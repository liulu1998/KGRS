from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
from models.utils.metrics import L2Loss


class Aggregator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, aggregator_type: str = ""):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.aggregator_type = aggregator_type
        self.activation = nn.LeakyReLU()

        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)
        else:
            raise NotImplementedError

    def forward(self, h, N_h):
        if self.aggregator_type == "gcn":
            out = self.activation(self.W(h + N_h))
        elif self.aggregator_type == "graphsage":
            out = self.activation(self.W(torch.cat([h, N_h], dim=1)))
        elif self.aggregator_type == "bi-interaction":
            out = self.activation(self.W1(h + N_h)) + self.activation(self.W2(h * N_h))

        out = self.dropout(out)
        return out


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, aggregator_type: str):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        # W: projection matrix
        self.W = nn.Linear(self.in_dim, self.hid_dim, bias=False)
        nn.init.trunc_normal_(self.W.weight, mean=0., std=sqrt(2. / (self.in_dim + self.hid_dim)))

        self.bias = nn.Parameter(torch.zeros([self.hid_dim]))
        self.h = nn.Parameter(
            torch.full([self.hid_dim, 1], fill_value=0.01)
        )
        self.activation = nn.ReLU()

        if aggregator_type is not None and aggregator_type != "":
            self.aggregator = Aggregator(
                in_dim=self.in_dim, out_dim=self.in_dim,
                dropout=0.1, aggregator_type=aggregator_type
            )
        else:
            self.aggregator = None

    def forward(self, item, r, t, mask):
        """
        :param item: (batch, in)
        :param r: (batch, max_i_r, in)
        :param t: (batch, max_i_r, in)
        :param mask: (batch, max_i_r)
        """
        score = self.activation(self.W(r * t) + self.bias)
        # masked softmax
        score = torch.exp(torch.einsum("abc, ck->abk", score, self.h))
        score = torch.einsum("ab, abc->abc", mask, score)  # (batch, max_i_r, 1)

        # softmax
        score_sum = torch.sum(score, dim=1, keepdim=True)  # (batch, 1, 1)
        # add epsilon to prevent division by zero
        weight = torch.div(score, score_sum + 1e-9)  # (batch, max_i_r, 1)

        item_neighbors = torch.sum(torch.mul(weight, t), dim=1)

        if not self.aggregator:
            out = item + item_neighbors
            return out
        return self.aggregator(item, item_neighbors)

    def L2(self):
        loss = torch.tensor(0., requires_grad=True) + L2Loss(self.W.weight) + L2Loss(self.h)
        return loss


class GAT(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, heads: int, dropout: float, aggregator_type: str):
        super(GAT, self).__init__()
        # multi-head attention
        self.attentions = nn.ModuleList(
            [GATLayer(in_dim, hid_dim, aggregator_type) for _ in range(heads)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, item, r, t, mask):
        """
        :param item: items' basic embeddings
        :param r: relation embeddings
        :param t: tail entity embeddings
        :param mask:
        :return: out
        """
        head_dim = 1
        out = torch.stack([att(item, r, t, mask) for att in self.attentions], dim=head_dim)  # [batch, head, emb]

        # avg
        out = torch.torch.mean(out, dim=head_dim)
        out = self.dropout(out)
        return out

    def L2(self):
        loss = torch.tensor(0., requires_grad=True)
        for att in self.attentions:
            loss = loss + att.L2()
        return loss


class KGRS(nn.Module):
    def __init__(self, n_users, n_items, n_relations, n_entities, max_i_u, max_i_r,
                 negative_c, negative_ck, emb_size, attention_size,
                 weight_task_kg, weight_L2_kg, dropout_cf, dropout_kg,
                 n_heads=1, aggregator_type=None):
        """
        :param n_users: number of users in all data
        :param n_items:
        :param n_relations:
        :param n_entities:
        :param max_i_u: Maximum number of interactions between item to users
        :param max_i_r: Maximum number of connections between item (head entity) and relations
        :param negative_c:
        :param negative_ck:
        :param emb_size: embedding size in KG Embedding
        :param dropout_cf: dropout ratio in CF part
        :param dropout_kg: dropout ratio in KG part
        :param n_heads: number of heads in multi-head attention
        """
        super(KGRS, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_relations = n_relations
        self.n_entities = n_entities

        # Maximum number of interactions between item to users
        self.max_i_u = max_i_u
        # Maximum number of connections between item (head entity) and relations
        self.max_i_r = max_i_r

        self.emb_size = emb_size
        self.attention_size = attention_size

        # weight of pos & neg samples in KG, respectively
        self.negative_c = negative_c
        self.negative_ck = negative_ck

        # weight of pos and neg samples in CF, respectively
        self.c_v_pos = 1.0
        self.c_v_neg = 0.2

        # weight of multi-tasks
        self.weight_cf_loss = 1.0
        self.weight_kg_loss = weight_task_kg
        # weight of L2 Regularization
        self.weight_cf_l2 = 1e-5
        self.weight_kg_l2 = weight_L2_kg

        # Dropout
        self.dropout_kg = nn.Dropout(p=dropout_kg, inplace=False)
        self.dropout_cf = nn.Dropout(p=dropout_cf, inplace=False)

        # embedding layers
        self.User = nn.Embedding(self.n_users + 1, self.emb_size)

        # >>> SimplE as KGE model
        self.Head = nn.Embedding(max(self.n_items, self.n_entities) + 1, self.emb_size)
        self.Tail = nn.Embedding(max(self.n_items, self.n_entities) + 1, self.emb_size)

        # r and r^{-1} in SimplE
        self.R = nn.Embedding(self.n_relations + 1, self.emb_size)
        self.R_inv = nn.Embedding(self.n_relations + 1, self.emb_size)

        # StayPositive
        self.psi = 2.
        # weight of regularization term in StayPositive loss
        self.alpha = 1e-6

        # prediction vector in NCF
        self.pre_vec = nn.Parameter(
            torch.full([self.emb_size, 1], fill_value=0.01),
            requires_grad=True
        )

        # >>> attention
        self.n_heads = n_heads
        self.GAT = GAT(in_dim=self.emb_size, hid_dim=self.attention_size, heads=self.n_heads, dropout=dropout_kg,
                       aggregator_type=aggregator_type)
        self.init_weights()
        self.hardtanh = nn.Hardtanh()
        self.cf_hardtanh = nn.Hardtanh(min_val=0., max_val=1.)

    def init_weights(self):
        # init Embedding layers
        for layer in self.modules():
            if isinstance(layer, nn.Embedding):
                nn.init.trunc_normal_(tensor=layer.weight, mean=0.0, std=0.01)

    def cal_kg_loss(self, score, item, item_inv, batch_r, batch_t):
        """
        calculate StayPositive loss for KG Embedding
        :param score, score of triples, 2D Tensor, (batch_size, max_i_r)
        :param item, a batch of items' Head Representation
        :param item_inv, a batch of items' Tail Representation
        :param batch_r, a batch of relations, 2D Tensor, size (batch_size, max_i_r)
        :param batch_t, a batch of tail entities, 2D Tensor, same size as batch_r
        :return: KGE loss
        """
        # filter out padding values
        r_set = batch_r[batch_r != self.n_relations].unique(sorted=False)
        # 2D Tensor
        r = self.R(r_set)
        r_inv = self.R_inv(r_set)

        t_set = batch_t[batch_t != self.n_entities].unique(sorted=False)
        # t_set = torch.masked_select(batch_t, torch.ne(batch_t, self.n_entities))
        # t_set = torch.unique(t_set, sorted=False)
        # 2D Tensor
        t = self.Tail(t_set)
        h_t = self.Head(t_set)

        # StayPositive loss
        # score_all = torch.mean(torch.stack([
        #     torch.einsum("ad,bd,cd->abc", item, r, t),
        #     torch.einsum("ad,bd,cd->abc", item_inv, r_inv, h_t)
        # ]), dim=0)

        score_all = torch.sum(torch.mean(
            torch.stack([
                torch.sum(item, dim=0) * torch.sum(r, dim=0) * torch.sum(t, dim=0),
                torch.sum(item_inv, dim=0) * torch.sum(r_inv, dim=0) * torch.sum(h_t, dim=0),
            ])
        ))
        # pos loss + regularization term
        loss_kg = torch.sum(F.softplus(-(score + self.psi), threshold=200)) + self.alpha * torch.norm(score_all, p=1)
        return loss_kg

    def cal_squared_cf_loss(self, pre, item, weight):
        loss_cf = torch.sum(
            (self.c_v_pos - weight) * torch.pow(pre, 2) - 2.0 * self.c_v_pos * pre
        ) + torch.sum(
            torch.matmul(self.pre_vec, self.pre_vec.t())
            * torch.einsum("ui,uj->ij", self.User.weight, self.User.weight)
            * torch.einsum("vi,vj->ij", weight * item, item)
        )
        return loss_cf

    def L2(self):
        """ L2 Regularization """
        l2_kg = torch.tensor(0., requires_grad=True) + L2Loss(self.User.weight) + L2Loss(self.Head.weight) \
                + L2Loss(self.Tail.weight) + L2Loss(self.R.weight) + L2Loss(self.R_inv.weight)

        l2_cf = self.GAT.L2()

        l2_loss = self.weight_kg_l2 * l2_kg + self.weight_cf_l2 * l2_cf
        return l2_loss

    def cf_score(self, pos_items, users, r_test, t_test):
        """ predict user-item preference
        :param pos_items:
        :param users:
        :param r_test: relations
        :param t_test: tail entities
        :return:
        """
        r_emb = self.R(r_test)
        t_emb = self.Tail(t_test)

        # mask
        mask = torch.ne(r_test, self.n_relations).float()
        r_emb = torch.einsum("ab, abc->abc", mask, r_emb)
        t_emb = torch.einsum("ab, abc->abc", mask, t_emb)

        # >>> cal items embedding
        item_emb = self.Head(pos_items).squeeze_()
        item_emb = self.GAT(item=item_emb, r=r_emb, t=t_emb, mask=mask)

        r_inv_emb = self.R_inv(r_test)
        t_h_emb = self.Head(t_test)

        item_emb_inv = self.Tail(pos_items).squeeze_()
        item_emb_inv = self.GAT(item=item_emb_inv, r=r_inv_emb, t=t_h_emb, mask=mask)

        # information fusion
        item_emb = 0.8 * item_emb + 0.2 * item_emb_inv
        # <<< cal items' embedding

        user_emb = self.User(users)
        dot = torch.einsum("ac, bc->abc", user_emb, item_emb)

        # user-item preference
        pre = torch.einsum("ajk, kl->ajl", dot, self.pre_vec)
        if torch.any(torch.isnan(pre)):
            raise ValueError("NaN value in user-item preference !")

        # TODO 激活到 [0, 1] ?
        pre = self.cf_hardtanh(pre)
        return pre

    def infer(self, input_i, input_iu, input_hr, input_ht):
        """
        :param input_i: item ID
        :param input_iu: (batch_size, max_i_u), corresponding user ID
        :param input_hr: (batch_size, max_i_r), corresponding relation ID
        :param input_ht: (batch_size, max_i_r), corresponding tail entity ID
        :return: total loss
        """
        item_emb = self.Head(input_i).squeeze_()  # (batch_size, emb_size)
        item_t_emb = self.Tail(input_i).squeeze_()

        # weights
        c = self.negative_c[input_i]  # (batch_size, 1)
        # ck = self.negative_ck[input_i]

        # Dropout
        item_emb_kg = self.dropout_kg(item_emb)
        item_t_emb_kg = self.dropout_kg(item_t_emb)

        # >> KG, calculate tuples' score
        # r and r^{-1}
        r_emb = self.R(input_hr)  # (batch_size, max_i_r, emb_size)
        r_inv_emb = self.R_inv(input_hr)

        t_emb = self.Tail(input_ht)  # (batch_size, max_i_r, emb_size)
        t_h_emb = self.Head(input_ht)

        # mask
        mask = torch.ne(input_hr, self.n_relations).float()
        pos_r_emb = torch.einsum("ab, abc->abc", mask, r_emb)
        pos_r_inv_emb = torch.einsum("ab, abc->abc", mask, r_inv_emb)

        pos_t_emb = torch.einsum("ab, abc->abc", mask, t_emb)
        pos_t_h_emb = torch.einsum("ab, abc->abc", mask, t_h_emb)

        # SimplE scoring function
        pos_rt = pos_r_emb * pos_t_emb
        pos_hrt = torch.einsum("ac, abc->ab", item_emb_kg, pos_rt).squeeze_()  # (batch_size, max_i_r)

        pos_rt_inv = pos_r_inv_emb * pos_t_h_emb
        pos_hrt_inv = torch.einsum("ac, abc->ab", item_t_emb_kg, pos_rt_inv).squeeze_()  # (batch_size, max_i_r)

        # tuples' score
        # TODO 激活 ?
        tuple_score = self.hardtanh(torch.mean(torch.stack([pos_hrt, pos_hrt_inv]), dim=0))  # (batch_size, max_i_r)
        # <<< SimplE scoring function

        # loss on Knowledge Graph Embedding task
        loss_kg = self.cal_kg_loss(score=tuple_score, item=item_emb_kg, item_inv=item_t_emb_kg,
                                   batch_r=input_hr, batch_t=input_ht)
        # << KG

        # >> CF, predict user-item preference
        # information 1
        item_emb_cf = self.dropout_cf(item_emb)  # basic embedding, (batch_size, emb_size)
        item_emb_qv = self.GAT(item=item_emb_cf, r=pos_r_emb, t=pos_t_emb,
                               mask=mask)  # aggregated embedding, (batch_size, emb_size)

        # information 2
        item_emb_cf_inv = self.dropout_cf(item_t_emb)  # basic embedding, (batch_size, emb_size)
        item_emb_qv_inv = self.GAT(item=item_emb_cf_inv, r=pos_r_inv_emb, t=pos_t_h_emb,
                                   mask=mask)  # aggregated embedding, (batch_size, emb_size)

        # information fusion
        item_emb_qv = 0.8 * item_emb_qv + 0.2 * item_emb_qv_inv  # items' final embedding

        user_emb = self.User(input_iu)  # (batch_size, max_i_u, emb_size)

        # mask
        pos_num_u = torch.ne(input_iu, self.n_users).float()
        user_emb = torch.einsum("ab, abc->abc", pos_num_u, user_emb)

        # predict user-item preference
        pos_iu = torch.einsum("ac, abc->abc", item_emb_qv, user_emb)
        pos_iu = torch.einsum("ajk, kl->ajl", pos_iu, self.pre_vec)
        # TODO 激活到 [0, 1] ?
        pos_iu = self.cf_hardtanh(torch.reshape(pos_iu, [-1, self.max_i_u]))

        # loss on User Preference Prediction task
        loss_cf = self.cal_squared_cf_loss(pre=pos_iu, item=item_emb_qv, weight=c)
        # << CF

        # L2 regularization
        l2_regular = self.L2()

        # multi-task learning loss
        tot_loss = self.weight_kg_loss * loss_kg + self.weight_cf_loss * loss_cf + l2_regular
        return tot_loss, loss_kg.item(), loss_cf.item()

    def forward(self, mode: str, **kwargs):
        """
        :param mode: "cal_loss" to train model, "predict" to predict user-item preference
        :param kwargs: other arguments, in key-value format
        :return: total loss when mode is "cal_loss", user-item preference when mode is "predict"
        """
        if mode == "cal_loss":
            return self.infer(**kwargs)
        elif mode == "predict":
            return self.cf_score(**kwargs)
        else:
            raise NotImplementedError
