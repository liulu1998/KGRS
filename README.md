# KGRS
My graduation project "Knowledge Graph Enhanced Recommender System"

## Environment Requirements

The code has been tested under Python 3.8.8

Required packages are as follows:

```
numpy==1.19.2
pandas==1.2.1
scikit-learn==0.23.2
scipy==1.6.1
torch==1.8.1
```

## Run the code
- Amazon-book dataset  
```python  
python main.py --data-name amazon-book --heads 1 --epochs 40 --lr 0.03 --dropout_kg 0.3 --dropout_cf 0.2 \
--weight_task_kg 0.5 --weight_L2_kg 1e-05 --c0 300 --c1 600 --embed_size 64 --attention_size 32 --gpus 0 --seed 2021 
```

- Yelp2018 dataset
```python  
python main.py --data-name yelp2018 --heads 2 --epochs 90 --lr 0.03 --dropout_kg 0.3 --dropout_cf 0.1 \
--weight_task_kg 0.8 --weight_L2_kg 1e-05 --c0 1000 --c1 7000 --embed_size 64 --attention_size 32 --gpus 0 --seed 2021 
```

Log will be saved to `./log/`
