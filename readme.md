# CNN LSTM Attention 股价预测

- CNN + LSTM
- CNN + LSTM + SE(attention)
- CNN + LSTM + HW(attention)
- CNN + LSTM + CBAM(attention)

| 模型  | RMSE |
|-----|------|
| CNN + LSTM | 0.0002811193857765333  |
| CNN + LSTM + SE | 0.0001978238473053683  |
| CNN + LSTM + HW | 0.000607791225775145  |
| CNN + LSTM + CBAM | 0.00033038058675381103  |

# train
```python
python train.py -m Base
python train.py -m SE
python train.py -m HW
python train.py -m CBAM
```
模型会存在best_model路径下

# test
```python
python test.py -m Base
python test.py -m SE
python test.py -m HW
python test.py -m CBAM
```
预测结果会存在result_picture下

# 预测结果

## CNN + LSTM
![CNN + LSTM](./result_picture/Base_fic.jpg)
## CNN + LSTM + SE
![CNN + LSTM + SE](./result_picture/SE_fic.jpg)
## CNN + LSTM + HW
![CNN + LSTM + HW](./result_picture/HW_fic.jpg)
## CNN + LSTM + CBAM
![CNN + LSTM + CBAM](./result_picture/CBAM_fic.jpg)

