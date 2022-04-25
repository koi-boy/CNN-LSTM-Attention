# CNN LSTM Attention 股价预测

- CNN + LSTM
- CNN + LSTM + ECA(attention)
- CNN + LSTM + SE(attention)
- CNN + LSTM + HW(attention)
- CNN + LSTM + CBAM(attention)

| 模型                | RMSE                  |
|-------------------|-----------------------|
| CNN + LSTM        | 0.00011371035229372369 |
| CNN + LSTM + ECA  | 0.0001245921911587092 |
| CNN + LSTM + SE   | 0.00009550479312152179 |
| CNN + LSTM + HW   | 0.00041322291971565306 |
| CNN + LSTM + CBAM | 0.0003162174993617968 |

# train
```python
python train.py -m Base
python train.py -m ECA
python train.py -m SE
python train.py -m HW
python train.py -m CBAM
```
模型会存在best_model路径下

# test
```python
python test.py -m Base
python test.py -m ECA
python test.py -m SE
python test.py -m HW
python test.py -m CBAM
```
预测结果会存在result_picture下

# 预测结果

## CNN + LSTM
![CNN + LSTM](./result_picture/Base_fic.jpg)
## CNN + LSTM + ECA
![CNN + LSTM + ECA](./result_picture/ECA_fic.jpg)
## CNN + LSTM + SE
![CNN + LSTM + SE](./result_picture/SE_fic.jpg)
## CNN + LSTM + HW
![CNN + LSTM + HW](./result_picture/HW_fic.jpg)
## CNN + LSTM + CBAM
![CNN + LSTM + CBAM](./result_picture/CBAM_fic.jpg)

