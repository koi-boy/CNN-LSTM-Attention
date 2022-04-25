import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
import numpy as np

class StockDataset(Dataset):
    def __init__(self, dataPath, window, is_test=False):
        df1=pd.read_csv(dataPath)
        df1=df1.iloc[:,2:]
        min_max_scaler = preprocessing.MinMaxScaler()
        df0=min_max_scaler.fit_transform(df1)
        df = pd.DataFrame(df0, columns=df1.columns)
        input_size=len(df.iloc[1,:])
        stock=df
        seq_len=window
        amount_of_features = len(stock.columns)#有几列
        data = stock.values #pd.DataFrame(stock) 表格转化为矩阵
        sequence_length = seq_len + 1#序列长度
        result = []
        for index in range(len(data) - sequence_length):#循环数据长度-sequence_length次
            result.append(data[index: index + sequence_length])#第i行到i+sequence_length
        result = np.array(result)#得到样本，样本形式为6天*3特征
        row = round(0.9 * result.shape[0])#划分训练集测试集
        train = result[:int(row), :]
        x_train = train[:, :-1]
        y_train = train[:, -1][:,-1]
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1][:,-1]
        #reshape成 6天*3特征
        X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
        X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  
        if not is_test:
            self.data = X_train
            self.label = y_train
        else:
            self.data = X_test
            self.label = y_test
            
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self,idx): 
        return torch.from_numpy(self.data[idx]).to(torch.float32), torch.FloatTensor([self.label[idx]])
