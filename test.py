import torch
from dataloader import StockDataset
from torch.utils.data import DataLoader 
from model import *
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists("result_picture"):
    os.makedirs("result_picture")
    
if not os.path.exists("best_model"):
    os.makedirs("best_model")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", "-m", help="which model", required=True)
    args = parser.parse_args()
    return args


args = parse_args()

test_data = StockDataset('dataset/data.csv', 5, is_test=True)
print_step = 10

test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)

model_dict = {
    "SE": CNNLSTMModel_SE,
    "Base": CNNLSTMModel,
    "CBAM": CNNLSTMModel_CBAM,
    "HW": CNNLSTMModel_HW
}


model = model_dict[args.model]()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
params = torch.load(f"best_model/{args.model}_best.pth")
model.load_state_dict(params)

eval_loss = 0.0
with torch.no_grad():
    y_gt = []
    y_pred = []
    for data, label in test_loader:
        y_gt += label.numpy().squeeze(axis=1).tolist()
        out = model(data)
        loss = criterion(out, label)
        eval_loss += loss.item()
        y_pred += out.numpy().squeeze(axis=1).tolist()
    print(len(y_gt), len(y_pred))

y_gt = np.array(y_gt)
y_gt = y_gt[:,np.newaxis]
y_pred = np.array(y_pred)
y_pred = y_pred[:,np.newaxis]


draw=pd.concat([pd.DataFrame(y_gt),pd.DataFrame(y_pred)],axis=1)
draw.iloc[200:500,0].plot(figsize=(12,6))
draw.iloc[200:500,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') #添加标题
plt.savefig(f"result_picture/{args.model}_fic.jpg")
print("{}'s eval loss is {}".format(args.model, eval_loss/len(test_loader)))