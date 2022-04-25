import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):

    def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNNLSTMModel_ECA(nn.Module):

    def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
        super(CNNLSTMModel_ECA, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.attn = nn.Linear(lstm_units * 2, lstm_units * 2)
        self.act3 = nn.Sigmoid()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        attn = self.attn(x)  # bs, 2*lstm_units
        attn = self.act3(attn)
        x = x * attn
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNNLSTMModel_SE(nn.Module):

    def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
        super(CNNLSTMModel_SE, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.se_fc = nn.Linear(window, window)

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)

        # se
        avg = x.mean(dim=1)  # bs, window
        se_attn = self.se_fc(avg).softmax(dim=-1)  # bs, window
        x = torch.einsum("bnd,bd->bnd", x, se_attn)

        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNNLSTMModel_CBAM(nn.Module):

    def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
        super(CNNLSTMModel_CBAM, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.se_fc = nn.Linear(window, window)
        self.hw_fc = nn.Linear(lstm_units, lstm_units)

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)

        # chanal
        avg = x.mean(dim=1)  # bs, window
        se_attn = self.se_fc(avg).softmax(dim=-1)  # bs, window
        x = torch.einsum("bnd,bd->bnd", x, se_attn)

        # wh
        avg = x.mean(dim=2)  # bs, lstm_units
        hw_attn = self.hw_fc(avg).softmax(dim=-1)  # bs, lstm_units
        x = torch.einsum("bnd,bn->bnd", x, hw_attn)

        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNNLSTMModel_HW(nn.Module):

    def __init__(self, window=5, dim=4, lstm_units=16, num_layers=2):
        super(CNNLSTMModel_HW, self).__init__()
        self.conv1d = nn.Conv1d(dim, lstm_units, 1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.hw_fc = nn.Linear(lstm_units, lstm_units)

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)

        # wh
        avg = x.mean(dim=2)  # bs, lstm_units
        hw_attn = self.hw_fc(avg).softmax(dim=-1)  # bs, lstm_units
        x = torch.einsum("bnd,bn->bnd", x, hw_attn)

        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x
