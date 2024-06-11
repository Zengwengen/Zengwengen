import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 读取股票数据
data = pd.read_csv('D:\\Users\\ZWG\\PycharmProjects\\pythonProject1\\stockdata_1500days(ver2)\\000049.csv', header=None)

# 选择要预测的特征列和需要预测的目标列(开盘价、最高价、最低价、收盘价、交易量)
feature_columns = [0, 1, 2, 4]
target_column = 3
features = data.iloc[:, feature_columns].values
target = data.iloc[:, target_column].values.reshape(-1, 1)

# 归一化数据
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))
features_normalized = scaler_features.fit_transform(features)
target_normalized = scaler_target.fit_transform(target)

# 划分训练集和测试集
train_size = int(len(features_normalized) * 0.8)
test_size = len(features_normalized) - train_size
train_features, test_features = features_normalized[0:train_size, :], features_normalized[train_size:len(features_normalized), :]
train_target, test_target = target_normalized[0:train_size, :], target_normalized[train_size:len(target_normalized), :]


# 构建特征和标签
def create_dataset(features, target, look_back):
    X, Y = [], []
    for i in range(len(features) - look_back - 1):
        a = features[i:(i + look_back), :]
        X.append(a)
        Y.append(target[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 20  # 使用过去*天的数据作为特征(自行选择，1就是不使用历史数据进行常规预测，由于LSTM的独特功能，建议大于1)
train_X, train_Y = create_dataset(train_features, train_target, look_back)
test_X, test_Y = create_dataset(test_features, test_target, look_back)

# 将数据转换为PyTorch张量
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).float()
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).float()


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


input_size = len(feature_columns)
hidden_size = 64
num_layers = 2
model = LSTM(input_size, hidden_size, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    outputs = model(train_X)
    optimizer.zero_grad()
    loss = criterion(outputs.squeeze(), train_Y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在测试集上评估模型
with torch.no_grad():
    test_outputs = model(test_X)
    test_loss = criterion(test_outputs.squeeze(), test_Y)
    print(f'Test Loss: {test_loss.item():.4f}')

# 反归一化并计算预测误差
test_predictions = scaler_target.inverse_transform(test_outputs.cpu().numpy())
test_actual = scaler_target.inverse_transform(test_Y.unsqueeze(-1).cpu().numpy())
rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
print(f'Test RMSE: {rmse:.4f}')

# 可视化预测结果
plt.plot(test_actual, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
