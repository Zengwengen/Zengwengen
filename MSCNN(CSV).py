import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv('data.csv')
X = data.iloc[:, :13].values  # 特征   选择从第一列到第13列的特征数据
Y = data.iloc[:, -1].values  # 标签   选择最后一列的标签数据

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=False)

# 将数据转换成 PyTorch 的 Tensor 格式
X_train = torch.tensor(X_train, dtype=torch.float).unsqueeze(1)  # 第一个维度为-1，表示保持原来的数据个数不变
X_test = torch.tensor(X_test, dtype=torch.float).unsqueeze(1)  # 第二个维度为1，表示每个数据项有一个通道（channel）
Y_train = torch.tensor(Y_train, dtype=torch.float)  # 第三个维度为13，表示每个数据项有13个特征
Y_test = torch.tensor(Y_test, dtype=torch.float)

# 将数据转换成 Dataset 对象
train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=False)


# 定义多尺度CNN模型
class MultiScaleCNN(nn.Module):
    def __init__(self):
        super(MultiScaleCNN, self).__init__()
        # 尺度1
        self.conv1_scale1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)  # 注意输入通道数改为1
        self.conv2_scale1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        # self.fc1_scale1 = nn.Linear(64 * ((13 - 3) + 1), 128)                           # 根据错误，修改卷积层和全连接层之间，全连接层的大小
        self.fc1_scale1 = nn.Linear(64 * 3, 128)

        # 尺度2
        self.conv1_scale2 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2_scale2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        # self.fc1_scale2 = nn.Linear(128 * ((13 - 3) + 1), 128)
        self.fc1_scale2 = nn.Linear(64 * 6, 128)

        # 公共层
        self.fc2 = nn.Linear(256, 1)  # 结合来自两个尺度的特征
        self.pool = nn.MaxPool1d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 尺度1
        x_scale1 = self.relu(self.conv1_scale1(x))
        # print(x_scale1.size())
        x_scale1 = self.pool(x_scale1)
        x_scale1 = self.relu(self.conv2_scale1(x_scale1))
        x_scale1 = self.pool(x_scale1)
        # x_scale1 = x_scale1.view(-1, 64 * ((13 - 3) + 1))
        x_scale1 = x_scale1.view(x_scale1.size(0), -1)
        # print(x_scale1.size())
        x_scale1 = self.relu(self.fc1_scale1(x_scale1))

        # 尺度2
        x_scale2 = self.relu(self.conv1_scale2(x))
        # print(x_scale1.size())
        x_scale2 = self.pool(x_scale2)
        x_scale2 = self.relu(self.conv2_scale2(x_scale2))
        x_scale2 = self.pool(x_scale2)
        # x_scale2 = x_scale2.view(-1, 128 * ((13 - 3) + 1))
        x_scale2 = x_scale2.view(x_scale2.size(0), -1)
        # print(x_scale1.size())
        x_scale2 = self.relu(self.fc1_scale2(x_scale2))

        # 结合来自两个尺度的特征
        x_combined = torch.cat((x_scale1, x_scale2), dim=1)

        # 分类
        x = self.fc2(x_combined)
        return x


# 初始化模型、损失函数和优化器
model = MultiScaleCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 早停

"""提前停止迭代（精度已达要求，梯度停止下降）"""

# 在训练集上训练模型
train_model(model, train_loader, criterion, optimizer, epochs=10)

# 在测试集上进行预测
model.eval()
with torch.no_grad():
    y_pred = model(X_test)


# 计算RMSE和MAPE
def calculate_metrics(actual, predicted):
    RMSE = np.sqrt(np.mean((actual - predicted)**2))
    MAPE = np.mean(np.abs((actual - predicted) / actual)) * 100
    return RMSE, MAPE


# 在测试集上计算RMSE和MAPE
Y_test_np = Y_test.numpy().flatten()
y_pred_np = y_pred.numpy().flatten()
RMSE, MAPE = calculate_metrics(Y_test_np, y_pred_np)
print(f'RMSE: {RMSE}, MAPE: {MAPE}%')


def R2_torch(actual, predicted):
    actual_mean = np.mean(actual)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual_mean) ** 2)
    R2 = 1 - ss_res / (ss_tot + 1e-8)  # 避免除以零的情况
    return R2


# Y_test = torch.Tensor(Y_test)
# y_pred = torch.Tensor(y_pred)

R2 = R2_torch(Y_test_np, y_pred_np)
print('R2: {}'.format(R2.item()))  # 使用 item() 方法获取张量的数值

# 可视化预测值与实际值的曲线
plt.figure(figsize=(10, 6))
plt.plot(Y_test_np, label='Actual Value', color='b',)
plt.plot(y_pred_np, label='Predicted Value', color='r', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()




