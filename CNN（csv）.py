import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# 读取CSV文件并处理数据
data = pd.read_csv('data.csv')
X = data.iloc[:, :13].values  # 特征
y = data.iloc[:, -1].values  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换数据为Tensor并调整维度
X_train_tensor = torch.Tensor(X_train).unsqueeze(1)  # 在第二维上增加一个维度，将64个通道对应到第二维
X_test_tensor = torch.Tensor(X_test).unsqueeze(1)
y_train_tensor = torch.Tensor(y_train)
y_test_tensor = torch.Tensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.fc = nn.Linear(16 * ((13 - 3) + 1), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 初始化模型、损失函数和优化器
model = CNN()
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
train_model(model, train_loader, criterion, optimizer, epochs=100)

# 在测试集上进行预测
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)


# 计算RMSE和MAPE
def calculate_metrics(actual, predicted):
    RMSE = np.sqrt(np.mean((actual - predicted)**2))
    MAPE = np.mean(np.abs((actual - predicted) / actual)) * 100
    return RMSE, MAPE


# 在测试集上计算RMSE和MAPE
RMSE, MAPE = calculate_metrics(y_test, y_pred.numpy().flatten())
print(f'RMSE: {RMSE}, MAPE: {MAPE}%')

# 可视化预测值与实际值的曲线
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Value', color='b',)
plt.plot(y_pred, label='Predicted Value', color='r', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

# 保存预测值和实际值到CSV文件
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.numpy().flatten()})
results_df.to_csv('prediction_cnn.csv', index=False)


