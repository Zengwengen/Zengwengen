import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
X = data.iloc[:, :-1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)


class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


model = SAE(input_dim=X.shape[1], hidden_dim=64)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_outputs, _ = model(torch.Tensor(X))
    train_loss = criterion(train_outputs, torch.Tensor(X))
    train_loss.backward()
    optimizer.step()

# 在模型训练后，获取并分析重构后的输出
model.eval()
with torch.no_grad():
    reconstructed_outputs, _ = model(torch.Tensor(X))

# 分析重构输出中被激活的特征
# 计算每个特征在重构输出中被激活的比例（即值大于0的比例）
activated_features_ratio_reconstructed = (reconstructed_outputs > 0).float().mean(axis=0)
print("重构输出中特征被激活的比例:", activated_features_ratio_reconstructed)

# 根据需要选择特征进行下一步训练
# 这里只是一个示例，实际选择应基于任务需求和进一步的分析
selected_features_indices_reconstructed = activated_features_ratio_reconstructed > 0.3
selected_features_indices = selected_features_indices_reconstructed.nonzero().squeeze().numpy()
print("重构输出中选择的特征索引:", selected_features_indices)

# 保存选择的特征索引到文件
np.save('selected_features_indices.npy', selected_features_indices)
# 加载特征索引文件
selected_features_indices = np.load('selected_features_indices.npy')

# 提取对应特征列
selected_features = data.iloc[:, selected_features_indices]

# 合并提取的特征与原始数据的最后一列（假设最后一列为输出）
output_column = data.iloc[:, -1]
train_data = pd.concat([selected_features, output_column], axis=1)

# 保存新的训练数据到文件
train_data.to_csv('select_data.csv', index=False)
# 加载训练数据
data_1 = pd.read_csv('select_data.csv')

# 绘制每一列的曲线图
plt.figure(figsize=(12, 6))
for column in data_1.columns:
    plt.plot(data_1[column], label=column)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plots of Each Column')
plt.legend()
plt.show()

# 取最后一列的数据
last_column = data_1.columns[-1]

# 绘制最后一列的曲线图
plt.figure(figsize=(12, 6))
plt.plot(data_1[last_column], label=last_column, color='b')  # 假设使用蓝色作为曲线颜色
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plot of Last Column')
plt.legend()
plt.show()

X = data_1.iloc[:, :-1].values
y = data_1.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)


class NNRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NNRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = NNRegression(input_dim=X.shape[1], hidden_dim=64)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=False)

epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(torch.Tensor(X_train))
    loss = criterion(outputs, torch.Tensor(y_train).view(-1, 1))

    loss.backward()
    optimizer.step()

# 使用模型进行预测
model.eval()
with torch.no_grad():
    y_pred = model(torch.Tensor(X_test)).squeeze().numpy()

# 可视化预测值与实际值的曲线
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Value', color='b')
plt.plot(y_pred, label='Predicted Value', color='r', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

y_test_tensor = torch.Tensor(y_test)
predicted_values_tensor = torch.Tensor(y_pred)

# 计算均方根误差（RMSE）
RMSE = torch.sqrt(torch.mean((predicted_values_tensor - y_test_tensor)**2))
print("Root Mean Squared Error (RMSE):", RMSE.item())

# 计算平均绝对百分比误差（MAPE）
MAPE = torch.mean(torch.abs((predicted_values_tensor - y_test_tensor) / y_test_tensor)) * 100
print("Mean Absolute Percentage Error (MAPE):", MAPE.item())
# 将预测结果保存为CSV文件
results_df = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': y_pred})

# 保存实际值和预测值到 CSV 文件
results_df.to_csv('predictions_actual_values_sae-nn.csv', index=False)
