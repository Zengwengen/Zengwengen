import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
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
    train_outputs, _ = model(torch.Tensor(X))  # 注意这里的改动
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
selected_features_indices_reconstructed = activated_features_ratio_reconstructed > 0.5
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

# 提取特征和输出
X = data_1.iloc[:, :-1]
y = data_1.iloc[:, -1]

# 数据归一化处理（可选）
# X = X / 255.0

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=False)
epochs = 100


class CNN:
    def __init__(self, X_train):
        self.history = None
        self.model = Sequential()
        self.model.add(Conv1D(filters=5, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
        self.model.add(MaxPooling1D(pool_size=5))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='linear'))  # 修改激活函数为linear
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=2):
        self.history = self.model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test.reshape(-1, X_test.shape[1], 1), y_test))

    def predict(self, X):
        return self.model.predict(X.reshape(-1, X.shape[1], 1)).reshape(-1)


# 实例化模型
cnn_model = CNN(X_train)

# 训练模型
cnn_model.train(X_train, y_train, X_test, y_test, epochs=100)

# 使用模型进行预测，假设X是你想要预测的数据
y_pred = cnn_model.predict(X)

# 可视化预测值与实际值的曲线
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Value', color='b',)
plt.plot(y_pred, label='Predicted Value', color='r', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

# 将预测结果保存为CSV文件
predictions_df = pd.DataFrame(y_pred, columns=['Predicted Value'])
predictions_df.to_csv('predictions.csv', index=False)
