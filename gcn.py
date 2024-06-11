import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('D:\\Users\\ZWG\\PycharmProjects\\pythonProject1\\stockdata_1500days(ver2)\\000049.csv', header=None)

# 准备特征数据和目标数据
X = data.iloc[:, [0, 1, 2, 4]].values
y = data.iloc[:, 3].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=False)


# 构建GCN模型
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.gc1(torch.matmul(adj, x)))
        x = self.gc2(torch.matmul(adj, x))
        return x


# 超参数
input_dim = X.shape[1]
hidden_dim = 64
output_dim = 1
learning_rate = 0.001
epochs = 2000

# 初始化模型、优化器和损失函数
model = GCN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 转换数据为张量
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# 训练模型
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor, torch.eye(X_train.shape[0]))  # 邻接矩阵假设为单位矩阵
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

# 测试模型
model.eval()
with torch.no_grad():
    test_output = model(X_test_tensor, torch.eye(X_test.shape[0]))
    test_loss = criterion(test_output, y_test_tensor)
    print('Test Loss: {:.4f}'.format(test_loss.item()))

# 打印预测结果
predicted_prices = test_output.numpy()
print(predicted_prices)
# 可视化预测结果与实际值
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predicted_prices, label='Predicted')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()
