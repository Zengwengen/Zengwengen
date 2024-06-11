import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义多尺度CNN模型
class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiScaleCNN, self).__init__()
        # 尺度1
        self.conv1_scale1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 注意输入通道数改为1
        self.conv2_scale1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1_scale1 = nn.Linear(64 * 7 * 7, 128)  # 修改全连接层输入维度

        # 尺度2
        self.conv1_scale2 = nn.Conv2d(1, 64, kernel_size=5, padding=2)  # 注意输入通道数改为1
        self.conv2_scale2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.fc1_scale2 = nn.Linear(128 * 7 * 7, 128)  # 修改全连接层输入维度

        # 公共层
        self.fc2 = nn.Linear(256, num_classes)  # 结合来自两个尺度的特征
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 尺度1
        x_scale1 = self.relu(self.conv1_scale1(x))
        x_scale1 = self.pool(x_scale1)
        x_scale1 = self.relu(self.conv2_scale1(x_scale1))
        x_scale1 = self.pool(x_scale1)
        x_scale1 = x_scale1.view(-1, 64 * 7 * 7)  # 修改全连接层输入维度
        x_scale1 = self.relu(self.fc1_scale1(x_scale1))

        # 尺度2
        x_scale2 = self.relu(self.conv1_scale2(x))
        x_scale2 = self.pool(x_scale2)
        x_scale2 = self.relu(self.conv2_scale2(x_scale2))
        x_scale2 = self.pool(x_scale2)
        x_scale2 = x_scale2.view(-1, 128 * 7 * 7)  # 修改全连接层输入维度
        x_scale2 = self.relu(self.fc1_scale2(x_scale2))

        # 结合来自两个尺度的特征
        x_combined = torch.cat((x_scale1, x_scale2), dim=1)

        # 分类
        x = self.fc2(x_combined)
        return x


# 定义数据预处理和加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 图像标准化
])

train_dataset = datasets.MNIST(root='./data_MNIST', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
model = MultiScaleCNN(num_classes=10)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移到GPU上（如果可用）
model.to(device)

# 训练模型
model.train()

for epoch in range(epochs):
    total_train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到GPU上

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        if i % 100 == 0:  # 每100个mini-batches打印一次损失
            print('[{}, {}] loss: {:.4f}'.format(epoch + 1, i + 1, total_train_loss / 100))
            total_train_loss = 0.0

print('Finished Training')
