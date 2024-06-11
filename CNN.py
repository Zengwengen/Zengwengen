import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
from torch.autograd import Variable  # 导入自动微分模块
from torch.utils.data import DataLoader  # 导入数据加载器
import torchvision  # 导入计算机视觉库
from typing import Iterable  # 导入Iterable类型提示
import matplotlib.pyplot as plt  # 导入绘图库

EPOCH = 2  # 定义训练轮数，CNN巡视训练集的次数
BATCH_SIZE = 50  # 定义批大小
LR = 0.001  # 定义学习率
DOWNLOAD_MNIST = True  # 是否下载MNIST数据集


# 自定义collate函数，将PIL Image对象转换为Tensor对象
def custom_collate(batch):
    # data_set=[] #训练集
    # for item in batch:
    #     data_set.append(item[0])
    data_set = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data_set = torch.stack(data_set)
    target = torch.tensor(target)
    return [data_set, target]


# 使用transforms对数据进行预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# 构建训练数据集
train = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transform, download=DOWNLOAD_MNIST)

# 创建DataLoader并指定自定义collate函数
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

print(train.data.size())  # 打印训练数据大小
print(train.targets.size())  # 打印训练标签大小
print(train.data[0])  # 打印第一张训练图片
# 显示第一张训练图片
plt.imshow(train.data[0].numpy(), cmap='gray')
plt.title('%i' % train.targets[0])
plt.show()

# 构建测试数据集
test_data = torchvision.datasets.MNIST(root='./mnist', train=False, transform=transform)

# 获取部分测试数据
with torch.no_grad():
    test_x = torch.unsqueeze(test_data.data, dim=1).float()[:3000] / 255
    test_y = test_data.targets[:3000]

train_labels = train.targets
test_labels = test_data.targets
print(train_labels)  # 打印训练标签
print(test_labels)  # 打印测试标签


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, input_m):
        local_x = self.conv1(input_m)
        local_x = self.conv2(local_x)
        local_x = local_x.view(local_x.size(0), -1)
        output_m = self.out(local_x)
        return output_m


cnn = CNN()  # 实例化CNN模型

parameters_iterable: Iterable[torch.Tensor] = (param for param in list(cnn.parameters()))  # 定义参数可迭代对象
optimizer = torch.optim.Adam(parameters_iterable, lr=LR)  # 定义优化器
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数

step = 0
for epoch in range(EPOCH):
    for step, data in enumerate(train_loader):
        x, y = data
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)  # 前向传播
        loss = loss_fn(output, b_y)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if step % 50 == 0:
            test_output = cnn(test_x)
            y_pred = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = torch.eq(y_pred, test_y).float().mean().item()

            print('now epoch: ', epoch, '   |  loss: %.4f ' % loss.item(), '     |   accuracy:   ', accuracy)

test_output = cnn(test_x[:10])  # 获取前10个样本的预测结果
y_pred = torch.max(test_output, 1)[1].data.squeeze()
print(y_pred.tolist(), 'prediction Result')  # 打印预测结果
print(test_y[:10].tolist(), 'Real Result')  # 打印真实结果
