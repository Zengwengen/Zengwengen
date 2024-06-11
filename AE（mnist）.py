import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

'准备数据集'
# 准备MNIST数据集 (数据会下载到py文件所在的data_MNIST文件夹下)
train_data = torchvision.datasets.MNIST(root='../data_MNIST', train=True, transform=transforms.ToTensor(), download=True)
# test_data = torchvision.datasets.MNIST(root='../data_MNIST', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

'模型结构'


# nn.MOdule是pytorch中用于构建神经网络的基本类，所有的模型都应该继承这个类
# 作用：用于参数管理（方便访问模型中所有参数），模型组织（模块化，通过在模型中可以嵌套入其他类型)
class Encoder(torch.nn.Module):
    # 编码器，将input_size维度数据经过hidden_size大小的Linear1层，压缩为latent_size维度(Linear2层)
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()  # 继承刚定义好的Encoder这个类
        self.linear1 = nn.Linear(input_size, hidden_size)  # 包含了两个全连接层，比如input_size=784,hidden_size=128(输出的大小，即神经元的个数),latent_size=64
        self.linear2 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))  # 选取relu作为激活函数， 输入数据x进去网络linear,wx+b,后面经激活函数，变为relu(wx+b)
        x = torch.relu(self.linear2(x))  # 因为有2层，所以还有一个relu
        return x


class Decoder(torch.nn.Module):
    # 解码器，将latent_size维度的压缩数据解码升维度，经过hidden_size的Linear1层，转换为output_size维度(Linear2层)的数据
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))  # 这里改用Sigmoid,relu出来的效果不太好；利用Sigmoid将输出重构的数据都映射到(0,1)区间
        return x


class AE(torch.nn.Module):
    # 将编码器解码器组合，数据先后通过编码器、解码器处理
    def __init__(self, input_size, output_size, latent_size, hidden_size):  # 这个__init__括号里的 参数  顺序可以随意
        super(AE, self).__init__()  # 同样，继承所定义的AE这个类
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 确定模型，导入已训练模型（如有）
input_size = output_size = 28 * 28  # 原始图片和生成图片的维度  （mnist数据集每张图片包含28像素*28像素)
hidden_size = 128  # encoder和decoder分别中间层的维度
latent_size = 16

model = AE(input_size, output_size, latent_size, hidden_size)  # 具象化模型，里面4个变量的顺序可以随机
# loss_BCE = torch.nn.BCELoss()  # 交叉熵，衡量各个像素原始数据与重构数据的误差
loss_MSE = torch.nn.MSELoss()  # 均方误差，可作为交叉熵替代使用.衡量各个像素原始数据与重构数据的误差
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 优化器
epochs = 20  # 训练轮数                                                                   # 每轮训练样本数

'训练及测试'
loss_history = {'train': [], 'test': []}  # 创建了一个字典，存储训练的损失和测试的损失，其中train和test一开始都为空

for epoch in range(epochs):
    # 训练
    train_loss = 0  # 一开始训练的损失为0，训练的样本也为0
    train_sample = 0
    data1 = tqdm(train_loader, desc="[train]epoch:{}".format(epoch + 1))  # tqdm 会用在文件读取操作中，创建一个进度条，以展示文件读取进度；desc参数表示进度条前缀文字
    # epoch本身从0开始，+1后使进度条前缀文字从epoch1开始显示

    for imgs, lbls in data1:
        bs = imgs.shape[0]  # 获取数据,用shape获取图片的形状大小，并辅助给bs
        imgs = imgs.view(bs, input_size)  # imgs:(bs,28*28)，修改图片的形状，确保张量的形状与模型输入期望的形状一致

        output = model(imgs)  # 模型运算
        loss = loss_MSE(output, imgs)  # 计算损失,重构与原始数据的差距，且顺序不能换，输出output写前面，输入imgs写后面；output也就是re_imgs即重构后的图片

        optimizer.zero_grad()  # 梯度清零，反向传播，参数优化
        loss.backward()
        optimizer.step()

        # 计算平均损失，设置进度条
        train_loss = train_loss + loss.item()  # 用.item( ) 的方式获取张量loss中的数值；  在深度学习的训练循环中，经常会看到.item()用来把损失函数的输出 转换为一个标准Python数值，用于记录或输出显示。
        train_sample = train_sample + bs
        data1.set_postfix({'loss': train_loss})  # data表示一个 tqdm 进度条的实例，通常用于循环迭代（比如一个 DataLoader）
        # set_postfix( ) 方法用于在进度条中添加字符串后缀以显示额外的信息，这些信息将在每次迭代时更新。

    # 每个epoch记录总损失
    loss_history['train'].append(train_loss)  # 损失值除以样本数表示的是平均损失值（average loss），也被称为损失函数的期望或平均损失
    # 这个量化指标提供了关于模型性能在单个数据点上表现的信息，因此是一个实用的性能度量，比单纯的总损失更加直观

    # # 测试
    # test_loss = 0  # 每个epoch重置损失，设置进度条
    # test_sample = 0
    # data2 = tqdm(test_loader, desc="[test]epoch:{}".format(epoch + 1))
    #
    # for imgs, lbls in data2:
    #     bs = imgs.shape[0]
    #     imgs = imgs.view(bs, input_size)
    #
    #     output = model(imgs)
    #     loss = loss_MSE(output, imgs)
    #
    #     test_loss = test_loss + loss.item()
    #     test_sample = test_sample + bs
    #     data2.set_postfix({'loss': test_loss / test_sample})
    #
    # # 每个epoch记录总损失
    # loss_history['test'].append(test_loss / test_sample)
