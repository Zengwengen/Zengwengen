import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers

# 读取数据
data = pd.read_excel('data.xlsx')

# 将数据转换为numpy数组
data = data.values

# 确定输入和输出变量的变化范围
print(f"化合物A的范围: {data[:,0].min()} 到 {data[:,0].max()}")
print(f"底物的范围: {data[:,1].min()} 到 {data[:,1].max()}")
print(f"生物质的范围: {data[:,2].min()} 到 {data[:,2].max()}")

# 如果检测到异常值，根据需要纠正数据
data[data < 0] = 0

# 将数据分为两个子集：训练和测试
train_data = data[:1200]
test_data = data[1201:]

# 归一化和标准化
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

scaler2 = StandardScaler()
train_data = scaler2.fit_transform(train_data)
test_data = scaler2.transform(test_data)

# 设计神经网络
model = Sequential()
model.add(Dense(400, input_dim=2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='linear'))

# 编译模型
opt = Adam(learning_rate=0.00001)  # 使用自定义学习率的Adam优化器
model.compile(loss='mse', optimizer=opt)

# 早期停止回调
early_stop = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001)

# 训练模型
history = model.fit(train_data[:,:2], train_data[:, 2], epochs=400, batch_size=64, verbose=1, validation_split=0.2, callbacks=[early_stop])

# 验证模型
train_predictions = model.predict(train_data[:, :2])
test_predictions = model.predict(test_data[:, :2])

# 计算误差指数


def index_of_agreement(predicted, observed):
    return 1 - (np.sum((predicted - observed) ** 2)) / (np.sum((np.abs(predicted - np.mean(observed)) + np.abs(observed - np.mean(observed))) ** 2))


def root_mean_square_error(predicted, observed):
    return np.sqrt(mean_squared_error(observed, predicted))


def relative_standard_deviation(predicted):
    return np.std(predicted) / np.mean(predicted)


# 计算并打印IA，RMS和RSD
for dataset_name, predicted_biomass, actual_biomass in [("训练", train_predictions, train_data[:,2]), ("测试", test_predictions, test_data[:,2])]:
    print(f"{dataset_name}集:")
    print(f"一致性指数: {index_of_agreement(predicted_biomass, actual_biomass)}")
    print(f"均方根误差: {root_mean_square_error(predicted_biomass, actual_biomass)}")
    print(f"相对标准偏差: {relative_standard_deviation(predicted_biomass)}")

# 生成400个预测
predictions_400 = model.predict(test_data[:, :2])
predictions_400 = scaler.inverse_transform(np.column_stack((test_data[:,:2], predictions_400)))[:,2]

# 打印400个预测

# 绘制测试集的实际和预测生物质
plt.figure()
plt.plot(test_data[:,2], 'b', label='actual')
plt.plot(test_predictions, 'r', label='predict')
plt.legend()
plt.title('actual-predict')
plt.show()

# 绘制训练集的损失
plt.figure()
plt.plot(history.history['loss'], 'b', label='train-loss')
plt.legend()
plt.title('Loss in train')
plt.show()






