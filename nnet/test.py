import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers

# 加载数据
data = pd.read_excel('data.xlsx')

# 提取输入特征和输出标签
X = data.iloc[:, :2].values
y = data.iloc[:, 2].values

# 数据归一化
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 划分训练集和测试集
train_size = int(0.8 * len(data))
train_X, test_X = X_scaled[:train_size], X_scaled[train_size:]
train_y, test_y = y_scaled[:train_size], y_scaled[train_size:]

# 构建神经网络模型
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
opt = Adam(learning_rate=0.00001)
model.compile(loss='mse', optimizer=opt)

# 提前停止回调
early_stop = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001)

# 训练模型
history = model.fit(train_X, train_y, epochs=400, batch_size=64, verbose=1, validation_split=0.2, callbacks=[early_stop])

# 验证模型
train_predictions = model.predict(train_X)
test_predictions = model.predict(test_X)

# 计算指标


def index_of_agreement(predicted, observed):
    return 1 - (np.sum((predicted - observed) ** 2)) / (np.sum((np.abs(predicted - np.mean(observed)) + np.abs(observed - np.mean(observed))) ** 2))


def root_mean_square_error(predicted, observed):
    return np.sqrt(mean_squared_error(observed, predicted))


def relative_standard_deviation(predicted):
    return np.std(predicted) / np.mean(predicted)

# 打印指标


for dataset_name, predicted, actual in [("Training", train_predictions, train_y), ("Test", test_predictions, test_y)]:
    print(f"{dataset_name} set:")
    print(f"Index of Agreement: {index_of_agreement(predicted, actual)}")
    print(f"Root Mean Square Error: {root_mean_square_error(predicted, actual)}")
    print(f"Relative Standard Deviation: {relative_standard_deviation(predicted)}")

# 生成预测
predictions = model.predict(test_X)
predictions = scaler_y.inverse_transform(predictions)

# 绘制真实值和预测值
plt.figure()
plt.plot(test_y, 'b', label='Real')
plt.plot(predictions, 'r', label='Predicted')
plt.legend()
plt.title('Real and Predicted Values')
plt.show()

# 绘制训练集的损失曲线
plt.figure()
plt.plot(history.history['loss'], 'b', label='Training Loss')
plt.legend()
plt.title('Loss for Training Set')
plt.show()
