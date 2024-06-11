import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms

# 读取数据
data = pd.read_excel("data.xlsx")
data = data.values.flatten()

# 数据预处理
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, 1))

# 将数据划分为训练集和测试集
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]


# 定义适应度函数
def evaluate(individual):
    # 将个体转换为LSTM模型的参数
    n_units = individual[0]
    n_epochs = individual[1]
    batch_size = individual[2]

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(n_units, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练模型
    model.fit(train_data, train_data, epochs=n_epochs, batch_size=batch_size, verbose=0)

    # 使用训练好的模型进行预测
    train_predict = model.predict(train_data)
    test_predict = model.predict(test_data)

    # 计算均方根误差
    train_mape = np.mean(np.abs((train_data - train_predict) / np.where(train_data!=0, train_data, 1))) * 100
    test_rmse = np.sqrt(mean_squared_error(test_data, test_predict))

    # 计算平均绝对百分比误差
    train_mape = np.mean(np.abs((train_data - train_predict) / np.where(train_data!=0, train_data, 1))) * 100
    test_mape = np.mean(np.abs((test_data - test_predict) / np.where(test_data != 0, test_data, 1))) * 100
    # 返回适应度值和平均绝对百分比误差
    return test_rmse, test_mape


# 创建遗传算法的工具箱
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_units", np.random.randint, 1, 50)  # 增加LSTM单元数的范围
toolbox.register("attr_epochs", np.random.randint, 10, 200)  # 增加训练轮数的范围
toolbox.register("attr_batch_size", np.random.randint, 1, 50)  # 增加批处理大小的范围
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_units, toolbox.attr_epochs, toolbox.attr_batch_size), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=10, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 设置种群大小和迭代次数
population_size = 5
generations = 2

# 创建初始种群
population = toolbox.population(n=population_size)

# 记录每一代的最优适应度值和平均绝对百分比误差
best_fitness_values = []
best_mape_values = []

# 运行遗传算法
for generation in range(generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

    # 记录当前代的最优适应度值和平均绝对百分比误差
    best_fitness = tools.selBest(population, k=1)[0].fitness.values[0]
    best_mape = tools.selBest(population, k=1)[0].fitness.values[1]
    best_fitness_values.append(best_fitness)
    best_mape_values.append(best_mape)

# 打印最优个体
best_individual = tools.selBest(population, k=1)[0]
print("Best individual:", best_individual)

# 使用最优个体构建LSTM模型并进行预测
n_units = best_individual[0]
n_epochs = best_individual[1]
batch_size = best_individual[2]

model = Sequential()
model.add(LSTM(n_units, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_data, train_data, epochs=n_epochs, batch_size=batch_size, verbose=0)

train_predict = model.predict(train_data)
test_predict = model.predict(test_data)

# 反标准化预测结果
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# 计算均方根误差
train_rmse = np.sqrt(mean_squared_error(train_data, train_predict))
test_rmse = np.sqrt(mean_squared_error(test_data, test_predict))

# 计算平均绝对百分比误差
train_mape = np.mean(np.abs((train_data - train_predict) / np.where(train_data!=0, train_data, 1))) * 100

test_mape = np.mean(np.abs((test_data - test_predict) / np.where(test_data!=0, test_data, 1))) * 100

# 绘制训练集和测试集的实际值和预测值曲线，并显示平均绝对百分比误差
plt.subplot(2, 1, 1)
plt.plot(train_data, label='Actual')
plt.plot(train_predict, label='Predicted')
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("LSTM Prediction - Training Set")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(test_data, label='Actual')
plt.plot(test_predict, label='Predicted')
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("LSTM Prediction - Test Set")
plt.legend()

plt.tight_layout()

# 显示平均绝对百分比误差
plt.text(0.5, 0.95, "Train MAPE: {:.2f}%".format(train_mape), ha='center', va='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.05, "Test MAPE: {:.2f}%".format(test_mape), ha='center', va='center', transform=plt.gca().transAxes)

plt.show()

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
