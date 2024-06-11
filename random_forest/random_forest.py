import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import numpy as np
import chardet


def train_model(data_path, save_path):

    # 读取文件并检测编码
    with open(data_path, 'rb') as f:
        result = chardet.detect(f.read())

    # 获取检测到的编码方式
    encoding = result['encoding']

    # 读取数据集
    data = pd.read_csv(data_path, encoding=encoding)
    data[data < 0] = 0
    # 准备特征和目标变量
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建随机森林回归模型，设置决策树数量为10
    rf = RandomForestRegressor(n_estimators=10)

    # 拟合模型
    rf.fit(X_train, y_train)

    # 保存模型
    joblib.dump(rf, save_path)

    # 计算训练集的均方根误差
    train_predictions = rf.predict(X_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_rmse = np.sqrt(train_mse)
    print("Train RMSE:", train_rmse)

    # 计算测试集的均方根误差
    test_predictions = rf.predict(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    print("Test RMSE:", test_rmse)

    # 绘制训练集的预测结果与真实值的对比图
    plt.scatter(y_train, train_predictions)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='black', linewidth=1)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Training Set - True Values vs Predictions (Train RMSE: {:.4f})'.format(train_rmse))
    plt.show()

    # 绘制测试集的预测结果与真实值的对比图
    plt.scatter(y_test, test_predictions)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linewidth=1)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Test Set - True Values vs Predictions (Test RMSE: {:.4f})'.format(test_rmse))
    plt.show()


def predict_new_data(model_path, new_data_path):
    # 读取文件并检测编码
    with open(new_data_path, 'rb') as f:
        result = chardet.detect(f.read())

    # 获取检测到的编码方式
    encoding = result['encoding']

    # 读取新的表格数据
    new_data = pd.read_csv(new_data_path, encoding=encoding)

    # 准备特征变量
    X_new = new_data.iloc[:, :-1]

    # 加载训练好的随机森林模型
    rf = joblib.load(model_path)

    # 进行预测
    predictions = rf.predict(X_new)
    # 创建包含预测结果的数据框
    output_data = pd.DataFrame(predictions, columns=['Predicted Output'])

    # 将预测结果保存为 Excel 文件
    output_data.to_csv('predict.csv', index=False)

    # 计算均方根误差
    rmse = np.sqrt(mean_squared_error(new_data.iloc[:, -1], predictions))
    print("RMSE:", rmse)

    # 计算相对标准偏差（RSD）
    rsd = np.std(predictions) / np.mean(predictions)
    print("RSD:", rsd)

    # 计算指标准偏差的逆向分析（IA）
    ia = 1 / rsd
    print("IA:", ia)

    # 绘制新数据的预测结果与真实值的对比图
    plt.plot(range(len(predictions)), predictions, label='Predictions')
    plt.plot(range(len(predictions)), new_data.iloc[:, -1], label='True Values')
    plt.xlabel('Data Index')
    plt.ylabel('Values')
    plt.title('New Data - Predictions vs True Values (RMSE: {:.4f}, RSD: {:.4f}, IA: {:.4f})'.format(rmse, rsd, ia))
    plt.legend()
    plt.show()
