import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data = pd.read_csv('train_data.csv')
exclude_columns = ['P_009_HAY10EZ010G00', 'P_009_HAY10EZ020G00', 'P_009_HAY10EZ030G00', 'P_009_MKC02EZ001G00', 'P_009_HAY01EZ001G00', 'P_009_AGC01EM001G00', 'P_009_AGC01EM001G00',
                   'P_009_LCB20AP001G00', 'P_009_MAG01CP001G00', 'P_009_MAJ10AP301G00',
                   'P_009_MAJ10AP001G00', 'P_009_MAJ20AP301G00', 'P_009_MAJ20AP001G00', 'P_009_MAJ30AP001G00', 'P_009_PGC20AP001G00', 'P_009_PCC10AP001G00', 'P_009_PCC10AP301G00',
                   'P_009_PCC20AP001G00', 'P_009_MKF11AP301G00', 'P_009_MKF11AP001G00',
                   'P_009_LCB10AP201G00', 'P_009_LCB10AP301G00', 'P_009_MKF12AP301G00', 'P_009_MKF12AP001G00', 'P_009_HFC10AJ001G00', 'P_009_LCB20AP201G00', 'P_009_LAC30AP201G00',
                   'P_009_MKA01GS001G00', 'P_009_MAX10AP001G00', 'P_009_MAX20AP001G00',
                   'P_009_MAX10AP201G00', 'P_009_MKA01CE104G00', 'P_009_MAY10EZ013G00', 'P_009_MAY10EZ014G00', 'P_009_MAY10EZ015G00', 'P_009_HFC30CF001G00', 'P_009_MAY01EZ001G00',
                   'P_009_MAY10EZ001G00', 'P_009_MAY10EZ002G00', 'P_009_MAY10EZ003G00',
                   'P_009_MAY10EZ004G00', 'P_009_MAY10EZ005G00', 'P_009_MAY10EZ006G00', 'P_009_MAY10EZ007G00', 'P_009_MAY10EZ009G00', 'P_009_MAY10EZ010G00', 'P_009_MAY10EZ011G00',
                   'P_009_MAY10EZ012G00', 'P_009_HFC20AJ301G00', 'P_009_HFC20AJ001G00',
                   'P_009_HFC30AJ001G00', 'P_009_HFD10AN001G00', 'P_009_HFD10AN301G00', 'P_009_HTF40CF001G00', 'P_009_HTQ10AP001G00', 'P_009_HTQ20AP001G00', 'P_009_HNC20AN301G00',
                   'P_009_HNC20AN001G00', 'P_009_LAE11CF001G00', 'P_009_LAE12CF001G00',
                   'P_009_LAE22CF001G00', 'P_009_LAE21CF001G00', 'P_009_HTF20CT002G00', 'P_009_HTF20CT007G00', 'P_009_HTF20CT008G00', 'P_009_HTF20CT009G00', 'P_009_HTF20CT003G00',
                   'P_009_HTF20CT004G00', 'P_009_HTF40CP001G00', 'P_009_HTF40CP002G00',
                   'P_009_HFC40AJ301G00', 'P_009_HSG10AN001G00', 'P_009_HTF20CE001G00', 'P_009_HFC60AJ301G00', 'P_009_HFC60AJ001G00', 'P_009_HFC60CF001G00', 'P_009_HSG10AP101G00',
                   'P_009_HSG20AN001G00', 'P_009_HSG20AP101G00', 'P_009_LAF01CF001G00',
                   'P_009_LAF02CF001G00', 'P_009_SIS01CF003G00', 'P_009_SIS01CF002G00', 'P_009_HFC50AJ001G00', 'P_009_HTQ20AP101G00', 'P_009_HLB10AN001G00', 'P_009_HLB10AN301G00',
                   'P_009_HNC10AN301G00', 'P_009_HNC10AN001G00', 'P_009_HFB10AF001G00',
                   'P_009_HFB20AF001G00', 'P_009_HFB30AF001G00', 'P_009_HFB40AF001G00', 'P_009_HFB50AF001G00', 'P_009_HFB60AF001G00', 'P_009_HLC10AE002G00', 'P_009_HLC10AE301G00',
                   'P_009_HLC10AE001G00', 'P_009_LCB10AP001G00', 'P_009_MAJ30AP301G00',
                   'P_009_HTQ10AP101G00', 'P_009_HLA14CF002G00', 'P_009_HFC10AJ301G00', 'P_009_HTF20CT001G00', 'P_009_HTF11AP001G00', 'P_009_HTF12AP001G00', 'P_009_HTF13AP001G00',
                   'P_009_HTF14AP001G00', 'P_009_HTF11EZ001G00', 'P_009_HTF12EZ001G00',
                   'P_009_HTF13EZ001G00', 'P_009_HTF14EZ001G00', 'P_009_PAC10AP001G00', 'P_009_PAC20AP001G00', 'P_009_HTG10AN001G00', 'P_009_HTG20AN001G00', 'P_009_HTG10AN101G00',
                   'P_009_HTG20AN101G00', 'P_009_HAG03CL003G00', 'P_009_MAA30AA020G00', 'P_009_AGC01CE201G00', ]

# 检查列名是否存在，并删除存在的列
columns_to_drop = [col for col in exclude_columns if col in data.columns]
if columns_to_drop:
    data.drop(columns_to_drop, axis=1, inplace=True)
    print("成功删除以下列:", columns_to_drop)
else:
    print("没有找到需要删除的列")

# 保存特征列的列名
X = data
feature_columns = X.columns.tolist()

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=19, min_samples=2)  # 调整参数
dbscan.fit(X_scaled)
labels = dbscan.labels_
outliers = np.where(labels == -1)[0]  # 更新异常值判断条件

row_indices = np.arange(X_scaled.shape[0])  # 定义行的序数
column_indices = np.arange(X_scaled.shape[1])  # 定义列的序数

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111, projection='3d')

# 绘制正常点
normal_points = X_scaled[labels != -1]
x_normal, y_normal = np.meshgrid(column_indices, row_indices[:normal_points.shape[0]])
color_normal = np.array([68, 153, 69]) / 255  # 标准化 RGB 值
ax.scatter(x_normal.flatten(), y_normal.flatten(), normal_points.flatten(), c=[color_normal], marker='o', label='正常点', s=0.5)

# 绘制异常点
outlier_points = X_scaled[labels == -1]
x_outliers, y_outliers = np.meshgrid(column_indices, row_indices[normal_points.shape[0]:])
color_outliers = np.array([234, 120, 39]) / 255  # 标准化 RGB 值
ax.scatter(x_outliers.flatten(), y_outliers.flatten(), outlier_points.flatten(), c=[color_outliers], marker='*', label='离群点', s=5)


# 设置坐标轴标题和标签字体大小
ax.set_xlabel('列', fontsize=16, labelpad=10)  # 增加labelpad来调整标签位置
ax.set_ylabel('行', fontsize=16, labelpad=10)
ax.set_zlabel('数值大小', fontsize=16, labelpad=15)

# 设置图表标题和字体大小
ax.set_title('离群点检测', fontsize=16, x=0.5, y=0.9)  # 增加pad来调整标题位置

# 设置坐标轴刻度字号
ax.tick_params(axis='x', labelsize=16, pad=5)  # 增加pad来调整刻度位置
ax.tick_params(axis='y', labelsize=16, pad=5)
ax.tick_params(axis='z', labelsize=16, pad=5)

# 避免标签遮挡坐标轴
fig.tight_layout()

# 显示图例并设置绝对位置和文本大小
plt.legend(bbox_to_anchor=(0.8, 0.88), prop={'size': 14})

# 增加网格线
ax.grid(True)

plt.show()
