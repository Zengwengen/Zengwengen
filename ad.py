import random

import pandas as pd

# 读取表格数据
data = pd.read_excel('x.xlsx')  # 替换为你的表格文件名或路径

# 遍历每一列的数据
for column in data.columns:
    # 处理绝对值大于20的数
    new_column = data[column].apply(lambda x: x + random.uniform(5, 10) if x < -20 else (x - random.uniform(8, 15) if x > 20 else x))

    # 处理绝对值大于40的数
    new_column = new_column.apply(lambda x: x + random.uniform(20, 28) if x < -40 else (x - random.uniform(15, 25) if x > 40 else x))

    # 处理绝对值小于20的数
    new_column = new_column.apply(lambda x: x + random.uniform(0, 5) if abs(x) < 20 else x)

    # 将数据写入原有数据后一列
    new_column_name = column + '_new'
    data[new_column_name] = new_column

# 将结果写入新的表格文件
data.to_excel('modified_table.xlsx', index=False)
