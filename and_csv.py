import pandas as pd
"""拼接有标签数据"""
"""拼接后形式：{X_train,y_train
                X_test, y_test}"""
# 读取 CSV 文件
df1 = pd.read_csv('X_train.csv')  # 第一个文件
df2 = pd.read_csv('y_train.csv')   # 第二个文件
df3 = pd.read_csv('X_test.csv')  # 第三个文件
df4 = pd.read_csv('y_test.csv')   # 第四个文件
# print(df1.shape)
# print(df2.shape)
# print(df3.shape)
# print(df4.shape)
# print(df1.index.equals(df2.index))
# print(df3.index.equals(df4.index))

# 假设df1, df2, df3, df4已经加载

# 重置所有DataFrame的索引
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df3.reset_index(drop=True, inplace=True)
df4.reset_index(drop=True, inplace=True)

# 按列拼接第一对DataFrame，确保仅拼接一次
df_1_2_combined = pd.concat([df1, df2], axis=1)

# 按列拼接第二对DataFrame，确保仅拼接一次
df_3_4_combined = pd.concat([df3, df4], axis=1)

# 为 df_1_2_combined 赋予列名 X_i 和 Y
df_1_2_combined.columns = [f'X_{i}' for i in range(13)] + ['Y']

# 为 df_3_4_combined 赋予列名 X_i 和 Y
df_3_4_combined.columns = [f'X_{i}' for i in range(13)] + ['Y']

# 确认拼接后的DataFrame尺寸
print("第一对拼接后尺寸:", df_1_2_combined.shape)  # 预期为 (999或1389, 14)
print("第二对拼接后尺寸:", df_3_4_combined.shape)  # 同上

# 按行拼接得到最终DataFrame，确保仅拼接一次
final_df = pd.concat([df_1_2_combined, df_3_4_combined], axis=0, )

# 最终尺寸检查
print("最终DataFrame尺寸:", final_df.shape)  # 预期为 (2388, 14)

# 如果尺寸正确，保存到CSV
if final_df.shape == (2388, 14):
    final_df.to_csv('final_combined_data.csv', index=False)
else:
    print("尺寸不匹配预期，请检查数据和代码。")
# 如果需要，保存最终的DataFrame到CSV文件
final_df.to_csv('data.csv', index=False)

# 输出结果
print("完成数据集拼接")
