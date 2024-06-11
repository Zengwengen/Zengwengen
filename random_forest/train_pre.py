# 训练模型并保存
from random_forest import train_model, predict_new_data

train_model('data.csv', 'random_forest_model.pkl', )

# 预测新数据(重复预测只需运行以下代码)
predict_new_data('random_forest_model.pkl', 'new_data.csv')
