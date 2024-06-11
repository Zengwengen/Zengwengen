# 线性判别法
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.metrics import accuracy_score
import numpy as np
x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = LinearDiscriminantAnalysis()
clf.fit(x, y)
y_pred = clf.predict([[-0.8, -1]])
# accuracy = accuracy_score(y, y_pred)
print(y_pred)
