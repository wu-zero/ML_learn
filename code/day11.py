# 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../dataset/Social_Network_Ads.csv")
X = dataset.iloc[:, 2:4].values.astype(float)  # X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# K-NN进行训练
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# metric='minkowski', p=2  闵氏距离 p=2, 欧式距离
classifier.fit(X_train, Y_train)

# 预测
Y_pred = classifier.predict(X_test)

# 生成混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
