# 1. 
import pandas as pd
import numpy as np

dataset = pd.read_csv('./dataset/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# 类别数据数字化
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# #可以替代上一段,适用于skleran version 0.22
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# columntransformer = ColumnTransformer(
#     transformers=[("onehot",OneHotEncoder(),[3])],remainder="passthrough"
# )
# X = columntransformer.fit_transform(X)

X = X[:, 1:] # 躲避虚拟变量陷阱

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# 2.多元线性回归模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print(Y_pred)