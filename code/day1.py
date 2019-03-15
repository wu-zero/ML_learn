# 1
import numpy as np
import pandas as pd

# 2. 导入数据集
dataset = pd.read_csv('../dataset/Data.csv')
X = dataset.iloc[:, :-1].values  # iloc 行列号索引 # values 去掉index、colums,变numpy
Y = dataset.iloc[:, 3].values
# print(X)
# print(Y)
# print('='*20)

# 3. 处理丢失数据
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])

#可以替代上一段,适用于skleran version 0.22
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# print(X)
# print(Y)
# print('='*20)

# 4. 解析分类数据
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# # 处理X
# LabelEncoder_X = LabelEncoder()  # 标签变数值
# X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features=[0])  # 数字变one-hot编码
# X = onehotencoder.fit_transform(X).toarray()  # 注意toarray()

#可以替代上一段,适用于skleran version 0.22
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columntransformer = ColumnTransformer(
    transformers=[("onehot", OneHotEncoder(), [0])], remainder="passthrough"
)
X = columntransformer.fit_transform(X).astype(float)  # astype()


# 处理Y
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(X)
print(Y)
print('='*20)

# 5. 拆分数据集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(X_train)
# print(X_test)
# print('='*20)

# 6. 特征量化 标准化均值0,方差1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# print(X_train)
# print(X_test)
# print('='*20)
