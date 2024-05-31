import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data_set = pd.read_csv('Data.csv')
x = data_set.iloc[:,:-1]
y = data_set.iloc[:,-1]

imputer = SimpleImputer(strategy='mean')
x.iloc[:,1:3] = imputer.fit_transform(x.iloc[:,1:3])

#分類數據
#將文字數據進行numpy陣列轉換,以方便後面的機器學習的運算
label_encoder_x = LabelEncoder()
x.iloc[:,0] = label_encoder_x.fit_transform(x.iloc[:,0])
"""
0       0  44.000000  72000.000000
1       2  27.000000  48000.000000
2       1  30.000000  54000.000000
3       2  38.000000  61000.000000
4       1  40.000000  63777.777778
5       0  35.000000  58000.000000
6       2  38.777778  52000.000000
7       0  48.000000  79000.000000
8       1  50.000000  83000.000000
9       0  37.000000  67000.000000
"""
onehotencoder = OneHotEncoder()
column_transformer = ColumnTransformer(transformers=[('encoder', onehotencoder, [0])]
                                       ,remainder='passthrough')

x = column_transformer.fit_transform(x)
x = x.astype(int)

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

#同一筆數據裡自變量跟應變量訓練集跟測試集的產生
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=0)

#標準化特徵縮放 (訓練集跟測試集)
sc_x = StandardScaler()
sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test = sc_x.transform(x_test)
print(x_train)
print(x_test)