import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data_set = pd.read_csv('50_Startups.csv')
"""
 R&D Spend  Administration  Marketing Spend       State     Profit
0  165349.20       136897.80        471784.10    New York  192261.83
1  162597.70       151377.59        443898.53  California  191792.06
2  153441.51       101145.55        407934.54     Florida  191050.39
3  144372.41       118671.85        383199.62    New York  182901.99
4  142107.34        91391.77        366168.42     Florida  166187.94

"""
#先做出自變量跟應變量
#以我們的數據來說,我們的profit是我們要預測的目標（因變量）,而前面四欄位是我們可以使用訓練的自變量
dep_data = data_set.iloc[:,0:4]
indep_data = data_set.iloc[:,-1]
#先做缺失數據跟虛擬變量轉換因目前數據無缺失數據，所我們對分類自變量進行虛擬變量的轉換
#你的數據的虛擬變量設定要根據欄位進行設定
label_encoder = LabelEncoder()
label_encoder.fit(dep_data.iloc[:,-1])
dep_data.iloc[:,-1] = label_encoder.transform(dep_data.iloc[:,-1])
print(dep_data)
onehotencoder = OneHotEncoder()
column_transformer = ColumnTransformer(transformers=[("onehotencoder",onehotencoder,[3])],
                                       remainder='passthrough')
column_transformer.fit(dep_data)
dep_data = column_transformer.transform(dep_data)
dep_data = dep_data.astype(int)
dep_data = dep_data[:,1:6]

#劃分訓練集跟測試集：
dep_train,dep_test,indep_train,indep_test = train_test_split(dep_data,indep_data,
                                                             test_size=0.2,train_size=0.8,
                                                             random_state=0)

sc_x = StandardScaler()
sc_x.fit(dep_train)
dep_train = sc_x.transform(dep_train)
dep_test = sc_x.transform(dep_test)

#因為因變量唯一為陣列,但特徵縮放只能放在二維陣列,所以我們要先進行轉換
indep_train = np.array(indep_train).reshape(-1,1)
indep_test = np.array(indep_test).reshape(-1,1)
sc_y = StandardScaler()
sc_y.fit(indep_train)
indep_train = sc_y.transform(indep_train)
indep_test = sc_y.transform(indep_test)

