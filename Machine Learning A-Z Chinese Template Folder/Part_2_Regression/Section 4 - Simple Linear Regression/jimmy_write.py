import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
#導入數據
data_set = pd.read_csv('Salary_Data.csv')
#做出應變量跟自變量
#自變量
dependent_value = data_set.iloc[:,0].values
#因變量
independent_value = data_set.iloc[:,1].values

#因無缺失數據或文字數據,所以我們可以直接做訓練集跟測試集
dependent_train,dependent_test,independent_train,independent_test = train_test_split(
    dependent_value,independent_value,test_size=0.2,train_size=0.8,random_state=0
)
#標值準話進行特徵縮放,只能使用二維陣列,將一維陣列進行轉換
sc_x = StandardScaler()
dependent_train = np.array(dependent_train).reshape(len(dependent_train),1)
dependent_test = np.array(dependent_test).reshape(-1,1)
sc_x.fit(dependent_train)
dependent_train = sc_x.transform(dependent_train)
dependent_test = sc_x.transform(dependent_test)
independent_train = np.array(independent_train).reshape(-1,1)
independent_test = np.array(independent_test).reshape(-1,1)
sc_y = StandardScaler()
sc_y.fit(independent_train)
independent_train = sc_y.transform(independent_train)
independent_test = sc_y.transform(independent_test)
#創建線性回歸物件
linear_regression = LinearRegression()
#擬合我們的線性回歸的訓練集資料
linear_regression.fit(dependent_train,independent_train)
#預測結果
y_pred = linear_regression.predict(dependent_test)
y_train_pred = linear_regression.predict(dependent_train)

#在畫圖之前先將特徵縮放回歸到原始數據
independent_test = sc_y.inverse_transform(independent_test)
y_pred = sc_y.inverse_transform(y_pred)
independent_train = sc_y.inverse_transform(independent_train)
dependent_train = sc_x.inverse_transform(dependent_train)
dependent_test = sc_x.inverse_transform(dependent_test)
y_train_pred = sc_y.inverse_transform(y_train_pred)

#使用matplotlib.pyplot畫出我們的每一個點
#先畫我們訓練集的點：
plt.scatter(dependent_train,independent_train,color='red')
#畫出我們機器學習的預測器
plt.plot(dependent_train,y_train_pred,color='blue')
plt.title('Salary VS experience (testing set)')
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()

#使用matplotlib.pyplot畫出我們的每一個點
#先畫我們測試集的點：
plt.scatter(dependent_test,independent_test,color='red')
#畫出我們機器學習的預測器
plt.plot(dependent_test,y_pred,color='blue')
plt.title('Salary VS experience (testing set)')
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()