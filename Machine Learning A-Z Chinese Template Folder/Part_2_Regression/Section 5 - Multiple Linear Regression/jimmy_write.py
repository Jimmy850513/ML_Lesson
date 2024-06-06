import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

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
#創建線性回歸器
linear_regression = LinearRegression()
#回歸器擬合我們的訓練集
linear_regression.fit(dep_train,indep_train)
indep_data_pred = linear_regression.predict(dep_test)
#把我們的特徵縮放進行反項縮放,把預測的跟測試集裡面的資料型比對
#先畫出用一般all-in的圖

def backward_elimination(X, y, significance_level=0.05):
    num_features = X.shape[1]
    for i in range(0, num_features):
        regressor_OLS = sm.OLS(y, X).fit()
        max_pvalue = max(regressor_OLS.pvalues)
        if max_pvalue > significance_level:
            max_pvalue_index = np.where(regressor_OLS.pvalues == max_pvalue)[0][0]
            X = np.delete(X, max_pvalue_index, axis=1)
    return X

#加上b0*全部都是1陣列的值,因為反向淘汰沒有這個b0係數
dep_train = np.append(np.ones((40,1)),dep_train,axis=1)

dep_opt = backward_elimination(dep_train,indep_train)

print(dep_opt.astype(int))
#創建完我們的線性回歸器
regresser_OLS = sm.OLS(endog=indep_train,exog=dep_opt)
#擬合回歸器
regresser_OLS = regresser_OLS.fit()

dep_test = np.append(np.ones((10,1)),dep_test,axis=1)

Y_pred2 = regresser_OLS.predict(dep_test[:,[0,3]])

"""
[104667.24473855 134150.43724489 135208.15316323  72170.15800112
 179090.58942745 109824.67436935  65644.17001221 100481.51709069
 111431.6872018  169438.29295526]

"""

# plt.figure(figsize=(10, 6))

# plt.scatter(indep_test, Y_pred2, color='blue', label='Predictions without feature selection')
# plt.scatter(indep_test, Y_pred2, color='red', label='Predictions with feature selection')
# plt.plot([min(indep_test), max(indep_test)], [min(indep_test), max(indep_test)], color='gray', linestyle='--')

# plt.title('Comparison of Predictions with and without Feature Selection')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.legend()
# plt.grid(True)
# plt.show()