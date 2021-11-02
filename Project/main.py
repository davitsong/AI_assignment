import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
import numpy as np
from model import Linear_Regression, Ridge_Regression, Lasso_Regression, RandomForest_Regression, Polynomial_Regression
from data_analysis import data_analysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from self_linear import self_LinearRegression

# 데이터 분류 (8:2)
data = pd.read_csv('insurance.csv')
data.drop_duplicates() # 중복성 제거

data_analysis(data) # 데이터 분석 결과

data.loc[:,['sex','smoker','region']]= data.loc[:,['sex', 'smoker', 'region']].apply(LabelEncoder().fit_transform) # 범주형 데이터 처리
columns=data.columns

# 정규화 진행
StandardScaler()
standard_scaled_data= StandardScaler().fit_transform(data)
standard_scaled_data = pd.DataFrame(standard_scaled_data, columns = columns)

MinMaxScaler()
min_max_scaled_data= MinMaxScaler().fit_transform(data)
min_max_scaled_data = pd.DataFrame(min_max_scaled_data, columns = columns)

# data=data.drop(['region','sex'],axis=1) # Feature 삭제 (정확도가 낮아져서 미포함)

# Train Data, Test Data 분류
train_data,test_data= train_test_split(data, test_size=0.3, random_state=5)
train_data_inputs, train_data_outputs, test_data_inputs,test_data_outputs= (pd.DataFrame(train_data.iloc[:,:-1]),pd.DataFrame(train_data.iloc[:,-1]),pd.DataFrame(test_data.iloc[:,:-1]),pd.DataFrame(test_data.iloc[:,-1]))

standard_scaled_train_data,standard_scaled_test_data= train_test_split(standard_scaled_data, test_size=0.3, random_state=5)
standard_scaled_train_data_inputs, standard_scaled_train_data_outputs, standard_scaled_test_data_inputs,standard_scaled_test_data_outputs= (pd.DataFrame(standard_scaled_train_data.iloc[:,:-1]),pd.DataFrame(standard_scaled_train_data.iloc[:,-1]),pd.DataFrame(standard_scaled_test_data.iloc[:,:-1]),pd.DataFrame(standard_scaled_test_data.iloc[:,-1]))

min_max_scaled_train_data,min_max_scaled_test_data= train_test_split(min_max_scaled_data, test_size=0.3, random_state=5)
min_max_scaled_train_data_inputs, min_max_scaled_train_data_outputs, min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs= (pd.DataFrame(min_max_scaled_train_data.iloc[:,:-1]),pd.DataFrame(min_max_scaled_train_data.iloc[:,-1]),pd.DataFrame(min_max_scaled_test_data.iloc[:,:-1]),pd.DataFrame(min_max_scaled_test_data.iloc[:,-1]))


# OLS 분석
lm = sm.OLS(data['charges'], data[['age', 'bmi', 'children','sex','smoker','region']]).fit()
print(lm.summary())

# 모델별 정확도
print(" <--- 정규화 처리 전 --->")
print()
print(f" Linear_Regression: {Linear_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs)}")
print(f" Ridge_Regression: {Ridge_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs)}")
print(f" Lasso_Regression: {Lasso_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs)}")
print(f" RandomForest_Regression: {RandomForest_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs)}")
print(f" Polynomial_Regression: {Polynomial_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs)}")
print()

print(" <--- StandardScaler() 처리 후 --->")
print()
print(f" Linear_Regression: {Linear_Regression(standard_scaled_train_data_inputs,standard_scaled_train_data_outputs,standard_scaled_test_data_inputs,standard_scaled_test_data_outputs)}")
print(f" Ridge_Regression: {Ridge_Regression(standard_scaled_train_data_inputs,standard_scaled_train_data_outputs,standard_scaled_test_data_inputs,standard_scaled_test_data_outputs)}")
print(f" Lasso_Regression: {Lasso_Regression(standard_scaled_train_data_inputs,standard_scaled_train_data_outputs,standard_scaled_test_data_inputs,standard_scaled_test_data_outputs)}")
print(f" RandomForest_Regression: {RandomForest_Regression(standard_scaled_train_data_inputs,standard_scaled_train_data_outputs,standard_scaled_test_data_inputs,standard_scaled_test_data_outputs)}")
print(f" Polynomial_Regression: {Polynomial_Regression(standard_scaled_train_data_inputs,standard_scaled_train_data_outputs,standard_scaled_test_data_inputs,standard_scaled_test_data_outputs)}")
print()

print(" <--- MinMaxScaler() 처리 후 --->")
print()
print(f" Linear_Regression: {Linear_Regression(min_max_scaled_train_data_inputs,min_max_scaled_train_data_outputs,min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs)}")
print(f" Ridge_Regression: {Ridge_Regression(min_max_scaled_train_data_inputs,min_max_scaled_train_data_outputs,min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs)}")
print(f" Lasso_Regression: {Lasso_Regression(min_max_scaled_train_data_inputs,min_max_scaled_train_data_outputs,min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs)}")
print(f" RandomForest_Regression: {RandomForest_Regression(min_max_scaled_train_data_inputs,min_max_scaled_train_data_outputs,min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs)}")
print(f" Polynomial_Regression: {Polynomial_Regression(min_max_scaled_train_data_inputs,min_max_scaled_train_data_outputs,min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs)}")

# 모델별 결정계수 막대그래프 출력
models=['Linear','Ridge','Lasso','Polynomial','RandomForest']
original_score=[Linear_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs),Ridge_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs),Lasso_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs),Polynomial_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs),RandomForest_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs)]
standard_scaled_score=[Linear_Regression(standard_scaled_train_data_inputs,standard_scaled_train_data_outputs,standard_scaled_test_data_inputs,standard_scaled_test_data_outputs),Ridge_Regression(standard_scaled_train_data_inputs,standard_scaled_train_data_outputs,standard_scaled_test_data_inputs,standard_scaled_test_data_outputs),Lasso_Regression(standard_scaled_train_data_inputs,standard_scaled_train_data_outputs,standard_scaled_test_data_inputs,standard_scaled_test_data_outputs),Polynomial_Regression(standard_scaled_train_data_inputs,standard_scaled_train_data_outputs,standard_scaled_test_data_inputs,standard_scaled_test_data_outputs),RandomForest_Regression(standard_scaled_train_data_inputs,standard_scaled_train_data_outputs,standard_scaled_test_data_inputs,standard_scaled_test_data_outputs)]
min_max_scaled_score=[Linear_Regression(min_max_scaled_train_data_inputs,min_max_scaled_train_data_outputs,min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs),Ridge_Regression(min_max_scaled_train_data_inputs,min_max_scaled_train_data_outputs,min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs),Lasso_Regression(min_max_scaled_train_data_inputs,min_max_scaled_train_data_outputs,min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs),Polynomial_Regression(min_max_scaled_train_data_inputs,min_max_scaled_train_data_outputs,min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs),RandomForest_Regression(min_max_scaled_train_data_inputs,min_max_scaled_train_data_outputs,min_max_scaled_test_data_inputs,min_max_scaled_test_data_outputs)]
scores={'Original':original_score,'StandardScaler':standard_scaled_score,'MinMaxScaler':min_max_scaled_score}
df=pd.DataFrame(scores,columns=list(scores.keys()),index=models)
df.plot.barh()
plt.title('R2 Score')
plt.xlabel('R2')
plt.ylabel('Models')
plt.yticks(fontsize=7)
plt.legend(loc='center right', bbox_to_anchor=(1.12, 1.055),fontsize=9)
plt.show()



