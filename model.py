# 다양한 Regression 모델
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

def Linear_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs):
    linear = LinearRegression(fit_intercept=False).fit(train_data_inputs, train_data_outputs)
    return linear.score(test_data_inputs,test_data_outputs)

def Ridge_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs):
    ridge= Ridge(alpha=0.001).fit(train_data_inputs,train_data_outputs)
    return ridge.score(test_data_inputs,test_data_outputs)

def Lasso_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs):
    lasso= Lasso(alpha=0.001).fit(train_data_inputs,train_data_outputs)
    return lasso.score(test_data_inputs,test_data_outputs)

def RandomForest_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs):
    Rfr = rf(n_estimators=100, criterion='mse',     # n_estimators : randomforest 개수
              random_state=1,                       # random_state=1 seed 값 고정
              n_jobs=-1)                            # n_jobs=-1 컴퓨터의 모든 cpu 사용
    Rfr.fit(train_data_inputs,np.ravel(train_data_outputs))
    return Rfr.score(test_data_inputs, test_data_outputs)

def decision_tree_regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs):
    dtr = DecisionTreeRegressor()                   # 파라미터 설정 안할 시 overfitting됨!
    dtr.fit(train_data_inputs,train_data_outputs)
    return dtr.score(test_data_inputs,test_data_outputs)

def Polynomial_Regression(train_data_inputs,train_data_outputs,test_data_inputs,test_data_outputs):
    pol=PolynomialFeatures(degree=2)
    train_data_inputs = pol.fit_transform(train_data_inputs)
    test_data_inputs = pol.fit_transform(test_data_inputs)
    poly=LinearRegression(fit_intercept=False).fit(train_data_inputs,train_data_outputs)
    return poly.score(test_data_inputs,test_data_outputs)


