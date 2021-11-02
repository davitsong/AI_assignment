import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# 직접 만든 LinearRegression
class self_LinearRegression():

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # 모델 fitting
    def fit(self, X, Y):
        self.m, self.n = X.shape
        # weight 초기화
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.W_update()

        return self

    # gradient descent learning으로 파라미터 수정
    def W_update(self):
        Y_pred = self.predict(self.X)
        dW = (- (2 * (self.X.T).dot(self.Y - Y_pred))) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m
        # weight 최신화
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    # 활성화 함수
    def predict(self, X):
        return X.dot(self.W) + self.b

def main():
    data = pd.read_csv("insurance.csv")
    data.loc[:, ['sex', 'smoker', 'region']] = data.loc[:, ['sex', 'smoker', 'region']].apply(
        LabelEncoder().fit_transform)
    # data=data.drop(['region','sex'],axis=1) # Feature 삭제 (정확도가 낮아져서 미포함)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    # 데이터셋 분류
    train_data_inputs, test_data_inputs, train_data_outputs, test_data_outputs = train_test_split(X, Y,test_size= 0.3, random_state=5)

    # Model 학습
    model = self_LinearRegression(iterations=300000, learning_rate=0.000375)
    model.fit(train_data_inputs, train_data_outputs)

    # 예측값 비교분석
    Y_pred = model.predict(test_data_inputs)
    print(list(zip(Y_pred,test_data_outputs)))
    print(r2_score(Y_pred,test_data_outputs))

if __name__ == "__main__":
    main()
