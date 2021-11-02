import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

def data_analysis(data):
    print("Data Head")
    print(data.head())  # 데이터의 0~5번째 부분 (샘플)
    print("====================================")
    print("Data Info")
    print(data.info())  # 데이터의 형태
    print("====================================")
    print("Numerical Datas")
    print(data.describe())  # 숫자형 데이터
    print("====================================")
    print("Object Datas")
    cat_features = data.dtypes[data.dtypes == 'object'].index
    print(data[cat_features].describe())
    print("====================================")
    print("Sex , Smoker, Region")
    print(data.sex.value_counts())
    print(data.smoker.value_counts())
    print(data.region.value_counts())
    print(data.region.value_counts())
    print(data.region.value_counts())
    print(data.region.value_counts())
    print("====================================")
    print("Data visualize")
    data.loc[:, ['sex', 'smoker', 'region']] = data.loc[:, ['sex', 'smoker', 'region']].apply(
        LabelEncoder().fit_transform) #Label Encoder를 사용하여 sex, smoker, region을 바꾼다
    le = LabelEncoder()
    le.fit(data.sex)
    le.fit(data.smoker)
    le.fit(data.region)
    data.hist(bins=10, figsize=(10,10))
    plt.show()
    print("====================================")



    scatter_matrix(data, figsize=(40, 40));     #각 features들의 관계를 그래프로 보여줌
    plt.figure(figsize=(12, 10))
    plt.show()

    ax = sns.heatmap(data.corr(), annot=True,cmap='Blues')   # 각 feature들의 상관관계를 히트맵으로 표현
    # 나이와 가격을 그래프를 흡연자 기준으로 그래프 그림
    plt.show()


    ax = sns.lmplot(x='age', y='charges', data=data, hue='smoker', palette='Set2')
    # bmi와 가격을 그래프를 흡연자 기준으로 그래프 그림
    plt.show()


    ax = sns.lmplot(x='bmi', y='charges', data=data, hue='smoker', palette='Set2')
    # children과 가격을 그래프를 흡연자 기준으로 그래프 그림
    plt.show()


    ax = sns.lmplot(x='children', y='charges', data=data, hue='smoker', palette='Set2')
    plt.show()
