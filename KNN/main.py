from sklearn.datasets import load_iris
import numpy as np


iris=load_iris()            # load iris dataset
X=iris.data                 # iris input data (150*4)
y=iris.target               # iris target data (150*1)
y_name=iris.target_names    # iris name of target data (3*1)

# split train and target data by index of data (multiples of 15 for test data)
X_train= np.array([X[i] for i in range(0, 150) if (i+1) % 15 != 0])
X_test=np.array([X[i] for i in range(0, 150) if (i+1) % 15 == 0])
y_train=np.array([y[i] for i in range(0, 150) if (i+1) % 15 != 0])
y_test=np.array([y[i] for i in range(0, 150) if (i+1) % 15 == 0])


from KNN import KNN          # import KNN.py


# Majority Vote
for k in [3,5,10]:

    print("k=",k, " (Majority Vote)")

    clf=KNN(k,X_train,y_train,y_name)

    for i in range(len(y_test)):
        #target_result = clf.majority_vote(X_test[i],iris.data,iris)
        print("Test Data Index: ", i, " Computed class: ", clf.majority_vote(X_test[i], iris.data, iris),
              ",   True class: ", y_name[y_test[i]])

    print("\n")

# Weighted Majority Vote
for k in [3,5,10]:
    print("k =", k, " (Weighted Majority Vote)")

    clf = KNN(k, X_train, y_train, y_name)

    for i in range(len(y_test)):
        print("Test Data Index: ", i, " Computed class: ",
              clf.weighted_majority_vote(X_test[i], iris.data,iris),
              ",   True class: ", y_name[y_test[i]])

    print("\n")

