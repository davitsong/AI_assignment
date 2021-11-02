#import from parent directory
import sys, os
sys.path.append(os.pardir)

#import module for train and test mnist
import numpy as np
from dataset.mnist import load_mnist
from KNN import KNN
import time


# load mnist / to do handcraft, setting flatten: False normarlize: False
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=False, normalize=False)


#############################################HANDCRAFT############################################################

# initailize variables and arrays
matrix = []
matrix1 = []
mat1=[]
k = 0
sum1 = 0

# hand craft for x_train data, sum each row of data and make new array // x_train(60000,784) to mat1(60000,28)
for t in range(60000):
    for i in x_train[t][k]:
        matrix.append([])
        for j in i:
            if (j > 0):
                matrix[k].append(j)
        if matrix[k] == []:
            matrix[k].append(0)
        k += 1

    k=0

    for i in matrix:
        for j in i:
            sum1 += j

        matrix1.append(sum1)
        sum1 = 0

    mat1.append(matrix1)
    matrix1=[]
    matrix=[]

# hand craft for x_test data, sum each row of data and make new array t_train(10000,784) to mat2(10000,28)

# initailize variables and arrays
matrix = []
matrix2 = []
mat2=[]
k = 0
sum1 = 0

for t in range(10000):
    for i in x_test[t][k]:
        matrix.append([])
        for j in i:
            if (j > 0):
                matrix[k].append(j)
        if matrix[k] == []:
            matrix[k].append(0)
        k += 1

    k = 0

    for i in matrix:
        for j in i:
            sum1 += j

        matrix2.append(sum1)
        sum1 = 0

    mat2.append(matrix2)
    matrix2 = []
    matrix = []

#########################################HANDCRAFT#############################################################

# convert mat1, mat2 array to np.array format and save x_train,x_test
x_train=np.array(mat1)
x_test=np.array(mat2)



#  #check data size
# print("x_train.shape:", x_train.shape)
# print("t_train.shape:", t_train.shape)
# print("x_test.shape:", x_test.shape)
# print("t_test.shape", t_test.shape)
#
# print("x_train.dim:", x_train.ndim)
# print("t_train.dim:", t_train.ndim)
# print("x_test.dim:", x_test.ndim)
# print("t_test.dim", t_test.ndim)


# set new data set for random sampling
idx1 = np.random.randint(0, x_train.shape[0], size=60000)
set_x_train = x_train[idx1]
set_t_train = t_train[idx1]

idx2 = np.random.randint(0, x_test.shape[0], size=100)
set_x_test = x_test[idx2]
set_t_test = t_test[idx2]


# #check data size
# print("set_x_train:", set_x_train.shape)
# print("set_x_test:", set_x_test.shape)



k = 3
check_acu = 0
print("\n")
print("k =", k)

start_time = time.time()
# use weighted majority vote from KNN algorithm and check the trained to label
mnist_k = KNN(k, set_x_train, set_t_train, set_t_test)
for i in range(set_x_test.shape[0]):
    tested = mnist_k.weighted_majority_vote(set_x_test[i], set_x_train, set_t_train)
    print("no.",i+1,"\t" ,idx2[i],"\tth data ", " \tresult:", tested, " \tlabel: ", set_t_test[i])
    if int(tested) == set_t_test[i]:
        check_acu += 1


# reset for next train
mnist_k.reset()

# running time for knn
print("Time :" + str(time.time() - start_time) + "sec")

# print accuracy
print("accuracy:" + str((check_acu / set_t_test.shape[0])))
