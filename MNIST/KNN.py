import numpy as np

# Make KNN class
class KNN:

    # initialize class
    def __init__(self,k,x_train,t_train,label):

        self.k=k
        self.x_train=x_train
        self.t_train=t_train
        self.label=label

    # Euclidien distance metric by numpy
    def euclidean_distance(self,X1,X2):

        return np.sqrt(np.sum(np.power(X1-X2,2)))

    # manhattan_distance metric by numpy
    def manhattan_distance(self,X1,X2):

        distance=0
        for i in range(len(X1)):
            distance +=abs(X1[i]-X2[i])
        return distance

    # Weighted Majority Vote
    def weighted_majority_vote(self,s_point,points,data):

        # initialize distance_matrix numpy array size of points numpy array
        distance_matrix=np.zeros(len(points))

        # save distance from s_points to points to distance_matrix using euclidean_distance method
        for i in range(len(points)):
           distance_matrix[i]=self.euclidean_distance(s_point,points[i])
           #distance_matrix[i] = self.manhattan_distance(s_point, points[i])

        # Sort and store k indices in ascending order in distance_matrix using the argsort function of a numpy array.
        k_index = np.argsort(distance_matrix)[:self.k]

        # initialize k_target_name numpy array size of k
        k_target = np.zeros(self.k)

        # return nearest k distance to k_target
        for i in range(self.k):
            k_target[i] = data[k_index[i]]

        # initialize weight for each number to choose
        weight_0=weight_1=weight_2=weight_3=weight_4=weight_5=weight_6=weight_7=weight_8=weight_9=0

        # accumulation of weights for reciprocal of (distance+1) to avoid divide by 0 , k_index[i] is distance
        for i in range(self.k):
            if k_target[i] == 0:
                weight_0 += 1 / (k_index[i]+0.1)
            elif k_target[i] == 1:
                weight_1 += 1 / (k_index[i]+0.1)
            elif k_target[i] == 2:
                weight_2 += 1 / (k_index[i]+0.1)
            elif k_target[i] == 3:
                weight_3 += 1 / (k_index[i]+0.1)
            elif k_target[i] == 4:
                weight_4 += 1 / (k_index[i]+0.1)
            elif k_target[i] == 5:
                weight_5 += 1 / (k_index[i]+0.1)
            elif k_target[i] == 6:
                weight_6 += 1 / (k_index[i]+0.1)
            elif k_target[i] == 7:
                weight_7 += 1 / (k_index[i]+0.1)
            elif k_target[i] == 8:
                weight_8 += 1 / (k_index[i]+0.1)
            elif k_target[i] == 9:
                weight_9 += 1 / (k_index[i]+0.1)

        # choose max weight and save
        max_weight = max(weight_0,weight_1,weight_2,weight_3,weight_4,weight_5,weight_6,weight_7,weight_8,weight_9)

        # return class of number of Mnist data which has max_weight
        if (max_weight == weight_0):
            return '0'
        elif (max_weight == weight_1):
            return '1'
        elif (max_weight == weight_2):
            return '2'
        elif (max_weight == weight_3):
            return '3'
        elif (max_weight == weight_4):
            return '4'
        elif (max_weight == weight_5):
            return '5'
        elif (max_weight == weight_6):
            return '6'
        elif (max_weight == weight_7):
            return '7'
        elif (max_weight == weight_8):
            return '8'
        elif (max_weight == weight_9):
            return '9'

    #KNN Class reset
    def reset(self):
         self.x_train = 0
         self.t_train = 0
         self.label = 0



