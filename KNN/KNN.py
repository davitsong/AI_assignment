import numpy as np

# Make KNN class
class KNN:

    # initialize class
    def __init__(self,k,X_train,y_train,y_name):

        self.k=k
        self.X_train=X_train
        self.y_train=y_train
        self.y_name=y_name

    # Euclidien distance metric by numpy
    def euclidean_distance(self,X1,X2):

        return np.sqrt(np.sum(np.power(X1-X2,2)))

    # To return the best count index
    def best_count(self,labels):

        cnt0=cnt1=cnt2=0            # initialize

        for i in labels:
            if i==0:                # index for count setosa
                cnt0=1+cnt0

            elif i==1:              # index for count versicolor
                cnt1=1+cnt1
            else:                   # index for count virginica
                cnt2=1+cnt2
        return np.argmax([cnt0, cnt1, cnt2])

    # majority_vote
    def majority_vote(self, s_point, points, data):

        # initialize distance_matrix numpy array size of points numpy array
        distance_matrix = np.zeros(len(points))

        # save distance from s_points to points to distance_matrix using euclidean_distance method
        for i in range(len(points)):
            distance_matrix[i]=self.euclidean_distance(s_point,points[i])

        # Sort and store k indices in ascending order in distance_matrix using the argsort function of a numpy array
        k_index=np.argsort(distance_matrix)[:self.k]

        # initialize k_target_name numpy array size of k
        k_target=np.zeros(self.k)

        # return k nearest target value
        for i in range(self.k):
            k_target[i] = data.target[k_index[i]]  #ex) k=4 k_target=[0,1,2,1]

        # count k_target the best index and save idx_result
        idx_result = self.best_count(k_target)

        # return if index 0 'setosa' 1 'versicolor' 2 'virginica
        if (idx_result == 0):
            return 'setosa'
        elif (idx_result == 1):
            return 'versicolor'
        elif (idx_result == 2):
            return 'virginica'

    # Weighted Majority Vote
    def weighted_majority_vote(self,s_point,points,data):

        # initialize distance_matrix numpy array size of points numpy array
        distance_matrix=np.zeros(len(points))

        # save distance from s_points to points to distance_matrix using euclidean_distance method
        for i in range(len(points)):
            distance_matrix[i]=self.euclidean_distance(s_point,points[i])

        # Sort and store k indices in ascending order in distance_matrix using the argsort function of a numpy array.
        k_index = np.argsort(distance_matrix)[:self.k]

        # initialize k_target_name numpy array size of k
        k_target = np.zeros(self.k)

        # print(sorted_index)
        for i in range(self.k):
            k_target[i] = data.target[k_index[i]]

        # initialize weight for each iris weight_0: setos,a weight_1:versiocolor, weight_2:virginica
        weight_0=weight_1=weight_2=0.0

        # accumulation of weights which k_target points
        # accumulation of weights for reciprocal of (distance+1) to avoid divide by 0 , k_index[i] is distance
        for i in range(self.k):
            if k_target[i] == 0:
                weight_0 += 1 / (k_index[i]+1)
            elif k_target[i] == 1:
                weight_1 += 1 / (k_index[i]+1)
            elif k_target[i] == 2:
                weight_2 += 1 / (k_index[i]+1)

        # choose max weight and save
        max_weight = max(weight_0,weight_1,weight_2)

        # return class of iris which has max_weight
        if (max_weight == weight_0):
            return 'setosa'
        elif (max_weight == weight_1):
            return 'versicolor'
        elif (max_weight == weight_2):
            return 'virginica'
