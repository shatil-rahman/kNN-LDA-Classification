# -*- coding: utf-8 -*-
"""
COMP 551 A2

Author: Shatil Rahman
ID:     260606042

Contains the algorithm for the 2-class k-Nearest Neighbours
"""
import numpy as np
import matplotlib.pyplot as plt
import performance as perf

def loadData(fname):
    #for easy loading of data into Input space and output space
    data = np.loadtxt(fname, delimiter=',',dtype=float)
    X = data[:,:-1]
    Y = data[:,-1].reshape((X.shape[0],1))
    
    return [X, Y]
    

def kNN(X, k, X_train, Y_train):
    #kNN classifier, outputs the class of the instance X using the 
    #empirical probability of the k-nearest neighbours
    distances = np.empty((X_train.shape[0],1))
    
    for i in range(0,X_train.shape[0]):
        distance = np.linalg.norm(X_train[i,:] - X)
        distances[i] = distance
        
    sorted_indices = np.argsort(distances, axis=0)
    nearest_neighbours = sorted_indices[:k,0]
    
    n0 = 0.0    
    
    for index in nearest_neighbours:
        if Y_train[index] == 1.0:
            n0 = n0 + 1
    
    p_0 = n0/(float(k))
    if p_0 >0.5:
        return 1.0
    else:
        return -1.0

def bestK(X_train, Y_train, X_test, Y_test):
    #iterates k, and finds optimum k for this dataset
    k=1
    kList = []
    f1_list =[]
    for i in range(1,20):
        Y_predicted = np.empty((Y_test.shape[0],1))

        for i in range(0, Y_test.shape[0]):
            X = X_test[i,:]
            Y_predicted[i] = kNN(X, k, X_train, Y_train)
            
        accuracy, precision, recall, f1_score = perf.evaluate(Y_test, Y_predicted)
        
        kList.append(k)
        f1_list.append(f1_score)
        k = k + 1
        
    
    plt.plot(kList, f1_list, 'b-')
    plt.show
        
        
    

############################## Main Code ###################################   

#Select k for k-NN here:

k = 15

#Load data     
X_train, Y_train = loadData('DS2_training.csv')
X_test, Y_test = loadData('DS2_test.csv')

'''
#Just testing with 2 features

m0 = [1.3, 1.3]
m1 = [3.30, 2.0]

cov = np.array([[0.5, 1.20], [1.20, 5.6]])

 
DS1_pos = np.random.multivariate_normal(m0,cov,size=(2000))
pos = 1.0*np.ones((2000,1))
DS1_pos = np.concatenate((DS1_pos,pos),axis=1)
DS1_neg = np.random.multivariate_normal(m1,cov,size=(2000))
neg = -1.0*np.ones((2000,1))
DS1_neg = np.concatenate((DS1_neg,neg),axis=1)
DS1_training = np.concatenate((DS1_pos[600:,:], DS1_neg[600:,:]))
X_train = DS1_training[:,:-1]
Y_train = DS1_training[:,-1].reshape((X_train.shape[0],1))
DS1_test = np.concatenate((DS1_pos[:600,:], DS1_neg[:600,:]))
X_test = DS1_test[:,:-1]
Y_test = DS1_test[:,-1].reshape((X_test.shape[0],1))
'''
#Prediction

Y_predicted = np.empty((Y_test.shape[0],1))

for i in range(0, Y_test.shape[0]):
    X = X_test[i,:]
    Y_predicted[i] = kNN(X, k, X_train, Y_train)

accuracy, precision, recall, f1_score = perf.evaluate(Y_test, Y_predicted)

print "Performance of kNN with k=" + str(k) + ":"
print "Accuracy: " + str(accuracy)
print "Precision: " + str(precision)
print "Recall: " + str(recall)
print "f1_score: " + str(f1_score)



    
    
