# -*- coding: utf-8 -*-
"""
COMP 551 A2

Author: Shatil Rahman
ID:  260606042

This module deals with generating the datasets

"""

import numpy as np

def genDS1():
    
    #Import the means and the covariance matrix
    m0 = np.loadtxt('hwk2_datasets_corrected/DS1_m_0.txt',delimiter=',',ndmin=1, dtype=float)
    m1 = np.loadtxt('hwk2_datasets_corrected/DS1_m_1.txt',delimiter=',',ndmin=1, dtype=float)
    cov1 = np.loadtxt('hwk2_datasets_corrected/DS1_Cov.txt',delimiter=',',ndmin=2, dtype=float)
    
    #Generate samples for class 0, label them as positive by adding a column of 1's
    DS1_pos = np.random.multivariate_normal(m0,cov1,size=(2000))
    pos = 1.0*np.ones((2000,1))
    DS1_pos = np.concatenate((DS1_pos,pos),axis=1)
    
    #Generate samples for class 1, label them as negative by adding a column of -1's
    DS1_neg = np.random.multivariate_normal(m1,cov1,size=(2000))
    neg = -1.0*np.ones((2000,1))
    DS1_neg = np.concatenate((DS1_neg,neg),axis=1)
    
    #Split the dataset into test and training, using a 30-70 split
    DS1_test = np.concatenate((DS1_pos[:600,:], DS1_neg[:600,:])) 
    DS1_training = np.concatenate((DS1_pos[600:,:], DS1_neg[600:,:]))
    
    #Save the datasets as csv
    np.savetxt('DS1_test.csv', DS1_test, delimiter=',')
    np.savetxt('DS1_training.csv', DS1_training, delimiter=',')


def genDS2():    
    #Import the means and the covariance matrices
    c1_m1 = np.loadtxt('hwk2_datasets_corrected/DS2_c1_m1.txt',delimiter=',',ndmin=1, dtype=float)
    c1_m2 = np.loadtxt('hwk2_datasets_corrected/DS2_c1_m2.txt',delimiter=',',ndmin=1, dtype=float)
    c1_m3 = np.loadtxt('hwk2_datasets_corrected/DS2_c1_m3.txt',delimiter=',',ndmin=1, dtype=float)
    c2_m1 = np.loadtxt('hwk2_datasets_corrected/DS2_c2_m1.txt',delimiter=',',ndmin=1, dtype=float)
    c2_m2 = np.loadtxt('hwk2_datasets_corrected/DS2_c2_m2.txt',delimiter=',',ndmin=1, dtype=float)
    c2_m3 = np.loadtxt('hwk2_datasets_corrected/DS2_c2_m3.txt',delimiter=',',ndmin=1, dtype=float)
    cov1 = np.loadtxt('hwk2_datasets_corrected/DS2_Cov1.txt',delimiter=',',ndmin=2, dtype=float)
    cov2 = np.loadtxt('hwk2_datasets_corrected/DS2_Cov2.txt',delimiter=',',ndmin=2, dtype=float)
    cov3 = np.loadtxt('hwk2_datasets_corrected/DS2_Cov3.txt',delimiter=',',ndmin=2, dtype=float)
        
        
    #Generate samples for class 0, label them as positive by adding a column of 1's
    DS2_pos = np.empty((1,20))
    for i in range(0,2000):
        g1_sample = np.random.multivariate_normal(c1_m1, cov1).reshape(20,1).T
        g2_sample = np.random.multivariate_normal(c1_m2, cov2).reshape(20,1).T
        g3_sample = np.random.multivariate_normal(c1_m3, cov3).reshape(20,1).T
        
        choice = np.random.choice(3,p=[0.1,0.42,0.48])
        
        
        if choice == 0:
            DS2_pos = np.concatenate((DS2_pos,g1_sample), axis=0)
        if choice == 1:
            DS2_pos = np.concatenate((DS2_pos,g2_sample), axis=0)
        if choice == 2:
            DS2_pos = np.concatenate((DS2_pos,g3_sample), axis=0)
    
    pos = 1.0*np.ones((2000,1))
    DS2_pos = DS2_pos[1:,:]
    DS2_pos = np.concatenate((DS2_pos,pos),axis=1)
    
    #Generate samples for class 1, label them as negative by adding a column of -1's
    DS2_neg = np.empty((1,20))
    for i in range(0,2000):
        g1_sample = np.random.multivariate_normal(c2_m1, cov1).reshape(20,1).T
        g2_sample = np.random.multivariate_normal(c2_m2, cov2).reshape(20,1).T
        g3_sample = np.random.multivariate_normal(c2_m3, cov3).reshape(20,1).T
        
        choice = np.random.choice(3,p=[0.1,0.42,0.48])
        
        
        if choice == 0:
            DS2_neg = np.concatenate((DS2_neg,g1_sample), axis=0)
        if choice == 1:
            DS2_neg = np.concatenate((DS2_neg,g2_sample), axis=0)
        if choice == 2:
            DS2_neg = np.concatenate((DS2_neg,g3_sample), axis=0)
    
    neg = -1.0*np.ones((2000,1))
    DS2_neg = DS2_neg[1:,:]
    DS2_neg = np.concatenate((DS2_neg,neg),axis=1)
    
    
    
    
    
    #Split the dataset into test and training, using a 30-70 split
    DS2_test = np.concatenate((DS2_pos[:600,:], DS2_neg[:600,:])) 
    DS2_training = np.concatenate((DS2_pos[600:,:], DS2_neg[600:,:]))
    
    #Save the datasets as csv
    np.savetxt('DS2_test.csv', DS2_test, delimiter=',')
    np.savetxt('DS2_training.csv', DS2_training, delimiter=',')


    


