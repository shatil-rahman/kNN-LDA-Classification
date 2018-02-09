# -*- coding: utf-8 -*-
"""
COMP 551 A2

Author: Shatil Rahman
ID:  260606042

This module deals with generating the datasets

"""

import numpy as np

def genQ1():
    
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
    



