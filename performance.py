# -*- coding: utf-8 -*-
"""
COMP 551 A2

Author: Shatil Rahman
ID:     260606042

Contains the functions to evaluate performance of classifiers, using 
the following metrics: accuracy, precision, recall and F1-score
"""

import numpy as np

def evaluate(Y_test, Y_predicted):
    '''
    -Calculates the accuracy, precision, recall and F1-score and the 
     confusion matrix from the results of the classification
    -Class 0 (1.0s) is a "positive" and Class 1 (-1.0's) is a "negative"
    -The format of the confusion matrix is:
            _______________________________
            |true positive | false positive|
            |______________|_______________|
            |false negative| true negative |
            |______________|_______________|
            
    '''
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    for i in range(0,len(Y_test)):
        if Y_predicted[i] == 1.0:
            if Y_test[i] == 1.0:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if Y_test[i] == -1.0:
                tn = tn + 1
            else:
                fn = fn + 1
    
    conf_mat = np.array([[tp, fp], [fn, tn]])
    
    accuracy = (tp + tn)/ (tp + tn + fp + fn)
    precision = tp/(tp+fp)
    recall = tp/(tp + fn)
    f1_score = 2.0*(precision * recall)/(precision + recall)
    
    return accuracy, precision, recall, f1_score
    
    
         
        
    