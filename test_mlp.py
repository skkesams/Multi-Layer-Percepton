# Test_mlp.py file

import numpy as np
import pandas as pd

STUDENT_NAME = ['NIKHILA VALIVARTHI','SAI KRISHNAK KESAMSETTY','VENKATA BHAGYASREE POTAMSETTI']
STUDENT_ID = ['20870695','20870700','20924096']

def test_mlp(data_file):
    # Load the test set
    # START
    test_data = pd.read_csv(data_file)
    data = np.float64(test_data)
    # END

    # Load your network
    # START
    weight = np.load('weights.npy', allow_pickle=True)
    # END
    
    # Predict test set - one-hot encoded
    p = np.atleast_2d(data)
    p = np.c_[p, np.ones((p.shape[0]))]
    # loop over our layers in the network
    for layer in np.arange(0, len(weight)):
        p = 1.0 / (1 + np.exp(-(np.dot(p, weight[layer]))))
    predictions = p.argmax(axis=1)
    predict = np.zeros((predictions.size, predictions.max()+1))
    predict[np.arange(predictions.size),predictions] = 1
    return predict

# After reading the test_labels file which is in a .csv format into "test_labels", please convert 
#test_variable into a numpy array by including following line of code: 
#test_labels=np.float64(test_labels)


'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''