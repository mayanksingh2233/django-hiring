import numpy as np

#Load weights
weights = np.load("weights.npy")

"""
Receive 'input_data' here
"""

def my_function(X, W):
    '''
            Parameters:
                    X (float): A tensor of shape (200, 4)
                    W (float): A tensor of shape (200, 1)

            Returns:
                    y (int): A value between 1 to 4 
    '''
    y = np.argmax(X.T.dot(W))
    return y

#Make prediction
prediction = my_function(X=input_data, W=weights)

"""
Send prediction 
"""