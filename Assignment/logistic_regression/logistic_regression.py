import numpy as np 
import pandas as pd 
import math
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self):
        
        #learning rate
        self.lr = 1
        #number of iterations for loop 
        self.iter = 60000 
        # w and b are the model parameters
        self.w = None
        self.b = None
        

    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        # init model parameters
        self.w = np.zeros(n_features)
        self.b = 0
        self.compute_gradient(X, y, n_samples)

    def compute_gradient(self, X, y, n_samples):
        
        # gradient descent
        for _ in range(self.iter):
            #linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.w) + self.b
            # sigmoid function
            y_pred = sigmoid(linear_model)
            # compute gradients
            diff = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, diff) 
            db = (1 / n_samples) * np.sum(diff) 
            # update model  parameters
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
        
  
    def predict(self, X):
        
        linear_model = np.dot(X, self.w) + self.b
        y_predicted = sigmoid(linear_model)
        return np.array(y_predicted)

    
    def compute_fi(self,X):
        
        #compute expanded feature space to obtain a linear separation surface in the space defined by the mapping fi
        dim = int(math.pow(X.shape[1],2))
        D = np.zeros((X.shape[0], dim))

        for i in range(X.shape[0]):
            x1 = np.dot(np.array(X.loc[i]).T.reshape(2,1), np.array(X.loc[i]).reshape(1,2))
            D[i, :] = x1.reshape((-1, 1), order="F").reshape(dim,)
        return np.hstack([D, X])
        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

