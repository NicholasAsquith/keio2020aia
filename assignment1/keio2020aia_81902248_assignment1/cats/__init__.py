##
import numpy as np
import pickle as pkl
import random
##

##
def sigmoid(z):
    
    return 1.0 / (1.0 + np.exp(-z))

def lrloss(yhat, y):
    
    return 0.0 if yhat == y else -1.0*(y*np.log(yhat) + (1 - y)*np.log(1 - yhat))

def lrpredict(self, x):
    
    return 1.0 if self(x) > 0.5 else 0.0
##
