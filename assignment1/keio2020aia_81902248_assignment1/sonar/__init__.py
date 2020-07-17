##
import numpy as np
import pickle as pkl
import random
##

##
def perceptron(z):
    return -1 if z <= 0 else 1

def ploss(yhat, y):
    return max(0, -yhat*y)

def ppredict(self, x):
     return 1.0 if self(x) > 0.5 else 0.0
##
