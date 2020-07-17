##
import numpy as np
import pickle as pkl
import random
##

##
# all inputs should be of shape (batch_size, input_size)
# all outputs will be of shape (batch_size, output_size)

class Linear:
    
    def __init__(self, input_size, output_size):
        
        self.params = {"w": np.random.randn(input_size, output_size), "b": np.random.randn(output_size)}
        self.grads = {}

    def forward(self, inputs):
        
        self.inputs = inputs
        
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
       
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        
        return grad @ self.params["w"].T
    
class Tanh:
    
    def __init__(self):
        
        self.params = {}
        self.grads = {}
    
    def forward(self, inputs):
        
        self.inputs = inputs
        return np.tanh(inputs)

    def backward(self, grad):
        
        return (lambda x: 1 - np.tanh(x) ** 2)(self.inputs) * grad
    
class CrossEntropy:
    
    def loss(self, outputs, targets):
        
        probs = softmax(outputs)
        return -1.0*np.mean(np.diag(np.log(probs + 1e-30) @ targets.T))
    
    def grad(self, predicted, actual):
        
        probs = softmax(predicted)
        
        return (probs - actual)

def softmax(inputs):
    """numerically stabilized softmax along innermost dimension"""
    
    if len(inputs.shape) == 1:
        largest = np.max(inputs)
        exps = np.exp(inputs - largest)
        sum_of_exps = np.sum(exps)

        return exps/sum_of_exps
    
    else:
        
        return np.array([softmax(x) for x in inputs])
##
