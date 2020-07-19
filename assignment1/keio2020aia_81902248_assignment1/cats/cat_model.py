from . import *

class Cat_Model:
    #initializations
    def __init__(self, dimension=1, weights=None, bias=None, activation=(lambda x: x), predict=lrpredict):

        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)*np.sqrt(1/dimension)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)
        self.train = True
        self.train_mean = 0.0
        self.train_sd = 1.0

    def __str__(self):

        return "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)

    def __call__(self, x):
        
        yhat = None
        #getting the predicted value and returning the predicted value of the model
        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)
            
        return yhat
    
    def preprocess(self, x):
        #standardizes the images: averages the colors, subtracting the mean of each image and divides by the standard deviation
        return (x - self.train_mean) / self.train_sd

    def load_model(self, file_path):

        with open(file_path, mode='rb') as f:
            mm = pkl.load(f)
            
        self._dim = mm._dim
        self.w = mm.w
        self.b = mm.b
        self._a = mm._a

    def save_model(self):

        f = open('results/cat_model.pkl','wb')
        pkl.dump(self, f)
        f.close
