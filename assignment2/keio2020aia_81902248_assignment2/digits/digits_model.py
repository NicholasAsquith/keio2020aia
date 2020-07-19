from . import *

class Digits_Model:

    def __init__(self, dim_input=784, dim_hidden=[100], dim_out=10):

        self.layers = []
        #adding layers to the layers array
        self.layers.append(Linear(dim_input,dim_hidden[0]))#The linear input layer taking arguments (input dimension, first hidden layer dimension)
        self.layers.append(Tanh())#The activation function
        for i in range(0,len(dim_hidden)-1):# We loop over all elements in the hidden layers based on dim_hidden array. e.g., if dim_hidden=[100,80] then another hidden layer+activation function will be added
            self.layers.append(Linear(dim_hidden[i],dim_hidden[i+1]))
            self.layers.append(Tanh())
        self.layers.append(Linear(dim_hidden[len(dim_hidden)-1],dim_out))#The final layer with dim_out as the output
        
        self._predict = softmax 
        
    def __str__(self):

        return "Simple Neural Network\n\
        \tInput dimension: %d\n\
        \tHidden dimensions: %d\n\
        \tOutput dimension: %d\n" % (self._dim_input, self._dim_hidden, self._dim_out)

    def __call__(self, x):
        
        prediction = None
        outputs=self.forward(x)# The batch input is passed through all the layers via the forward function and gives the model's probability outputs
        prediction=self._predict(outputs)# These probability outputs are passed through the softmax layer via the ._predict method to get the predicted outputs
        pred=[]
        for i in prediction: #picking the highest probability digit by looping each element in the predictions, and calling np.argmax over the probability distribution for all the batches. E.g., [0.1,0.2,0.1,0.4] output would be 4
            pred.append(np.argmax(i))
        prediction=np.array(pred)
        # hint: use the forward and predict functions to obtain a probability distribution over digits, then pick the highest probability digit
        return prediction

    def load_model(self, file_path):
        
        with open(file_path, mode='rb') as f:
            loaded_model = pkl.load(f)

        self.__dict__.update(loaded_model.__dict__)

    def save_model(self):
## The below is optional and drops saved inputs in each layer before saving the model to conserve memory
        for layer in self.layers: 
            layer.inputs = None
##
        with open('results/digits_model.pkl','wb') as f:
            pkl.dump(self, f)
    
    def forward(self, inputs):
        
        outputs = None
        outputs=inputs
        #passing the output of the previous layer as input to the next layer and saving them as outputs
        for layer in self.layers:
          outputs=layer.forward(outputs)        
        # hint: transform a batch of inputs (images) into a batch of outputs (final layer activations) using your model's layers in the order that they are supposed to be applied  
        return outputs

    def backward(self, grad):
        
        for layer in reversed(self.layers):
            
            grad = layer.backward(grad)
            
        return grad

    def params_and_grads(self):
        
        for layer in self.layers:
            
            for name, param in layer.params.items():
                
                grad = layer.grads[name]
                yield param, grad