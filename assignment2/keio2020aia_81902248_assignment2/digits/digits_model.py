from . import *

class Digits_Model:

    def __init__(self, dim_input=784, dim_hidden=[100], dim_out=10):

        self.layers = []
        #adding layers to the layers array
        self.layers.append(Linear(dim_input,dim_hidden[0]))#linear layer input layer(input dimension, first hidden layer dimension)
        self.layers.append(Tanh())#activation function
        for i in range(0,len(dim_hidden)-1):#add's hidden layers based on dim_hidden array.if dim_hidden=[100,80] then another hidden layer will be added and activation function will be added
            self.layers.append(Linear(dim_hidden[i],dim_hidden[i+1]))
            self.layers.append(Tanh())
        self.layers.append(Linear(dim_hidden[len(dim_hidden)-1],dim_out))#final layer with dim_out as output
        
        self._predict = softmax 
        
    def __str__(self):

        return "Simple Neural Network\n\
        \tInput dimension: %d\n\
        \tHidden dimensions: %d\n\
        \tOutput dimension: %d\n" % (self._dim_input, self._dim_hidden, self._dim_out)

    def __call__(self, x):
        
        prediction = None
        outputs=self.forward(x)#batch input is passed through all the layers with forward function and gives model probability outputs
        prediction=self._predict(outputs)#probability outputs are passed through softmax layer with ._predict to get predicted outputs
        #picking the highest probability digit by calling np.argmax over the probability distribution for all the batches. For example, [0.1,0.2,0.1,0.4] output would be 4
        pred=[]
        for i in prediction:
            pred.append(np.argmax(i))
        prediction=np.array(pred)
        # hint: use the forward and predict functions to obtain a probability distribution over digits, then pick the highest probability digit
        return prediction

    def load_model(self, file_path):
        
        with open(file_path, mode='rb') as f:
            loaded_model = pkl.load(f)

        self.__dict__.update(loaded_model.__dict__)

    def save_model(self):

        for layer in self.layers: 
            layer.inputs = None #Drops saved inputs in each layer before saving model to conserve memory
        with open('results/digits_model.pkl','wb') as f:
            pkl.dump(self, f)
    
    def forward(self, inputs):
        
        outputs = None
        outputs=inputs
        #passing output of previous as input to next layer to get model outputs.
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