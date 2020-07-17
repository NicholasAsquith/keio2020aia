from . import *

class Digits_Trainer:
    
    def __init__(self, dataset, model):
        
        self.dataset = dataset
        self.model = model
        self.loss = CrossEntropy()
        
    def accuracy(self):

        acc = None
        #appending whole data i,e 60000 training data to dat.for calculating accuracy on entire data.when self.dataset is called a batch of 938 values from whole data is returned
        dat=[]
        for data in self.dataset:
            dat.append([data['inputs'],data['targets']])
               
        acc=0
        #self.model(x) gives predicted values for batch of inputs.y is target vector.
        #Example: pred is [1,5,9] and y is [[0,0,0,0,1,0,0,0],[0,1,0,0,0,0,0],[...]]
        #calculating accuracy for each batch and summing up for all batches
        for x, y in dat:
            pred=self.model(x)
            acc+= 100*float(sum([1 for i in range(len(pred)) if pred[i] == np.argmax(y[i],axis=0)])/float(len(pred)))  
        #getting average accuracy for all the batches combined
        acc=acc/len(dat)
        
        # hint: return the accuracy (i.e. the percentage of digits classified correctly) of the current model on the dataset given in the constructor
        return acc
    
    def step(self, lr):

        for param, grad in self.model.params_and_grads():
            
            param -= lr * grad
            
    def train(self, lr, ne):
        
        print('initial accuracy: %.3f\n\n' %(self.accuracy()))
        
        print('training model on data...\n')
        print('='*80+'\n')
    
        for epoch in range(1, ne + 1):

            epoch_loss = 0.0

            for batch in self.dataset:
                predicted = None
                #model.forward gives predicted values of the model
                predicted=self.model.forward(np.array(batch['inputs']))
                #cross entropy loss is called with predicted values of the model and batch target values and summing up loss for all the batches in epoch
                epoch_loss+=self.loss.loss(predicted,np.array(batch['targets']))
                # hint: use the model to generate predictions (digit labels) for a batch, and then use the loss given in the constructor to update the epoch_loss variable
                
                grad = None
                #gradient is called with predicted values of the model and batch targets
                grad=self.loss.grad(predicted,np.array(batch['targets'])) 
                # hint: compute the gradient of the loss with respect to its inputs (not the model inputs - watch the lecture video for explanation)
               
                self.model.backward(grad)
                self.step(lr)
                
            print("""epoch %d:\n
            \t loss = %.3f\n
            \t accuracy=%.3f""" % (epoch, epoch_loss, self.accuracy()))
            
        print('='*80+'\n')
        print('training complete!\n\n')
        print('final accuracy: %.3f' %(self.accuracy()))









#x=np.argmax([756,279,892],axis=0)