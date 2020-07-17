from . import *

class Cat_Trainer:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.model.train_mean = self.dataset.train_mean
        self.model.train_sd = self.dataset.train_sd
        self.loss = lrloss

    def accuracy(self, data):
        #calculating number of correct predictions  out of total data by giving 1 if predicted value and label value are same,summing up all the correcting prediction and dividing by total data
        #next(iter(data)) gives the dataset ,x is input vector and y is label 0 0r 1.if predicted value of model is equal to actual label
        #then append 1 to list. sum all these 1 to get numbe of correct predictions out of all the predictions.to get accuracy divide
        #the total number of correct predictions by total data length ,to get percentage multiply with 100.
        return 100*float(sum([1 for x, y in next(iter(data)) if self.model.predict(x) == y]))/float(len(data))

    def train(self, lr, ne):
        
        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))
        
        costs = []
        accuracies = []
        #looping through number of epochs
        for epoch in range(1, ne+1):
            #intializing the cost variable
            J = 0
            #looping through each row of the dataset
            for x, y in next(iter(self.dataset)):
                #converting list to numpy array
                x = np.array(x)
                #yhat is the predicted value for  the given x vector
                yhat = self.model(x)
                #calling lr loss by passing predicted value and actual value and summing up total loss for entire data 
                #J+=self.loss(self.model.predict(x),y)
                J+=self.loss(yhat,y)
                #changing the model weights to based the difference between actual and predicted value
                self.model.w += lr*(y-yhat)*x
                #changing bias based on the difference between actual and predicted value
                self.model.b += lr*(y-yhat)

            J /= len(self.dataset)

            accuracy = self.accuracy(self.dataset)
            if epoch%10 == 0:
                print('--> epoch=%d, accuracy=%.3f' % (epoch, accuracy))
            costs.append(J)
            accuracies.append(accuracy)
        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))
        
        costs = list(map(lambda t: np.mean(t), [np.array(costs)[i-10:i+11] for i in range(1, len(costs)-10)]))
        accuracies = list(map(lambda t: np.mean(t), [np.array(accuracies)[i-10:i+11] for i in range(1, len(accuracies)-10)]))
        
        return (costs, accuracies)
