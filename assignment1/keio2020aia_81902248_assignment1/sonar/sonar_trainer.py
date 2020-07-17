from . import *

class Sonar_Trainer:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.loss = ploss

    def accuracy(self, data):
        #calculating number of correct predictions out of total data by giving 1 if predicted value and label value are the same, summing up all the correct predictions and dividing by the total data size
        #return 100*float(sum([1 for x, y in data if self.model.predict(x) == y]))/float(len(data))
        return 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data]) 
        

    def train(self, lr, ne):

        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))
        
        costs = []
        accuracies = []

        for epoch in range(ne):
            J = 0
            for x, y in self.dataset:

                x = np.array(x)
                yhat = self.model(x)
                #calling ploss by passing the predicted value and actual value and summing up the total loss for entire data 
                J+=self.loss(self.model.predict(x),y)
                #changing the model weights based on the difference between actual and predicted value
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
