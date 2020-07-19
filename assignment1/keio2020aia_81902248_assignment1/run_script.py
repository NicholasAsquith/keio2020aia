##
from sonar.sonar_data import * 
from sonar.sonar_model import * 
from sonar.sonar_trainer import * 
from cats.cat_data import *
from cats.cat_model import *
from cats.cat_trainer import *
##

##
def main():
    #sonar model
  #loading the data into dcx from pickle file by calling Sonar_Data with datapath as the default parameter
  Data=Sonar_Data()
  #calling Sonar_model class with input dimension and activation function,the variable dimension is assigned as the length of the input vector
  model =Sonar_Model(dimension=len([next(iter(Data))][0][0][0]), activation=perceptron)
  #calling Sonar_Trainer class with data and model ,the first parameter is the data which is converted into a numpy array
  trainer = Sonar_Trainer(next(iter(Data)), model)
  #Storing the cost and accuracies from trainer.train, hyperparameters for learning rate and epoch are set here
  costs, accuracies =trainer.train(0.1, 500)#(learning rate and epochs)
  #saving the model into pickle file
  model.save_model()
 
    #cat model
  Data=Cat_Data()
  model =Cat_Model(dimension=len([next(iter(Data))][0][0][0]), activation=sigmoid)
  trainer = Cat_Trainer(Data, model)
  costs, accuracies =trainer.train(0.1, 80)#(learning rate and epochs)
  model.save_model()
##

##
if __name__ == "__main__":
    main()
    
