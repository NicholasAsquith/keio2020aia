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
  #loading the data into dcx from pickle file by calling Sonar_Data with datapath as parameter
  Data=Sonar_Data()
  #calling Sonar_model class with input dimension and actiavtion function,the variable dimension is assigned with length of input vector
  model =Sonar_Model(dimension=len([next(iter(Data))][0][0][0]), activation=perceptron)
  #calling Sonar_Trainer class with data and model ,the first parameter is the data which is converted from object into numpy array
  trainer = Sonar_Trainer(next(iter(Data)), model)
  #As train from trainer returns cost and accuracies  and storing it into variables.and parameters to train as learing rate and number of epochs
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
    
