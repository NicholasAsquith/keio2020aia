from . import *

class Sonar_Data:

    def __init__(self, data_file_path='../data/', data_file_name='sonar_data.pkl'):
                
        self.data = []
         #appending file path with filename
        filepath=data_file_path+data_file_name
        #loading the pickle file 
        with open(filepath, 'rb') as f:
            dat = pkl.load(f)
        #based on keys rock and mine in pickle file creating a list of data format as [[vector],[label]] and labelling the r as 0 and m as 1
        k=["r","m"] #create an empty list K: [r,m]

        for i in k: #for each element in K
            for j in dat[i]: #Loop over each index of the dat list, i.e., rock, no rock
                if i=="r": #For every rock value
                    self.data.append([j,0]) #save this to the training data with the key 0
                else: #Same logic for every mine value 
                    self.data.append([j,1])

        self.shuffle()
        

    def __iter__(self):
        return self

    def __next__(self):
        
        return self.data

    def shuffle(self):
        
        random.shuffle(self.data)        
    
    def __len__(self):
        
        return len(self.data)


