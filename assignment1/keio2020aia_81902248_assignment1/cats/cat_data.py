from . import *

class Cat_Data:

    def __init__(self, data_file_path='../data/', data_file_name='cat_data.pkl'):
        
        self.data = []
        self.train_mean = 0.0
        self.train_sd = 0.0
        #appending file path with filename
        filepath=data_file_path+data_file_name
        #loading the pickle file
        with open(filepath, 'rb') as f:
            dat = pkl.load(f)
        #loading the training data into a new list based on the key 'train' from the pickle file    
        dat=dat["train"]
        #standardizing and flattening the 3d image data into a 1d vector and labelling the values 'no_cat' and 'cat' as 0 and 1 respectively
        k=["no_cat","cat"] #creates an empty list K: [cat, no cat]
        for i in k: #loop over each element in list K
            for j in dat[i]: #for each element in list K, loop over each index of the dat list, i.e., no_cat, cat:
                if i=="no_cat": #For every no_cat index
                    j=self.standardize(j) #standardize the values, save them as j
                    s=j.flatten().tolist() #flatten the dimensions to make a 1-d vector & return a copy of the array data as a nested python list, i.e. 28x3
                    self.data.append([s,0]) #save this to the training data with the key 0
                else: #Same logic for every cat value 
                    j=self.standardize(j) 
                    s=j.flatten().tolist()
                    self.data.append([s,1])

        self.shuffle()

    def __iter__(self):
        return self

    def __next__(self):
        #returning the data when Cat_Data function is called
        return self.data

    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, i):
        
        return self.data[i]

    def shuffle(self):
        
        random.shuffle(self.data)
    
    def standardize(self, rgb_images):
        
        mean = np.mean(rgb_images, axis=(0, 1, 2), keepdims=True)
        sd = np.std(rgb_images, axis=(0, 1, 2), keepdims=True)
        return (rgb_images - mean) / sd
