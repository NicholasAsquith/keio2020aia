from . import *

class Digits_Data:

    def __init__(self, relative_path='../data/', data_file_name='digits_data.pkl', batch_size=64):

        self.batch_size = batch_size
        self.index = -1
        
        with open('%s%s' % (relative_path, data_file_name), mode='rb') as f:
            
            digits_data = pkl.load(f)
            
        self.samples = []
        
        #taking the training data from pickle file,flattening into vectors and mapping labels as one-hot encoded values
        train=digits_data['train']
        for j in range(10):
            for i in train[j]:
                c=np.zeros(10)#array of 10 zeros to create one hot encoding
                c[j]=1# if label is 1 then c would be [0,1,0,0,0,0,0,0,0,0].j is label.
                self.samples.append([i.flatten(),c]) 
                # hint: you will need to flatten the images to represent them as vectors (numpy arrays) and pair them with digit labels from the training data
                
        self.shuffle()
        
        self.starts = np.arange(0, len(self.samples), self.batch_size)

    def __iter__(self):
                
        return self

    def __next__(self):

        self.index += 1
        
        if self.index + 1 > len(self.starts):
            
            self.index = -1
            self.shuffle()
            raise StopIteration
        #self.starts gives us a random set of 938 values from (0 to 60000) i,e  60000 points in our training data are split into 64 batches, or, 938   
        inputs = None
        targets = None
        batch=[]
        tar=[]
        #based on the values from self.starts, the data is split into batches of size 938 points. This batch is called when the _next_ method is called.
        for index in self.starts:
            batch.append(self.samples[index][0])
            tar.append(self.samples[index][1])
    
        inputs=batch
        targets=tar
    # hint: use the starts initialized in the last line of the constructor and the batch size to generate a batch of inputs and the corresponding batch of targets

        return {'inputs': inputs, 'targets': targets}

    def shuffle(self):
        
        random.shuffle(self.samples)

