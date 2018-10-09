import numpy
import copy
from collections import OrderedDict
from settings import config
import torch
from torch.autograd import Variable

class Data:
    def __init__(self):
        # Load captions as array of strings corresponding to an array of image feature vectors
        self.load_dataset(name=config["dataset"])        

        # Create vocab dictionaries
        self.create_dictionaries()

        # Reset counter & batch size
        self.batch_size = config["batch_size"]        
        self.batch_number = 0   

    def __iter__(self):
        return self

    def __next__(self):        
        """
        Return a batch of data ready to go into the model
        """

        # Upper and lower indexes for the batches
        upper_idx = (self.batch_number+1)*self.batch_size
        lower_idx = self.batch_number*self.batch_size

        # If our lower index is in the bounds of our data, we can return a new batch
        if lower_idx < len(self.train[0]):            
            self.batch_number += 1 # Increment the batch number so next-time we will return the next batch
            captions = self.train[0][lower_idx:upper_idx] # Extract caption batch
            image_features = self.train[1][lower_idx:upper_idx] # Extract image feature batch
            captions, image_features = self.preprocess_data(captions, image_features) # Preprocess our data, converting raw text to embedded numbers etc                        
            return captions, image_features
            
        self.batch_number = 0
        raise StopIteration

    def load_dataset(self, name='f8k', path_to_data = 'data/'):
        """
        Load captions and image features
        """
        print("loading dataset")

        loc = path_to_data + name + '/'

        # Captions
        train_caps, dev_caps = [], []
        with open(loc+name+'_train_caps.txt', 'rb') as f:
            for line in f:
                train_caps.append(line.strip())
        with open(loc+name+'_dev_caps.txt', 'rb') as f:
            for line in f:
                dev_caps.append(line.strip())

        # Image features
        train_ims = numpy.load(loc+name+'_train_ims.npy')
        dev_ims = numpy.load(loc+name+'_dev_ims.npy')        

        self.train = (train_caps, train_ims)
        self.dev = (dev_caps, dev_ims)
        return

    def create_dictionaries(self):
        """
        Create the dictionaries to go from a word to an index and an index to a words.
        """        
        captions = self.train[0] + self.dev[0]

        # TODO: Add beginning and end of setence tokens thats not zero
        self.word_to_index = {'<eos>':0, 'UNK':1}
        self.index_to_word = {0:'<eos>', 1:'UNK'}

        words = set()                
        for idx, caption in enumerate(captions):  
            for word in caption.split():  
                words.add(word)                

        for idx, word in enumerate(words):
            self.word_to_index[word] = idx + 2
            self.index_to_word[idx+2] = word              

        print("dictionary contains", len(self.word_to_index)-2, "words.")        
        return

    def preprocess_data(self, captions, image_features, n_words=10000):
        """
        Convert raw data to go into the model.
        """               
        # Convert a caption to an array of indexes of words from the dictionary, self.word_to_index  
        sequences = []        
        for idx, caption in enumerate(captions):
            sequences.append([self.word_to_index[word] if word in self.word_to_index.keys() else 1 for word in caption.split()])                    

        sequence_lengths = [len(seq) for seq in sequences] # the lengths of all sequences in an array
        processed_captions = numpy.zeros((max(sequence_lengths)+1, len(sequences))).astype('int64') # create matrix w/ biggest length of sequence by length of all sequences
        for idx, seq in enumerate(sequences):
            processed_captions[:sequence_lengths[idx], idx] = seq # populate matrix with sequences

        # Just convert image features to numpy array
        processed_image_features = numpy.asarray(image_features, dtype=numpy.float32)

        if torch.cuda.is_available() and config["cuda"] == True:
            return Variable(torch.from_numpy(processed_captions)).cuda(), Variable(torch.from_numpy(processed_image_features)).cuda()
        
        return Variable(torch.from_numpy(processed_captions)), Variable(torch.from_numpy(processed_image_features))