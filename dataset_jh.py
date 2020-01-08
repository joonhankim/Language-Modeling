# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 04:03:55 2019

@author: eric
"""

import unidecode
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

dataset='shakespeare_train.txt'

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
            You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):

        # write your codes here
        #filename='shakespeare_train.txt'
        self.file=unidecode.unidecode(open(input_file).read())
        self.chars = tuple(sorted(set(self.file)))
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.encoded = np.array([self.char2int[ch] for ch in self.file])
        
        
    def __len__(self):
        
        return int((len(self.file)-30)/5)

        # write your codes here

    
    def __getitem__(self,idx):

        # write your codes here
        input = self.encoded[idx*5:idx*5+30]
        target = self.encoded[idx*5+1:idx*5+30+1]
        input = torch.LongTensor(input)
        target = torch.LongTensor(target)
        
        return input, target

#if __name__ == '__main__':
    

