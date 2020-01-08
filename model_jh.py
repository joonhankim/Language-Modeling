# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 04:03:58 2019

@author: eric
"""

import torch.nn as nn
import torch

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
#is_cuda = torch.cuda.is_available()
#
## If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
#if is_cuda:
#    device = torch.device("cuda")
#    print("GPU is available")
#else:
#    device = torch.device("cpu")
#    print("GPU not available, CPU used")
    
    
class CharRNN(nn.Module):
    def __init__(self,chars,hidden_dim,n_layers,drop_prob,dict_size,emved_dim):
        super(CharRNN,self).__init__()
        
        self.hidden_dim=hidden_dim
        self.n_layers = n_layers
        self.chars = chars
        self.drop_prob= drop_prob
        
        self.embed  = nn.Embedding(dict_size,emved_dim)
        self.rnn = nn.RNN(emved_dim,self.hidden_dim,self.n_layers,batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc=nn.Linear(self.hidden_dim,len(self.chars))
        # write your codes here

        
        
    def forward(self, input,hidden):

        # write your codes here
        input=self.embed(input)
        output, hidden = self.rnn(input,hidden)
        output = self.dropout(output)
        output = output.contiguous().view(-1,self.hidden_dim)
        output= self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        weight = next(self.parameters()).data
        initial_hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        
        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self,chars,hidden_dim,n_layers,drop_prob,dict_size,emved_dim):
        super(CharLSTM,self).__init__()
        
        self.drop_prob= drop_prob
        self.hidden_dim = hidden_dim
        self.chars = chars
        self.n_layers = n_layers
        
        
        self.embed  = nn.Embedding(dict_size,emved_dim)
        self.lstm = nn.LSTM(emved_dim,self.hidden_dim,self.n_layers,batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc=nn.Linear(self.hidden_dim,len(self.chars))

        # write your codes here

    def forward(self, input, hidden):

        # write your codes here
        input=self.embed(input)
        output, hidden = self.lstm(input,hidden)
        output = self.dropout(output)
        output = output.contiguous().view(-1,self.hidden_dim)
        output= self.fc(output)
        
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
       
        initial_hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
              weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        

        return initial_hidden