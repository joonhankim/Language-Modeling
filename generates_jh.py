# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 04:04:00 2019

@author: eric
"""

# import some packages you need here
from dataset_jh import Shakespeare
from model_jh import CharRNN, CharLSTM
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable



def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = DATA.char2int[string[c]]
    return Variable(tensor)


def generate(model, seed_characters, temperature, DATA):
    

    model.eval()
    batch_size=1

    inp = char_tensor(seed_characters)
    hidden = model.init_hidden(batch_size)
    x=inp
    #prob= prob.numpy().squeeze()
    #inp = torch.unsqueeze(inp, 0)
    samples= seed_characters
    
    for i in range(200):
        output, hidden = model(x.view(1, 1),hidden)

        output_dist = output.data.view(-1)
        output_dist = np.array(torch.nn.functional.softmax(output_dist/temperature, dim=0).data)
        pred_char= np.random.choice(np.arange(len(DATA.chars)), p= output_dist/output_dist.sum())
        
        pred_char = DATA.int2char[pred_char]
        
        samples= samples+ pred_char
        
        x = char_tensor(pred_char)
        
    return samples
            
    


data_dir='shakespeare_train.txt'
DATA = Shakespeare(data_dir)
model = CharRNN(chars=DATA.chars,hidden_dim=512,n_layers=3,drop_prob=0.5,dict_size=len(DATA.char2int),emved_dim=20)
model.load_state_dict(torch.load('./model.pt'))
model.cpu()


# 결
result_list=[] 
for temperature in [0.5, 1, 1.5,2]:
     result=generate(model,'A',temperature,DATA)      
     result_list.append(result)