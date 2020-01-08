# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 04:03:59 2019

@author: eric
"""

from dataset_jh import Shakespeare
from model_jh import CharRNN, CharLSTM
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch
import numpy as np
from matplotlib import pyplot as plt

"""
for i, data in enumerate(data_loader):
    input,target = data
target.shape
"""
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    
    
def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """
    running_loss = 0
    count = 0
    model.train()
    for i, (input_, target_) in enumerate(trn_loader):
        input_, target_ = input_.to(device), target_.to(device)
        batch_size= input_.shape[0]
#        optimizer.zero_grad()
        hidden = model.init_hidden(batch_size)
        model.zero_grad()

#        hidden= hidden.to(device)
        output, hidden= model(input_, hidden)
        loss = criterion(output, target_.view(-1).long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        count += len(input_)
        
    
    
    trn_loss = running_loss / len(trn_loader)
    
    

    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    # write your codes here
    val_loss = 0
    correct= 0
    with torch.no_grad():
        for input , target in val_loader:
            input , target = input.to(device), target.to(device)
            batch_size= input.shape[0]
            hidden = model.init_hidden(batch_size)
            #hidden= hidden.to(device)
            output, hidden= model(input, hidden)
            loss = criterion(output, target.view(-1).long())
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            val_loss+= loss
            
        val_loss /= len(val_loader)
        

        
            
    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    data_dir='shakespeare_train.txt'
    DATA=  Shakespeare(data_dir) 
    
    validation_split = .3
    shuffle_dataset = True
    random_seed= 42
    batch_size = 168
    dataset_size = len(DATA)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    trn_loader = DataLoader(DATA, batch_size=batch_size,sampler=train_sampler)
    val_loader = DataLoader(DATA, batch_size=batch_size,sampler=valid_sampler)
    
    
    rnn = CharRNN(chars=DATA.chars,hidden_dim=512,n_layers=3,drop_prob=0.5,dict_size=len(DATA.char2int),emved_dim=20)
    rnn= rnn.to(device)
    lstm = CharLSTM(chars=DATA.chars,hidden_dim=512,n_layers=3,drop_prob=0.5,dict_size=len(DATA.char2int),emved_dim=20)
    lstm= lstm.to(device)

    
    lr=0.001 

    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr=lr)
    optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=lr)
    
    save_dir=r'./model.pt'
    trn_rnn_loss = []
    val_rnn_loss = []
    best_val_loss = None
    
    for epoch in range(10):
        training_loss = train(rnn, trn_loader,device, criterion, optimizer_rnn)
        print("{0} epoch = train_rnn_loss : {1}".format(epoch, training_loss))
        trn_rnn_loss.append(training_loss)
        testing_loss = validate(rnn, val_loader,device, criterion)
        print("{0} epoch = validate_rnn_loss : {1}".format(epoch, testing_loss))
        val_rnn_loss.append(testing_loss)
        if not best_val_loss or testing_loss < best_val_loss:
            torch.save(rnn.state_dict(),save_dir)
            best_val_loss = testing_loss
    
    plt.figure()
    
    plt.plot(range(10),trn_rnn_loss,color='blue')
    plt.plot(range(10),val_rnn_loss,color='red')
    plt.legend(['Train Loss RNN','Validate Loss RNN'],loc='upper right')
    
    trn_lstm_loss = []
    val_lstm_loss = []
    
    
    for epoch in range(10):
        training_loss = train(lstm, trn_loader,device, criterion, optimizer_lstm)
        print("{0} epoch = train_lstm_loss : {1}".format(epoch, training_loss))
        trn_lstm_loss.append(training_loss)
        testing_loss = validate(lstm, val_loader,device, criterion)
        print("{0} epoch = validate_lstm_loss : {1}".format(epoch, testing_loss))
        val_lstm_loss.append(testing_loss)


    plt.figure()
    
    plt.plot(range(10),trn_lstm_loss,color='green')
    plt.plot(range(10),val_lstm_loss,color='red')
    plt.legend(['Train Loss LSTM','Validate Loss LSTM'],loc='upper right')
    



if __name__ == '__main__':
    main()
