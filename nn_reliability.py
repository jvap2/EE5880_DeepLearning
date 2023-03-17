import matplotlib.pyplot as plt
import torch 
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() ##Call constructor from parent class
        self.layer1=nn.Linear(input_size,hidden_size)##Give the input and output size
        self.layer2=nn.Linear(hidden_size,output_size)
        
    def forward(self,x):
        '''This function will take an input pattern and do the forward pass'''
        x=self.layer1(x) 
        x=nn.LeakyReLU(.1)(x)
        x=self.layer2(x)
        return x
    
def Loss(output,x,Load,A,T_max):
    '''x[:,0]=Pg and x[:,1]=Pd and x[:,2]=Pl'''
    T=torch.matmul(A,x[:,0]+output-x[:,1])
    loss=nn.MSELoss(x[:,0]+output-x[:,1])+nn.MSELoss(nn.ReLU(torch.sum(x[:,0])-3405))+\
        nn.MSELoss(nn.ReLU(Load-torch.sum(x[:,1])))+nn.MSELoss(nn.ReLU(T-T_max))+\
        nn.MSELoss(nn.ReLU(output-x[:,1]))
    return loss


def Train(model,input,Load,A,T_max,optimizer):
    num_epochs=500
    pred=torch.empty(size=input.size(dim=1))
    for i in range(num_epochs):
        for (j,val) in enumerate(input):
            pred[j]=model(input)
            loss=Loss(pred,val,Load,A,T_max)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return pred

def Network(input_size, hidden_size, output_size,x,Load,A,T_max):
    lr=.001
    mod=Model(input_size,hidden_size,output_size)
    optimizer=torch.optim.Adam(mod.parameters(), lr=lr)
    pred=torch.empty(size=x.size(dim=1))
    pred=Train(mod,x,Load,A,T_max,optimizer)
    return pred

