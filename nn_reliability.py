import matplotlib.pyplot as plt
import torch 
import numpy as np
import torch.nn as nn
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
    loss=torch.sum((-output)**2)
    # loss=nn.MSELoss()(x[:,0]+output,x[:,1])+torch.sum(output)
    # +nn.ReLU(nn.MSELoss()(torch.sum(x[:,0]),3405))+\
    #     nn.ReLu(nn.MSELoss()(Load-torch.sum(x[:,1])))+nn.MSELoss()(nn.ReLU(T-T_max))+\
    #     nn.MSELoss()(nn.ReLU(output-x[:,1]))
    return loss


def Train(model,input,Load,A,T_max,optimizer):
    num_epochs=500
    s=np.shape(A)[1]
    pred=torch.zeros(size=(s,))
    for i in range(num_epochs):
        total_loss=0
        for (j,val) in enumerate(input):
            optimizer.zero_grad()
            pred=pred.clone()
            pred[j]=model(val)
            loss=Loss(pred,input,Load,A,T_max)
            loss.backward(inputs= pred,retain_graph=True)
            total_loss=total_loss+loss.item()
            optimizer.step()
    return pred

def Network(input_size, hidden_size, output_size,x,Load,A,T_max):
    lr=.001
    torch.autograd.set_detect_anomaly(True)
    s=np.shape(A)[1]
    mod=Model(input_size,hidden_size,output_size)
    optimizer=torch.optim.Adam(mod.parameters(), lr=lr)
    pred=torch.empty(size=(s,))
    pred=Train(mod,x,Load,A,T_max,optimizer)
    return pred

