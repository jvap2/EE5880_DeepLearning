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
        self.relu_1=nn.LeakyReLU(.1)
        self.output_layer=nn.ReLU()
        
    def forward(self,x):
        '''This function will take an input pattern and do the forward pass'''
        x=self.layer1(x) 
        x=self.relu_1(x)
        x=self.layer2(x)
        x=self.output_layer(x)
        return x
    
def Loss(output,x,Load,A,T_max):
    '''x[:,0]=Pg and x[:,1]=Pd and x[:,2]=Pl'''
    T=torch.matmul(A,x[:,0]+output-x[:,1])
    loss=torch.sum((-output)**2)+torch.sum((x[:,0]+output-x[:,1])**2)+\
    nn.ReLU()(torch.sum(x[:,0])-3405)+nn.ReLU()(Load-torch.sum(x[:,1]))+nn.ReLU()(torch.sum(T-T_max))+\
    nn.ReLU()(torch.sum(output-x[:,1]))
    return loss

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1).requires_grad_()


def Train(model,input,Load,A,T_max,optimizer):
    num_epochs=50
    s=np.shape(A)[1]
    pred=torch.zeros(size=(s,))
    weights_init(model=model)
    for i in range(num_epochs):
        total_loss=0
        for (j,val) in enumerate(input):
            optimizer.zero_grad()
            pred=pred.clone()
            pred[j]=model(val)
            constraint=(torch.autograd.grad(nn.ReLU()(torch.sum(input[:,0])-3405)+nn.ReLU()(Load-torch.sum(input[:,1]))+nn.ReLU()(torch.sum(torch.matmul(A,input[:,0]+pred-input[:,1])-T_max))+\
            nn.ReLU()(torch.sum(pred-input[:,1])),inputs=(input),retain_graph=True))
            for column in range(constraint[0].shape[1]):
                pred[j]=pred[j]-constraint[0][j,column]
            loss=Loss(pred,input,Load,A,T_max)
            loss.backward(inputs= pred,retain_graph=True)
            total_loss=total_loss+loss.item()
            optimizer.step()
    print("Cost: ", total_loss/s)
    return pred

def Network(input_size, hidden_size, output_size,x,Load,A,T_max):
    lr=.01
    torch.autograd.set_detect_anomaly(True)
    s=np.shape(A)[1]
    mod=Model(input_size,hidden_size,output_size)
    optimizer=torch.optim.Adam(mod.parameters(), lr=lr)
    pred=torch.empty(size=(s,))
    pred=Train(mod,x,Load,A,T_max,optimizer)
    return pred

