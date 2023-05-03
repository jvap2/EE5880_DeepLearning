import torchvision
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision.transforms import ToTensor
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# function to generate uniform random data
def generate_random(size):
    random_data = torch.rand(size)
    return random_data

#Function to generate normalized random data
def generate_random_normal(size):
    #Using randn function instead of rand
    random_data = torch.randn(size)
    return random_data



class Generator(nn.Module):
    def __init__(self):
        #initialize parent PyTorch class
        super().__init__()    
        #Define the neural network layers
        self.model = nn.Sequential(
            #First layer with a single input and 200 outputs
            nn.ConvTranspose1d(16,32,3,2,1,1),
            #Apply LeakyRELU activation functions
            nn.LeakyReLU(0.02),
            nn.LayerNorm(32),
            #Second layer with 200 inputs and 784 outputs
            nn.ConvTranspose1d(32,48,3,1,4,3),
            nn.Sigmoid()
        )
        #Generator doesn't have its own loss function
        #Create Adam Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.0001)
        
        #Variables to accumulate progress of training
        self.counter = 0
        self.progress = []
        
    #Overridden Forward Function
    def forward(self,inputs):
        #Run the model
        return self.model(inputs)
    
    #Training Function
    def train(self, Disc, inputs,targets):
        #Compute the outputs of the network
        g_outputs = self.forward(inputs)
        
        #Pass the generator output as input to the discriminator
        d_output = Disc.forward(g_outputs)
        
        #Calculate Loss
        loss = Disc.loss_function(d_output,targets)
        #Accumlate error every ten counts
        self.counter += 1
        if(self.counter % 10 == 0):
            self.progress.append(loss.item())
            
        #Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    #Plotting Function of the progress
    def plot_progress(self):
        df = pd.DataFrame(self.progress,columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5,1.0,5.0))


class Discriminator(nn.Module):
    def __init__(self):
        #initialize parent PyTorch class
        super().__init__()    
        #Define the neural network layers
        self.model = nn.Sequential(
            #First layer with 784 inputs and 200 outputs
            nn.Linear(48,200),
            #Apply LeakyRELU activation functions
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            #Second layer with 200 inputs and a single output
            nn.Linear(200,1),
            nn.Sigmoid()
        )
        #Create the loss function using the Binary Cross Entropy Loss
        self.loss_function = nn.BCELoss()
        
        #Create Adam Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.0001)
        
        #Variables to accumulate progress of training
        self.counter = 0
        self.progress = []
        
    #Overridden Forward Function
    def forward(self,inputs):
        #Run the model
        return self.model(inputs)
    
    #Training Function
    def train(self,inputs,targets):
        #Compute the outputs of the network
        outputs = self.forward(inputs)
        #Calculate Loss
        loss = self.loss_function(outputs,targets)
        #Accumlate error every ten counts
        self.counter += 1
        if(self.counter % 10 == 0):
            self.progress.append(loss.item())
        if(self.counter % 10000 == 0):
            print('counter = ',self.counter)
            
        #Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    #Plotting Function of the progress
    def plot_progress(self):
        df = pd.DataFrame(self.progress,columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5,1.0,5.0))