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
from cnn_reliability import Data_NN, Clean_Data

global dev
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# function to generate uniform random data
def generate_random(size):
    random_data = torch.rand(size).to(dev)
    return random_data

#Function to generate normalized random data
def generate_random_normal(size):
    #Using randn function instead of rand
    random_data = torch.randn(size).to(dev)
    return random_data



class Generator(nn.Module):
    def __init__(self):
        #initialize parent PyTorch class
        super().__init__()    
        #Define the neural network layers
        self.model = nn.Sequential(
            #First layer with a single input and 200 outputs
            nn.BatchNorm1d(1),
            nn.Upsample(scale_factor=4),
            nn.Conv1d(100,32,3,1,1),
            nn.LeakyReLU(.01),
            nn.BatchNorm1d(32),
            nn.Upsample(scale_factor=3),
            nn.Conv1d(16,2,3,1,1),
            nn.LeakyReLU(.01),
            nn.Tanh(),
        )
        self.model.to(dev)
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
        inputs=inputs.to(dev).float()
        targets=targets.to(dev).float()
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
            nn.Conv1d(2,16,3,1,1),
            nn.LeakyReLU(.1),
            nn.MaxPool1d(2,2),
            #Apply LeakyRELU activation functions
            nn.Conv1d(16,32,3,1,1),
            nn.LeakyReLU(.1),
            nn.MaxPool1d(2,2),

            nn.Flatten(),
            #Second layer with 200 inputs and a single output
            nn.Linear(192,96),
            nn.LeakyReLU(.05),
            nn.Linear(96,1),
            nn.Sigmoid()
        )
        self.model.to(dev)
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
        inputs=inputs.to(dev).float()
        targets=targets.to(dev).float()
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


if __name__=="__main__":
    D = Discriminator()
    G = Generator()
    epochs=4
    Data,Label=Clean_Data()
    print(np.shape(Data)[0])
    dataset=Data_NN(Data,Label)
    gen_1=torch.Generator().manual_seed(42)
    train, val, test =random_split(dataset,[.8,.1,.1],gen_1)
    train_dl=DataLoader(train,128,True)
    val_dl=DataLoader(val,32,False)
    test_dl=DataLoader(test,len(test),False)
    for epoch in range(epochs):
        print("epoch = ", epoch + 1)
        for image_data_tensor, target_tensor in train_dl:
            #train the discriminator on true
            print(image_data_tensor.shape)
            D.train(image_data_tensor,torch.ones(size=(128,1)))
            #train discriminator on false
            #Use detach so that the gradients for the Generator are not calculated
            print(G.forward(generate_random_normal(100).view(100,1,1)).detach().shape)
            D.train(G.forward(generate_random_normal(100)).detach(),torch.zeros(size=(128,1)))

            #train the generator
            G.train(D,generate_random_normal(100),torch.ones(size=(128,1)))
