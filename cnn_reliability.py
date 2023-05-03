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
import json
import ast


global dev
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential()
        self.model.add_module('conv1',nn.Conv1d(2,16,3,1,1))
        self.model.add_module('relu1',nn.LeakyReLU(.1))
        self.model.add_module('pool1',nn.MaxPool1d(2,2))

        self.model.add_module('conv2',nn.Conv1d(16,32,3,1,1))
        self.model.add_module('relu2',nn.LeakyReLU(.1))
        self.model.add_module('pool2',nn.MaxPool1d(2,2))

        self.model.add_module('conv3',nn.Conv1d(32,64,3,1,1))
        self.model.add_module('relu3',nn.LeakyReLU(.1))
        self.model.add_module('pool3',nn.MaxPool1d(2,2))

        self.model.add_module('flat', nn.Flatten())
        self.model.add_module('d1', nn.Dropout(.2))
        self.model.add_module('fc1',nn.Linear(196,64))
        self.model.add_module('relu4', nn.LeakyReLU(.05))
        self.model.add_module('fc2',nn.Linear(64,1))
        self.model.add_module('out', nn.Sigmoid())

        self.model.to(dev)

        self.loss_fn=nn.BCELoss()

        self.optim=torch.optim.Adam(self.parameters(), lr=1e-3)
    def forward(self,inputs):
        return self.model(inputs)


def Train(Model, train_input, val_input):
    num_epochs=30
    train_acc=[0]*num_epochs
    val_acc=[0]*num_epochs
    train_loss=[0]*num_epochs
    val_loss=[0]*num_epochs
    for epch in range(num_epochs):
        Model.train()
        for x_batch, y_batch in train_input:
            x_batch=x_batch.to(dev).float()
            y_batch=y_batch.to(dev).float()
            pred=Model(x_batch)[:0]
            loss=Model.loss_fn(pred, y_batch.float())
            loss.backward()
            Model.optim.step()
            Model.optim.zero_grad()
            train_loss[epch]+=loss.item()*y_batch.size(0)
            is_correct=((pred>=0.5).float() == y_batch).float()
            train_acc[epch]+=is_correct.sum().cpu()

        train_loss[epch]/=len(train_input)
        train_acc[epch]/=len(train_input)

        Model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_input:
                x_batch=x_batch.to(dev)
                y_batch=y_batch.to(dev)
                pred = Model(x_batch)[:, 0]
                loss = Model.loss_fn(pred, y_batch.float())
                val_loss[epch] += loss.item()*y_batch.size(0) 
                is_correct = ((pred>=0.5).float() == y_batch).float()
                val_acc[epch] += is_correct.sum().cpu()
        val_loss[epch] /= len(val_input.dataset)
        val_acc[epch] /= len(val_input.dataset)
        print(f'Epoch {epch+1} accuracy: {train_acc[epch]:.4f} val_accuracy: {val_acc[epch]:.4f}')
    return train_loss, val_loss, train_acc, val_acc


class Data_NN(Dataset):
    def __init__(self,d,labels):
        self.data=d
        self.labels=labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        data= self.data[index,:,:]
        lab=self.labels[index]
        return data, lab


def Clean_Data():
    df_1=pd.read_csv("NN_data_1.csv")
    df_2=pd.read_csv("NN_data_2.csv")
    df_3=pd.read_csv("NN_data_3.csv")
    df_4=pd.read_csv("NN_data_4.csv")
    LD_1=df_1['LoadData'].to_numpy()
    LD_2=df_2['LoadData'].to_numpy()
    LD_3=df_3['LoadData'].to_numpy()
    LD_4=df_4['LoadData'].to_numpy()
    PD_1=df_1['PowerData'].to_numpy()
    PD_2=df_2['PowerData'].to_numpy()
    PD_3=df_3['PowerData'].to_numpy()
    PD_4=df_4['PowerData'].to_numpy()
    Label_1=df_1['Fail'].to_numpy()
    Label_2=df_2['Fail'].to_numpy()
    Label_3=df_3['Fail'].to_numpy()
    Label_4=df_4['Fail'].to_numpy()
    Data=np.empty(shape=(np.shape(LD_1)[0]+np.shape(LD_2)[0]+np.shape(LD_3)[0]+np.shape(LD_4)[0],2,24))
    Labels=np.empty(shape=(np.shape(LD_1)[0]+np.shape(LD_2)[0]+np.shape(LD_3)[0]+np.shape(LD_4)[0]))
    for j,(dummy,pdummy) in enumerate(zip(LD_1,PD_1)):
        dummy=LD_1[j].split('.')
        pdummy=PD_1[j].split('.')
        for i,x in enumerate(dummy):
            if x[:2]=='\r\n':
                x=x[2:]
            if x[0]=='[':
                x=x[1:]
            if x==']':
                break
            dummy[i]=float(x)
        for i,x in enumerate(pdummy):
            if x[:2]=='\r\n':
                x=x[2:]
            if x[0]=='[':
                x=x[1:]
            if x==']':
                break
            pdummy[i]=float(x)
        del dummy[-1]
        del pdummy[-1]
        Data[j,0,:]=dummy
        Data[j,1,:]=pdummy
        Labels[j]=Label_1[j]
    for j,(dummy,pdummy) in enumerate(zip(LD_2,PD_2)):
        dummy=LD_2[j].split('.')
        pdummy=PD_2[j].split('.')
        for i,x in enumerate(dummy):
            if x[:2]=='\r\n':
                x=x[2:]
            if x[0]=='[':
                x=x[1:]
            if x==']':
                break
            dummy[i]=float(x)
        for i,x in enumerate(pdummy):
            if x[:2]=='\r\n':
                x=x[2:]
            if x[0]=='[':
                x=x[1:]
            if x==']':
                break
            pdummy[i]=float(x)
        del dummy[-1]
        del pdummy[-1]
        Data[j+np.shape(LD_1)[0],0,:]=dummy
        Data[j+np.shape(LD_1)[0],1,:]=pdummy
        Labels[j+np.shape(LD_1)[0]]=Label_2[j]
    for j,(dummy,pdummy) in enumerate(zip(LD_3,PD_3)):
        dummy=LD_3[j].split('.')
        pdummy=PD_3[j].split('.')
        for i,x in enumerate(dummy):
            if x[:2]=='\r\n':
                x=x[2:]
            if x[0]=='[':
                x=x[1:]
            if x==']':
                break
            dummy[i]=float(x)
        for i,x in enumerate(pdummy):
            if x[:2]=='\r\n':
                x=x[2:]
            if x[0]=='[':
                x=x[1:]
            if x==']':
                break
            pdummy[i]=float(x)
        del dummy[-1]
        del pdummy[-1]
        Data[j+np.shape(LD_1)[0]+np.shape(LD_2)[0],0,:]=dummy
        Data[j+np.shape(LD_1)[0]+np.shape(LD_2)[0],1,:]=pdummy
        Labels[j+np.shape(LD_1)[0]+np.shape(LD_2)[0]]=Label_3[j]
    for j,(dummy,pdummy) in enumerate(zip(LD_4,PD_4)):
        dummy=LD_4[j].split('.')
        pdummy=PD_4[j].split('.')
        for i,x in enumerate(dummy):
            if x[:2]=='\r\n':
                x=x[2:]
            if x[0]=='[':
                x=x[1:]
            if x==']':
                break
            dummy[i]=float(x)
        for i,x in enumerate(pdummy):
            if x[:2]=='\r\n':
                x=x[2:]
            if x[0]=='[':
                x=x[1:]
            if x==']':
                break
            pdummy[i]=float(x)
        del dummy[-1]
        del pdummy[-1]
        Data[j+np.shape(LD_1)[0]+np.shape(LD_2)[0]+np.shape(LD_3)[0],0,:]=dummy
        Data[j+np.shape(LD_1)[0]+np.shape(LD_2)[0]+np.shape(LD_3)[0],1,:]=pdummy
        Labels[j+np.shape(LD_1)[0]+np.shape(LD_2)[0]+np.shape(LD_3)[0]]=Label_3[j]
    return Data,Labels



if __name__=='__main__':
    Data,Label=Clean_Data()
    print(Data)
    dataset=Data_NN(Data,Label)
    gen_1=torch.Generator().manual_seed(42)
    train, val, test =random_split(dataset,[.8,.1,.1],gen_1)
    train_dl=DataLoader(train,32,True)
    val_dl=DataLoader(val,32,False)
    test_dl=DataLoader(test,32,False)
    mod=Model_CNN()
    train_loss, val_loss, train_acc, val_acc = Train(mod,train_dl,val_dl)

    