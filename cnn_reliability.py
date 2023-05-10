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
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn


global dev
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
global mod

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
        self.model.add_module('d1', nn.Dropout(.5))
        self.model.add_module('fc1',nn.Linear(192,64))
        self.model.add_module('relu4', nn.LeakyReLU(.05))
        self.model.add_module('d2', nn.Dropout(.2))
        self.model.add_module('fc2',nn.Linear(64,1))
        self.model.add_module('out', nn.Sigmoid())

        self.model.to(dev)

        self.loss_fn=nn.BCELoss()

        self.optim=torch.optim.Adam(self.parameters(), lr=1e-3)
    def forward(self,inputs):
        return self.model(inputs)


def Train(Model, train_input, val_input):
    num_epochs=45
    train_acc=[0]*num_epochs
    val_acc=[0]*num_epochs
    train_loss=[0]*num_epochs
    val_loss=[0]*num_epochs
    for epch in range(num_epochs):
        Model.train()
        for x_batch, y_batch in train_input:
            x_batch=x_batch.to(dev).float()
            y_batch=y_batch.to(dev).float()
            pred=Model(x_batch)[:,0]
            loss=Model.loss_fn(pred, y_batch.float())
            loss.backward()
            Model.optim.step()
            Model.optim.zero_grad()
            train_loss[epch]+=loss.item()*y_batch.size(0)
            is_correct=((pred>=0.5).float() == y_batch).float()
            train_acc[epch]+=is_correct.sum().cpu()

        train_loss[epch]/=len(train_input.dataset)
        train_acc[epch]/=len(train_input.dataset)

        Model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_input:
                x_batch=x_batch.to(dev).float()
                y_batch=y_batch.to(dev).float()
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

def Model_Eval(Load,Power):
    mod=Model_CNN()
    mod.cuda()
    mod.eval()
    data=torch.empty(size=(1,2,24), requires_grad=False, device='cuda:0')
    ld_tensor=torch.Tensor(Load)
    pd_tensor=torch.Tensor(Power)
    data[0,0,:]=ld_tensor
    data[0,1,:]=pd_tensor
    data = data.cuda().float()
    out = mod(data)[:,0]
    predicted = (out>=.5).bool()
    pred=predicted.detach().cpu().numpy().item()
    return pred


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
        Labels[j+np.shape(LD_1)[0]+np.shape(LD_2)[0]+np.shape(LD_3)[0]]=Label_4[j]
    return Data,Labels

def predict_with_pytorch(model,val_x):    
    y_preds = []
    val_x = val_x.cuda().float()
    out = model(val_x)[:,0]
    predicted = (out>=.5).float()
    for p in predicted:
        y_preds.append(p.detach().cpu().numpy().item())
    return y_preds    

def Fin():
    Data,Label=Clean_Data()
    print(np.shape(Data))
    print(np.shape(Data)[0])
    dataset=Data_NN(Data,Label)
    gen_1=torch.Generator().manual_seed(42)
    train, val, test =random_split(dataset,[.8,.1,.1],gen_1)
    train_dl=DataLoader(train,512,True)
    val_dl=DataLoader(val,32,False)
    test_dl=DataLoader(test,len(test),False)
    mod=Model_CNN()
    print(summary(mod,input_size=(2,24)))
    hist = Train(mod,train_dl,val_dl)
    x_arr = np.arange(len(hist[0])) + 1

    fig, (ax1)= plt.subplots(1,1)
    ax1.plot(x_arr, hist[0], '-o', label='Train loss')
    ax1.plot(x_arr, hist[1], '-->', label='Validation loss')
    ax1.grid()
    ax1.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Average Loss')
    ax1.set_title('Training and Validation Loss')
    plt.savefig('fig_1.png')
    plt.show()

    fig, (ax1)= plt.subplots(1,1)
    ax1.plot(x_arr, hist[2], '-o', label='Train Accuracy')
    ax1.plot(x_arr, hist[3], '-->', label='Validation Accuracy')
    ax1.grid()
    ax1.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    plt.savefig('fig_2.png')
    plt.show()

    input_ft=next(iter(test_dl))[0]
    labels=next(iter(test_dl))[1]
    accuracy_test=0.0

    #Set the trained model for evaluation to avoid calculation of gradients

    mod.eval()

    ##Begin evaluation without any gradient calculations
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            ##Move the test ba
            x_batch=x_batch.to(dev).float()
            y_batch=y_batch.to(dev).float()
            pred=mod(x_batch)[:,0]
            is_correct=((pred>=.5).float()==y_batch).float()
            accuracy_test+=is_correct.sum()
    ##Divide by the length to get a value between 0 and 1
    accuracy_test/=len(test)
    print("Test Accuracy: {0:.4f}".format(accuracy_test))
    pred=[]
    pred.append(predict_with_pytorch(mod,input_ft))

    cm = confusion_matrix(labels.numpy(),pred[0])
    plt.figure(figsize = (8,8),dpi=250)
    seaborn.heatmap(cm,annot=True)
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.title("Confusion Matrix for Test Data")
    plt.show()

    fpr, tpr, _ = roc_curve(labels.numpy(),  pred[0])

    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC Curve for Test Data")
    plt.show()