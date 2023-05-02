import torchvision
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

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
    


def Train(Model, train_input, val_input):
    num_epochs=30
    train_acc=[0]*num_epochs
    val_acc=[0]*num_epochs
    train_loss=[0]*num_epochs
    val_loss=[0]*num_epochs
    for epch in range(num_epochs):
        Model.train()
        for x_batch, y_batch in train_input:
            x_batch=x_batch.to(dev)
            y_batch=y_batch.to(dev)
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




    