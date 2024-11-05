# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets,transforms


# create ANN model for MNIST 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(784, 64)
        
        self.hidden_layer_01 = nn.Linear(64,64)
        self.hidden_layer_02 = nn.Linear(64,32)
        
        self.output = nn.Linear(32,10)
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1) # Because data shape is (28*28) for ANN must at first flatten
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_layer_01(x))
        x = F.relu(self.hidden_layer_02(x))
        x = self.output(x)   
        return torch.log_softmax(x, axis=1) # We use NLLLoss function for this
        
# Definition
def definition(model):
    net = model()
    loss = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    return net, loss, optimizer


# train function 
def train(epoch, net, loss, optimizer, train_loader, test_loader):
    train_accuracy  = []
    test_accuracy = []
    losses  = torch.zeros(epoch)
    
    for epochi in range(epoch):
        batch_accuracy  = []
        batch_loss = []
        
        # net.train()
        
        for X,y in train_loader:
            yHAT = net(X)
            lossfun = loss(yHAT, y)

            # 
            optimizer.zero_grad()
            lossfun.backward()
            optimizer.step()
            
            # Evaluate Metrics for this Batch
            batch_loss.append(lossfun.item())
            batch_accuracy.append( 100*torch.mean((torch.argmax(yHAT,axis=1) == y).float())) 
            
        # and get average losses across the batches
        losses[epochi] = np.mean(batch_loss)
        train_accuracy.append(np.mean(batch_accuracy))
        
        # net.eval()
        with torch.no_grad():
            X,y = next(iter(test_loader))
            yHAT = net(X)
            test_accuracy.append(100*torch.mean((torch.argmax(yHAT,axis=1) == y).float()))
            
        print("Epoch: {}, Loss:  {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(epochi + 1, losses[epochi].item(), train_accuracy[epochi], test_accuracy[epochi]))
        
    return net, losses, train_accuracy, test_accuracy