# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets,transforms


model_path = os.path.join(os.getcwd(), "models", "mnist")
model_path += ".pth"

fig_path = os.path.join(os.getcwd(), "iamges")


# import data
def load_data():
    # download datasets
    train_data = datasets.MNIST(
        root='data',
        train = True,
        download=True,
        transform=transforms.ToTensor())

    test_data = datasets.MNIST(
        root='data',
        train = False,
        download=True,
        transform=transforms.ToTensor())
    
    # use DaraLoader for batching datasets
    train_loader = DataLoader(train_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=test_data.targets.size()[0])
    
    return train_data, test_data, train_loader, test_loader


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
    train_losses  = torch.zeros(epoch)
    
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
        train_losses[epochi] = np.mean(batch_loss)
        train_accuracy.append(np.mean(batch_accuracy))
        
        # net.eval()
        with torch.no_grad():
            X,y = next(iter(test_loader))
            yHAT = net(X)
            test_accuracy.append(100*torch.mean((torch.argmax(yHAT,axis=1) == y).float()))
            # test_losses = loss(y, yHAT)
            
        print("Epoch: {}, Loss:  {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(epochi + 1, train_losses[epochi].item(), train_accuracy[epochi], test_accuracy[epochi]))
        
    return net, train_losses, train_accuracy, test_accuracy

def evaluate(net, test_loader):
    net.eval()
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct/labels.size(0)
        print(f'Accuracy for this batch {accuracy}%')
        

def save_model(net, model_apth):
    torch.save(net.state_dict(), model_apth)
        

def load_model(model_path):
    # Load Model
    mnist_model = Net()
    mnist_model.load_state_dict(torch.load(model_path, weights_only=True))
    mnist_model.eval()
    return mnist_model


def plot_data(data):
    # schow 12 randomly iamges
    fig,axs = plt.subplots(3,4,figsize=(20,10))
    for ax in axs.flatten():
        # pick a random image
        random_index = np.random.randint(0,high=data.targets.size(0))
        img, label = data[random_index]
        ax.imshow(img.numpy()[0], cmap='gray')
        ax.set_title(f'The number: {label}')

    plt.suptitle('Sample of MNIST Dataset',fontsize=20)
    plt.tight_layout(rect=[0,0,1,.95])
    plt.show()
    
    
def plot_results(train_losses, test_losses, train_accuracy, test_accuracy):
    fig, axs = plt.subplots(1, 2, figsize=(20,10))
    axs[0].plot(train_losses)
    axs[0].plot(test_losses)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss function")
    axs[0].set_title("Train Loss: {:.4f}   Test Loss: {:.4f}".format(train_losses[-1], test_losses[-1]))

    axs[1].plot(train_accuracy)
    axs[1].plot(test_accuracy)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Train Accuracy: {:.2f}%   Test Accuracy: {:.2f}%".format(train_accuracy[-1], test_accuracy[-1]))

    plt.suptitle("ANN MNIST Results", fontsize=20)
    
    
def main():
    train_data, test_data, train_loader, test_loader = load_data()
    net, loss, optimizer = definition(Net)
    net, train_losses, train_accuracy, test_accuracy = train(10, net, loss, optimizer, train_loader, test_loader)
    # print(net)
    
    
if __name__ == "__main__":
    main()