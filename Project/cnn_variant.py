'''
Final Project
This code implements a variant of CNN with high classification accuracy on the Fashion-MNIST dataset.
Yue Lin (lin.3326 at osu.edu)
Created: 12/2/2020
'''

# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # TensorBoard support

# import torchvision module to handle image manipulation
import torchvision
import torchvision.transforms as transforms


# Use standard FashionMNIST dataset
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)

test_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = False,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)


# Build the baseline neural network
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # define layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Linear(6*6*48, 1000),
            nn.ReLU())
        self.fc1 = nn.Linear(in_features=1000, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)


    # define forward function
    def forward(self, t):
        t = self.layer1(t)
        t = self.layer2(t)
        t = t.reshape(t.size(0), -1)
        t = self.layer3(t)
        t = self.fc1(t)
        t = self.fc2(t)
        t = self.fc3(t)
        return t


# Auxiliary function that reports the accuracy on a dataset
def get_accuracy(model, dataloader):
    count = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0]
            labels = batch[1]
            preds = network(images)
            batch_correct = preds.argmax(dim=1).eq(labels).sum().item()
            batch_count = len(batch[0])
            count += batch_count
            correct += batch_correct
    model.train()
    return correct/count


# Train the model for three epochs (by default); report the training set accuracy after each epoch
lr = 0.001
batch_size = 1000
shuffle = True
epochs = 10

network = Network()
loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
optimizer = optim.Adam(network.parameters(), lr=lr)

# set the network to training mode
network.train()
for epoch in range(epochs):
    for batch in loader:
        images = batch[0]
        labels = batch[1]
        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch {0}: train set accuracy {1}'.format(epoch,get_accuracy(network,loader)))

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
print('Epoch {0}: test set accuracy {1}'.format(epoch,get_accuracy(network,test_loader)))