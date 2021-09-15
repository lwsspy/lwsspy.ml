# %% Create Nerual Net
from torchsummary import summary
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pprint import pprint
import torch
from torch.utils import data
from lwsspy.ml.nn.akshaynn import AkshayNet
from lwsspy.ml.dataset.ccpdataset import CCPDataset
from torch.utils.data import DataLoader

# %%
model = AkshayNet()
input = torch.randn(1, 1, 55, 55)
out = model(input)
# %% Check out the layers
summary(model, input_size=(1, 55, 55))


# %% Check learnable parameters

params = list(model.parameters())
print(len(params))
for param in params:
    print(param.size())

# %% Try some random input


print(out)

# %% Loading an existing dataset

filename = \
    '/Users/lucassawade/OneDrive/Research/RF/DATA/GLImER/' \
    'ccps/US_P_0.58_minrad_3D_it_f2_volume_labeled.npz'
sq_size = 55
dataset = CCPDataset(filename, sq_size=sq_size)

print("Labels: ")
print(dataset.labeldict)

# %%

# %% Checkout the implemented sample plotting tool (nothing crazy)
dataset.plot_samples()

# %% Create Training data


def trainTestSplit(dataset, TTR=0.7, rand=True):
    """Randomized splitting"""
    N = len(dataset)
    if rand:
        indices = torch.randperm(len(dataset))
    else:
        indices = range(0, N)
    indices0 = indices[:int(TTR * N)]
    indices1 = indices[int(TTR * N):]
    trainDataset = torch.utils.data.Subset(
        dataset, indices0)
    valDataset = torch.utils.data.Subset(
        dataset, indices1)
    return trainDataset, valDataset


# %%
training_data, test_data = trainTestSplit(dataset, TTR=0.7)

# %%
# Decimate training/tests data
for _ in range(10):
    training_data, test_data = trainTestSplit(training_data, TTR=0.7)


print("Ntrain:", len(training_data))
print("Ntest:", len(test_data))

# %%
# Create dataloaders
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# %% Setup Optimization

# Hyper Parameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Initialize Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %% Define Training Loop


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# %% Define test_loop

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# %% Optimization

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
