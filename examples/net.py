# %% Create Nerual Net
# from torchinfo import summary
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pprint import pprint
import torch
from torch.utils import data
import numpy as np
from lwsspy.ml.nn.ccpnet import CCPNet
from lwsspy.ml.dataset.ccpdataset import CCPDataset
from lwsspy.ml.sampling.imbalanced import ImbalancedDatasetSampler
from lwsspy.ml.dataset.subset import CustomSubset
from lwsspy.ml.plot.plot_weights import plot_weights
from torch.utils.data import DataLoader, dataloader


# %%
model = CCPNet()
model = model.float()
tin = torch.randn(1, 3, 76, 76)
out = model.forward(tin, v=True)

# %% Check out the!pip layers
summary(model, input_size=(1, 3, 76, 76))


# %% Check learnable parameters

params = list(model.parameters())
print(len(params))
for param in params:
    print(param.size())

# %% Try some random input

# out = model(input)
# print(out)

# print(torch.max(out, 1))

# %% Loading an existing dataset

filename = \
    '/Users/lucassawade/OneDrive/Research/RF/DATA/GLImER/ccps/US_P_0.58_minrad_3D_it_f2_volume_labeled.npz'
sq_size = 151
dataset = CCPDataset(filename, sq_size=sq_size, ups=2.0)

print("Labels: ")
print(dataset.labeldict)

# %% Checkout the implemented sample plotting tool (nothing crazy)
dataset.plot_samples()


# %% Resample the dataset

def WRS(dataset, N=None):
    # distribution of classes in the dataset
    class_sample_count = np.array(
        [len(np.where(dataset.targets == t)[0]) for t in np.unique(dataset.targets)])
    weight = 1. / class_sample_count

    samples_weight = np.array(weight[dataset.targets])

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=samples_weight,
        num_samples=N,
        replacement=False)
    return sampler


wrs = WRS(dataset, 10000)

# c = {i: 0 for i in range(4)}
# for _i in wrs:
#     c[int(dataset[_i][1])] += 1

# %%
subset_indices = [i for i in wrs]
datasubset = CustomSubset(dataset, subset_indices,
                          labels=dataset.targets[subset_indices])

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

    trainDataset = CustomSubset(
        dataset, indices0, labels=dataset.targets[indices0])
    valDataset = CustomSubset(
        dataset, indices1, labels=dataset.targets[indices1])
    return trainDataset, valDataset


# %%
training_data, test_data = trainTestSplit(datasubset, TTR=0.7)

# # %%
# # Decimate training/tests data
# for _ in range(10):
#     training_data, test_data = trainTestSplit(training_data, TTR=0.7)

# %%
print("Ntrain:", len(training_data))
print("Ntest: ", len(test_data))

# %% Create dataloaders


# %%
batch_size = 64
train_dataloader = DataLoader(
    training_data, batch_size=batch_size,
    sampler=WRS(training_data, len(training_data)),
    num_workers=0)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size,
    sampler=WRS(test_data, len(test_data)),
    num_workers=0)


# train_dataloader = DataLoader(
#     training_data, batch_size=batch_size,
#     sampler=ImbalancedDatasetSampler(training_data, callback_get_label=lambda x: x.targets))
# test_dataloader = DataLoader(
#     test_data, batch_size=batch_size,
#     sampler=ImbalancedDatasetSampler(test_data, callback_get_label=lambda x: x.targets))

# %% Setup Optimization

# Hyper Parameters
learning_rate = 1e-2

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Initialize Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# %% Define Training Loop


def train_loop(dataloader, model, loss_fn, optimizer, device='cpu', num_batches=None):
    size = len(dataloader.dataset)

    if not num_batches:
        num_batches = len(dataloader)
    losses = []
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if num_batches == batch:
            break

    return np.sum(losses)/len(losses)


# %% Define test_loop

def test_loop(dataloader, model, loss_fn, device='cpu', num_batches=None):
    size = len(dataloader.dataset)

    if not num_batches:
        num_batches = len(dataloader)

    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if num_batches == batch:
                break

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct


# %% Optimization
epochs = 10
train_losses = []
test_losses = []
test_accs = []

# %%
plt.figure()
ax = plt.gca()
ax2 = ax.twinx()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    tloss, tacc = test_loop(test_dataloader, model, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(tloss)
    test_accs.append(tacc)
    eps = list(range(len(train_losses)))
    ax.plot(eps, train_losses, 'k', label='Train')
    ax.plot(eps, test_losses, 'r', label='Test')
    ax2.plot(eps, test_accs, 'b', label='Accuracy')
    if t == 0:
        ax.legend()
        ax2.legend()
    plt.pause(0.01)
print("Done!")


# %% Load model state

model = CCPNet()
model.load_state_dict(torch.load("model.state"))

# %% Get array from dataset to classify

dataset
x_np = torch.from_numpy(dataset.V[70, 60:93, 25:58]).reshape(
    1, 1, 33, 33).float()

plt.imshow(dataset.V[70, 60:93, 25:58].T, cmap='rainbow', aspect='auto')


# %% Testing in and output

pos = np.where(~np.equal(dataset.targets.numpy(), 3.0))
counter = 0

# %%
for i in np.random.choice(pos[0], size=100, replace=False):
    inn, out = dataset[i]
    counter += torch.max(model(inn.reshape(1, *inn.shape)),
                         1)[1].item() - out.item()

# %%
label = torch.ones_like(dataset.labeled['lV'])

pad = int((76 - 1)/2)
padval = 0
paddedV = torch.nn.functional.pad(
    dataset.V, pad=(pad, pad, pad, pad, pad, pad), mode='constant', value=0)
# padmask = dataset.labeldict['none']

# %%
L, M, N = label.shape
for idx in [223]:
    print(idx)
    for idy in range(M):
        print(idy)
        for idz in range(N):

            # Get excerpts
            ximage = paddedV[idx + pad,
                             idy: idy + 2 * pad + 1,
                             idz: idz + 2 * pad + 1].T

            yimage = paddedV[
                idx: idx + 2 * pad + 1,
                idy + pad,
                idz: idz + 2 * pad + 1].T

            zimage = paddedV[
                idx: idx + 2 * pad + 1,
                idy: idy + 2 * pad + 1,
                idz + pad]

            image = torch.stack((ximage, yimage, zimage), axis=0)

            label[idx, idy, idz] = torch.max(
                model(image.reshape((1, *image.shape))), 1)[1].item()


# %%
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(label[idx, :, :].T, cmap=dataset.bcmap,
           norm=dataset.bnorm, aspect='auto')
plt.gca().axis('off')
plt.title('ML - labeled')


plt.subplot(2, 2, 2)
plt.imshow(dataset.V[idx, :, :].T, cmap=dataset.imgcmap,
           norm=dataset.imgnorm, aspect='auto')
plt.gca().axis('off')
plt.title('RF section')

plt.subplot(2, 2, 3)
plt.imshow(dataset.labeled['lV'][idx, :, :].T, cmap=dataset.bcmap,
           norm=dataset.bnorm, aspect='auto')
plt.gca().axis('off')
plt.title('Train labels')


# # visualize weights for alexnet - first conv layer
# plot_weights(model, 0, single_channel=False)
