

import torch
import torch.nn as nn
import torch.nn.functional as F


class AkshayNet(nn.Module):
    """
    Here I think I will want to make it customizable in terms of input grayscale
    image, because compared to a lot of the grinder images, mine are quite
    coarse.
    """

    def __init__(self, outputdim=4):
        super().__init__()
        # 1 input image channel, 64 output channels, 5x5 square convolutiond
        # kernel
        self.conv1 = nn.Conv2d(1, 64, 7)
        # 64 input channels, 64 output channels but now 3x3 square convolutioned
        self.conv2 = nn.Conv2d(64, 64, 3)

        # This step makes no sense to me
        # self.conv3 = nn.Conv2d(64, 4, 21)

        # an affine operation: y = Wx + b

        # Change this number depending on the image input size
        # self.fc1 = nn.Linear(64 * 2 * 2, 120)  # 5*5 from image dimension
        self.fc1 = nn.Linear(1600, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, outputdim)

    def forward(self, x, verbose=False):

        # Max pooling over a (2, 2) window
        if verbose:
            print(0, x.shape)

        x = self.conv1(x)
        if verbose:
            print(1, x.shape)

        x = F.relu(x)
        if verbose:
            print(2, x.shape)

        x = F.max_pool2d(x, 4)
        if verbose:
            print(3, x.shape)

        # If the size is a square, you can specify with a single number
        x = self.conv2(x)
        if verbose:
            print(4, x.shape)

        x = F.relu(x)
        if verbose:
            print(5, x.shape)

        x = F.max_pool2d(x, 2)
        if verbose:
            print(6, x.shape)

        # Softmax
        # x = self.conv3(x)
        x = F.relu(x)
        if verbose:
            print(7, x.shape)

        x = F.softmax(x, dim=1)
        if verbose:
            print(8, x.shape)

        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        if verbose:
            print(9, x.shape)

        x = F.relu(self.fc1(x))
        if verbose:
            print(10, x.shape)

        x = F.relu(self.fc2(x))
        if verbose:
            print(11, x.shape)

        x = self.fc3(x)
        if verbose:
            print(12, x.shape)

        return x

        # return x
