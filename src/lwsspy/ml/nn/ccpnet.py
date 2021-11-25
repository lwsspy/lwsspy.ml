import torch
import torch.nn as nn
import torch.nn.functional as F


class CCPNet(nn.Module):

    def __init__(self, nclasses=4):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.ups = nn.Upsample(
            scale_factor=(2, 2), mode='nearest')
        self.conv1 = nn.Conv2d(3, 16, 75, stride=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, 30, stride=(1, 1))
        # self.conv3 = nn.Conv2d(8, 10, 2)
        # self.conv4 = nn.Conv2d(10, 12, 2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(288, 96)  # 5*5 from image dimension
        self.fc2 = nn.Linear(96, 32)
        self.fc3 = nn.Linear(32, nclasses)

    def forward(self, x, v=False):
        # Max pooling over a (2, 2) window
        # x = self.ups(x)
        # if v:
        #     print(x.shape)
        if v:
            print("First layer")
        x = F.relu(self.conv1(x))
        if v:
            print(x.shape)
        x = F.dropout2d(F.max_pool2d(x, 2))
        if v:
            print(x.shape)
            print()
        # If the size is a square, you can specify with a single number
        if v:
            print("Second layer")
        x = F.relu(self.conv2(x))
        if v:
            print(x.shape)
        x = F.dropout2d(F.max_pool2d(x, 3))
        if v:
            print(x.shape)
            print()

        # # If the size is a square, you can specify with a single number
        # if v:
        #     print("Third layer")
        # x = F.relu(self.conv3(x))
        # if v:
        #     print(x.shape)
        # x = F.dropout2d(F.max_pool2d(x, 2))
        # if v:
        #     print(x.shape)
        #     print()

        # # If the size is a square, you can specify with a single number
        # if v:
        #     print("Forth layer")
        # x = F.relu(self.conv4(x))
        # if v:
        #     print(x.shape)
        # x = F.dropout2d(F.max_pool2d(x, 2))
        # if v:
        #     print(x.shape)
        #     print()

        # flatten all dimensions except the batch dimension
        if v:
            print("Flatten")
        x = torch.flatten(x, 1)
        if v:
            print(x.shape)
            print("")
            print("Linear 1")
        x = F.dropout(F.relu(self.fc1(x)))
        if v:
            print(x.shape)
            print("")
            print("Linear 2")
        x = F.dropout(F.relu(self.fc2(x)))
        if v:
            print(x.shape)
            print("")
            print("Linear 3")
        x = self.fc3(x)
        if v:
            print(x.shape)
            print("")
        return x
