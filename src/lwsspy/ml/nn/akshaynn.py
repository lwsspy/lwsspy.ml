

import torch
import torch.nn as nn
import torch.nn.functional as F


class AkshayNet(nn.Module):
    """
    Here I think I will want to make it customizable in terms of input grayscale
    image, because compared to a lot of the grinder images, mine are quite
    coarse.
    """

    def __init__(self, outputdim=3):
        super(AkshayNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolutiond
        # kernel
        self.conv1 = nn.Conv2d(1, 64, 7)
        # 64 input channels, 64 output channels but now 3x3 square convolutioned
        self.conv2 = nn.Conv2d(64, 64, 3)

        # This step makes no sense to me
        # self.conv3 = nn.Conv2d(64, 4, 21)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64 * 2 * 2, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, outputdim)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        print("1")
        x = self.conv1(x)
        print(x.size())
        x = F.relu(x)
        print(x.size())
        x = F.max_pool2d(x, 4)
        print(x.size())

        print("2")
        # If the size is a square, you can specify with a single number
        x = self.conv2(x)
        print(x.size())
        x = F.relu(x)
        print(x.size())
        x = F.max_pool2d(x, 2)
        print(x.size())

        # Softmax
        print("3")
        # x = self.conv3(x)
        x = F.relu(x)
        print(x.size())
        x = F.softmax(x, dim=1)
        print(x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = self.fc3(x)
        print(x.size())
        

        return x

        
        
        # return x
