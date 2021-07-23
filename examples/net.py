#%% Create Nerual Net
import torch
from lwsspy.ml.nn.akshaynn import AkshayNet

an = AkshayNet()

#%% Check out the layers
print(an)


#%% Check learnable parameters

params = list(an.parameters())
print(len(params))
for param in params:
    print(param.size())

#%% Try some random input


input = torch.randn(1, 1, 33, 33)
out = an(input)
print(out)
# %% Zero the gradient buffers of all parameters and backprops with random gradients:

# an.zero_grad()
x = torch.randn((1, 3))
print(x)
out.backward(x)