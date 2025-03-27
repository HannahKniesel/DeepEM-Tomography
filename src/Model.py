import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deepEM.Model import AbstractModel

# positional encoding maps the 3d input position to fourier features 
def posenc_fct(n_posenc, x):
    rets = [x]
    for i in range(n_posenc):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn((np.pi*2.**i)*x))
    return torch.cat(rets,-1)

# define model as MLP to predict single density value given a 3D position in the reconstruction space
class Model(AbstractModel):
    def __init__(self, n_posenc=10, n_features=256, n_layers=6, skip_layer=0):
        super(Model, self).__init__()
        self.n_posenc = n_posenc
        self.skip_layer = skip_layer

        input_size = n_posenc * 2 * 3 # compute input size based on positional encoding

        self.layers = torch.nn.ModuleList()
        
        # Input layer
        self.layers.append(torch.nn.Linear(input_size, n_features))
        self.layers.append(torch.nn.ReLU())
        
        # Hidden layers
        for i in range(n_layers - 1):
            if i == skip_layer - 1:
                self.layers.append(torch.nn.Linear(n_features + input_size, n_features))  
            else:
                self.layers.append(torch.nn.Linear(n_features, n_features))
            self.layers.append(torch.nn.ReLU())
        
        # Output layer
        self.output_layer = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        # application of positional encoding
        x = posenc_fct(self.n_posenc, x)
        x = x[:, 3:]
        identity = x

        for i, layer in enumerate(self.layers):
            if i == (2 * self.skip_layer - 1):  
                x = torch.cat((x, identity), axis=-1)
            x = layer(x)

        x = self.output_layer(x)

        return x



