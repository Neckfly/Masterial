from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class TransformerHyper(nn.Module):
    def __init__(self, ray_hidden_dim=100, model = None):
        super().__init__()
        self.ray_mlp = nn.Sequential(
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        self.names = model.state_dict().keys()
        self.shapes = []
        #self.model = model

        for name, W in model.state_dict().items():
            shape = ()
            dim = 1

            for i in range(len(W.size())):
                shape += (W.size(dim=i),)
                dim *= W.size(dim=i)
            
            self.shapes.append(shape)
            
            setattr(self, name, nn.Linear(ray_hidden_dim, dim))
            

        #self.in_dim = 12
        #self.dims = [256, 256, 256, 2] # [256, 256, 256, 7]

        #prvs_dim = self.in_dim
        #for i, dim in enumerate(self.dims):
            #setattr(self, f"fc_{i}_weights", nn.Linear(ray_hidden_dim, prvs_dim * dim))
            #setattr(self, f"fc_{i}_bias", nn.Linear(ray_hidden_dim, dim))
            #prvs_dim = dim

    def forward(self, ray):
        out_dict = OrderedDict()
        features = self.ray_mlp(ray)

        '''
        for name, W in self.model.state_dict().items(): 
            dim = ()

            for i in range(len(W.size())):
                dim += (W.size(dim=i),)

            out_dict[name] = self.__getattr__(name)(
                features
            ).reshape(dim)   
        '''

        for i, name in enumerate(self.names): 
            out_dict[name] = self.__getattr__(name)(
                features
            ).reshape(self.shapes[i])  

        return out_dict

    '''
    def __init__(self, ray_hidden_dim=100):
        super().__init__()
        self.ray_mlp = nn.Sequential(
            nn.Linear(2, ray_hidden_dim), # nn.Linear(7, ray_hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        self.in_dim = 12
        self.dims = [256, 256, 256, 2] # [256, 256, 256, 7]

        prvs_dim = self.in_dim
        for i, dim in enumerate(self.dims):
            setattr(self, f"fc_{i}_weights", nn.Linear(ray_hidden_dim, prvs_dim * dim))
            setattr(self, f"fc_{i}_bias", nn.Linear(ray_hidden_dim, dim))
            prvs_dim = dim
    '''

    '''
    def forward(self, ray):
        out_dict = dict()
        features = self.ray_mlp(ray)

        prvs_dim = self.in_dim
        for i, dim in enumerate(self.dims):
            out_dict[f"fc_{i}_weights"] = self.__getattr__(f"fc_{i}_weights")(
                features
            ).reshape(dim, prvs_dim)
            out_dict[f"fc_{i}_bias"] = self.__getattr__(f"fc_{i}_bias")(
                features
            ).flatten()
            prvs_dim = dim

        return out_dict
    '''


class TargetTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weights):
        for i in range(int(len(weights) / 2)):
            x = F.linear(x, weights[f"fc_{i}_weights"], weights[f"fc_{i}_bias"])
            if i < int(len(weights) / 2) - 1:
                x = F.relu(x)
        return x
