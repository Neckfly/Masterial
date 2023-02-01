from torch import nn
from collections import OrderedDict

class HyperNet(nn.Module):
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

        for name, W in model.state_dict().items():
            shape = ()
            dim = 1

            for i in range(len(W.size())):
                shape += (W.size(dim=i),)
                dim *= W.size(dim=i)
            
            self.shapes.append(shape)
            
            setattr(self, name, nn.Linear(ray_hidden_dim, dim))

    def forward(self, ray):
        out_dict = OrderedDict()
        features = self.ray_mlp(ray)

        for i, name in enumerate(self.names): 
            out_dict[name] = self.__getattr__(name)(
                features
            ).reshape(self.shapes[i])  

        return out_dict