import torch 
import numpy as np
import torch.nn as nn

def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid

class PositionalEncoder():
    def __init__(self, embedding_size=256, coordinates_size=2, scale=4, device='cuda'):
        self.B = torch.randn((embedding_size, coordinates_size)) * scale
        self.B = self.B.to(device)

    def embedding(self, x):
        x_embedding = (2. * np.pi * x) @ self.B.t()
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding
    

# ref: https://github.com/liyues/NeRP/blob/1c7d0b980246a80f2694daf9e77af00b9b24b7f6/networks.py#L58
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)
    
class SIREN(nn.Module):
    def __init__(self, network_depth, network_width, network_input_size, network_output_size, device='cuda'):
        super(SIREN, self).__init__()

        self.network_depth = network_depth
        self.network_width = network_width
        self.network_input_size = network_input_size
        self.network_output_size = network_output_size
        self.loss = nn.MSELoss()

        num_layers = network_depth
        hidden_dim = network_width
        input_dim = network_input_size
        output_dim = network_output_size

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.device = device

    def forward(self, x):
        out = self.model(x)
        return out