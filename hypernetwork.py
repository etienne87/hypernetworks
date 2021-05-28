"""
Naive Implementation

Should look more closely at
https://github.com/vsitzmann/siren/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from mlp_mixer import MLPMixer


def get_mgrid(dims):
    tensors = tuple([torch.linspace(-1, 1, steps=sidelen) for sidelen in dims])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(dims))
    return mgrid


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        #input: [B,N,D]
        if input.dim() == 3:
            b,n,d = input.shape
            x = input.view(-1,d)
        else:
            x = input
        y = torch.sin(self.omega_0 * self.linear(x))
        if input.dim() == 3:
            y = y.view(b,n,-1)
        return y



class DynamicSineLayer(nn.Module):
    def __init__(self, in_features, out_features, in_activations):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        total_len = in_features*out_features
        self.weight = nn.Sequential(nn.Linear(in_activations, 256, bias=True), nn.LeakyReLU(),
                                    nn.Linear(256, total_len))

        self.bias = nn.Sequential(nn.Linear(in_activations, 256, bias=True), nn.LeakyReLU(),
                                    nn.Linear(256, out_features))

    def forward(self, coords, fuel):
        weight = self.weight(fuel).reshape(len(coords), self.in_features, self.out_features)
        bias = self.bias(fuel).unsqueeze(1)
        return self._forward(coords, weight, bias)

    def _forward(self, coords, weight, bias, omega=1):
        #coords: [B,N,D]
        #weight: [B,D,K]
        #output: [B,N,K]
        weight *= omega
        y = torch.bmm(coords, weight) + bias
        return torch.sin(y)


class HyperRenderer(nn.Module):
    def __init__(self, in_activations, out_features, hidden_features=[128]*5, skip_input=0):
        super().__init__()
        self.out_features = out_features

        last = 2 + skip_input
        self.layers = []
        for hidden in hidden_features:
            self.layers.append(DynamicSineLayer(last, hidden, in_activations))
            last = hidden
        self.layers = nn.ModuleList(self.layers)

        self.final_linear = nn.Linear(hidden_features[-1], out_features)

        self.hidden_features = hidden_features
        self.grid = None

    def set_size(self, x):
        height, width = x.shape[-2:]
        if self.grid is not None and self.grid.shape[-2:] == (height,width):
            return
        self.grid = get_mgrid([height,width]).to(x)
        self.height = height
        self.width = width

    def forward(self, fuel):
        assert self.grid is not None
        batch_size = fuel.shape[0]
        coords = self.grid[None].repeat([batch_size,1,1])

        for layer in self.layers:
            coords = layer(coords, fuel)

        b,n,d = coords.shape
        x = coords.view(-1,d)
        y = self.final_linear(x)
        y = y.reshape(b,self.height,self.width,self.out_features)
        y = y.permute(0,3,1,2).contiguous()
        return y





class HyperNetwork(nn.Module):
    def __init__(self, cout=3, hyper_hidden=32, hypo_hidden=256, n_layers=2):
        super().__init__()
        self.hypo = MLPMixer(image_size=128, patch_size=16, cin=3, dim=128, num_classes=hypo_hidden, depth=3)
        self.hyper = HyperRenderer(hypo_hidden, 3)

    def forward(self, x):
        self.hyper.set_size(x)
        #TODO: add coordinates
        # coords = self.grid[None].repeat([batch_size,1,1])
        # x = torch.cat((x, coords), dim=1)
        """
        x = self.hypo[0](x)
        x = self.hypo[1](x)
        for i in range(2, len(self.hypo)):
            x = self.hypo[i](x)
            #mat-mul/ cross-attention to weights
            #perceiver style modification of weights
        """
        x = self.hypo(x)
        y = self.hyper(x)
        return y



if __name__ == '__main__':
    # hyper = HyperRenderer(100, 3)
    # hyper.set_size(torch.randn(64,64))
    # b,c = 4,100
    # x = torch.randn(b,c)
    # y = hyper(x)
    x = torch.randn(4,3,128,128)
    hyper = HyperNetwork(x)
    y = hyper(x)

    print(y.shape)
