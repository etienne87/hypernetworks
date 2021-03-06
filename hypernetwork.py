"""
Naive Implementation

Should look more closely at
https://github.com/vsitzmann/siren/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from mlp_mixer import MLPMixer, MLPUnMixer


import cv2
import numpy as np
from torchvision.utils import make_grid

def my_make_grid(tensor):
    batch_size = len(tensor)
    nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
    return make_grid(tensor, nrow=nrows, padding=4, normalize=True).permute(1,2,0).contiguous().cpu().numpy()

def colorize(img):
    return cv2.applyColorMap((img*255).astype(np.uint8), cv2.COLORMAP_JET)

def viz_coords(coords, height, width, name='coords.jpg'):
    coords = coords.view(height,width,coords.shape[-1])
    coords = coords.permute(2,0,1)[:,None].contiguous()
    grid = my_make_grid(coords)
    grid = colorize(grid)
    cv2.imwrite(name, grid)


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
        self.cin = in_features
        self.cout = out_features

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-math.sqrt(6 / self.in_features) / self.omega_0,
                                             math.sqrt(6 / self.in_features) / self.omega_0)

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


def fourier_encode(x, max_freq, num_bands = 8, base = 2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(0., math.log(max_freq / 2) / math.log(base), num_bands, base = base, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)

    c,d = x.shape[-2:]
    x = x.view(len(x), c*d)
    return x




class DynamicSineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False):
        super().__init__()
        self.cin = in_features
        self.cout = out_features
        self.total_len = self.cin * self.cout
        self.omega_0 = 30.0/in_features if is_first else 1.0*math.sqrt(6.0/in_features)

    def forward_weight_and_bias(self, coords, weight, bias):
        #coords: [B,N,D]
        #weight: [B,D,K]
        #bias: [B,1,K]
        #output: [B,N,K]
        y = torch.bmm(coords, weight) + bias
        return torch.sin(y)

    def local_forward_weight_and_bias(self, coords, weight, bias):
        #coords: [B,N,D]
        #weights: [B,N,D,K]
        #biases: [B,N,K]
        #output: [B,N,K]
        y = torch.einsum('bnd,bndk->bnk', coords, weight)
        y = y + bias
        return torch.sin(y)




class HyperRenderer(nn.Module):
    def __init__(self, cin, out_features, hidden_features=[128]*5, patch_size=8):
        super().__init__()
        self.out_features = out_features

        last = 32
        #self.first_layer = SineLayer(cin, last, is_first=True)
        self.layers = []
        for i, hidden in enumerate(hidden_features):
            self.layers.append(DynamicSineLayer(last, hidden))
            last = hidden

        #add fixed sirens
        self.fixed_layers = nn.ModuleList()
        for i in range(2):
            hidden = 64
            self.fixed_layers.append(SineLayer(last, hidden))
            last = hidden

        self.layers = nn.ModuleList(self.layers)

        self.final_linear = nn.Linear(last, out_features)
        with torch.no_grad():
            self.final_linear.weight.uniform_(-math.sqrt(6.0 / last) / 30,
                                               math.sqrt(6.0 / last) / 30)


        self.hidden_features = hidden_features
        self.grid = None

        self.cin_cout = self.get_cin_cout()
        self.weight_sizes = [a*b for a,b in self.cin_cout]
        self.bias_sizes = [b for a,b in self.cin_cout]
        self.patch_size = patch_size

    def set_size(self, x):
        height, width = x.shape[-2:]
        if self.grid is not None and self.grid.shape[-2:] == (height,width):
            return
        p = self.patch_size
        self.grid = get_mgrid([width,height]).to(x)
        self.height = height
        self.width = width

    def modulo_patch_size(self, coords):
        scale = torch.FloatTensor([self.width,self.height])[None,None,:]
        scale = scale.to(coords)
        coords = coords * scale
        coords = coords.long()%self.patch_size
        coords = coords.float() / self.patch_size
        return coords

    def get_cin_cout(self):
        self.cin_cout = []
        for layer in self.layers:
            self.cin_cout.append((layer.cin,layer.cout))
        return self.cin_cout

    def get_weights_total_len(self):
        return sum([item.total_len for item in self.layers])

    def get_biases_total_len(self):
        return sum([item.cout for item in self.layers])

    def forward_weights_and_biases(self, x, field):
        assert self.grid is not None
        batch_size = x.shape[0]

        coords = self.grid
        #coords = self.modulo_patch_size(coords)
        #coords = self.first_layer(coords)
        coords = fourier_encode(coords, 1024)
        #viz_coords(coords, self.height, self.width)

        if coords.ndim == 2:
            coords = coords[None]
        coords = coords.repeat([batch_size,1,1])

        weights, biases = self.get_weights_and_biases(x, field)

        for layer, weight, bias in zip(self.layers, weights, biases):
            if field:
                coords = layer.local_forward_weight_and_bias(coords, weight, bias)
            else:
                coords = layer.forward_weight_and_bias(coords, weight, bias)

        for layer in self.fixed_layers:
            coords = layer(coords)

        b,n,d = coords.shape
        x = coords.view(-1,d)
        y = self.final_linear(x)
        y = y.reshape(b,self.height,self.width,self.out_features)
        y = y.permute(0,3,1,2).contiguous()
        return y

    def get_weights_and_biases(self, x, field):
        b = len(x)
        if field:
            n = x.shape[1]
        x = torch.split(x, self.weight_sizes+self.bias_sizes, dim=-1)
        weights, biases = x[:len(self.layers)], x[len(self.layers):]
        if field:
            weights = [weight.view(b,n,cin,cout) for weight,(cin,cout) in zip(weights,self.cin_cout)]
        else:
            weights = [weight.view(b,cin,cout) for weight,(cin,cout) in zip(weights,self.cin_cout)]
            biases = [bias.unsqueeze(1) for bias in biases]

        return weights, biases


class HypoResNet(nn.Module):
    def __init__(self, total):
        super().__init__()
        self.hypo = models.resnet18(pretrained=True)
        self.conv = nn.Conv2d(256, total, 1, 1, 0)

    def forward(self, x):
        x = self.hypo.maxpool(self.hypo.relu(self.hypo.bn1(self.hypo.conv1(x))))
        x = self.hypo.layer1(x)
        x = self.hypo.layer2(x)
        x = self.hypo.layer3(x)
        x = self.conv(x)
        return x

class HypoMLPMixer(nn.Module):
    def __init__(self, image_size, total, patch_size):
        super().__init__()
        self.hypo = MLPMixer(image_size=image_size, patch_size=patch_size, cin=3, dim=256, num_classes=total, depth=5, do_reduce=False)
        self.patch_size = patch_size

    def forward(self, x):
        b,c,h,w = x.shape
        p = self.patch_size
        x = self.hypo(x)
        x = x.reshape(b,h//p,w//p,x.shape[-1])
        x = x.permute(0,3,1,2)
        return x



class HyperNetwork(nn.Module):
    def __init__(self, cout=3, field=True, img_size=128):
        super().__init__()
        self.hyper = HyperRenderer(2, 3, [8,8], patch_size=16)
        total = self.hyper.get_weights_total_len() + self.hyper.get_biases_total_len()
        #self.hypo = HypoResNet(total)
        self.hypo = HypoMLPMixer(image_size=img_size, total=total, patch_size=16)
        self.field = field

    def forward(self, x, grad=True):
        b,c,h,w = x.shape
        x = self.hypo(x)

        if self.field:
            coords = self.hyper.grid[None].repeat([b,1,1])
            coords = coords.unsqueeze(2)
            x = F.grid_sample(x, coords, mode='bilinear', align_corners=True).squeeze()
            x = x.permute(0,2,1)
        y = self.hyper.forward_weights_and_biases(x, self.field)
        return y



if __name__ == '__main__':
    x = torch.randn(4,3,128,128)
    hyper = HyperNetwork(x)
    y = hyper(x)

    print(y.shape)
