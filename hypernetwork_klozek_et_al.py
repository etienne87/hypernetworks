"""
We reproduce exactly here:

"Hypernetwork functional image representation"
Sylwester Klocek et al.

http://ww2.ii.uj.edu.pl/~smieja/publications/ICANN2019.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mgrid(dims):
    tensors = tuple([torch.linspace(-1, 1, steps=sidelen) for sidelen in dims])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(dims))
    return mgrid



def bnconv3relu(cin,cout,stride):
    return nn.Sequential(nn.BatchNorm2d(cin),
                         nn.Conv2d(cin,cout,3,stride,1),
                         nn.ReLU())

class NaiveInception(nn.Module):
    def __init__(self,nonlin='ReLU'):
        super().__init__()
        conv1 = nn.Conv2d(3,3,1,1,0)
        conv3 = nn.Conv2d(3,3,3,1,1)
        conv5 = nn.Conv2d(3,3,5,1,2)
        pool = nn.AvgPool2d(3,1,1)
        self.convs = nn.ModuleList([conv1,conv3,conv5,pool])
        self.nonlin = getattr(nn, nonlin)()

    def forward(self, x):
        ys = [conv(x) for conv in self.convs]
        y = torch.cat(ys, dim=1)
        return self.nonlin(y)


class HyperNetworkKlozek(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.naive_inception = NaiveInception()
        self.res_branch = nn.Sequential(
                            bnconv3relu(12,12,1),
                            bnconv3relu(12,12,1),
                            bnconv3relu(12,12,1))
        self.pool = nn.Sequential(
                        bnconv3relu(12,24,1),
                        nn.MaxPool2d(3,2,1),
                        bnconv3relu(24,dim,1))

        self.convs = nn.ModuleList()
        self.fc_sizes = [2,32,64,256,64,3]
        self.weight_makers = nn.ModuleList()
        self.bias_makers = nn.ModuleList()
        self.cin_cout = []
        for i in range(len(self.fc_sizes)-1):
            cin = self.fc_sizes[i]
            cout = self.fc_sizes[i+1]
            self.cin_cout.append((cin,cout))
            self.convs.append(bnconv3relu(dim,dim, 2))
            weight_module = nn.Linear(dim,cin*cout)
            bias_module = nn.Linear(dim,cout)
            self.weight_makers.append(weight_module)
            self.bias_makers.append(bias_module)

        self.grid = None

    def forward(self, x):
        height,width = x.shape[-2:]
        if self.grid is None:
            self.grid = get_mgrid([height,width]).to(x)
        # common branch
        x = self.naive_inception(x)
        x = self.res_branch(x) + x
        x = self.pool(x)

        b,c,h,w = x.shape
        #x = x.view(b,c,-1).permute(0,2,1).contiguous()
        #x = x.mean(dim=1)

        # weight 1,2
        z = self.grid.clone()[None].repeat(b,1,1)
        for i, (conv,wm,bm) in enumerate(zip(self.convs, self.weight_makers, self.bias_makers)):
            x = conv(x)
            y = x.view(b,c,-1).permute(0,2,1).contiguous()
            y = y.mean(dim=1)
            cin, cout = self.cin_cout[i]
            weight = wm(y).view(-1,cin,cout)
            bias = bm(y).view(-1,1,cout)
            z = torch.bmm(z, weight) + bias
            z = torch.sin(z) if i < len(self.weight_makers)-1 else torch.sigmoid(z)
        y = z.reshape(b,height,width,3)
        y = y.permute(0,3,1,2).contiguous()
        return y




