import os
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import kornia
import tqdm

from hypernetwork import HyperNetwork
from hypernetwork_klozek_et_al import HyperNetworkKlozek

def loader(im_path, w, h):
    im = cv2.imread(im_path)
    im = cv2.resize(im, (w,h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im).permute(2,0,1)


def make_grid(x):
    batch_size = len(x)
    nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
    return torchvision.utils.make_grid(x,nrow=nrows).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)



def main(path, save_path='hyper.ckpt', lr=0.001, batch_size=32, viz_batch_size=8, height=128, width=128, hr_height=128, hr_width=128, viz_every=10, epochs=10, num_workers=2, device='cuda:0', resume=False):

    loader_fn = lambda x:loader(x,hr_width,hr_height)
    dataset = torchvision.datasets.ImageFolder(path, loader=loader_fn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=num_workers)


    mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)[None,:,None,None]
    std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)[None,:,None,None]

    net = HyperNetwork(3, img_size=width)
    if os.path.exists(save_path) and resume:
        net.load_state_dict(torch.load(save_path))

    net.to(device)
    optim = torch.optim.AdamW(net.parameters(), lr)

    for epoch in range(epochs):

        with tqdm.tqdm(dataloader, total=len(dataloader)) as tq:
            for i, (x, _) in enumerate(tq):

                x = x.to(device)
                x_in = (x.float()/255.0-mean)/std

                net.hyper.set_size(x_in)

                x_net = F.interpolate(x_in, size=(height, width), mode='bilinear', align_corners=True)
                optim.zero_grad()
                y = net(x_net)
                loss = torch.nn.functional.smooth_l1_loss(x_in, y)
                loss += kornia.losses.ssim(x_in, y, 5, reduction='mean')
                loss.backward()
                optim.step()

                tq.set_description('\rtrain_loss: %.11f'%loss.item())

                if i % viz_every == 0:
                    x_in = x_in[:viz_batch_size]
                    y = y[:viz_batch_size]

                    x_in = (x_in*std+mean)*255.0
                    y = (y*std+mean)*255.0

                    im_x_in = make_grid(x_in)
                    im_y = make_grid(y)
                    cat = np.concatenate((im_x_in, im_y), axis=1)

                    cv2.imshow('res', cat)
                    cv2.waitKey(5)

        torch.save(net.state_dict(), save_path)



if __name__ == '__main__':
    import fire;fire.Fire(main)
