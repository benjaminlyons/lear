
import torch
import torch.nn as nn
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms

import numpy as np
from ae import AutoEncoder
from vae import VAE, Encoder, Decoder
from progress.bar import Bar
from prettytable import PrettyTable

def deepfake():
    # trump = torch.load("biden_model.pth")["model"]
    model = torch.load("models/best_model.pth")["model"]    
    # model = torch.load("latent128_model.pth")["model"]    

    # trump.eval()
    model.eval()
    
    count = 0
    row = 0
    LATENT = 256
    for val in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        mean_vector = np.zeros((1,LATENT), dtype=float)
        mean_vector[0][67] = val
        col = 0
        for i in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
            mean_vector[0][231] = i
            latent_sample = torch.Tensor(mean_vector).cuda()
            output = model.decoder(latent_sample)
            output = output.cpu().detach()
            output = torch.mul(output, 255).type(torch.uint8)
            output = output.squeeze()
            torchvision.io.write_jpeg(output, 'output/grids/dim_' + str(row) + '_' + str(col) + '.jpg')
            col += 1
        row += 1

deepfake()
