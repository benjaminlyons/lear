
import torch
import torch.nn as nn
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms

import numpy as np
from vae import VAE, Encoder, Decoder
from progress.bar import Bar
from prettytable import PrettyTable

def deepfake():
    model = torch.load("models/best_model.pth")["model"].cuda()

    # trump.eval()
    model.eval()
    
    count = 0
    row = 0
    LATENT = 256
    for i in range(81):
        mean_vector = np.zeros((1,LATENT), dtype=float)
        latent_sample = torch.Tensor(mean_vector).normal_().cuda()
        output = model.decoder(latent_sample)
        output = output.cpu().detach()
        output = torch.mul(output, 255).type(torch.uint8)
        output = output.squeeze()
        torchvision.io.write_jpeg(output, 'output/fakes/fake' + str(count) + '.jpg')
        count += 1

deepfake()
