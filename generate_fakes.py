
import torch
import torch.nn as nn
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms

import numpy as np
from deepfake import AutoEncoder
from vae import VAE, Encoder, Decoder
from progress.bar import Bar
from prettytable import PrettyTable

def deepfake():
    # trump = torch.load("biden_model.pth")["model"]
    model = torch.load("best_model.pth")["model"]    
    # model = torch.load("latent128_model.pth")["model"]    

    # trump.eval()
    model.eval()
    
    # img_transform = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
    # dataset = torchvision.datasets.ImageFolder('trump', transform=img_transform)
    # training_size = int(len(dataset)*.8)
    # validation_size = len(dataset) - training_size
    # [training_data, validation_data] = torch.utils.data.random_split(dataset, [training_size, validation_size])
    # training_loader = torch.utils.data.DataLoader(training_data, batch_size=512, shuffle=True)
    # validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1, shuffle=True)
    #
    count = 0
    row = 0
    LATENT = 48
    for i in range(81):
        mean_vector = np.zeros((1,LATENT), dtype=float)
        # mean_vector[0][0] = val
        # col = 0
        # for i in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        latent_sample = torch.Tensor(mean_vector).normal_().cuda()
        output = model.decoder(latent_sample)
        output = output.cpu().detach()
        output = torch.mul(output, 255).type(torch.uint8)
        output = output.squeeze()
        torchvision.io.write_jpeg(output, 'fakes/fake' + str(count) + '.jpg')
        count += 1

deepfake()
