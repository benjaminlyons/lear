
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms
from progress.bar import Bar
from prettytable import PrettyTable
from ae import AutoEncoder
from vae import VAE, Encoder, Decoder

img_transform = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
model = torch.load("models/best_model.pth")
vae = model['model']
validation_data = torchvision.datasets.ImageFolder('data/friends_faces', transform=img_transform)
# validation_data = torchvision.datasets.ImageFolder('data/img_align', transform=img_transform)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1, shuffle=True)
count = 0

# vae.train()
vae.eval()
for images, _ in validation_loader:
    images = images.cuda()
    output, mu, logvar = vae.forward(images)
    std = logvar.mul(0.5).exp_()
    eps = torch.empty_like(std).normal_()
    latent_sample =  eps.mul(std).add_(mu)
    # print(latent_sample)
    output = vae.decoder(latent_sample)
    images = images.cpu()

    images = torch.mul(images, 255).type(torch.uint8)
    images = images.squeeze()

    output = output.cpu().detach()
    output = torch.mul(output, 255).type(torch.uint8)
    output = output.squeeze()

    torchvision.io.write_jpeg(images, 'output/friend_outputs/real' + str(count) + '.jpg')
    torchvision.io.write_jpeg(images, 'output/friend_outputs/real' + str(count) + '.jpg')
    # torchvision.io.write_jpeg(output, 'output/recon/fake' + str(count) + '.jpg')
    # torchvision.io.write_jpeg(output, 'output/recon/fake' + str(count) + '.jpg')
    count += 1
