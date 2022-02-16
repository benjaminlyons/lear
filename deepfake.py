import torch
import torch.nn as nn

# https://www.codeproject.com/Articles/5298025/Building-and-Training-Deep-Fake-Autoencoders
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=5, strides=2)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Conv2d(256, 512, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Conv2d(512, 1024, kernel_size=5, stride=2),
                nn.ReLU(),

        )
