import os
import sys

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim = 256):
        super(Discriminator, self).__init__()

        self.linear1    = nn.Linear(input_dim,  hidden_dim)
        self.linear2    = nn.Linear(hidden_dim, hidden_dim)
        self.linear3    = nn.Linear(hidden_dim, hidden_dim)
        self.linear4    = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x): # input : (batch_size, 1 x 28 x 28) flattened image
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.sigmoid(self.linear4(x)) # 0 ~ 1 사이로 -> 확률로 표현
        return x

class Generator(nn.Module):
    def __init__(self, noise_dim, out_img_size, hidden_dim=256):
        super(Generator, self).__init__()
        self.linear1    = nn.Linear(noise_dim,  hidden_dim)
        self.linear2    = nn.Linear(hidden_dim, hidden_dim)
        self.linear3    = nn.Linear(hidden_dim, out_img_size)
        self.relu       = nn.ReLU()
        self.tanh       = nn.Tanh()

    def forward(self, noise):
        x = self.relu(self.linear1(noise))
        x = self.relu(self.linear2(x))
        x = self.tanh(self.linear3(x))

        return x
