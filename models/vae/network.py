import os
import sys

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torchvision.utils import save_image

def reparameterize(mu, logvar):

    std = torch.exp(logvar/2)
    eps = torch.randn_like(std)
    return mu + eps*std

class Encoder(nn.Module):
    def __init__(self, input_dim, noise_dim, hidden_dim=128):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, noise_dim)
        self.logvar = nn.Linear(hidden_dim, noise_dim)

        self.relu = nn.ReLU()

    def forward(self, x):  # input : (batch_size, 1 x 28 x 28) flattened image
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))

        mu     = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterize(mu, logvar)

        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, input_dim, noise_dim, hidden_dim=128):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(noise_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x    = self.relu(self.linear1(x))
        x    = self.relu(self.linear2(x))
        x_re = F.sigmoid(self.linear3(x))
        return x_re




