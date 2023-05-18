import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class AE(nn.Module):
    def __init__(self, IO_dim, batch_size = 69):
        super(AE, self).__init__()
        self.batch_size = batch_size
        self.encoder = nn.Linear(IO_dim, 36)
        self.decoder = nn.Linear(36, IO_dim)

    def enco(self, data):
        latent_vector = self.encoder(data)
        return latent_vector

    def deco(self, latent_vector):
        org_data = self.decoder(latent_vector)
        return org_data
