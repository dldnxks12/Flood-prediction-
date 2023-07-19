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
    def __init__(self, IO_dim, batch_size=69):
        super(AE, self).__init__()
        self.batch_size = batch_size

        self.enco0 = nn.Linear(IO_dim, 128)
        self.ebn0 = nn.BatchNorm1d(128)
        self.enco1 = nn.Linear(128, 64)
        self.ebn1 = nn.BatchNorm1d(64)
        self.enco2 = nn.Linear(64, 36)

        self.deco0 = nn.Linear(36, 64)
        self.dbn0 = nn.BatchNorm1d(64)
        self.deco1 = nn.Linear(64, 128)
        self.dbn1 = nn.BatchNorm1d(128)
        self.deco2 = nn.Linear(128, IO_dim)

    def enco(self, data):

        x = F.relu(self.ebn0(self.enco0(data)))
        x = F.relu(self.ebn1(self.enco1(x)))
        lv = self.enco2(x)
        return lv

    def deco(self, lv):
        x = F.relu(self.dbn0(self.deco0(lv)))
        x = F.relu(self.dbn1(self.deco1(x)))
        org = self.deco2(x)

        # output range 설정?
        return org


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(36, 72)
        self.bn1 = nn.BatchNorm1d(72)
        self.fc2 = nn.Linear(72, 72)
        self.bn2 = nn.BatchNorm1d(72)
        self.fc3 = nn.Linear(72, 36)
        self.do = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do(x)
        x = self.fc3(x)
        return x
