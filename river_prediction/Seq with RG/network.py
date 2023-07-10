import os
import sys
import random
import numpy as np
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, device):
        super(LSTM, self).__init__()
        # Network params
        self.device          = device
        self.num_classes     = num_classes
        self.input_size      = input_size
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers

        # Network components
        self.leaky_relu = nn.LeakyReLU()
        self.lstm = nn.LSTM(input_size  = self.input_size,
                            hidden_size = self.hidden_size,
                            num_layers  = self.num_layers,
                            batch_first = True)

        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        # Define hidden state & cell state
            x       = (batch_size, seq_length, input_size) <- when batch_first == True
            c_0/h_0 = (num_layers, batch_size, input_size)
        """
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)

        """
            output : the output features h_t from the last layer of LSTM for each t 
            hn     : the final hidden state for each element in the sequence 
            cn     : the final cell state for each element in the sequence 
        """
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # LSTM에 입력 -> xt, ht-1, ct-1

        if self.num_layers == 1:
            hn = hn.view(-1, self.hidden_size)
        else:
            hn = hn[-1].view(-1, self.hidden_size)  # 맨 마지막 layer의 출력만 가져가겠다.

        out = self.fc1(hn)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.fc3(out)

        return out


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
