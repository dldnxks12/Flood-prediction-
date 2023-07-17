import os
import sys
import random
import numpy as np
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder

import network

"""
StandardScaler : 각 feature의 특징을 평균 0, 분산 1로 변경  
MinMaxScaler   : Min/Max 값이 0, 1이 되도록 변경 
"""

# LSTM params
input_size  = 71  # Rainfall data_경안천
hidden_size = 100 # Number of features in hidden state
num_layers  = 4   # Number of stacked LSTM layers
num_classes = 1   # Next time step rainfall

def train(dataset, device):

    lstm = network.LSTM(num_classes, input_size, hidden_size, num_layers, device).to(device)

    learning_rate = 0.0009
    optimizer  = optim.Adam(lstm.parameters(), lr = learning_rate)

    dataloader = DataLoader(dataset, batch_size = 256, shuffle = True )
    EPOCHS = 2000

    print("Start Training..")
    for e in range(EPOCHS):
        LOSS = 0
        for _, samples in enumerate(dataloader):
            x_data, y_data = samples
            x_train = x_data.unsqueeze(1).to(device)
            y_train = (y_data).type(torch.int64).to(device)

            # Prediction
            x_lstm_output = lstm(x_train)
            optimizer.zero_grad()
            cost  = F.cross_entropy(x_lstm_output, y_train)
            cost.backward()
            optimizer.step()
            LOSS += cost

        print(f"Epoch : {e} / {EPOCHS} | LOSS : {LOSS}")
    torch.save(lstm.state_dict(), "rainfall_result_반월/rainfall_rounded_model_state_dict2.pt")

def main():
    goal = 'predict rainfall'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = 'cpu'
    print("--------------------------------")
    print(f"Goal : {goal} with {device}")
    print("--------------------------------")

    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # train-test data_경안천 split
    x_data = np.load('data_반월/x_aug.npy')
    print(x_data.shape)

    x_data     = np.load('data_반월/x_aug.npy')[:-10000, :]
    y_data     = np.load('data_반월/y_aug.npy')[:-10000]

    x_data = np.round(x_data)  # max : 90 - Catogory 90
    y_data = np.round(y_data)

    print("DATA SHAPE : ")
    print(x_data.shape)
    print(y_data.shape)

    print("Define DataLoader")
    x_data  = torch.FloatTensor(x_data)
    y_data  = torch.FloatTensor(y_data)
    dataset = TensorDataset(x_data, y_data)

    train(dataset, device)

if __name__ == '__main__':
    main()


