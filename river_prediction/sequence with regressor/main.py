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

import network

"""
StandardScaler : 각 feature의 특징을 평균 0, 분산 1로 변경  
MinMaxScaler   : Min/Max 값이 0, 1이 되도록 변경 
"""

# LSTM params
input_size  = 35  # Rainfall data
hidden_size = 100 # Number of features in hidden state
num_layers  = 2   # Number of stacked LSTM layers
num_classes = 1   # Next time step rainfall

def train(dataset, device):

    """
    x_data : 강우 데이터
    y_data : 다음 시간 강우량
    """

    lstm = network.LSTM(num_classes, input_size, hidden_size, num_layers, device).to(device)

    learning_rate = 0.001
    optimizer     = optim.Adam(lstm.parameters(), lr = learning_rate)
    loss_fn       = nn.MSELoss()

    dataloader = DataLoader(dataset, batch_size = 256, shuffle=True )

    EPOCHS = 1000

    for e in range(EPOCHS):
        LOSS = 0

        for batch_idx, samples in enumerate(dataloader):
            x_data, y_data = samples
            x_train = x_data.to(device)
            y_train = (y_data).unsqueeze(1).to(device)
            x_train = x_train.unsqueeze(1)

            # Prediction
            x_lstm_output  = lstm(x_train)

            optimizer.zero_grad()
            cost  = loss_fn(x_lstm_output, y_train)
            cost.backward()
            optimizer.step()
            LOSS += cost

        print(f"Epoch : {e} / {EPOCHS} | LOSS : {LOSS} : X : {x_lstm_output[0]} | Y : {y_train[0]}")

    torch.save(lstm.state_dict(), "rainfall_result/rainfall_model_state_dict2.pt")

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

    # train-test data split
    x_data     = np.load('data/x_aug.npy')[:-46, :]
    y_data     = np.load('data/y_aug.npy')[:-46]

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


