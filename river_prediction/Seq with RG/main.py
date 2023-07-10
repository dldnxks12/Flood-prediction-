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

import network

"""
StandardScaler : 각 feature의 특징을 평균 0, 분산 1로 변경  
MinMaxScaler   : Min/Max 값이 0, 1이 되도록 변경 
"""

# Scaler
MinMax_x = MinMaxScaler(feature_range = (-1,1))
MinMax_y = MinMaxScaler(feature_range = (-1,1))

# LSTM params
input_size  = 35  # Rainfall data
hidden_size = 200 # Number of features in hidden state
num_layers  = 6   # Number of stacked LSTM layers
num_classes = 1   # Next time step rainfall

def train(x_data, y_data, device):

    """
    x_data : 강우 데이터
    y_data : 다음 시간 강우량
    """

    # TODO : Define LSTM instead of Regressor
    lstm = network.LSTM(num_classes, input_size, hidden_size, num_layers, device).to(device)

    # TODO : tuning - learning reate
    learning_rate = 0.0005
    optimizer     = optim.Adam(lstm.parameters(), lr = learning_rate)
    loss_fn       = nn.MSELoss()

    EPOCHS = 30
    training_epochs = 1000
    for e in range(EPOCHS):
        LOSS = 0
        for i in range(training_epochs):
            # TODO : Scaler 사용 효과 Check
            x_train = MinMax_x.fit_transform(x_data)
            x_train = torch.FloatTensor(x_train).to(device)  # 36
            x_train = x_train.unsqueeze(1)

            # TODO : Scaler 사용 효과 Check
            y_train = torch.FloatTensor(y_data).unsqueeze(1).to(device)
            y_train = torch.FloatTensor(MinMax_y.fit_transform(y_train)).to(device)

            # Prediction
            x_lstm_output  = lstm(x_train)

            optimizer.zero_grad()
            cost  = loss_fn(x_lstm_output, y_train)
            cost.backward()
            optimizer.step()
            LOSS += cost
        print(f"Epoch : {e} | LOSS : {LOSS} : X : {x_lstm_output[0]} | Y : {y_train[0]}")

    # TODO : 추후 prediction 할 때, fit_transform -> predict -> inverse_transform 수행.
    torch.save(lstm.state_dict(), "./rainfall_result/rainfall_model_state_dict6.pt")

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
    x_data     = np.load('xs_data.npy')[:-15, :-1]  # 675 x 35
    y_data     = np.load('xs_data.npy')[:-15, -1]   # 675 x 1

    train(x_data, y_data, device)

if __name__ == '__main__':
    main()


