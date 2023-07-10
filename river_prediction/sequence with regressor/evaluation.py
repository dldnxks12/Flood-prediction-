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


# Scaler
MinMax = MinMaxScaler(feature_range = (-1,1))

# LSTM params
input_size  = 35  # Rainfall data
hidden_size = 50 # Number of features in hidden state
num_layers  = 4   # Number of stacked LSTM layers
num_classes = 1   # Next time step rainfall

def eval(x_data, y_data, device):

    """
    x_data : 강우 데이터
    y_data : 다음 시간 강우량
    """

    lstm = network.LSTM(num_classes, input_size, hidden_size, num_layers, device).to(device)
    lstm.load_state_dict(torch.load("rainfall_result/rainfall_model_state_dict1.pt"))
    lstm.eval()

    x_test = x_data
    x_test = torch.FloatTensor(x_test).to(device)  # 36
    x_test = x_test.unsqueeze(1)

    y_origin = y_data
    y_test   = y_origin
    y_test   = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    # Prediction
    x_lstm_output  = lstm(x_test).cpu().detach().numpy()
    x_lstm_output = np.squeeze(x_lstm_output, axis = 1)

    error = np.sum(np.abs(x_lstm_output - y_origin))
    print(error)

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
    x_data     = np.load('data/x_aug.npy')[-46:, :]
    y_data     = np.load('data/y_aug.npy')[-46:]

    eval(x_data, y_data, device)

if __name__ == '__main__':
    main()


