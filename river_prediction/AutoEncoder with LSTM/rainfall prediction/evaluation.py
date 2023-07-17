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
from sklearn.preprocessing import OneHotEncoder


import network


# Scaler
MinMax = MinMaxScaler(feature_range = (-1,1))
OH     = OneHotEncoder(handle_unknown='ignore')

# LSTM params
input_size  = 71  # Rainfall data_경안천
hidden_size = 100 # Number of features in hidden state
num_layers  = 4   # Number of stacked LSTM layers
num_classes = 1   # Next time step rainfall

def eval(x_data, y_data, device):

    """
    x_data : 강우 데이터
    y_data : 다음 시간 강우량
    """

    lstm = network.LSTM(num_classes, input_size, hidden_size, num_layers, device).to(device)
    lstm.load_state_dict(torch.load("rainfall_result_반월/rainfall_rounded_model_state_dict1.pt"))
    lstm.eval()

    x_data = torch.FloatTensor(x_data).unsqueeze(1).to(device)  # 36
    y_data = torch.FloatTensor(y_data).type(torch.int64).to(device)

    # Predict
    x_lstm_output  = lstm(x_data)

    # One-Hot
    max_index = torch.argmax(x_lstm_output, axis = 1)

    print(max_index[:50])
    print("")
    print(y_data[:50])

    print("Error", sum(np.abs(max_index - y_data)))

    sys.exit()


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
    x_data = np.load('data_반월/x_aug.npy')[-10000:, :]
    y_data = np.load('data_반월/y_aug.npy')[-10000:]

    x_data     = np.round(x_data)
    y_data     = np.round(y_data)

    eval(x_data, y_data, device)

if __name__ == '__main__':
    main()


