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

# LSTM params
input_size  = 71  # Rainfall data_경안천
hidden_size = 200 # Number of features in hidden state
num_layers  = 4   # Number of stacked LSTM layers
num_classes = 1   # Next time step rainfall

def prediction(x_origin, x_modi, idx, device):

    lstm = network.LSTM(num_classes, input_size, hidden_size, num_layers, device).to(device)
    lstm.load_state_dict(torch.load("rainfall_result_반월/rainfall_rounded_model_state_dict.pt"))
    lstm.eval()

    x_test = torch.FloatTensor(x_modi).unsqueeze(0).unsqueeze(0).to(device)  # 36

    # Prediction
    x_lstm_output  = lstm(x_test)

    max_index = torch.argmax(x_lstm_output, axis = 1)[0]

    error = np.abs(x_origin[idx] - max_index)
    print("Error", error)

    return max_index, error

def main():
    goal = 'predict rainfall'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # train-test data_경안천 split
    x_data = np.load('data_반월/x_aug.npy')[-10000:, :]

    x_data     = np.round(x_data)

    x_origin = x_data[8356+106*5]
    x_origin_cut = x_origin[18:]

    # fix this

    x_ = [0.0 for i in range(len(x_origin) - 6)]
    x_modi = x_origin_cut[:6]
    x_modi = np.append(x_, x_modi, axis = 0)

    #print("Modified : ", x_modi)
    E = 0

    for idx in range(len(x_origin) - 25):
        pred, error = prediction(x_origin, x_modi, idx+25, device)
        x_modi = np.append(x_modi[1:], pred)
        E += error

    print("-------------------------------------------------------------------")
    print("Original x_data : ", x_origin)
    print("-------------------------------------------------------------------")
    print("Pred x_data : ", x_modi)
    print("-------------------------------------------------------------------")
    print("Total Error : ", E)

if __name__ == '__main__':
    main()


