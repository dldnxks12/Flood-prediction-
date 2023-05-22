# 0518 ae_regressor test (not bad but not so good, too)

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import cv2
import random
import network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import multiprocessing as mp

def make_data_dict(train_data):
    data_dict = dict()
    for i in range(200):
        data_dict[i] = train_data[:,1150*i:1150*(i+1)]
    data_dict[200] = train_data[:, 230000:]
    return data_dict

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    # Seed
    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(888)

    # train-test data split
    x_test  = np.load('xs_data.npy')[-30:, :] # 30 x 36
    y_test  = np.load('ys_data.npy')[-30:, :] # 30 x 231432

    test_data_dict  = make_data_dict(y_test) # 200개로 분할

    # model
    batch_size    = 69
    ae = network.AE(IO_dim = 1150, batch_size=batch_size).to(device)
    rg = network.Regressor().to(device)

    total_pred      = []
    total_pred_dict = dict()

    for idx in range(len(x_test)):
        if idx % 10 == 0:
            print(f"c idx : {idx}")
        x = np.expand_dims(x_test[idx], axis=0)
        for i in range(len(test_data_dict)):
            if i == 200:
                ae = network.AE(IO_dim=1432).to(device)
            else:
                ae = network.AE(IO_dim=1150).to(device)
            ae.load_state_dict(torch.load("./phase1_0521/model_state_dict" +str(i) + ".pt"))  # part i에 대한 ae network
            rg.load_state_dict(torch.load("./phase2_0521/model_state_dict" +str(i) + ".pt"))  # part i에 대한 rg network
            ae.eval()
            rg.eval()

            latent_x = rg(torch.FloatTensor(x).to(device))
            pred_y   = ae.deco(latent_x).squeeze(0).detach().cpu().numpy()
            total_pred.extend(pred_y)
        total_pred_dict[idx] = total_pred
        total_pred = []

    np.save('total_pred_dict_0521.npy', total_pred_dict)

if __name__ == '__main__':
    main()
