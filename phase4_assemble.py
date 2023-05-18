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
    x_data  = np.load('xs_data.npy')            # 690 x 36
    x_test  = np.load('xs_data.npy')[-30:, :]   # 30 x 36

    y_data  = np.load('ys_data.npy')[:,:230000] # 690 x 230000
    y_test  = y_data[-30:,:]  # 30  x 230000

    test_data_dict  = make_data_dict(y_test) # 200개로 분할

    # model
    batch_size    = 69
    IO_D          = 1150

    ae = network.AE(IO_dim=IO_D, batch_size=batch_size).to(device)
    rg = network.Regressor().to(device)

    total_pred      = []
    total_pred_dict = dict()

    for idx in range(len(x_data)):
        if idx % 10 == 0:
            print(f"c idx : {idx}")
        x = x_data[idx]
        for i in range(len(test_data_dict)):
            # load weight & eval
            ae.load_state_dict(torch.load("./phase1/model_state_dict" +str(i) + ".pt"))  # part i에 대한 ae network
            rg.load_state_dict(torch.load("./phase2/model_state_dict" +str(i) + ".pt"))  # part i에 대한 rg network
            ae.eval()
            rg.eval()

            latent_x = rg(torch.FloatTensor(x).to(device))
            pred_y   = ae.deco(latent_x).detach().cpu().numpy()
            total_pred.extend(pred_y)
        total_pred_dict[idx] = total_pred
        total_pred = []

    test_pic = y_test[0]
    pred_pic = total_pred_dict[0]

    test_pic = np.reshape(test_pic, (500, 460))
    pred_pic = np.reshape(pred_pic, (500, 460))

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pred", cv2.WINDOW_NORMAL)
    cv2.imshow('label', test_pic)
    cv2.imshow('pred', pred_pic)

    cv2.waitKey(0)
    cv2.destroyWindow()

if __name__ == '__main__':
    main()
