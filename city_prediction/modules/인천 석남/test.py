import sys
import torch
import cv2
import random
import network
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def make_data_dict(train_data):
    data_dict = dict()
    for i in range(200):
        data_dict[i] = train_data[: , 367*i : 367*(i+1)]
    data_dict[200] = train_data[:, 73400:]
    return data_dict

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(888)

    # train-test data split
    x_data       = np.load('./data/x_data.npy')     # 40 x 36
    x_train_data = x_data[:-5, :]                   # 35 x 36
    x_test_data  = x_data[-5:, :]                   #  5 x 36

    y_data       = np.load('./data/y_data.npy')     # 40 x 73883
    y_train_data = y_data[:-5,:]                    # 35 x 73883
    y_test_data  = y_data[-5:,:]                    #  5 x 73883
    print(y_test_data.shape)

    train_data_dict = make_data_dict(y_train_data)
    test_data_dict  = make_data_dict(y_test_data)

    batch_size    = 5
    IO_D          = 367

    # define model
    ae = network.AE(IO_dim=IO_D, batch_size=batch_size).to(device)
    rg = network.Regressor().to(device)

    # load weight & eval
    ae.load_state_dict(torch.load("phase1/model_state_dict3.pt")) # part 0에 대한 ae network
    rg.load_state_dict(torch.load("phase2/model_state_dict3.pt")) # part 0에 대한 rg network
    ae.eval()
    rg.eval()

    test_x  = x_test_data[0]
    test_y  = test_data_dict[3][0]

    latent_x = rg(torch.FloatTensor(test_x).unsqueeze(0).to(device))
    pred_y   = ae.deco(latent_x).detach().cpu().numpy()

    error = np.sum(abs(test_y - pred_y))
    print(error)

    # test_y = np.reshape(test_y, (50, 23))
    # pred_y = np.reshape(pred_y, (50, 23))
    #
    # cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("pred", cv2.WINDOW_NORMAL)
    # cv2.imshow('label', test_y)
    # cv2.imshow('pred' , pred_y)
    #
    # cv2.waitKey(0)
    # cv2.destroyWindow()


if __name__ == '__main__':
    main()