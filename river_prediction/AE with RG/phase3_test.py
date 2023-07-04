import torch

import cv2
import random
import network
import numpy as np


def make_data_dict(train_data):
    data_dict = dict()
    for i in range(200):
        data_dict[i] = train_data[:,1150*i:1150*(i+1)]
    return data_dict


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(888)

    # train-test data split
    x_data     = np.load('xs_data.npy')            # 690 x 36
    y_data     = np.load('ys_data.npy')[:,:230000] # 690 x 230000
    train_data = y_data[:-30,:]  # 660 x 230000
    test_data  = y_data[-30:,:]  # 30  x 230000

    data_dict = make_data_dict(train_data)

    batch_size    = 69
    IO_D          = 1150

    # define model
    ae = network.AE(IO_dim=IO_D, batch_size=batch_size).to(device)
    rg = network.Regressor().to(device)

    # load weight & eval
    ae.load_state_dict(torch.load("./model_weights_0518/model_state_dict0.pt"))
    rg.load_state_dict(torch.load("./phase2_model_weights_0518/model_state_dict0.pt"))
    ae.eval()
    rg.eval()

    test_x_data1 = x_data[600]
    test_y_data1 = data_dict[0][600]
    test_x_data2 = x_data[601]
    test_y_data2 = data_dict[0][601]

    prox_latent_x1 = rg(torch.FloatTensor(test_x_data1).to(device))
    prox_latent_x2 = rg(torch.FloatTensor(test_x_data2).to(device))
    org1 = ae.deco(prox_latent_x1).detach().cpu().numpy()
    org2 = ae.deco(prox_latent_x2).detach().cpu().numpy()

    org1 = np.reshape(org1, (50, 23))
    org2 = np.reshape(org2, (50, 23))
    test_y_data1 = np.reshape(test_y_data1, (50, 23))
    test_y_data2 = np.reshape(test_y_data2, (50, 23))
    cv2.namedWindow("pic1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pic2", cv2.WINDOW_NORMAL)
    cv2.imshow('pic1', test_y_data1)
    cv2.imshow('pic2', org1)

    cv2.namedWindow("pic3", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pic4", cv2.WINDOW_NORMAL)
    cv2.imshow('pic3', test_y_data2)
    cv2.imshow('pic4', org2)

    cv2.waitKey(0)
    cv2.destroyWindow()
    print(org.shape)





if __name__ == '__main__':
    main()
