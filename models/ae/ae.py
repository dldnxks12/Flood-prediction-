"""

Auto-Encoder for making reasonable dimension of latent space.

"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import network

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    # Seed
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(888)

    learning_rate = 0.001
    batch_size    = 69
    IO_D          = 2300

    ae = network.AE(IO_dim=IO_D, batch_size=batch_size).to(device)
    optimizer  = optim.Adam(ae.parameters(), lr = learning_rate)
    loss_fn    = nn.MSELoss()

    training_epochs = 10

    ae_data = np.load('../../ys_data.npy')[:, :2300] # 690 x 231432 - test : 23000

    kfold   = KFold(n_splits = 5)
    kfold.get_n_splits(ae_data)
    print(kfold)

    for i in range(training_epochs):
        for train_idx, _ in kfold.split(ae_data):
            x_train = torch.FloatTensor(ae_data[train_idx]).to(device)
            latent_vector = ae.enco(x_train)
            output        = ae.deco(latent_vector)
            optimizer.zero_grad()

            cost   = loss_fn(output, x_train)
            cost.backward()
            optimizer.step()
            #print(cost)

    print(ae.state_dict())

    # # Test
    # ae_data_test = ae_data[10]
    # with torch.no_grad():
    #     lv = ae.enco(torch.FloatTensor(ae_data_test).to(device))
    #     ae_data_output = ae.deco(lv)
    #     ae_data_output = ae_data_output.detach().cpu().numpy()
    #
    # ae_data_test   = np.reshape(ae_data_test, (23, 100))
    # ae_data_output = np.reshape(ae_data_output, (23, 100))
    # cv2.namedWindow("pic1", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("pic2", cv2.WINDOW_NORMAL)
    # cv2.imshow('pic1', ae_data_test)
    # cv2.imshow('pic2', ae_data_output)
    # cv2.waitKey(0)
    # cv2.destroyWindow()


if __name__ == '__main__':
    main()
