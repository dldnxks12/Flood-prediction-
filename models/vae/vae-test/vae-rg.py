import os
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
import itertools, random

def train(x_data, y_data, device):
    lr = 0.001
    b1 = 0.9
    b2 = 0.99
    noise_dim = 2000
    input_dim = 73883

    vae_enco = network.Encoder(input_dim=input_dim, noise_dim=noise_dim).to(device)
    vae_enco.load_state_dict(torch.load("vae_enco_model_state_dict.pt"))
    vae_enco.eval()

    rg = network.Regressor().to(device)

    learning_rate = 0.0005
    optimizer  = optim.Adam(rg.parameters(), lr = learning_rate)
    loss_fn    = nn.MSELoss()

    kfold = KFold(n_splits=3)
    kfold.get_n_splits(y_data)
    training_epochs = 1000

    for i in range(training_epochs):
        for train_idx, _ in kfold.split(y_data):
            x_train = torch.FloatTensor(x_data[train_idx]).to(device)  # 36
            y_train = torch.FloatTensor(y_data[train_idx]).to(device)  # 1150

            with torch.no_grad():
                z, _, _ = vae_enco(y_train)

            x_reg_output  = rg(x_train)

            optimizer.zero_grad()
            cost  = loss_fn(x_reg_output, z)
            cost.backward()
            optimizer.step()
        print(f"Cost : {cost} ")
    torch.save(rg.state_dict(), "../test-visualize/vae_rg_model_state_dict.pt")
    print(f"Regressor done & cost : {cost} & saved state_dict")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(888)

    # train-test data split
    x_data = np.load('../test-VAE/data/x_data.npy') # data for regressor
    y_data = np.load('../test-VAE/data/y_data.npy') # data for latent vector

    print("#--Data configs--#")
    train_x_data = x_data[:-5, :] # 35 x 36
    train_y_data = y_data[:-5, :] # 35 x 73883

    train(x_data, y_data, device)

if __name__ == '__main__':
    main()
