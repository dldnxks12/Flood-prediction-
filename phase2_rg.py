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
def train(x_data, y_data, device, p_idx):

    # Load ae model
    batch_size    = 69
    IO_D          = 1150
    ae = network.AE(IO_dim=IO_D, batch_size=batch_size).to(device)
    ae.load_state_dict(torch.load("./phase1/model_state_dict"+str(p_idx) +".pt"))
    ae.eval()

    rg = network.Regressor().to(device)

    learning_rate = 0.0001
    optimizer  = optim.Adam(rg.parameters(), lr = learning_rate)
    loss_fn    = nn.MSELoss()

    kfold = KFold(n_splits=3)
    kfold.get_n_splits(y_data)
    training_epochs = 2000

    for i in range(training_epochs):
        for train_idx, _ in kfold.split(y_data):
            x_train = torch.FloatTensor(x_data[train_idx]).to(device)  # 36
            y_train = torch.FloatTensor(y_data[train_idx]).to(device)  # 1150

            with torch.no_grad():
                y_latent_vector = ae.enco(y_train)

            x_reg_output  = rg(x_train)
            optimizer.zero_grad()
            cost  = loss_fn(x_reg_output, y_latent_vector)
            cost.backward()
            optimizer.step()

    torch.save(rg.state_dict(), "./phase2/model_state_dict" + str(p_idx) + ".pt")
    print(f"{p_idx} regressor done & cost : {cost} & saved state_dict")

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
    x_data     = np.load('xs_data.npy')[:-30,:]    # 660 x 36
    y_data     = np.load('ys_data.npy')[:,:230000] # 690 x 230000
    train_data = y_data[:-30,:]  # 660 x 230000

    data_dict = make_data_dict(train_data)

    # multi-process
    processes   = []
    process_num = 5

    mp.set_start_method('spawn')
    print("MP start method :" , mp.get_start_method())

    for i in range(40): # 10개 thread x 10 train
        print(f" train {i+1} / 40 group ")
        for rank in range(process_num):
            p = mp.Process(target = train, args=(x_data, data_dict[rank + (process_num*i)], device, rank + (process_num*i)))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == '__main__':
    main()
