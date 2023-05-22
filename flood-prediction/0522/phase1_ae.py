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

def train(data, device, p_idx, IO_D = 1150):

    batch_size    = 69
    ae = network.AE(IO_dim=IO_D, batch_size=batch_size).to(device)

    learning_rate = 0.005
    optimizer  = optim.Adam(ae.parameters(), lr = learning_rate)
    loss_fn    = nn.MSELoss()

    kfold = KFold(n_splits=5)
    kfold.get_n_splits(data)
    training_epochs = 2000

    for i in range(training_epochs):
        for train_idx, _ in kfold.split(data):
            x_train = torch.FloatTensor(data[train_idx]).to(device)
            latent_vector = ae.enco(x_train)
            output = ae.deco(latent_vector)
            optimizer.zero_grad()

            cost = loss_fn(output, x_train)
            cost.backward()
            optimizer.step()

    torch.save(ae.state_dict(), "./phase1_0522/model_state_dict" + str(p_idx) + ".pt")
    print(f"{p_idx} done & saved state_dict")

def make_data_dict(train_data):
    data_dict = dict()
    for i in range(200):
        data_dict[i] = train_data[:,1150*i:1150*(i+1)]
    data_dict[200] = train_data[:, 230000:]
    return data_dict

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")
    # device = 'cpu'
    # print(f"Device : {device}")
    # Seed
    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(888)

    # train-test data split
    data    = np.load('ys_data.npy')

    # ae_data = data[:, :230000]  # 690 x 230000
    # ae_rest = data[:,230000:-1]
    # print(ae_data.shape)
    # print(ae_rest.shape)
    train_data = data[:-30,:]  # 660 x 231431
    test_data  = data[-30:,:]  # 30  x 231431

    # train data split - to 101 network
    data_dict = make_data_dict(train_data)

    # multi-process
    processes   = []
    process_num = 5

    mp.set_start_method('spawn')
    print("MP start method :" , mp.get_start_method())

    # train network 200 + 1개
    for i in range(40):
        print(f" train {i+1} / 40 group ")
        for rank in range(process_num):
            p = mp.Process(target = train, args=(data_dict[rank + (process_num*i)], device, rank + (process_num*i)))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    train(data_dict[200], device, 200, IO_D=1432)

if __name__ == '__main__':
    main()
