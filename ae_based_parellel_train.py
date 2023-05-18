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

def train(ae, data, device, p_idx):

    learning_rate = 0.001
    optimizer  = optim.Adam(ae.parameters(), lr = learning_rate)
    loss_fn    = nn.MSELoss()

    kfold = KFold(n_splits=5)
    kfold.get_n_splits(data)
    training_epochs = 200

    for i in range(training_epochs):
        for train_idx, _ in kfold.split(data):
            x_train = torch.FloatTensor(data[train_idx]).to(device)
            latent_vector = ae.enco(x_train)
            output = ae.deco(latent_vector)
            optimizer.zero_grad()

            cost = loss_fn(output, x_train)
            cost.backward()
            optimizer.step()

    torch.save(ae.state_dict(), "./model_weights_0518/model_state_dict" + str(p_idx) + ".pt")
    print(f"{p_idx} done & saved state_dict")
    # save latent spaces weigths
    # np.save(weigths)


    """
    이후 model.load_state_dict(~) 한 후 
    model.eval()을 꼭 호출해야한다. 
    """

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
    ae_data    = np.load('ys_data.npy')[:,:230000] # 690 x 230000
    train_data = ae_data[:-30,:]  # 660 x 230000
    test_data  = ae_data[-30:,:]  # 30  x 230000

    # train data split - to 100 network
    data_dict = dict()
    for i in range(100):
        data_dict[i] = train_data[:,2300*i:2300*(i+1)]

    batch_size    = 69
    IO_D          = 2300 # 231400 / 200
    ae = network.AE(IO_dim=IO_D, batch_size=batch_size).to(device)

    # multi-process
    processes   = []
    process_num = 10

    mp.set_start_method('spawn')
    print("MP start method :" , mp.get_start_method())

    for i in range(10): # 10개 thread x 10 train
        print(f" train {i} / 10 group ")
        for rank in range(process_num):
            p = mp.Process(target = train, args=(ae, data_dict[rank + (10*i)], device, rank + (10*i)))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == '__main__':
    main()
