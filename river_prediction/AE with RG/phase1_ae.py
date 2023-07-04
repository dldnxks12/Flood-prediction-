import torch
import torch.nn as nn
import torch.optim as optim

import random
import network
import numpy as np
from sklearn.model_selection import KFold

import multiprocessing as mp

def train(data, device, p_idx):

    batch_size    = 69
    IO_D          = 1150 # 230000 / 200
    ae = network.AE(IO_dim=IO_D, batch_size=batch_size).to(device)

    learning_rate = 0.005
    optimizer  = optim.Adam(ae.parameters(), lr = learning_rate)
    loss_fn    = nn.MSELoss()

    kfold = KFold(n_splits=3)
    kfold.get_n_splits(data)
    training_epochs = 1000

    for i in range(training_epochs):
        for train_idx, _ in kfold.split(data):
            x_train = torch.FloatTensor(data[train_idx]).to(device)
            latent_vector = ae.enco(x_train)
            output = ae.deco(latent_vector)
            optimizer.zero_grad()

            cost = loss_fn(output, x_train)
            cost.backward()
            optimizer.step()

    torch.save(ae.state_dict(), "./phase1/model_state_dict" + str(p_idx) + ".pt")
    print(f"{p_idx} done & saved state_dict")

def make_data_dict(train_data):
    data_dict = dict()
    for i in range(200):
        data_dict[i] = train_data[:,1150*i:1150*(i+1)]
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
    ae_data    = np.load('ys_data.npy')[:,:230000] # 690 x 230000
    train_data = ae_data[:-30,:]  # 660 x 230000
    test_data  = ae_data[-30:,:]  # 30  x 230000

    # train data split - to 100 network
    data_dict = make_data_dict(train_data)

    # multi-process
    processes   = []
    process_num = 5

    mp.set_start_method('spawn')
    print("MP start method :" , mp.get_start_method())

    for i in range(40): # 10개 thread x 10 train
        print(f" train {i+1} / 40 group ")
        for rank in range(process_num):
            p = mp.Process(target = train, args=(data_dict[rank + (process_num*i)], device, rank + (process_num*i)))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == '__main__':
    main()
