# 0518 ae_regressor test (not bad but not so good, too)

import torch

import random
import network
import numpy as np


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
    x_test  = np.load('xs_data.npy')[-10:, :] # 10 x 36
    y_test  = np.load('ys_data.npy')[-10:, :] # 10 x 231432

    test_data_dict  = make_data_dict(y_test) # 200개로 분할

    ae = network.AE(IO_dim = 1150).to(device)
    rg = network.Regressor().to(device)

    # make interpolation data x 50
    inter_range = 50
    # calc diff
    diff_ = x_test[1] - x_test[0]

    diff = diff_ / inter_range

    x_list = []
    for i in range(inter_range):
        x_list.append(x_test[0] + (diff * (i+1)))

    x_list = x_list[:-1]

    total_pred      = []
    total_pred_dict = dict()
    for idx, x in enumerate(x_list):
        print(f"Current idx : {idx}")
        x = np.expand_dims(x, axis = 0)
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

    np.save('interpolation_0521.npy', total_pred_dict)

if __name__ == '__main__':
    main()
