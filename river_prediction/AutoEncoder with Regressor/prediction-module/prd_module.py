import sys
import torch
import time
import random
import network
import numpy as np


def prediction(x_data, y_data, pred_once):
    if y_data.shape[1] != 231432:
        raise Exception("Invalid data shape : check y data shape...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    # Seed
    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # model
    batch_size    = 69
    ae = network.AE(IO_dim = 1150, batch_size=batch_size).to(device)
    rg = network.Regressor().to(device)

    test_data_dict = 200 # 200 networks

    # Prediction 1 data
    if pred_once:
        total_pred = []
        x = np.expand_dims(x_data, axis=0)
        start = time.time()
        for i in range(test_data_dict + 1):
            if i == 200:
                ae = network.AE(IO_dim=1432).to(device)
            else:
                ae = network.AE(IO_dim=1150).to(device)
            ae.load_state_dict(torch.load("./phase1/model_state_dict" +str(i) + ".pt"))  # part i에 대한 ae network
            rg.load_state_dict(torch.load("./phase2/model_state_dict" +str(i) + ".pt"))  # part i에 대한 rg network
            ae.eval()
            rg.eval()

            latent_x = rg(torch.FloatTensor(x).to(device))
            pred_y   = ae.deco(latent_x).squeeze(0).detach().cpu().numpy()
            total_pred.extend(pred_y)

        t = time.time() - start
        return total_pred, t

    else:
        total_pred = []
        total_pred_dict = dict()
        for idx in range(len(x_data)):
            x = np.expand_dims(x_data[idx], axis=0)
            start = time.time()
            for i in range(len(test_data_dict)):
                if i == 200:
                    ae = network.AE(IO_dim=1432).to(device)
                else:
                    ae = network.AE(IO_dim=1150).to(device)
                ae.load_state_dict(torch.load("./phase1/model_state_dict" + str(i) + ".pt"))  # part i에 대한 ae network
                rg.load_state_dict(torch.load("./phase2/model_state_dict" + str(i) + ".pt"))  # part i에 대한 rg network
                ae.eval()
                rg.eval()

                latent_x = rg(torch.FloatTensor(x).to(device))
                pred_y = ae.deco(latent_x).squeeze(0).detach().cpu().numpy()
                total_pred.extend(pred_y)

            total_pred_dict[idx] = total_pred
            total_pred = []

        t = time.time() - start
        return total_pred_dict, t
