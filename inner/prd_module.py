"""
파트 별로 예측하는 모델
"""
import sys
import torch
import time
import random
import network
import numpy as np
def prediction(x_data, pred_once):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    # Seed
    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # models
    batch_size    = 5
    ae = network.AE(IO_dim = 367, batch_size=batch_size).to(device)
    rg = network.Regressor().to(device)

    test_data_dict = 200 # 200 networks

    # Prediction 1 data
    if pred_once:
        total_pred = []
        x = np.expand_dims(x_data, axis=0)
        for i in range(test_data_dict+1):
            if i == 200:
                ae = network.AE(IO_dim=483).to(device)
            else:
                ae = network.AE(IO_dim=367).to(device)
            ae.load_state_dict(torch.load("./phase1/model_state_dict" +str(i) + ".pt"))  # part i에 대한 ae network
            rg.load_state_dict(torch.load("./phase2/model_state_dict" +str(i) + ".pt"))  # part i에 대한 rg network
            ae.eval()
            rg.eval()

            latent_x = rg(torch.FloatTensor(x).to(device))
            pred_y   = ae.deco(latent_x).squeeze(0).detach().cpu().numpy()
            total_pred.extend(pred_y)

        return total_pred

    # Prediction all data
    else:
        total_pred = []
        total_pred_dict = dict()
        for idx in range(len(x_data)):
            x = np.expand_dims(x_data[idx], axis=0)
            for i in range(test_data_dict+1):
                if i == 200:
                    ae = network.AE(IO_dim=483).to(device)
                else:
                    ae = network.AE(IO_dim=367).to(device)

                ae.load_state_dict(torch.load("./phase1/model_state_dict" + str(i) + ".pt"))  # part i에 대한 ae network
                rg.load_state_dict(torch.load("./phase2/model_state_dict" + str(i) + ".pt"))  # part i에 대한 rg network
                ae.eval()
                rg.eval()

                latent_x = rg(torch.FloatTensor(x).to(device))
                pred_y = ae.deco(latent_x).squeeze(0).detach().cpu().numpy()
                total_pred.extend(pred_y)

            total_pred_dict[idx] = total_pred
            total_pred = []

        return total_pred_dict
