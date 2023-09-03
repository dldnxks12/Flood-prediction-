import sys
import torch
import time
import random
import network
import numpy as np


def prediction(x_data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    # Seed
    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # model
    noise_dim = 2000
    input_dim = 73883

    vae_enco = network.Encoder(input_dim=input_dim, noise_dim=noise_dim).to(device)
    vae_deco = network.Decoder(input_dim=input_dim, noise_dim=noise_dim).to(device)

    rg = network.Regressor().to(device)

    total_pred_dict = dict()
    for idx in range(len(x_data)):
        x = np.expand_dims(x_data[idx], axis=0)
        start = time.time()

        vae_deco.load_state_dict(torch.load("vae_deco_model_state_dict.pt"))
        rg.load_state_dict(torch.load("../test-visualize/vae_rg_model_state_dict.pt"))
        vae_deco.eval()
        rg.eval()

        latent_x = rg(torch.FloatTensor(x).to(device))
        pred_y   = vae_deco(latent_x).squeeze(0).detach().cpu().numpy()

        total_pred_dict[idx] = pred_y

        t = time.time() - start
        print("One sample inference time : ", t)

    return total_pred_dict


