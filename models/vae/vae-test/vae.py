import itertools, random
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data

import network
from sklearn.model_selection import KFold

def train(data, device):
    lr = 0.0005
    b1 = 0.9
    b2 = 0.99
    noise_dim = 2000
    input_dim = 73883

    vae_enco = network.Encoder(input_dim=input_dim, noise_dim=noise_dim).to(device)
    vae_deco = network.Decoder(input_dim=input_dim, noise_dim=noise_dim).to(device)

    optimizer = torch.optim.Adam(itertools.chain(vae_enco.parameters(), vae_deco.parameters()), lr = lr, betas = (b1, b2))

    kfold = KFold(n_splits=3)
    kfold.get_n_splits(data)
    training_epochs = 1000

    loss_fn = nn.MSELoss()
    # train
    for i in range(training_epochs):
        for train_idx, _ in kfold.split(data):

            x = torch.FloatTensor(data[train_idx]).to(device)

            z, mu, logvar = vae_enco(x)
            x_re          = vae_deco(z)

            recon_loss    = loss_fn(x_re, x)
            kl_div        = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

            cost = recon_loss + kl_div
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        print(f"Epoch : {i} | Cost : {cost}")
    torch.save(vae_enco.state_dict(), "vae_enco_model_state_dict.pt")
    torch.save(vae_deco.state_dict(), "vae_deco_model_state_dict.pt")
    print(f"Done & final cost : {cost} & saved state_dict")

def main():
    location = 'Seongnam'
    print(f"# ------------- VAE training in {location} ------------- #")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(888)

    y_data    = np.load('data/y_data.npy') # Auto Encoder train

    print("#--Data configs--#")
    train_data = y_data[:-5, :] # 35 x 73883
    valid_data = y_data[-5:, :] #  5 x 73883
    print(train_data.shape, valid_data.shape)

    train(train_data, device)

if __name__ == "__main__":
    main()
