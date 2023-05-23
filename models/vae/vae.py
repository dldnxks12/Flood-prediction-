# https://velog.io/@hong_journey/VAEVariational-AutoEncoder-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0
# https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

import os
import sys
import itertools
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms as T
from torchvision.utils import save_image
import torch.nn.functional as F

import network


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    lr = 0.0003
    b1 = 0.9
    b2 = 0.99
    img_size    = 28 * 28
    noise_dim   = 100
    batch_size  = 128
    dir_name = 'fake_image'

    # Test with MNIST dataset
    MNIST_dataset = torchvision.datasets.MNIST(root = './data/', train = True, transform = transform, download = True)
    data_loader = torch.utils.data.DataLoader(dataset = MNIST_dataset, batch_size=batch_size, shuffle = True)

    encoder = network.Encoder(input_dim=img_size, noise_dim=noise_dim).to(device)
    decoder = network.Decoder(input_dim=img_size, noise_dim=noise_dim).to(device)

    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr = lr, betas = (b1, b2))

    # train
    for epoch in range(200):
        train_loss = 0
        for idx, (x, _) in enumerate(data_loader):

            #x = x.view(-1, img_size).to(device)
            x = x.reshape(batch_size, -1).to(device)

            if x.shape[1] != 784:
                continue

            z, mu, logvar = encoder(x)
            x_re          = decoder(z)

            recon_loss = F.binary_cross_entropy(x_re, x, reduction='sum')
            kl_div     = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

            loss = recon_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch : {epoch} | loss : {loss.item()}")

        if epoch % 10 == 0:
            samples = x_re.reshape(batch_size, 1, 28, 28)
            save_image(samples, os.path.join(dir_name, 'VAE_fake_sample{}.png'.format(epoch + 1)))

if __name__ == "__main__":
    main()

