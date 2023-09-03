"""

    Conditional Generator

"""

import torch
import torch.nn as nn
import torch.utils.data

import random
import numpy as np
import network
from sklearn.model_selection import KFold

def train(rainfall_data, y_data, device):
    learning_rate = 0.0001
    img_size      = 236424 # 반월 # 73883 석남
    noise_dim     = 2000

    discriminator = network.Conditional_Discriminator(input_dim=img_size, condition_dim=72).to(device)
    generator     = network.Conditional_Generator(noise_dim=noise_dim, condition_dim=72, out_img_size=img_size).to(device)

    criterion   = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

    kfold = KFold(n_splits=3)
    kfold.get_n_splits(y_data)
    training_epochs = 500

    for i in range(training_epochs):
        for train_idx, _ in kfold.split(y_data):
            batch_size = len(train_idx)
            x_train = torch.FloatTensor(y_data[train_idx]).to(device)
            x_rainfall = torch.FloatTensor(rainfall_data[train_idx]).to(device)
            real_label = torch.full( (batch_size, 1), 1, dtype = torch.float32).to(device)
            fake_label = torch.full( (batch_size, 1), 0, dtype = torch.float32).to(device)

            # image flatten (batch_size, 1 x 28 x 28 )
            real_images             = x_train.reshape(batch_size, -1).to(device)
            conditioned_real_iamges = torch.cat((real_images, x_rainfall), 1)

            if conditioned_real_iamges.shape[1] != 236424+72: #73919:
                print("SIZE ERROR")
                continue

            for _ in range(5):
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()

                # make noise input
                z              = torch.randn(batch_size, noise_dim).to(device)
                conditioned_z  = torch.cat((z, x_rainfall), 1)
                fake_images    = generator(conditioned_z)
                conditioned_fake_images = torch.cat((fake_images, x_rainfall), 1)

                g_loss = criterion(discriminator(conditioned_fake_images), real_label)

                g_loss.backward()
                g_optimizer.step()

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            # 위의 과정을 한 번 학습을 거친 generator에서 fake image 한 번 더 뽑아내기
            z = torch.randn(batch_size, noise_dim).to(device)
            conditioned_z = torch.cat((z, x_rainfall), 1)
            fake_images = generator(conditioned_z)
            conditioned_fake_images = torch.cat((fake_images, x_rainfall), 1)

            fake_loss = criterion(discriminator(conditioned_fake_images), fake_label)
            real_loss = criterion(discriminator(conditioned_real_iamges), real_label)
            d_loss    = (fake_loss + real_loss) / 2

            d_loss.backward()
            d_optimizer.step()

        print(f"Epoch : {i} | d_loss : {d_loss.item()} | g_loss : {g_loss.item()}")
    torch.save(discriminator.state_dict(), "conditioned_discriminator_1000_model_state_dict.pt")
    torch.save(generator.state_dict(), "conditioned_generator_1000_model_state_dict.pt")
    print(f"Done & final cost : {d_loss.item() + g_loss.item()} & saved state_dict")


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device : {device}")

    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(888)

    # train-test data split
    x_data = np.load('x_data.npy')
    y_data = np.load('y_data1.npy')

    print("#--Data configs--#")
    rainfall_data = x_data[:1000, :]
    train_data    = y_data[:1000, :]

    print(rainfall_data.shape, train_data.shape)

    train(rainfall_data, train_data, device)


if __name__ == "__main__":
    main()
