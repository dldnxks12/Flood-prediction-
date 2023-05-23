import os
import sys

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms as T
from torchvision.utils import save_image

import network

def train():
    pass

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learning_rate = 0.0002
    img_size    = 28 * 28
    noise_dim   = 128
    batch_size  = 128
    dir_name    = 'fake_image'

    transform = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    # Test with MNIST dataset
    MNIST_dataset = torchvision.datasets.MNIST(root = './data/', train = True, transform = transform, download = True)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset = MNIST_dataset, batch_size=batch_size, shuffle = True)

    discriminator = network.Discriminator(input_dim = img_size).to(device)
    generator     = network.Generator(noise_dim = noise_dim, out_img_size = img_size).to(device)

    criterion   = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = learning_rate)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr = learning_rate)

    # train
    for epoch in range(200):
        for idx, (images, labels) in enumerate(data_loader):
            real_label = torch.full( (batch_size, 1), 1, dtype = torch.float32).to(device)
            fake_label = torch.full( (batch_size, 1), 0, dtype = torch.float32).to(device)

            # image flatten (batch_size, 1 x 28 x 28 )
            real_images = images.reshape(batch_size, -1).to(device)

            if real_images.shape[1] != 784:
                continue

            # ---- train generator ---- #
            """
            discriminator가 제대로 판단을 할경우 (fake image 임을 알아챌 경우)
            generator는 올바른 방향으로 데이터 생성을 못했다고 생각하게 된다.            
            discriminator가 제대로 판단 못했을 경우 generator는 옳타구나 fake image를 잘 만들었다라고 생각한다.
            """
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # make noise input
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(z)

            g_loss = criterion(discriminator(fake_images), real_label)

            g_loss.backward()
            g_optimizer.step()


#             # ---- train discriminator ---- #
#             for _ in range(5):
#               d_optimizer.zero_grad()
#               g_optimizer.zero_grad()

#               # 위의 과정을 한 번 학습을 거친 generator에서 fake image 한 번 더 뽑아내기
#               z = torch.randn(batch_size, noise_dim).to(device)
#               fake_images = generator(z)

#               fake_loss = criterion(discriminator(fake_images), fake_label)
#               real_loss = criterion(discriminator(real_images), real_label)
#               d_loss    = (fake_loss + real_loss) / 2

#               d_loss.backward()
#               d_optimizer.step()

            # ---- train discriminator ---- #
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            # 위의 과정을 한 번 학습을 거친 generator에서 fake image 한 번 더 뽑아내기
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(z)

            fake_loss = criterion(discriminator(fake_images), fake_label)
            real_loss = criterion(discriminator(real_images), real_label)
            d_loss    = (fake_loss + real_loss) / 2

            d_loss.backward()
            d_optimizer.step()

        print(f"epoch : {epoch} | d_loss : {d_loss.item()} | g_loss : {g_loss.item()}")

        if epoch % 10 == 0:
            samples = fake_images.reshape(batch_size, 1, 28, 28)
            save_image(samples, os.path.join(dir_name, 'GAN_fake_sample{}.png'.format(epoch + 1)))

if __name__ == "__main__":
    main()

