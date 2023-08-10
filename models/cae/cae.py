import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.datasets as dsets  # for 4. MNIST dataset
import torchvision.transforms as transforms  # for tensor transforms
from torch.utils.data import DataLoader  # for batch learning
import numpy as np
import matplotlib.pyplot as plt
from network import *

# 4. MNIST dataset load
mnist_train = dsets.MNIST("MNIST_DATA/", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST("MNIST_DATA/", train=False, transform=transforms.ToTensor(), download=True)

# Hyperparameter
training_epochs = 10
batch_size      = 100
learning_rate   = 0.1

# Dataloder
train_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


# Model 객체 생성, optimizer, loss function
encoder = Encoder()
decoder = Decoder()

# encoder와 decoder의 parameter를 모두 학습해야한다
# 다음과 같은 방식으로 묶어서 Optimizer에 넣어주면 된다!

parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.SGD(parameters, lr=learning_rate)
loss_func = nn.MSELoss()  # pixel wise 계산할 것

for epoch in range(training_epochs):
    print(f"epoch : {epoch}")
    for x, y, in train_loader:
        out = encoder(x)
        out = decoder(out)

        optimizer.zero_grad()
        loss = loss_func(out, x)

        loss.backward()
        optimizer.step()

    print(loss)

# Input - Output 이미지 비교 확인
plt.subplot(1, 2, 1)
plt.imshow(torch.squeeze(out.data[1]).numpy(), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(torch.squeeze(x.data[1]).numpy(), cmap='gray')
plt.show()
