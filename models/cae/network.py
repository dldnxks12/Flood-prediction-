import torch
import torch.nn as nn

# Hyperparameter
training_epochs = 10
batch_size = 100
learning_rate = 0.1

# Model Architecture
class Encoder(nn.Module):  # Convolution
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(

        nn.Conv2d(1, 16, 3, padding=1),  # batch x 16 x 28 x 28
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, padding=1),  # batch x 32 x 28 x 28
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x 28 x 28
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2)  # batch x 64 x 14 x 14
        )
        
        self.layer2 = nn.Sequential(

            nn.Conv2d(64, 128, 3, padding=1),  # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # batch x 128 x 7 x 7
            nn.Conv2d(128, 256, 3, padding=1),  # batch x 256 x 7 x 7
            nn.ReLU()
        )

    def forward(self, x):
        z = self.layer1(x)
        z = self.layer2(z)
        out = z.view(batch_size, -1)  # latent space vector로써 1차원 tensor로
        return out


# ******************************* Deconvolution *********************************** #

class Decoder(nn.Module):  # Deconvolution
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(

            # nn.ConvTranspose2d(in-channel, out-channel, kernel, stide, padding, output-padding, ...)
            # padding 추가시, outline padding 크기 만큼 제거

            # ( batch x 256 x 7 x 7 ) -> ( batch x 128 x 14 x 14 ) # 크기 증가
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64)

        )

        self.layer2 = nn.Sequential(

            nn.ConvTranspose2d(64, 16, 3, 1, 1),  # batch x 16 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),  # batch x 1 x 28 x 28 # 크기 증가
            nn.ReLU()

        )

    def forward(self, x):
        x = x.view(batch_size, 256, 7, 7)
        out = self.layer1(x)
        out = self.layer2(out)

        return out  # batch x 1 x 28 x 28

