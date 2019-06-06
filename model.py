import torch.nn.functional as F
import torch.nn as nn
import torch

class Concat2d(nn.Module):
    def forward(self, a, b):
        return torch.cat([a, b], 1)

class UNet(nn.Module):
    def __init__(self):
        # Define the network components
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv2 = nn.Sequential(
            Concat2d(),
            nn.ConvTranspose2d(256, 128, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv3 = nn.Sequential(
            Concat2d(),
            nn.ConvTranspose2d(128, 64, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv4 = nn.Sequential(
            Concat2d(),
            nn.ConvTranspose2d(64, 32, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv5 = nn.Sequential(
            Concat2d(),
            nn.ConvTranspose2d(32, 16, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv6 = nn.Sequential(
            Concat2d(),
            nn.ConvTranspose2d(16, 1, kernel_size = (5, 5), stride=(2, 2), padding=2),
        )

        # Define loss list
        self.loss_list_vocal = []
        self.Loss_list_vocal = []

        # Define the criterion and optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.crit  = nn.L1Loss()
        self.to('cuda')

    def forward(self, mix):
        conv1_out = self.conv1(mix)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        deconv1_out = self.deconv1(conv6_out)
        deconv2_out = self.deconv2(deconv1_out, conv5_out)
        deconv3_out = self.deconv3(deconv2_out, conv4_out)
        deconv4_out = self.deconv4(deconv3_out, conv3_out)
        deconv5_out = self.deconv5(deconv4_out, conv2_out)
        deconv6_out = self.deconv6(deconv5_out, conv1_out)
        out = F.sigmoid(deconv6_out)
        return out

    def backward(self, mix, voc):
        self.optim.zero_grad()
        msk = self.forward(mix)
        loss = self.crit(msk * mix, voc)
        self.loss_list_vocal.append(loss.item())
        loss.backward()
        self.optim.step()