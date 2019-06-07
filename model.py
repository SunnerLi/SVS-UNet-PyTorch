import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os

"""
    This script define the structure and update schema of U-Net

    @Reference: https://github.com/Jeongseungwoo/Singing-Voice-Separation
    @Revise: SunnerLi
"""

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the network components
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True)
        )
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv1_BAD = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv2_BAD = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv3_BAD = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv4_BAD = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv5 = nn.ConvTranspose2d(64, 16, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_BAD = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size = (5, 5), stride=(2, 2), padding=2)

        # Define loss list
        self.loss_list_vocal = []
        self.Loss_list_vocal = []

        # Define the criterion and optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.crit  = nn.L1Loss()
        self.to('cuda')

    # ==============================================================================
    #   IO
    # ==============================================================================
    def load(self, path):
        if os.path.exists(path):
            print("Load the pre-trained model from {}".format(path))
            state = torch.load(path)
            for (key, obj) in state.items():
                if len(key) > 10:
                    if key[1:9] == 'oss_list':
                        setattr(self, key, obj)
            self.conv1.load_state_dict(state['conv1'])
            self.conv2.load_state_dict(state['conv2'])
            self.conv3.load_state_dict(state['conv3'])
            self.conv4.load_state_dict(state['conv4'])
            self.conv5.load_state_dict(state['conv5'])
            self.conv6.load_state_dict(state['conv6'])
            self.deconv1.load_state_dict(state['deconv1'])
            self.deconv2.load_state_dict(state['deconv2'])
            self.deconv3.load_state_dict(state['deconv3'])
            self.deconv4.load_state_dict(state['deconv4'])
            self.deconv5.load_state_dict(state['deconv5'])
            self.deconv6.load_state_dict(state['deconv6'])
            self.deconv1_BAD.load_state_dict(state['deconv1_BAD'])
            self.deconv2_BAD.load_state_dict(state['deconv2_BAD'])
            self.deconv3_BAD.load_state_dict(state['deconv3_BAD'])
            self.deconv4_BAD.load_state_dict(state['deconv4_BAD'])
            self.deconv5_BAD.load_state_dict(state['deconv5_BAD'])
            self.optim.load_state_dict(state['optim'])
        else:
            print("Pre-trained model {} is not exist...".format(path))

    def save(self, path):
        # Record the parameters
        state = {
            'conv1': self.conv1.state_dict(),
            'conv2': self.conv2.state_dict(),
            'conv3': self.conv3.state_dict(),
            'conv4': self.conv4.state_dict(),
            'conv5': self.conv5.state_dict(),
            'conv6': self.conv6.state_dict(),
            'deconv1': self.deconv1.state_dict(),
            'deconv2': self.deconv2.state_dict(),
            'deconv3': self.deconv3.state_dict(),
            'deconv4': self.deconv4.state_dict(),
            'deconv5': self.deconv5.state_dict(),
            'deconv6': self.deconv6.state_dict(),
            'deconv1_BAD': self.deconv1_BAD.state_dict(),
            'deconv2_BAD': self.deconv2_BAD.state_dict(),
            'deconv3_BAD': self.deconv3_BAD.state_dict(),
            'deconv4_BAD': self.deconv4_BAD.state_dict(),
            'deconv5_BAD': self.deconv5_BAD.state_dict(),
        }

        # Record the optimizer and loss
        state['optim'] = self.optim.state_dict()
        for key in self.__dict__:
            if len(key) > 10:
                if key[1:9] == 'oss_list':
                    state[key] = getattr(self, key)
        torch.save(state, path)

    # ==============================================================================
    #   Set & Get
    # ==============================================================================
    def getLoss(self, normalize = False):
        loss_dict = {}
        for key in self.__dict__:
            if len(key) > 9 and key[0:9] == 'loss_list':
                if not normalize:
                    loss_dict[key] = round(getattr(self, key)[-1], 6)
                else:
                    loss_dict[key] = np.mean(getattr(self, key))
        return loss_dict

    def getLossList(self):
        loss_dict = {}
        for key in self.__dict__:
            if len(key) > 9 and key[0:9] == 'Loss_list':
                loss_dict[key] = getattr(self, key)
        return loss_dict

    def forward(self, mix):
        """
            Generate the mask for the given mixture audio spectrogram

            Arg:    mix     (torch.Tensor)  - The mixture spectrogram which size is (B, 1, 512, 128)
            Ret:    The soft mask which size is (B, 1, 512, 128)
        """
        conv1_out = self.conv1(mix)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        deconv1_out = self.deconv1(conv6_out, output_size = conv5_out.size())
        deconv1_out = self.deconv1_BAD(deconv1_out)
        deconv2_out = self.deconv2(torch.cat([deconv1_out, conv5_out], 1), output_size = conv4_out.size())
        deconv2_out = self.deconv2_BAD(deconv2_out)
        deconv3_out = self.deconv3(torch.cat([deconv2_out, conv4_out], 1), output_size = conv3_out.size())
        deconv3_out = self.deconv3_BAD(deconv3_out)
        deconv4_out = self.deconv4(torch.cat([deconv3_out, conv3_out], 1), output_size = conv2_out.size())
        deconv4_out = self.deconv4_BAD(deconv4_out)
        deconv5_out = self.deconv5(torch.cat([deconv4_out, conv2_out], 1), output_size = conv1_out.size())
        deconv5_out = self.deconv5_BAD(deconv5_out)
        deconv6_out = self.deconv6(torch.cat([deconv5_out, conv1_out], 1), output_size = mix.size())
        out = F.sigmoid(deconv6_out)
        return out

    def backward(self, mix, voc):
        """
            Update the parameters for the given mixture spectrogram and the pure vocal spectrogram

            Arg:    mix     (torch.Tensor)  - The mixture spectrogram which size is (B, 1, 512, 128)
                    voc     (torch.Tensor)  - The pure vocal spectrogram which size is (B, 1, 512, 128)
        """
        self.optim.zero_grad()
        msk = self.forward(mix)
        loss = self.crit(msk * mix, voc)
        self.loss_list_vocal.append(loss.item())
        loss.backward()
        self.optim.step()