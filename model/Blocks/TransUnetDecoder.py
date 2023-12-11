import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class CUDModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CUDModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, feature_1, feature_2):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat([x, feature_1, feature_2], dim=1)
        x = self.conv(x)
        return x


class SAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x,dim=1).unsqueeze(dim=1)
        max_out = torch.max(x,dim=1,keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.fc(x)
        return x


class DEMModule(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=512):
        super(DEMModule, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.SA = SAModule(2, 1)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)+x
        x = torch.multiply(x, self.SA(x))
        return x
