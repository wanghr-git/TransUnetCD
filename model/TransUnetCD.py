import torch
import torch.nn as nn
import torch.nn.functional as F

from ResNet_50 import ResNet50
from Transformer import TransformerEncoder, PatchEmbedding
from TransUnetDecoder import CUDModule, DEMModule, ConvBlock2


class TransUnetCD(nn.Module):
    def __init__(self, img_size, in_channels, feature_channels, out_channels):
        super(TransUnetCD, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ConvBlock2_1 = ConvBlock2(768, 512)
        self.ConvBlock2_2 = ConvBlock2(64, 16)

        self.ResNet50_1 = ResNet50()
        self.ResNet50_2 = ResNet50()

        self.PatchEmbedding = PatchEmbedding(patch_size=1, in_channels=feature_channels)
        self.TransformerEncoder = TransformerEncoder()

        self.CUDModule_1 = CUDModule(512*3, 256)
        self.CUDModule_2 = CUDModule(256*3, 128)
        self.CUDModule_3 = CUDModule(64*4, 64)

        self.DEMModule = DEMModule(3, 16)


    def forward(self,x_1, x_2):
        diff = x_1-x_2

        feature_1 = self.ResNet50_1(x_1)
        feature_2 = self.ResNet50_2(x_2)

        feature = torch.cat([feature_1[3], feature_2[3]], dim=1)
        PatchEmbed = self.PatchEmbedding(feature)
        feature = self.TransformerEncoder(PatchEmbed)
        batch_size = feature.shape[0]
        feature = feature.reshape(batch_size, 768, int(self.img_size/16), int(self.img_size/16))
        feature = self.ConvBlock2_1(feature)

        x = self.CUDModule_1(feature, feature_1[2], feature_2[2])
        x = self.CUDModule_2(x, feature_1[1], feature_2[1])
        x = self.CUDModule_3(x, feature_1[0], feature_2[0])

        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = self.ConvBlock2_2(x)
        DEM_feature = self.DEMModule(diff)

        out = torch.multiply(x, DEM_feature)
        return out


if __name__ == "__main__":
    input_1 = torch.rand((8, 3, 512, 512))
    input_2 = torch.rand((8, 3, 512, 512))
    net = TransUnetCD(512, 3, 2048, 16)
    test = net(input_1, input_2)
    print('11') 