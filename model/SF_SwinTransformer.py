import torch.nn as nn
import torch

from .Blocks.Base import DeConv, Conv3Relu
from .Blocks.FPN import FPNNeck_SwinTiny
from .encoders import get_encoder


class SF_SwinTransformer(nn.Module):
    def __init__(self, encoder_name='Swin-T', neck_name='fpn', out_channels=1, inplanes=96, pretrained=None):
        super(SF_SwinTransformer, self).__init__()
        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=pretrained)
        self.neck = FPNNeck_SwinTiny(inplanes=inplanes, neck_name=neck_name)
        self.deconv_for_change_5 = DeConv(8 * inplanes, inplanes, scale=8)   # change5--> 64*64*inplanes
        self.deconv_for_change_4 = DeConv(4 * inplanes, inplanes, scale=4)   # change4--> 64*64*inplanes
        self.deconv_for_change_3 = DeConv(2 * inplanes, inplanes, scale=2)   # change3--> 64*64*inplanes
        self.conv_for_change_2 = Conv3Relu(1 * inplanes, inplanes)           # change2--> 64*64*inplanes
        self.up_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(inplanes, out_channels, (1, 1), bias=False)

    def forward(self, A, B):
        _, fa1, fa2, fa3, fa4, fa5 = self.encoder(A)
        _, fb1, fb2, fb3, fb4, fb5 = self.encoder(B)
        ms_feats = [fa1, fa2, fa3, fa4, fa5, fb1, fb2, fb3, fb4, fb5]
        ms_feats = self.neck(ms_feats)
        change_1, change_2, change_3, change_4, change_5 = ms_feats
        change_5 = self.deconv_for_change_5(change_5)  # 64*64*inplanes
        change_4 = self.deconv_for_change_4(change_4)  # 64*64*inplanes
        change_3 = self.deconv_for_change_3(change_3)  # 64*64*inplanes
        change_2 = self.conv_for_change_2(change_2)  # 64*64*inplanes
        change_fuse = change_2 + change_3 + change_4 + change_5
        change_fuse = self.up_1(change_fuse)  # 256*256*inplanes
        change_fuse = change_fuse + self.up_2(change_1)  # 256*256*inplanes

        change_map = self.final_conv(change_fuse)  # 256*256*2
        return torch.sigmoid(change_map)
