import torch
import torch.nn as nn

from .Base import Conv3Relu
from .Drop import DropBlock
from .Field import PPM, ASPP, SPP


class FPNNeck_ResNet(nn.Module):
    """
    用于ResNet的FPN,ResNet的输出为5个特征图的通道数为(3, 64, 256, 512, 1024, 2048)
    """
    def __init__(self, inplanes, neck_name='fpn+fuse'):
        super().__init__()
        self.conv_1 = Conv3Relu(inplanes//2, inplanes)
        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)       # channel: 2*inplanes  ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)   # channel: 4*inplanes  ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)   # channel: 8*inplanes  ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        self.stage1_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 16, inplanes * 8)

        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)

        # PPM/ASPP比SPP好
        if "+ppm+" in neck_name:
            self.expand_field = PPM(inplanes * 8)
        elif "+aspp+" in neck_name:
            self.expand_field = ASPP(inplanes * 8)
        elif "+spp+" in neck_name:
            self.expand_field = SPP(inplanes * 8)
        else:
            self.expand_field = None

        if "fuse" in neck_name:
            self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)   # 降维
            self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
            self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)

            self.final_Conv = Conv3Relu(inplanes * 4, inplanes)

            self.fuse = True
        else:
            self.fuse = False

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = DropBlock(rate=0, size=0, step=0)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fa5, fb1, fb2, fb3, fb4, fb5 = ms_feats  # 10个特征图

        [fa1, fa2, fa3, fa4, fa5, fb1, fb2, fb3, fb4, fb5] = self.drop([fa1, fa2, fa3, fa4, fa5, fb1, fb2, fb3, fb4, fb5])  # dropblock

        # 前后时相特征做差
        diff_1 = self.conv_1(fa1 - fb1)     # 128*128*inplanes
        diff_2 = fa2 - fb2     # 64*64*2inplanes
        diff_3 = fa3 - fb3     # 32*32*4inplanes
        diff_4 = fa4 - fb4     # 16*16*8inplanes
        diff_5 = fa5 - fb5     # 8*8*  16inplanes

        # stage5
        change_5 = diff_5  # 8*8* 16inplanes,  第一个特征变化图

        # stage4
        diff_5_up = self.up(diff_5)         # 16*16* 16inplanes
        diff_5_up = self.stage4_Conv_after_up(diff_5_up)  # 16*16* 8inplanes
        change_4 = self.stage4_Conv1(torch.cat([diff_5_up, diff_4], 1))  # 16*16* 8inplanes

        # stage3
        diff_4_up = self.up(diff_4)         # 32*32* 8inplanes
        diff_4_up = self.stage3_Conv_after_up(diff_4_up)  # 32*32* 4inplanes
        change_3 = self.stage3_Conv1(torch.cat([diff_4_up, diff_3], 1))  # 32*32* 4inplanes

        # stage2
        diff_3_up = self.up(diff_3)         # 64*64* 4inplanes
        diff_3_up = self.stage2_Conv_after_up(diff_3_up)  # 64*64* 2inplanes
        change_2 = self.stage2_Conv1(torch.cat([diff_3_up, diff_2], 1))  # 64*64* 2inplanes

        # stage1
        diff_2_up = self.up(diff_2)         # 128*128* 2inplanes
        diff_2_up = self.stage1_Conv_after_up(diff_2_up)  # 128*128* inplanes
        change_1 = self.stage1_Conv2(torch.cat([diff_2_up, diff_1], 1))  # 128*128* inplanes

        return [change_1, change_2, change_3, change_4, change_5]


class FPNNeck_SwinTiny(nn.Module):
    """
    用于Swin的FPN,SwinBase的输出为5个特征图的通道数为(3, 96, 96, 192, 384, 768),Resnet的输出为5个特征图的通道数为(3, 64, 256, 512, 1024, 2048)
    """
    def __init__(self, inplanes, neck_name='fpn+fuse'):
        super().__init__()
        self.conv_1 = Conv3Relu(inplanes, inplanes)
        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)       # channel: 2*inplanes  ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 2, inplanes * 1)   # channel: 4*inplanes  ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)   # channel: 8*inplanes  ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)  # channel: 16*inplanes ---> 8*inplanes

        self.stage1_Conv_after_up = Conv3Relu(inplanes, inplanes)
        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes * 1)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)

        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)

        # PPM/ASPP比SPP好
        if "+ppm+" in neck_name:
            self.expand_field = PPM(inplanes * 8)
        elif "+aspp+" in neck_name:
            self.expand_field = ASPP(inplanes * 8)
        elif "+spp+" in neck_name:
            self.expand_field = SPP(inplanes * 8)
        else:
            self.expand_field = None

        if "fuse" in neck_name:
            self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)   # 降维
            self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
            self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)

            self.final_Conv = Conv3Relu(inplanes * 4, inplanes)

            self.fuse = True
        else:
            self.fuse = False

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = DropBlock(rate=0, size=0, step=0)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fa5, fb1, fb2, fb3, fb4, fb5 = ms_feats  # 10个特征图

        [fa1, fa2, fa3, fa4, fa5, fb1, fb2, fb3, fb4, fb5] = self.drop([fa1, fa2, fa3, fa4, fa5, fb1, fb2, fb3, fb4, fb5])  # dropblock

        # 前后时相特征做差
        diff_1 = fa1 - fb1     # 128*128*inplanes
        diff_2 = fa2 - fb2     # 64*64*inplanes
        diff_3 = fa3 - fb3     # 32*32*2inplanes
        diff_4 = fa4 - fb4     # 16*16*4inplanes
        diff_5 = fa5 - fb5     # 8*8*  8inplanes

        # stage5
        change_5 = diff_5  # 8*8* 8inplanes,  第一个特征变化图

        # stage4
        diff_5_up = self.up(diff_5)         # 16*16* 8inplanes
        diff_5_up = self.stage4_Conv_after_up(diff_5_up)  # 16*16* 8inplanes
        change_4 = self.stage4_Conv1(torch.cat([diff_5_up, diff_4], 1))  # 16*16* 4inplanes

        # stage3
        diff_4_up = self.up(diff_4)         # 32*32* 8inplanes
        diff_4_up = self.stage3_Conv_after_up(diff_4_up)  # 32*32* 4inplanes
        change_3 = self.stage3_Conv1(torch.cat([diff_4_up, diff_3], 1))  # 32*32* 2inplanes

        # stage2
        diff_3_up = self.up(diff_3)         # 64*64* 4inplanes
        diff_3_up = self.stage2_Conv_after_up(diff_3_up)  # 64*64* 2inplanes
        change_2 = self.stage2_Conv1(torch.cat([diff_3_up, diff_2], 1))  # 64*64* inplanes

        # stage1
        diff_2_up = self.up(diff_2)         # 128*128* 2inplanes
        diff_2_up = self.stage1_Conv_after_up(diff_2_up)  # 128*128* inplanes
        change_1 = self.stage1_Conv2(torch.cat([diff_2_up, diff_1], 1))  # 128*128* inplanes

        return [change_1, change_2, change_3, change_4, change_5]

