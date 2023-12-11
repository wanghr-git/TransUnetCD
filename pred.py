import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torchvision import transforms

from model.SF_ResNet import SF_ResNet


# 预测一张图片
def predict_one_image(model, data_root_path, device, image_name):
    # 1.读取图片
    image_A = Image.open(os.path.join(data_root_path, 'A', image_name))
    image_B = Image.open(os.path.join(data_root_path, 'B', image_name))
    # 2.预处理
    transform = transforms.Compose([
        transforms.ToTensor()])
    image_A = transform(image_A)
    image_B = transform(image_B)
    # 3.转移到GPU
    image_A = image_A.to(device)
    image_B = image_B.to(device)

    # 图片在512大小的数据上训练，但是测试数据是1024的，需要将测试图片裁剪为512分别预测
    image_A_1 = image_A[:, :, :512, :512]
    image_A_2 = image_A[:, :, :512, 512:]
    image_A_3 = image_A[:, :, 512:, :512]
    image_A_4 = image_A[:, :, 512:, 512:]
    image_B_1 = image_B[:, :, :512, :512]
    image_B_2 = image_B[:, :, :512, 512:]
    image_B_3 = image_B[:, :, 512:, :512]
    image_B_4 = image_B[:, :, 512:, 512:]

    # 4.预测
    model.Eval()
    with torch.no_grad():
        change_map_1 = model(image_A_1, image_B_1).cpu().numpy()
        change_map_2 = model(image_A_2, image_B_2).cpu().numpy()
        change_map_3 = model(image_A_3, image_B_3).cpu().numpy()
        change_map_4 = model(image_A_4, image_B_4).cpu().numpy()

    # 5.拼接
    change_map = np.zeros((1, 1, 1024, 1024))
    change_map[:, :, :512, :512] = change_map_1
    change_map[:, :, :512, 512:] = change_map_2
    change_map[:, :, 512:, :512] = change_map_3
    change_map[:, :, 512:, 512:] = change_map_4

    # 保存为图片
    change_map = change_map.squeeze()
    change_map = change_map * 255
    change_map = change_map.astype(np.uint8)
    change_map = Image.fromarray(change_map)
    change_map.save(os.path.join(data_root_path, 'change_map', image_name))

    return change_map


if __name__ == '__main__':
    Model = SF_ResNet(encoder_name='resnet50', neck_name='fpn', out_channels=1, inplanes=128, pretrained='None')
    Model.load_state_dict(torch.load('/Users/wanghr/Data/LEVIR-CD/checkpoint/SF_ResNetmodel_best.pt'))
    predict = predict_one_image(Model, '/Users/wanghr/Data/LEVIR-CD/test', torch.device('cuda:0'), 'test_1.png')
