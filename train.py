import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import LEVIR_CD_Dataset
from losses.dice import DiceLoss
from model.SF_ResNet import SF_ResNet
from model.SF_SwinTransformer import SF_SwinTransformer
import json


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(seed=1024)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)


def train(epoches=100):
    train_dataset = LEVIR_CD_Dataset('/Users/wanghr/Data/LEVIR-CD/train',
                                     sub_dir_1='A',
                                     sub_dir_2='B',
                                     img_suffix='.png',
                                     ann_dir='/Users/wanghr/Data/LEVIR-CD/train/label',
                                     debug=True)

    valid_dataset = LEVIR_CD_Dataset('/Users/wanghr/Data/LEVIR-CD/val',
                                     sub_dir_1='A',
                                     sub_dir_2='B',
                                     img_suffix='.png',
                                     ann_dir='/Users/wanghr/Data/LEVIR-CD/val/label',
                                     debug=True,
                                     test_mode=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    # model = SF_ResNet(encoder_name='resnet50', neck_name='fpn', out_channels=1, inplanes=128, pretrained=None)
    model = SF_SwinTransformer(encoder_name='Swin-T', neck_name='fpn', out_channels=1, inplanes=96, pretrained=None)
    # 损失函数使用DiceLoss和BCELoss的结合
    loss = DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    model.to(torch.device(DEVICE))
    # 记录训练历史数据
    history = {'train_loss': [], 'train_acc': [],
               'valid_loss': [], 'valid_acc': [],
               'best_epoch': 0
               }

    best_epoch = 0
    best_f1 = 0

    for epoch in range(epoches):
        model.train()
        epoch_start = time.time()  # 每轮开始时间记录
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        f1 = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epoches}', unit='img', leave=True) as pbar:
            for i, (item) in enumerate(train_loader):
                img1, img2, ann, filename = item
                # 转移到CUDA并且转换为float
                img1 = img1.to(torch.device(DEVICE)).float()
                img2 = img2.to(torch.device(DEVICE)).float()
                ann = ann.to(torch.device(DEVICE)).float() // 255
                pred = model(img1, img2)
                loss_value = loss(pred, ann)
                optimizer.zero_grad()
                loss_value.backward()  # 计算梯度
                optimizer.step()  # 更新参数
                train_loss += loss_value.item()
                pred_for_acc = pred.cpu().detach().numpy().reshape(-1)
                pred_for_acc[np.where(pred_for_acc > 0.5)] = 1
                pred_for_acc[np.where(pred_for_acc <= 0.5)] = 0
                ann_for_acc = ann.cpu().detach().numpy().reshape(-1)
                train_acc += accuracy_score(ann_for_acc, pred_for_acc)

                pbar.set_postfix(**{'loss (batch)': loss_value.item()})
                pbar.update(img1.shape[0])

        with tqdm(total=len(valid_loader), desc=f'Epoch {epoch + 1}/{epoches}', unit='img', leave=True) as pbar:
            model.eval()
            for i, (item) in enumerate(valid_loader):
                img1, img2, ann, filename = item
                # 转移到CUDA并且转换为float
                img1 = img1.to(torch.device(DEVICE)).float()
                img2 = img2.to(torch.device(DEVICE)).float()
                ann = ann.to(torch.device(DEVICE)).float() // 255
                pred = model(img1, img2)
                loss_value = loss(pred, ann)
                valid_loss += loss_value.item()

                pred_for_acc = pred.cpu().detach().numpy().reshape(-1)
                pred_for_acc[np.where(pred_for_acc > 0.5)] = 1
                pred_for_acc[np.where(pred_for_acc <= 0.5)] = 0
                ann_for_acc = ann.cpu().detach().numpy().reshape(-1)
                valid_acc += accuracy_score(ann_for_acc, pred_for_acc)
                f1 += f1_score(ann_for_acc, pred_for_acc)
                pbar.set_postfix(**{'loss (batch)': loss_value.item()})
                pbar.update(img1.shape[0])

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_valid_acc = valid_acc / len(valid_loader)
        avg_valid_f1 = f1 / len(valid_loader)

        # 学习率调整策略
        avg_valid_f1 = torch.tensor(avg_valid_f1)
        scheduler.step(avg_valid_f1)
        avg_valid_f1 = avg_valid_f1.item()

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['valid_loss'].append(avg_valid_loss)
        history['valid_acc'].append(avg_valid_acc)
        if best_f1 < avg_valid_f1:
            best_f1 = avg_valid_f1
            best_epoch = epoch
            history['best_epoch'] = best_epoch

        epoch_end = time.time()

        print(
            "\nEpoch: {:03d}\n"
            "Training:   Loss: {:.4f};\n"
            "            Accuracy: {:.4f}%;\n"
            "Validation: Loss: {:.4f};\n"
            "            Accuracy: {:.4f}%;\n"
            "            F1-Score: {:.4f};\n"
            "Time Spent: {:.4f}s".format(
                epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100, avg_valid_f1 * 100,
                epoch_end - epoch_start
            ))
        print("Best F1-Score for validation : {:.4f} at epoch {:03d}".format(best_f1, best_epoch))
        if best_epoch == epoch + 1:
            torch.save(model, './checkpoint/' + 'SF_ResNet' + 'model_best.pt')
    return history


if __name__ == '__main__':
    log = train(100)
    with open('./checkpoint/' + 'SF_ResNet' + 'log.log', 'w') as f:
        json.dump(log, f)
