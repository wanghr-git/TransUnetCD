import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import LEVIR_CD_Dataset


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


def Eval(model_path):
    valid_dataset = LEVIR_CD_Dataset('/Users/wanghr/Data/LEVIR-CD/test',
                                     sub_dir_1='A',
                                     sub_dir_2='B',
                                     img_suffix='.png',
                                     ann_dir='/Users/wanghr/Data/LEVIR-CD/test/label',
                                     debug=True,
                                     test_mode=False)

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = torch.load(model_path)
    model.to(torch.device(DEVICE))

    model.Eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for i, (image_A, image_B, ann, filename) in tqdm(enumerate(valid_loader)):
            image_A = image_A.to(torch.device(DEVICE)).float()
            image_B = image_B.to(torch.device(DEVICE)).float()
            ann = ann.to(torch.device(DEVICE))

            pred = model(image_A, image_B)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

            y_true.append(ann.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        acc = accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))
        f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))
        precision = precision_score(y_true.reshape(-1), y_pred.reshape(-1))
        recall = recall_score(y_true.reshape(-1), y_pred.reshape(-1))
        IoU = np.sum(y_true * y_pred) / np.sum(y_true + y_pred - y_true * y_pred)

        print('acc: ', acc)
        print('f1: ', f1)
        print('precision: ', precision)
        print('recall: ', recall)
        print('IoU: ', IoU)
