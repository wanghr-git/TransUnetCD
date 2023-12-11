import os.path as osp
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .custom import CustomDataset
from .transforms.albu import ChunkImage, ToTensorTest
import cv2


class LEVIR_CD_Dataset(CustomDataset):
    """LEVIR-CD dataset"""

    def __init__(self, img_dir, sub_dir_1='A', sub_dir_2='B', ann_dir=None, img_suffix='.png', seg_map_suffix='.png',
                 transform=None, split=None, data_root=None, test_mode=False, size=256, debug=False):
        super().__init__(img_dir, sub_dir_1, sub_dir_2, ann_dir, img_suffix, seg_map_suffix, transform, split,
                         data_root, test_mode, size, debug)

    def get_default_transform(self):
        """Set the default transformation."""

        default_transform = A.Compose([
            A.RandomCrop(self.size, self.size),
            # A.ShiftScaleRotate(),
            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'})
        return default_transform

    def get_test_transform(self):
        """Set the test transformation."""

        test_transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'})
        return test_transform

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if not self.ann_dir:
            ann = None
            img1, img2, filename = self.prepare_img(idx)
            transformed_data = self.transform(image=img1, image_2=img2)
            img1, img2 = transformed_data['image'], transformed_data['image_2']
            # HWC to CHW, numpy to tensor
            if img1.shape[2] == 3:
                img1 = img1.transpose(2, 0, 1)
                img2 = img2.transpose(2, 0, 1)
            return img1, img2, filename
        else:
            img1, img2, ann, filename = self.prepare_img_ann(idx)
            transformed_data = self.transform(image=img1, image_2=img2, mask=ann)
            img1, img2, ann = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']
            # 可视化img1, img2, ann，检测是否正确
            ann = ann[:, :, np.newaxis]
            img1 = (img1 * 255).astype(np.uint8)
            img2 = (img2 * 255).astype(np.uint8)
            ann = (ann * 255).astype(np.uint8)
            cv2.imshow('img1', img1)
            cv2.imshow('img2', img2)
            cv2.imshow('ann', ann)
            # HWC to CHW, numpy to tensor
            if img1.shape[2] == 3:
                img1 = img1.transpose(2, 0, 1)
                img2 = img2.transpose(2, 0, 1)
                # ann只需要新建一个维度
                ann = ann[np.newaxis, :, :]
            return img1, img2, ann, filename


if __name__ == "__main__":
    LEVIR_CD_Dataset('dir')
