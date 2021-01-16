import os
import cv2
import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from config import Config as cfg

class Dataset(data.Dataset):
    def __init__(self, data_dir, data_list, phase):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.data_list = np.loadtxt(data_list, dtype=np.str)
        self.phase = phase
        # self.train_transform = transforms.Compose([
        #     transforms.Pad(4, padding_mode='reflect'),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         np.array([125.3, 123.0, 113.9]) / 255.0,
        #         np.array([63.0, 62.1, 66.7]) / 255.0),
        # ])
        # self.test_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         np.array([125.3, 123.0, 113.9]) / 255.0,
        #         np.array([63.0, 62.1, 66.7]) / 255.0),
        # ])

    def __getitem__(self, index):
        label = int(self.data_list[index].split('_')[1])
        img_path = os.path.join(self.data_dir, self.data_list[index])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
        img /= 255.0
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # if self.phase == 'train':
        #     img = self.train_transform(img)
        # else:
        #     img = self.test_transform(img)
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))

        return {'inputs': img, 'labels': label}

    def __len__(self):
        return self.data_list.shape[0]