import os
import cv2
import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from config import Config as cfg


class Dataset(data.Dataset):
    def __init__(self, data_dir, data_list, phase):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.data_list = np.loadtxt(data_list, dtype=np.str)
        self.phase = phase
        self.transforms_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), ## function: 1. pixel value : [0, 255] => [0.0, 1.0]; 2. premute(2,0,1): [H, W, C] => [C, H, W]
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __getitem__(self, index):
        label = int(self.data_list[index].split('_')[1])
        img_path = os.path.join(self.data_dir, self.data_list[index])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.phase == 'train':
            img = self.transforms_train(Image.fromarray(img))
        else:
            img = self.transforms_test(Image.fromarray(img))

        # img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))

        return {'inputs': img, 'targets': label}

    def __len__(self):
        return self.data_list.shape[0]