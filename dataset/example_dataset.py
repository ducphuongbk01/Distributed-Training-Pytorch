import os
import random

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self, data_path, labels, height, width, phase):
        super().__init__()
        self.data_path = data_path
        self.labels = labels
        self.data_list = self._load_data(self.data_path, self.labels)
        random.shuffle(self.data_list)

        self.height = height
        self.width = width
        self.phase = phase
        self.transforms = self._build_transform(self.phase, self.height, self.width)

    def _load_data(self, data_path, labels):
        data_list = []
        for idx, lb in enumerate(labels):
            lb_path = os.path.join(data_path, lb)
            map_fnc = lambda name: (os.path.join(lb_path, name), idx)
            data_list.extend(list(map(map_fnc, sorted(os.listdir(lb_path)))))
        return data_list
    
    def _build_transform(self, phase, height, width):
        prob = 0.5
        if phase=="train":
            transforms = A.Compose([A.Resize(height=height, width=width),
                                    A.RandomRotate90(p=prob),
                                    A.HorizontalFlip(p=prob),
                                    A.VerticalFlip(p=prob),
                                    A.Blur(p=prob),
                                    A.MedianBlur(p=prob),
                                    A.CLAHE(p=prob),
                                    A.RandomBrightnessContrast(p=prob),
                                    A.RandomGamma(p=prob),
                                    A.ImageCompression(quality_lower=20, quality_upper=100, p=prob),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                                    ToTensorV2()])
        else:
            transforms = A.Compose([A.Resize(height=height, width=width),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                                    ToTensorV2()])
        return transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, lb = self.data_list[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transforms(image=img)["image"], lb
