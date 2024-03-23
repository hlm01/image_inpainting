import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import choice


class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose(
            [transforms.Resize(256), transforms.ToTensor()]
        )
        self.image_filenames = [
            filename
            for filename in os.listdir(self.img_dir)
            if filename.endswith(".png")
        ]
        self.mask_filenames = [
            filename
            for filename in os.listdir(self.mask_dir)
            if filename.endswith(".png")
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, choice(self.mask_filenames))
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = self.transform(image)
        mask = self.transform(mask)
        mask = 1 - mask
        mask = mask.expand(3,-1,-1)

        return image, mask
