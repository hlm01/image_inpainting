import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, dir):
        self.root_dir = dir
        self.transform = transforms.Compose(
            [transforms.Resize(256), transforms.ToTensor()]
        )
        self.image_filenames = [
            filename
            for filename in os.listdir(self.root_dir)
            if filename.endswith(".png")
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_filenames[idx])
        image = Image.open(image_path)

        image = self.transform(image)

        return image

if __name__ == "__main__":
    data = CustomDataset("images")
    dataloader = DataLoader(data, batch_size=32, shuffle=True, pin_memory=True)
    import matplotlib.pyplot as plt
    import numpy as np

    for batch_idx, images in enumerate(dataloader):
        print(f"Batch {batch_idx+1}/{len(dataloader)}")
        
        # Convert images back to numpy arrays
        images = images.numpy().transpose(0, 2, 3, 1)
        
        # Plot images
        fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i])
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Break after displaying one batch
        break
