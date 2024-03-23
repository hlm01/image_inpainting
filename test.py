import argparse
from model import *
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def display_batch_images(a, b, c, msg):
    """
    Displays the last two images in a PyTorch tensor batch using Matplotlib.

    Args:
        tensor (torch.Tensor): A PyTorch tensor with dimensions b x c x h x w (batch, channels, height, width).
        normalize (bool, optional): Whether to normalize the tensor values to the range [0, 1]. Default is True.

    Returns:
        None
    """
    # Move the tensor to CPU
    a = a.to("cpu").detach().numpy().astype(float)
    b = b.to("cpu").detach().numpy().astype(float)
    b = np.clip(b, 0, 1)
    c = c.to("cpu").detach().numpy().astype(float)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].imshow(a.transpose(1, 2, 0))
    ax[0].set_title(f"Original")
    ax[0].axis("off")

    ax[1].imshow(b.transpose(1, 2, 0))
    ax[1].set_title(f"Masked")
    ax[1].axis("off")

    ax[2].imshow(c.transpose(1, 2, 0))
    ax[2].set_title(f"Output")
    ax[2].axis("off")

    f_path = os.path.join(folder, msg)
    plt.savefig(f_path, bbox_inches="tight")
    plt.close("all")


folder = "out"

net = Inpaint().to("cuda")
checkpoint = torch.load("kaiming1_epoch_34.pth")


def bias_off(m):
    if isinstance(m, PartialConv):
        m.mask_conv.bias = None
        m.mask_conv.weight.requires_grad = False


def mask_w(m):
    if isinstance(m, nn.Conv2d) and m.bias is None:
        nn.init.constant_(m.weight, 1)

    # Load the model and optimizer state dictionaries


net.load_state_dict(checkpoint["model_state_dict"], strict=False)
net.apply(bias_off)
net.apply(mask_w)


parser = argparse.ArgumentParser(description="Apply Inpainting model to input files")
parser.add_argument("input_image", type=str, help="Path to the image input file")
parser.add_argument("input_mask", type=str, help="Path to the mask input file")

transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
args = parser.parse_args()
input_mask = args.input_mask
input_image = args.input_image

mask = Image.open(input_mask)
gt = Image.open(input_image)

mask = 1 - transform(mask)
mask = mask.expand(3, -1, -1).to("cuda")

gt = transform(gt).to("cuda")

removed = (mask * gt).to("cuda")
o_remove = 1 - mask
o_remove = o_remove.unsqueeze(0)

gt = gt.unsqueeze(0)
o_remove += gt

removed = removed.unsqueeze(0)
mask = mask.unsqueeze(0)
output = net(removed, mask)

display_batch_images(gt[0], o_remove[0], output[0], "2.png")
