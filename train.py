from model import Inpaint, PartialConv
from loss import InpaintingLoss
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from data import CustomDataset
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch.nn as nn
import os

start = 28
load = f"kaiming1_epoch_{start}.pth"
folder = "train_observe"
matplotlib.use("Agg")
torch.set_default_device("cuda")
# Creates model and optimizer in default precision


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        else:
            nn.init.constant_(m.weight, 1)


def bias_off(m):
    if isinstance(m, PartialConv):
        m.mask_conv.bias = None
        m.mask_conv.weight.requires_grad = False


def mask_w(m):
    if isinstance(m, nn.Conv2d) and m.bias is None:
        nn.init.constant_(m.weight, 1)


# Assuming you have defined your PyTorch module as `model`

model = Inpaint().to("cuda")
model.eval()
model.apply(init_weights)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0002)

if load:
    checkpoint = torch.load(load)

    # Load the model and optimizer state dictionaries
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

model.apply(bias_off)
model.apply(mask_w)
# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()
loss_f = InpaintingLoss()
dataset = CustomDataset("images", "mask")
gen = torch.Generator(device="cuda")
gen.seed()
data = DataLoader(dataset, 12, shuffle=True, pin_memory=True, generator=gen)


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
    ax[0].set_title(f"Input ")
    ax[0].axis("off")

    ax[1].imshow(b.transpose(1, 2, 0))
    ax[1].set_title(f"Output")
    ax[1].axis("off")

    ax[2].imshow(c.transpose(1, 2, 0))
    ax[2].set_title(f"GT")
    ax[2].axis("off")

    f_path = os.path.join(folder, msg)
    plt.savefig(f_path, bbox_inches="tight")
    plt.close('all')


for epoch in range(start + 1, 40):
    for i, d in enumerate(data):
        gt, mask = d
        optimizer.zero_grad()
        mask = mask.half()

        # Runs the forward pass with autocasting.
        with autocast(dtype=torch.float16):
            gt = gt.to("cuda")
            mask = mask.to("cuda")
            removed = gt * mask
            output = model(removed, mask)
            loss = loss_f(removed, mask, output, gt)
            n_loss = (
                1 * loss[0] + 6 * loss[1] + 0.05 * loss[2] + 120 * loss[3] + 0.1 * loss[4]
            )

        print(f"epoch {epoch} iteration {i}/{len(data)} loss {n_loss}")
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(n_loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # Zero your gradients for every batch!
        # optimizer.zero_grad()
        # gt = gt.to("cuda")
        # mask = mask.to("cuda")
        # removed = gt * mask
        # # Make predictions for this batch
        # output = model(removed, mask)

        # # Compute the loss and its gradients
        # loss = loss_f(removed, mask, output, gt)
        # n_loss = loss[0] + 6 * loss[1] + 0.05 * loss[2] + 120 * loss[3]
        # n_loss.backward()

        # # Adjust learning weights
        # optimizer.step()
        # print(
        #     f" hole {loss[0]}, valid {loss[1]}, perception {loss[2]}, style {loss[3]}, total {n_loss.item()}"
        # )
        if i % 100 == 0:
            display_batch_images(
                removed[-1], output[-1], gt[-1], f"epoch_{epoch}_i_{i}.png"
            )
            pass

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, f"kaiming1_epoch_{epoch}.pth")
