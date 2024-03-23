import torch
import torch.nn as nn
from torchvision import models


class InpaintingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(MEAN).view(-1, 1, 1)
        self.std = torch.tensor(STD).view(-1, 1, 1)

        self.feat1 = nn.Sequential(*self.net.features[:5])
        self.feat2 = nn.Sequential(*self.net.features[5:10])
        self.feat3 = nn.Sequential(*self.net.features[10:17])

    def extract(self, x):
        phi = [self.feat1(x)]
        phi.append(self.feat2(phi[-1]))
        phi.append(self.feat3(phi[-1]))
        return phi

    def gram(self, mat):
        (b, c, h, w) = mat.shape
        mat = mat.view(b, c, -1)
        mat_t = mat.transpose(1, 2)
        input = torch.zeros(b, c, c).type(mat.type())
        res = torch.baddbmm(
            input, mat, mat_t, beta=0, alpha=1.0 / (c * h * w), out=None
        )
        return res

    def forward(self, input, mask, output, gt):
        comp = mask * input + (1 - mask) * output

        hole = self.l1((1 - mask) * gt, (1 - mask) * output)
        valid = self.l1(mask * gt, mask * output)
        norm_gt = (gt - self.mean) / self.std
        norm_out = (output - self.mean) / self.std

        feats_out = self.extract(norm_out)
        feats_comp = self.extract(comp)
        feats_gt = self.extract(norm_gt)

        perception = 0
        style = 0
        for i in range(3):
            perception += self.l1(feats_out[i], feats_gt[i])
            perception += self.l1(feats_comp[i], feats_gt[i])
            style += self.l1(self.gram(feats_out[i]), self.gram(feats_gt[i]))
            style += self.l1(self.gram(feats_comp[i]), self.gram(feats_gt[i]))

        smooth_loss = (
            torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:]).mean()
            + torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :]).mean()
        )

        return [hole, valid, perception, style, smooth_loss]
