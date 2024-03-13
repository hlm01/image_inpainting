import torch
import torch.nn as nn


class Inpaint(nn.Module):
    def __init__(self):
        #         K5S1C32 - K3S2C64 - K3S1C64 - K3S2C128 -
        # K3S1C128 - K3S1C128 - K3D2S1C128 - K3D4S1C128 -
        # K3D8S1C128 - K3D16S1C128 - K3S1C128 - K3S1C128 -
        # resize (2×) - K3S1C64 - K3S1C64 - resize (2×) - K3S1C32
        # - K3S1C16 - K3S1C3 - clip.

        super().__init__()
        ch = 32
        self.upsample = nn.Sequential(
            GatedConv(2 * ch, 4 * ch),
            GatedConv(2 * ch, 4 * ch),
            nn.Upsample(scale_factor=2),
            GatedConv(2 * ch, 2 * ch),
            GatedConv(ch, 2 * ch),
            nn.Upsample(scale_factor=2),
            GatedConv(ch, ch),
            GatedConv(ch // 2, ch),
            GatedConv(ch // 2, ch // 2),
            GatedConv(ch // 4, 3),
            nn.Tanh()
        )
        self.course = nn.Sequential(
            GatedConv(5, ch, kernel=5),
            GatedConv(ch // 2, 2 * ch, stride=2),
            GatedConv(ch, 2 * ch),
            GatedConv(ch, 4 * ch, stride=2),
            GatedConv(2 * ch, 4 * ch),
            GatedConv(2 * ch, 4 * ch),
            GatedConv(2 * ch, 4 * ch, rate=2),
            GatedConv(2 * ch, 4 * ch, rate=4),
            GatedConv(2 * ch, 4 * ch, rate=8),
            GatedConv(2 * ch, 4 * ch, rate=16),
            self.upsample
        )
        self.refine1 = nn.Sequential(
            GatedConv(5, ch, kernel=5),
            GatedConv(ch // 2, 2 * ch, stride=2),
            GatedConv(ch, 2 * ch),
            GatedConv(ch, 4 * ch, stride=2),
            GatedConv(2 * ch, 4 * ch),
            GatedConv(2 * ch, 4 * ch),
            GatedConv(2 * ch, 4 * ch, rate=2),
            GatedConv(2 * ch, 4 * ch, rate=4),
            GatedConv(2 * ch, 4 * ch, rate=8),
            GatedConv(2 * ch, 4 * ch, rate=16),
        )
        self.refine2 = nn.Sequential(
            GatedConv(5, ch, kernel=5),
            GatedConv(ch // 2, 2 * ch, stride=2),
            GatedConv(ch, 2 * ch),
            GatedConv(ch, 4 * ch, stride=2),
            GatedConv(2 * ch, 4 * ch),
            GatedConv(2 * ch, 4 * ch, activation=nn.ReLU),
        )
        self.concat_conv = GatedConv(4 * ch, 4 * ch)

    def forward(self, x):
        original = x
        mask = x[:, None, 3, :, :]
        x = self.course(x)
        x = x * mask + original[:, :3, :, :] * (1 - mask)
        x_course = torch.concat([x, mask, torch.ones_like(x)[:, None, 0, :, :]], dim=1)
        x = self.refine1(x_course)
        dilate_refine = x
        # attention_refine = self.refine2(x_course)
        # x = torch.concat([dilate_refine, attention_refine], axis=1)
        # x = self.concat_conv(x)

        return self.upsample(x)



class GatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, rate=1, activation=nn.ELU):
        super().__init__()
        self.out = out_ch
        self.activation = activation()
        self.sigmoid = nn.Sigmoid()
        pad = (kernel - 1) * rate // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, (kernel, kernel), stride=stride, padding=pad, dilation=rate
        )

    def forward(self, x):
        x = self.conv(x)
        if self.out == 3:
            return x

        features, gating = torch.chunk(x, 2, 1)
        features = self.activation(features)
        gating = self.sigmoid(gating)
        return features * gating


class ContextualAttention(nn.Module):
    def __init__(self, rate=2):
        super().__init__()
        self.rate = rate

    def forward(self, x, mask):
        x_shape = x.shape


model = Inpaint()

model.to("cuda")


with torch.cuda.amp.autocast():
    print(model(torch.ones(32, 5, 256, 256, dtype=torch.float16).to("cuda")).shape)
