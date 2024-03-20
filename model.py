import torch
import torch.nn as nn
import torch.nn.functional as F


class Inpaint(nn.Module):
    def __init__(self):
        super(Inpaint, self).__init__()
        self.relu = nn.ReLU()
        self.Lrelu = nn.LeakyReLU(0.2)
        self.pconv1 = PartialConv(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.pconv2 = PartialConv(
            in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2
        )
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.pconv3 = PartialConv(
            in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2
        )
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.pconv4 = PartialConv(
            in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
        )
        self.batch_norm4_8 = nn.BatchNorm2d(512)
        self.pconv5 = PartialConv(
            in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1
        )
        self.pconv6 = PartialConv(
            in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1
        )
        self.pconv7 = PartialConv(
            in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1
        )
        self.pconv8 = PartialConv(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.pconv9 = PartialConv(
            in_channels=1024, out_channels=512, kernel_size=3, stride=1
        )
        self.pconv10 = PartialConv(
            in_channels=1024, out_channels=512, kernel_size=3, stride=1
        )
        self.pconv11 = PartialConv(
            in_channels=1024, out_channels=512, kernel_size=3, stride=1
        )
        self.pconv12 = PartialConv(
            in_channels=1024, out_channels=512, kernel_size=3, stride=1
        )
        self.pconv13 = PartialConv(
            in_channels=768, out_channels=256, kernel_size=3, stride=1
        )
        self.pconv14 = PartialConv(
            in_channels=384, out_channels=128, kernel_size=3, stride=1
        )
        self.pconv15 = PartialConv(
            in_channels=192, out_channels=64, kernel_size=3, stride=1
        )
        self.pconv16 = PartialConv(
            in_channels=67, out_channels=3, kernel_size=3, stride=1
        )
        self.batch_norm9 = nn.BatchNorm2d(512)
        self.batch_norm15 = nn.BatchNorm2d(64)

    def forward(self, x, mask):
        xin = x
        mask_in = mask
        p1, mask1 = self.pconv1(x, mask)
        p1 = self.relu(p1)

        p2, mask2 = self.pconv2(p1, mask1)
        p2 = self.relu(self.batch_norm2(p2))

        p3, mask3 = self.pconv3(p2, mask2)
        p3 = self.relu(self.batch_norm3(p3))

        p4, mask4 = self.pconv4(p3, mask3)
        p4 = self.relu(self.batch_norm4_8(p4))

        p5, mask5 = self.pconv5(p4, mask4)
        p5 = self.relu(self.batch_norm4_8(p5))

        p6, mask6 = self.pconv6(p5, mask5)
        p6 = self.relu(self.batch_norm4_8(p6))

        p7, mask7 = self.pconv7(p6, mask6)
        p7 = self.relu(self.batch_norm4_8(p7))

        p8, mask8 = self.pconv8(p7, mask7)
        p8 = self.relu(self.batch_norm4_8(p8))

        concat = torch.cat((p7, p8), dim=1)

        mask = torch.cat((mask7, mask8), dim=1)
        p9, mask = self.pconv9(concat, mask)
        p9 = self.Lrelu(self.batch_norm9(p9))

        p9 = self.upsample(p9)

        concat = torch.cat((p6, p9), dim=1)
        mask = self.upsample(mask)

        mask = torch.cat((mask6, mask), dim=1)
        p10, mask = self.pconv10(concat, mask)
        p10 = self.Lrelu(self.batch_norm9(p10))

        p10 = self.upsample(p10)
        concat = torch.cat((p5, p10), dim=1)
        mask = torch.concat((mask5, self.upsample(mask)), dim=1)
        p11, mask = self.pconv11(concat, mask)
        p11 = self.Lrelu(self.batch_norm9(p11))

        p11 = self.upsample(p11)
        concat = torch.cat((p4, p11), dim=1)
        mask = torch.concat((mask4, self.upsample(mask)), dim=1)
        p12, mask = self.pconv12(concat, mask)
        p12 = self.Lrelu(self.batch_norm9(p12))

        p12 = self.upsample(p12)
        concat = torch.cat((p3, p12), dim=1)
        mask = torch.concat((mask3, self.upsample(mask)), dim=1)

        p13, mask = self.pconv13(concat, mask)
        p13 = self.Lrelu(self.batch_norm3(p13))

        p13 = self.upsample(p13)
        concat = torch.cat((p2, p13), dim=1)
        mask = torch.concat((mask2, self.upsample(mask)), dim=1)

        p14, mask = self.pconv14(concat, mask)
        p14 = self.Lrelu(self.batch_norm2(p14))

        p14 = self.upsample(p14)
        concat = torch.cat((p1, p14), dim=1)
        mask = torch.concat((mask1, self.upsample(mask)), dim=1)
        p15, mask = self.pconv15(concat, mask)
        p15 = self.Lrelu(self.batch_norm15(p15))

        p15 = self.upsample(p15)
        concat = torch.cat((xin, p15), dim=1)
        mask = torch.concat((mask_in, self.upsample(mask)), dim=1)

        p16, _ = self.pconv16(concat, mask)

        return p16


class PartialConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.mask_conv.weight = torch.nn.Parameter(
            torch.ones_like(self.mask_conv.weight)
        )
        self.sum = kernel_size**2
        self.out_ch = out_channels

    def forward(self, x, mask):
        with torch.no_grad():
            out_mask = self.mask_conv(mask)
        scale = self.sum / (out_mask + 1e-6)
        out_mask = torch.clamp(out_mask, 0, 1)
        x = self.conv(torch.mul(x, mask))
        bias = self.conv.bias.view(1, self.out_ch, 1, 1)
        output = torch.mul(x - bias, scale) + bias
        output = torch.mul(output, scale)

        return x, out_mask


model = Inpaint()

model.to("cuda")


with torch.cuda.amp.autocast():
    out = model(
        torch.ones(10, 3, 256, 256, dtype=torch.float16).to("cuda"),
        torch.ones(10, 3, 256, 256, dtype=torch.float16).to("cuda"),
    )
    print(out.shape)
