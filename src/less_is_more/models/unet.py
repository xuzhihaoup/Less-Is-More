"""U-Net implementation used in the experiments."""

import torch
import torch.nn as nn

__all__ = ["UNet"]


def cut_img(tensor_img, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor_img.size()[2]
    delta = tensor_size - target_size
    delta_left = delta // 2
    delta_right = delta - delta_left
    return tensor_img[:, :, delta_left:tensor_size - delta_right, delta_left:tensor_size - delta_right]


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, normalize_input=True, return_features=True):
        super().__init__()
        self.normalize_input = normalize_input
        self.return_features = return_features

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(in_channels, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.up_conv_1 = double_conv(1024, 512)
        self.up_conv_2 = double_conv(512, 256)
        self.up_conv_3 = double_conv(256, 128)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, image):
        if self.normalize_input:
            image = image / 255.0

        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        x = self.up_trans_1(x9)
        y = cut_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], dim=1))

        x = self.up_trans_2(x)
        y = cut_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], dim=1))

        x = self.up_trans_3(x)
        y = cut_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], dim=1))

        x = self.up_trans_4(x)
        y = cut_img(x1, x)
        feature_map = self.up_conv_4(torch.cat([x, y], dim=1))
        logits = self.out(feature_map)

        if self.return_features:
            return logits, feature_map
        return logits

