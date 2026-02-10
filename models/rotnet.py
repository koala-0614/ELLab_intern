import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
  
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = BasicBlock(in_channels, out_channels, kernel_size=kernel_size)
        self.conv2 = BasicBlock(out_channels, out_channels, kernel_size=1)
        self.conv3 = BasicBlock(out_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))
    
class RotNetEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 96, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32 -> 16

        self.block2 = ConvBlock(96, 192, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16 -> 8

        self.block3 = ConvBlock(192, 192, kernel_size=3)
        self.block4 = ConvBlock(192, 192, kernel_size=3)

        # Rotation classifier head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.rot_fc = nn.Linear(192, num_rotations)
        self.num_features = 192

    def forward(self, x, return_block2 = False):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        feature_b2 = x # [192, 8, 8]
        if return_block2:
            return feature_b2

        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        # x = self.rot_fc(x)

        return x

