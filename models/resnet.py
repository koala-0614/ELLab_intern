import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # CIFAR-ResNet style shortcut
        if in_channels != out_channels:
            def shortcut(x):
                x = x[:, :, ::2, ::2]
                diff = out_channels - in_channels
                pad_l = diff // 2
                pad_r = diff - pad_l
                return F.pad(x, (0,0,0,0,pad_l,pad_r))
            self.shortcut = shortcut
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, num_blocks=[9,9,9]):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(16, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(32, 64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

        self.num_features = 64

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x # [B, 64]
