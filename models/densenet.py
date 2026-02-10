import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(p=self.drop_rate)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        if self.drop_rate > 0:
            out = self.dropout(out)
        out = torch.cat([x, out], dim=1)
        return out
    
class DenseLayer_bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate=0.2):
        super().__init__()
        mid_channels = 4 * growth_rate
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(p=self.drop_rate)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.drop_rate > 0:
            out = self.dropout(out)
        out = torch.cat([x, out], dim=1)
        return out
    
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, drop_rate=0.2, bottleneck=False):
        super().__init__()
        self.layers = nn.ModuleList()
        layer_cls = DenseLayer_bottleneck if bottleneck else DenseLayer
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(layer_cls(layer_in_channels, growth_rate, drop_rate=drop_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, theta=1):
        super().__init__()
        out_channels = int(in_channels * theta)

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.pool(self.conv(self.bn(x)))
        return out


class DenseNetEncoder(nn.Module):
    def __init__(self, bottleneck=False, growth_rate=12, theta=1.0):
        super().__init__()
        self.bottleneck = bottleneck
        self.growth_rate = growth_rate
        self.theta = theta

        # 초기 convolution
        if not bottleneck:
            num_init_features = 16
        else:
            num_init_features = 2 * growth_rate

        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.relu = nn.ReLU()

        # Dense Block 1
        self.block1 = DenseBlock(num_layers=12, in_channels=num_init_features, growth_rate=growth_rate)
        num_features = num_init_features + 12 * growth_rate
        self.trans1 = TransitionLayer(in_channels=num_features, theta=theta)

        # Dense Block 2
        num_features = int(num_features * theta)
        self.block2 = DenseBlock(num_layers=12, in_channels=num_features, growth_rate=growth_rate)
        num_features = num_features + 12 * growth_rate
        self.trans2 = TransitionLayer(in_channels=num_features, theta=theta)

        # Dense Block 3
        num_features = int(num_features * theta)
        self.block3 = DenseBlock(num_layers=12, in_channels=num_features, growth_rate=growth_rate)
        num_features = num_features + 12 * growth_rate

        # Final layers, fc 제거
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_features = num_features

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
