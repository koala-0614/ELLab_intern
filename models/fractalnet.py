import torch
import torch.nn as nn
import random


def local_drop(stack, p, training):
    S = stack.shape[0]  # 2
    if not training or p == 0:
        return stack

    mask = torch.rand(S, device=stack.device) > p  # [2]

    # 최소 한 개 branch 남겨두기
    if mask.sum() == 0:
        keep_idx = torch.randint(0, S, (1,), device=stack.device)
        mask[keep_idx] = True

    mask = mask.view(S, 1, 1, 1, 1).expand_as(stack)
    # view: [2,1,1,1,1], expand_as: [2, 100, 64, 32, 32]
    return stack * mask


def get_column_indices(C):
    seq = []

    def visit(level):
        seq.append(level)
        if level == 1:
            return
        visit(level - 1)
        visit(level - 1)

    visit(C)
    cols = {k: [] for k in range(1, C + 1)}
    for i, c in enumerate(seq):
        cols[c].append(i)

    # cols : [col_idx, B, C, H, W]
    return cols


def global_drop(stack, num_col, active_col, training):
    # active_col는 남길 컬럼 하나
    if not training:
        return stack

    S = stack.shape[0]  # stack된 텐서 수
    cols = get_column_indices(num_col)
    keep_branches = cols[active_col]

    mask = torch.zeros(S, device=stack.device)
    mask[keep_branches] = 1.0
    mask = mask.view(S, 1, 1, 1, 1).expand_as(stack)

    return stack * mask


class Join(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C

    def forward(self, out, mode, p=None):
        if self.C == 1:
            return out[0]
        else:
            stack = torch.stack(out, dim=0)  # [S, B, C, H, W]
            training = self.training

            if mode == "local":
                stack = local_drop(stack, p=p, training=training)

            out = torch.mean(stack, dim=0)  # [B, C, H, W]
            return out


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unit = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.unit(x)


class FractalBlock(nn.Module):
    def __init__(self, C, in_channels, out_channels, drop_path):
        super().__init__()

        self.C = C
        self.max_depth = 2 ** (C - 1)

        # left path
        self.path1 = ConvUnit(in_channels, out_channels)

        if C > 1:
            # sub path
            self.path2 = nn.ModuleList([
                FractalBlock(C - 1, in_channels, out_channels, drop_path),
                FractalBlock(C - 1, out_channels, out_channels, drop_path),
            ])

        self.join = Join(C)

    # x -> [y1, y2, ..., yC]
    def forward(self, x, mode, p, active_col, join_index):
        if self.C == 1:
            return self.path1(x)

        left = self.path1(x)

        # path2 module 통과하면서 쭉 join 내부에서 stack해주기
        sub1 = self.path2[0](x, mode, p, active_col, join_index + 1)
        sub2 = self.path2[1](sub1, mode, p, active_col, join_index + 1)

        # 마지막 join 직전에 global drop 적용
        if join_index == (self.max_depth - 1) and mode == "global":
            sub2 = global_drop(
                sub2,
                num_col=self.C,
                active_col=active_col,
                training=self.training,
            )

        out = self.join([left, sub2], mode, p)
        return out


class FractalNetEncoder(nn.Module):
    def __init__(self, columns=4, drop_path=0.15):
        super().__init__()

        self.columns = columns
        self.drop_path = drop_path

        self.block1 = FractalBlock(columns, 3, 64, drop_path)
        self.block2 = FractalBlock(columns, 64, 128, drop_path)
        self.block3 = FractalBlock(columns, 128, 256, drop_path)
        self.block4 = FractalBlock(columns, 256, 512, drop_path)
        self.block5 = FractalBlock(columns, 512, 512, drop_path)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_features = 512

    def forward(self, x):
        if self.training:
            if random.random() < 0.5:
                mode = "local"
                active_col = None
            else:
                mode = "global"
                active_col = random.randint(1, self.columns)
            p = self.drop_path
        else:
            # 평가 / 추론은 결정적으로
            mode = "none"
            active_col = None
            p = 0.0

        x = self.block1(x, mode, p, active_col, join_index=0)
        x = self.pool(x)

        x = self.block2(x, mode, p, active_col, join_index=0)
        x = self.pool(x)

        x = self.block3(x, mode, p, active_col, join_index=0)
        x = self.pool(x)

        x = self.block4(x, mode, p, active_col, join_index=0)
        x = self.pool(x)

        x = self.block5(x, mode, p, active_col, join_index=0)

        x = self.gap(x)
        x = x.flatten(1)
        return x
