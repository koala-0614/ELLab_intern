from random import random
import torch
import torch.nn as nn
from utils.head import classification_head

class RotNetMethod(nn.Module):
    def __init__(self, encoder, num_rotations=4):
        super().__init__()
        self.encoder = encoder
        self.rot_fc = classification_head(encoder, n_cls=num_rotations)
        self.criterion = nn.CrossEntropyLoss()

    def make_rotation_batch(self, x):
        # x: (B, C, H, W)
        B = x.size(0)
        device = x.device
        rot_id = torch.randint(0, self.num_rotations, (B,), device=device)  # [B] on same device
        x_rot = x.clone()

        for k in range(1, self.num_rotations):
            m = (rot_id == k)
            if m.any():
                x_rot[m] = torch.rot90(x[m], k=k, dims=(2, 3))  # H,W

        return x_rot, rot_id
    
    def forward(self, batch):
        x, _ = batch
        x_rot, rot_id = self.make_rotation_batch(x)
        h = self.encoder(x_rot)
        logits = self.rot_fc(h)
        loss = self.criterion(logits, rot_id)
        return loss
    
