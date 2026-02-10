import torch
import torch.nn as nn
from utils.head import projection_head_v2

def l2_normalize(x, eps=1e-12):
    return x / (x.norm(dim=1, keepdim=True) + eps)

class SimSiamMethod(nn.Module):
    def __init__(self, encoder, feat_dim=64, proj_dim=128, hidden_dim=512):
        super().__init__()
        self.encoder = encoder
        self.proj = projection_head_v2(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        self.pred = projection_head_v2(in_dim=proj_dim, hidden_dim=hidden_dim, out_dim=proj_dim)

    def encode(self, x):
        y = self.encoder(x)     # [B, 64]
        z = self.proj(y)        # [B, 128]
        p = self.pred(z)        # [B, 128]
        return p, z

    def loss_fn(self, p, z):
        z = z.detach()
        p = l2_normalize(p)
        z = l2_normalize(z)
        return -(p * z).sum(dim=1).mean()

    def compute_loss(self, v1, v2):
        p1, z1 = self.encode(v1)
        p2, z2 = self.encode(v2)
        return 0.5 * self.loss_fn(p1, z2) + 0.5 * self.loss_fn(p2, z1)

        
    def forward(self, batch):
        x1, x2 = batch
        loss = self.compute_loss(x1, x2)
        return loss