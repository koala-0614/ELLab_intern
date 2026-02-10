import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.head import projection_head_v2
import math

def l2_normalize(x, eps=1e-12):
    return x / (x.norm(dim=1, keepdim=True) + eps)

def byol_tau(step, total_steps, tau_base=0.99):
    return 1 - (1 - tau_base) * (math.cos(math.pi * step / total_steps) + 1) / 2

class BYOLMethod(nn.Module):
    def __init__(self, encoder, feat_dim=64, proj_dim=128, hidden_dim=512):
        super().__init__()
        self.online_encoder = encoder
        self.online_proj = projection_head_v2(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        self.online_pred = projection_head_v2(in_dim=proj_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_proj = copy.deepcopy(self.online_proj)

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_proj.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau: float):
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1 - tau)
        for p_t, p_o in zip(self.target_proj.parameters(), self.online_proj.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1 - tau)

    def forward_online(self, x):
        y = self.online_encoder(x)   # [B, 64]
        z = self.online_proj(y)      # [B, 128]
        p = self.online_pred(z)      # [B, 128]
        return p

    @torch.no_grad()
    def forward_target(self, x):
        y = self.target_encoder(x)   # [B, 64]
        z = self.target_proj(y)      # [B, 128]
        return z

    def loss_fn(self, p, z_t):
        p = l2_normalize(p)
        z_t = l2_normalize(z_t)
        return 2 - 2 * (p * z_t).sum(dim=1).mean()

    def compute_loss(self, v1, v2):
        p1 = self.forward_online(v1)
        p2 = self.forward_online(v2)
        with torch.no_grad():
            zt1 = self.forward_target(v1)
            zt2 = self.forward_target(v2)
        return self.loss_fn(p1, zt2) + self.loss_fn(p2, zt1)

    def forward(self, batch):
        v1, v2 = batch
        return self.compute_loss(v1, v2)