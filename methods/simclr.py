import torch
import torch.nn as nn
from utils.head import projection_head
from utils.contrastive_loss import nt_xent_loss

class SimCLRMethod(nn.Module):
    def __init__(self, encoder, z_dim=128, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.projector = projection_head(encoder, z_dim)
        self.temperature = temperature

    def forward(self, batch):
        x1, x2 = batch

        h1 = self.encoder(x1)  # [B, d]
        h2 = self.encoder(x2)  # [B, d]

        z1 = self.projector(h1)  # [B, z_dim]
        z2 = self.projector(h2)  # [B, z_dim]

        loss = nt_xent_loss(z1, z2, self.temperature)
        return loss
    