import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.head import projection_head

class MoCoMethod(nn.Module):
    def __init__(self, encoder, dim=128, K=4096, m=0.999, T=0.2):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.dim = dim

        # query encoder
        self.encoder_q = encoder
        self.projector_q = projection_head(self.encoder_q, z_dim=dim)

        # key encoder
        self.encoder_k = copy.deepcopy(encoder)  
        self.projector_k = projection_head(self.encoder_k, z_dim=dim)
        
        # initialize + stop grad for key
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False
        for q, k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False

        # create queue: (dim, K)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data = k.data * self.m + q.data * (1.0 - self.m)
        for q, k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            k.data = k.data * self.m + q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # keys: (B, dim)
        B = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        self.queue[:, ptr:ptr + B] = keys.T
        ptr = (ptr + B) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, batch):
        x_q, x_k = batch
        # 1) query: grad update O
        q = self.encoder_q(x_q)          # (B, 512)
        q = self.projector_q(q)          # (B, dim)
        q = F.normalize(q, dim=1)

        # 2) key: grad update X, momentum update
        with torch.no_grad():
            self._momentum_update()
            k = self.encoder_k(x_k)
            k = self.projector_k(k)
            k = F.normalize(k, dim=1)

        # 3) logits: positive + negatives
        # positive: (B, 1)
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(1)
        # negative: (B, K)
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())

        logits = torch.cat([l_pos, l_neg], dim=1)  # (B, 1+K)
        logits /= self.T

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # 4) update queue
        self._dequeue_and_enqueue(k)

        # 5) InfoNCE = cross-entropy
        loss = F.cross_entropy(logits, labels)
        return loss