import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat

class PatchEmbedding(nn.Module):  # input [B, 3, 32, 32]
    def __init__(self, in_channels=3, patch_size=4, emb_size=48, img_size=32):
        super().__init__() # emb_size = 3*4*4
        self.patch_size = patch_size

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, patch_size, stride=patch_size), # [B, 48, 8, 8]
            Rearrange('b e (h) (w) -> b (h w) e') # [B, 64, 48]
        )

        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size)) # [1, 1, 48]
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size)) # [65, 48]

    def forward(self, x):
        b = x.shape[0] # batch size B
        x = self.projection(x) # [b, 64, 48]
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b) # [1, 1, 48] -> [b, 1, 48]
        x = torch.cat([cls_tokens, x], dim=1) # [b, 65, 48]
        x += self.positions
        return x

class MultiHeadAttention(nn.Module): # input: [b,65,48]
    def __init__(self, emb_size = 48, num_heads = 4, dropout = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads # d_head=(48/4)=12
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3) # 여기서 d는 헤드당 차원
        # self.qkv: [b,65,144]
        # rearrange: [b,65,(4*12*3)] -> qkv: [3,b,4,65,12]

        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # queries, keys, values: [b,4,65,12]

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # [b,4,65,65]

        d_head = self.emb_size // self.num_heads
        scaling = d_head ** (1/2)
        att = F.softmax(energy/scaling, dim=-1) # [b,4,65,65]

        att = self.att_drop(att)

        out = torch.einsum('bhqk, bhkd -> bhqd ', att, values)
        # [b,4,65,65],[b,4,65,12] -> [b,4,65,12]
        out = rearrange(out, "b h n d -> b n (h d)") # [b,65,48]
        out = self.projection(out) # [b,65,48]
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 2, drop_p = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size = 48, drop_p = 0., forward_expansion = 2, forward_drop_p = 0., ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ViTEncoder(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 4, emb_size = 48, img_size = 32, depth = 6, **kwargs):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer_encoder = TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        # self.classification_head = ClassificationHead(emb_size, n_classes)
        self.num_features = emb_size

    def forward(self, x):
        x = self.patch_embedding(x)  # [b, 65, 48]
        x = self.transformer_encoder(x)  # [b, 65, 48]
        # x = self.classification_head(x)  # [b, n_classes]
        return x[:, 0, :]  # cls token [b, 48]
    

