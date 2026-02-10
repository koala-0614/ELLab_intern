import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5): # [N, D]
    N = z_i.shape[0] # N
    z = torch.cat([z_i, z_j], dim=0) # [2N, D]
    z = F.normalize(z, dim=1) # dot product -> cosine simmilarity

    sim = torch.matmul(z, z.T) / temperature # [2N, D] @ [D, 2N] -> [2N, 2N]


    mask = (~torch.eye(2*N, dtype=bool)).to(z.device)
    exp_sim = torch.exp(sim) * mask
    denominator = exp_sim.sum(dim=1) # [2N]

    idx = torch.arange(N, device=z.device)
    pos_sim = torch.cat([sim[idx, idx + N], sim[idx + N, idx]], dim=0)  # [2N]
    positives = torch.exp(pos_sim)  # [2N]

    loss = -torch.log(positives / denominator)
    return loss.mean() # 2N