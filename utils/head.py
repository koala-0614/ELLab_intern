import torch.nn as nn

def classification_head(encoder, n_cls):
    d = encoder.num_features
    head = nn.Linear(d, n_cls)
    return head

def projection_head(encoder, z_dim=128):
    d = encoder.num_features
    head = nn.Sequential(
        nn.Linear(d, d),
        nn.ReLU(),
        nn.BatchNorm1d(d),
        nn.Linear(d, z_dim)
    )
    return head

def projection_head_v2(in_dim=None, hidden_dim=None, out_dim=None):
    if hidden_dim is None:
        hidden_dim = in_dim
    head = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Linear(hidden_dim, out_dim)
    )
    return head
