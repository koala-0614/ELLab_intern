import torch
import torch.nn as nn
from utils.head import classification_head



class SupervisedLearningMethod(nn.Module):
    def __init__(self, encoder, n_cls):
        super().__init__()
        self.encoder = encoder
        self.head = classification_head(encoder, n_cls)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        x, y = batch

        h = self.encoder(x)
        logit = self.head(h)
        loss = self.criterion(logit, y)
        return loss