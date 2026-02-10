import torch
import torch.nn.functional as F

@torch.no_grad()
def build_feature_bank(train_loader, encoder, device):
    features, labels = [], []

    encoder.eval()
    for batch in train_loader:
        x, y = batch
        x = x.to(device)
        h = encoder(x)
        h = F.normalize(h, dim=1)
        features.append(h.cpu())
        labels.append(y)

    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def knn_evaluate_k1(test_loader, encoder, train_features, train_labels, device):
    correct, total = 0, 0
    encoder.eval()

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        h = encoder(x)
        h = F.normalize(h, dim=1)

        sim = torch.mm(h, train_features.T)
        idx = sim.argmax(dim=1)
        pred = train_labels[idx]

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total