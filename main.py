# main.py
import torch
from torch.utils.data import DataLoader
import torchvision

from datasets import TwoViewWrapper, SupervisedWrapper
from transforms import train_transform, test_transform, two_view_transform
from models.resnet import ResNetEncoder
from methods.simsiam import SimSiamMethod
from evaluation.knn import build_feature_bank, knn_evaluate_k1

def main():
    device = torch.device("cpu")
    torch.manual_seed(0)

    # 1) Raw dataset
    tv_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=None)
    tv_test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=None)

    # 2) transform + wrapper
    tr_tf = two_view_transform()
    te_tf = test_transform()

    train_ds = TwoViewWrapper(tv_train, tr_tf)
    test_ds  = SupervisedWrapper(tv_test, te_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    # 4) Encoder + Method
    encoder = ResNetEncoder(num_blocks=[3,3,3]).to(device)
    method = SimSiamMethod(encoder, feat_dim=64, proj_dim=128, hidden_dim=512).to(device)
    method.train()

    # 5) Optimizer
    optimizer = torch.optim.SGD(method.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # 6) 1 epoch run
    running_loss = 0.0
    for step, batch in enumerate(train_loader):
        x1, x2 = batch
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        batch = (x1, x2)

        optimizer.zero_grad(set_to_none=True)
        loss = method(batch)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())

        if step in (0, 5, 10):
            print(f"step={step:04d} loss={loss.item():.4f}")

        if step >= 11:
            break

    print(f"done. avg_loss={running_loss/(step+1):.4f}")


    # 7) kNN evaluation
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    knn_train_ds = SupervisedWrapper(tv_train, te_tf)
    knn_train_loader = DataLoader(
        knn_train_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    
    train_features, train_labels = build_feature_bank(knn_train_loader, encoder, device)
    knn_acc = knn_evaluate_k1(test_loader, encoder, train_features, train_labels, device)
    print(f"kNN (k=1) accuracy: {knn_acc*100:.2f}%")

if __name__ == "__main__":
    main()
