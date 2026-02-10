from torchvision import transforms as T

class TwoTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010))
    ])


def test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010))
    ])


def two_view_transform():
    base = T.Compose([
        T.RandomResizedCrop(32),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010))
    ])
    return TwoTransform(base)
