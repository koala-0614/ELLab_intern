[WrapperDataset] → [DataLoader] → [Method] → [Encoder] → [Head] → [Loss]

# WrapperDataset
## Supervised / Classification
```python
# Input from BaseDataset
(x_raw, y_raw)

# Inside Wrapper
x = train_transform(x_raw)

# Output (tuple)
(x, y)
# x : Tensor[C,H,W]
# y : int (class label)
```
## RotNet
```python
# Input from BaseDataset
(x_raw, y_raw) # y_raw는 버려짐

# Inside Wrapper
x = train_transform(x_raw)
x_rot, rot_id = random_rotate(x)

# Output (tuple)
(x_rot, rot_id)
# x_rot : Tensor[C,H,W]
# rot_id: int in {0,1,2,3}
```
wrapper에서 label 바꿔주기

## TwoViews (SimCLR/MoCo)
```python
# Input from BaseDataset
(x_raw, y_raw)

# Inside Wrapper
x1, x2 = two_view_transform(x_raw)

# Output (tuple)
(x1, x2)
# x1 : Tensor[C,H,W]
# x2 : Tensor[C,H,W]
```  
# Transform
train, test, two_view 총 3가지
## train
```python
T.RandomResizedCrop(32),
T.RandomHorizontalFlip(),
T.RandomGrayscale(p=0.2),
T.ToTensor(),
T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
```
## test
```python
T.ToTensor(),
T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
```
## two_view
```python
# input 
x_raw 

# inside
transform = T.Compose([
    T.RandomResizedCrop(32),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
x1 = transform(x_raw)
x2 = transform(x_raw)

# output
(x1, x2)
```



# DataLoader
```python
# Input (from Wrapper)
list_of_samples = [
  (x, y),
  (x, y),
  ...
] # x: Tensor[C,H,W], y: int
or
list_of_samples = [
  (x1, x2),
  (x1, x2),
  ...
] # x1, x2: Tensor[C,H,W]

# Output

## Supervised / RotNet
batch = (x, y)

x: Tensor[B,C,H,W]
y: Tensor[B]          # dtype: torch.int64


## SimCLR / MoCo
# Output
batch = (x1, x2)

x1: Tensor[B,C,H,W]
x2: Tensor[B,C,H,W]
```
# Method
batch를 받아서 loss 계산함  
## Classification / RotNet 
```python
# Input Batch (from DataLoader)
(x, y)

# Inside
h = encoder(x)
logit = head(h)
loss = CE(logit, y)

# Output
loss
```
## SimCLR
```python
# Input
(x1, x2)

# Inside
h1 = encoder(x1)
h2 = encoder(x2)

z1 = projector(h1)
z2 = projector(h2)

loss = NTXent(z1, z2, T)

# Output
loss
```
## MoCo
```python
# Input
(x1, x2)

# Inside
q = enc_q(x1) → proj_q → normalize
k = enc_k(x2) → proj_k → normalize  # no grad

logits = [q·k_pos, q·queue] / T
loss = CE(logits, zeros)

update_queue(k)
```
## BYOL
```python
(x1, x2)

# Inside
# online branch
p1 = pred(proj(enc_o(x1))) → normalize
p2 = pred(proj(enc_o(x2))) → normalize

# target branch (no grad, EMA)
z1 = proj(enc_t(x1)) → normalize
z2 = proj(enc_t(x2)) → normalize

# loss (symmetric, no negatives)
loss = ||p1 - z2||^2 + ||p2 - z1||^2

# update
optimize(enc_o, proj, pred)
enc_t, proj_t ← EMA(enc_o, proj_o)
```
# SimSiam
```python
# Input
(x1, x2)

# Inside
# shared siamese encoder (same weights for both views)
z1 = proj(enc(x1)) → normalize          # projection
z2 = proj(enc(x2)) → normalize

# predictor (only on the "p" side)
p1 = pred(z1) → normalize               # prediction
p2 = pred(z2) → normalize

# stop-gradient on target z
# D(p, z) = - <p, z>  (negative cosine similarity)
loss = 0.5 * D(p1, stopgrad(z2)) + 0.5 * D(p2, stopgrad(z1))

# update
optimize(enc, proj, pred)
```
# Encoder (model)
ResNet, FractalNet, DenseNet, Vit 모두 공통
```python
input  : Tensor[B,C,H,W]
output : Tensor[B,d]
attr   : encoder.num_features == d
```
딱 표현 추출 까지만 담당함
# Head
## Classification / RotNet
```python
input  : h [B,d]
output : logits [B,n_cls]
```
## SimCLR / MoCo projector
```python
input  : h [B,d]
output : z [B,z]
```
# Loss
scalar로 출력
## Supervised / RotNet
```python
 CrossEntropy(logits, y) → scalar
```
## SimCLR
```python
NT-Xent(z1, z2, T) → scalar
```
## MoCo
```python
CE([l_pos, l_neg], label=0) → scalar
```
# file structure
```
root/  
  main.py  
 
  datasets.py # Wrapper datasets (Supervised / RotNet / TwoViews)  
 
  transforms.py # train / test / simclr transform + TwoTransform  

  models/ # encoder
    __init__.py  
    resnet.py
    fractalnet.py  
    vit.py  
    densenet.py  

  methods/  
    __init__.py  
    supervised_learning.py  
    rotnet.py  
    simclr.py
    moco.py
    BYOL.py
    SimSiam.py  

  evaluation/
    knn_evaluation.py
    
  utils/
   contrastive_loss.py
   head.py
```
