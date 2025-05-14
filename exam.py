import torch
from torch import nn

Q = torch.randint(-3,3,[16,64,128], dtype=torch.float32)
K = torch.randint(-3,3,[16,64,128], dtype=torch.float32)
V = torch.randint(-3,3,[16,64,128], dtype=torch.float32)

multiheadattn = nn.MultiheadAttention(128, 8, 0.2)
a, _ = multiheadattn(Q,K,V)

print(a)
print(a.shape)







