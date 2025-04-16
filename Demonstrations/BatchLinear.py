import torch
from torch import nn

B, N, C = 16, 1024, 256

x = torch.randn(B, N, C).cuda()
linear = nn.Linear(C, C*4).cuda()

def batchlinear(x, w):
    # return torch.einsum('bnc,ncd->bnd', x, w)
    return torch.matmul(x.transpose(0,1), w).transpose(0,1)

# warm up
y = linear(x)
w = torch.randn(N, C, C*4).cuda()
y = batchlinear(x, w)

# Measure memory peak for the custom batchlinear function
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
_ = batchlinear(x, w)
torch.cuda.synchronize()
batchlinear_peak = torch.cuda.max_memory_allocated()

# Measure memory peak for nn.Linear
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
_ = linear(x)
torch.cuda.synchronize()
linear_peak = torch.cuda.max_memory_allocated()

print("Custom batchlinear peak CUDA memory:", batchlinear_peak/1e6, "MB")
print("nn.Linear peak CUDA memory:   ", linear_peak/1e6, "MB")