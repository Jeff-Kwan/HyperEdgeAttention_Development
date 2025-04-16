'''
This script demonstrates the equivalence of two dot product calculations:
(Ax)T (Bx) = xTAT Bx = xT (ATB)x
1. Ax @ Bx.T
2. ATBx @ x.T

Even if A and B are down-projections
Thus the multihead attention mechanism Q, K projections can be merged
'''
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.use_deterministic_algorithms(True)  
# Oops @ is not deterministic because it uses CuBLAS and we have CUDA >= 10.2.

N, E = 4096, 256
A = torch.randn(E, E//4, device=device)
B = torch.randn(E, E//4, device=device)
x = torch.randn(N, E, device=device)

Ax = x @ A
Bx = x @ B
ATBx = x @ A @ B.T
print(f'Ax shape: {Ax.shape}, Bx shape: {Bx.shape}, ATBx shape: {ATBx.shape}')

dot1 = Ax @ Bx.T / E**0.5
dot2 = ATBx @ x.T / E**0.5

diff = torch.abs(dot1 - dot2)
print(f'Difference - Mean: {diff.mean()}, Std: {diff.std()}, Max: {diff.max()}, Min: {diff.min()}')