'''
'''
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N, E = 4096, 256
A = torch.randn(E, E, device=device)
x = torch.randn(N, E, device=device)

# Polar Decomposition
U, S, V = torch.linalg.svd(A, full_matrices=False)  # A = U @ torch.diag(S) @ V)
Q = U @ V
P = (V.T * S) @ V

# diff = torch.abs(A - Q@P)
# print(f'Difference - Mean: {diff.mean()}, Std: {diff.std()}, Max: {diff.max()}, Min: {diff.min()}')

dot1 = x @ (x @ A).T / E**0.5
dot2 = x @ (x @ Q).T / E**0.5
diff = torch.abs(dot1 - dot2)
print(f'Difference - Mean: {diff.mean()}, Std: {diff.std()}, Max: {diff.max()}, Min: {diff.min()}')