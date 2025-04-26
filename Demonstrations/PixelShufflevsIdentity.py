import torch
from torch import nn
import time
import torch.nn.functional as F


patch = 4
pixel_unshuffle = nn.PixelUnshuffle(1)
Identity = nn.Identity()

device = torch.device("cuda")

iterations = 1000

# Create a random tensor
input_tensor = torch.randn(64, 128, 128, 128, device=device)

print(f"Input shape: {input_tensor.shape}")
print(f"PixelUnshuffle shape: {pixel_unshuffle(input_tensor).shape}")
print(f"Identity shape: {Identity(input_tensor).shape}")
print()

# Benchmark PixelUnshuffle
start_time = time.time()
for _ in range(iterations):
    _ = pixel_unshuffle(input_tensor)
pixel_unshuffle_time = (time.time() - start_time)/iterations

# Benchmark Identity
start_time = time.time()
for _ in range(iterations):
    _ = Identity(input_tensor)
Identity_time = (time.time() - start_time)/iterations

print(f"PixelUnshuffle time: {pixel_unshuffle_time * 1000:.3f} ms")
print(f"Identity time: {Identity_time * 1000:.3f} ms")