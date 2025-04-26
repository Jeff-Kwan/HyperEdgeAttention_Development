import torch
from torch import nn
import time
import torch.nn.functional as F

device = torch.device("cuda")

patch = 8
in_c = 512
out_c = 16

iterations = 100
input_tensor = torch.randn(256, in_c, 32, 32, device=device)

torch.cuda.empty_cache()

# Benchmark PixelShuffle
pixel_shuffle = nn.Sequential(
    nn.Conv2d(in_c, out_c*patch**2, 1, 1, 0),
    nn.PixelShuffle(patch) if patch > 1 else nn.Identity()).to(device)
start_time = time.time()
torch.cuda.reset_peak_memory_stats(device)
for _ in range(iterations):
    y = pixel_shuffle(input_tensor)
    loss = y.sum()
    loss.backward()
pixel_shuffle_time = (time.time() - start_time)/iterations
pixel_shuffle_peak_memory = torch.cuda.max_memory_allocated(device)

torch.cuda.empty_cache()

# Benchmark ConvTranspose2d
ConvTranspose2d = nn.ConvTranspose2d(in_c, out_c, patch, patch, 0).to(device)
start_time = time.time()
torch.cuda.reset_peak_memory_stats(device)
for _ in range(iterations):
    y = ConvTranspose2d(input_tensor)
    loss = y.sum()
    loss.backward()
ConvTranspose2d_time = (time.time() - start_time)/iterations
ConvTranspose2d_peak_memory = torch.cuda.max_memory_allocated(device)

print(f"PixelShuffle time: {pixel_shuffle_time * 1000:.3f} ms")
print(f"PixelShuffle peak memory: {pixel_shuffle_peak_memory / 1024**2:.3f} MB")
print(f"ConvTranspose2d time: {ConvTranspose2d_time * 1000:.3f} ms")
print(f"ConvTranspose2d peak memory: {ConvTranspose2d_peak_memory / 1024**2:.3f} MB")