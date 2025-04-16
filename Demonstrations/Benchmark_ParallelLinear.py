import torch
from torch import nn
import torch.nn.functional as F

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.ParallelLinear import ParallelLinear
import time
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
batch_size = 32
num_vectors = 1024
in_features = 256
out_features = 1024
num_iterations = 100

# Input tensor on device
x = torch.randn(num_vectors, batch_size, in_features, device=device)

############################
# Benchmark nn.Linear
############################
linear = nn.Linear(in_features, out_features).to(device)
# Warm-up
_ = linear(x)
torch.cuda.synchronize()

# Reset peak memory and record time
torch.cuda.reset_peak_memory_stats()
start_linear = time.time()
for _ in range(num_iterations):
    y = linear(x)
    loss = torch.sum(y)
    loss.backward()
    torch.cuda.synchronize()
end_linear = time.time()
linear_time = (end_linear - start_linear) / num_iterations
linear_peak_mem = torch.cuda.max_memory_allocated()

print(f"nn.Linear Average Forward Time per Iteration: {linear_time*1000:.3f}ms")
print(f"nn.Linear Peak CUDA Memory: {linear_peak_mem / (1024**2):.2f} MB")

############################
# Benchmark ParallelLinear
############################
parallel_linear = ParallelLinear(in_features, out_features, num_vectors).to(device)
# Warm-up
_ = parallel_linear(x)
torch.cuda.synchronize()

# Reset peak memory and record time
torch.cuda.reset_peak_memory_stats()
start_parallel = time.time()
for _ in range(num_iterations):
    y = parallel_linear(x)
    loss = torch.sum(y)
    loss.backward()
    torch.cuda.synchronize()
end_parallel = time.time()
parallel_time = (end_parallel - start_parallel) / num_iterations
parallel_peak_mem = torch.cuda.max_memory_allocated()

print(f"ParallelLinear Average Forward Time per Iteration: {parallel_time*1000:.3f}ms")
print(f"ParallelLinear Peak CUDA Memory: {parallel_peak_mem / (1024**2):.2f} MB")
