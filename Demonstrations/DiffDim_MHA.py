import torch
from torch import nn

device = "cuda"

B, N1, E1 = 4,  512, 256
N2, E2 =        512, 2048


# Create random tensors on the appropriate device
x1 = torch.randn(B, N1, E1, device=device)
x2 = torch.randn(B, N2, E2, device=device)

# Initialize the MultiheadAttention and move it to device
MHA = nn.MultiheadAttention(embed_dim=E1, num_heads=4, batch_first=True, kdim=E2, vdim=E2).to(device)

# Reset CUDA memory stats before execution
torch.cuda.reset_peak_memory_stats(device)
start_memory = torch.cuda.memory_allocated(device)

# Perform the operation
out = MHA(x1, x2, x2, need_weights=False)
torch.cuda.synchronize()  # Wait for all operations to finish
peak_memory = torch.cuda.max_memory_allocated(device)
end_memory = torch.cuda.memory_allocated(device)

print("Output shape:", out[0].shape)
print(f"Initial allocated memory (MB): {start_memory / (1024**2):.2f}")
print(f"Memory allocated after operation (MB): {end_memory / (1024**2):.2f}")
print(f"Peak memory during operation (MB): {peak_memory / (1024**2):.2f}")