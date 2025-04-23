'''2D Sliding Window attention with flex-attention'''
import torch
from torch.nn.attention import flex_attention
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def SW_mask2D(b, h, q_idx, kv_idx):
    q_x, q_y = q_idx // W, q_idx % W
    kv_x, kv_y = kv_idx // W, kv_idx % W
    return ((q_x - kv_x).abs() <= WINDOW) & ((q_y - kv_y).abs() <= WINDOW)

# Create dummy data for query, key, and value tensors
B = 1
heads = 1
H, W = 96, 96
embed_dim = 64
WINDOW = 8

query = torch.randn(B, heads, H*W, embed_dim).to(device)
key = torch.randn(B, heads, H*W, embed_dim).to(device)
value = torch.randn(B, heads, H*W, embed_dim).to(device)

# Create the block mask using the sliding window mask modification function
block_mask = flex_attention.create_block_mask(SW_mask2D, B, heads, H*W, H*W, 
                                              device=query.device, BLOCK_SIZE=64)

# Warm‑up runs (optional, helps stabilize CUDA performance)
for _ in range(10):
    _ = F.scaled_dot_product_attention(query, key, value)
    _ = flex_attention.flex_attention(query, key, value, block_mask=block_mask)

# Measure vanilla scaled dot‑product attention
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start_vanilla = torch.cuda.Event(enable_timing=True)
end_vanilla   = torch.cuda.Event(enable_timing=True)
start_vanilla.record()
out_vanilla = F.scaled_dot_product_attention(query, key, value)
end_vanilla.record()
torch.cuda.synchronize()
time_vanilla = start_vanilla.elapsed_time(end_vanilla)
peak_memory_vanilla = torch.cuda.max_memory_allocated(query.device) / (1024 * 1024)  # in MB

# Measure flex‑attention
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start_flex = torch.cuda.Event(enable_timing=True)
end_flex   = torch.cuda.Event(enable_timing=True)
start_flex.record()
out_flex = flex_attention.flex_attention(query, key, value, block_mask=block_mask)
end_flex.record()
torch.cuda.synchronize()
time_flex = start_flex.elapsed_time(end_flex)
peak_memory_flex = torch.cuda.max_memory_allocated(query.device) / (1024 * 1024)  # in MB

print(f"SDPA: {time_vanilla:.2f} ms, Peak Memory: {peak_memory_vanilla:.2f} MB")
print(f"Flex: {time_flex:.2f} ms, Peak Memory: {peak_memory_flex:.2f} MB")