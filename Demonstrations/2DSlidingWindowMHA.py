'''2D Sliding Window attention with flex-attention'''
import torch
from torch.nn.attention import flex_attention
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.reset_peak_memory_stats()

def SW_mask2D(b, h, q_idx, kv_idx):
    q_x, q_y = q_idx // W, q_idx % W
    kv_x, kv_y = kv_idx // W, kv_idx % W
    return ((q_x - kv_x).abs() <= WINDOW) & ((q_y - kv_y).abs() <= WINDOW)

def SW_mask2D_score(score, b, h, q_idx, kv_idx):
    q_x, q_y = q_idx // W, q_idx % W
    kv_x, kv_y = kv_idx // W, kv_idx % W
    return torch.where(
        ((q_x - kv_x).abs() <= WINDOW) & ((q_y - kv_y).abs() <= WINDOW),
        score,
        float('-inf')
    )

# Create dummy data for query, key, and value tensors
B = 4
heads = 8
H, W = 128, 128
embed_dim = 64
WINDOW = 8

x = torch.randn(B, heads, H*W, embed_dim).to(device)
QKV = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=False).to(device)
query, key, value = QKV(x).chunk(3, dim=-1)

flex_attn = torch.compile(flex_attention.flex_attention)

# Create the block mask using the sliding window mask modification function
block_mask = flex_attention.create_block_mask(SW_mask2D, B, heads, H*W, H*W, 
                                              device=x.device, _compile=True)

# Warm‑up runs (optional, helps stabilize CUDA performance)
for _ in range(10):
    _ = F.scaled_dot_product_attention(query, key, value)
    _ = flex_attn(query, key, value, block_mask=block_mask)
    # _ = flex_attn(query, key, value, score_mod=SW_mask2D_score)

peak_memory_warmup = torch.cuda.max_memory_allocated(x.device) / (1024 * 1024)
print(f"Peak Memory during warmup: {peak_memory_warmup:.2f} MB")

# Measure vanilla scaled dot‑product attention
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
x = torch.randn(B, heads, H*W, embed_dim).to(device)
QKV = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=False).to(device)
query, key, value = QKV(x).chunk(3, dim=-1)
start_vanilla = torch.cuda.Event(enable_timing=True)
end_vanilla   = torch.cuda.Event(enable_timing=True)
start_vanilla.record()
out_vanilla = F.scaled_dot_product_attention(query, key, value)
loss = out_vanilla.sum()
loss.backward()
end_vanilla.record()
torch.cuda.synchronize()
time_vanilla = start_vanilla.elapsed_time(end_vanilla)
peak_memory_vanilla = torch.cuda.max_memory_allocated(x.device) / (1024 * 1024)  # in MB

# Measure flex‑attention
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
x = torch.randn(B, heads, H*W, embed_dim).to(device)
QKV = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=False).to(device)
query, key, value = QKV(x).chunk(3, dim=-1)
start_flex = torch.cuda.Event(enable_timing=True)
end_flex   = torch.cuda.Event(enable_timing=True)
start_flex.record()
out_flex = flex_attn(query, key, value, block_mask=block_mask)
# out_flex = flex_attn(query, key, value, score_mod=SW_mask2D_score)
loss = out_flex.sum()
loss.backward()
end_flex.record()
torch.cuda.synchronize()
time_flex = start_flex.elapsed_time(end_flex)
peak_memory_flex = torch.cuda.max_memory_allocated(x.device) / (1024 * 1024)  # in MB

print(f"SDPA: {time_vanilla:.2f} ms, Peak Memory: {peak_memory_vanilla:.2f} MB")
print(f"Flex: {time_flex:.2f} ms, Peak Memory: {peak_memory_flex:.2f} MB")