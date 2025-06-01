import torch
from torch import nn
from torch.nn import functional as F

class CausalBinnedAttention(nn.Module):
    def __init__(self, embed_dim, heads, bin_num):
        super(CausalBinnedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.bin_num = bin_num

        self.bin_linear = nn.Linear(embed_dim, bin_num, bias=False)
        self.norm = nn.RMSNorm(embed_dim, elementwise_affine=False)
        self.mha = nn.MultiheadAttention(embed_dim, heads, 0, False, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        B, S, C = x.shape

        # Create memory bins
        weights = self.bin_linear(x)
        weights = F.relu(weights) * F.softmax(weights, dim=-1)
        z = self.norm(torch.cumsum(weights.unsqueeze(-1) @ x.unsqueeze(-2), dim=1))

        # Apply MHA to each element with corresponding memory
        x = x.reshape(B*S, 1, C)
        z = z.reshape(B*S, self.bin_num, C)
        y = self.mha(x, z, z, need_weights=False)[0]
        y = y.reshape(B, S, C)
        return y
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # Test HyperEdgeAttention
    B, S, E = 1, 2048, 256
    heads = 8

    x = torch.randn(B, S, E, device=device)
    CBA = CausalBinnedAttention(E, heads, E).to(device)

    # Profile the forward pass
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        y = CBA(x)
        loss = torch.sum(y)
        loss.backward()

    print(prof.key_averages().table(sort_by=f"{device}_time_total", row_limit=12))
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
    print("Total trainable parameters:", round(sum(p.numel() for p in CBA.parameters() if p.requires_grad)/1e6, 2), 'M')
    print("IO is size:", x.element_size() * x.nelement() / 1024 / 1024, 'MB')
    print("I/O has elements: ", round(y.nelement() / 1e6, 2), 'M')