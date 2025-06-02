import torch
from torch import nn
from torch.nn import functional as F

class HyperEdgeAttention(nn.Module):
    '''HyperEdge partitioning and attention.'''
    def __init__(self, channels, edges, heads, bias=False):
        super(HyperEdgeAttention, self).__init__()
        assert channels%heads == 0, f"Channels {channels} not divisble by heads {heads}"
        self.edges = edges
        self.heads = heads
        self.h_dim = channels // heads
        self.hyperedge = nn.Conv2d(channels, edges*heads, 1, 1, 0, bias=bias)
        self.QKV = nn.Conv2d(channels, channels*3, 1, 1, 0, bias=bias)
        self.O = nn.Conv2d(channels, channels, 1, 1, 0, bias=bias)

        nn.init.kaiming_uniform_(self.QKV.weight, nonlinearity='linear')
        

    def forward(self, x):
        # x: (batch_size, channels, height, width) as input
        B, C, H, W = x.shape

        # Compute Q, K, V
        q, k, v = self.QKV(x).view(B, 3, self.heads, self.h_dim, H*W).unbind(1)
        w = self.hyperedge(x).view(B, self.heads, self.edges, H*W)
        w = F.relu(w) * F.softmax(w, dim=2) * (self.edges/H/W)

        # Contraction of K, V
        q = q.transpose(2, 3).contiguous()
        zk = torch.einsum('bhdn,bhen->bhed', k, w).contiguous()
        zv = torch.einsum('bhdn,bhen->bhed', v, w).contiguous()

        zk = zk / torch.sum(zk**2, dim=(1, 3), keepdim=True).clamp(min=1e-6).sqrt()
        zv = zv / torch.sum(zv**2, dim=(1, 3), keepdim=True).clamp(min=1e-6).sqrt()

        # SDPA
        y = F.scaled_dot_product_attention(q, zk, zv)
        y = self.O(y.permute(0, 2, 1, 3).reshape(B, C, H, W))
        return y


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # Test HyperEdgeAttention
    B, C, H, W = 4, 128, 128, 128
    heads = 4

    x = torch.randn(B, C, H, W).to(device)
    CA = HyperEdgeAttention(C, C*2, heads).to(device)

    # Profile the forward pass
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        y = CA(x)
        loss = torch.sum(y)
        loss.backward()

    print(prof.key_averages().table(sort_by=f"{device}_time_total", row_limit=12))
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
    print("Total trainable parameters:", round(sum(p.numel() for p in CA.parameters() if p.requires_grad)/1e6, 2), 'M')
    print("IO is size:", x.element_size() * x.nelement() / 1024 / 1024, 'MB')
    print("I/O has elements: ", round(y.nelement() / 1e6, 2), 'M')