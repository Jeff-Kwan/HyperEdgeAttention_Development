import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

class HyperEdgeAttention(nn.Module):
    def __init__(self, channels, edges, heads, bias=False, checkpoint=True):
        super().__init__()
        assert channels % heads == 0
        self.edges = edges
        self.checkpoint = checkpoint              # on/off flag
        self.hyperedge = nn.Linear(channels, edges, bias=bias)
        self.MHA = nn.MultiheadAttention(channels, heads, bias=bias,
                                         batch_first=True)
        self.norm = nn.RMSNorm(channels)

    def _inner(self, x, H, W):                    # <-- checkpointable block
        w = self.hyperedge(x)
        w = F.relu(w) * F.softmax(w, dim=-1) * (self.edges / H / W)
        z = self.norm(w.transpose(1, 2) @ x)
        return z

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)

        if self.training and self.checkpoint:
            z = checkpoint.checkpoint(
                    lambda _x: self._inner(_x, H, W),   # closure captures modules
                    x,
                    use_reentrant=False)                # new non-reentrant API
        else:
            z = self._inner(x, H, W)

        y = self.MHA(x, z, z, need_weights=False)[0]

        return y.permute(0, 2, 1).reshape(B, C, H, W)



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