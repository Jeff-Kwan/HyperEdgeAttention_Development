import torch
from torch import nn
from torch.nn import functional as F

class HyperEdgeAttention(nn.Module):
    '''Fully data-dependent Hypergraph Convolution by Attention-like mechanism'''
    def __init__(self, channels, edges, heads, bias=False):
        super(HyperEdgeAttention, self).__init__()
        assert channels%heads == 0, f"Channels {channels} not divisble by heads {heads}"

        self.hyperedge = nn.Linear(channels, edges, bias=bias)
        self.MHA = nn.MultiheadAttention(channels, heads, bias=bias, batch_first=True)
        self.norm = nn.RMSNorm(channels, elementwise_affine=False)
        

    def forward(self, x):
        # x: (batch_size, channels, height, width) as input
        B, C, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, height*width, C)

        weights = self.hyperedge(x)
        weights = F.relu(weights) * F.softmax(weights, dim=-1)
        z = self.norm(torch.bmm(x.transpose(1, 2), weights))

        y = self.MHA(x, z, z, need_weights=False)[0]

        return y.transpose(1, 2).reshape(B, C, height, width)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # Test HyperEdgeAttention
    B, C, height, width = 4, 128, 128, 128
    heads = 4

    x = torch.randn(B, C, height, width).to(device)
    CA = HyperEdgeAttention(C, C, heads).to(device)

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