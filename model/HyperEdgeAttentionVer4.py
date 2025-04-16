import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

class HyperEdgeAttention(nn.Module):
    '''Fully data-dependent Hypergraph Convolution by Attention-like mechanism'''
    def __init__(self, channels, edges, heads, bias=False):
        super(HyperEdgeAttention, self).__init__()
        self.channels = channels
        self.edges = edges
        self.heads = heads
        self.head_dim = channels // heads
        self.sqrt_dk = self.head_dim**0.5
        assert channels%heads == 0, f"Channels {channels} not divisble by heads {heads}"

        # Linear Projections
        self.hyperedge = nn.Linear(channels, edges, bias=bias)
        self.MHA1 = nn.MultiheadAttention(channels, heads, bias=bias)
        self.MHA2 = nn.MultiheadAttention(channels, heads, bias=bias)

        # Initializations
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        # x: (batch_size, channels, height, width) as input
        B, C, height, width = x.shape
        x = x.reshape(B, C, height*width).permute(2, 0, 1).contiguous()

        # Create Hyperedges
        weights = F.softmax(self.hyperedge(x), dim=0)
        edges = F.rms_norm(torch.einsum('nhc,nhe->ehc', x, weights), [C])

        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            z = self.MHA1(edges, x, x, need_weights=False)[0]
            y = self.MHA2(x, z, z, need_weights=False)[0]

        return y.permute(1, 2, 0).reshape(B, C, height, width)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # Test HyperEdgeAttention
    B, C, height, width = 4, 64, 256, 256
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