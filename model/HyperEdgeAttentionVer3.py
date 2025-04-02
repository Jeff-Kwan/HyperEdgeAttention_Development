import torch
from torch import nn
from torch.nn import functional as F

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
        self.V = nn.Linear(channels, channels, bias=bias)
        self.QK = nn.Linear(channels, channels*2, bias=bias)
        self.WE = nn.Linear(channels, heads, bias=bias)
        self.O = nn.Conv2d(channels, channels, 1, 1, 0, bias=bias)

        # Initializations
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        # x: (batch_size, channels, height, width) as input
        B, C, height, width = x.shape
        x = x.view(B, C, height*width).transpose(1, 2)

        # Create Hyperedges
        weights = F.softmax(self.hyperedge(x), dim=1)
        # weights = F.relu(self.hyperedge(x))
        edges = F.rms_norm(weights.transpose(1, 2) @ x, [C])

        # Queries, Keys, Edge weights
        q, k = self.QK(x).view(B, height*width, self.heads, self.head_dim, 2).transpose(1,2).unbind(dim=-1)
        v = self.V(x).view(B, height*width, self.heads, self.head_dim).transpose(1,2)
        hq, hk = self.QK(edges).view(B, self.edges, self.heads, self.head_dim, 2).transpose(1,2).unbind(dim=-1)

        # Positive hyperedge weights
        WE = F.normalize(F.softplus(self.WE(edges)).transpose(1,2).unsqueeze(-1), p=1, dim=-2) * self.edges

        # Scaled dot-product cross attention implementation
        # Theoretically almost equivalent to Hypergraph Convolution DinvHW @ (BinvH_T @ v)
        # Except for normalization Dinv because WE and H are not normalized together
        # hq, hk, q, k, v, WE = map(lambda x: x.contiguous(), (hq, hk, q, k, v, WE))
        z = F.scaled_dot_product_attention(hq, k, v)
        y = F.scaled_dot_product_attention(q, hk, z*WE)

        # Output Projection & Reshape
        y = self.O(y.transpose(2, 3).reshape(B, C, height, width).contiguous(memory_format=torch.channels_last))
        return y


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # Test HyperEdgeAttention
    B, C, height, width = 4, 64, 256, 256
    heads = 4

    x = torch.randn(B, C, height, width).to(device).contiguous(memory_format=torch.channels_last)
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