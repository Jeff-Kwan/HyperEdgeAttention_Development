import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseConv(nn.Module):
    '''
    Global data-dependent phase convolution?
    Computes a phase-only transfer function H between q→k,
    using rfft2 on a zero-padded grid of size (2H, 2W).
    Then applies the phase-only filter to v.
    '''
    def __init__(self, channels, heads, eps=1e-6):
        super(PhaseConv, self).__init__()
        assert channels % heads == 0, "channels must be divisible by heads"

        self.QKV = nn.Conv2d(channels, channels * 3, 1, 1, 0, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.heads = heads
        self.head_dim = channels // heads
        self.eps = eps

        nn.init.kaiming_normal_(self.QKV.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.out.weight, nonlinearity='linear')

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) Q K V projection and FFT
        q, k, v= torch.fft.rfft2(self.QKV(x), s=(2*H, 2*W), norm="ortho").chunk(3, dim=1)
        
        # 2) Head-wise cross‐spectrum of q and k to phase maps H
        S = (q * k.conj()).real
        S = F.softmax(S.view(S.size(0), S.size(1), -1).view(S.size()), dim=-1)
        # H_phase = torch.polar(torch.ones(S.shape, device=S.device), S.angle())

        # 3) Apply phase‐only filter to v
        y = torch.fft.irfft2(v * S, s=(2*H, 2*W), norm="ortho")[..., :H, :W]
        y = self.out(y)
        return y


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # Test HyperEdgeAttention
    B, C, height, width = 4, 64, 256, 256
    heads = 4

    x = torch.randn(B, C, height, width).to(device)
    CA = PhaseConv(channels=C, heads=heads).to(device)

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