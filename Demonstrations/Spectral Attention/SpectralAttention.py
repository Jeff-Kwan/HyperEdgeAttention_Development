import torch
from torch import nn
from torch.nn import functional as F


class SpectralAttention(nn.Module):
    def __init__(self, channels):
        super(SpectralAttention, self).__init__()
        self.QKV = nn.Conv2d(channels, channels * 3, 1, 1, 0, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) Q K V projection and FFT
        q, k, v = torch.fft.rfft2(self.QKV(x), s=(2*H, 2*W), norm="ortho").chunk(3, dim=1)
        
        # 2) Cross Correlation Attention
        S = (q * k.conj()).real
        S = self.tanh(S * self.alpha) * self.gamma
        S = F.softmax(S.view(B, C, -1), dim=-1).view(S.size())

        # 3) Apply filter to v
        y = torch.fft.irfft2(v * S, norm="ortho")[..., :H, :W]
        y = self.out(y)

        # Plot heat map of S
        # import matplotlib.pyplot as plt
        # S = torch.complex(S, torch.zeros_like(S))
        # S_kernel = torch.fft.irfft2(S, norm="ortho")
        # S_kernel = torch.log(torch.mean(S_kernel, dim=1)[0]).cpu().detach().numpy()
        # plt.figure(figsize=(12, 12))
        # plt.imshow(S_kernel, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title('Heat Map of S Kernel')
        # plt.show()
        return y
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # Test HyperEdgeAttention
    B, C, height, width = 4, 64, 128, 128

    x = torch.randn(B, C, height, width).to(device)
    CA = SpectralAttention(channels=C).to(device)

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