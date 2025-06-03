import torch
from torch import nn
from torch.nn import functional as F


class SpectralAttention(nn.Module):
    def __init__(self, channels):
        super(SpectralAttention, self).__init__()
        self.QKV = nn.Conv2d(channels, channels * 3, 1, 1, 0, bias=False)
        self.spectral_mix = nn.Sequential(
            nn.Conv2d(channels, channels*2, 1, 1, 0, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels, channels*2, 1, 1, 0, bias=False)
        )
        self.out = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.kaiming_normal_(self.QKV.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.out.weight, nonlinearity='linear')

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) Q K V projection and FFT
        qkv = F.pad(self.QKV(x), (W//2, W//2, H//2, H//2))
        q, k, v = torch.fft.rfft2(qkv, norm="ortho").chunk(3, dim=1)
        
        # 2) Cross Correlation Attention
        S = (q * k.conj()).real
        S = F.softmax(S.view(S.size(0), S.size(1), -1), dim=-1).view(S.size())

        # 3) Apply filter to v
        v = v + self.spectral_mix(v)
        y = torch.fft.irfft2(v * S, norm="ortho")[..., H//2:H+H//2, W//2:W+W//2]
        y = self.out(y)

        # Plot heat map of S
        import matplotlib.pyplot as plt
        S = torch.complex(S, torch.zeros_like(S))
        S_kernel = torch.fft.irfft2(S, norm="ortho").cpu().detach().numpy()
        plt.imshow(S_kernel[0, 0], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Heat Map of S Kernel')
        plt.show()
        return y
    

B, C, H, W = 1, 64, 32, 32
x = torch.randn(B, C, H, W)
GC = SpectralAttention(C)

y = GC(x)