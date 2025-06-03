import torch
from torch import nn
from torch.nn import functional as F


class SpectralAttention(nn.Module):
    def __init__(self, channels, heads):
        super(SpectralAttention, self).__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.QKV = nn.Conv2d(channels, channels * 3, 1, 1, 0, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.heads = heads
        self.head_dim = channels // heads
        self.sqrt_dk = self.head_dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)

        nn.init.kaiming_normal_(self.QKV.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.out.weight, nonlinearity='linear')

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1) Q K V projection and FFT
        q, k, v = torch.fft.rfft2(self.QKV(x).view(B, 3*self.heads, self.head_dim, H, W), s=(2*H, 2*W), norm="ortho").chunk(3, dim=1)
        
        # 2) Cross Correlation Attention
        S = torch.sum((q * k.conj()), dim=2, keepdim=True).real / self.sqrt_dk
        S = F.softmax(S.view(B, self.heads, 1, -1), dim=-1).view(S.size())

        # 3) Apply filter to v
        y = torch.fft.irfft2(v * S, s=(2*H, 2*W), norm="ortho")[..., :H, :W]
        y = self.out(y.view(B, C, H, W))
        return y
    

B, C, H, W = 1, 64, 128, 128
heads = 4
x = torch.randn(B, C, H, W)
GC = SpectralAttention(C, heads)

y = GC(x)