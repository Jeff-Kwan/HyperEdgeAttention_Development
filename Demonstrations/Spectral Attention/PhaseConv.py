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
        q, k, v= torch.fft.rfft2(self.QKV(x).view(B, 3*self.heads, self.head_dim, H, W), s=(2*H, 2*W)).chunk(3, dim=1)
        
        # 2) Head-wise cross‐spectrum of q and k to phase maps H
        S = torch.sum(q * k.conj(), dim=2)
        H_phase = S / (S.abs().clamp_min(self.eps))

        # 3) Apply phase‐only filter to v
        y = torch.fft.irfft2(v * H_phase.unsqueeze(2), s=(2*H, 2*W))[..., :H, :W]
        y = self.out(y.view(B, C, H, W))
        return y


if __name__ == "__main__":
    # create test input
    B, C, H, W = 4, 64, 128, 128
    heads = 4
    x = torch.randn(B, C, H, W)

    # create PhaseConv module
    phase_attention = PhaseConv(C, heads)

    # apply phase attention
    y = phase_attention(x)

    # check output shape
    print(f"Output shape: {y.shape}")
    assert y.shape == (B, C, H, W), "Output shape mismatch"