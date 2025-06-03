import torch
from torch import nn
from torch.nn import functional as F
from pytorch_wavelets import DWTForward, DWTInverse

class WaveletAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.QKV  = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.out  = nn.Conv2d(channels, channels,     kernel_size=1, bias=False)

        # --- wavelet set-up --------------------------------------------------
        self.levels  = 3
        self.wavelet = 'haar'
        self.dwt  = DWTForward(J=self.levels, wave=self.wavelet, mode='zero')
        self.idwt = DWTInverse(              wave=self.wavelet, mode='zero')

        # --- init -------------------------------------------------------------
        nn.init.kaiming_normal_(self.QKV.weight,  nonlinearity='linear')
        nn.init.kaiming_normal_(self.out.weight,  nonlinearity='linear')

    def _attn(self, q, k, v):
        """
        Vectorized attention on tensors of shape (B, C, *spatial_dims).
        Flattens all spatial dims, does softmax per channel, then reshapes back.
        """
        # flatten spatial dims into one
        B, C, *spatial = q.shape
        N = 1
        for d in spatial:
            N *= d
        # (B, C, N)
        S = (q * k).reshape(B, C, N)
        S = F.softmax(S, dim=2)
        # back to (B, C, *spatial)
        S = S.view(B, C, *spatial)
        return v * S

    def forward(self, x):
        # 1) get Q, K, V
        q, k, v = self.QKV(x).chunk(3, dim=1)

        # 2) perform DWT
        q_l, q_h = self.dwt(q)   # q_l: (B,C,H/2,W/2),   q_h: list of length levels each (B,C,3,H',W')
        k_l, k_h = self.dwt(k)
        v_l, v_h = self.dwt(v)

        # 3) attention on low-pass
        out_l = self._attn(q_l, k_l, v_l)

        # 4) attention on each high-pass level
        out_h = []
        for lvl in range(self.levels):
            q_band = q_h[lvl]  # (B, C, 3, H', W')
            k_band = k_h[lvl]
            v_band = v_h[lvl]

            B, C, O, H, W = q_band.shape
            # merge orientation into channel dim → (B, C*O, H, W)
            q_flat = q_band.reshape(B, C*O, H, W)
            k_flat = k_band.reshape(B, C*O, H, W)
            v_flat = v_band.reshape(B, C*O, H, W)

            # single vectorized attention
            out_flat = self._attn(q_flat, k_flat, v_flat)

            # reshape back → (B, C, 3, H, W)
            out_h.append(out_flat.reshape(B, C, O, H, W))

        # 5) inverse DWT and final projection
        y = self.idwt((out_l, out_h))
        return self.out(y)


if __name__ == "__main__":
    # create test input
    B, C, H, W = 1, 64, 128, 128
    x = torch.randn(B, C, H, W)

    # create WaveletAttention module
    wavelet_attention = WaveletAttention(C)

    # apply wavelet attention
    y = wavelet_attention(x)

    # check output shape
    print(f"Output shape: {y.shape}")
    assert y.shape == (B, C, H, W), "Output shape mismatch"