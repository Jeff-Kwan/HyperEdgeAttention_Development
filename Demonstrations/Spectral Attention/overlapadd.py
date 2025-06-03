import torch
from torch.nn import functional as F

B, C, H, W = 1, 1, 128, 128
x = torch.randn(B, C, H, W)

# 1. Convolution method (with correct flipping and shape)
# Flip spatial dims for autocorrelation kernel
kernel = x.flip(-1).flip(-2)
# conv2d expects (out_channels, in_channels/groups, kH, kW)
kernel = kernel  # shape (B, C, H, W)
kernel = kernel.view(B*C, 1, H, W)  # flatten batch and channel if necessary

# Pad the input to (2H-1, 2W-1)
x_pad = F.pad(x, (W-1, W-1, H-1, H-1), mode='constant', value=0)
x_pad = x_pad.view(B*C, 1, x_pad.shape[-2], x_pad.shape[-1])  # flatten

# Do convolution (cross-correlation with flipped kernel)
x_acorr_full = F.conv2d(x_pad, kernel, groups=1)
x_acorr_full = x_acorr_full.view(B, C, x_acorr_full.shape[-2], x_acorr_full.shape[-1])

# 2. FFT method
x_padded = F.pad(x, (0, W-1, 0, H-1), mode='constant', value=0)  # zero pad to (2H-1, 2W-1)
y_fft = torch.fft.rfft2(x_padded, norm='ortho')
fft_acorr_full = torch.fft.irfft2(y_fft * y_fft.conj(), norm='ortho', s=(2*H-1, 2*W-1))
fft_acorr_full = fft_acorr_full.real  # Only take the real part

# 3. Compare
print("FFT Reconstruction Error:", (x_acorr_full - fft_acorr_full).abs().mean().item())
