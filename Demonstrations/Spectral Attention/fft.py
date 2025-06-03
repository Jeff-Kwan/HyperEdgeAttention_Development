import torch

x = torch.randn(8, 64, 128, 128)  # Example input tensor

# Perform 2D FFT
X = torch.fft.rfft2(x, s=(2 * x.shape[2], 2 * x.shape[3]))
# Perform inverse 2D FFT
x_reconstructed = torch.fft.irfft2(X, s=(2 * x.shape[2], 2 * x.shape[3]))
# Crop the reconstructed tensor to the original size
x_reconstructed_cropped = x_reconstructed[..., :x.shape[2], :x.shape[3]]

# Check if the original and reconstructed tensors are close
print(f"Max absolute difference: {(x - x_reconstructed_cropped).abs().max().item():.6e}")
print(f"Mean absolute difference: {(x - x_reconstructed_cropped).abs().mean().item():.6e}")