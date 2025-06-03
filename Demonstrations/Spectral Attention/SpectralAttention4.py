import torch
from torch import nn
import torch.nn.functional as F

class SpectralAttention(nn.Module):
    def __init__(self, channels, block_size=(64, 64), overlap=(16, 16)):
        super(SpectralAttention, self).__init__()
        self.channels = channels
        self.block_size = block_size  # (block_height, block_width)
        self.overlap = overlap  # (overlap_height, overlap_width)
        self.QKV = nn.Conv2d(channels, channels * 3, 1, 1, 0, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        bh, bw = self.block_size
        oh, ow = self.overlap
        sh, sw = bh - oh, bw - ow  # strides

        # Calculate necessary padding
        pad_h = (sh - (H - bh) % sh) % sh
        pad_w = (sw - (W - bw) % sw) % sw
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, H_pad, W_pad = x_padded.shape

        # Unfold input into overlapping blocks
        x_unfold = F.unfold(x_padded, kernel_size=(bh, bw), stride=(sh, sw))
        x_unfold = x_unfold.view(B, C, bh, bw, -1)  # (B, C, bh, bw, num_blocks)
        num_blocks = x_unfold.shape[-1]

        # Reshape for batch processing
        x_blocks = x_unfold.permute(0, 4, 1, 2, 3).reshape(-1, C, bh, bw)  # (B*num_blocks, C, bh, bw)

        # Apply QKV projection and FFT
        qkv = self.QKV(x_blocks)
        q, k, v = torch.chunk(torch.fft.rfft2(qkv, norm='ortho'), 3, dim=1)

        # Cross-correlation attention
        S = (q * k.conj()).real
        S = self.tanh(S * self.alpha) * self.gamma
        S = F.softmax(S.view(S.size(0), S.size(1), -1), dim=-1).view_as(S)

        # Apply attention to v and inverse FFT
        y = torch.fft.irfft2(v * S, s=(bh, bw), norm='ortho')

        # Apply output projection
        y = self.out(y)

        # Reshape back to blocks
        y_blocks = y.view(B, num_blocks, C, bh, bw).permute(0, 2, 3, 4, 1)

        # Fold blocks back to image
        y = y_blocks.reshape(B, C * bh * bw, num_blocks)
        output = F.fold(y, output_size=(H_pad, W_pad), kernel_size=(bh, bw), stride=(sh, sw))

        # Create overlap-add divisor to normalize overlapping regions
        ones = torch.ones_like(x_padded)
        ones_unfold = F.unfold(ones, kernel_size=(bh, bw), stride=(sh, sw))
        ones_unfold = ones_unfold.view(B, C, bh, bw, -1)
        ones_blocks = ones_unfold.permute(0, 4, 1, 2, 3).reshape(-1, C, bh, bw)
        ones_blocks = ones_blocks.view(B, num_blocks, C, bh, bw).permute(0, 2, 3, 4, 1)
        ones_blocks = ones_blocks.reshape(B, C * bh * bw, num_blocks)
        divisor = F.fold(ones_blocks, output_size=(H_pad, W_pad), kernel_size=(bh, bw), stride=(sh, sw))
        divisor[divisor == 0] = 1.0  # Avoid division by zero

        # Normalize output
        output = output / divisor

        # Crop to original size
        return output[:, :, :H, :W]


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