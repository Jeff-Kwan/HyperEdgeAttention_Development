import torch
from torch import nn
from torch.nn import functional as F

def linear_phase_map(x, y, eps=1e-6):
    """
    Compute the phase‐only transfer function H between x→y,
    using rfft2 on a zero‐padded grid of size (2H,2W).
    """
    B, C, H, W = x.shape

    # 1) real→complex FFTs (zero‐pads internally to 2H, 2W)
    X = torch.fft.rfft2(x, s=(2 * H, 2 * W))
    Y = torch.fft.rfft2(y, s=(2 * H, 2 * W))

    # 1) cross‐spectrum summed over channels → (B, Hpad, Wpad//2+1)
    S = torch.sum(X.conj() * Y, dim=1)

    # 4) normalize to unit magnitude → pure phase
    H_phase = S / (S.abs().clamp_min(eps))

    return H_phase

def apply_phase_map(z, H_phase):
    """
    Apply a phase‐only filter H_phase to z,
    on the same zero‐padded grid, then crop back.
    """
    B, C, H, W = z.shape

    # 1) FFT z → (B, C, Hpad, Wpad//2+1)
    Z = torch.fft.rfft2(z, s=(2 * H, 2 * W))

    # 2) apply phase‐only filter (broadcast over C)
    Z_shift = Z * H_phase.unsqueeze(1)   # → (B, C, Hpad, Wpad//2+1)

    # 3) inverse FFT back to real, shape = (B, C, Hpad, Wpad)
    z_padded = torch.fft.irfft2(Z_shift, s=(2 * H, 2 * W))

    # 4) crop top left H×W region
    z_shifted = z_padded[..., :H, :W]

    return z_shifted


if __name__ == "__main__":

    # create test input
    B, C, H, W = 1, 3, 128, 128
    x = torch.randn(B, C, H, W)

    # compute phase map for identity (x→x)
    H_phase = linear_phase_map(x, x)

    # apply to x should return back x
    x_shifted = apply_phase_map(x, H_phase)

    # check maximum error
    max_diff = (x_shifted - x).abs().max().item()
    print(f"Max absolute difference after identity phase map: {max_diff:.6e}")

    assert max_diff < 1e-5, "Padding/cropping or FFT steps are incorrect!"