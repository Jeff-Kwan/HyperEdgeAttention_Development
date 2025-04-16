import torch
from torch import nn
import torch.nn.functional as F

def intrinsic_dim_effective_rank(x, eps=1e-12):
    """
    Compute the effective intrinsic dimensionality using the effective rank measure.
    
    Args:
        x (torch.Tensor): A tensor of shape (N, D) where N is the number of vectors.
        eps (float): A small constant to avoid log(0).
    
    Returns:
        float: The effective rank (a soft measure of intrinsic dimensionality).
    """
    # Center the data (remove the mean)
    x_centered = x - x.mean(dim=0, keepdim=True)
    # Compute the Singular Value Decomposition (SVD)
    _, S, _ = torch.linalg.svd(x_centered, full_matrices=False)
    # Compute the energy of each singular value (variance explained)
    energy = S ** 2
    energy_norm = energy / energy.sum()
    effective_rank = torch.exp(-torch.sum(energy_norm * torch.log(energy_norm + eps)))
    return effective_rank.item()

def intrinsic_dim_variance_threshold(x, variance_threshold=0.95):
    """
    Compute the intrinsic dimensionality as the number of principal components
    needed to reach a specified variance threshold.
    
    Args:
        x (torch.Tensor): A tensor of shape (N, D) where N is the number of vectors.
        variance_threshold (float): The fraction of total variance to be captured.
    
    Returns:
        int: The number of components required to explain at least the given 
             variance fraction.
    """
    # Center the data (remove the mean)
    x_centered = x - x.mean(dim=0, keepdim=True)
    # Compute SVD (which is equivalent to PCA here)
    _, S, _ = torch.linalg.svd(x_centered, full_matrices=False)
    # Variances are the squared singular values
    variances = S ** 2
    total_variance = variances.sum()
    # Compute cumulative variance ratio
    cumulative_variance = torch.cumsum(variances, dim=0) / total_variance
    # Count how many components are needed to reach the threshold
    intrinsic_dim = (cumulative_variance < variance_threshold).sum().item() + 1
    return intrinsic_dim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
N, E, C = 4096, 256, 1024
x = torch.randn(N, E, device=device)
A = nn.Linear(E, C, device=device)
B = nn.Linear(E, C, device=device)

# Compute the intrinsic dimensionality using both methods
eff_rank = intrinsic_dim_effective_rank(x)
var_dim = intrinsic_dim_variance_threshold(x, variance_threshold=0.99)

print("Effective Rank (Intrinsic Dimensionality):", eff_rank)
print("Intrinsic Dimensionality (95% Variance Threshold):", var_dim)

y = A(x)
eff_rank_y = intrinsic_dim_effective_rank(y)
var_dim_y = intrinsic_dim_variance_threshold(y, variance_threshold=0.99)

print("Effective Rank (Intrinsic Dimensionality) of y:", eff_rank_y)
print("Intrinsic Dimensionality (95% Variance Threshold) of y:", var_dim_y)

z = A(x) * F.silu(B(x))
eff_rank_z = intrinsic_dim_effective_rank(z)
var_dim_z = intrinsic_dim_variance_threshold(z, variance_threshold=0.99)

print("Effective Rank (Intrinsic Dimensionality) of z:", eff_rank_z)
print("Intrinsic Dimensionality (95% Variance Threshold) of z:", var_dim_z)