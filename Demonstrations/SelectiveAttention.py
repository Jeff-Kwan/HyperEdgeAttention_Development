import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B, S, E = 128, 4096, 128
heads = 4
mha = nn.MultiheadAttention(embed_dim=E, num_heads=heads, batch_first=True).eval().to(device)

class TokenSelection(nn.Module):
    def __init__(self):
        super(TokenSelection, self).__init__()
        self.hyperedges = nn.Parameter(torch.randn(1, E, E))
        # self.hyperedges = nn.Parameter(torch.eye(E, E).unsqueeze(0))
        self.norm = nn.RMSNorm(E, elementwise_affine=False)

    def forward(self, x):
        return self.norm(F.scaled_dot_product_attention(self.hyperedges, x, x))
        
selection = TokenSelection().to(device)

with torch.no_grad():
    x = F.rms_norm(torch.randn(B, S, E, device=device), [E])
    y_target = mha(x, x, x, need_weights=False)[0]
    z = F.rms_norm(torch.randn_like(x, device=device), [E])
    z = selection(x)
    y_select = mha(x, z, z, need_weights=False)[0]
print(f"Initial Stats")
print(f"MSE: {F.mse_loss(y_select, y_target).item():.6f}")
print(f"Max Error: {torch.max(torch.abs(y_select - y_target)).item():.6f}")
print(f"Mean Error: {torch.mean(torch.abs(y_select - y_target)).item():.6f}\n")


# criterion = nn.MSELoss()
# optimizer = torch.optim.AdamW(selection.parameters(), lr=1e-2, weight_decay=2e-3)
# steps = 1000
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

# p_bar = tqdm(range(steps))
# for step in p_bar:
#     optimizer.zero_grad()

#     # Generate random input and self attention target
#     with torch.no_grad():
#         x = F.rms_norm(torch.randn(B, S, E, device=device), [E])
#         y_target = mha(x, x, x, need_weights=False)[0]

#     # Selective token low rank attention
#     z = selection(x)
#     y_select = mha(x, z, z, need_weights=False)[0]

#     loss = criterion(y_select, y_target)
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
#     p_bar.set_description(f"Loss: {loss.item():.6f}")