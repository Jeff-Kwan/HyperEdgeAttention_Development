import torch
import torch.nn as nn
from torch.nn.attention import flex_attention


class SlidingWindowMHA(nn.Module):
    def __init__(self, in_c, heads, window, bias=False):
        super(SlidingWindowMHA, self).__init__()
        self.channels = in_c
        self.window = window
        self.heads = heads
        self.head_dim = in_c // heads
        self.QKV = nn.Conv2d(in_c, in_c*3, 1, 1, 0, bias=bias)
        self.O = nn.Conv2d(in_c, in_c, 1, 1, 0, bias=bias)
        self.flex_attn = torch.compile(flex_attention.flex_attention)
        self.W = None

        for m in self.modules():
            if hasattr(m, 'weight'):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def SW_mask2D_score(self, score, b, h, q_idx, kv_idx):
        q_x, q_y = q_idx // self.W, q_idx % self.W
        kv_x, kv_y = kv_idx // self.W, kv_idx % self.W
        return torch.where(
            ((q_x - kv_x).abs() <= self.window) & ((q_y - kv_y).abs() <= self.window),
            score,
            float('-inf')
        )

    def forward(self, x):
        B, _, H, W = x.shape
        self.W = W          
        q, k, v = self.QKV(x).view(B, 3, self.heads, self.head_dim, -1).transpose(3, 4).unbind(dim=1)

        q, k, v = map(lambda x: x.contiguous(), (q, k, v))
        y = self.flex_attn(q, k, v, score_mod=self.SW_mask2D_score)

        y = self.O(y.transpose(2, 3).reshape(B, self.channels, H, W))
        return y



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example usage
    B = 128
    heads = 4
    H, W = 16, 16
    embed_dim = 32
    WINDOW = 4

    x = torch.randn(B, embed_dim, H, W).to(device)
    model = SlidingWindowMHA(embed_dim, heads, WINDOW).to(device)
    
    # Profile the pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        y = model(x)
        loss = torch.sum(y)
        loss.backward()

    print(prof.key_averages().table(sort_by=f"{device}_time_total", row_limit=12))
    if torch.cuda.is_available():
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    print("Total trainable parameters:", round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2), 'M')
    print("IO is size:", x.element_size() * x.nelement() / 1024 / 1024, 'MB')
    print("I/O has elements: ", round(y.nelement() / 1e6, 2), 'M')