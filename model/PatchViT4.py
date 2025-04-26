import torch
from torch import nn
import torch.nn.functional as F

class RMSNormTranspose(nn.Module):
    def __init__(self, dim, features, eps=1e-6, elementwise_affine=True):
        super(RMSNormTranspose, self).__init__()
        self.dim = dim
        self.norm = nn.RMSNorm(features, eps, elementwise_affine)

    def forward(self, x):
        return self.norm(x.transpose(self.dim, -1)).transpose(self.dim, -1)
    

class ConvBlock(nn.Module):
    def __init__(self, patch, in_c, h_c, out_c, bias=True):
        super(ConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.PixelUnshuffle(patch),
            RMSNormTranspose(1, in_c*patch**2),
            nn.Conv2d(in_c*patch**2, h_c, 3, 1, 1, bias=bias),
            nn.SiLU(),
            nn.Conv2d(h_c, out_c*patch**2, 3, 1, 1, bias=bias),
            nn.PixelShuffle(patch))

    def forward(self, x):
        return self.convs(x)
    

class CSwiGLU(nn.Module):
    def __init__(self, patch, in_c, h_c, out_c, bias=True):
        super(CSwiGLU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.PixelUnshuffle(patch),
            RMSNormTranspose(1, in_c*patch**2),
            nn.Conv2d(in_c*patch**2, h_c*2, 1, 1, 0, bias=bias),
            nn.Conv2d(h_c*2, h_c*2, 3, 1, 1, bias=bias, groups=h_c*2))
        self.act = nn.SiLU()
        self.conv2 = nn.ConvTranspose2d(h_c, out_c, patch, patch, 0, bias=bias)

    def forward(self, x):
        x1, x2 = self.conv1(x).chunk(2, dim=1)
        return self.conv2(self.act(x1) * x2)


class PatchMHA(nn.Module):
    def __init__(self, patch, in_c, h_c, out_c, heads, bias=False):
        super(PatchMHA, self).__init__()
        assert h_c % heads == 0, "h_c must be divisible by heads."
        self.heads = heads
        self.head_dim = h_c // heads
        self.h_c = h_c
        self.patch = patch
        self.QKV = nn.Sequential(
            nn.PixelUnshuffle(patch),
            RMSNormTranspose(1, in_c*patch**2),
            nn.Conv2d(in_c*patch**2, h_c*3, 1, 1, 0, bias=bias))
        self.O = nn.ConvTranspose2d(h_c, out_c, patch, patch, 0, bias=bias)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (Batch, Channels, H, W)
        B, _, H, W = x.shape
        q, k, v = self.QKV(x).view(B, 3, self.heads, self.head_dim, -1).transpose(3, 4).unbind(dim=1)

        q, k, v = map(lambda x: x.contiguous(), (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)

        y = self.O(y.transpose(2, 3).reshape(B, self.h_c, H//self.patch, W//self.patch))
        return y


class Layer(nn.Module):
    def __init__(self, in_c, convs, attns, mlps, bias=False):
        super(Layer, self).__init__()
        self.ConvBlock = nn.ModuleList([
            ConvBlock(p, in_c, c, in_c, bias=bias)
            for p, c in convs])
        self.PatchMHA = nn.ModuleList([
            PatchMHA(p, in_c, c, in_c, h, bias=bias)
            for p, c, h in attns])
        self.CSwiGLU = nn.ModuleList([
            CSwiGLU(p, in_c, c, in_c, bias=bias)
            for p, c in mlps])


    def forward(self, x):      
        # Sequential Blocks
        for conv in self.ConvBlock:
            x = x + conv(x)
        for attn in self.PatchMHA:
            x = x + attn(x)
        for mlp in self.CSwiGLU:
            x = x + mlp(x)
        return x


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, init_patch):
        super(ConvEmbedding, self).__init__()
        self.embed= nn.Sequential(
            nn.Conv2d(in_channels+4, out_channels, init_patch, init_patch, 0, bias=False),
            RMSNormTranspose(1, out_channels, elementwise_affine=False))

    def forward(self, x):
        # x shape: (Batch, in_channels, H, W)
        with torch.no_grad():
            B, _, H, W = x.shape
            h_frac = torch.linspace(0, 1, H, device=x.device, requires_grad=False).view(-1, 1)
            w_frac = torch.linspace(0, 1, W, device=x.device, requires_grad=False).view(1, -1)
            pos = torch.stack([
                h_frac.expand(-1, W),
                (1-h_frac).expand(-1, W),
                w_frac.expand(H, -1),
                (1-w_frac).expand(H, -1)
            ], dim=0).expand(B, -1, -1, -1)
            x = torch.cat([x, pos], dim=1)
        x = self.embed(x)
        return x


class PatchViT(nn.Module):
    def __init__(self, model_params):
        super(PatchViT, self).__init__()
        in_channels = model_params['in_channels']
        classes = model_params['out_channels']
        init_patch, init_channels = model_params['init_pc']
        out_patch, out_channels = model_params['out_pc']
        convs = model_params['convs']
        attns = model_params['attns']
        mlps = model_params['mlps']
        num_layers = model_params['layers']

        convs = [[p // init_patch, c] for p, c in convs]
        attns = [[p // init_patch, c, h] for p, c, h in attns]
        mlps = [[p // init_patch, c] for p, c in mlps]
        assert all(all(v > 0 for v in params) for params in convs), "All patch sizes and channel sizes in convs must be greater than 0."
        assert all(all(v > 0 for v in params) for params in attns), "All patch sizes, channel sizes, and head counts in attns must be greater than 0."
        assert all(all(v > 0 for v in params) for params in mlps), "All patch sizes and channel sizes in mlps must be greater than 0."

        self.conv_embed = ConvEmbedding(in_channels, init_channels, init_patch)
        self.layers = nn.ModuleList([
            Layer(init_channels, convs, attns, mlps, bias=False)
            for _ in range(num_layers)])
        self.out = nn.Sequential(
            nn.Conv2d(init_channels, out_channels, out_patch, out_patch, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.RMSNorm(out_channels, elementwise_affine=False),
            nn.Linear(out_channels, classes))

    def forward(self, x):
        # Input Embedding
        x = self.conv_embed(x)

        # Triple Scale Parallel Layers
        for layer in self.layers:
            x = layer(x)
        
        # Classifier Output
        y = self.out(x)
        return y
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parameters for PatchViT
    B, H, W = 8, 224, 224
    model_params = {
        'in_channels': 3,
        'out_channels': 1000,
        'init_pc': [1, 4],
        'out_pc': [32, 1024],
        'convs': [[2, 32], [4, 64], [8, 128]],
        'attns': [[16, 512, 16]],
        'mlps': [[16, 1024]],
        'layers': 8
    }

    # Create random input tensor representing an image batch
    x = torch.randn(B, model_params['in_channels'], H, W).to(device)
    
    # Instantiate PatchViT
    model = PatchViT(model_params).to(device)
    
    # Profile the forward and backward pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        y = model(x)
        loss = y.sum()
        loss.backward()
        
    print(prof.key_averages().table(sort_by=f"{device}_time_total", row_limit=12))
    if torch.cuda.is_available():
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", round(total_params / 1e6, 2), 'M')
    
    # Calculate I/O sizes for input and output
    input_size_mb = x.element_size() * x.nelement() / 1024 / 1024
    output_size_mb = y.element_size() * y.nelement() / 1024 / 1024
    print("Input is size:", input_size_mb, 'MB')
    print("Output is size:", output_size_mb, 'MB')