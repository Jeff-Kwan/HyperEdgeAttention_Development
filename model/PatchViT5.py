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
    

class HyperEdgeAttention(nn.Module):
    '''HyperEdge partitioning and attention.'''
    def __init__(self, patch, in_c, edges, heads, bias=False):
        super(HyperEdgeAttention, self).__init__()
        channels = in_c * patch**2
        assert channels%heads == 0, f"Channels {channels} not divisble by heads {heads}"
        self.edges = edges
        self.in_norm = nn.Sequential(
            nn.PixelUnshuffle(patch) if patch > 1 else nn.Identity(),
            RMSNormTranspose(1, channels))
        self.hyperedge = nn.Linear(channels, edges, bias=bias)
        self.MHA = nn.MultiheadAttention(channels, heads, bias=bias, batch_first=True)
        self.norm = nn.RMSNorm(channels)
        self.out = nn.PixelShuffle(patch) if patch > 1 else nn.Identity()
        

    def forward(self, x):
        # x: (batch_size, channels, height, width) as input
        x = self.in_norm(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)

        # "Classify" which edge it belongs to & linear weighting
        weights = self.hyperedge(x)
        weights = F.relu(weights) * F.softmax(weights, dim=-1) * (self.edges/H/W)
        z = self.norm(weights.transpose(1, 2) @ x)

        # Cross Attention with representative bins
        y = self.MHA(x, z, z, need_weights=False)[0]
        y = self.out(y.permute(0, 2, 1).reshape(B, C, H, W))
        return y



class ConvBlock(nn.Module):
    def __init__(self, patch, in_c, h_c, out_c, bias=True):
        super(ConvBlock, self).__init__()
        assert h_c % 2 == 0, "h_c must be divisible by 2."
        self.in_norm = nn.Sequential(
            nn.PixelUnshuffle(patch) if patch > 1 else nn.Identity(),
            RMSNormTranspose(1, in_c*patch**2))
        self.convs = nn.ModuleList([
            nn.Conv2d(in_c*patch**2, h_c//2, 3, 1, 1, bias=bias),
            nn.Conv2d(in_c*patch**2, h_c//2, 3, 1, 2, dilation=2, bias=bias),
        ])
        self.out_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(h_c, out_c*patch**2, 3, 1, 1, bias=bias),
            nn.PixelShuffle(patch) if patch > 1 else nn.Identity())

    def forward(self, x):
        x = self.in_norm(x)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.out_proj(x)
        return x
    

class SwiGLU(nn.Module):
    def __init__(self, patch, in_c, h_c, out_c, bias=True):
        super(SwiGLU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.PixelUnshuffle(patch) if patch > 1 else nn.Identity(),
            RMSNormTranspose(1, in_c*patch**2),
            nn.Conv2d(in_c*patch**2, h_c*2, 1, 1, 0, bias=bias))
        self.act = nn.SiLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(h_c, out_c*patch**2, 1, 1, 0, bias=bias),
            nn.PixelShuffle(patch) if patch > 1 else nn.Identity())

    def forward(self, x):
        x1, x2 = self.conv1(x).chunk(2, dim=1)
        return self.conv2(self.act(x1) * x2)



class Layer(nn.Module):
    def __init__(self, in_c, convs, attns, mlps, bias=False):
        super(Layer, self).__init__()
        self.ConvBlock = ConvBlock(convs[0], in_c, convs[1], in_c, bias=bias)
        self.HyperEdgeAttn = HyperEdgeAttention(attns[0], in_c,
                attns[1], attns[2], bias=bias)
        self.SwiGLU = SwiGLU(mlps[0], in_c, mlps[1], in_c, bias=bias)


    def forward(self, x):      
        # Sequential Blocks
        x = x + self.ConvBlock(x)
        x = x + self.HyperEdgeAttn(x)
        x = x + self.SwiGLU(x)
        return x


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, init_patch):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4 for 2-D sin-cos PE"
        self.channels = out_channels
        self.embed = nn.Conv2d(in_channels, out_channels,
                               kernel_size=init_patch, stride=init_patch,
                               padding=0, bias=False)
        self.pos_embed = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)
        self.norm = RMSNormTranspose(1, out_channels, elementwise_affine=False)
        self.register_buffer(
            "radians",
            2048 ** torch.linspace(0, 1, out_channels//4).view(-1, 1, 1)
        )

    @torch.no_grad()
    def _build_2d_sincos(self, H: int, W: int, device):
        y_embed = self.radians * torch.linspace(0, 1, H, device=device).view(1, H, 1)
        x_embed = self.radians * torch.linspace(0, 1, W, device=device).view(1, 1, W)
        pos = torch.cat([
            torch.sin(y_embed).expand(1, -1, -1, W),
            torch.cos(y_embed).expand(1, -1, -1, W),
            torch.sin(x_embed).expand(1, -1, H, -1),
            torch.cos(x_embed).expand(1, -1, H, -1)
        ], dim=1)
        return pos

    def forward(self, x):
        x = self.embed(x)
        pos = self._build_2d_sincos(x.size(2), x.size(3), x.device)
        x = self.norm(x + self.pos_embed(pos))
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

        convs[0] = convs[0] // init_patch
        attns[0] = attns[0] // init_patch
        mlps[0] = mlps[0] // init_patch

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
    B, H, W = 16, 256, 256
    model_params = {
        "in_channels": 3,
        "out_channels": 1000,
        "init_pc": [4, 48],
        "out_pc": [8, 192],
        "convs": [4, 64],
        "attns": [8, 192, 6],
        "mlps": [8, 512],
        "layers": 8
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
        with torch.autocast('cuda', torch.bfloat16):
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