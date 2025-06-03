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
    def __init__(self, channels, edges, heads, bias=False):
        super(HyperEdgeAttention, self).__init__()
        assert channels%heads == 0, f"Channels {channels} not divisble by heads {heads}"
        self.edges = edges
        self.in_norm = RMSNormTranspose(1, channels)
        self.hyperedge = nn.Linear(channels, edges, bias=bias)
        self.MHA = nn.MultiheadAttention(channels, heads, bias=bias, batch_first=True)
        self.norm = nn.RMSNorm(channels)
        

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
        return y.permute(0, 2, 1).reshape(B, C, H, W)


class ReshapeSelfMHA(nn.Module):
    def __init__(self, in_c, heads, bias=False):
        super(ReshapeSelfMHA, self).__init__()
        self.in_norm = nn.RMSNorm(in_c)
        self.MHA = nn.MultiheadAttention(in_c, heads, bias=bias, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.in_norm(x.permute(0, 2, 3, 1).reshape(B, H*W, C))
        y = self.MHA(x, x, x, need_weights=False)[0]
        return y.permute(0, 2, 1).reshape(B, C, H, W)


class ConvBlock(nn.Module):
    def __init__(self, in_c, h_c, out_c, bias=True):
        super(ConvBlock, self).__init__()
        assert h_c % 2 == 0, "h_c must be divisible by 2."
        self.in_conv = nn.Sequential(
            RMSNormTranspose(1, in_c),
            nn.Conv2d(in_c, h_c, 1, 1, 0, bias=bias))
        self.path1 = nn.Conv2d(h_c//2, h_c//2, 3, 1, 1, bias=bias)
        self.path2 = nn.Conv2d(h_c//2, h_c//2, 3, 1, 2, dilation=2, bias=bias)
        self.out_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(h_c, out_c, 1, 1, 0, bias=bias))

    def forward(self, x):
        x1, x2 = self.in_conv(x).chunk(2, dim=1)
        # Further 3x3 and dilated 3x3 paths
        x = torch.cat([self.path1(x1), self.path2(x2)], dim=1)
        x = self.out_proj(x)
        return x
    

class SwiGLU(nn.Module):
    def __init__(self, in_c, h_c, out_c, bias=True):
        super(SwiGLU, self).__init__()
        self.conv1 = nn.Sequential(
            RMSNormTranspose(1, in_c),
            nn.Conv2d(in_c, h_c*2, 1, 1, 0, bias=bias))
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(h_c, out_c, 1, 1, 0, bias=bias)

    def forward(self, x):
        x1, x2 = self.conv1(x).chunk(2, dim=1)
        return self.conv2(self.act(x1) * x2)


class Layer(nn.Module):
    def __init__(self, in_c, conv, attn, mlp, bias=False, HyperAttn=True):
        super(Layer, self).__init__()
        self.ConvBlock = ConvBlock(in_c, conv, in_c, bias=bias)
        if HyperAttn:
            self.Attention = HyperEdgeAttention(in_c, attn[0], attn[1], bias=bias)
        else:
            self.Attention = ReshapeSelfMHA(in_c, attn, bias=bias)
        self.SwiGLU = SwiGLU(in_c, mlp, in_c, bias=bias)


    def forward(self, x):      
        # Sequential Blocks
        x = x + self.ConvBlock(x)
        x = x + self.Attention(x)
        x = x + self.SwiGLU(x)
        return x
    

class DownSample(nn.Module):
    def __init__(self, in_c, out_c, patch=2):
        super(DownSample, self).__init__()
        self.patch = patch
        self.patchmerge = nn.Conv2d(in_c, out_c, patch, patch, 0, bias=False)
        self.norm = RMSNormTranspose(1, out_c, elementwise_affine=False)
        nn.init.kaiming_normal_(self.patchmerge.weight, nonlinearity='linear')

    def forward(self, x):
        return self.norm(self.patchmerge(x))


class CNNEmbedding(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        assert out_c % 4 == 0, "out_channels must be divisible by 4."
        self.channels = out_c
        self.downs = nn.ModuleList([
            DownSample(in_c, out_c//4, 2),  # 1x1 -> 2x2
            DownSample(out_c//4, out_c, 2), # 2x2 -> 4x4
        ])
        self.convs = nn.ModuleList([
            ConvBlock(out_c//4, out_c//4, out_c//4, bias=False),  # 2x2
            ConvBlock(out_c, out_c, out_c, bias=False)            # 4x4
        ])

        # Positional Embedding
        self.pos_embed = nn.Conv2d(out_c, out_c, 1, 1, 0, bias=False)
        self.norm = RMSNormTranspose(1, out_c, elementwise_affine=False)
        self.register_buffer(
            "radians",
            2048 ** torch.linspace(0, 1, out_c//4).view(-1, 1, 1))

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
        for down, conv in zip(self.downs, self.convs):
            x = down(x)
            x = x + conv(x)
        pos = self._build_2d_sincos(x.size(2), x.size(3), x.device)
        x = self.norm(x + self.pos_embed(pos))
        return x



class PatchViT(nn.Module):
    def __init__(self, model_params):
        super(PatchViT, self).__init__()
        in_channels = model_params['in_channels']
        classes = model_params['out_channels']
        channels = model_params['channels']
        convs = model_params['convs']
        attns = model_params['attns']
        mlps = model_params['mlps']
        layers = model_params['layers']
        assert len(channels) == len(layers), "Channels length must match Layers length"
        assert len(convs) == len(attns) == len(mlps) == len(layers), \
            "Convs, Attns, MLPs must match Layers length"
        assert isinstance(attns, list) and all(isinstance(a, (list, int)) for a in attns), \
            "attns must be a list containing either integers or lists"

        self.CNN_embed = CNNEmbedding(in_channels, channels[0])
        self.downs = nn.ModuleList([
            DownSample(channels[i], channels[i+1], 2)
            for i in range(len(channels)-1) 
        ])
        self.layers = nn.ModuleList([
            nn.Sequential(*[
            Layer(channels[i], convs[i], attns[i], mlps[i], bias=False, 
                  HyperAttn=isinstance(attns[i], list))
            for _ in range(layers[i])
            ])
            for i in range(len(channels))
        ])
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.RMSNorm(channels[-1], elementwise_affine=False),
            nn.Linear(channels[-1], classes))

    def forward(self, x):
        # Input Embedding
        x = self.CNN_embed(x)

        # Layers
        x = self.layers[0](x)
        for down, layer in zip(self.downs, self.layers[1:]):
            x = down(x)
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
        "channels": [64, 128, 256, 512],
        "convs": [64, 64, 128, 128],
        "attns": [[256, 4], [256, 4], 8, 16],
        "mlps": [256, 512, 1024, 2048],
        "layers": [2, 2, 2, 2]
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