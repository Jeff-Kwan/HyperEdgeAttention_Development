import torch
from torch import nn

from ParallelLinear import ParallelLinear


class ParallelSwiGLU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_vectors, bias=True):
        super(ParallelSwiGLU, self).__init__()
        self.linear1 = ParallelLinear(in_channels, hidden_channels * 2, n_vectors, bias=bias)
        self.linear2 = ParallelLinear(hidden_channels, out_channels, n_vectors, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        # x shape: (N_parallel, Batch, Channels)
        z1, z2 = self.linear1(x).chunk(2, dim=-1)
        return self.linear2(z1 * self.act(z2))
    

class ConvSwiGLU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, bias=True):
        super(ConvSwiGLU, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels * 2, bias=bias)
        self.linear2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.conv = nn.Conv2d(hidden_channels*2, hidden_channels*2, 3, 1, 1, groups=hidden_channels*2, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x, x_shape):
        # x shape: (H*W, Batch, Channels)
        B, _, H, W = x_shape
        z = self.linear1(x)
        z1, z2 = self.conv(z.permute(1, 2, 0).view(B, -1, H, W)).view(B, -1, H*W).permute(2, 0, 1).chunk(2, dim=-1)
        return self.linear2(z1 * self.act(z2))


class DualPathBlock(nn.Module):
    def __init__(self, channels, hidden_channels, n_vectors, heads, bias=False):
        super(DualPathBlock, self).__init__()
        self.ImgCrossLatent = nn.MultiheadAttention(channels, heads, bias=False)
        self.LatentSelfMHA = nn.MultiheadAttention(channels, heads, bias=False)
        self.PSwiGLU = ParallelSwiGLU(channels, hidden_channels, channels, n_vectors, bias=bias)
        self.LatentCrossImg = nn.MultiheadAttention(channels, heads, bias=False)
        self.CSwiGLU = ConvSwiGLU(channels, hidden_channels, channels, bias=bias)

        self.norms = nn.ModuleList([nn.RMSNorm(channels) for _ in range(7)])

    def forward(self, x, z, x_shape):
        # Assume x, z come in (N, Batch, Channels format)

        # Cross Attention from Image to Latent
        x_norm = self.norms[0](x)
        z_norm = self.norms[1](z)
        z = z + self.ImgCrossLatent(z_norm, x_norm, x_norm, need_weights=False)[0]

        # Latent Self Attention
        z_norm = self.norms[2](z)
        z = z + self.LatentSelfMHA(z_norm, z_norm, z_norm, need_weights=False)[0]

        # Latent Parallel SwiGLU
        z_norm = self.norms[3](z)
        z = z + self.PSwiGLU(z_norm)

        # Cross Attention from Latent to Image
        x_norm = self.norms[4](x)
        z_norm = self.norms[5](z)
        x = x + self.LatentCrossImg(x_norm, z_norm, z_norm, need_weights=False)[0]

        # Image Conv SwiGLU
        x_norm = self.norms[6](x)
        x = x + self.CSwiGLU(x_norm, x_shape)
        return x, z
    

class ImgBlock(nn.Module):
    def __init__(self, channels, hidden_channels, heads, bias=False):
        super(ImgBlock, self).__init__()
        self.LatentCrossImg = nn.MultiheadAttention(channels, heads, bias=False)
        self.CSwiGLU = ConvSwiGLU(channels, hidden_channels, channels, bias=bias)
        self.norms = nn.ModuleList([nn.RMSNorm(channels) for _ in range(3)])

    def forward(self, x, z, x_shape):
        # Cross Attention from Latent to Image
        x_norm = self.norms[0](x)
        z_norm = self.norms[1](z)
        x = x + self.LatentCrossImg(x_norm, z_norm, z_norm, need_weights=False)[0]

        # Image Conv SwiGLU
        x_norm = self.norms[2](x)
        x = x + self.CSwiGLU(x_norm, x_shape)
        return x
    

class LatentBlock(nn.Module):
    def __init__(self, channels, hidden_channels, n_vectors, heads, bias=False):
        super(LatentBlock, self).__init__()
        self.ImgCrossLatent = nn.MultiheadAttention(channels, heads, bias=False)
        self.LatentSelfMHA = nn.MultiheadAttention(channels, heads, bias=False)
        self.PSwiGLU = ParallelSwiGLU(channels, hidden_channels, channels, n_vectors, bias=bias)
        self.norms = nn.ModuleList([nn.RMSNorm(channels) for _ in range(4)])

    def forward(self, x, z):
        # Imge Cross Attention from Image to Latent
        x_norm = self.norms[0](x)
        z_norm = self.norms[1](z)
        z = z + self.ImgCrossLatent(z_norm, x_norm, x_norm, need_weights=False)[0]

        # Latent self attention
        z_norm = self.norms[2](z)
        z = z + self.LatentSelfMHA(z_norm, z_norm, z_norm, need_weights=False)[0]

        # Latent Parallel SwiGLU
        z_norm = self.norms[3](z)
        z = z + self.PSwiGLU(z_norm)
        return z


class DenseConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, growth, bias=True):
        super(DenseConvEmbedding, self).__init__()
        assert out_channels % growth == 0, "Output channels must be divisible by growth factor."
        assert growth >= 8 and growth % 2 == 0, "Growth factor must be >= 8 and even."
        self.layers = out_channels // growth
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, growth*2, 1, 1, 0, bias=bias),
            nn.GroupNorm(4, growth*2))
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(growth*i, growth, 3, 1, 1, bias=bias),
                nn.GroupNorm(1, growth),
            )
            for i in range(1, self.layers)
        ])

    def forward(self, x):
        # x shape: (Batch, in_channels, H, W)
        x1, x2, x3, x4 = self.in_conv(x).chunk(4, dim=1)
        with torch.no_grad():
            h_frac = torch.linspace(0, 1, x.shape[2], device=x.device, requires_grad=False).view(1, 1, -1, 1)
            w_frac = torch.linspace(0, 1, x.shape[3], device=x.device, requires_grad=False).view(1, 1, 1, -1)
        x = torch.cat([x1*h_frac + x2*(1-h_frac), x3*w_frac + x4*(1-w_frac)], dim=1)
        for conv in self.convs:
            x = torch.cat([x, conv(x)], dim=1)
        return x


class SemanticViT(nn.Module):
    def __init__(self, model_params):
        super(SemanticViT, self).__init__()
        in_channels = model_params["in_channels"]
        n_embed = model_params["n_embed"]
        out_channels = model_params["out_channels"]
        num_layers = model_params["num_layers"]
        n_vectors = model_params["n_vectors"]
        dgrowth = model_params["dgrowth"]
        heads = model_params["heads"]

        self.latents = nn.Parameter(torch.randn(n_vectors, 1, n_embed))
        self.dense_embed = DenseConvEmbedding(in_channels, n_embed, dgrowth)
        self.in_norm = nn.Sequential(nn.Linear(n_embed, n_embed, bias=False),
                                     nn.RMSNorm(n_embed, elementwise_affine=False))
        # self.layers = nn.ModuleList([
        #     DualPathBlock(n_embed, n_embed, n_vectors, heads) 
        #     for _ in range(num_layers)
        # ])
        self.n_layers = num_layers
        self.latent_layers = nn.ModuleList([
            LatentBlock(n_embed, n_embed, n_vectors, heads) 
            for _ in range(num_layers+1)
        ])
        self.img_layers = nn.ModuleList([
            ImgBlock(n_embed, n_embed, heads) 
            for _ in range(num_layers)
        ])

        self.out_norm = nn.RMSNorm(n_embed, elementwise_affine=False)
        out_embed = (out_channels // n_embed + 1)
        out_embed = out_embed + out_embed % 2
        self.out_lin = ParallelLinear(n_embed, out_embed, n_vectors, bias=False)
        self.out = nn.Sequential(nn.RMSNorm(out_embed*n_vectors, elementwise_affine=False),
                                   nn.Linear(out_embed*n_vectors, out_channels))

    def forward(self, x):
        # Input Embedding
        x = self.dense_embed(x)
        x_shape = x.shape
        x = self.in_norm(x.view(x_shape[0], x_shape[1], -1).permute(2, 0, 1).contiguous())
        z = self.latents.repeat(1, x_shape[0], 1)

        # Dual Pathway ViT Blocks
        z = self.latent_layers[0](x, z)
        z = z - self.latents / (self.n_layers + 1)
        for i in range(self.n_layers):
            x = self.img_layers[i](x, z, x_shape)
            z = self.latent_layers[i+1](x, z)
            z = z - self.latents / (self.n_layers + 1)

        # Classifier Output
        z = self.out_lin(self.out_norm(z)).permute(1, 0, 2).reshape(x_shape[0], -1)
        y = self.out(z)
        return y
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parameters for SemanticViT
    B, H, W = 4, 224, 224
    model_params = {
        "in_channels": 3,  
        "n_embed": 96,   
        "out_channels": 1000,  
        "num_layers": 4,               
        "n_vectors": 128,              
        "dgrowth": 32,              
        "heads": 4                    
    }

    # Create random input tensor representing an image batch
    x = torch.randn(B, model_params['in_channels'], H, W).to(device)
    
    # Instantiate SemanticViT
    model = SemanticViT(model_params).to(device)
    
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