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
    def __init__(self, patch, in_channels, hidden_channels, out_channels, bias=True):
        super(ConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels, patch, patch, 0, bias=bias),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=bias),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=bias),
            nn.Conv2d(hidden_channels, out_channels, patch, patch, 0, bias=bias))
        nn.init.kaiming_uniform_(self.convs[0].weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.convs[-1].weight, nonlinearity='linear')

    def forward(self, x):
        return self.convs(x)
    

class CSwiGLU(nn.Module):
    def __init__(self, in_c, h_c, out_c, bias=True):
        super(CSwiGLU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, h_c*2, 1, 1, 0, bias=bias),
            nn.Conv2d(h_c*2, h_c*2, 3, 1, 1, bias=bias, groups=h_c*2))
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(h_c, out_c, 1, 1, 0, bias=bias)

    def forward(self, x):
        x1, x2 = self.conv1(x).chunk(2, dim=1)
        return self.conv2(self.act(x1) * x2)


class PatchMHA(nn.Module):
    def __init__(self, patch, in_c, patch_c, heads, bias=False):
        super(PatchMHA, self).__init__()
        assert patch_c % heads == 0, "patch_c must be divisible by heads."
        self.heads = heads
        self.head_dim = patch_c // heads
        self.patch_c = patch_c
        self.patch = patch
        self.QKV = nn.Conv2d(in_c, patch_c*3, patch, patch, 0, bias=bias)
        self.O = nn.ConvTranspose2d(patch_c, in_c, patch, patch, 0, bias=bias)

        for m in self.modules():
            if hasattr(m, 'weight'):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (Batch, Channels, H, W)
        B, _, H, W = x.shape
        q, k, v = self.QKV(x).view(B, 3, self.heads, self.head_dim, -1).transpose(3, 4).unbind(dim=1)

        q, k, v = map(lambda x: x.contiguous(), (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)

        y = self.O(y.transpose(2, 3).reshape(B, self.patch_c, H//self.patch, W//self.patch))
        return y


class Layer(nn.Module):
    def __init__(self, channels, patches, heads, bias=False):
        super(Layer, self).__init__()
        self.ConvBlock = nn.Sequential(
            RMSNormTranspose(1, channels[1]),
            ConvBlock(patches[0], channels[1], channels[0], channels[1], bias=bias))
        self.CSwiGLU = nn.Sequential(
            RMSNormTranspose(1, channels[1]),
            CSwiGLU(channels[1], channels[1] * 4, channels[1], bias=bias))
        self.PatchMHA = nn.Sequential(
            RMSNormTranspose(1, channels[1]),
            PatchMHA(patches[2], channels[1], channels[2], heads, bias=bias))


    def forward(self, x):      
        # Parallel Blocks
        x = x + self.ConvBlock(x) + self.CSwiGLU(x) + self.PatchMHA(x)
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
        out_channels = model_params['out_channels']
        patches = model_params['patches']
        channels = model_params['latent_channels']
        heads = model_params['heads']
        num_layers = model_params['layers']

        assert type(patches) == list and len(patches)==3, "Patches must be a list of length 3."
        assert type(channels) == list and len(channels)==3, "Channels must be a list of length 3."
        patches[0] = patches[1]//patches[0]  # Sub-Token expansion ratio
        patches[2] = patches[2]//patches[1]  # Patch Global MHA ratio
        

        self.conv_embed = ConvEmbedding(in_channels, channels[1], patches[1])
        self.layers = nn.ModuleList([
            Layer(channels,  patches, heads)
            for _ in range(num_layers)])
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.RMSNorm(channels[1], elementwise_affine=False),
            nn.Linear(channels[1], out_channels))

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
        'patches': [2, 8, 16],
        'latent_channels': [32, 128, 256],
        'heads': 8,
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