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
    def __init__(self, in_channels, hidden_channels, out_channels, bias=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1,  bias=bias)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))
    

class PatchMHA(nn.Module):
    def __init__(self, patch_size, in_c, vec_embed, heads, bias=False):
        super(PatchMHA, self).__init__()
        assert vec_embed % heads == 0, "vec_embed must be divisible by heads."
        self.heads = heads
        self.head_dim = vec_embed // heads
        self.vec_embed = vec_embed
        self.patch_size = patch_size
        self.QKV = nn.Conv2d(in_c, vec_embed*3, patch_size, patch_size, 0, bias=bias)
        self.O = nn.ConvTranspose2d(vec_embed, in_c, patch_size, patch_size, 0, bias=bias)

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

        y = self.O(y.transpose(2, 3).reshape(B, self.vec_embed, H//self.patch_size, W//self.patch_size))
        return y


class Layer(nn.Module):
    def __init__(self, channels, n_embed, patches, heads, bias=False, last=False):
        super(Layer, self).__init__()
        self.PatchMHA = nn.ModuleList([
            PatchMHA(p, channels, n_embed, heads, bias=bias)
            for p in patches])

        self.CBlock = ConvBlock(channels, channels, channels, bias=bias)
        self.norms = nn.ModuleList([RMSNormTranspose(1, channels),
                                    RMSNormTranspose(1, channels)])

    def forward(self, x):      
        # Image Patched self attention
        x_norm = self.norms[0](x)
        for mha in self.PatchMHA:
            x = x + mha(x_norm)

        # Conv Block
        x_norm = self.norms[1](x)
        x = x + self.CBlock(x_norm)
        return x


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, init_patch):
        super(ConvEmbedding, self).__init__()
        assert out_channels % 4 == 0, "Embedding channels must be divisible by 4."
        self.in_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.mixnorm = nn.Sequential(
            nn.Conv2d(out_channels//2, out_channels, init_patch, init_patch, 0, bias=False),
            RMSNormTranspose(1, out_channels, elementwise_affine=False))

    def forward(self, x):
        # x shape: (Batch, in_channels, H, W)
        x1, x2, x3, x4 = self.in_conv(x).chunk(4, dim=1)
        with torch.no_grad():
            h_frac = torch.linspace(0, 1, x.shape[2], device=x.device, requires_grad=False).view(1, 1, -1, 1)
            w_frac = torch.linspace(0, 1, x.shape[3], device=x.device, requires_grad=False).view(1, 1, 1, -1)
        x = torch.cat([x1*h_frac + x2*(1-h_frac), x3*w_frac + x4*(1-w_frac)], dim=1)
        x = self.mixnorm(x)
        return x


class PatchViT(nn.Module):
    def __init__(self, model_params):
        super(PatchViT, self).__init__()
        in_channels = model_params['in_channels']
        init_patch = model_params['init_patch']
        n_channels = model_params['latent_channels']
        vec_embed = model_params['latent_dim']
        attn_patch = [p // init_patch for p in model_params['attn_patch']]
        out_channels = model_params['out_channels']
        num_layers = model_params['layers']
        heads = model_params['heads']

        self.conv_embed = ConvEmbedding(in_channels, n_channels, init_patch)

        self.n_layers = num_layers
        self.layers = nn.ModuleList([
            Layer(n_channels, vec_embed, attn_patch, heads, last=(i == num_layers)) 
            for i in range(num_layers+1)
        ])

        self.outpatch = nn.ModuleList([
            nn.Conv2d(n_channels, vec_embed, p, p, 0, bias=False)
            for p in attn_patch
        ])
        self.out = nn.Sequential(nn.RMSNorm(vec_embed*len(attn_patch), elementwise_affine=False),
                                   nn.Linear(vec_embed*len(attn_patch), out_channels))

    def forward(self, x):
        # Input Embedding
        x = self.conv_embed(x)

        # Dual Pathway ViT Blocks
        for layer in self.layers:
            x = layer(x)
        
        # Classifier Output
        y = []
        for outp in self.outpatch:
            y.append(torch.mean(outp(x), dim=(2,3)))
        y = torch.cat(y, dim=-1)
        y = self.out(y)
        return y
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parameters for PatchViT
    B, H, W = 8, 224, 224
    model_params = {
        'in_channels': 3,
        'init_patch': 4,
        'latent_channels': 128,
        'latent_dim': 256,
        'attn_patch': [4, 16],
        'out_channels': 1000,
        'layers': 8,
        'heads': 8,          
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