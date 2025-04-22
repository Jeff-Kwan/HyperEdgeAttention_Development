import torch
from torch import nn
from torch.nn import functional as F

from .ParallelLinear import ParallelLinear

class RMSNormTranspose(nn.Module):
    def __init__(self, dim, features, eps=1e-6, elementwise_affine=True):
        super(RMSNormTranspose, self).__init__()
        self.dim = dim
        self.norm = nn.RMSNorm(features, eps, elementwise_affine)

    def forward(self, x):
        return self.norm(x.transpose(self.dim, -1)).transpose(self.dim, -1)

class SwiGLU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, bias=True):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels * 2, bias=bias)
        self.linear2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.act = nn.SiLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (Batch, N, Channels)
        z1, z2 = self.linear1(x).chunk(2, dim=-1)
        return self.linear2(z1 * self.act(z2))
    

class DenseConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, growth, bias=True):
        super(DenseConvEmbedding, self).__init__()
        assert out_channels > 8 and (out_channels - 8) % growth == 0, \
        "Output channels - 8 must be divisible by growth factor and larger than 8."
        self.layers = (out_channels - 8) // growth
        intermediate_c = [8 + growth * i for i in range(self.layers)]
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1, 1, 0, bias=bias),
            nn.GroupNorm(4, 16))
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(intermediate_c[i], growth, 3, 1, 1, bias=bias),
                nn.GroupNorm(1, growth),
            )
            for i in range(self.layers)
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


    
class CSwiGLU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, bias=True):
        super(CSwiGLU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels*2, 1, 1, 0, bias=bias),
            nn.Conv2d(hidden_channels*2, hidden_channels*2, 3, 1, 1, groups=hidden_channels*2, bias=bias))
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        x1, x2 = self.conv1(x).chunk(2, dim=1)
        return self.conv2(x1 * self.act(x2))


class Latent2PatchMHA(nn.Module):
    def __init__(self, patch_size, in_c, vec_embed, heads, bias=False):
        super(Latent2PatchMHA, self).__init__()
        assert vec_embed % heads == 0, "vec_embed must be divisible by heads."
        self.heads = heads
        self.head_dim = vec_embed // heads
        self.patch_size = patch_size
        self.vec_embed = vec_embed
        self.Q = nn.Conv2d(in_c, vec_embed, patch_size, patch_size, 0, bias=bias)
        self.KV = nn.Linear(vec_embed, vec_embed * 2, bias=bias)
        self.O = nn.ConvTranspose2d(vec_embed, in_c, patch_size, patch_size, 0, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, z):
        # x shape: (Batch, Channels, H, W), z shape: (Batch, N, E)
        B, _, H, W = x.shape
        q = self.Q(x).view(B, self.heads, self.head_dim, -1).transpose(2, 3)
        k, v = self.KV(z).view(B, -1, self.heads, self.head_dim, 2).transpose(1, 2).unbind(dim=-1)

        q, k, v = map(lambda x: x.contiguous(), (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)

        y = self.O(y.transpose(2, 3).reshape(B, self.vec_embed, H//self.patch_size, W//self.patch_size))
        return y
    

class Patch2LatentMHA(nn.Module):
    def __init__(self, patch_size, in_c, vec_embed, heads, bias=False):
        super(Patch2LatentMHA, self).__init__()
        assert vec_embed % heads == 0, "vec_embed must be divisible by heads."
        self.heads = heads
        self.head_dim = vec_embed // heads
        self.vec_embed = vec_embed
        self.Q = nn.Linear(vec_embed, vec_embed, bias=bias)
        self.KV = nn.Conv2d(in_c, vec_embed*2, patch_size, patch_size, 0, bias=bias)
        self.O = nn.Linear(vec_embed, vec_embed, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, z):
        # x shape: (Batch, Channels, H, W), z shape: (Batch, N, E)
        B = x.shape[0]; N = z.shape[1]
        q = self.Q(z).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.KV(x).view(B, 2, self.heads, self.head_dim, -1).transpose(3, 4).unbind(dim=1)

        q, k, v = map(lambda x: x.contiguous(), (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)

        y = self.O(y.transpose(2, 3).reshape(B, N, self.vec_embed))
        return y
    

class Layer(nn.Module):
    def __init__(self, patches, in_c, vec_embed, heads, bias, last=False):
        super(Layer, self).__init__()
        self.last = last

        self.SwiGLU = SwiGLU(vec_embed, vec_embed, vec_embed, bias)
        self.latentMHA = nn.MultiheadAttention(vec_embed, heads, bias=False, batch_first=True)
        self.patch2latents = nn.ModuleList([
            Patch2LatentMHA(patches[i], in_c, vec_embed, heads, bias) 
            for i in range(len(patches))
        ])

        self.z_norms = [nn.RMSNorm(vec_embed)] * 3
        self.x_norms = [RMSNormTranspose(1, in_c)] * 2

        if not last:
            self.latent2patchs = nn.ModuleList([
                Latent2PatchMHA(patches[i], in_c, vec_embed, heads, bias) 
                for i in range(len(patches))
            ])
            self.CSwiGLU = CSwiGLU(in_c, in_c, in_c, bias)

            self.x_norms += [RMSNormTranspose(1, in_c)]
            self.z_norms += [nn.RMSNorm(vec_embed)]
            
        self.x_norms = nn.ModuleList(self.x_norms)
        self.z_norms = nn.ModuleList(self.z_norms)


    def forward(self, x, z):
        # x shape: (Batch, Channels, H, W), z shape: (Batch, N, E)

        # Patch to Latents
        x_norm = self.x_norms[0](x)
        z_norm = self.z_norms[0](z)
        for p2l in self.patch2latents:
            z = z + p2l(x_norm, z_norm)

        # Latent Self Attention & SwiGLU
        z_norm = self.z_norms[1](z)
        z = z + self.latentMHA(z_norm, z_norm, z_norm, need_weights=False)[0]

        z_norm = self.z_norms[2](z)
        z = z + self.SwiGLU(z_norm)

        # Latents to Patch
        if not self.last:
            x_norm = self.x_norms[1](x)
            z_norm = self.z_norms[3](z)
            for l2p in self.latent2patchs:
                x = x + l2p(x_norm, z_norm)
            x_norm = self.x_norms[2](x)
            x = x + self.CSwiGLU(x_norm)

        return x, z
    

class LatentViT(nn.Module):
    def __init__(self, model_params):
        super(LatentViT, self).__init__()
        in_c = model_params['in_channels']
        n_c = model_params['latent_channels']
        n_embed = model_params['latent_dim']
        n_vectors = model_params['n_latents']
        patches = model_params['patches']
        out_c = model_params['out_channels']
        layers = model_params['layers']
        heads = model_params['heads']
        dgrowth = model_params['dgrowth']
        bias = False
        assert n_embed % heads == 0, "latent_dim must be divisible by heads."

        self.dense_embed = DenseConvEmbedding(in_c, n_c, dgrowth, bias)
        self.in_norm = nn.Sequential(nn.Conv2d(n_c, n_c, 1, 1, 0, bias=False),
                                     RMSNormTranspose(1, n_c, elementwise_affine=False))
        self.latents = nn.Parameter(torch.randn(1, n_vectors, n_embed))
        self.layers = nn.ModuleList([
            Layer(patches, n_c, n_embed, heads, bias, last=(i == layers-1)) 
            for i in range(layers)
        ])
        
        self.out_norm = nn.RMSNorm(n_embed, elementwise_affine=False)
        out_embed = (out_c // n_embed + 1)
        out_embed = out_embed + out_embed % 2
        self.out_lin = ParallelLinear(n_embed, out_embed, n_vectors, bias=False)
        self.out = nn.Sequential(nn.RMSNorm(out_embed*n_vectors, elementwise_affine=False),
                                   nn.Linear(out_embed*n_vectors, out_c))

    def forward(self, x):
        # x shape: (Batch, Channels, H, W), z shape: (Batch, N, E)
        x = self.dense_embed(x)
        x = self.in_norm(x)
        z = self.latents.expand(x.shape[0], -1, -1)

        for layer in self.layers:
            x, z = layer(x, z)

        z = z - self.latents

        z = self.out_lin(self.out_norm(z).transpose(0,1)).transpose(0,1).reshape(x.shape[0], -1)
        y = self.out(z)
        return y


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 8, 3, 224, 224
    N, E = 128, 64

    model = LatentViT({
        'in_channels': C,
        'latent_channels': 32,
        'latent_dim': E,
        'n_latents': N,
        'patches': [4, 16],
        'out_channels': 1000,
        'layers': 9,
        'heads': 8,
        'dgrowth': 4
    }).to(device)

    x = torch.randn(B, C, H, W).to(device)


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