import torch
from torch import nn
import torch.nn.functional as F

from .ParallelLinear import ParallelLinear

class RMSNormTranspose(nn.Module):
    def __init__(self, dim, features, eps=1e-6, elementwise_affine=True):
        super(RMSNormTranspose, self).__init__()
        self.dim = dim
        self.norm = nn.RMSNorm(features, eps, elementwise_affine)

    def forward(self, x):
        return self.norm(x.transpose(self.dim, -1)).transpose(self.dim, -1)


class ParallelSwiGLU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_vectors, bias=True):
        super(ParallelSwiGLU, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels * 2, bias=bias)
        self.linear2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        # x shape: (N_parallel, Batch, Channels)
        z1, z2 = self.linear1(x).chunk(2, dim=-1)
        return self.linear2(z1 * self.act(z2))
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, bias=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1,  bias=bias)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


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
            if hasattr(m, 'weight'):
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
            if hasattr(m, 'weight'):
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
    

class BidirectionalMHA(nn.Module):
    def __init__(self, patch_size, in_c, vec_embed, heads, bias=False):
        super(BidirectionalMHA, self).__init__()
        assert vec_embed % heads == 0, "vec_embed must be divisible by heads."
        self.heads = heads
        self.head_dim = vec_embed // heads
        self.vec_embed = vec_embed
        self.patch_size = patch_size
        self.ImgQKV = nn.Conv2d(in_c, vec_embed*3, patch_size, patch_size, 0, bias=bias)
        self.ImgO = nn.ConvTranspose2d(vec_embed, in_c, patch_size, patch_size, 0, bias=bias)
        self.LatQKV = nn.Linear(vec_embed, vec_embed*3, bias=bias)
        self.LatO = nn.Linear(vec_embed, vec_embed, bias=bias)

        for m in self.modules():
            if hasattr(m, 'weight'):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, z):
        # x shape: (Batch, Channels, H, W); z shape: (Batch, N, E)
        B, _, H, W = x.shape; N = z.shape[1]
        qx, kx, vx = self.ImgQKV(x).view(B, 3, self.heads, self.head_dim, -1).transpose(3, 4).unbind(dim=1)
        qz, kz, vz = self.LatQKV(z).view(B, N, self.heads, self.head_dim, 3).transpose(1, 2).unbind(dim=-1)

        qx, kx, vx, qz, kz, vz = map(lambda x: x.contiguous(), (qx, kx, vx, qz, kz, vz))
        yx = F.scaled_dot_product_attention(qx, kz, vz)
        yz = F.scaled_dot_product_attention(qz, kx, vx)

        yx = self.ImgO(yx.transpose(2, 3).reshape(B, self.vec_embed, H//self.patch_size, W//self.patch_size))
        yz = self.LatO(yz.transpose(2, 3).reshape(B, N, self.vec_embed))
        return yx, yz


class Layer(nn.Module):
    def __init__(self, channels, n_embed, n_vectors, patches, heads, bias=False, last=False):
        super(Layer, self).__init__()
        self.Img2Latent = nn.ModuleList([
            Patch2LatentMHA(p, channels, n_embed, heads, bias=bias)
             for p in patches])
        self.LatentSelfMHA = nn.MultiheadAttention(n_embed, heads, bias=False, batch_first=True)
        self.PSwiGLU = ParallelSwiGLU(n_embed, n_embed*4, n_embed, n_vectors, bias=bias)
        self.normsL = nn.ModuleList([RMSNormTranspose(1, channels)] + 
                                    [nn.RMSNorm(n_embed)] * 3)
        # self.BiMHA = nn.ModuleList([
        #     BidirectionalMHA(p, channels, n_embed, heads, bias=bias)
        #      for p in patches])
        
        self.last = last
        if not last:
            self.Latent2Img = nn.ModuleList([
                Latent2PatchMHA(p, channels, n_embed, heads, bias=bias)
                for p in patches])
            self.CBlock = ConvBlock(channels, channels, channels, bias=bias)
            self.normsI = nn.ModuleList([RMSNormTranspose(1, channels),
                                        nn.RMSNorm(n_embed),
                                        RMSNormTranspose(1, channels)])

    def forward(self, x, z):      
        # Latent self attention
        z_norm = self.normsL[2](z)
        z = z + self.LatentSelfMHA(z_norm, z_norm, z_norm, need_weights=False)[0]

        # Image Cross Attention from Image to Latent
        x_norm = self.normsL[0](x)
        z_norm = self.normsL[1](z)
        for i in range(len(self.Img2Latent)):
            z = z + self.Img2Latent[i](x_norm, z_norm)
            # Ax, Az = self.BiMHA[i](x_norm, z_norm)
            # x = x + Ax
            # z = z + Az

        if not self.last:
            # Cross Attention from Latent to Image
            x_norm = self.normsI[0](x)
            z_norm = self.normsI[1](z)
            for L2I in self.Latent2Img:
                x = x + L2I(x_norm, z_norm)

            # Image Conv SwiGLU
            x_norm = self.normsI[2](x)
            x = x + self.CBlock(x_norm)

        # Latent Parallel SwiGLU
        z_norm = self.normsL[3](z)
        z = z + self.PSwiGLU(z_norm)

        return x, z


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


class LatentViT(nn.Module):
    def __init__(self, model_params):
        super(LatentViT, self).__init__()
        in_channels = model_params['in_channels']
        init_patch = model_params['init_patch']
        n_channels = model_params['latent_channels']
        vec_embed = model_params['latent_dim']
        n_vectors = model_params['n_latents']
        attn_patch = [p // init_patch for p in model_params['attn_patch']]
        out_channels = model_params['out_channels']
        num_layers = model_params['layers']
        heads = model_params['heads']

        self.latents = nn.Parameter(torch.randn(n_vectors, vec_embed))
        self.conv_embed = ConvEmbedding(in_channels, n_channels, init_patch)
        self.Img2LatInit = nn.ModuleList([
            Patch2LatentMHA(p, n_channels, vec_embed, heads, bias=False)
            for p in attn_patch])

        self.n_layers = num_layers
        self.layers = nn.ModuleList([
            Layer(n_channels, vec_embed, n_vectors, attn_patch, heads, last=(i == num_layers)) 
            for i in range(num_layers+1)
        ])

        self.out_norm = nn.RMSNorm(vec_embed, elementwise_affine=False)
        out_embed = (out_channels // n_vectors + 1)
        out_embed = out_embed + out_embed % 2
        self.out_lin = ParallelLinear(vec_embed, out_embed, n_vectors, bias=False)
        self.out = nn.Sequential(nn.RMSNorm(out_embed*n_vectors, elementwise_affine=False),
                                   nn.Linear(out_embed*n_vectors, out_channels))
        # self.out = nn.Sequential(nn.RMSNorm(vec_embed, elementwise_affine=False),
        #                            nn.Linear(vec_embed, out_channels))

    def forward(self, x):
        # Input Embedding
        x = self.conv_embed(x)
        z = self.latents.expand(x.shape[0], -1, -1)
        
        # 1-head weighted sum to latents
        for init in self.Img2LatInit:
            z = z + init(x, z)

        # Dual Pathway ViT Blocks
        for layer in self.layers:
            x, z = layer(x, z)
        z = z - self.latents
        
        # Classifier Output
        z = self.out_lin(self.out_norm(z)).reshape(x.shape[0], -1)
        y = self.out(z)
        return y
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parameters for LatentViT
    B, H, W = 8, 224, 224
    model_params = {
        'in_channels': 3,
        'init_patch': 4,
        'latent_channels': 128,
        'latent_dim': 256,
        'n_latents': 256,
        'attn_patch': [16],
        'out_channels': 1000,
        'layers': 8,
        'heads': 8,          
    }

    # Create random input tensor representing an image batch
    x = torch.randn(B, model_params['in_channels'], H, W).to(device)
    
    # Instantiate LatentViT
    model = LatentViT(model_params).to(device)
    
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