'''
HyperEdge Attention Transformer (HAT??) - PyTorch
'''
import torch
from torch import nn

from .HyperEdgeAttentionVer2 import HyperEdgeAttention


class RMSNormPermute(nn.Module):
    def __init__(self, num_features, dim, eps=1e-5, elementwise_affine=True,
                 device=None, dtype=None):
        """
        Wrapper for RMSNorm that permutes the input tensor before and after normalization,
        for normalizing across convolutional channels only.
        """
        super(RMSNormPermute, self).__init__()
        self.dim = dim
        self.norm = nn.RMSNorm(num_features, eps, elementwise_affine, device, dtype)

    def forward(self, x):
        dim = self.dim if self.dim >= 0 else x.dim() + self.dim
        permute_order = [i for i in range(x.dim()) if i != dim] + [dim]
        reverse_order = [permute_order.index(i) for i in range(x.dim())]
        x = self.norm(x.permute(*permute_order)).permute(*reverse_order)
        return x


class CSwiGLU(nn.Module):
    '''SiLU-Gated Convolutional Layer with InstanceNorm'''
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, bias=False):
        super(CSwiGLU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=bias),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels, bias=bias))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=bias),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels),
            nn.SiLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=out_channels, bias=bias),
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=bias))
        
        for m in self.modules():
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        y = self.conv3(self.conv1(x) * self.conv2(x))
        return y


class Block(nn.Module):
    '''Residual block with Linear Attention and SiLU-Gated Convolutional Layers'''
    def __init__(self, in_channels, edges, heads):
        super(Block, self).__init__()
        self.CS = CSwiGLU(in_channels, in_channels*4, in_channels, bias=False)
        self.HA = HyperEdgeAttention(in_channels, edges, heads, bias=False)
        self.norm1 = RMSNormPermute(in_channels, dim=1, elementwise_affine=False)
        self.norm2 = RMSNormPermute(in_channels, dim=1, elementwise_affine=False)
    
    def forward(self, x):
        x = self.norm1(x + self.HA(x))
        x = self.norm2(x + self.CS(x))
        return x
    

class HAT_Encoder(nn.Module):
    '''Hypergraph Attention Transformer Encoder'''
    def __init__(self, model_params):
        super(HAT_Encoder, self).__init__()
        in_channels = model_params['in_channels']
        init_patch = model_params['init_patch']
        channels = model_params['channels']
        heads = model_params['heads']
        edges = model_params['edges']
        depths = model_params['depths']
        layers = len(channels) - 1
        
        # Modules
        self.in_conv = nn.Conv2d(in_channels, channels[0], init_patch, stride=init_patch, padding=0, bias=None)
        self.blocks = nn.ModuleList([
            nn.Sequential(*[
                Block(channels[i], edges[i], heads[i])
                for _ in range(depths[i])])
            for i in range(layers)])
        self.downsample = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i+1], 2, stride=2, padding=0, bias=None)
            for i in range(layers)])
        
        self.bottleneck = nn.Sequential(*[Block(channels[-1], edges[-1], heads[-1]) for _ in range(depths[-1])])
        
        # Initizalizations
        nn.init.kaiming_uniform_(self.in_conv.weight, nonlinearity='linear')
        for d in self.downsample:
            nn.init.kaiming_uniform_(d.weight, nonlinearity='linear')
        
    def forward(self, x):
        # Patch Embedding
        x = self.in_conv(x)

        # Downsampling stage
        for block_seq, down in zip(self.blocks, self.downsample):
            x = block_seq(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)
        return x



class HAT_Classifier(nn.Module):
    '''Hypergraph Attention Transformer'''
    def __init__(self, model_params):
        super(HAT_Classifier, self).__init__() 
        in_channels = model_params['in_channels']
        out_channels = model_params['out_channels']
        init_patch = model_params['init_patch']
        channels = model_params['channels']
        heads = model_params['heads']
        edges = model_params['edges']
        depths = model_params['depths']
        layers = len(channels) - 1
        assert len(depths) == layers+1, "Number of depths not equal to number of layers + 1."
        assert len(edges) == layers+1, "Number of edges not equal to number of layers + 1."
        assert len(heads) == layers+1, "Number of heads not equal to number of layers + 1."

        self.encoder = HAT_Encoder(model_params)

        # Classifier Prediction head
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.RMSNorm(channels[-1], elementwise_affine=False),
            nn.Linear(channels[-1], out_channels, bias=False)
        )
        
    def encode(self, x):
        x = self.encoder(x)
        return x
        
    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Classifier
        x = self.out(x)
        return x
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Example Image tensor
    B, C, H, W = 1, 3, 224, 224
    x = torch.randn(B, C, H, W, device=device)

    # Example model
    import json
    model_params = json.load(open('model/configs/HAT_Base.json'))
    HAT = HAT_Classifier(model_params).to(device)
    HAT.eval()

    # Clear cache and reset memory stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Profile memory usage
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, 
                    torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        output = HAT(x)
        loss = output.sum()
        loss.backward()

    print(prof.key_averages().table(sort_by=f"{device}_time_total", row_limit=8))
    if torch.cuda.is_available():
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1048**2:.2f} MB")
    print("Total trainable parameters:", round(sum(p.numel() for p in HAT.parameters() if p.requires_grad)/1e6, 2), 'M')
    print("IO is size:", x.element_size() * x.nelement() / 1048 / 1048, 'MB')
    print("I/O has elements: ", round(output.nelement() / 1e6, 2), 'M')

