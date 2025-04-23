import torch
from torch import nn

class ParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, num_parallel, bias=True):
        super(ParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_parallel = num_parallel

        # Initialize the weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(num_parallel, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_parallel, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        # Kaiming uniform initialization for weights in parallel
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = fan_in ** -0.5
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        '''Assume x is in form (Batch, N_parallel, Channels)'''
        # y = torch.einsum('bnc,ncd->bnd', x, self.weight)
        y = torch.matmul(x.transpose(0, 1), self.weight).transpose(0, 1)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)
        return y
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Test ParallelLinear
    B, N, C = 64, 256, 128  # batch size, sequence length, channel features

    x = torch.randn(B, N, C).to(device)
    model = ParallelLinear(C, C, num_parallel=N).to(device)

    # Profile the forward pass
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