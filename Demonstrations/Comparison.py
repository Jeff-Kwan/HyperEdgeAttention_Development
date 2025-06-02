#!/usr/bin/env python
"""
benchmark_hyperedge_vs_mha.py
 
Compare HyperEdgeAttention with a plain MultiheadAttention “reshape & self-attend”
baseline on an image-like feature map.

Example:
    python benchmark_hyperedge_vs_mha.py \
        --batch 8 --h 32 --w 32 --c 256 --heads 8 --edges 64 --iters 200 --device cuda
"""
import argparse
import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils import benchmark

# ----------  Model definitions -------------------------------------------------
class HyperEdgeAttention(nn.Module):
    '''HyperEdge partitioning and attention.'''
    def __init__(self, channels, edges, heads, bias=False):
        super().__init__()
        assert channels % heads == 0, "channels not divisible by heads"
        self.edges = edges
        self.hyperedge = nn.Linear(channels, edges, bias=bias)
        self.MHA = nn.MultiheadAttention(channels, heads, bias=bias, batch_first=True)
        self.norm = nn.RMSNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)        # (B, N, C)
        weights = self.hyperedge(x)                           # (B, N, E)
        weights = torch.relu(weights) * torch.softmax(weights, dim=-1) * (
            self.edges / (H * W)
        )
        z = self.norm(weights.transpose(1, 2) @ x)            # (B, E, C)
        y, _ = self.MHA(x, z, z, need_weights=False)
        return y.permute(0, 2, 1).reshape(B, C, H, W)

class ReshapeSelfAttention(nn.Module):
    """Baseline: flatten H✕W and run MultiheadAttention on itself."""
    def __init__(self, channels, heads, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, heads, bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        z = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        out, _ = self.mha(z, z, z, need_weights=False)
        return out.permute(0, 2, 1).reshape(B, C, H, W)

# ----------  Helpers -----------------------------------------------------------
@torch.no_grad()
def param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def _benchmark_cpu(fn, inputs, iters, desc):
    t = benchmark.Timer(
        stmt="fn(inp)", globals={"fn": fn, "inp": inputs}, num_threads=torch.get_num_threads()
    )
    res = t.timeit(iters)
    print(f"{desc:<35} | {res.median * 1e3:8.3f} ms (median of {iters})")

def _benchmark_gpu(fn, inputs, iters, desc, backward=False):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        starter.record()
        out = fn(inputs)
        if backward:
            out.mean().backward()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms
    print(f"{desc:<35} | {torch.tensor(times).median():8.3f} ms (median of {iters})")
    print(f"{' ' * 35} | peak memory {torch.cuda.max_memory_allocated() / 1e6:8.1f} MB")

def run_single(model, x, device, iters, backward):
    model.eval()
    model = model.to(device)
    x = x.to(device).requires_grad_(backward)
    fn = lambda inp: model(inp)
    torch.cuda.empty_cache()
    is_gpu = device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=is_gpu):
        if is_gpu:
            _benchmark_gpu(fn, x, iters, model.__class__.__name__, backward)
        else:
            _benchmark_cpu(fn, x, iters, model.__class__.__name__)

def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    B, C, H, W = args.batch, args.c, args.h, args.w

    x = torch.randn(B, C, H, W)

    models = [
        HyperEdgeAttention(C, args.edges, args.heads, bias=args.bias),
        ReshapeSelfAttention(C, args.heads, bias=args.bias),
    ]

    if args.compile:
        print("Compiling models with torch.compile()...")
        for i, m in enumerate(models):
            models[i] = torch.compile(m)

    print(f"\nInput  : (B={B}, C={C}, H={H}, W={W}) on device={device}")
    print(f"Iter   : {args.iters}  | Backward = {args.backward}\n")
    for m in models:
        print(f"{m.__class__.__name__} parameters: {param_count(m):,}")
    print()

    for m in models:
        run_single(m, x, device, args.iters, args.backward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--h", type=int, default=128, help="feature map height")
    parser.add_argument("--w", type=int, default=128, help="feature map width")
    parser.add_argument("--c", type=int, default=256, help="channels")
    parser.add_argument("--heads", type=int, default=8, help="attention heads")
    parser.add_argument("--edges", type=int, default=256, help="hyperedges")
    parser.add_argument("--iters", type=int, default=100, help="benchmark iterations")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--bias", default=False, help="include bias terms")
    parser.add_argument("--backward", default=True, help="include backward pass")
    parser.add_argument("--compile", default=True, help="compile models with torch.compile")
    main(parser.parse_args())
