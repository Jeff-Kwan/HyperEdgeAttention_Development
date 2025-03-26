import torch
from torch import nn

from HyperEdgeAttentionVer1 import HyperEdgeAttention
import time


def measure_performance(model, dummy_input, num_runs=100):
    # Ensure model is on the same device as dummy_input and set it to eval mode
    device = dummy_input.device
    model = model.to(device)
    model.eval()

    # Warm up GPU (if applicable) to prevent startup overhead
    if device.type == "cuda":
        for _ in range(10):
            _ = model(dummy_input)
        torch.cuda.synchronize()

    total_time = 0.0

    if device.type == "cuda":
        # Reset peak memory stats before timing
        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model(dummy_input)
            end_event.record()

            torch.cuda.synchronize()  # Ensure events are recorded
            elapsed_time = start_event.elapsed_time(end_event)  # In milliseconds
            total_time += elapsed_time

        avg_time = total_time / num_runs
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

        print(f"Average inference time on CUDA: {avg_time:.2f} ms")
        print(f"Peak VRAM usage on CUDA: {peak_vram:.2f} MB")
        return avg_time, peak_vram
    else:
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds

        print(f"Average inference time on CPU: {avg_time:.2f} ms")
        return avg_time, None


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a dummy input tensor to measure inference time
    dummy_input = torch.randn(1, 32, 224, 224).to(device)

    # Create an instance of the model
    model = HyperEdgeAttention(32, 32, 4).to(device)

    # Measure the performance of the model
    measure_performance(model, dummy_input, num_runs=100)

    print("Now with compilation enabled:")
    model = torch.compile(model)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    measure_performance(model, dummy_input, num_runs=100)