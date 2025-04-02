import torch
import json

from HAT import HAT_Classifier
import time


def measure_performance(model, dummy_input, num_runs=100, autocast=False):
    # Ensure model is on the same device as dummy_input and set it to eval mode
    device = dummy_input.device
    model = model.to(device)
    model.eval()

    # Warm up GPU (if applicable) to prevent startup overhead
    if device.type == "cuda":
        for _ in range(10):
            y = model(dummy_input)
            loss = y.sum()
            loss.backward()
        torch.cuda.synchronize()

    total_time = 0.0

    if device.type == "cuda":
        # Reset peak memory stats before timing
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            if autocast:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    y = model(dummy_input)
            else:
                y = model(dummy_input)
            loss = y.sum()
            loss.backward()
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
    dummy_input = torch.randn(64, 3, 256, 256).to(device)

    # Create an instance of the model
    config = json.load(open('model/configs/HAT_Base.json'))
    model = HAT_Classifier(config)

    # Measure the performance of the model
    print("Base Speed")
    measure_performance(model, dummy_input, num_runs=100)

    print("\nNow with compilation enabled")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    model = torch.compile(model)
    measure_performance(model, dummy_input, num_runs=100)

    print("\nNow with flash attention and autocast")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    measure_performance(model, dummy_input, num_runs=100, autocast=True)

    print("\nNow with channels last memory as well")
    dummy_input = dummy_input.to(memory_format=torch.channels_last).contiguous()
    measure_performance(model, dummy_input, num_runs=100, autocast=True)
