import torch
import time
import json
from torchvision import transforms, datasets
from multiprocessing import cpu_count
from torch.nn.attention import sdpa_kernel, SDPBackend
from HAT import HAT_Classifier

def profile_training_run(model, optimizer, criterion, data_loader, 
                         use_autocast=False, channels_last=False):
    num_iters = len(data_loader)
    device = next(model.parameters()).device
    model.train()

    # --- Warm-up Phase ---
    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad()
        if use_autocast:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if i >= 10:  # Warm-up for ~10 iterations
            break

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # --- Profiling Phase ---
    total_data_loading_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_optim_time = 0.0

    start_cpu = time.perf_counter()
    start_gpu = torch.cuda.Event(enable_timing=True)
    end_gpu = torch.cuda.Event(enable_timing=True)
    start_gpu.record()

    for i, (inputs, targets) in enumerate(data_loader):
        # Data Loading (CPU timing)
        data_start = time.perf_counter()
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        data_end = time.perf_counter()
        data_loading_time = (data_end - data_start) * 1000.0  # ms

        # Forward Pass Timing
        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_fwd.record()
        optimizer.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        end_fwd.record()
        torch.cuda.synchronize(device)
        forward_time = start_fwd.elapsed_time(end_fwd)

        # Backward Pass Timing
        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)
        start_bwd.record()
        loss.backward()
        end_bwd.record()
        torch.cuda.synchronize(device)
        backward_time = start_bwd.elapsed_time(end_bwd)

        # Optimizer Step Timing
        start_opt = torch.cuda.Event(enable_timing=True)
        end_opt = torch.cuda.Event(enable_timing=True)
        start_opt.record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.zero_grad(set_to_none=True)
        optimizer.step()
        end_opt.record()
        torch.cuda.synchronize(device)
        optim_time = start_opt.elapsed_time(end_opt)

        total_data_loading_time += data_loading_time
        total_forward_time += forward_time
        total_backward_time += backward_time
        total_optim_time += optim_time

    avg_data_loading = total_data_loading_time / num_iters
    avg_forward = total_forward_time / num_iters
    avg_backward = total_backward_time / num_iters
    avg_optim = total_optim_time / num_iters

    # Run again to get total CPU and GPU time
    

    # for i, (inputs, targets) in enumerate(data_loader):
    #     inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    #     optimizer.zero_grad(set_to_none=True)
    #     if use_autocast:
    #         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    #             outputs = model(inputs)
    #             loss = criterion(outputs, targets)
    #     else:
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #     optimizer.step()

    end_gpu.record()
    torch.cuda.synchronize(device)
    total_gpu_time = start_gpu.elapsed_time(end_gpu)
    total_cpu_time = (time.perf_counter() - start_cpu) * 1000.0  # ms
    avg_cpu_iter = total_cpu_time / num_iters
    avg_gpu_iter = total_gpu_time / num_iters
    avg_total_iter = (avg_cpu_iter + avg_gpu_iter) / 2  # Average of CPU and GPU iteration times

    # Calculate unaccounted time (CPU side)
    unaccounted_time = avg_total_iter - (avg_data_loading + avg_forward + avg_backward + avg_optim)

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB

    print(f"Average Data Loading Time: {avg_data_loading:.2f} ms")
    print(f"Average Forward Pass Time: {avg_forward:.2f} ms")
    print(f"Average Backward Pass Time: {avg_backward:.2f} ms")
    print(f"Average Optimizer Step Time: {avg_optim:.2f} ms")
    print(f"Average CPU Iter Time: {avg_cpu_iter:.2f} ms")
    print(f"Average GPU Iter Time: {avg_gpu_iter:.2f} ms")
    print(f"Unaccounted Iter Time: {unaccounted_time:.2f} ms")
    print(f"Peak VRAM Usage: {peak_vram:.2f} MB")

    return avg_data_loading, avg_forward, avg_backward, avg_optim, peak_vram

if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 128
    num_iters = 20
    cpu_workers = min(max(1, cpu_count() - 1), 64)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dummy_dataset = datasets.FakeData(
        size=batch_size * num_iters,
        image_size=(3, 224, 224),
        num_classes=1000,
        transform=train_transforms
    )
    dummy_loader = torch.utils.data.DataLoader(
        dummy_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=cpu_workers, pin_memory=True, persistent_workers=True
    )

    with open('model/configs/HAT_Base.json', 'r') as f:
        config = json.load(f)
    model = HAT_Classifier(config).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # print("Baseline Training")
    # profile_training_run(model, optimizer, criterion, dummy_loader, use_autocast=False)

    # print("\nWith Compilation")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    # compiled_model = torch.compile(model)
    # profile_training_run(compiled_model, optimizer, criterion, dummy_loader, use_autocast=False)

    print("\nWith Autocast, MATH Attention")
    with sdpa_kernel(SDPBackend.MATH):
        compiled_model = torch.compile(model)
        profile_training_run(compiled_model, optimizer, criterion, dummy_loader, use_autocast=False)

    print("\nWith Autocast, EFFICIENT Attention")
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        compiled_model = torch.compile(model)
        profile_training_run(compiled_model, optimizer, criterion, dummy_loader, use_autocast=False)

    print("\nWith Autocast, CUDNN Attention")
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        compiled_model = torch.compile(model)
        profile_training_run(compiled_model, optimizer, criterion, dummy_loader, use_autocast=False)

    print("\nWith Autocast, FLASH Attention")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        compiled_model = torch.compile(model)
        profile_training_run(compiled_model, optimizer, criterion, dummy_loader, use_autocast=False)

    print("\nChannels Last Memory (Flash)")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        compiled_model = torch.compile(model)
        profile_training_run(compiled_model, optimizer, criterion, dummy_loader, use_autocast=False, channels_last=True)