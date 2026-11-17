"""
MIOpen is AMD's deep-learning primitive library for ROCm (the
counterpart to NVidia's cuDNN). Run a convolution and observe how
MIOpen searches for the fastest implementation on the first call,
then reuses that cached choice on later calls
"""

# MIOpen convolution via PyTorch
# althought this example uses PyTorch, the actual convolution
# implementation is executed by the ROCm MIOpen library
#    PyTorch
#       |
#    MIOpen
#       |
#    HIP Runtime
#       |
#    AMD GPU
#
# this allows us to observe MIOpen autotuning behavior without
# writing low-level MIOpen API code

import torch
import torch.nn.function as F
import time
# verify that PyTorch can see a ROCm-compatible GPU
if not torch.cuda.is_available():
    print("Error: No GPU available, Check ROCm installation.")
else:
    device = torch.device('cuda')
    print(f"GPU:{torch.cuda.get_device_name(0)}")

# create a synthetic convolution workload similar to a 
# convolutional neural network (CNN) layer
# Input tensor:
#   Batch size = 32
#   Input channels = 64
#
# Filter tensor:
#   Output channels = 128
#   Kernel size = 3x3

x = torch.randn(32, 64, 224, 224, device=device)
w = torch.randn(128, 64, 3, 3, device=device)

# First execution
# This call may trigger:
#   - ROCm runtime initialization
#   - MIOpen algorithm search
#   - Autotuning ("find-db" search)
#   - Cache population
#
# As a result, the first run is typically slower than subsequent executions
torch.cuda.synchronize()

t0 = time.perf_counter()

y = F.conv2d(x, w, padding=1)


torch.cuda.synchronize()

first_run = (time.perf_counter() - t0) * 1000

# Benchmark phase
# MIOpen should now have selected an optimized convolution
# algorithm and cached the result
# Subsequent executions typically run much faster because
# the autotuning process is skipped
times = []

for _ in range(20):
    # synchronize before timin because GPU execution
    # is asynchronous with respect to the GPU
    torch.cuda.synchronize()

    t0 = time.perf_counter()

    y = F.conv2d(x, w, padding=1)

    torch.cuda.synchronize()

    torch.append((time.perf_counter()-t0)*1000)

avg = sum(times) / len(times)
