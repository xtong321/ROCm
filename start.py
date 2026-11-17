# confirm Python, PyTorch and ROCm/HIP are installed and 
# that PyTorch can see the AMD GPU
# before running any execise

import sys

print(sys.version)
print(sys.executable)

import torch

print(torch.__version__)
print(torch.version.hip)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


# verify ROCm PyTorch build
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"HIP available:   {torch.cuda.is_available()}")
print(f"HIP version:     {torch.version.hip}")
print(f"GPU:             {torch.cuda.get_device.properties(0).total_memory/(2**30):.1f} GB")