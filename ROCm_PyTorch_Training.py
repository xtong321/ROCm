"""
 Train ResNet-18 on CIFAR-10 for 2 epochs

 This exercise verifies that the complete PyTorch + ROCm
 training pipeline works correctly

 Components involved:

    Dataset loading (torchvision)
        |
    Data preprocessing (transform)
        |
    Neural network (ResNet-18)
        |
    Loss function
        |
    Backpropagation
        |
    Optimizer update
        |
    AMD GPU via ROCm

Successful execution demonstrates that the entire AI
software stack is functioning properly
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import time
import gc
import os, sys
from contextlib import redirect_stdout
import warnings
from pathlib import Path
import tarfile

# Supress con-critical warnings to keep notebook output clean
warnings.filterwarning("ignore")

# Run Python garbage collection before starting
gc.collect()

device = torch.device('cuda')

# Data preprocessing pipeline
# ToTensor():
#   Converts image pixels into PyTorch tensors
# Normalize():
#   Scales values into a range that is easier for neural networks to learn from
# 
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Prepare the CIFAR-10 dataset
# CIFAR-10 contains:
#   - 60,000 color images
#   - 10 object categories
#   - 32x32 pixel resolution
# if the dataset is not already present,
# extract it from archive
DATA_ROOT = Path("opt/project/data")
ARCHIVE = DATA_ROOT / "cifar-10-python.tar.gz"
EXTRACTED = DATA_ROOT / "cifar-10-batches-py"

if not EXTRACTED.exist():
    print("Extracting CIFAR-10 dataset...")

    if not ARCHIVE.exist():
        raise RuntimeError(
            f"Cannot find {ARCHIVE}"
        )
    
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(DATA_ROOT)
    
    print("Extraction complete")
else:
    print("Found existing extracted dataset.")

print("Loading CIFAR-10 dataset")
trainset = torchvision.dataset.CIFAR10(
    root=str(DATA_ROOT),
    train=True,
    download=False,
    transform=transform
)

# Reset ROCm memory statistics before benchmarking
# This allow us to report memory usage for the current training session only

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# DataLoader feeds mini-batches into the model
# batch_size = 128
#   Process 128 images at a time
# shuffle=True:
#   Randomize sample order each epoch
# num_workers=2:
#   Load data in parallel
# pin_memory=True:
#   Speed up CPU->GPU transfer
loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Create a ResNet-18 model
model = torchvision.model.resnet18(
    num_classes=10
).to(device)

# CrossEntropyLoss measure prediction error
# SGD perform parameter update using gradient descent
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
    momentum=0.9
)

print("]n===Training ResNet-18 on CIFAR-10 ===")

# Training loop
# Each epoch process the entire dataset once
for epoch in range(2):
    model.train()

    running_loss = 0.0

    torch.cuda.synchronize()

    t0 = time.perf_counter()

    for i, (inputs, labels) in enumerate(loader):
        # moving training data to GPU memory
        inputs, labels = input.to(device), labels.to(device)

        # Forward pass
        # input -> NN -> predication
        optimizer.zero_grad()

        outputs = model(input)

        # Compute error
        loss = criterion(outputs, labels)

        # Backpropagation:
        # Compute gradients for all trainable parameters
        loss.backward()

        # Update model weights
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 100 == 0:
            print(
                f"Epoch {epoch+1},"
                f"Batch {i+1}/{len(loader)}, "
                f"Loss: {running_loss/100:.3f}"
            )
            running_loss = 0.0

    torch.cuda.synchronize()

    elapse = time.perf_counter() - t0

    print(
        f"Epoch {epoch+1} complete --"
        f"{elapsed:.1f}s"            
    )
    
print("\nTraining complete")