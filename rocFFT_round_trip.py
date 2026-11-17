"""
rocFFT round-trip via PyTorch
although this example uses the PyTorch FFT API,
the underlying FFT implementation is provided by
the ROCm rocFFT library

    PyTorch
       |
    rocFFT
       |
    HIP Runtime
       |
    AMD GPU

this exercise verifies that a forward FFT followed 
by an inverse FFT reconstructs the original signal
"""

import torch

device = torch.device('cuda')

# create a synthetic test signal
# in a real application this could represent:
#   - Audio samples
#   - Sensor measurements
#   - Radio signals
#   - Scientific data

N = 8192

signal = torch.randn(N, dtype=torch.float32, device=device)

# Forward FFT
# Converts the signal from the time domain into frequency domain
# Result:
#   spectrum[k]
# contrains the frequency content of the signal
spectrum = torch.fft.fft(signal)

# Inverse FFT
# Converts the frequency-domain representation
# back into the original time-domain signal
recovered = torch.fft.ifft(spectrum).real

# Numerical accuracy test
# Because floating-point arithmetic is not exact
# we expect a very small reconstruction error
# A correct FFT implementation should produce 
# an error closed to machine precision
error = torch.max(torch.abs(signal-recovered)).item()

print("=== rocFFT Round-Trip Test===")
print(f"Signal length: {N}")
print(f"Max error: {error:.2e}")
print(f"Status:   "
      f"{'PASS' if error < 1e-5 else 'FAIL'}"
      )