import torch
import ctypes
import numpy as np

# Load shared library
lib = ctypes.cdll.LoadLibrary("./libsoftmax.so")

lib.softmax_forward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
]

# Test sizes
rows = 256
cols = 512

# Input
x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
y_ref = torch.softmax(x, dim=1)

# Output buffer
y_out = torch.empty_like(x)

# Launch kernel
lib.softmax_forward(
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(y_out.data_ptr()),
    rows,
    cols
)

torch.cuda.synchronize()

# Error check
max_err = (y_ref - y_out).abs().max().item()
mean_err = (y_ref - y_out).abs().mean().item()

print("Max error :", max_err)
print("Mean error:", mean_err)

assert max_err < 1e-5
