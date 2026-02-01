import torch
import ctypes

# Load CUDA kernel
lib = ctypes.cdll.LoadLibrary("./libsoftmax.so")

lib.softmax_forward.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
]

# Problem size
rows = 1024
cols = 512
iters = 200

x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
y_out = torch.empty_like(x)

for _ in range(10):
    torch.softmax(x, dim=1)
    lib.softmax_forward(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y_out.data_ptr()),
        rows,
        cols
    )

torch.cuda.synchronize()

def benchmark(fn):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms

# PyTorch baseline
torch_time = benchmark(lambda: torch.softmax(x, dim=1))

# CUDA kernel
cuda_time = benchmark(lambda: lib.softmax_forward(
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(y_out.data_ptr()),
    rows,
    cols
))

print(f"PyTorch softmax time   : {torch_time:.4f} ms")
print(f"CUDA kernel time       : {cuda_time:.4f} ms")
print(f"Speedup                : {torch_time / cuda_time:.2f}x")
