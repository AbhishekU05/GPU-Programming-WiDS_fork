import torch
import time
from triton_kernel import triton_softmax

def check_correctness():
    x = torch.randn(1024, 1024, device="cuda")
    y_ref = torch.softmax(x, dim=1)
    y_tri = triton_softmax(x)

    max_err = (y_ref - y_tri).abs().max().item()
    print("Max error:", max_err)


def benchmark(fn, x, iters=100):
    for _ in range(10):
        fn(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) * 1000 / iters


if __name__ == "__main__":
    check_correctness()

    x = torch.randn(4096, 1024, device="cuda")

    t_triton = benchmark(triton_softmax, x)
    t_torch = benchmark(lambda z: torch.softmax(z, dim=1), x)

    print(f"Triton softmax time : {t_triton:.4f} ms")
    print(f"PyTorch softmax time: {t_torch:.4f} ms")
