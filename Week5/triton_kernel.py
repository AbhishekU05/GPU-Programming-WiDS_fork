import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(X_ptr, Y_ptr, stride_x, stride_y, M,
                   BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    cols = offs

    mask = cols < M
    x_ptrs = X_ptr + row_id * stride_x + cols
    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))

    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)

    y = exp_x / denom
    y_ptrs = Y_ptr + row_id * stride_y + cols
    tl.store(y_ptrs, y, mask=mask)


def triton_softmax(x: torch.Tensor):
    assert x.is_cuda
    N, M = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(M)

    grid = (N,)
    softmax_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        M,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y
