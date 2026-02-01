import triton
import triton.language as tl

@triton.jit
def heat_kernel(
    T_ptr, Tn_ptr, P_ptr,
    H, W,
    alpha, dt,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    mask = offs < H * W
    i = offs // W
    j = offs % W

    center = tl.load(T_ptr + offs, mask)
    up = tl.load(T_ptr + offs - W, mask & (i > 0))
    down = tl.load(T_ptr + offs + W, mask & (i < H - 1))
    left = tl.load(T_ptr + offs - 1, mask & (j > 0))
    right = tl.load(T_ptr + offs + 1, mask & (j < W - 1))

    lap = up + down + left + right - 4.0 * center
    out = center + alpha * dt * lap + tl.load(P_ptr + offs, mask)

    tl.store(Tn_ptr + offs, out, mask)
