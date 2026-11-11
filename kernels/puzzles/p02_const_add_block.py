import triton
import triton.language as tl


@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * B0 + tl.arange(0, B0)
    mask = offsets < N0
    x = tl.load(x_ptr + offsets, mask=mask)
    z = x + 10.0
    tl.store(z_ptr + offsets, z, mask=mask)
