import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    offsets = tl.arange(0, B0)
    x = tl.load(x_ptr + offsets)
    z = x + 10.0
    tl.store(z_ptr + offsets, z)
    
