import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # TODO: add 10 to each element of x and store the result in z
    pass
