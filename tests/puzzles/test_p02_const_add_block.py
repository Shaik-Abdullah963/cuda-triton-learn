import pytest
import torch
import triton

from kernels.puzzles.p02_const_add_block import add_mask2_kernel


def add2_spec(x):
    return x + 10.0


N0 = 200


@pytest.mark.parametrize(
    "B0",
    [
        8,    # divides 200 evenly: 25 full blocks, no partial block
        32,   # 7 blocks, last one partial
        64,   # 4 blocks, last one partial
        128,  # 2 blocks, last one partial
        256,  # 1 block, bigger than N0, mostly masked
    ],
)
def test_add_mask2(B0):
    x = torch.rand(N0)
    z = torch.empty_like(x)

    grid = (triton.cdiv(N0, B0),)
    add_mask2_kernel[grid](x, z, N0, B0=B0)

    torch.testing.assert_close(z, add2_spec(x))


@pytest.mark.parametrize("B0", [32, 64, 256])
def test_add_mask2_does_not_write_past_end(B0):
    # z is the first N0 elements of a bigger buffer. If the kernel stores
    # without a mask, it overwrites the extra elements after z.
    sentinel = -1.0
    buffer = torch.full((N0 + B0,), sentinel)
    z = buffer[:N0]
    x = torch.rand(N0)

    grid = (triton.cdiv(N0, B0),)
    add_mask2_kernel[grid](x, z, N0, B0=B0)

    torch.testing.assert_close(z, add2_spec(x))
    assert torch.all(buffer[N0:] == sentinel), "kernel wrote past the end of z"
