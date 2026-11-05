
import pytest
import torch

from kernels.puzzles.p01_const_add import add_kernel


def add_spec(x):
    return x + 10.0


@pytest.mark.parametrize("N0", [8, 32, 128])
def test_constant_add(N0):
    x = torch.rand(N0)
    z = torch.empty_like(x)

    add_kernel[(1,)](x, z, N0, B0=N0)

    torch.testing.assert_close(z, add_spec(x))