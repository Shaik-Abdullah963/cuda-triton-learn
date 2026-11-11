# Triton Puzzles: Tests and Kernels

My solutions to the Triton Puzzles (https://github.com/gpu-mode/Triton-Puzzles), worked test-first:
write the test, see it fail, write the kernel, see it pass.

For the concepts behind these (grids, program instances, `tl.program_id`, blocks, masks, pointers), see [kernel-fundamentals.md](kernel-fundamentals.md).

Run any puzzle's tests on the CPU with:

```
TRITON_INTERPRET=1 .venv/bin/pytest tests/puzzles/<test file> -v
```

| # | Puzzle | Kernel | Test |
|---|---|---|---|
| 1 | Constant Add | `kernels/puzzles/p01_const_add.py` | `tests/puzzles/test_p01_const_add.py` |
| 2 | Constant Add Block | `kernels/puzzles/p02_const_add_block.py` | `tests/puzzles/test_p02_const_add_block.py` |

---

## Puzzle 1: Constant Add

### The task

Add 10 to every element of a vector:

$$z_i = x_i + 10$$

The block size `B0` equals the vector length `N0`, so **one program instance covers the whole vector**.

### The test

```python
@pytest.mark.parametrize("N0", [8, 32, 128])
def test_constant_add(N0):
    x = torch.rand(N0)
    z = torch.empty_like(x)

    add_kernel[(1,)](x, z, N0, B0=N0)

    torch.testing.assert_close(z, add_spec(x))
```

- **`parametrize` over `N0`:** runs the test for three vector sizes. Each must be a power of 2, because `B0 = N0` and `tl.arange` needs a power of 2.
- **`torch.empty_like(x)`:** the output buffer. It's uninitialized, so if the kernel doesn't write to it, the test sees garbage and fails. That's what made the first run fail with huge values like `5e35`.
- **Grid `(1,)`:** one program instance, because one block of size `N0` covers everything.
- **`B0=N0`:** the test chooses the block size. The kernel doesn't.
- **`assert_close`:** compares every element of `z` with `x + 10`, within a small float tolerance.

### The kernel

```python
@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    offsets = tl.arange(0, B0)
    x = tl.load(x_ptr + offsets)
    z = x + 10.0
    tl.store(z_ptr + offsets, z)
```

- **`offsets = tl.arange(0, B0)`:** the positions `0 .. B0-1`. It must use `B0`, not `N0`: `tl.arange` needs a `tl.constexpr` size, and `N0` is a runtime value.
- **`tl.load(x_ptr + offsets)`:** `x_ptr` is the address of `x[0]`. Adding `offsets` gives a block of addresses, and `tl.load` reads them into registers. The result is named `x`.
- **`z = x + 10.0`:** adds 10 to every element of the block at once.
- **`tl.store(z_ptr + offsets, z)`:** writes the block into `z`'s memory. It returns nothing.

### Why no `program_id` and no mask

- There's only one instance, so `tl.program_id(0)` is always 0. The offsets already start at 0.
- `B0 = N0`, so every offset is inside the vector. Nothing needs masking.

### Mistakes I made along the way

- `tl.arange(0, N0)` failed with `arange's arguments must be of type tl.constexpr`. The fix was to use `B0`.
- Assigning the load result to `z_ptr` overwrote the output pointer. Pointers and values need different names.
- Using `x` before defining it in the kernel. The kernel's `x` is the loaded block, not the test's tensor.

---

## Puzzle 2: Constant Add Block

### The task

The same math, `z_i = x_i + 10`, but now:

- `N0 = 200`, which isn't a power of 2.
- `B0` is smaller than `N0`, so the vector needs **several blocks**, one program instance each.
- 200 doesn't divide evenly by most block sizes, so the **last block runs past the end** of the vector and needs a **mask**.

Kernel signature from the official puzzle:

```python
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr)
```

### The tests

#### Test 1: correct output

```python
@pytest.mark.parametrize("B0", [8, 32, 64, 128, 256])
def test_add_mask2(B0):
    x = torch.rand(N0)
    z = torch.empty_like(x)

    grid = (triton.cdiv(N0, B0),)
    add_mask2_kernel[grid](x, z, N0, B0=B0)

    torch.testing.assert_close(z, add2_spec(x))
```

The block sizes are chosen to cover each way `B0` can relate to `N0`:

| `B0` | blocks | case |
|---|---|---|
| 8 | 25 | divides 200 evenly, no partial block |
| 32 | 7 | last block partial |
| 64 | 4 | last block partial |
| 128 | 2 | last block partial |
| 256 | 1 | block bigger than the whole vector, mostly masked |

- Every `B0` is a power of 2. A non-power of 2 would be rejected by Triton before the kernel runs, so it tests Triton, not my kernel.
- **`triton.cdiv(N0, B0)`** is division rounded up, the same as `(N0 + B0 - 1) // B0`. It gives the number of blocks needed to cover all 200 elements.

#### Test 2: no writes past the end

```python
@pytest.mark.parametrize("B0", [32, 64, 256])
def test_add_mask2_does_not_write_past_end(B0):
    sentinel = -1.0
    buffer = torch.full((N0 + B0,), sentinel)
    z = buffer[:N0]
    x = torch.rand(N0)

    grid = (triton.cdiv(N0, B0),)
    add_mask2_kernel[grid](x, z, N0, B0=B0)

    torch.testing.assert_close(z, add2_spec(x))
    assert torch.all(buffer[N0:] == sentinel), "kernel wrote past the end of z"
```

Test 1 can pass even if the kernel stores without a mask, because it only looks inside `z`. Test 2 catches writes **outside** `z`:

- **`torch.full((N0 + B0,), -1.0)`** makes a buffer of `200 + B0` slots, all `-1`.
- **`z = buffer[:N0]`** is a *view* of the first 200 slots, so the extra slots sit right after `z` in memory.
- The extra slots hold a **sentinel**: a marker value the real output can never produce. `x` is between 0 and 1, so `x + 10` is between 10 and 11, never `-1`.
- After the kernel runs, `buffer[N0:] == sentinel` gives one `True`/`False` per extra slot, and `torch.all` requires all of them to be `True`. If any slot changed, the kernel wrote past the end of `z`.

```
buffer:  [ z0  z1  ...  z199 | -1  -1  ...  -1 ]
           <--- z (200) --->   <-- sentinel -->
```

`B0 = 8` is left out, since it divides 200 evenly and has no positions past the end.

Test 2 also runs the same `assert_close` as test 1, so it checks everything test 1 does for its block sizes. Running it with all five block sizes would make test 1 unnecessary.

Test 2 only catches a missing **store** mask. A missing **load** mask reads garbage past the end of `x`, but if the store mask is correct, that garbage is never written, so neither test sees it. The load mask is still needed for the kernel to be safe on a real GPU.

### The kernel

```python
@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * B0 + tl.arange(0, B0)
    mask = offsets < N0
    x = tl.load(x_ptr + offsets, mask=mask)
    z = x + 10.0
    tl.store(z_ptr + offsets, z, mask=mask)
```

- **`pid = tl.program_id(0)`:** this instance's number, from 0 up to the grid size minus 1.
- **`offsets = pid * B0 + tl.arange(0, B0)`:** shifts the local positions `0 .. B0-1` to this instance's block.
- **`mask = offsets < N0`:** `True` for positions inside the vector, `False` past the end.
- **`tl.load(..., mask=mask)`:** reads only the valid positions.
- **`z = x + 10.0`:** same math as Puzzle 1.
- **`tl.store(..., mask=mask)`:** writes only the valid positions. This is what keeps test 2's sentinel slots at `-1`.

The same mask works for the load and the store, because `x` and `z` have the same length and use the same offsets.

With `B0 = 64`, the 4 instances cover:

| `pid` | offsets | real elements | masked |
|---|---|---|---|
| 0 | 0–63 | 64 | 0 |
| 1 | 64–127 | 64 | 0 |
| 2 | 128–191 | 64 | 0 |
| 3 | 192–255 | 8 (192–199) | 56 |

### What's new compared with Puzzle 1

| | Puzzle 1 | Puzzle 2 |
|---|---|---|
| `B0` vs `N0` | `B0 = N0` | `B0` smaller (or larger) than `N0` |
| grid | `(1,)` | `(cdiv(N0, B0),)` |
| `tl.program_id` | not needed | shifts each instance to its block |
| mask | not needed | needed for the partial last block |
