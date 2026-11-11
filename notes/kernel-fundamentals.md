# Triton Kernel Fundamentals

Notes from the Triton puzzles, written as a reference for the ideas that come up again and again.

## 1. Kernel launch and the grid

A kernel is launched with a grid in brackets, then its arguments:

```python
add_kernel[(4,)](x, z, N0, B0=64)
```

- `(4,)` is the **grid**: a tuple with one entry per axis. It's a tuple, not a scalar.
- The grid decides how many **program instances** run.
- A grid can have 1, 2, or 3 axes. Triton supports at most 3.

## 2. Program instances

A **program instance** is one running copy of the kernel. All instances run the same code, but each gets its own ID values, so each can work on different data.

The number of instances is the product of the grid entries:

| grid | instances |
|---|---|
| `(4,)` | 4 |
| `(4, 2)` | 4 × 2 = 8 |
| `(2, 3, 4)` | 2 × 3 × 4 = 24 |

Instances don't need to coordinate when their data doesn't overlap, so they can run in any order or at the same time.

## 3. `tl.program_id(axis)`

`tl.program_id(axis)` returns the ID of the current instance along one axis, counting from 0 up to that axis's grid size minus 1.

- `tl.program_id(0)` is valid for any grid.
- `tl.program_id(1)` and `tl.program_id(2)` are valid only if the grid has those axes.

Assigning it to a variable is common:

```python
pid0 = tl.program_id(0)
```

`pid0` is just a variable name. The function is `tl.program_id`.

With a 2-axis grid, each instance is one **pair** `(pid0, pid1)`. With a 3-axis grid, each instance is one **triple** `(pid0, pid1, pid2)`. Every combination of IDs is one instance.

## 4. Blocks and `B0`

A **block** is the group of elements one program instance handles.

`B0` is the block size along axis 0: how many elements that instance handles along that axis.

- `B0` must be a power of 2, because `tl.arange` requires it.
- It's declared as `tl.constexpr`, so its value is fixed at compile time.
- Larger values are allowed, but hardware resources limit how large a block can be.

The number of blocks along an axis is `ceil(N / B)`, where `N` is the size of the data along that axis:

```python
(N0 + B0 - 1) // B0
```

If `N0 = 200` and `B0 = 64`, that's `4` blocks. The last block is only partly filled.

For 2-D and 3-D data, each axis has its own block size: `B0`, `B1`, and `B2`.

## 5. Offsets

Each instance computes which positions it owns:

```
offsets = pid * B0 + tl.arange(0, B0)
```

- `tl.arange(0, B0)` gives the local positions `0 .. B0-1`.
- Multiplying the instance's ID by `B0` shifts those positions to the instance's own region.

If the grid has one instance (`N0 == B0`), `pid` is always 0, so the offsets are just `tl.arange(0, B0)`.

## 6. Tiles in 2-D and 3-D

In a 2-D problem, a **tile** is a rectangle of the data, one per instance:

- `pid0` picks the row tile: rows `pid0 * B0` to `pid0 * B0 + B0 - 1`.
- `pid1` picks the column tile: columns `pid1 * B1` to `pid1 * B1 + B1 - 1`.

For an 8 × 4 matrix with 2 × 2 tiles, the grid is `(4, 2)`, and the 8 instances cover all 32 elements without overlap.

In 3-D, the same idea gives boxes of size `B0 × B1 × B2`.

## 7. Masks

When a block runs past the end of the data, some offsets point outside valid memory. A **mask** says which positions are real:

- `mask = offsets < N0` is `True` for valid positions and `False` for the rest.
- `tl.load` and `tl.store` accept a `mask=` argument, and they skip the masked-out positions.

Puzzle 1 needs no mask, because `B0 = N0` means every offset is valid.

## 8. Pointers and values

- A **pointer** is an address. `x_ptr` in a kernel is the address of the first element of `x`.
- `x_ptr + offsets` is a **block of pointers**, one address per position.
- `tl.load(pointer_block)` reads the values at those addresses into registers and returns them as a block tensor.
- `tl.store(pointer_block, values)` writes the values back to those addresses. It doesn't return a value.

Name the pointers and the loaded values differently, so the code shows which is which. Overwriting a pointer with a loaded value is a common bug.
