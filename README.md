# cuda-triton-learn

My hands-on path to learning GPU kernels: Triton first, then CUDA, then how a real inference engine (SGLang) uses them.

Everything is done **test-first**:

1. Write the test.
2. Run it and watch it fail.
3. Write the kernel.
4. Run it and watch it pass.
5. Benchmark (once GPU work starts).

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | **Triton Puzzles**: solve every puzzle from [gpu-mode/Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles) | In progress |
| 2 | **Kernel papers**: implement each kernel in Triton and in CUDA, then compare the two | Not started |
| 3 | **SGLang**: study how SGLang uses these kernels and techniques | Not started |

### Phase 1 progress

| # | Puzzle | Kernel | Test | Status |
|---|---|---|---|---|
| 1 | Constant Add | [`p01_const_add.py`](kernels/puzzles/p01_const_add.py) | [`test_p01_const_add.py`](tests/puzzles/test_p01_const_add.py) | Done |
| 2 | Constant Add Block | [`p02_const_add_block.py`](kernels/puzzles/p02_const_add_block.py) | [`test_p02_const_add_block.py`](tests/puzzles/test_p02_const_add_block.py) | Done |
| 3 | Outer Vector Add | | | Next |

## Repository layout

```
.
├── kernels/
│   └── puzzles/          # my kernel solutions, one file per puzzle (pNN_<name>.py)
├── tests/
│   └── puzzles/          # pytest tests, one file per puzzle (test_pNN_<name>.py)
├── notes/
│   ├── kernel-fundamentals.md   # grids, program instances, program_id, blocks, masks, pointers
│   └── puzzles.md               # per-puzzle explanation of each test and kernel
├── benchmarks/           # GPU benchmarks (later phases)
├── .github/workflows/
│   └── ci.yml            # runs all tests on the CPU on every push
├── pyproject.toml        # pytest config
└── requirements.txt      # Python dependencies
```

### Where to start reading

- **New to Triton?** Start with [notes/kernel-fundamentals.md](notes/kernel-fundamentals.md).
- **Want to see a solved puzzle?** Read its section in [notes/puzzles.md](notes/puzzles.md), then open the kernel and test files from the table above.

## Running the tests

The tests run on a CPU-only machine through Triton's interpreter, so no GPU is needed.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install pytest
```

### Run

```bash
# all tests
TRITON_INTERPRET=1 pytest -v

# one puzzle
TRITON_INTERPRET=1 pytest tests/puzzles/test_p02_const_add_block.py -v
```

`TRITON_INTERPRET=1` makes Triton run kernels as Python on the CPU. Without it, Triton looks for a GPU.

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the full test suite on GitHub Actions on every push and pull request, using the same CPU interpreter. Results are in the repo's **Actions** tab.

GPU tests and benchmarks will run on [Modal](https://modal.com) starting in Phase 2.

## Adding a new puzzle

1. **Test:** create `tests/puzzles/test_pNN_<name>.py`. Use `@pytest.mark.parametrize` to cover several sizes.
2. **Stub:** create `kernels/puzzles/pNN_<name>.py` with the kernel's signature and a `pass` body.
3. **Red:** run the test and confirm it fails on the output check.
4. **Kernel:** write the kernel body until the test passes.
5. **Notes:** add a section to [notes/puzzles.md](notes/puzzles.md) and a row to its table.
6. **README:** add a row to the Phase 1 progress table above and update its status.
7. **Commit and push**, then check that CI is green.

Name files with a two-digit puzzle number (`p03`, `p04`, ...) so they sort in order.

## Keeping this README up to date

- When a puzzle is finished, mark it **Done** in the progress table and mark the next one **Next**.
- When a phase starts, change its status in the roadmap and add a progress table for it.
- When a new top-level folder is added, add it to the layout above.
