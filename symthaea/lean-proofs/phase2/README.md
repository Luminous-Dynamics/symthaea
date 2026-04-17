# Phase 2 Lean Verification Project

This directory is a tiny Lake project whose job is to **resolve Mathlib imports** so that `.lean` files emitted by `cargo run -p symthaea-lean-bridge --example prove_fol_arith` can be externally verified.

## Why this exists

Phase 1's Lean bridge targets **core Lean 4** and uses term-mode proofs (`exact fun h => h`, `Classical.em`, etc). No Mathlib needed. Phase 2 extends the bridge with arithmetic goals (equality + ordering + quantifiers over ℤ/ℕ/ℝ), whose decision procedures live in Mathlib as `omega` / `linarith` / `nlinarith`. Importing Mathlib requires a Lake project with Mathlib in its dependency tree — hence this directory.

## One-time setup

```bash
cd lean-proofs/phase2

# Pull prebuilt Mathlib olean cache (~1-2 GB, downloads in ~3-10 min).
# Much faster than compiling Mathlib from source (which would take hours).
lake exe cache get

# Verify the cache resolves and all Mathlib modules typecheck.
# First run ~1 min; subsequent runs are incremental.
lake build
```

After setup, `lake env lean <path>` resolves `import Mathlib.Tactic` and
verifies any `.lean` file that imports Mathlib, in typically <10s per file.

## Running the Phase 2 verification

Once the setup above is done:

```bash
# from repo root:
LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example prove_fol_arith
```

The example's `lake_check` column in the emitted CSV will report
`accepted` / `rejected` per fixture instead of `skipped`.

Or run one file manually:

```bash
cd lean-proofs/phase2
lake env lean ../../proofs/fol_arith/square_nonneg_pow.lean
# expected: clean exit (no output)
```

## What's pinned

- `lean-toolchain`: `leanprover/lean4:v4.12.0`
- `lakefile.lean` requires `mathlib4` at tag `v4.12.0`

Both pins are synchronized so Mathlib's `lean-toolchain` matches our toolchain. When Mathlib bumps a major tag (quarterly-ish), update both files in lock-step, run `lake update`, and re-run the test suite.

## Troubleshooting

- **"Unknown tactic `linarith`"**: the file was run with plain `lean` instead of `lake env lean`. Mathlib tactics are only available inside a Lake project with the Mathlib dependency.
- **"Command 'cache' not found"**: `lake exe cache get` requires a Lake project that has Mathlib as a dependency (this one does). If you see this error, check that `lake build` was run first.
- **Cache download is slow**: Mathlib's prebuilt olean cache is hosted by the Lean community; residual download times of 5-15 min are normal on first-time setup.

## Scope

This Lake project is intentionally minimal. It's not an end-user library; it's scaffolding so we can externally verify what the Rust bridge emits. The actual proof artifacts live in `../../proofs/fol_arith/*.lean`.
