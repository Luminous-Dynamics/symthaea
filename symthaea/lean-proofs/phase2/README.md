# Phase 2 Lean Verification Project

Minimal Lake project that resolves Mathlib imports for the arithmetic `.lean` fixtures emitted by `cargo run -p symthaea-lean-bridge --example prove_fol_arith`.

## Setup strategy: `elan` via nix flake

The `symthaea/flake.nix` adds `elan` (the official Lean version manager) to the dev shell's `buildInputs`. When you `cd` into this directory and run any `lake`/`lean` command, elan reads the `lean-toolchain` file and auto-downloads the matching Lean binary (to `~/.elan/toolchains/`), regardless of what's in nixpkgs.

This sidesteps the version-alignment problem Phase 2 Week 2 hit: nixpkgs's `lean4` package tracks recent stable Lean versions, but Mathlib's Lake dependency graph (specifically `proofwidgets`) requires exact version matching. With elan, the Lean ecosystem's own versioning takes over.

## Pins

- `lean-toolchain`: `leanprover/lean4:v4.12.0`
- `lakefile.lean`: requires `mathlib4` at tag `v4.12.0`

Both pins are synchronized — Mathlib v4.12.0's own `lean-toolchain` specifies v4.12.0, so elan + Mathlib converge.

## One-time setup

From the workspace root, inside `nix develop`:

```bash
cd lean-proofs/phase2

# 1. Elan reads lean-toolchain and downloads Lean v4.12.0 (~1 min).
#    This is a noop on subsequent shells once the toolchain is cached
#    under ~/.elan/toolchains/.
elan toolchain install leanprover/lean4:v4.12.0

# 2. Resolve Mathlib + transitive deps.
lake update

# 3. Pull prebuilt Mathlib olean cache (~1-2 GB, ~3-10 min).
#    Much faster than compiling Mathlib from source.
lake exe cache get

# 4. Verify the cache resolves cleanly.
lake build
```

After `lake build` succeeds, `lake env lean <path>` verifies any `.lean` file that imports Mathlib.

## Running the Phase 2 verification

```bash
# from workspace root:
LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example prove_fol_arith
```

The example's `lake_check` column in the emitted CSV will report `accepted` / `rejected` per fixture.

Or verify one file manually:

```bash
cd lean-proofs/phase2
lake env lean ../../proofs/fol_arith/square_nonneg_pow.lean
# expected: clean exit (no output)
```

## Troubleshooting

- **`elan: command not found`**: Enter `nix develop` first (the flake adds elan).
- **`toolchain 'leanprover/lean4:v4.12.0' is not installed`**: Run step 1 of the one-time setup.
- **`lake: command not found`**: Same — elan's shim for `lake` is installed alongside Lean.
- **`Unknown tactic 'linarith'`**: The file was run with plain `lean` instead of `lake env lean`. Mathlib tactics need a Lake project with the Mathlib dep.
- **Cache download is slow**: Mathlib's prebuilt olean cache is hosted by the Lean community; 5-15 min on first-time setup is normal.

## Bumping the Mathlib pin

Mathlib releases track Lean bumps every few months. To update:

```bash
cd lean-proofs/phase2
# Pick the new Mathlib release tag.
# Check https://github.com/leanprover-community/mathlib4/releases
NEW_TAG=v4.13.0
sed -i "s|v4.12.0|$NEW_TAG|g" lakefile.lean
echo "leanprover/lean4:${NEW_TAG}" > lean-toolchain
rm -rf .lake lake-manifest.json
lake update && lake exe cache get && lake build
```

## Scope

This Lake project is intentionally minimal. It's not an end-user library; it's scaffolding so we can externally verify what the Rust bridge emits. The actual proof artifacts live in `../../proofs/fol_arith/*.lean`.
