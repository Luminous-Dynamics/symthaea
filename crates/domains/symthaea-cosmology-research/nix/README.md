# DE-001A Nix environment

This subflake is the isolated numerical environment for the first cosmology reproduction lane.

It is intentionally separate from Symthaea's root development shell so Cobaya/CAMB dependencies do not become ambient dependencies of unrelated work.

## Inspect

```bash
nix develop ./crates/domains/symthaea-cosmology-research/nix#cosmology-verify
```

The dev shell is for inspection only. It sets `PIP_NO_INDEX=1`, `PYTHONNOUSERSITE=1`, and `UV_OFFLINE=1` to reduce accidental environment contamination, but a shell is not itself a network sandbox.

## Qualify

```bash
nix flake check ./crates/domains/symthaea-cosmology-research/nix --no-update-lock-file
nix build ./crates/domains/symthaea-cosmology-research/nix#checks.x86_64-linux.environment --no-update-lock-file
```

The check runs inside a Nix build sandbox and asserts exact distribution versions plus successful imports.

The scientific DESI reproduction is **not** performed by this subflake yet. Environment qualification licenses only the statement that the frozen numerical environment built and satisfied its internal version/import contract.
