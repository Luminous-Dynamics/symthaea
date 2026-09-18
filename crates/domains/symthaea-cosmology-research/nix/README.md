# DE-001A Nix environment

This subflake is the isolated numerical environment and fixed-output input closure for the first cosmology reproduction lane.

It is intentionally separate from Symthaea's root development shell so Cobaya/CAMB dependencies do not become ambient dependencies of unrelated work.

## Realize the A0 scientific bytes

```bash
nix build ./crates/domains/symthaea-cosmology-research/nix#a0-artifacts --no-update-lock-file
```

The eight inputs are fixed-output derivations. Nix will refuse to realize a source whose bytes do not match its preregistered SHA-256. The package copies them into regular files named by A0 role so the Rust `de001a-a0-verify` binary can independently verify them.

The A0 check also verifies every expected byte count and SHA-256:

```bash
nix build ./crates/domains/symthaea-cosmology-research/nix#checks.x86_64-linux.a0-artifacts --no-update-lock-file
```

Neither realization nor this check evaluates a cosmological likelihood.

## Inspect the numerical environment

```bash
nix develop ./crates/domains/symthaea-cosmology-research/nix#cosmology-verify
```

The dev shell is for inspection only. It sets `PIP_NO_INDEX=1`, `PYTHONNOUSERSITE=1`, and `UV_OFFLINE=1` to reduce accidental environment contamination, but a shell is not itself a network sandbox.

## Qualify

```bash
nix flake check ./crates/domains/symthaea-cosmology-research/nix --no-update-lock-file
nix build ./crates/domains/symthaea-cosmology-research/nix#checks.x86_64-linux.environment --no-update-lock-file
```

The checks run inside Nix build sandboxes. One asserts exact scientific-input bytes; the other asserts exact distribution versions plus successful numerical-package imports.

The scientific DESI reproduction is **not** performed by this subflake yet. These checks license only their narrow integrity/environment statements.
