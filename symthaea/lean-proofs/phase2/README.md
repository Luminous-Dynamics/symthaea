# Phase 2 Lean Verification Project

Minimal Lake project intended to resolve Mathlib imports for the arithmetic `.lean` fixtures emitted by `cargo run -p symthaea-lean-bridge --example prove_fol_arith`.

## Current status: setup BLOCKED on Lean/Mathlib/nix version alignment

**First attempt** (April 17 2026): `lake update` against `v4.12.0` and then against `main` both failed with `proofwidgets` Lake API errors. Root cause: our nix flake provides `lean4` version **4.26.0** (from `nixpkgs-unstable`), while Mathlib tags and even `main` assume Lean versions that live at different points in Lake's evolving API.

The conflict is not with Mathlib itself — it's with Mathlib's transitive dependency on `proofwidgets`, which in older releases uses Lake APIs that don't exist in Lean 4.26's Lake 5.0, and in `main` uses APIs that are newer than Lean 4.26 exposes.

## Paths forward (pick one)

### Option A: pin an older Lean via elan (abandons nix flake control)

Add elan to the flake, install the Lean version Mathlib's `main` requires:

```bash
# in flake.nix, swap `lean4` for `elan`:
#   elan
# then:
cd lean-proofs/phase2
cat > lean-toolchain <<EOF
leanprover/lean4:v4.13.0
EOF
lake update   # pulls Mathlib main + its Lean version
lake exe cache get
lake build
```

Downside: elan downloads its own Lean toolchain outside nix, duplicating what our flake already provides.

### Option B: update the flake's lean4 to match a stable Mathlib release

Pin `lean4` in the flake to a version that has a corresponding stable Mathlib release:

```nix
# flake.nix: replace `lean4` with a version matching Mathlib's release.
# Mathlib's v4.13.0 tag wants Lean 4.13; our current Lean 4.26 is too new.
# Either downgrade to a cohabitating pair, or follow Mathlib main as it
# catches up to Lean 4.26+.
```

Downside: requires nixpkgs override or a Lean overlay; not all Lean versions are in nixpkgs.

### Option C: defer Mathlib verification, keep structural validation only

This is the current state. The bridge emits well-formed `.lean` files (14 committed under `proofs/fol_arith/`). Any reader can:

- Verify the structural emission with `cargo test -p symthaea-lean-bridge --lib fol_ext_bridge` (7 tests).
- Spot-check the generated Lean surface syntax by eye — the committed `.lean` files are self-explanatory.

External semantic verification (did Mathlib's `linarith`/`nlinarith`/`omega` actually close each goal?) is blocked until Option A or B lands.

## What works today without Mathlib

- **SMT decisions via Z3.** Every fixture that `detect_fragment` classifies as `QF_LIA` / `QF_LRA` / `QF_NIA` / `QF_NRA` can be checked with Z3 alone (no Lean needed). The `cargo test -p symthaea-core --test fol_ext_round_trip` suite exercises 10 such round-trips end-to-end (8 tautologies unsat + 2 non-tautologies sat).

That's the Phase 2 Week 1 result, and it's the authoritative evidence of semantic correctness. The Mathlib Lean emission is the *presentation* layer for Lean audiences; the SMT result is the *proof*.

## One-time setup (when one of the Options above is chosen)

```bash
cd lean-proofs/phase2
lake exe cache get      # ~3-10 min, 1-2 GB
lake build              # ~1 min after cache
```

Then in the workspace root:

```bash
LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example prove_fol_arith
```

The CSV `lake_check` column will report `accepted` / `rejected` per fixture.

## Pins

- `lakefile.lean` requires `mathlib` from git `main` (will be lake-manifest-pinned after first `lake update`).
- `lean-toolchain` is **deliberately absent** so Lake uses whatever `lean` is on PATH. Re-create it with the appropriate version once Option A or B is chosen.

## Troubleshooting

- **`proofwidgets` errors about `BuildJob`, `inputTextFile'`, `afterReleaseAsync`**: Lean / Lake version mismatch. See the status note above.
- **`lake: command not found`**: Enter `nix develop` first (the flake adds `lean4` which ships `lake`).
- **`cache: command not found`**: Mathlib's `cache` executable is only available after `lake update` pulls Mathlib. Run `lake update` first.
