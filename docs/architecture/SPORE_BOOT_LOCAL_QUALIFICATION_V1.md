# Spore Boot Local Qualification v1

Status: tooling/runbook only. This document does not qualify a boot or renderer build by itself.

## Why this exists

The Symthaea repository can have substantial GitHub Actions queue pressure. Local qualification provides an independent exact-ref path on a NixOS/Rust workstation without weakening hosted merge gates.

The current Boot Ecology v0.3.2 dedicated workflow reached an actual runner and failed at its first substantive step: Rust 1.96 `cargo fmt --check`. All check/Clippy/test/renderer-probe/gallery/lint/evidence steps after formatting were skipped. Treat that as a formatting blocker, not a renderer failure.

## Tool branch

The local-only helpers live on:

`boot/local-qualification-v1`

They are intentionally not part of the Spore product branch and do not have an active PR.

## A. Repair and qualify Boot Ecology #238

Start from the exact Boot Ecology branch and fetch the tools branch:

```bash
git fetch origin spore/boot-ecology-v0.3 boot/local-qualification-v1
git switch spore/boot-ecology-v0.3
git status --short
```

The first run may apply only pinned Rust 1.96 formatting to a clean matching checkout:

```bash
bash <(git show origin/boot/local-qualification-v1:scripts/qualify-spore-boot-ecology-local.sh) \
  --ref HEAD \
  --apply-format
```

Expected formatter-repair outcome when #238 is still at the known failing head:

- exit code `4`;
- `BOOT_ECOLOGY_RUSTFMT.patch` exported under `/tmp/spore-boot-ecology-<sha>/`;
- the same patch applied to the current checkout only because `--ref HEAD` matches and the relevant files were clean;
- no PASS claim.

Review the formatting-only diff before committing:

```bash
git diff --check
git diff --stat
git diff
```

Then commit the exact formatter output:

```bash
git add \
  crates/core/symthaea-boot-ecology \
  crates/core/symthaea-boot-state \
  crates/domains/symthaea-quicken-fb

git commit -m 'style(spore): apply Rust 1.96 rustfmt to boot ecology'
```

Now run the exact local qualification again without `--apply-format`:

```bash
bash <(git show origin/boot/local-qualification-v1:scripts/qualify-spore-boot-ecology-local.sh) \
  --ref HEAD
```

That run requires the committed target to be format-clean before it proceeds to Rust check, Clippy, tests, exact renderer cost probe, lifecycle previews, Inoculation previews, install-path previews, visual lint, and SHA-256 evidence sealing.

Use `--skip-galleries` only for a faster diagnostic pass. It is not a substitute for final exact visual evidence.

## B. Qualify the consolidated Spore foundation

The product consolidation branch is:

`boot/spore-foundation-v0.1`

It is a true descendant of the boot micro-PR stack and a two-parent merge with current `main`; it was prepared to preserve commit ancestry while removing the stale-main gap.

Fetch both branches, then run:

```bash
git fetch origin boot/spore-foundation-v0.1 boot/local-qualification-v1

bash <(git show origin/boot/local-qualification-v1:scripts/qualify-spore-boot-local.sh) \
  --ref origin/boot/spore-foundation-v0.1
```

Important exit states:

- `0`: exact committed target passed and its committed Cargo.lock already matched resolver output;
- `1`: a focused Rust/Nix gate failed;
- `2`: local tooling/precondition failure;
- `3`: semantic lane passed only after Cargo produced a different lock. The exported `Cargo.lock.resolved` must be committed unchanged to the product lineage and the new exact head rerun before PASS can be claimed.

The receipt binds exact commit, Git tree, committed/resolved lock hashes, workspace manifest hash, qualification-script hash, and actual Rust/Cargo versions.

## C. Inspect queue pressure safely

The queue helper is dry-run by default:

```bash
python3 <(git show origin/boot/local-qualification-v1:scripts/prune-superseded-actions.py)
```

It preserves manual/scheduled runs and the newest run in every selected PR/workflow group. It mutates GitHub only when both `--apply` and `--yes` are supplied.

Do not use queue pruning as a replacement for PR-stack consolidation. Superseded-run pruning handles stale queue entries; consolidated review topology reduces the number of independently active PR heads that create new work.

## Architecture exit gate

After these three facts are true:

1. Boot Ecology exact renderer/evidence gates pass on a format-clean committed head;
2. consolidated Spore boot foundation passes its exact locked focused qualification;
3. render-projection seam passes that same focused lane;

stop adding renderer-independent boot abstractions.

The next product tranche is integration-only:

`EcologyFrameInput -> RenderProjection -> qualified Boot Ecology renderer`

Then evaluate exact pixels and performance for fast healthy, slow healthy, degraded-to-recovered, and Ready-with-Unknown cases before renaming `quicken-fb` or optimizing rendering.
