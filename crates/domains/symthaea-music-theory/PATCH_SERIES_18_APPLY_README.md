# Applying Symthaea Music Theory Patch Series 18

## Base

Apply this series to the exact Series-17 source tree:

```text
32f5fcc14037e0a3b81227f2dd710ddf49f9b7dc
```

That is the Git tree contained in
`symthaea-music-theory-improved-round17.tar.gz`.

## Apply the mail patches

From the crate root:

```text
git am --3way patches/*.patch
```

The patch filenames are ordered numerically and must be applied in that order.

## Verify patch integrity

The release archive contains a SHA-256 manifest. Verify it before applying:

```text
sha256sum -c SHA256SUMS
```

## Canonical Rust verification

The patch-building environment did not contain a Rust or Nix toolchain. Run all
of the following in the canonical project shell before merging:

```text
cargo fmt --all -- --check
cargo check --all-targets
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

## Focused operator smoke tests

```text
cargo run --example evidence_publication_checkpoint -- --help
cargo run --example evidence_publication_mirror -- --help
cargo run --example evidence_publication_witness -- --help
cargo run --example evidence_publication_head_bundle -- --help
```

## Review priorities

Reviewers should pay particular attention to:

- preservation of Series-17 schema-role ordinals;
- exact record/event prefix comparison in consistency proofs;
- the distinction between mirror-ledger integrity and detected conflicts;
- external-verifier enforcement at witness-threshold evaluation;
- mandatory mirror observation of a packaged head;
- the machine-readable limitations in the catalog-head bundle.

## Rollback

The series is additive. If it must be removed before downstream adoption,
reverse the commits in order or reset to the Series-17 tree. It does not alter
the existing publication catalog, publication record, publication event,
delegation, or status-proof persistence schemas.
