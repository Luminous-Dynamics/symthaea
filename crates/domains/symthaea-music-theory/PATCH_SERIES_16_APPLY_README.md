# Applying Symthaea Music Theory Patch Series 16

## Expected base

Apply this series to Patch Series 15 with Git tree:

```text
5d2c8164aab3718aa64265e9e42377d5cec546f5
```

Confirm the tree before applying:

```bash
git rev-parse HEAD^{tree}
```

## Apply the mail series

Extract the patch archive and run from the `symthaea-music-theory` repository:

```bash
git am --3way patches/*.patch
```

If the archive was extracted directly into a directory containing numbered patch files:

```bash
git am --3way ./*.patch
```

After application, compare the resulting tree with the final tree recorded in the release summary accompanying the patch archive.

## Verify

Run the canonical project checks:

```bash
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
```

Run the repository's normal Nix verification lane.

Useful governance-tool smoke checks include:

```bash
cargo run --example evidence_selective_disclosure -- --help
cargo run --example evidence_retention_snapshot -- --help
cargo run --example evidence_governance_receipt -- --help
cargo run --example evidence_governance_receipt_chain -- --help
cargo run --example evidence_governance_export -- --help
cargo run --example evidence_governance_attestation_payload -- --help
```

## Revert

Before publishing evidence produced by the series, the cleanest rollback is:

```bash
git reset --hard <series-15-tip>
```

After the commits are shared, revert them in reverse order rather than rewriting shared history.

## Important trust limits

- A withdrawal receipt proves an exact governed bundle transition, not deletion of uncontrolled external copies.
- A retention snapshot uses externally governed logical epochs and is not legal advice or certification.
- Selective disclosure removes private fields from its versioned schema but does not make outside contextual re-identification impossible.
- Attestation payload bytes are not authenticated until an external trusted verifier accepts a signature or transparency proof.
