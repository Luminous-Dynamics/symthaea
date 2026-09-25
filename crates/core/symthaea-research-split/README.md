# symthaea-research-split

Content-addressed contracts for **declared** train/calibration/evaluation separation.

The crate exists because random example-level holdouts are often an invalid validation strategy for spatially or temporally autocorrelated data such as remote sensing, ecological observations, repeated subjects, orbital acquisitions, and time series.

## What the crate can establish

Given an immutable sample manifest, it can fail closed when:

- an evaluation sample shares a configured group such as `spatial-block`, `acquisition`, `watershed`, `site`, or `subject` with development data;
- a configured group is missing from a sample;
- a group is reused across roles under `AllRolesDisjoint`;
- evaluation begins before the declared forward-time embargo after the latest training/calibration sample;
- sample ids or group dimensions are duplicated;
- a manifest is mutated after its content digest is frozen;
- a serialized manifest is loaded with a stale or forged digest.

A split manifest therefore records concrete structural facts such as:

> evaluation does not reuse any declared `spatial-block` or `acquisition` group from development, and evaluation begins at least N milliseconds after the latest development sample.

## What the crate deliberately does not establish

A group id is not proof of statistical independence.

The crate does **not** infer that:

- two geographic blocks are far enough apart;
- a spatial buffer exceeds the process autocorrelation range;
- two dates are temporally independent;
- different Sentinel products imply independent atmospheric or hydrological conditions;
- different watersheds are exchangeable;
- an evaluation split represents a target deployment distribution.

Those stronger claims require domain analysis. `SeparationEvidence` records the statement and content digest of supporting artifacts without converting them into a universal `independent = true` flag.

## Core types

- `SplitUnit` — immutable sample identity, timestamp, content digest, and named groups.
- `AssignedUnit` — a sample assigned to Training, Calibration, or Evaluation.
- `GroupSeparationPolicy` — `None`, `EvaluationDisjoint`, or `AllRolesDisjoint` over explicit group dimensions.
- `TemporalSeparationPolicy` — currently `None` or strict forward evaluation with an embargo.
- `SeparationEvidence` — attributable evidence for claims the structural contract cannot prove itself.
- `ResearchSplitManifest` — validated, content-addressed frozen assignment manifest.

## Remote-sensing profile

A serious Sentinel experiment should normally consider more than one grouping dimension. For example:

```text
sample
  spatial-block = watershed-042
  acquisition   = S2A-product-...
  orbit/swath   = ...
  season        = wet-2026
```

A different spatial block does not prevent leakage if the development and evaluation samples are crops from the same acquisition. Conversely, different products may still be highly autocorrelated in time or space. The manifest records the enforced grouping; domain-specific buffer/autocorrelation analysis remains separate evidence.

For real Wetland Watch validation, random pixel-level train/test splits should not be the default. Prefer a preregistered combination of geographic grouping, acquisition grouping, and forward-time evaluation appropriate to the scientific claim.

## Digest boundary

`ResearchSplitManifest::new` validates the semantics and freezes a BLAKE3 digest over the complete assignment/policy/evidence view. `verify_digest` re-runs semantic validation and verifies the digest.

Deserialization is routed through `TryFrom<ResearchSplitManifestRepr>`, so loading a stored manifest also performs semantic validation and digest verification. Serialized input is not a bypass around the constructor.

## Required gates

```bash
cargo fmt --all -- --check
cargo check -p symthaea-research-split --all-targets
cargo test -p symthaea-research-split
cargo clippy -p symthaea-research-split --all-targets -- -D warnings
```

Do not promote a split from `implemented` to `scientifically adequate` merely because these gates pass. The gates validate the contract implementation; the scientific adequacy of the chosen groups, buffers, embargo, and target distribution remains an empirical question.