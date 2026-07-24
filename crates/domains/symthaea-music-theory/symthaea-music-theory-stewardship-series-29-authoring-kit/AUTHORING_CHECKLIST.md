# Series 29 authoring checklist

## Grounding

- Begin from the exact reviewed and landed Series 28 final tree.
- Preserve the Series 21–27 evidence and claim boundaries.
- Convert each plan into one reviewable, intentional patch.
- Record exact base/final commits and trees.
- Replay in a sanitized clean environment.

## Required code gates

```text
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

- Run the canonical Nix lane.
- Run independent-verifier conformance.
- Run race, rollback, restart, privacy, hostile-input, compatibility, and deterministic packaging lanes.
- Generate claims from observed evidence only.

## Change discipline

- No new lifecycle authority semantics in this series.
- Every code change maps to a review, regression, compatibility, or maintenance requirement.
- Every confirmed defect receives a frozen regression fixture.
- No telemetry, review signoff, or maintainer role becomes catalog evidence authority.
