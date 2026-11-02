# Connectome structural-world verifier V0.1

A standalone Rust verifier for a deliberately narrow, independently reproducible finite-universe contract. It is designed to supply the independently computed evidence required by `PairedWorldSetCoverage` in `connectome-query-impact`.

## Paradigm shift

Do not infer completeness from a boolean, a digest-shaped string, or equality between two author-supplied ID lists. Define a finite universe as a product of explicit axes and alternatives, then independently regenerate every admissible combination.

V1 deliberately supports only:

`cartesian_product_all_combinations_v1`

Every combination of every declared axis is admissible. There are no arbitrary predicates, hidden filters, pruning rules, weights or post-hoc exclusions. This restriction makes closure independently checkable. The implementation caps the universe at 100,000 worlds and 64 axes to bound memory use and recursion depth.

## What the executable verifies

- canonical manifest digest using the pinned `SERDE_JSON_ORDERED_CANONICAL_V1` encoding;
- actual SHA-256 bytes of local artifacts, with path traversal and symlink escape rejected;
- pre-query freeze receipt, source-universe snapshot, base graph snapshot, hypothetical graph snapshot and candidate patch bindings;
- exact query, scenario, candidate, operation, sample, task-route and threshold context;
- each alternative's explicit evidence-artifact ID and digest against the exact file whose bytes were read and hashed;
- caps axis count at 64 before recursive enumeration to avoid hostile singleton-axis stack exhaustion;
- deterministic world IDs generated from canonical assignment objects;
- full Cartesian enumeration, exact expected/evaluated world-set equality, no duplicates and no unresolved frontier;
- matched before/after values inside their corresponding marginal intervals and paired differences inside the supplied effect interval;
- a coverage-evidence receipt containing universe, policy, enumerator, expected-world-manifest, closure-proof and verification-receipt digests; the trusted parent `closure_verified` flag remains false until external trust checks pass.

## Run

```sh
cargo fmt --manifest-path tools/connectome-worldset-verifier/Cargo.toml -- --check
cargo test --manifest-path tools/connectome-worldset-verifier/Cargo.toml --locked
cargo clippy --manifest-path tools/connectome-worldset-verifier/Cargo.toml --locked --all-targets -- -D warnings
cargo run --manifest-path tools/connectome-worldset-verifier/Cargo.toml --locked -- path/to/universe.json path/to/receipt.json
```

Artifact paths in the manifest are relative to the manifest's directory. The verifier never follows paths outside that directory. The optional second argument writes the same JSON receipt printed to stdout.

## Receipt integration

The output's `coverage` object contains the structural fields consumed by `PairedWorldSetCoverage` in `tools/connectome-query-impact/src/lib.rs`. It sets `finite_cartesian_enumeration_verified=true` after hashing artifacts and independently matching the complete declared product, but intentionally leaves the parent kernel's trusted `closure_verified=false`. This prevents an untrusted manifest from self-promoting into a trusted finite-universe claim. The receipt explicitly says pre-query freeze authentication and outcome-independent universe selection were not performed. Only a separate production adapter/operator can promote the trusted flag after:

1. independently pin the verifier binary SHA-256 in a trust policy outside the submitted manifest;
2. hash and retain the emitted receipt bytes (the receipt includes its own canonical digest);
3. authenticate the pre-query freeze receipt and verify it was logged before query outcomes were opened;
4. check that the universe and admissibility policy were selected without using target labels or desired outcomes.

A digest included only in the same untrusted manifest is not an authentication mechanism. A valid receipt proves exact coverage of the declared finite Cartesian universe and artifact lineage; it does not prove the universe contains every biologically possible connectome or that any alternative is anatomically correct.

## Explicit limitations

- The freeze booleans are assertions. The receipt reports `prequery_freeze_authentication=NOT_PERFORMED_EXTERNAL_TRUST_REQUIRED` and `universe_selection_independence=NOT_ASSESSED`. Authentic pre-query timing/authorization requires a separately authenticated receipt and operator-pinned trust policy.
- The verifier validates the declared alternatives and evidence digests; it does not decide whether an alternative is biologically admissible.
- Cartesian worlds are not assigned probabilities.
- The paired interval and graph endpoint values are inputs; this tool does not run connectome inference or apply edits.
- No ConnectomeBench2 V2 dataset pin, Parquet rows, EM volume, or biological outcome is bundled.
- Rust tests in this change are authored but must run on an exact reviewed head before the tool is called executable in production.

## Tests

Ten authored Rust tests cover:
1. full Cartesian product closure and adapter-ready receipt emission;
2. omission rejection even after the manifest is resealed;
3. duplicate-world rejection;
4. changed artifact-byte rejection;
5. paired values outside marginal intervals;
6. unresolved-frontier rejection;
7. stale manifest-digest rejection;
8. parent-directory traversal rejection;
9. exact alternative-to-evidence-artifact ID/digest binding;
10. the 64-axis recursion-safety cap.

They have not been compiled or executed in the authoring environment.
