# Fusion-frame closure V0.1 fixture

This is a synthetic negative-control fixture. It contains no biological data and does not assert that two real custodians are independent.

## Files

- `connectome-fusion-frame-closure-input-v0.1.schema.json`: strict input schema.
- `fixtures/connectome-fusion-frame-closure-omitted-use-v0.1.json`: two source exports and a deliberately incomplete candidate frame.
- `CONNECTOME_FUSION_FRAME_CLOSURE_V0.1.md`: protocol and claim boundary.

## Canonicalization and digest domains

For this fixture, canonical JSON is UTF-8 JSON with object keys sorted lexicographically, no insignificant whitespace, Unicode emitted directly, and array order preserved. This is a fixture convention, not a general claim of RFC 8785 compatibility. The executable verifier must pin its exact implementation and test it independently.

Digest domains are:

- Scope: `SHA256("connectome-fusion-scope-v0.1\\0" || canonical_json(scope))`
- Projection: `SHA256("connectome-fusion-projection-v0.1\\0" || canonical_json({projection_id, rule}))`
- Canonical use key: `SHA256("connectome-fusion-use-key-v0.1\\0" || canonical_json(use_material))`
- Event export: `SHA256("connectome-fusion-events-v0.1\\0" || canonical_json(ordered_events))`
- Frame: `SHA256("connectome-fusion-frame-v0.1\\0" || canonical_json(assertions))`
- Extraction identity: `SHA256("connectome-fusion-extraction-v0.1\\0" || UTF8(source_id))`

The canonical use material binds the scope digest, query ID, claim ID, certificate ID, operation ID, fusion-policy digest, verifier digest, dependence-bundle digest, sorted source-release IDs and sorted evidence IDs. Source-local event IDs and sequence numbers are deliberately excluded from the cross-source use key. This prevents two different certificates or dependence-bundle revisions from aliasing to one semantic-use identity.

## Expected known answer

Both the fusion execution ledger and claim/certificate ledger report two contributing fusion uses, with matching canonical use keys and evidence sets.

The candidate audit frame contains only the first use. The second operation consumes `evidence-B` and `evidence-C` but has no corresponding frame assertion.

Expected disposition: `FRAME_GAP_DETECTED`.

Expected anti-join:
- required uses absent from frame: one canonical use key (operation `fusion-op-002`);
- frame assertions without a contributing use: zero;
- execution uses absent from claim/certificate ledger: zero;
- claim/certificate uses absent from execution ledger: zero.

The frame must not proceed to #10414 sampling. Filling the missing assertion after freeze requires a new frame digest and a new sample commitment.

## Limits

The fixture demonstrates only deterministic digest conventions, cross-source event identity and frame-undercoverage detection. The declared source-coverage fields are synthetic assertions; the fixture does not authenticate custodians, prove source independence, test an event journal's completeness or establish biological independence.


## Standalone Rust verifier

The standalone crate lives at `tools/connectome-fusion-frame-closure-audit` and is deliberately excluded from the root workspace.

Run the synthetic omitted-use known answer from the repository root:

```sh
cargo run --locked --manifest-path tools/connectome-fusion-frame-closure-audit/Cargo.toml -- research/connectomics/fixtures/connectome-fusion-frame-closure-omitted-use-v0.1.json
```

Expected disposition: `FRAME_GAP_DETECTED`, with two required assertions, one declared assertion and one missing assertion. A validly evaluated blocked disposition exits successfully; malformed input or unsupported/digest-invalid input emits `NOT_ESTIMABLE`.

The crate uses exact dependency pins and its own lockfile. The workflow runs formatting, locked tests and strict Clippy on the crate independently of root-workspace compilation.

V0.1 remains intentionally narrow: it validates supplied export structure and declared coverage labels; it cannot independently authenticate custodians or prove that real-world executions were all logged.
