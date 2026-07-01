# Alpha.9 Release Manifest

Alpha.9 adds a small release manifest that records blocked claims and recommended local verification commands.

The manifest exists because research crates can become dangerous when examples, reports, or downstream notebooks accidentally imply more than the crate can support.

## Explicitly blocked claims

Alpha.9 blocks interpretation as:

- quantum consciousness evidence
- quantum advantage evidence
- physical QPU execution unless an external adapter attaches raw backend metadata
- cryptographic Mycelix attestation from local fingerprints
- medical, safety-critical, or production engineering guidance

## CLI

Run:

`cargo run --bin symthaea-quantum-comp -- manifest`

## Example

Run:

`cargo run --example release_manifest`

## Recommended local verification

The manifest recommends format checks, full tests, smoke gates, matrix examples, and research receipt examples before publishing local reports.

These checks are still local-only. They do not replace peer review, external audit, real backend validation, or formal Mycelix source-chain signing.
