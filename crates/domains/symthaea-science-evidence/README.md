# symthaea-science-evidence

A cross-domain experiment manifest and verification-receipt contract for
Symthaea research. Current schema: `symthaea-science-evidence-v2`.

## What it binds

- Full source commit SHA and whether the source tree was clean.
- Toolchain and execution platform labels.
- Coordinate and unit conventions.
- Equations, assumptions, configuration and RNG seed policy.
- Content-addressed input and output artifacts using BLAKE3.
- Explicit baseline, primary metrics and uncertainty method.
- A frozen, exact set of required checks.
- Claims, non-claims, and conditions that would falsify the claim.
- Verification status and exact-head log evidence.

## Hard integrity rules

- A source revision must be a full 40-character Git SHA.
- An artifact digest must be a full lowercase BLAKE3 digest over non-empty bytes.
- `ArtifactDigest::verify_bytes` recomputes the digest and size against actual
  bytes; shape validation alone does not authenticate an artifact's contents.
- Input and output artifact paths must be unique within their respective sets.
- Every check must identify the exact manifest source SHA.
- Every receipt must contain exactly the required check names frozen in the
  manifest; missing, duplicated or unexpected check names are rejected.
- A passed check must carry a log artifact digest.
- A passed run must come from a clean source tree, match the manifest toolchain
  and platform, contain every required check and output artifact, and have
  every required check passed.
- Queued, running, skipped, not-run and failed are distinct from passed.

The manifest digest is computed from the versioned serialized schema. Preserve
the schema identifier and exact manifest bytes when archiving the run. The
digest binds the serialized manifest; it does not establish who supplied it.
The required-check list is part of that digest, so a caller cannot silently
remove a check while reusing the same manifest identity.

## Demonstration

The example refuses to infer a commit or environment from hidden defaults.
Supply the exact values explicitly:

    SYMTHAEA_SOURCE_COMMIT=<full-40-char-sha> \
    SYMTHAEA_TOOLCHAIN="rustc 1.96.0" \
    SYMTHAEA_PLATFORM="x86_64-unknown-linux-gnu" \
    SYMTHAEA_SOURCE_TREE_CLEAN=true \
    cargo run -p symthaea-science-evidence --example manifest_demo

The demonstration records every preregistered check as `NotRun`, attaches no
logs or outputs, and validates the receipt structure. It must report
`receipt_valid=true` and `qualifies_as_pass=false`; the example does not
pretend that an experiment was executed.

## Important boundary

The crate records locally supplied provenance. It does not authenticate a
runner, sign results, establish physical truth, or replace Mycelix attestation.
BLAKE3 proves byte identity only. A passed local simulation is still a local
simulation; a passed numerical benchmark is not an experimental confirmation.

## Tests

    cargo test -p symthaea-science-evidence
