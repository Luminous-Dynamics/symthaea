# symthaea-science-evidence

A cross-domain experiment manifest and verification-receipt contract for
Symthaea research crates.

## What it binds

- Full source commit SHA and whether the source tree was clean.
- Toolchain and execution platform labels.
- Coordinate and unit conventions.
- Equations, assumptions, configuration and RNG seed policy.
- Content-addressed input and output artifacts using BLAKE3.
- Explicit baseline, primary metrics and uncertainty method.
- Claims, non-claims, and conditions that would falsify the claim.
- Verification status and exact-head log evidence.

## Hard integrity rules

- A commit must be a full 40-character SHA.
- An artifact digest must be a full lowercase BLAKE3 digest over non-empty bytes.
- A passed check must carry a log artifact and refer to the exact manifest commit.
- A passed run must contain at least one required check and one output artifact.
- Every required check in a passed run must be passed.
- Queued, running, skipped, not-run and failed are distinct from passed.

The manifest digest is computed from the versioned serialized schema. Preserve
the schema identifier and exact manifest bytes when archiving the run.

## Important boundary

The crate records locally supplied provenance. It does not authenticate a
runner, sign results, establish physical truth, or replace external attestations.
BLAKE3 proves byte identity only. A passed local simulation is still a local
simulation; a passed numerical benchmark is not an experimental confirmation.

## Run

    cargo test -p symthaea-science-evidence
