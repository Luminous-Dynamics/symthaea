# Symthaea REA Evidence Verifier

A small Rust CLI for independently checking a downloaded Unblob provider-smoke evidence bundle. It performs no network calls and never executes recovered firmware.

## What it verifies

- The independently supplied source-head SHA matches the receipt's source head.
- The receipt schema and qualification ceiling are exactly the supported Unblob 26.6.4 provider-smoke tier; provider-only evidence cannot claim REA CLI/MCP parity, firmware execution, emulation, or device qualification.
- SHA-256 identities of the checked-out normalizer, fixture manifest, oracle, exact Python dependency lock, provider-smoke script, and provider-metamorphic script match the source receipt.
- The resolved `pip freeze` matches the checked-in version lock byte-for-byte; original and mutated gzip sources match their expected input digests.
- Baseline and mutated input/payload manifests match the actual input and source payload files.
- Raw provider report files and semantic projection files match their receipt digests; equivalent runs have equal projection digests, while the controlled semantic change has a different digest.
- Extracted payload files match independent expected hashes.
- The provider-metamorphic sub-receipt is byte-bound to the outer receipt and agrees on report, normalizer, projection, and variant-count identities.
- A sorted bundle manifest covers every produced artifact file except the outer receipt and the manifest itself; missing, extra, symlinked, malformed, or digest-mismatched files are rejected.
- Receipt and digest-manifest inputs are bounded; symlinked files and extraction-search roots are rejected.

## Build and test

From the Symthaea workspace:

    cargo fmt --package symthaea-rea-evidence-verifier
    cargo test -p symthaea-rea-evidence-verifier

## Verify a downloaded artifact

Unzip the artifact from [the pinned provider workflow](https://github.com/Luminous-Dynamics/symthaea/actions/workflows/rea-provider-smoke.yml). Check out the exact source head named by that run, then execute:

    cargo run -p symthaea-rea-evidence-verifier -- \
      --artifact-dir /path/to/rea-unblob-26.6.4-provider-evidence \
      --source-root benchmarks/rea-determinism/v1 \
      --expected-head 40_HEX_SOURCE_HEAD

The verifier emits deterministic JSON with verification_mode set to integrity-consistency-only. To additionally compare the raw receipt to an independently recorded receipt SHA-256, add:

    --expected-receipt-sha256 TRUSTED_RECEIPT_SHA256

That produces verification_mode set to receipt-digest-anchored only when the digest matches. The digest must come from a separately trusted channel; computing it from the same artifact does not create an independent trust anchor.

## Claims ceiling

The provider dependency lock pins the observed Linux/Python 3.12 package set by version, not by wheel hash; it reduces resolver drift but is not a supply-chain signature or hermetic build proof. This checks internal consistency and file digests, not authorship or a digital signature. A malicious party able to rewrite every artifact file and the receipt can construct a new self-consistent bundle; an independent receipt digest or signed CI attestation is needed to anchor it. Even anchored, the evidence only qualifies one pinned Unblob CLI smoke on inert generated fixtures. It does not establish extraction completeness for arbitrary firmware, content safety, REA CLI/MCP parity, firmware behavior, or device qualification.
