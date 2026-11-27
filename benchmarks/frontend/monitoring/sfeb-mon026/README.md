# MON-026 — Evidence Custody and Reproducibility Envelope

**Status:** Go standard-library reference verifier, deterministic manifest digest, Ed25519 signature validation, artifact hashing, chain-link checks and adversarial tests added; MON-025 packager integration, trusted-key operations and independent custody qualification pending.

MON-025 produces source, CDC, backup, restore and state-comparison artifacts. MON-026 makes those bytes independently verifiable after collection.

It answers:

> Given a run bundle copied to another machine, can a verifier establish that every declared artifact is present and unchanged, the manifest has not been substituted, its signer is trusted, its clock claims are bounded, and the bundle links to the expected previous run?

## The paradigm

Instead of trusting a dashboard, CI result, connector status or recovery orchestrator, preserve a portable evidence bundle that can be checked offline by a separate verifier.

The envelope binds:
- exact artifact paths, lengths, SHA-256 digests, media types, roles and producer identifiers;
- a shared run/source/database/timeline/snapshot/recovery boundary;
- collection clock domain, uncertainty and timestamp;
- ordered bundle sequence and previous-manifest digest;
- a deterministic semantic manifest root;
- an Ed25519 signature over the root digest.

The verifier requires a separately configured trusted public key. A key embedded in the bundle is not automatically trusted. A valid signature proves that a holder of the trusted private key signed the manifest bytes; it does not prove that the signed claims are true. Source/target facts still need independent observers.

## Current implementation

`custody/` is a Go 1.23 standard-library-only reference package. It provides:
- deterministic, field-framed manifest hashing (JSON whitespace/key order do not change the semantic root);
- Ed25519 signing and verification helpers;
- artifact path traversal and symlink rejection;
- file length and SHA-256 verification;
- signer trust lookup and invalid-signature rejection;
- run sequence / previous-root chain validation;
- capture freshness, clock-domain and uncertainty checks;
- fail-closed outcomes: `qualified`, `blocked` or `indeterminate`;
- adversarial tests, fuzz target and microbenchmark.

## Use

The package exposes `ComputeRoot`, `SignManifest` and `VerifyBundle`. The caller supplies an out-of-band trusted-key map and expected previous root. No private key or trust anchor is bundled with the run.

See `custody/README.md` for the model, offline CLI and test commands, and `EVIDENCE-CUSTODY-CONTRACT-v1.md` for the complete trust boundary.

## Explicit limits

- No MON-025 packager is included yet; adapters must produce the manifest and include the raw files.
- No trusted key provisioning, revocation service, transparency log, HSM, timestamp authority or remote WORM object storage is implemented.
- The file verifier rejects symlinks and path traversal but assumes the bundle filesystem is not concurrently modified by a hostile actor during verification. Stronger hostile-filesystem guarantees require platform-specific no-follow/openat-style handles or immutable storage.
- A trusted signature authenticates who signed a manifest, not whether the source state, WAL interval or observer conclusion is truthful.
- It does not promote a MON-025 run to production qualification by itself.

## Files

- `EVIDENCE-CUSTODY-CONTRACT-v1.md` — manifest semantics, trust model, threat model and chain rules.
- `MANIFEST-v1.schema.json` — strict JSON manifest structure.
- `SCENARIOS-v1.json` / schema — custody adversarial seed corpus.
- `BENCHMARKS-v0.1.md` — integrity, overhead and operator acceptance targets.
- `custody/` — Go verifier, tests, fuzz target and benchmark.
