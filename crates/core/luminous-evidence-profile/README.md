# Luminous Evidence Profile verifier

This crate provides semantic conformance checks for the LEP v0.1 predicate carried by an in-toto Statement v1.

## Run locally

From the Symthaea repository root:

    cargo run -p luminous-evidence-profile -- docs/research/fixtures/luminous-evidence-v0.1/valid-passed.json

The CLI prints a JSON report. It exits with code 0 if the LEP semantic checks pass, 1 if the statement is semantically invalid, and 2 if the input cannot be read or parsed.

Run the unit tests with:

    cargo test -p luminous-evidence-profile

The crate includes fixture-backed tests for a valid illustrative statement, queued-but-passed execution, missing target identity, and a stale checked target.

## What it checks

- in-toto Statement v1 and the LEP v0.1 predicate type/version.
- Required claim and target fields.
- Pinned lowercase 40- or 64-character hexadecimal source revisions.
- Supported digest algorithms and digest shapes: sha256, sha512, and blake3.
- Execution status/result consistency.
- Exact structured equality of the declared target and verification checked target.
- Verifier identity, non-empty executed checks, valid check outcomes, and unique check IDs.
- The rule that a passed verification requires completed, passed execution and all recorded checks to pass.
- The rule that not_run cannot carry a verifier, checked target, or checks.

## What it does not establish

This semantic checker is one layer in the validation pipeline. It does not:

- validate the full JSON Schema;
- authenticate DSSE signatures or verify trusted signer identity/authority;
- fetch artifacts or recompute hashes of their bytes;
- prove that a claimed run was actually executed;
- independently rerun the named checks;
- prove scientific correctness, physical validity, governance legitimacy, or generalization.

Use the JSON Schema validator and the in-toto/DSSE verification implementation separately. Before treating a record as trusted evidence, the caller must resolve its referenced artifacts, recompute hashes, verify the authenticated statement, compare target identities, and apply the domain's independent verifier. A passing CLI result alone is not a production assurance verdict.

The included JSON fixtures contain synthetic placeholder digests and are for conformance testing only.
