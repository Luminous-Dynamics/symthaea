# ADR-011: Independent RSK Qualification Receipt Verification and Exact-Head CI

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

ADR-010 introduced an RSK qualification generator that executes pinned Rust gates and emits source/tool/input-bound receipts.

A receipt generator should not be the only implementation capable of deciding whether its own output is internally valid. A defect or weakening in the generator could otherwise affect both the evidence and the logic used to interpret that evidence.

A second evidence issue exists in pull-request CI: GitHub's default `pull_request` checkout commonly represents a synthetic merge ref. That is valuable integration evidence, but it is not identical to qualifying the exact PR head that a Class A reviewer is evaluating.

This ADR contains no physical replication mechanism, fabrication recipe, molecular design, biological implementation, or autonomous manufacturing path.

## Decision

Add an independent verifier:

```text
scripts/verify_rsk_qualification.py
```

and adversarial verifier tests:

```text
scripts/test_verify_rsk_qualification.py
```

The focused RSK workflow must:

1. check out the exact pull-request head SHA for RSK Class A, format, and core qualification;
2. run the qualification generator;
3. run the independent verifier against the generated receipt/logs and the same checkout;
4. retain the capsule artifact whether qualification succeeds or fails;
5. run generator and verifier self-tests in the governance job.

Repository-wide CI remains useful as broader merge/integration evidence. The focused RSK lane is specifically exact-candidate evidence.

## Independent implementation rule

The verifier does **not** import the generator's command-plan implementation.

It independently encodes the v0.1 required command names and argument vectors. This controlled duplication is intentional:

> a unilateral weakening or accidental drift in generator or verifier should produce disagreement, not silently redefine the qualification profile for both implementations.

Both files are Class A and changes require an RSK Class A ADR.

## Receipt properties independently checked

The verifier must reject a receipt unless it can independently establish at least:

- exact receipt schema and qualification scope;
- explicit production-admission denial marker;
- canonical receipt SHA-256;
- optional `receipt.sha256` sidecar consistency;
- no duplicate JSON object keys;
- exact v0.1 qualification phase;
- exact required command count, order, names, and argv;
- `--locked` on semantic Cargo commands;
- `--all-targets` on the declared semantic target surface;
- `-D warnings` on Clippy;
- safe relative log paths with no traversal/escape;
- each retained log exists and matches its SHA-256;
- bounded receipt/log sizes;
- exact selected input-path set;
- valid before/after input digests;
- `inputs.unchanged` agrees with the actual before/after hash maps;
- dirty flags agree with their path lists;
- recorded Rust pin-match agrees with the retained `rustc --version --verbose` evidence;
- qualification status is the status independently implied by commands/toolchain/worktree/input facts;
- `admissible_evidence` is independently implied by those same facts.

The verifier can additionally bind a receipt to a local checkout by checking:

- current Git commit;
- current Git tree;
- current selected input hashes;
- current repository Rust channel.

CI uses this checkout-bound mode.

## Exact PR head rule

For `pull_request` runs, the focused RSK workflow explicitly checks out:

```text
github.event.pull_request.head.sha
```

rather than relying on the default PR merge ref.

The qualification receipt therefore records the exact candidate commit/tree under Class A review.

For `push` runs, the workflow checks out `github.sha`.

This does not eliminate the value of merge-candidate testing. It separates two evidence questions:

```text
focused RSK qualification -> exact candidate head
ordinary repository CI    -> broader integration/merge behavior
```

Neither evidence class silently substitutes for the other.

## Path and parser hardening

The verifier treats the receipt as untrusted input.

It therefore:

- bounds receipt and log sizes before hashing/processing;
- rejects duplicate JSON keys;
- rejects unknown top-level receipt fields;
- rejects absolute log paths and `..` traversal;
- resolves log paths and requires them to remain under the declared capsule root;
- validates digest syntax and expected scalar types;
- rejects unknown statuses/phases/schemas.

This is evidence-parser hardening, not a claim that Python itself is a production cryptographic TCB.

## Status recomputation

For non-blocked receipts, the verifier derives the expected state from evidence rather than trusting summary fields.

Conceptually:

```text
commands_pass
pin_match
pre_run_clean
post_run_clean
inputs_unchanged
    -> derived qualification_status
    -> derived admissible_evidence
```

A receipt cannot make itself admissible by changing `admissible_evidence: true` and recomputing its outer SHA-256.

Likewise, a receipt cannot omit a required command, alter an argv, claim a favorable pin-match, or point a log outside the capsule and remain valid merely by recomputing the receipt digest.

## Adversarial verifier self-test

The candidate verifier self-test covers at least:

1. valid receipt/logs against the exact generating checkout;
2. receipt-field tamper without digest update;
3. retained-log tamper;
4. command omission followed by attacker recomputation of the receipt digest;
5. false `toolchain.pin_match` followed by receipt rehash;
6. false `inputs.unchanged` followed by receipt rehash;
7. log path traversal followed by receipt rehash;
8. substitution of a selected input in the current checkout.

These tests use fake Rust/Cargo tools so verifier semantics can execute without requiring a Rust installation.

## Failure receipts remain useful

The verifier checks the integrity and internal consistency of both passing and failing qualification receipts.

A validly verified failure receipt means:

> the retained evidence consistently describes a failed qualification.

It does **not** turn failure into success.

The generator step's nonzero exit still fails the CI job; the verifier runs with `if: always()` so the failure evidence itself can be checked and retained.

## Non-claims

Independent receipt verification does not establish:

- signer/executor identity;
- trusted hardware execution;
- a hermetic build;
- formal correctness;
- runtime binary identity;
- physical containment;
- protected-branch enforcement;
- production admission.

A SHA-256 receipt digest is an integrity commitment, not an attestation of who produced it.

## Alternatives considered

### Let the generator verify itself

Rejected. A shared bug or weakened command definition could validate its own weakened output.

### Import the generator's `command_plan()` into the verifier

Rejected for the v0.1 safety boundary. It reduces useful implementation diversity and allows one changed function to redefine both production and verification expectations.

### Qualify only GitHub's synthetic PR merge ref

Rejected as the sole focused safety evidence. Merge-ref integration testing is useful, but Class A review also needs exact-head qualification identity.

### Require a cryptographic signature on receipts immediately

Deferred. Signature provenance is useful future work, but it does not replace independent semantic verification of receipt contents.

## Verification discipline

The verifier branch itself is not considered qualified until its exact-head governance/self-test workflow executes.

The current execution container could not resolve GitHub's raw-content host, so an attempted exact-branch local execution path was unavailable. That failed retrieval is **not** counted as test evidence.

Authored adversarial tests are design evidence until independently executed on the exact candidate head.

## Consequences

### Positive

- generator output is checked by a separate implementation;
- a recomputed outer digest cannot hide semantic receipt weakening;
- exact PR-head identity becomes explicit;
- log retention becomes independently verifiable;
- command-profile drift becomes detectable;
- evidence parser/path hardening is explicit;
- failing qualifications can retain trustworthy failure evidence.

### Residual risk

- generator and verifier are still implemented in the same language/repository and may share conceptual mistakes;
- receipt digests are unsigned;
- executor identity is not cryptographically attested;
- GitHub runner availability remains external;
- exact build/runtime identity and production admission remain separate gates.

## Related work

- ADR-010 — qualification capsule and receipts
- #1335 — production-admission umbrella blocker
- #1672 — formal refinement
- #1682 — exact build/runtime identity
- #1806 — qualification capsule candidate
- #1776 — trusted-time boundary
