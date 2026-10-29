# MEL-MR-EVAL-007 — Content-Bound Symbolic Edit-Locality Evidence

**Status:** Rust implementation draft; test execution pending  
**Tracking:** [Issue #8934](https://github.com/Luminous-Dynamics/symthaea/issues/8934)  
**Depends on:** MEL-MR-EVAL-004 symbolic edit locality and MEL-MR-EVAL-006 audio preservation

## Research question

Can symbolic edit-locality evidence be bound to the exact source score, target score, request, thresholds, property measurements, and verdict so that later consumers can reject accidental cross-input reuse and report tampering?

This increment addresses the symbolic receipt, not the whole symbolic-to-audio proof. A receipt hash is an unkeyed integrity commitment: it does not authenticate the producer, prove the score came from a particular musician, or prove a renderer rendered that score.

## Implemented shape

`EditLocalityEvidence` contract version `edit-locality-v1.1` now records:

- a domain-separated SHA-256 digest of the exact source event sequence;
- a domain-separated SHA-256 digest of the exact target event sequence;
- a receipt SHA-256 over the serialized evidence with its receipt field blank, including request, config, requested-property result, all protected-property results, verdict, and failure reasons.

The new `verify_edit_locality_evidence(source, target, evidence)` verifier checks the report digest, recomputes source/target content digests, replays the request with the recorded configuration, and requires the complete replayed evidence to match. It returns false for modified inputs or modified evidence; malformed/unsupported replay still returns the existing edit-locality error. Serialization failure is explicit and fail-closed.

Runtime `serde_json` dependency is used for serialization, and `sha2` already belongs to this crate's runtime dependency set. The contract version is bumped because the receipt format changed.

## Unit-test additions

The edit-locality corpus replay now verifies each generated evidence object against its inputs. Added tests cover source substitution, target substitution, verdict tampering, tolerance tampering, property-check tampering, receipt digest tampering, and a successful serialize/deserialize/replay round trip.

These tests are committed but have **not been executed** in the current environment. Rust compilation, rustfmt, and clippy status must be tied to an exact commit before this draft can be considered validated.

## Next cross-modal binding step

The system still needs a separate render-binding object tying together:

1. verified source and target symbolic score digests;
2. the edit request and symbolic receipt digest;
3. the rendered reference/output audio digests from MEL-MR-EVAL-006;
4. renderer/version, sample rate, channel layout, instrument/effect configuration, and latency policy;
5. a verifier that can recompute the binding and rejects a score/audio pair that was substituted after rendering.

Only after this binding exists should the overall edit acceptance gate require symbolic pass **and** audio-composite pass, with a critical fail dominating and missing evidence yielding abstention. This implementation does not yet claim that end-to-end guarantee.

## Benchmarks and acceptance gates

- 100% of the declared symbolic fixture corpus yields the same expected property outcomes and replay-verifiable evidence.
- Source/target substitutions and any changed verdict/config/property report are rejected.
- Serializing and deserializing evidence preserves exact replay behavior.
- The later cross-modal suite includes mismatched render manifests, changed instrument banks, renderer-version drift, and swapped score/audio inputs.
- No reliability or SOTA claim follows from unit fixtures alone; the end-to-end guard must be evaluated in a paired editor ablation against the same tasks and resource budget.

The goal is a content-bound, independently replayable claim about what changed and what remained protected—not a hash attached to an unverified assertion.
