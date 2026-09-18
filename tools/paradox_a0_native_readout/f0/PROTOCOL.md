# PARADOX-A0R-F0 — Label-Blind Native-State Transport V4

**Authority:** `DevelopmentOnly / Representational MeasurementOnly / TransportOnly`  
**Program issue:** #3733  
**Exact parent:** qualified M0 subject `306b0471a0854d2e77c1d50438fc493806fe1564`  
**Supersedes unqualified V3 subject:** `1760cf4a331793814ec47552a86565d8f6163646`

F0 V4 keeps the V3 execution-binding and strict-canonicalization design but removes the last unrelated dependency path before qualification.

## Dependency-plane isolation

The frozen M0 manifest explicitly records that root `[[example]]` targets require `[dev-dependencies]` and can therefore be blocked by unrelated broken dev crates. V4 does not use an example target and does not alter the manifest.

The research worker lives at:

`src/bin/paradox_a0r_f0_worker.rs`

Cargo auto-discovers it as a binary because the root package does not disable automatic binary discovery. V4 changes neither `Cargo.toml` nor `Cargo.lock`.

The worker uses only dependencies already present in the frozen root package (`serde`, `serde_json`, `uuid`) and implements the small SHA-256 commitment primitive internally. The SHA-256 implementation executes built-in startup known-answer tests for the empty message and `abc` before any request is accepted. Qualification independently checks additional SHA-256 vectors against Python `hashlib`.

This SHA implementation is evidence-transport plumbing only; it is not production cognition or cryptographic key material.

## Public production path

Every agent-visible UTF-8 event enters the existing public `CognitiveLoopService::cycle(&str)` path. The worker observes only already-public `CycleResult.output`, `thought_vector`, and `wisdom_hv`.

`cycle_with_hv()` and harness-side text pre-encoding remain forbidden.

## Strict request decoding

The worker decodes directly into a typed `serde` struct with `deny_unknown_fields`:

- duplicate known keys fail deserialization;
- unknown/semantic keys fail deserialization;
- numeric fields must have their declared integer type;
- event arrays contain strings only;
- no generic JSON `Value` pass can collapse duplicate keys before validation.

The four transport IDs are exactly 32 lowercase hexadecimal characters. Technical replicate index is exactly 0 or 1.

## Final-cycle rule

`measurement_cycle_index == agent_visible_events.len() - 1`

No event executes after the measured cycle.

## Event-stream commitment

Domain: `PARADOX-A0R-F0-EVENTS-V1`.

The exact ordered UTF-8 event stream is length-framed and SHA-256 committed. Provenance binds both event count and event-stream digest; neither is a decoder feature.

## Scientific / provenance / execution separation

### ScientificPayloadV1

Only measurement facts and exact raw-channel commitments. It excludes opaque IDs, service identity, Git identities, event digest, executable/environment/source commitments, labels, splits, probes, predictions, and scores.

### ProvenanceEnvelopeV2

Binds the scientific digest to exact lineage, strict opaque IDs, replicate index, post-science service UUID, event count/digest, config projection, executable, environment capsule, and source binding.

### ExecutionBindingV1

A separately frozen label-blind manifest supplies the expected F0 subject, opaque IDs, event count/digest, final cycle, config projection, executable, environment capsule, and source binding. The independent verifier rejects two mutually consistent receipts if they disagree with this external binding.

### TechnicalPairReceiptV2

The independent Python verifier uses strict duplicate-key/non-finite/float-token/NFC JSON rules, exact scalar types, raw f32 reconstruction, independent raw/scientific/provenance rehashing, frozen diagnostics, external execution-binding checks, and exact peer-invariant provenance.

A missing peer, reused service identity, mismatched science, type ambiguity, parser ambiguity, provenance drift, or external-binding mismatch fails the pair as `INVALID_MEASUREMENT_NONDETERMINISTIC`. Technical repeats never increase statistical `n`.

## Frozen configuration projection

The worker starts from `CognitiveLoopConfig::with_cfc()` and fixes:

- genesis `PARADOX-A0R-DEV-V1-GENESIS-2026-09-16`;
- CfC neurons/input dimension 256/256;
- delta-t 0.02;
- horizons `[0.02, 0.1, 0.2]`;
- async and online learning off;
- episodic replay training off;
- memory graduation off;
- recurrent and spectral masking off;
- effective-dimension override absent;
- attention budget 60,000,000 us;
- timezone offset 0.0.

Projection SHA-256:
`2cd99b06cf2cf11cfc0612c0db818355099b48b04bd687ca66a31a08680be1f9`.

Exact source blobs, manifest/lockfile, executable, target, and environment remain separately bound.

## Qualification boundary

Committing V4 earns only:

`F0_IMPLEMENTED_UNQUALIFIED`

A separate workflow-only exact-subject qualifier must build only the auto-discovered F0 binary with locked/offline dependencies, independently validate SHA-256, execute two fresh-service replicas and the frozen mutation controls, retain attempt evidence, archive a complete evidence bundle, and prove postflight subject immutability before `FEATURE_ACQUISITION_QUALIFIED` may be promoted.

That promotion would qualify only the transport mechanism; it would not mean that a PARADOX feature dataset has been collected.

## Claim ceiling

Neither implementation nor future transport qualification establishes latent decodability, native PARADOX behavior, production-policy use, causal necessity, metacognitive recruitment, ontology repair, cross-genesis generality, consciousness, sentience, phenomenology, or superiority of a consciousness theory.
