# PARADOX-A0R-F0 — Label-Blind Native-State Transport V3

**Authority:** `DevelopmentOnly / Representational MeasurementOnly / TransportOnly`  
**Program issue:** #3733  
**Exact parent:** qualified M0 subject `306b0471a0854d2e77c1d50438fc493806fe1564`  
**Supersedes unqualified V2 subject:** `56abcc0c79b84320ebbb4576c5dd110fac68ee48`

F0 V3 repairs only prequalification transport/canonicalization defects. It does not authorize a PARADOX feature-acquisition campaign, semantic labels, split construction, decoder fitting, held-out scoring, behavioral mapping, or production-cognition changes.

## Public production path

Every agent-visible UTF-8 event enters the existing public `CognitiveLoopService::cycle(&str)` path. The worker observes only already-public `CycleResult.output`, `thought_vector`, and `wisdom_hv`.

`cycle_with_hv()` and harness-side text pre-encoding remain forbidden because they alter the frozen perceptual-control surface.

## Strict request decoding

The Rust worker decodes directly into a typed `serde` request struct with `deny_unknown_fields`. This is intentional:

- duplicate known keys fail deserialization;
- unknown/semantic metadata keys fail deserialization;
- numeric fields must be JSON integers of the declared Rust type;
- arrays must contain strings only;
- there is no generic `serde_json::Value` preprocessing step that can collapse duplicate keys before validation.

The four transport IDs are mechanically opaque and MUST be exactly 32 lowercase hexadecimal characters:

- `opaque_measurement_id`
- `opaque_base_fixture_id`
- `opaque_transform_id`
- `technical_pair_id`

`technical_replicate_index` is exactly 0 or 1.

## Final-cycle rule

The A0-R receipt contract preregisters one final relational-state measurement cycle. Therefore V3 requires:

`measurement_cycle_index == agent_visible_events.len() - 1`

No event is executed after the measured cycle.

## Event-stream commitment

The exact ordered agent-visible UTF-8 event sequence is committed before execution provenance is sealed:

`agent_visible_events_sha256 = SHA256(frame("PARADOX-A0R-F0-EVENTS-V1") || frame(event_0) || ... || frame(event_n))`

The digest and exact event count are provenance only. They are never decoder features. This binds a feature receipt to the causal input sequence without exposing semantic fixture labels to Plane F.

## Three evidence objects plus execution binding

### ScientificPayloadV1

Contains only measurement facts and exact raw-channel commitments. It excludes opaque IDs, service identity, Git identities, event digest, executable/environment/source commitments, labels, splits, probes, predictions, and scores.

### ProvenanceEnvelopeV2

Binds the scientific digest to:

- exact production/G2b/A0/M0/F0 lineage;
- strict opaque IDs;
- technical replicate index;
- post-science service UUID;
- exact ordered-event digest and event count;
- config projection;
- executable, environment-capsule, and source-binding commitments.

### ExecutionBindingV1

A separate label-blind manifest is frozen by the qualification/acquisition launcher before the pair auditor runs. It contains only expected execution identity:

- F0 subject SHA;
- opaque measurement/base/transform/pair IDs;
- event count and ordered-event digest;
- final measurement-cycle index;
- config-projection SHA;
- executable SHA;
- environment-capsule SHA;
- source-binding SHA.

It contains no semantic labels, condition names, expected responses, oracle output, split membership, probes, predictions, or scores.

The pair verifier MUST reject receipts that merely agree with each other but do not exactly match this external binding.

### TechnicalPairReceiptV2

Created only after both worker receipts exist. It independently:

- strict-loads JSON with duplicate-key rejection;
- rejects JSON floating-point tokens and non-standard `NaN`/`Infinity` constants;
- requires NFC strings;
- requires exact scalar types rather than `int(...)`, `str(...)`, or truthiness coercions;
- decodes raw f32 bytes and derives length/nonfinite/all-zero diagnostics;
- re-derives raw-channel, feature-bundle, scientific, and provenance commitments;
- checks frozen config diagnostics;
- checks exact fixed lineage and config identities;
- checks each receipt against `ExecutionBindingV1`;
- requires peer-invariant provenance for every field except replicate index, service UUID, and receipt hash;
- requires technical indices `[0, 1]`, distinct service UUIDs, identical scientific hashes, and distinct provenance hashes.

Any violation fails the pair as `INVALID_MEASUREMENT_NONDETERMINISTIC`. Technical repeats never increase statistical `n`.

## Science/provenance ordering

All raw scientific bytes and `scientific_payload_sha256` are sealed before the worker creates the random `service_instance_id`. Provenance randomness is therefore downstream of the scientific path.

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

Canonical projection SHA-256:
`2cd99b06cf2cf11cfc0612c0db818355099b48b04bd687ca66a31a08680be1f9`.

This projection is not represented as a hash of every production-config field. Exact source blobs, lockfile, executable, target/features, and environment are separately bound.

## Qualification boundary

Committing V3 earns only:

`F0_IMPLEMENTED_UNQUALIFIED`

A separate workflow-only exact-subject qualifier must execute compile/runtime/noninterference/mutation/attempt-retention/postflight gates before promotion to `FEATURE_ACQUISITION_QUALIFIED`. That state would mean only that the transport mechanism is qualified; it would not mean that a PARADOX feature dataset has been collected.

## Claim ceiling

Neither implementation nor future transport qualification establishes latent decodability, native PARADOX behavior, production-policy use, causal necessity, metacognitive recruitment, ontology repair, cross-genesis generality, consciousness, sentience, phenomenology, or superiority of a consciousness theory.
