# PARADOX-A0R-F0 — Label-Blind Native-State Transport V2

**Authority:** `DevelopmentOnly / Representational MeasurementOnly / TransportOnly`  
**Program issue:** #3733  
**Exact parent:** qualified M0 subject `306b0471a0854d2e77c1d50438fc493806fe1564`  
**Supersedes unqualified v1 subject:** `876ff2d308fd10ed0873f912079e2aa02a58c834`

F0 V2 repairs only verifier weaknesses found before v1 qualification. It does not authorize a PARADOX feature-acquisition campaign, semantic labels, split construction, decoder fitting, held-out scoring, behavioral mapping, or production-cognition changes.

## Frozen public path

Every agent-visible UTF-8 event enters the existing public `CognitiveLoopService::cycle(&str)` path. The worker observes only already-public `CycleResult.output`, `thought_vector`, and `wisdom_hv`.

`cycle_with_hv()` and harness-side text pre-encoding remain forbidden because they change the frozen perceptual-control surface.

## Three acyclic evidence objects

1. **ScientificPayloadV1** contains only measurement facts and exact raw-channel commitments. It excludes all provenance identity, labels, splits, probes, predictions, and scores.
2. **ProvenanceEnvelopeV1** binds the scientific digest to exact lineage, strict opaque IDs, replicate index, service identity, config/executable/environment/source commitments.
3. **TechnicalPairReceiptV1** is created after both worker receipts. It independently revalidates raw bytes, derives f32 diagnostics, revalidates science/provenance hashes, requires peer-invariant provenance, and then verifies the two-replicate relation.

## Strict opaque identity

The four experiment-transport identifiers are mechanically opaque:

- `opaque_measurement_id`
- `opaque_base_fixture_id`
- `opaque_transform_id`
- `technical_pair_id`

Each MUST be exactly 32 lowercase hexadecimal characters (`[0-9a-f]{32}`). Semantic names are invalid. They are provenance only and must not affect genesis, configuration, event bytes/order, cycle selection, retries, feature transforms, or scientific bytes.

`technical_replicate_index` MUST be exactly 0 or 1.

## Frozen science/provenance ordering

The worker must complete the scientific path and seal `scientific_payload_sha256` **before** generating the random `service_instance_id`. The UUID is therefore downstream provenance only.

A bijective opaque-ID rename with unchanged events must preserve the scientific payload hash and change the provenance receipt hash.

## Independent raw-diagnostic reconstruction

The pair auditor does not trust diagnostic fields merely because they rehash consistently. From the raw f32 little-endian bytes it independently derives:

- recurrent vector length;
- thought-vector length;
- recurrent nonfinite count;
- thought-vector nonfinite count;
- recurrent exact-positive-zero all-zero status (`u32 bits == 0` for every element);
- wisdom-HV byte length;
- all three raw channel SHA-256 commitments;
- the feature-bundle commitment.

It then independently derives the expected invalidity-reason list and measurement-validity status and requires the worker declaration to match exactly.

The pair auditor also requires:

- `recurrent_masking_enabled == false`;
- `spectral_entropy_masking_enabled == false`;
- `effective_dim_fraction_override_is_none == true`;
- exact fixed production/G2b/A0/M0 identities;
- exact frozen config-projection identity;
- syntactically valid F0/executable/environment/source commitments.

## Peer-invariant provenance

Across the two technical replicas, every provenance field must be identical except:

- `technical_replicate_index` (exactly 0 then 1);
- `service_instance_id` (must differ);
- `receipt_binding_sha256` (must differ).

In particular, measurement/base/transform/pair IDs, all lineage identities, config projection, executable, environment capsule, source binding, and scientific payload digest must match.

Any violation fails the whole pair as `INVALID_MEASUREMENT_NONDETERMINISTIC`. Technical repeats never increase statistical `n`.

## Frozen runner configuration projection

The worker starts from `CognitiveLoopConfig::with_cfc()` and explicitly fixes:

- genesis `PARADOX-A0R-DEV-V1-GENESIS-2026-09-16`;
- CfC neurons/input dimension 256/256;
- delta-t 0.02;
- horizons [0.02, 0.1, 0.2];
- async and online learning off;
- episodic replay training off;
- memory graduation off;
- recurrent and spectral masking off;
- effective-dimension override absent;
- attention budget 60,000,000 us;
- timezone offset 0.0.

Canonical projection SHA-256:
`2cd99b06cf2cf11cfc0612c0db818355099b48b04bd687ca66a31a08680be1f9`.

This is a declared causal projection, not a hash of every field in the large production config. Exact source blobs, lockfile, executable, target/features, and environment are separately bound.

## Evidence state and claim ceiling

Committing this implementation earns only:

`F0_IMPLEMENTED_UNQUALIFIED`

A separate workflow-only exact-subject qualifier must execute compile/runtime/noninterference/mutation/attempt-retention/postflight gates before promotion to `FEATURE_ACQUISITION_QUALIFIED`. That promotion would still mean only that the transport mechanism is qualified; it would **not** mean that a PARADOX feature dataset has been collected.

Neither implementation nor future transport qualification establishes latent decodability, native PARADOX behavior, production-policy use, causal necessity, metacognitive recruitment, ontology repair, cross-genesis generality, consciousness, sentience, phenomenology, or superiority of a consciousness theory.
