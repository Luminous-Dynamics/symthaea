# PARADOX-A0R-F0 — Label-Blind Native-State Transport V1

**Authority:** `DevelopmentOnly / Representational MeasurementOnly / TransportOnly`  
**Program issue:** #3733  
**Exact qualified parent:** M0 subject `306b0471a0854d2e77c1d50438fc493806fe1564`

F0 is the first executable Plane-F transport boundary after the M0 firewall. It does **not** run a PARADOX feature-acquisition campaign and does not read semantic labels, condition names, expected responses, oracle data, split assignments, probe artifacts, predictions, or scores.

## Frozen purpose

Prove that an external research worker can transport already-public native state from the exact frozen production path without giving semantic experiment metadata any route into cognition, cycle selection, configuration, retries, or scientific feature bytes.

The only production observations are:

- `CycleResult.output` — primary recurrent channel;
- `CycleResult.thought_vector` — 32-D text-path perceptual projection control;
- `CycleResult.wisdom_hv` — cached binary text-perception HDC control.

F0 adds **no** production accessor and changes **no** production cognition source.

## Text-path binding

Every V1 event is an agent-visible UTF-8 string and MUST enter the exact public `CognitiveLoopService::cycle(&str)` path. `cycle_with_hv()` is forbidden for this lineage. Harness-side text-to-HV encoding is also forbidden.

This matters because the frozen `cycle_with_hv()` helper constructs a different control surface: it hardcodes `thought_vector` to 32 zeros and derives `wisdom_hv` from the supplied external HV. Using it would invalidate the preregistered recurrent-vs-perceptual-control comparison.

## Three acyclic evidence objects

### 1. ScientificPayloadV1

Contains only measurement facts that must remain invariant when provenance-only names change. Raw recurrent and control bytes are encoded exactly and committed before any later semantic join.

The scientific commitment excludes opaque IDs, replicate index, service identity, Git identities, executable/environment commitments, labels, splits, probes, predictions, and scores.

### 2. ProvenanceEnvelopeV1

Binds the scientific digest to exact subject lineage, opaque row/pair identities, technical-replicate index, a fresh service identity, runner-config projection, executable, environment capsule, and source-binding commitment.

Opaque provenance is never a decoder feature.

### 3. TechnicalPairReceiptV1

Created only after both worker receipts exist. It verifies:

- replicate indices are exactly `[0, 1]`;
- service-instance IDs differ;
- scientific payload hashes are identical;
- provenance receipt hashes differ;
- both worker receipts independently revalidate from their raw channel bytes.

A missing peer, reused service identity, raw/scientific mismatch, or invalid peer fails the complete pair as `INVALID_MEASUREMENT_NONDETERMINISTIC`. Technical repeats never increase statistical `n`.

## Worker capability boundary

The worker accepts NDJSON on stdin only. Qualification launches it with no sidecar arguments and an explicit environment allowlist. Unknown request keys and recursively detected semantic key names fail closed before a `CognitiveLoopService` is constructed.

Allowed request data is limited to:

- protocol version;
- opaque measurement/base/transform/pair IDs;
- technical-replicate index;
- ordered agent-visible text events;
- preregistered measurement-cycle index;
- exact F0 subject identity supplied by the qualification runner;
- frozen config-projection commitment;
- executable, environment-capsule, and source-binding commitments.

Opaque IDs and service identity may affect only provenance framing. They may not affect genesis, configuration, event contents/order, cycle selection, retry policy, feature transforms, or scientific payload bytes.

## Frozen service configuration

The research worker starts from `CognitiveLoopConfig::with_cfc()` and then explicitly fixes the A0 V1 transport-relevant projection:

- genesis phrase `PARADOX-A0R-DEV-V1-GENESIS-2026-09-16`;
- CfC `num_neurons = 256`;
- CfC `input_dim = 256`;
- `delta_t = 0.02`;
- prediction horizons `[0.02, 0.1, 0.2]`;
- asynchronous training off;
- online learning off;
- episodic replay training off;
- memory graduation off;
- recurrent dimension masking off;
- spectral-entropy masking off;
- effective-dimension override `None`;
- attention budget override `60_000_000 us`;
- timezone offset `0.0`.

The canonical projection digest is `2cd99b06cf2cf11cfc0612c0db818355099b48b04bd687ca66a31a08680be1f9`. It is explicitly a **projection** commitment, not a claim to hash every field of the large `CognitiveLoopConfig`; qualification separately binds source, executable, features, target triple, lockfile, and environment capsule.

## Scientific/provenance noninterference

A bijective rename of opaque measurement/base/transform/pair IDs with identical agent-visible events MUST satisfy:

```text
scientific_payload_sha256(before) == scientific_payload_sha256(after)
receipt_binding_sha256(before)    != receipt_binding_sha256(after)
```

Two technical replicas of the same measurement MUST satisfy:

```text
scientific payload bytes/hash     == bit-identical
service_instance_id[0]            != service_instance_id[1]
receipt_binding_sha256[0]         != receipt_binding_sha256[1]
```

Plan-order permutations may change the order in which independent rows are executed, but not any row's scientific bytes when its event stream and frozen cycle selector are unchanged.

## Validity

A worker measurement is transport-eligible only when:

- recurrent length is exactly 256;
- thought-vector length is exactly 32;
- `wisdom_hv` is exactly 2048 bytes;
- recurrent and thought-vector values are finite;
- recurrent/spectral masking remain disabled;
- effective-dimension override remains absent;
- exact source/config/executable/environment bindings pass externally.

An all-zero recurrent vector is recorded but is not automatically invalidated. Presence, nonzero values, or reproducibility do not establish semantic content.

## Qualification boundary

This F0 implementation commit earns only:

`F0_IMPLEMENTED_UNQUALIFIED`

A separate exact-subject qualifier must execute the preregistered negative controls, locked/offline build gates, independent receipt recomputation, pair repeatability, source/config/path binding, attempt retention, and postflight immutability before `FEATURE_ACQUISITION_QUALIFIED` can be considered.

No real PARADOX feature-acquisition dataset may be collected under this implementation commit merely because the static audit passes.

## Claim ceiling

Even a later qualified F0 establishes only reproducible, label-blind transport of the specified already-exported channels under the frozen subject/configuration and noninterference controls.

It does **not** establish latent decodability, a native PARADOX response, behavioral competence, production-policy use, causal necessity, metacognitive recruitment, ontology repair, cross-genesis generality, consciousness, sentience, phenomenology, or superiority of any consciousness theory.
