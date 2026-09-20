# HUM-OBS-001D — Typed Observatory Evidence Values

Status: source-design candidate
Issue: #4943
Parent: HUM-OBS-003A / #4930
Authority: evidence representation only; **no physical, safety, or deployment authority**

## Purpose

Prevent exact benchmark/equipment metadata from being omitted, packed into compound strings, or lossy-converted into `f64` merely because the original Observatory measurement type is continuous-numeric.

## Compatibility strategy

`HumanoidCapabilityObservationV1` remains unchanged.

`HumanoidCapabilityObservationV2` wraps an already-valid V1 observation and adds a bounded exact-metadata vector. This avoids breaking existing V1 struct literals and preserves the semantics of continuous measurements while introducing lossless exact values incrementally.

```text
V1 continuous observation
        +
bounded exact metadata
        ↓
HumanoidCapabilityObservationV2
```

A V1 observation can be admitted into V2 with an empty exact-metadata set. No V1 measurement is reinterpreted during migration.

## Exact metadata values

`CapabilityExactMetadataValueV1` supports:

- signed `i64`;
- unsigned `u64`;
- boolean;
- bounded semantic token;
- bounded opaque artifact/configuration/calibration reference.

Each field also carries:

- a unique field ID;
- evidence provenance;
- optional bounded unit;
- optional bounded schema/type identity.

There are no arbitrary metadata maps and no unbounded free-form payloads.

## Continuous values remain continuous

Floating-point observations continue to use `CapabilityMeasurementV1` and retain finite-value validation.

Exact metadata is not a second generic measurement system. It exists for evidence that must not be coerced into floating point or hidden inside unrelated IDs.

## Canonical commitment

V2 computes a domain-separated BLAKE3 commitment over the exact metadata set.

Metadata field IDs must be unique. Commitment construction sorts by field ID before hashing, so insertion/serialization order does not change evidence identity.

The canonical commitment binds:

```text
schema identity
+ field ID
+ provenance
+ optional unit
+ optional schema/type ID
+ exact value type
+ exact value bytes
```

## HumanoidBench migration

`HumanoidBenchEpisodeResultV1::to_observation_v2` preserves the existing HUM-OBS-003A V1 lowering and adds exact metadata for:

- task ID;
- robot ID;
- control mode;
- execution status;
- exact upstream commit;
- runner artifact reference;
- explicit seed state;
- exact `u64` random seed when present;
- exact episode length;
- terminated/truncated booleans.

An unspecified seed is represented as an explicit `seed_state = unspecified` token. It is not rewritten as seed zero.

This makes the distinction explicit:

```text
seed = 0
!=
seed unspecified
```

and:

```text
true / false
!=
1.0 / 0.0 as the authoritative exact representation
```

The legacy V1 HumanoidBench lowering is retained for compatibility.

## Tests

Source tests cover:

- `u64::MAX` serde round-trip without floating-point coercion;
- typed boolean preservation;
- canonical commitment independence from insertion order;
- duplicate field rejection;
- bounded-reference rejection;
- exact V1-to-V2 migration;
- exact HumanoidBench seed and episode-length retention;
- explicit unspecified-seed state;
- exact runner artifact reference retention;
- reuse of HUM-OBS-003A source validation.

## Nonclaims

Typed evidence improves evidence fidelity. It does not improve benchmark performance, establish physical capability, prove sim-to-real transfer, create safety qualification, or grant deployment authority.
