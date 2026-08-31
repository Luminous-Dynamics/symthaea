# Sentinel custody and split integration contract

This document defines the intended boundary between the offline-first `symthaea-sentinel-eo` Earth-observation bridge and the separate research-integrity layers used by future Wetland Watch / Planetary Perception experiments.

The package suffix `-eo` is deliberate: the workspace already has an unrelated core `symthaea-sentinel` audio-pattern-recognition crate. The Earth-observation bridge therefore has a distinct Cargo identity.

It is deliberately a contract, not a claim that real Sentinel fixture custody has already been implemented.

## Responsibilities

### `symthaea-sentinel-eo`

Owns provider-facing Sentinel Earth-observation semantics:

- Sentinel mission and product kind;
- stable product id;
- acquisition time;
- provider-neutral footprint;
- sensor modality/bands;
- source content digest;
- processing lineage;
- deterministic frozen catalogue replay.

It does **not** decide which products are training, calibration, or evaluation data and does not grant a model access to held-out payloads.

### `symthaea-research-split`

Owns content-addressed research assignment/separation semantics:

- Training / Calibration / Evaluation role;
- configured group dimensions such as `spatial-block`, `acquisition`, `orbit-swath`, `watershed`, `season`, or `event`;
- evaluation-disjoint or all-role-disjoint policies;
- forward-time embargo;
- attributable separation evidence.

The split contract proves declared assignment/separation facts. It does not prove statistical independence or scientific adequacy of chosen blocks/buffers.

### External fixture custody / verification layer

A later custody layer must own actual access control over held-out evaluation payloads and labels/verification outcomes.

A digest is an integrity commitment, not secrecy. Do not expose small-state held-out labels to untrusted candidate code merely because their hashes are frozen.

## Canonical Sentinel -> split mapping

After `symthaea-research-split` qualifies, a Sentinel observation should be mapped into a `SplitUnit` using provider-neutral, immutable facts.

Recommended fields:

```text
SplitUnit.sample_id
    = Sentinel product id or a stable derived-observation id

SplitUnit.observed_at_unix_ms
    = acquisition time

SplitUnit.content_digest
    = digest of the exact immutable source/derived product used by analysis
```

Recommended group dimensions are experiment-specific, but a serious remote-sensing evaluation should consider at least:

```text
spatial-block
acquisition
```

and often also:

```text
watershed/site
orbit-swath
season/hydrological-regime
event-id
sensor/processing-campaign
```

Group values must be produced by a frozen methodology. `spatial-block = A` is an identifier, not proof that block A is sufficiently separated from block B.

## Raw product vs derived product identity

Do not reuse the raw Sentinel product digest for a derived feature cube, cloud-masked raster, terrain-corrected SAR product, resampled ROI, or learned embedding.

Every materially transformed artifact should receive its own content digest and processing lineage while retaining references to its source products.

This prevents a result from claiming it used one frozen input while actually depending on a mutable downstream preprocessing result.

## Acquisition leakage

Random pixel or patch splits are prohibited by default for claims of geographic or temporal generalization.

Two samples may be in different spatial blocks yet derive from the same Sentinel acquisition. If `acquisition` is a configured separation dimension, the split must reject that reuse.

Likewise, different product ids are not automatically independent if they share an orbit, event, weather regime, preprocessing artifact, or near-adjacent time window. Those leakage mechanisms should be represented by additional groups or separation evidence where they matter to the preregistered claim.

## Forward-time evaluation

For forecasting claims, evaluation should normally be later than all development data under a preregistered embargo appropriate to the process.

The embargo is an enforced clock separation, not proof that temporal autocorrelation has decayed. Evidence supporting the chosen embargo should remain separately attributable.

## Held-out custody

A production empirical campaign should separate at least these concerns:

```text
catalogue / metadata preparation
        |
        v
split assignment + preregistration
        |
        v
fixture custody
        |
        +--> development payloads -> model development
        |
        +--> evaluation payloads/labels remain sealed
                              |
                              v
                      forecast receipt
                              |
                              v
                    independent verification
```

Candidate model processes should receive only the data authorized for their phase. Evaluation outcome/label material should not be present in an ambient filesystem, process environment, serialized manifest, debug representation, or model-accessible API before forecast commitment.

## Commit/reveal boundary

The Living Watershed prequential witness demonstrates a code-level ordering:

```text
forecast -> commit -> reveal -> verify -> score
```

Real Sentinel experiments should preserve that ordering while upgrading custody from an in-process synthetic seal to an external/process-separated mechanism.

Possible later implementations include:

- separate verifier process with inaccessible evaluation labels;
- encrypted fixture store with scoped decryption authority;
- Xenia capability grants for phase-specific access;
- signed forecast receipts before outcome release;
- blinded fixture identifiers with hidden nonces where commitments otherwise expose tiny state spaces.

These are future implementation options, not claims of the current Sentinel bridge.

## Frozen real-data witness gate

Before a real Wetland Watch predictive-skill claim is allowed, require an evidence package binding:

1. exact Sentinel product/source digests;
2. exact preprocessing artifact digests;
3. frozen `ResearchSplitManifest`;
4. frozen research protocol and analysis plan;
5. custody/verification procedure digest;
6. model source/artifact digest;
7. forecast ledger issued before held-out verification;
8. proper scoring + coverage/abstention metrics;
9. all preregistered primary metrics, including null or not-computed outcomes;
10. direct replication on new product/region/time lineage before broad generalization.

## Non-claims

This integration contract does not establish that:

- current Sentinel code downloads or preprocesses real raster products;
- a chosen split is scientifically adequate;
- held-out evaluation bytes are presently cryptographically secret;
- Sentinel-1 directly images arbitrary underground structure;
- Sentinel-1/2 predicts wetland state accurately;
- Symthaea/HDC improves predictive performance;
- a forecast authorizes physical or governance action.

Those remain later, separately gated claims.
