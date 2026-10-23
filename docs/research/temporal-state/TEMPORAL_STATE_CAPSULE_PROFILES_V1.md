# TEMPORAL-STATE V1 — State Capsule Profiles

Status: **contract amendment / no runtime claim**

Parent: #3604  
Root API repair: #3586  
Primary V1 contract: `TEMPORAL_STATE_API_V1.md`

## Purpose

The primary V1 contract separates projected observation, exact state restoration,
reset, perturbation, checkpoint state, and training continuation. This companion
contract makes one further distinction explicit:

> There is no single universal "exact temporal snapshot".

Prediction, post-learning restoration, and training/adaptation continuation need
state capsules with different persistence and identity semantics.

```text
CheckpointBoundInferenceSnapshot
!= LocalEvolutionBackupAcrossLearning
!= TrainingContinuationSnapshot
```

They may contain overlapping bytes. They do not grant the same capability.

---

## 1. Checkpoint-bound inference snapshot

### Purpose

Reproduce future inference from the same dynamic state under the **same exact
configuration and checkpoint**.

Typical uses:

- pure multi-horizon prediction;
- paired replay where checkpoint parameters are frozen;
- deterministic counterfactual probes;
- effectome same-start interventions that do not update parameters.

### Identity rule

Restore MUST fail if the target no longer has the checkpoint/config identity the
snapshot was captured against.

For classic CfC, a V1 guard may be derived without adding dependencies:

```text
BLAKE3(
    domain_separator
    || exact serialized CfCNetworkConfig
    || ordered exact weight/tau bits
)
```

The repository already carries BLAKE3, serde/bincode, and a deterministic
`CfCNetwork::get_weights()` ordering. A final implementation must freeze the exact
encoding rather than relying on this prose alone.

### Evolution payload

For classic CfC frozen-checkpoint inference, the current source audit indicates the
semantic evolution payload can be limited to the ordered hidden state of every
cell/layer, provided executable conformance confirms diagnostic counters do not
feed back into future inference.

### Must not contain by default

Do not add fields merely because they are mutable.

Examples that are not automatically part of this profile:

- diagnostic step counters;
- optimizer moments;
- training-loss statistics;
- online-adaptation counters when adaptation is disabled;
- redundant copies of checkpoint parameters already bound by identity.

### Required restore law

```text
capture S at checkpoint C
advance/probe without changing C
restore S
replay input/dt stream I
==
untouched reference from S,C over I
```

---

## 2. Local evolution backup across learning

### Purpose

Preserve the live online **evolution state** across an operation that is allowed to
change learned parameters.

Primary example: consolidation replay.

The desired semantics are:

```text
live hidden state before consolidation = S
checkpoint before consolidation        = C0
training/consolidation learns           = C1
restore live evolution state            = S
continue cognition using                = S + C1
```

Restoring `C0` would erase learning and is therefore incorrect.

### Identity rule

This capsule MUST NOT require equality with the pre-training checkpoint fingerprint,
because parameter change is expected.

It should instead be bound to a local owner/backend/profile lineage and exact
shape/config compatibility sufficient to make restoration meaningful.

V1 implementations should prefer a non-portable, crate-internal capsule until a
stable subject-lineage identity exists.

### Payload

For classic CfC the candidate payload is again all ordered layer hidden states,
with exact layer-count and dimension metadata.

Any additional field must be justified by a demonstrated effect on post-training
online evolution.

### Required restore law

```text
capture evolution S under C0
train -> C1
restore S without reverting C1
future(I) == reference using S and C1
```

A whole-network clone restore FAILS this profile if it rolls learned parameters
back to C0.

---

## 3. Training/adaptation continuation snapshot

### Purpose

Resume not just inference dynamics but the exact future trajectory of an adaptive
or training process.

This is strictly stronger than inference restoration.

Potential state includes, depending on the qualified training profile:

- all layer hidden states;
- mutable checkpoint parameters;
- optimizer moments and optimizer step;
- online-adaptation statistics that affect future thresholds/updates;
- mutable tau values;
- training-history state;
- profile-specific scheduler state.

### Identity rule

The profile must bind the exact training algorithm and numerical execution profile.
A snapshot qualified for frozen inference is not automatically a valid training
continuation snapshot.

### Required continuation law

```text
capture T
run training stream R
restore T
rerun R
==
identical parameter/evolution/optimizer trajectory
```

within the explicitly qualified numerical determinism envelope.

---

## 4. Projected observations are none of the above

A projected observation is a measurement view.

For current classic CfC:

```text
read_state() -> final cell state only
```

For current HdcLtc bridge:

```text
read_state() -> projected/cached bridge output
```

For current HierarchicalCfC wrapper:

```text
read_state() -> selected slow-layer projection
```

These values may be useful telemetry. They are not restore capsules unless a backend
profile separately proves that theorem.

Semantic similarity to an internal state does not grant restoration authority.

---

## 5. Atomic restore is mandatory

Existing classic-CfC compatibility setters are intentionally low-level:

- network `set_state()` zips supplied states with cells;
- cell `set_state()` directly replaces the array.

Therefore exact restore MUST NOT blindly delegate to them before validation.

The V1 preflight sequence is:

1. validate capsule schema and profile version;
2. validate backend/profile identity;
3. validate subject/config/checkpoint or local-owner guard required by the profile;
4. require exact layer-count equality;
5. validate every layer state dimension;
6. enforce finite-state requirements for the profile;
7. perform **zero mutations** if any validation fails;
8. commit all state only after the full preflight succeeds.

Failure returns a typed error. Backend/profile mismatch is never a silent no-op.

Required negative fixtures:

- too few layers;
- too many layers;
- wrong first/middle/last layer dimension;
- wrong backend variant;
- wrong profile version;
- wrong checkpoint guard for checkpoint-bound inference;
- non-finite state where forbidden;
- failed restore followed by replay equality against an untouched reference.

---

## 6. Backend mapping

### Classic CfC

Current status:

- projected observation: available, lossy for multilayer restore;
- detached-clone observational prediction: proposed/implemented in TEMPORAL-STATE-003;
- checkpoint-bound exact inference snapshot: planned;
- local evolution backup across learning: planned;
- exact training continuation: unqualified.

### HdcLtc

Current status:

- engine-level `NetworkStateSnapshot`: strong existing primitive;
- internally pure prediction: existing separate capability;
- wrapper-level exact lifecycle: incomplete until bridge-visible state such as
  `current_output` is included where required;
- generic projected-state injection is not an exact restore.

### HierarchicalCfC

Current status:

- projected observation: available;
- exact inference snapshot: unavailable;
- deterministic realized construction: prerequisite #3577;
- exact lifecycle: prerequisite #3578;
- truthful realized-tau observability/purity: prerequisite #3579;
- tau-algorithm choice remains separate experiment #3581.

No HCFC capsule profile is promoted by this document.

---

## 7. Resource accounting

State correctness is not free.

Receipts record separately:

- snapshot bytes retained;
- bytes copied during capture;
- bytes copied during restore;
- checkpoint/config fingerprint work;
- full-network clone bytes/work for containment paths;
- per-horizon clone count;
- restore validation work;
- wall-clock capture/restore latency.

Do not fold snapshot/clone overhead invisibly into ordinary inference cost.

---

## 8. Capability admission

A backend/profile may advertise only capabilities it has executed successfully under
the V1 conformance suite.

Examples:

```text
ClassicCfC / frozen-checkpoint inference
    ObservationalPrediction      Qualified|...
    ExactInferenceSnapshot       Qualified|...
    LocalEvolutionBackup         Qualified|...
    TrainingContinuation         Qualified|...

HierarchicalCfC / current historical profile
    ObservationalPrediction      NotQualified
    ExactInferenceSnapshot       NotQualified
```

A broader backend name is not evidence.

---

## 9. Migration rule

Containment and optimization are separate stages.

For classic CfC:

```text
freeze regression
    -> qualify detached-clone observational prediction
    -> route prediction through containment
    -> implement checkpoint-bound exact snapshot
    -> qualify same-start replay
    -> replace clone path only if the replacement preserves semantics
```

For consolidation:

```text
freeze live-state corruption regression
    -> implement local evolution backup
    -> prove learned parameters persist across restore
    -> only then re-enable/qualify destructive replay
```

Do not use the prediction snapshot as the consolidation backup merely because both
contain hidden states.

---

## 10. Evidence boundary

This document defines semantics only.

It does not establish:

- runtime correctness;
- exact snapshot implementation;
- performance acceptability;
- backend superiority;
- consciousness significance;
- production eligibility.

Those require executable profile-bound evidence.