# SPINE-000B — Integration / Lowering Qualification Domain v1

**Status:** preregistered measurement-domain contract

**Authority:** measurement-only

**Issue:** #3331

N1 classifies emitted `SubsystemOutput` values. N2 classifies the next boundary: the exact `IntegratedBits` produced by proposal integration and the scalar operands Phase C deterministically lowers from that integrated object.

The core distinction is:

```text
qualified emitted proposals
!=
qualified integrated result
!=
qualified lowered application operand
```

A cycle is not eligible for qualified R2 application-relevance analysis unless both `I_all` and every `I_withoutS` used by that analysis pass N2.

## 1. Why this is separate from N1

Finite individual proposals do not guarantee a finite integrated object. Current integration can overflow during f64 or f32 accumulation. In addition, current Phase C casts integrated confidence, LR and exploration from f64 to f32 before application, so a finite integrated f64 may become infinity at the actual application boundary.

N2 records these as measurement-domain failures. It does not silently repair production behavior.

## 2. Bound inputs

N2 consumes the canonical `IntegratedBits` shape frozen by C1/P1R:

```text
confidence_delta_bits  u64 / f64
lr_modulation_bits     u64 / f64
exploration_delta_bits u64 / f64
arousal_delta_bits     u32 / f32
valence_delta_bits     u32 / f32
flags                  u32
n_contributors         u32
```

It is bound to:

- `SPINE_000B_PHASE_C_APPLICATION_REGISTRY_V1.json` for scalar source/lowering rules and known flags;
- I1's frozen Stage-A capacity (`CAPACITY = 64`) as the v1 maximum qualified contributor domain.

Registry or capacity drift requires a new N2 lineage.

## 3. Native integrated-value checks

All five integrated scalar values must be finite in their native representation.

A native NaN or infinity is `INVALID_INTEGRATED_NONFINITE`, regardless of how production would later propagate it.

The integrated flag mask must contain only bits present in the frozen Phase-C registry.

`n_contributors` must be in `0..=64` for this evidence version.

## 4. Empty collector identity

For `n_contributors == 0`, the exact integrated object must be the collector identity:

```text
confidence = +0.0 f64
LR         =  1.0 f64
explore    = +0.0 f64
arousal    = +0.0 f32
valence    = +0.0 f32
flags      = 0
```

This rule is bit-exact. An empty collector carrying `-0.0` is numerically identity but is not the exact production empty-collector result and is therefore `INVALID_EMPTY_COLLECTOR_IDENTITY`.

For nonempty integration, signed zero follows current numeric trigger semantics and remains byte-distinct in the receipt layer.

## 5. Frozen scalar trigger and lowering rules

The v1 registry must continue to contain exactly these scalar mappings:

```text
CONFIDENCE_DELTA  integrated.confidence_delta as f32   identity 0.0
LR_MODULATION     integrated.lr_modulation as f32      identity 1.0
EXPLORATION_DELTA integrated.exploration_delta as f32  identity 0.0
AROUSAL_DELTA     integrated.arousal_delta             identity 0.0
VALENCE_DELTA     integrated.valence_delta             identity 0.0
```

All use `SCALAR_NON_IDENTITY`.

Trigger semantics reproduce production numeric comparison exactly. Therefore `-0.0 != 0.0` is false.

For the three f64 channels, N2 independently reproduces Rust's f64→f32 conversion and records the exact lowered f32 bits. If a triggered finite f64 lowers to non-finite f32, the object is `INVALID_LOWERED_NONFINITE`.

N2 does not claim that the mapped operation actually executed. It only qualifies the deterministic lowering domain.

## 6. Primary classification

All diagnostics are computed. Primary precedence is:

```text
1 INVALID_INTEGRATED_NONFINITE
2 INVALID_LOWERED_NONFINITE
3 INVALID_CONTRIBUTOR_DOMAIN
4 UNSUPPORTED_INTEGRATED_FLAGS
5 INVALID_EMPTY_COLLECTOR_IDENTITY
6 QUALIFIED_IDENTITY
7 QUALIFIED_APPLICATION_FINITE
```

`QUALIFIED_IDENTITY` means no scalar source is numerically non-identity and no flag is set. It can occur with nonzero contributors when proposals cancel or redundantly average to identity.

`QUALIFIED_APPLICATION_FINITE` means at least one scalar/flag source is non-identity and all deterministic scalar lowering remains inside the finite v1 evidence domain.

## 7. Required witness surface

A future runtime/deferred witness must preserve at least:

```text
primary_class
integrated_nonfinite_field_mask
lowered_nonfinite_field_mask
triggered_scalar_mask
lowered_f32_bits for all five scalar sources
unsupported_integrated_flag_bits
contributor_count
max_qualified_contributors
empty_collector_identity_mismatch
numeric_identity
```

Pre-lowering bits remain supplied by the canonical integration receipt.

## 8. Required controls

The executable N2 oracle must cover ordinary finite identity/nonidentity values, signed zero, exact empty identity, native f64/f32 non-finite values, a finite f64 that overflows to f32 infinity, synthetic f64 and f32 integration overflow, contributor overflow, unknown integrated flags, two distinct f64 values that collapse to identical f32 operand bits, and a larger f64 delta that survives lowering.

## 9. R2 dependency

R2 must compare deterministic Phase-C **lowering projections**, not fabricate counterfactual runtime application receipts.

For `I_all`, later runtime application receipts may confirm which operations actually executed. For `I_withoutS`, before SPINE-000D intervention there is no observed counterfactual execution; only source-condition/lowering projection is permitted.

Correct wording is therefore:

```text
leave-one-out changes Phase-C application reachability/lowered operand
```

not:

```text
the counterfactual application executed
```

## 10. Qualification boundary

A PASS establishes only that `IntegratedBits` and deterministic scalar lowering are inside the frozen v1 measurement domain and that the independent classifier matches the frozen registry/capacity contract on synthetic controls.

It establishes no actual operation execution, destination state change, subsystem causality, benefit, safety, epistemic authority or action authority.