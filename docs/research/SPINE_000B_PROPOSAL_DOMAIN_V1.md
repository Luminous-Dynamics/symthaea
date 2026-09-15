# SPINE-000B — Proposal Qualification Domain v1

**Status:** preregistered measurement-domain contract

**Authority:** measurement-only

**Issue:** #3295

This contract defines when an emitted `SubsystemOutput` is eligible for **qualified SPINE evidence**. It deliberately does not redefine current `OutputCollector` behavior.

## 1. Production behavior and evidence qualification are different

Current production admission is simple: `OutputCollector::record()` filters outputs for which `SubsystemOutput::is_neutral()` is true and records other outputs. The integration path applies the LR floor with `lr_modulation.max(0.01)` before `ln()`.

N1 does not silently change either behavior.

Instead:

```text
production accepted a proposal
!=
proposal is inside the qualified SPINE evidence domain
```

A cycle containing an out-of-domain emitted proposal may still execute in production, but must not be labeled a qualified SPINE influence/causal result.

## 2. Registry-bound flag domain

The known flag mask is derived from the exact `flag_sources[].value` entries in:

`docs/research/SPINE_000B_PHASE_C_APPLICATION_REGISTRY_V1.json`

The values must be unique powers of two. Any bit outside that frozen registry mask is `UNSUPPORTED_FLAG_BITS` for this evidence version.

This does not mean unknown future bits are inherently unsafe. It means v1 does not know how to interpret their Phase-C application semantics.

## 3. Frozen qualification checks

For the emitted proposal, evaluate all diagnostics without coercion:

- every floating channel must be finite;
- `lr_modulation > 0.0`;
- `_reserved == 0`;
- `flags & !known_registry_mask == 0`.

No NaN canonicalization, decimal tolerance, silent flag masking, or reserved-field normalization is permitted.

### LR floor

For a finite positive LR factor:

```text
0.0 < lr_modulation < 0.01
```

production currently integrates the factor as `0.01` through `max(0.01)`.

Such a proposal is classified `QUALIFIED_WITH_LR_CLAMP`, not ordinary `QUALIFIED_NON_NEUTRAL`. Evidence must retain the exact emitted LR bits as well as the resulting integrated bits.

Exactly `0.01` is not classified as clamped.

## 4. Signed zero

The canonical receipt layer preserves IEEE-754 payload bits, so `+0.0` and `-0.0` are byte-distinct.

Production neutrality uses numeric equality. Therefore negative zero in additive channels remains production-neutral when every other field is neutral. N1 preserves both facts:

- `production_neutral=true` under current equality semantics;
- exact signed-zero bits remain visible in the proposal commitment.

For LR, `+0.0` and `-0.0` are both nonpositive and therefore `INVALID_LR_NONPOSITIVE`.

## 5. Deterministic primary classification

All diagnostics are computed. The primary class uses this fixed precedence:

```text
1 INVALID_NONFINITE
2 INVALID_LR_NONPOSITIVE
3 UNSUPPORTED_FLAG_BITS
4 RESERVED_NONZERO
5 QUALIFIED_WITH_LR_CLAMP
6 QUALIFIED_NEUTRAL
7 QUALIFIED_NON_NEUTRAL
```

A record may contain multiple diagnostic flags even though only one primary class is selected.

`QUALIFIED_NEUTRAL` means the exact emitted proposal is inside the qualification domain and `SubsystemOutput::is_neutral()` would be true under current numeric equality semantics.

## 6. Required diagnostic surface

A future runtime proposal-domain witness should bind at least:

```text
primary_class
nonfinite_field_mask
lr_nonpositive
lr_clamped_by_integration
unsupported_flag_bits
reserved_nonzero
production_neutral
negative_zero_field_mask
known_registry_flag_mask
```

The proposal's exact canonical bits are already committed by the execution receipt and must not be duplicated as decimal values.

## 7. Required controls

N1 qualification must exercise:

- finite ordinary neutral and non-neutral proposals;
- qNaN and a distinct NaN payload;
- +infinity and -infinity;
- +0.0 and -0.0 LR;
- negative finite LR;
- positive LR immediately below 0.01;
- LR exactly 0.01;
- unknown flag bit outside the registry;
- nonzero `_reserved`;
- additive negative zero with otherwise-neutral state;
- mixed proposal with more than one invalid diagnostic to prove precedence.

## 8. Runtime consequence

Until a separate behavior-changing patch is qualified, N1 is observational:

- do not silently drop an out-of-domain proposal from production;
- do not silently repair it;
- record that the cycle is outside the qualified SPINE proposal domain;
- do not promote that cycle to `LOAD_BEARING` / causal evidence.

A later hardening change may reject malformed proposals at runtime, but that change must have its own regression and causal-lineage boundary.

## 9. Qualification boundary

A PASS for N1 establishes only that the proposal-domain classifier is deterministic, registry-bound, and agrees with the frozen current production neutrality/LR-floor semantics on synthetic controls.

It does not establish runtime influence, causal load, correctness of a manager proposal, or epistemic/action authority.
