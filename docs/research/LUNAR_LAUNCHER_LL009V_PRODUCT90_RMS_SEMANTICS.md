# LL-009V — Product 90 RMS terrain-height semantics

## Purpose

LL-009V is a narrow semantic refinement for NASA PGDA Product 90's large-area `LDEM_*_ADJ_ERR.TIF` products.

LL-009O intentionally left the far-field error semantics as `unknown` because the PGDA file listing alone says only “surface height error.” Subsequent source-method review supports a stronger—but still statistical—classification: **RMS terrain-height uncertainty**.

V adds that refinement without rewriting O underneath review.

## Methodology basis

Product 90 publishes the `ADJ_ERR` rasters as LDEM surface-height error maps in meters and cites Barker et al. 2023 as the product methodology.

Barker et al. 2023 describes improved large-area south-polar LDEMs, at multiple pixel scales, with associated uncertainty estimates.

The follow-on Barker et al. polar-method work states that the corresponding gridded LDEM height/slope uncertainty maps represent **RMS uncertainty relative to the true average height/slope inside each pixel**, using uncertainty-versus-effective-resolution relations derived by Barker et al. 2023. It also reports the South-Pole uncertainty behavior from that methodology.

Therefore V permits:

```text
ADJ_ERR -> rms_error
```

but never:

```text
ADJ_ERR -> hard_upper_bound
```

## Exact byte binding

The V qualifier binds:

- the exact LL-009N source-lock entry and SHA-256 for the configured Product 90 `ADJ_ERR` artifact;
- the exact LL-009L companion-layer uncertainty path/SHA-256;
- optionally the exact passing LL-009M receipt and its same uncertainty SHA-256.

If any of those byte identities differ, V fails closed.

## Quantity semantics

The output records:

```text
semantics_class = rms_error
quantity = gridded_surface_height_error_m_relative_true_average_pixel_height
deterministic_upper_bound_eligible = false
statistical_closure_required = true
```

The scalar height-error map is statistical source evidence. V does not convert it into a displacement vector or choose a multiplier.

## Explicitly blocked interpretations

A passing V receipt does **not** authorize:

- `1×RMS` as a maximum error;
- Gaussian `1σ`, `2σ`, or `3σ` tails;
- independent pixel errors;
- a hard terrain upper bound;
- a complete physical horizon.

Those require a separate, hashed statistical theorem/model.

## Local executed logic evidence

The exact committed V script passed local Python compilation and its self-test. The synthetic campaign verifies:

1. exact N→L→M uncertainty-byte binding;
2. deterministic replay;
3. rejection of source-hash drift;
4. rejection of any policy attempting to change `rms_error` to `hard_upper_bound`;
5. `deterministic_upper_bound_eligible=false` in a passing receipt.

The committed GitHub blob SHA matches the locally executed script's Git object hash.

## Consequence for LL-009U

V improves the far-field classification from `unknown` to `rms_error`, but LL-009U V1 deliberately requires a terrain-radial hard bound. Therefore current Site01 U remains blocked.

The next appropriate closure is a separately versioned far-field statistical theorem. A useful baseline is a distribution-free second-moment/familywise risk envelope; a scientifically tighter lane may use source-supported covariance/fractal structure or an explicit far-field ensemble.

## Non-claims

LL-009V does not choose a tail distribution, covariance model, confidence level, familywise risk budget, spatial-support bound, horizon statistic, or site/architecture decision.
