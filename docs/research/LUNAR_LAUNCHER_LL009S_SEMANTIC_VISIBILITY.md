# LL-009S — Semantic K-horizon visibility gate

## Purpose

LL-009S closes a downstream semantic leak in the lunar terrain evidence chain.

LL-009K already computes the authoritative bin-wise maximum terrain obstruction across admitted multi-resolution terrain. The older LL-009J V1 visibility lane predates K and can rebuild a horizon from terrain samples. Separately, LL-009O/Q/R now distinguish whether the numbers support deterministic, statistical/ensemble, sampled, or merely descriptive claims.

S keeps those concerns together without rewriting J underneath review:

1. **K is the sole numerical terrain-horizon authority.**
2. **O/R determine whether that exact geometry may be promoted as a deterministic visibility bound.**
3. If stronger semantics are not established, S still computes the geometry but labels the metrics `descriptive_geometry_only`.
4. Statistical/ensemble promotion remains blocked until the exact K pack states which executed statistical horizon it numerically materialized.

## Direct K-bin obstruction

K emits complete azimuth bins and the maximum admitted conservative terrain elevation in each bin.

For target azimuth `a`, S uses:

```text
b = floor((a + ε) / bin_width) mod bin_count
h(a) = K[b].conservative_elevation_deg + LOS_margin
```

where `ε` matches K's deterministic bin-boundary convention.

S uses the K bin as a **piecewise-constant maximum obstruction** across that bin. It does not linearly interpolate between winner samples or rebuild a competing skyline from raw terrain points.

This matters because interpolation between two winner azimuths could fall below the actual per-bin maximum envelope K already proved over its admitted samples.

## Capability contract

S binds exact LL-009O and LL-009R receipts to the same study as the K pack.

### Deterministic visibility

`deterministic_visibility_bound` is allowed only if:

```text
O.deterministic_upper_bound_eligible == true
AND
R.strongest_common_spatial_support == continuous_hard_bound
```

Both axes are required. Hard vertical-error semantics do not compensate for unresolved terrain support, and continuous spatial support does not compensate for statistical/RMS height uncertainty.

### Statistical / empirical visibility

K V1 currently has no field proving that a selected LL-009Q ensemble statistic, confidence expansion, or other executed statistical horizon was the numeric obstruction actually materialized into the K pack.

Therefore S V1 deliberately blocks:

- `risk_qualified_visibility`;
- `empirical_sampled_visibility`;

unless the exact K pack carries a future machine-readable:

```text
statistical_horizon_binding.status = bound
```

along with the upstream O/R eligibility required by that claim.

The mere existence of a valid Q receipt is not enough. Otherwise a Q ensemble could be scientifically valid while K still contains a different RMS-derived numeric envelope, and S would be falsely attributing ensemble semantics to the wrong numbers.

## Descriptive fallback

If a requested strong claim is blocked, S does not discard useful geometry.

It still calculates:

- Sun center visibility;
- Sun full-disc visibility;
- Earth LOS visibility;
- relay LOS visibility;
- low/central/high sampled-time visibility fractions;
- bracketed longest outage/occlusion intervals.

But the output status becomes:

```text
claim_blocked_geometry_available
```

and every metric receives:

```text
evidence_class = descriptive_geometry_only
```

This makes it possible to inspect real lunar geography early without allowing downstream site-viability logic to mistake exploratory geometry for qualified availability.

## K basis reuse

S uses K's recorded local north/east/up basis directly and validates that it is finite, unit length, and orthogonal.

This avoids recomputing a slightly different local basis downstream and gives the target ephemeris calculation exactly the same site-local orientation lineage as the K horizon.

## Target evidence

Target direction histories remain exact hash-verified artifacts. Each target must:

- identify Sun, Earth, or relay kind;
- use the exact K frame;
- bind one or more exact source IDs;
- contain strictly increasing epochs;
- obey the declared maximum temporal gap;
- declare apparent angular radius for the Sun.

As in J, S brackets visibility transitions between discrete time samples rather than inventing sub-sample crossing times.

## Sun-disc semantics

For target elevation `e`, K-bin horizon `h`, LOS margin `m`, and apparent solar radius `r`:

```text
center_visible = e > h + m
full_disc_visible = e - r > h + m
```

Center and full-disc availability remain distinct metrics.

## Metric evidence classes

S V1 can emit:

- `derived_deterministic_bound` — only when O and R jointly establish deterministic capability;
- `derived_risk_qualified` — reserved for a future exact K statistical-horizon binding plus O/R qualification;
- `derived_empirical_sampled` — reserved for a future exact K empirical-horizon binding plus O/R qualification;
- `descriptive_geometry_only` — the safe fallback when stronger claims are blocked.

This replaces the ambiguous practice of calling every numerically verified visibility metric simply `derived_verified`.

## Local executed logic evidence

The local S synthetic campaign passes and verifies:

- exact K bin validation and complete 360° coverage;
- orthonormal K basis validation;
- deterministic piecewise-constant bin lookup;
- exact 0°, 90°, 359.999°, and 360° boundary/wrap behavior;
- Sun center vs full-disc distinction;
- same numerical target geometry under different semantic capability receipts;
- deterministic claim blocked when O is non-hard or R is sample-points-only;
- deterministic claim passes when O is hard-bound eligible and R is continuous-hard;
- blocked metrics automatically downgrade to `descriptive_geometry_only`;
- risk-qualified claim remains blocked when K lacks `statistical_horizon_binding`;
- exact target source hashing;
- temporal transition bracketing.

The key invariant is that changing only O/R capability changes the **allowed claim and metric evidence class**, not the underlying K-based geometry.

## Current expected Site01 state

With the current evidence line:

```text
O: Site01 RMS + far-field unknown semantics
R: near sample_points_only + far resolution_qualified
K: no statistical_horizon_binding
```

S should still be able to show the exact computed Sun/Earth/relay geometry, but the full terrain-backed visibility claim should remain descriptive rather than deterministic or risk-qualified.

That is intentional. We want the Moon's actual geography to influence the architecture now without allowing the evidence vocabulary to outrun what the data proves.

## Next statistical integration

The next promotion step should not be a label change. It should be a new exact numeric K lineage that explicitly binds the statistical horizon it uses, for example:

```text
K-statistical pack
  input = selected LL-009Q per-bin empirical quantile / observed ensemble envelope
  + independently qualified far-field uncertainty treatment
  + LL-009R spatial-support policy/margins
  ↓
statistical_horizon_binding {
  method,
  quantile_or_envelope,
  source_receipt_hashes,
  covered_layer_ids,
  spatial_support_receipt_hash,
  claim_nonclaims
}
```

Only then should S unlock `empirical_sampled_visibility` or a genuinely risk-qualified lane.

## Evidence chain

```text
LL-009N/P/L/M
     ↓
LL-009O/Q  vertical meaning
     ↓
LL-009R    spatial support
     ↓
LL-009K    numeric horizon authority
     ↓
LL-009S    semantic K-bin visibility
     ↓
LL-009I    site viability with evidence class intact
```

## Non-claims

LL-009S does not create terrain certainty that is absent upstream.

It does not turn RMS into a hard bound, a finite clone ensemble into a tail theorem, effective resolution into continuous support, LOS into an RF link budget, solar visibility into delivered electrical power, or descriptive terrain geometry into site qualification.

Its role is to ensure those distinctions survive all the way into visibility metrics.
