# symthaea-epidemiology

Lightweight epidemiology primitives for Symthaea.

The crate now keeps two capabilities deliberately separate:

1. infectious-disease dynamics through the existing SIR model;
2. conservative statistical screening of already-aggregated surveillance time series.

The surveillance screen produces **change candidates**, not outbreak declarations, diagnoses, forecasts, or response instructions.

## SIR dynamics

`Sir { beta, gamma }` provides:

- `basic_reproduction_number` (`R₀ = β/γ`);
- `herd_immunity_threshold` (`1 − 1/R₀`);
- `final_size`;
- population-conserving Euler simulation with peak tracking.

## Aggregate surveillance screen

`surveillance::assess_latest_change` compares one latest aggregate measurement with an ordered historical baseline.

The v1 algorithm is deliberately auditable:

1. count explicit missing baseline observations;
2. require a configured minimum number of observed history points;
3. use the median historical point estimate as the baseline center;
4. compute median absolute deviation (MAD);
5. convert MAD to a robust scale with the conventional `1.4826` consistency factor;
6. compute a caller-configured robust standardized deviation for the latest point estimate;
7. require the latest source-supplied uncertainty interval to be clearly separated from the baseline threshold envelope before returning `ChangeCandidate`;
8. otherwise return `WithinBaseline`, `InsufficientBaseline`, or an explicit abstention reason.

The default screen configuration is only a convenience profile. Its thresholds are **not universal epidemiological or clinical cutoffs**. A real source/deployment should preregister and validate screening parameters against its own historical process and false-alert tolerance.

### Explicit abstention

The screen abstains rather than inventing certainty when:

- the latest aggregate is missing;
- historical robust spread is effectively zero but the latest interval no longer contains the baseline center;
- the point estimate crosses the configured robust threshold while its uncertainty still overlaps the threshold envelope.

A zero-MAD baseline never turns into an infinite anomaly score.

### Missing data

Missingness is represented explicitly with `SurveillancePoint::missing(timestamp)`, not NaN or sentinel numeric values. Missing rows do not count toward the baseline observation requirement.

### Time ordering

Historical timestamps must be strictly increasing, and the latest point must come after the baseline series. This prevents accidental reordering/duplicate-period analysis from silently producing a candidate.

## What this screen does not establish

A `ChangeCandidate` establishes only that this univariate statistical screen found a sufficiently separated excursion under the configured rule.

It does **not** establish:

- an outbreak;
- disease or pathogen identity;
- transmission rate or `R₀`;
- persistence of a signal;
- source authenticity;
- source independence;
- causal explanation;
- clinical severity;
- treatment recommendations;
- public-health or emergency authority.

Those require separate evidence and reasoning layers. In particular, Mycelix surveillance provenance/lineage evidence should remain external until its wire contracts are qualified rather than being duplicated here.

## Dependency posture

The crate remains lightweight and has no `symthaea-core` dependency. Aggregate screening reuses only the pure-`std` `symthaea-statistics` domain crate for the median primitive.

## Example

```rust
use symthaea_epidemiology::{
    ScreeningDisposition, SurveillancePoint, SurveillanceScreenConfig,
    assess_latest_change,
};

let history = [
    SurveillancePoint::observed(1, 8.0, 7.9, 8.1).unwrap(),
    SurveillancePoint::observed(2, 9.0, 8.9, 9.1).unwrap(),
    SurveillancePoint::observed(3, 10.0, 9.9, 10.1).unwrap(),
    SurveillancePoint::observed(4, 11.0, 10.9, 11.1).unwrap(),
    SurveillancePoint::observed(5, 12.0, 11.9, 12.1).unwrap(),
];
let latest = SurveillancePoint::observed(6, 20.0, 19.0, 21.0).unwrap();
let config = SurveillanceScreenConfig::new(5, 3.0).unwrap();
let assessment = assess_latest_change(&history, latest, config).unwrap();
assert!(matches!(
    assessment.disposition,
    ScreeningDisposition::ChangeCandidate(_)
));
```

Run:

```bash
cargo test -p symthaea-epidemiology
```

## Later layers

Possible later, separately qualified additions include persistence/change-point methods, lineage-aware corroboration, competing hypotheses, calibration against held-out surveillance histories, and forecasting through the Futures Laboratory.

They should remain evidence/reasoning capabilities, not autonomous response authority.
