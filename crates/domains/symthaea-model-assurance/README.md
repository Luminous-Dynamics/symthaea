# symthaea-model-assurance

Generic residual-based assurance for predictive models and digital twins.

The central rule is simple:

> A model is evidence only while its predictions remain compatible with observed
> reality under declared uncertainty, freshness, persistence, and provenance.

This crate generalizes the strongest ideas from the helicopter digital-twin
divergence monitor without embedding helicopter-specific signals.

## Inputs

Each `ResidualSample` binds:

- a stable sample id
- a timestamp
- an open-ended `SignalId`
- predicted and observed values
- the one-sigma uncertainty of their difference
- evidence references for prediction and observation provenance

A reviewed `ModelAssurancePolicy` declares required signals, residual thresholds,
persistence requirements, maximum sample age, and minimum samples per signal.

## Output states

```text
Aligned
Restricted
Unsafe
Incomplete
```

`Incomplete` takes precedence when the evidence case itself is defective. Missing,
stale, duplicate, future, numerically invalid, or provenance-free samples cannot
satisfy the minimum evidence requirement.

Persistent normalized residuals can then move a complete case from `Aligned` to
`Restricted` or `Unsafe`.

## Important properties

- stale samples cannot make a model look aligned
- duplicate sample IDs cannot double-count evidence
- missing provenance cannot satisfy evidence requirements
- optional, unconfigured signals cannot change the reviewed required-signal case
- the monitor never silently retunes a model
- a report is evidence, not physical authority
- canonical JSON is available for binding into stronger signed evidence systems

## Intended uses

The same primitive can monitor:

- vehicle and aircraft digital twins
- camera calibration/projection models
- atmospheric or optical propagation models
- weather and environmental models
- robotics dynamics models
- power/thermal/process models
- simulation-to-reality divergence

Domain adapters should map `ModelAssuranceStatus` into their own degraded-mode
state explicitly. A model becoming less trustworthy must never expand capability.

## Verification

```bash
cargo test -p symthaea-model-assurance
```

This crate contains no targeting, firing, interception, jamming, electronic attack,
weapon-control, or engagement-optimization logic.
