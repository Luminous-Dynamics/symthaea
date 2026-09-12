# symthaea-perception-zero-event-assurance

Statistically honest assurance for perception campaigns that observe zero false events.

This crate sits above `symthaea-perception-crucible` and `symthaea-assurance-statistics`.
It never converts `0 observed` into `0 true probability`.

A campaign must provide:

- a **passing** perception crucible report
- which zero-event metric is being claimed (false-positive observations, false tracks, or identity switches)
- an explicit exposure model
- a durable evidence reference justifying that exposure model
- a reviewed confidence level and target upper bound/rate

## Exposure models

Two models are supported:

1. **Independent Bernoulli exposures** — caller supplies a justified number of independent exposure units. Raw video frames are not automatically independent.
2. **Poisson exposure** — caller supplies positive exposure (for example independently justified operating hours) and a named unit.

The crate does not infer independence from frame count or test duration.

## Outcomes

- `Supported`: zero events were observed and the exact upper confidence bound meets the reviewed target.
- `NotSupported`: zero events were observed but the campaign is not large enough to meet the target.
- `ObservedEvents`: the selected metric had one or more observed events, so zero-event reasoning is not applicable.
- `Blocked`: the upstream crucible did not pass.
- `Invalid`: malformed policy/campaign or incompatible exposure/requirement model.

Every report grants no physical authority.

## Verification

```bash
cargo test -p symthaea-perception-zero-event-assurance
```
