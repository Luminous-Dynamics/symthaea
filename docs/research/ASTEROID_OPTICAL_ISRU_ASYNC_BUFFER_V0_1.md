# Optical ISRU v0.2 — asynchronous in-flight buffer and synthetic-placeholder isolation v0.1

**Status:** standalone Rust reference; 14 async tests plus the 12 shared resource-kernel regression tests are included in the exact-file test binary; validation pending  
**Parent:** Symthaea #12550 — typed asynchronous in-flight buffer and synthetic-placeholder isolation  
**Related:** #12525 EIG/EVSI; #12534 propensity receipts; #12536 resource vector; #12538 stage-aware campaign planning; #12549 causal stage workflow

## Research basis

Volk, Reyes, Ethier and Baldwin's 2026 paper *Pessimistic asynchronous sampling in high-cost Bayesian optimization* evaluates realistic, pessimistic, random, descending/ascending pessimism and lower-confidence-bound placeholder policies against serial sampling in simulated high-cost optimization. It reports that pessimistic placeholders can improve performance in many of its tested scenarios, with performance depending on dimension, buffer length and exploration settings. The authors also distinguish experiment-count improvements from effective-time compression and discuss assumptions such as parallel in-flight experiments, equal experiment duration and negligible scheduling overhead. These findings motivate matched baselines, not a general claim that pessimistic placeholders always win. DOI: https://doi.org/10.1039/D5DD00477B

This implementation addresses the systems boundary that a physical scientific platform needs around those algorithms: a synthetic fantasy may influence acquisition, but it must never become an observation or satisfy a metrology, custody or accepted-product requirement.

## First implementation

`research/asteroid-optical-isru-v0.2/async_buffer.rs` defines:

- **In-flight job proposals** with candidate/replicate identity, declared exclusive resource locks, six-dimensional cost, posterior mean and posterior standard deviation.
- **Four placeholder policies:** no placeholder, realistic mean, objective-direction-aware pessimistic bound, and seeded uniform fantasy within a caller-declared uncertainty radius.
- **An optimizer-only fantasy type** (`FantasyObservation`) separate from the actual qualified measurement type (`ActualObservation`).
- **Hard resource reservation** by component-wise checking of spent + reserved + proposed cost against the frozen `CampaignBudget` from #12536.
- **Exclusive resource locks** that prohibit conflicting active jobs while allowing independent jobs to run in parallel.
- **Duplicate in-flight candidate protection**, with an explicit replicate identity for intentional concurrent replicates.
- **Out-of-order result completion**, preserving observation time separately from result receipt time.
- **Calibration and qualification receipts** required before an actual result enters the actual-observation store.
- **Failure accounting:** failed jobs release locks but their declared resource cost becomes spent, not returned to the budget.
- **Closeout refusal** while any job remains in flight.

### Placeholder policy semantics

For predicted mean \(\mu\), standard deviation \(\sigma\), and non-negative \(\beta\):

[
y_{fantasy} = \begin{cases}
\mu - \beta\sigma & \text{maximize objective}\
\mu + \beta\sigma & \text{minimize objective}
\end{cases}
]

The pessimistic value is deliberately objective-direction-aware. Seeded-uniform fantasy samples from:

[
[\mu - r\sigma,\;\mu+r\sigma]
]

where r is a caller-specified non-negative radius multiplier. The research RNG is deterministic and explicitly not cryptographic. Its seed and launch sequence are retained with the fantasy. A hidden evaluation seed must be controlled by the sealed benchmark custodian, not this helper.

## Evidence isolation invariant

The acquisition model can request `fantasies_for_model()`. That API returns only `FantasyObservation` references held by active jobs.

The physical evidence store is `actual_observations`, which accepts only `ActualObservation` values created by `resolve_job` after:
- the job was actually launched;
- the measurement time is no earlier than launch;
- measurement time is no later than result receipt time;
- the numeric result is finite; and
- non-empty calibration and qualification receipt IDs are supplied.

No method promotes a fantasy into an actual observation. When a job resolves or fails, its fantasy disappears from the active model buffer. This code checks receipt presence, not authority or cryptographic validity; Mycelix/Symtropy must verify the referenced receipts.

## Resource and timing semantics

The resource vector is shared with the campaign-resource reference and retains rig-hours, electrical/source energy under the declared meter basis, feedstock, assay cost, maintenance and consumables. Pending jobs reserve all declared resources before launch. Completion and failure both settle reserved cost into spent cost. This is a conservative reference accounting rule: real rigs may need a more detailed stage-by-stage consumption and refund/unused-reservation model, backed by measurement.

The scheduler separates:
- **launch time:** when the physical job started;
- **observation time:** when the instrument/sample measurement was made; and
- **recorded time:** when the qualified result was received and entered into the ledger.

Results may arrive out of order. Event receipt times remain monotonic, while actual observation timestamps remain the instrument-declared measurement times. The helper does not implement clock synchronization, distributed leases, durable storage, crash recovery or idempotent remote transport.

## Authored tests

Fourteen async-specific tests cover:
1. realistic mean remains a fantasy, not an actual observation;
2. pessimistic bound sign changes with maximize/minimize objective;
3. seeded-uniform replay;
4. no-placeholder behavior;
5. exclusive lock contention and concurrent independent instrument use;
6. hard resource reservation and over-budget refusal;
7. duplicate candidate rejection with explicit replicate support;
8. out-of-order result arrival and preserved measurement timestamps;
9. failed-job cost settlement and lock release;
10. closeout refusal while work is pending;
11. actual result receipt/calibration/qualification requirements;
12. transition-time rejection atomicity;
13. invalid seeded policy refusal; and
14. difference between synthetic placeholder and actual measured value.

The standalone test binary also includes the 12 regression tests in the shared campaign-resource module.

### Reproduce locally

```sh
rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/async_buffer.rs -o /tmp/isru-async-tests
/tmp/isru-async-tests
```

**Validation honesty:** tests are authored. They are not considered executed until the dedicated exact-file format, strict Clippy and Rust test workflow passes on the current head.

## Benchmark design

Compare serial GP-EI, batch BO, asynchronous realistic-mean, pessimistic constant-liar, seeded-random fantasy, confidence-bound fantasy and no-fantasy asynchronous policies.

Freeze the candidate pool, objective, objective sense, external feasibility gate, budget, stage/resource locks, hidden world and paired seed schedule. Include:
- unequal stage durations and asynchronous return times;
- exclusive assay/spectrometer/reactor contention;
- calibration holds and optical-fouling-induced pauses;
- failed, cancelled and abandoned jobs;
- high-noise, censored and missing outcomes;
- multiple simultaneously pending jobs;
- material-family and batch holdouts; and
- campaign closure with results still pending.

Report both experiment-count and wall-clock views. At minimum report best independently accepted product/decision versus wall-clock time, decision regret, accepted product per total campaign hour, rig utilization, resource vector, duplicate in-flight candidate rate, pending-buffer size, failed-job cost, result-order robustness and fantasy-to-observation contamination (required zero).

## Claim boundary and production path

This is a queue/resource/fantasy-isolation reference. It is not a GP implementation, optimizer proof, hardware interlock, distributed scheduler or measured throughput result.

Next production steps are durable transactional job/event storage, lease-backed physical instrument reservations, authenticated calibration/qualification receipts, measured stage-duration distributions, proper censoring/failure likelihoods, and integration with Symtropy's workflow/evidence graph. A physical rig's safety authority must remain outside the optimizer.
