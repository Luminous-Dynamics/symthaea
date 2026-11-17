# Optical ISRU B1 stateful digital-twin reference v0.1

**Status:** first Rust implementation authored; exact-head Rust 1.96 format/Clippy/tests pending  
**Parent:** Symthaea #12574 — ISRU-BENCH-002  
**Benchmark programme:** #12570 and PR #12572 (B0 replay oracle + two-gate contract)  
**Related foundations:** stage workflow #12549, asynchronous buffer #12552, resource budget #12536, Symtropy evidence profile #5734

## Purpose

B0 replay answers: "Can two policies be compared fairly on the same recorded outcomes?"

B1 adds the operational realities that make a laboratory campaign different from a static array of rows:
- experiments take different amounts of time;
- jobs can run at the same time on different apparatus;
- one instrument cannot be double-booked;
- calibration becomes due;
- optical apparatus accumulates fouling;
- cleaning and recalibration consume time and resources;
- failed attempts consume resources but do not advance a candidate's stage;
- measurements include deterministic-seed noise and degradation bias; and
- the policy must choose when to perform maintenance, requalify or continue.

The B1 reference keeps the benchmark's two scientific gates: discovery over selected designs and learning over never-measured holdouts.

## Hard epistemic boundary

Every B1 result is a `SimulatedObservation` with a distinct `SimulatedOutputKind` and an explicit `simulation_only` marker. A simulated assay is named `SimulatedIndependentAssay`; it is not an actual independent assay. The type carries no calibration, custody, qualification or authority receipt fields.

This is a scheduling and inference environment, not a calibrated digital twin of an existing rig. No real instrument, physical safety, process yield, optical performance or product acceptance is established by this code.

## Current implementation

`research/asteroid-optical-isru-v0.2/digital_twin.rs` supplies:

1. **Causal stage order:** Characterization → Exposure → Capture → Assay. A candidate cannot skip a stage or execute two stages concurrently.
2. **Instrument eligibility and locks:** stages declare eligible instruments; distinct instruments can work concurrently; an active instrument cannot be double-booked.
3. **Heterogeneous durations:** stage duration includes a deterministic per-candidate/stage/attempt jitter and an instrument-specific duration scale. The event queue completes jobs by simulated completion time, not launch order.
4. **Hard experiment and wall-time limits:** a new candidate consumes one experiment slot at first characterization. A launch is rejected if its worst-case duration exceeds the remaining wall-time envelope.
5. **Resource reservation:** component-wise resource cost is reserved before launch. Completion or failure releases the reservation and settles actual rig time plus the declared non-time costs into spent resources.
6. **Stochastic failures:** each candidate/stage/attempt has a deterministic keyed failure draw. A failed attempt consumes resources, releases the instrument, and leaves the candidate at the same stage so a distinct retry can be attempted.
7. **Calibration expiry:** each instrument has a finite number of operations per calibration; new launches are refused when that limit is reached. Recalibration is explicit and consumes simulated time and budget.
8. **Fouling and maintenance:** successful and failed operations both accumulate declared fouling. An instrument at or above its threshold refuses new work until a maintenance action consumes time/resources and resets fouling.
9. **Noise and drift:** stage outputs add a deterministic noise draw and a fouling-dependent bias. The seed, candidate, stage and attempt identify the random stream.
10. **Holdout isolation:** candidate metadata is visible through the agent port, but hidden truth is not. The evaluator requires a prediction for every holdout and returns rank correlation and interval coverage, not raw labels.
11. **Closeout and replay:** grading requires closeout; the event log captures launches, completions, failures, maintenance, recalibration and closeout.

## What this does not yet model

This first B1 version is intentionally narrow. It does not yet implement:
- full multi-stage partial-process state and physical resumability beyond preserving stage index after a failed attempt;
- a continuous physical state model for heat transfer, phase change, material morphology, volatile transport or optical path geometry;
- coupled/common-cause calibration drift across sensors or shared standards;
- asynchronous calibration jobs (recalibration and maintenance currently advance the clock synchronously and are rejected while any experiment is in flight);
- censored or interval-valued observations, correlated sensor noise, clock uncertainty or metrology covariance;
- verified SiLA 2 / OPC UA equipment adapters;
- a durable distributed job ledger or crash/recovery semantics; or
- calibration against physical rig traces.

These are follow-on research steps, not features to assume.

## Benchmark use

Compare fixed sequential, random feasible, GP-EI, batch BO, multi-stage BO, async fantasy policies, Symthaea EIG/EVSI and the Luminous stage-aware planner under:
- identical measurable/holdout candidates;
- identical stage plans and instrument fleet;
- identical budgets and hard gates;
- paired world seeds and action-independent keyed random draws where possible; and
- sealed evaluation with independent custodian grading.

Report time-to-simulated-assay threshold and best-so-far versus wall time, experiment count, resource vector, instrument utilization, queue time, maintenance downtime, failure rate/cost, calibration refusal rate, fouling exposure, partial work salvage, holdout ranking, interval coverage and deterministic event replay equality.

The current keyed draws use candidate ID + stage + attempt so random duration/noise/failure for one candidate-stage attempt is stable across action order. This provides a common-random-number starting point, but full cross-policy pairing requires the same candidate retry semantics and a sealed action protocol. Document any divergence.

## Initial benchmark hypotheses (not results)

- A stage-aware policy should improve accepted simulated assay per total wall-clock hour over one-shot BO when heterogeneous stages and apparatus constraints matter.
- A policy that models calibration/fouling state should reduce wasted attempts after instruments become ineligible.
- Async policies should improve utilization only when the added parallelism offsets lock contention, queueing, calibration holds and maintenance.
- Pessimistic fantasies should be compared against realistic-mean, random and no-fantasy controls; no policy is assumed to win.
- Zero holdout leakage, zero resource overcommit, zero stale-calibration starts and zero post-close mutation are hard correctness requirements.

No numeric superiority target is claimed until a sealed, paired B1 tournament has run with at least 30 seeds and published confidence intervals.

## Reproduce locally

    rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/digital_twin.rs -o /tmp/isru-digital-twin-tests
    /tmp/isru-digital-twin-tests

This file is a standalone simulator. A passing test suite validates its state-machine semantics, not the physical validity of its noise/fouling/failure distributions. Those distributions must be calibrated against instrument logs and physical experiments before B1 results can be used as engineering predictions or to justify B2.