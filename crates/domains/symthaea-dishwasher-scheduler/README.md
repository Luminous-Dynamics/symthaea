# symthaea-dishwasher-scheduler

Deterministic event-driven simulation of the proposed InstantClean QuickCell + FullLoad architecture.

## Policies compared

- `FullLoadOnly`: accumulate eligible jobs until full-load capacity or maximum batch wait.
- `QuickCellOnly`: clean each eligible item individually as the single cell becomes available.
- `HybridResourceAware`: an **offline look-ahead reference** that uses known future arrivals to estimate the next full batch. Treat this as an optimistic benchmark, not a deployable policy.
- `HybridConservativeNoOracle`: a deployable-style conservative policy that never exposes future arrivals to the dispatch estimate. It assumes a partial batch may wait until its configured timeout, then checks deadline feasibility.

Both hybrid policies retain the no-weighted-score Pareto rule for timed-out partial batches. If QuickCell is no worse on energy, fresh water and total serial service duration, and strictly better on at least one, it diverts the items to QuickCell. If objectives conflict or FullLoad is no worse, it retains FullLoad. The conservative policy adds a worst-case timeout estimate for deadline checks; the offline look-ahead policy uses future arrivals only as an optimistic comparison bound.

## What is explicit

- Item arrival times and deadlines.
- Soil class and sanitization requirement.
- Mode capacity and process duration.
- Base and per-item energy and fresh-water coefficients.
- Soil support, sanitization support and caller-supplied cleaning/rinse validation flag.
- Maximum wait before a partial full-load batch is dispatched or compared with QuickCell.

A process with an unvalidated cleaning/rinse flag is never selected. Soil or sanitation incompatibility is not traded away for speed or resource use. Unprocessable jobs are reported rather than forced into a mode.

## Evidence-qualified measured occupancy curves

The `resource_curve` module accepts physical resource observations scoped to one exact machine, programme, soil class, surface material, protocol and raw-data SHA-256 digest. It requires at least two measured occupancies, at least five independent runs per occupancy, and zero cleaning/rinse failures at every occupancy point before constructing an eligible curve.

- Exact measured occupancy returns the stored observation.
- Intermediate occupancy uses piecewise-linear interpolation, explicitly marked as an estimate.
- Occupancy outside the measured range is rejected; no extrapolation is permitted.
- Min/mean/max values are descriptive observed ranges, not confidence intervals.
- Interpolated values are diagnostic only until the chosen interpolation model is validated on held-out physical occupancy points.
- Failed cleaning or rinse observations disqualify the curve; resource savings cannot compensate for endpoint failure.

This API is separate from the illustrative linear `ModeProfile` coefficients. It does not silently change scheduler defaults. Issue #10136 remains the physical measurement programme required to supply real curves.

## Critical evidence boundary

The included `illustrative_profiles()` values are hypothetical design inputs only:
- QuickCell: 5 minutes, 0.05 kWh and 0.7 L per item. This is a simulation fixture, not a declared soil capability or a claim for normally/heavily soiled items. The separate engineering target ladder limits the <=4-minute aspiration to lightly soiled glassware and treats <=15 minutes for normally soiled single items as a stretch hypothesis.
- FullLoad: 14-item capacity, 120 minutes, 0.39 kWh and 6.7 L at full capacity.
- These profiles have permissive eligibility flags so scheduling mechanics can be exercised; they do not prove cleaning, rinsing or sanitization for any soil/material class.

They are not measured product performance and are not pulled from an industry label. The linear base-plus-per-item model is a sensitivity scaffold; it must be replaced with measured partial-load and full-load resource curves before product decisions.

The simulator does not model:
- Physical cleaning or rinse outcomes.
- Detergent chemistry or soil kinetics.
- Actual dishwasher hydraulic/thermal transients.
- Standby electricity, household tariffs, wastewater, filter loading, repairs or wear.
- Forecast errors or a learned policy.

Cleaning/rinse and sanitation flags are supplied assumptions. This crate cannot certify those processes.

## Simulation semantics

The finite workload trace is sorted by arrival and replayed deterministically. `HybridResourceAware` uses known future arrivals and is therefore an offline look-ahead reference only. `HybridConservativeNoOracle` estimates completion from currently pending jobs plus the configured timeout; future arrivals do not alter its current dispatch decision. It is deliberately conservative and may choose QuickCell more often than an accurate production forecast would. A future predictive controller must declare its forecast, quantify forecast error and fall back to a conservative rule when confidence is inadequate.

The result reports:
- Energy and fresh water totals.
- QuickCell and FullLoad cycle counts.
- Unprocessed jobs and deadline misses.
- Mean and p95 queue wait (arrival to process start).
- Mean and p95 end-to-end latency (arrival to item release).
- Workload makespan measured from first arrival to final completion.

## Run

~~~sh
cargo test -p symthaea-dishwasher-scheduler
cargo run -p symthaea-dishwasher-scheduler --bin dispatch_bench
cargo clippy -p symthaea-dishwasher-scheduler --all-targets -- -D warnings
~~~

The benchmark binary emits raw CSV for 1,000 deterministic seeds per scenario family (sparse, meal-burst and irregular), comparing four policies. The output is 12,000 trace/policy rows and can be aggregated with confidence intervals by a separate analysis step. Synthetic workload generators are reproducible fixtures, not representative household-use data until calibrated. All current profile coefficients remain hypothetical.

## Illustrative crossover example

See [illustrative-crossover.md](docs/illustrative-crossover.md) for the arithmetic showing why QuickCell is most compelling on very sparse loads, while FullLoad wins resource efficiency on dense loads under the current hypothetical inputs. This is not a measured result.

## Interpreting output

Compare energy, fresh water, p95 end-to-end latency, deadline misses, unprocessed jobs and cycle counts together. Do not reduce the comparison to one weighted score. Keep eco-label tests separate from per-item QuickCell experiments.
