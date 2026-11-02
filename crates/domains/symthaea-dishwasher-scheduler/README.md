# symthaea-dishwasher-scheduler

Deterministic event-driven simulation of the proposed InstantClean QuickCell + FullLoad architecture.

## Policies compared

- `FullLoadOnly`: accumulate eligible jobs until full-load capacity or maximum batch wait.
- `QuickCellOnly`: clean each eligible item individually as the single cell becomes available.
- `HybridResourceAware`: use QuickCell for jobs that are ineligible for FullLoad, would miss their deadline in the next full batch, or belong to a timed-out partial batch that QuickCell Pareto-dominates on per-item energy, fresh water and total serial queue service duration.

The hybrid policy has no opaque weighted score. If QuickCell is no worse on energy, fresh water and total serial service duration for a timed-out partial batch, and strictly better on at least one, it diverts the items to QuickCell. If any objective conflicts or FullLoad is no worse, it retains FullLoad and reports the resulting trade-off.

## What is explicit

- Item arrival times and deadlines.
- Soil class and sanitization requirement.
- Mode capacity and process duration.
- Base and per-item energy and fresh-water coefficients.
- Soil support, sanitization support and caller-supplied cleaning/rinse validation flag.
- Maximum wait before a partial full-load batch is dispatched or compared with QuickCell.

A process with an unvalidated cleaning/rinse flag is never selected. Soil or sanitation incompatibility is not traded away for speed or resource use. Unprocessable jobs are reported rather than forced into a mode.

## Critical evidence boundary

The included `illustrative_profiles()` values are hypothetical design inputs only:
- QuickCell: 5 minutes, 0.05 kWh and 0.7 L per item.
- FullLoad: 14-item capacity, 120 minutes, 0.39 kWh and 6.7 L at full capacity.

They are not measured product performance and are not pulled from an industry label. The linear base-plus-per-item model is a sensitivity scaffold; it must be replaced with measured partial-load and full-load resource curves before product decisions.

The simulator does not model:
- Physical cleaning or rinse outcomes.
- Detergent chemistry or soil kinetics.
- Actual dishwasher hydraulic/thermal transients.
- Standby electricity, household tariffs, wastewater, filter loading, repairs or wear.
- Forecast errors or a learned policy.

Cleaning/rinse and sanitation flags are supplied assumptions. This crate cannot certify those processes.

## Simulation semantics

The finite workload trace is sorted by arrival and replayed deterministically. The hybrid deadline check uses known future arrivals in the replay trace to estimate the next possible full batch; that is an offline benchmark assumption, not a deployable forecast. A production controller must use a declared forecast or conservative no-forecast rule and report forecast error.

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

Synthetic workload generators use fixed seeds and explicit scenario families. They are reproducible but are not representative household-use data until calibrated.

## Illustrative crossover example

See [illustrative-crossover.md](docs/illustrative-crossover.md) for the arithmetic showing why QuickCell is most compelling on very sparse loads, while FullLoad wins resource efficiency on dense loads under the current hypothetical inputs. This is not a measured result.

## Interpreting output

Compare energy, fresh water, p95 end-to-end latency, deadline misses, unprocessed jobs and cycle counts together. Do not reduce the comparison to one weighted score. Keep eco-label tests separate from per-item QuickCell experiments.
