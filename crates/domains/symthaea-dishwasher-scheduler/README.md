# symthaea-dishwasher-scheduler

Deterministic event-driven simulation of the proposed InstantClean QuickCell + FullLoad architecture.

## What it compares

- `FullLoadOnly`: accumulate eligible jobs until the full-load capacity or maximum batch wait is reached.
- `QuickCellOnly`: clean eligible jobs individually as the single cell becomes available.
- `HybridDeadlineAware`: use QuickCell when FullLoad is not eligible or the next possible full batch would miss the job deadline; otherwise accumulate for FullLoad.

## What is explicit

- Item arrival times and deadlines.
- Soil class and sanitization requirement.
- Mode capacity and process duration.
- Base and per-item energy and fresh-water coefficients.
- Soil support, sanitization support and a caller-supplied cleaning/rinse validation flag.

A process with an unvalidated cleaning/rinse flag is never selected. Soil or sanitation incompatibility is not traded away for speed or resource use. Unprocessable jobs are reported rather than silently forced into a mode.

## Critical evidence boundary

The included `illustrative_profiles()` values are hypothetical design inputs only:
- QuickCell: 5 minutes, 0.05 kWh and 0.7 L per item.
- FullLoad: 14-item capacity, 120 minutes, 0.39 kWh and 6.7 L at full capacity.

They are not measured product performance and are not pulled from an industry label. The linear base-plus-per-item model is a convenient sensitivity scaffold; it must be replaced with measured partial-load and full-load resource curves before product decisions.

The simulator does not model:
- Physical cleaning or rinse outcomes.
- Detergent chemistry or soil kinetics.
- Actual dishwasher hydraulic/thermal transients.
- Standby electricity, household tariffs, wastewater, filter loading, repairs or wear.
- Forecast errors or a learned policy.

Cleaning/rinse and sanitation flags are supplied assumptions. This crate cannot certify those processes.

## Run

~~~sh
cargo test -p symthaea-dishwasher-scheduler
cargo run -p symthaea-dishwasher-scheduler --bin dispatch_bench
cargo clippy -p symthaea-dishwasher-scheduler --all-targets -- -D warnings
~~~

Synthetic workload generators use a fixed seed and explicit pattern, so results are reproducible but are not representative household data without calibration.

## Interpreting the output

Compare energy, fresh water, waiting-time percentiles, deadline misses, unprocessed jobs and cycle counts together. Never rank policies on one weighted score without showing Pareto alternatives. Keep labelled eco-cycle comparisons separate from per-item QuickCell experiments.
