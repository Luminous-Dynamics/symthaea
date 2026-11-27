# CMAQ-035 — Paired PRM benchmark harness

Run from the repository root:

    cargo +1.100.0 run --release --manifest-path research/cmaq-035/Cargo.toml --locked -- 10000 500 20261127 > cmaq-035-results.csv

Arguments are replications, horizon and seed. Defaults are 10,000, 500 and 20,261,127. Output is CSV on stdout, with simulation configuration on stderr and paired win/loss/tie comment rows in stdout. The dedicated PR workflow runs 10,000 replications per scenario at a 500-observation horizon and uploads the CSV plus run metadata as a 30-day artifact.

## Design

- Calibration sample: 99 independent synthetic cluster blocks.
- Miscoverage: alpha = 0.05.
- Both methods consume the same frozen reference and identical one-based ranks.
- Discrete ties use independent deterministic pseudo-random 64-bit keys, unique within each replication. This is simulation-only, not production cryptographic randomness.
- Score RNG and tie-key RNG use distinct deterministic seed domains.
- The null samples calibration and deployment scores from the same discrete uniform distribution.
- Upper/lower location shifts change score location.
- Dispersion expansion/contraction change spread around the same center.
- The symmetric three-point alternative preserves ideal mean rank and mean absolute rank deviation while altering higher-order shape. It probes a known feature blind spot for the two-feature ONS portfolio.

## Output and interpretation

For each scenario and method: empirical probability of quarantine by the horizon, 95% Wilson interval, median / 90th / 95th percentile detection delay conditional on detection. Paired win/loss/tie counts compare detection delay on common random-number streams for alternative shifts only. Null false alarms are reported solely as rates and confidence intervals; paired timing wins are not defined for the null.

Null false-alarm probability must be checked before interpreting alternative power. No alarm is not assigned a fabricated delay. This simulation does not establish safety or guarantee detection.

## Limits and next comparators

The harness invokes the actual CMAQ-031 and CMAQ-033 code but uses synthetic score receipt digests. It does not authenticate evidence or prove independence of physical environment clusters. Its RNG is not cryptographic.

CCTM and confidence-sequence risk monitors are not yet implemented here. Add faithful versions and reproduce the exact 2026 PRM paper comparison before claiming reproduction or superiority. Expand to multiple calibration sizes, at least 10,000 null replications, power intervals, feature ablations, and real frozen environment-family score corpora.
