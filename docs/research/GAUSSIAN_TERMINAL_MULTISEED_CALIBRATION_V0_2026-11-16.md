# Gaussian terminal-test multi-seed calibration v0

**Date:** 2026-11-16  
**Tracking issue:** #12472  
**Parent method:** #12426 / PR #12433  
**Implementation:** `gaussian_terminal_multiseed_calibration` in the standalone falsification-power-reference crate

## 1. Purpose

The fixed-design terminal-test induction campaign uses a deterministic 20,000-path root-seed simulation. Its exact-head alternative run reported empirical terminal Neyman–Pearson rejection of 79.97%, while the analytic terminal power was 79.24%; the 95% Wilson interval narrowly excluded the analytic value, although the pre-existing broad ±1.5 percentage-point workflow tolerance passed.

A single deterministic seed is replayable, but it does not characterize seed sensitivity. This follow-up freezes five additional root seeds and executes 20,000 paths per seed and scenario. Every seed is reported; no seed is discarded or selected after observing the outcome.

## 2. Frozen campaign

- Root seed count: 5.
- Paths per root seed and scenario: 20,000.
- Pooled paths per scenario: 100,000.
- Fixed horizon: 80 observations.
- Nominal alpha: 0.05.
- Gaussian alternative discrepancy: 0.22.
- Known standard deviation: 0.80.
- Induced-test stopping thresholds: 0.75, 0.90 and 0.99.
- Observation stream: SplitMix64 / Box–Muller, versioned in the report.
- Separate randomizer stream per threshold, derived from the path seed and a frozen threshold salt.
- Common observation path for the terminal test and all three stopping policies within each path.
- Simple Gaussian null and declared simple Gaussian alternative.

The five literal root seeds are committed in the Rust source and emitted in the JSON report. The evaluator rejects duplicate root seeds. These pseudorandom streams are for deterministic simulation and are not cryptographic or suitable for real test randomization.

## 3. Reported quantities

For each root seed and for the pooled 100,000-path campaign, report:
- fixed-horizon terminal NP rejection rate and 95% Wilson interval;
- analytic terminal NP power for the declared alternative;
- randomized rejection rate and Wilson interval for each stopping threshold;
- early stopping rate and mean stopping look for each threshold;
- exact terminal-induced mismatch count.

The terminal-induced decision must agree exactly with the fixed-horizon NP decision on every path. The same observation paths are used across policies; each threshold has a separate independent pseudorandomizer stream. Failed, malformed or incomplete reports fail the workflow rather than becoming zero rates.

## 4. Preregistered workflow guards

The focused workflow checks:
- schema, exact scenario IDs, five unique seeds, path counts, horizon and threshold grid;
- finite [0,1] rates and Wilson intervals that contain their observed rates;
- five per-seed records in every scenario, each containing 20,000 paths;
- zero terminal-induced mismatches at both per-seed and pooled levels;
- pooled null terminal and induced rejection rates in [0.04, 0.06];
- per-seed null rejection rates in the wider [0.03, 0.07] diagnostic band;
- pooled alternative terminal NP rejection within 0.007 absolute of analytic terminal power;
- non-increasing early-stop rates as the threshold rises.

The alternative rejection rates and cost/power frontier are reported even when a comparison is not monotone. The randomized alternative power is not used as an unconditional dominance gate: it remains an empirical operating characteristic of the particular frozen design and independent randomizer.

These are numerical calibration guards for this finite synthetic generator. They are not universal statistical theorems or a physical validation result.

## 5. What this does not establish

This multi-seed campaign checks Monte Carlo stability for one fixed simple-versus-simple Gaussian design. It does not qualify:
- unknown or estimated variances;
- composite or misspecified nulls;
- correlated or serially dependent residuals;
- adaptive future measurement designs;
- nuisance-parameter estimation;
- arbitrary model-adequacy power;
- sensor calibration or real physical measurements.

The fixed-design terminal induction remains distinct from the predictable Gaussian e-process. The former induces a bounded randomized test from a frozen terminal NP test; the latter is a likelihood-ratio e-process under its own predictable conditional Gaussian contract.

## 6. Promotion boundary

The exact-head Rust workflow must pass formatting, all-target tests, the original single-seed sequential and terminal campaigns, this multi-seed campaign, artifact upload and strict Clippy. The uploaded report, source SHA and SHA-256 digest must all bind to the same PR head.

A passing workflow establishes only implementation and Monte Carlo calibration for the declared generator, not scientific model discovery, SOTA performance, a general sequential-test guarantee or physical validation.
