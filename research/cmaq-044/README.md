# CMAQ-044 — Paired effective-block benchmark inference

## Purpose

CMAQ-044 prevents three common benchmark errors:

1. **Survivor-only delay:** calculating median detection delay only among runs that detected, while dropping misses.
2. **Pseudoreplication:** treating overlapping clips from one drive, asset, session, shared seed or common shock as independent observations.
3. **Unpaired comparisons:** comparing candidate and baseline over different scenarios or configurations, allowing scenario mix to masquerade as an algorithm improvement.

Each input record is one pre-registered paired effective common-cause block. Baseline and candidate share the same source-pair digest, scenario manifest, score definition and closure manifest. Their binary build digests are recorded separately.

## Primary delay estimand

All detection delays are measured from the true change point. For a frozen post-change evaluation horizon H:

    penalized_delay = detected_delay, if detected by H
                    = H + 1,           if missed

The H+1 penalty preserves the miss in the denominator and follows the conservative capped-delay convention used in the PF-CCTM reproduction protocol. The caller must pre-aggregate multiple episodes inside each effective block using a pre-registered worst-case rule before creating the block record. It must not average clips as though they were independent.

For each paired block:

    delta_block = candidate_penalized_delay - baseline_penalized_delay

Negative values favor the candidate. CMAQ-044 reports:
- paired mean penalized-delay difference;
- nearest-rank 2.5th/97.5th percentile bootstrap interval for the paired mean;
- candidate wins, ties and losses at block level;
- baseline and candidate mean penalized delay and miss counts;
- separate per-scenario-family reports.

Bootstrap resampling samples paired effective blocks with replacement. Both arms remain together within each draw. Each family with fewer than two effective blocks reports its paired point estimate but has no bootstrap interval. The overall evaluation requires at least two independent blocks.

## Operational outcomes

Each arm also reports:
- fallback completions before the independent recoverability deadline;
- safety violations;
- false warnings and false quarantines separately;
- safe task completions and useful progress;
- unioned safe exposure per asset/clock domain;
- false warnings and quarantines per safe operating hour, represented as events/hour × 1,000,000 (ppm).

Exposure intervals are unioned by domain, so overlapping clips cannot inflate the denominator. The rates always have a raw event count and safe-exposure denominator available. False-alarm rates are not folded into the delay score.

## Determinism and evidence bindings

The bootstrap PRNG is deterministic SplitMix64 with rejection sampling for unbiased bounded indices over its 64-bit source. A report records:
- horizon;
- bootstrap replicate count;
- bootstrap seed;
- paired block count and per-family block count;
- source-pair, scenario, score, closure and both build digests per block.

All digests are checked for 64-character lowercase SHA-256 shape only. Each block must bind one common alpha fraction, calibration-reference digest, score definition, tie protocol, scenario manifest, and fixed baseline/candidate build and configuration digests; mismatches across rows are rejected. Duplicate source-pair identities are rejected. This crate does not authenticate signatures or prove that the source pairs, family labels, closure IDs or truth labels are complete/correct.

## Interpretation limits

The bootstrap interval is conditional on effective blocks being representative/exchangeable of the target deployment population. It is not a distribution-free confidence interval under arbitrary dependence. The effective-block inventory must capture all common causes; the block-level worst-case aggregation must be fixed before looking at arm outcomes.

A narrower interval or faster candidate is not sufficient to declare superiority. The candidate should also have no unacceptable regression in false alerts per hour, fallback success, safety violations, task completion, useful progress or worst-family outcomes. Simulator, HIL and physical deployments must be reported as separate evidence tiers.

## Build and test

    cargo +1.100.0 test --manifest-path research/cmaq-044/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-044/Cargo.toml --locked --all-targets -- -D warnings

The dedicated workflow checks formatting, tests and strict Clippy.
