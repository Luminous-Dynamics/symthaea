# Symthaea BCI core (R&D)

Hardware-independent, fail-closed contracts for early neural-signal acquisition work.

**Status:** initial research implementation; not yet compiled/qualified and not a medical device.

## Scope

This package validates signal-frame schema, channel/sample shape, finite sample values, explicit producer-supplied quality metadata, caller-supplied monotonic time/freshness, contiguous frame and sample counters, sample-rate-derived inter-frame timing continuity, clock epoch, sampling rate, channel layout, and preprocessing fingerprint. After a stream reset, the next accepted frame must use a strictly greater clock epoch. The `new_with_epoch_floor` constructor lets callers restore the last durable epoch after process restart; the plain `new` constructor is only sufficient where no prior epoch needs to be protected.

It does not acquire from EEG hardware, decode intent, or estimate consciousness. It includes an independent `ActionGate` that converts a proposal paired with an opaque validator-issued receipt into bounded **sandbox-only** action values. The gate checks evidence freshness and quality, confidence and candidate margin, stream epoch/sequence replay, cursor displacement bounds, action allowlists, and out-of-band confirmation for selection. It is not a real-device authorization or a safety certification.

The `selective` module evaluates a preregistered confidence-threshold sweep and reports end-to-end coverage, selective risk, fixed-bin confidence-calibration error, and false action onsets per hour using caller-supplied idle exposure. Its denominator assumes disjoint time intervals, and a sustained command must only count once at onset.

The `benchmark` module audits split metadata before training/evaluation. It can enforce subject-disjoint or recording/session-disjoint splits and detects overlapping or insufficiently separated windows in the same recording. The sample guard is configured explicitly; it does not guess filtering-window leakage bounds. It also computes cell-mean balanced accuracy with an explicit participant × session × recording-context inventory, requiring each cell and every expected class to be present so missing cells cannot silently disappear from the score. Recalls and cell scores are rounded down to parts-per-million for deterministic reporting.

The same module defines a versioned `BenchmarkRunManifest` and integer-scaled `BenchmarkScorecard`. Scorecard validation binds dataset, split, protocol, preprocessing, model, code revision, and environment identities; validates metric uniqueness, sample counts, scales, uncertainty bounds, and presence of the declared primary metric. Canonical serialization and SHA-256 calculation remain the caller's responsibility and must be performed by a reviewed layer—the dependency-free core checks the supplied digest binding, not the correctness of external hashing. See issue [BCI-RD-002](https://github.com/Luminous-Dynamics/symthaea/issues/8186).

The quality score must be calibrated and documented by each dataset/device protocol; the core intentionally does not ship a universal threshold.

## Tests

When Rust is available:

```bash
cargo test --manifest-path research/bci-core/Cargo.toml
cargo clippy --manifest-path research/bci-core/Cargo.toml -- -D warnings
```

The unit fixtures cover accepted contiguous frames, missing quality, NaN, malformed/oversized data, duplicate channels, low-quality/clipped/flatline frames, stale/future inputs, stream gaps and cursor non-advancement after a rejection, clock/timestamp/rate/montage/preprocessing changes, explicit reset with strictly advancing epoch, restart epoch-floor recovery, 100,000 deterministic fail-closed adversarial mutations, timing gaps inconsistent with sample count/rate, invalid validator configuration, action confidence/margin rejection, stale evidence, bounded movement, required selection confirmation, replay-sequence rejection, older-clock-epoch rejection, and invalid action-gate configuration. Selective-metric fixtures cover threshold sweeps, risk/coverage, confidence calibration, no-accepted-output cases, quality-gated idle periods, invalid confidence, duplicate thresholds, and inconsistent claims of correct prediction alongside an idle action onset. Split-audit fixtures cover valid guarded windows, overlapping windows, too-small temporal gaps, nested overlap missed by adjacent-only checks, subject/recording leakage, malformed identities/ranges, and empty splits. Run/scorecard fixtures cover run-manifest digest binding, model substitution, missing licence identity, zero digest, invalid uncertainty bounds, duplicate metrics, and missing primary metric. Cell-score fixtures cover equal cell weighting, required class support, missing/unexpected cells, duplicate/unknown labels, and order invariance.

## Deliberate limitations

- No real dataset, device adapter, decoder, actuator, or clinical outcome.
- No claim that these structural tests establish safety.
- No external dependencies.
- The crate is isolated as a nested Cargo workspace to avoid changing the root workspace or lockfile before the Rust tests run.

Research track: https://github.com/Luminous-Dynamics/symthaea/issues/8179
Benchmark harness: https://github.com/Luminous-Dynamics/symthaea/issues/8186
External Track 02 evaluation (registration deadline Oct 24, 2026): https://github.com/Luminous-Dynamics/symthaea/issues/8195
Pull request: https://github.com/Luminous-Dynamics/symthaea/pull/8180


## Participant-level uncertainty

src/statistics.rs implements a deterministic paired bootstrap over one aggregate score per held-out participant. See docs/research/BCI_ASSISTIVE_INTERFACE_RND.md for the protocol and limitations. It complements the official cell-mean benchmark metric and does not replace it. The Rust tests have been authored, but execution and correctness remain unverified until the isolated crate is compiled and the tests are run on a declared Rust toolchain.


### Track 02 scorer compatibility

The benchmark module now exposes evaluate_track02_reference_compatible_cell_mean, which mirrors the public challenge metric's per-cell union of target and predicted classes plus equal cell weighting. It retains fixed-point ppm rounding and therefore is not a bit-for-bit float64 port. The stricter generic evaluator remains available when full class support is required. See the research RFC for the pinned upstream source and compatibility limits; the new tests are authored but have not been executed on a Rust toolchain.


### Paired uncertainty for the official cell-mean score

paired_cell_mean_unit_bootstrap follows the challenge website's published post-evaluation unit convention: it resamples participant × session × context cell scores and computes paired candidate-versus-baseline differences from the same sampled cells. paired_cell_mean_participant_bootstrap is provided separately as a participant-cluster sensitivity analysis, not as the website's default. Both require PairedLabeledPrediction rows with exact stable window-ID, cell and target alignment. Their method ID, seed, replication count, confidence level, observed delta and interval are recorded in the result. Fixed-point rounding and percentile-index conventions mean golden comparisons against the pinned reference procedure are still required. The expanded tests are authored, not yet compiled or run.


### Full-field paired inference and rank stability

analyze_leaderboard requires model-ID and independent-unit-ID inventories from a frozen benchmark manifest, plus one fixed-point score for every model × unit pair. This catches missing units even when every model omits the same unit. It reuses the same bootstrap sample of units for every model, reports paired deltas and percentile intervals, computes two-sided smaller-tail bootstrap p-values, applies Holm correction to the complete pairwise family, and reports per-model mean rank, normalized rank-frequency vector (summing exactly to 1000 permille), and top-1/top-3/top-5 frequencies. Ties share competition rank, so top-k frequencies are not mutually exclusive across models. The API enforces complete score grids, fixed-point score bounds, configuration validity, and a compute/storage budget. It assumes higher is better and frozen independent-unit scores; it does not prove SOTA or user benefit. Golden parity against the challenge's Python/NumPy method and actual Rust compilation/tests remain required.


### Calibration-efficiency non-inferiority gate

The calibration module evaluates one preregistered contrast: candidate model at a smaller calibration budget versus a reference at the larger budget, on exactly matched held-out participants. It requires an explicit unit such as labelled_examples or minutes and reports the budget reduction, paired score interval and one of SupportsNonInferiority, EvidenceOfInferiority or Inconclusive against a declared absolute margin. It does not search budgets on the final test set, train/adapt a model, or establish clinical benefit. The synthetic tests are authored; Rust compilation/execution remains unverified.
