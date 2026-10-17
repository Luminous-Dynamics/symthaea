# OPEN-OBS-001 Evaluator (v0.2)

This is a small standalone Rust evaluator for the [OPEN-OBS-001 research contract](../OPEN_OBS_001_BENCHMARK.md). It intentionally sits outside the root Symthaea Cargo workspace globs, so it can be developed and tested independently without resolving unrelated workspace members.

The core evaluator scores detector dispositions; the companion synthetic runner generates deterministic feature grids and applies a deliberately limited matched-track baseline. Neither creates physical observations or infers extra dimensions or agency. Input truth labels must be held by the evaluator, not exposed to a candidate model before predictions and thresholds are frozen.

## Run

From the repository root:

```sh
cargo test --manifest-path research/open-obs-001-evaluator/Cargo.toml
cargo run --manifest-path research/open-obs-001-evaluator/Cargo.toml \
  --bin open-obs-evaluator -- /path/to/frozen-holdout.tsv \
  --source-revision FULL_40_CHARACTER_COMMIT_SHA
```

The command prints a deterministic TSV report. It exits with:
- `0` when every v0.2 qualification gate passes;
- `1` when a gate fails or evidence is insufficient;
- `2` for an invalid manifest, missing file, or invalid CLI arguments.

A successful execution is scoped to the declared manifest/profile. For example, an all-simulated run can qualify only the synthetic benchmark envelope; it says nothing about the physical universe.

## Manifest format

UTF-8, tab-separated, exact header, LF newlines, final LF required, no blank records, no quoted fields:

```text
case_id background_group partition stratum truth decision family_truth family_decision capture data_kind observation_sha256 background_sha256 instrument_profile_sha256 scenario_profile_sha256 detector_profile_sha256
```

The exact header uses tabs, not spaces. Enumerations:

| Field | Allowed values |
|---|---|
| `partition` | `train`, `calibration`, `holdout` |
| `truth` | `signal`, `no_signal` |
| `decision` | `candidate`, `reject`, `unknown` |
| `family_truth` | `known`, `held_out`, `not_applicable` |
| `family_decision` | `known`, `unknown`, `not_assigned` |
| `capture` | `complete`, `incomplete` |
| `data_kind` | `raw_observation`, `derived`, `simulated`, `independent_measurement` |
| `observation_sha256` | 64 lowercase hexadecimal characters; unique per case |
| `background_sha256` | 64 lowercase hexadecimal characters; unique per trial bundle |
| `instrument_profile_sha256` | 64 lowercase hexadecimal characters; one profile per manifest |
| `scenario_profile_sha256` | 64 lowercase hexadecimal characters; one profile per stratum and data kind |
| `detector_profile_sha256` | 64 lowercase hexadecimal characters; one profile per manifest |

A signal row requires `family_truth=known` or `held_out`. A no-signal row uses `not_applicable`. Non-candidate decisions use `family_decision=not_assigned`. Incomplete capture may only produce `decision=unknown` and may not carry a known-family assignment.

## Independence and leakage rules

- A `background_group` may occur only once in the entire manifest, not merely once per stratum or partition. This v0.2 rule is intentionally strict: one acquisition/background group is one scored trial bundle. Related target/control or ON/OFF panels must be encoded inside that single bundle, not split into rows that look independent. Repeated windows, injections, or crops from the same acquisition must be aggregated by a preregistered rule or rejected. Do not inflate sample size by counting correlated windows.
- Different group IDs must be justified by acquisition provenance; this parser cannot prove independence from labels alone.
- `stratum` names a predeclared signal-family/SNR slice or negative-control family. Define its meaning before the holdout is evaluated.
- Metrics and confidence bounds are emitted by `(stratum, data_kind)`, avoiding silent pooling of simulated and measured evidence.
- Every row carries lowercase 64-character SHA-256 fields for the observation bytes, pristine/background bundle, instrument profile, scenario/injection recipe, and detector profile. The parser rejects duplicate observation/background digests, non-canonical digest strings, mixed instrument/detector profiles within one manifest, and mixed scenario profiles within one `(stratum, data_kind)`. The full manifest bytes are also SHA-256 hashed, and the report retains the supplied evaluator source revision.

## Metrics and gates

Recall treats both `reject` and `unknown` on a true signal as missed detections. Operational false-positive rate is the share of independent no-signal trials marked `candidate`; unknown controls are reported separately. Each metric includes the number of independent groups and one-sided 95% Wilson bounds.

A stratum qualifies for the detection gate only when the recall point estimate is at least 90% **and** its one-sided 95% lower bound is at least 90%. A negative-control stratum qualifies only when the false-positive point estimate is at most 1% **and** its one-sided 95% upper bound is at most 1%. Point-estimate success with insufficient trials is reported as `INCONCLUSIVE`, not PASS.

As a planning approximation, zero false positives in about 300 independent negative-control trials supports an upper 95% bound near 1%; about 3,000 independent zero-failure trials support a bound near 0.1%. Use actual computed bounds, and count independent background groups—not windows. Correlated or repeated examples invalidate this interpretation.

Each `data_kind` is gated independently: simulated results cannot compensate for missing measured controls, and measured controls cannot compensate for missing simulated positive recovery. The overall result passes only when detection, false-positive control, and open-set gates pass separately for every data kind represented in the holdout. This also means a mixed manifest with a missing evidence arm is inconclusive, not a pooled PASS.

The open-set gate fails if a held-out signal family is labelled as a known family, passes only when at least one held-out signal of that data kind is surfaced as a candidate and explicitly labelled unknown, and is otherwise inconclusive.

## Limits

- The evaluator aggregates supplied decisions and the optional synthetic runner exercises a simple fixed detector. Neither decides whether the chosen physical model is correct.
- Digest strings are schema-checked and bound into the report, but this tool does not independently open the original sensor files and recompute each digest, authenticate who captured them, or prove that different background groups are truly independent. A signature/build-attestation layer and data-source adapters remain necessary for real-world evidence receipts.
- It cannot establish that data acquisition was truly independent; source grouping must be grounded in acquisition provenance.
- It does not provide a signed cryptographic attestation or authenticate the owner of input files.
- The next tranche is tracked in [OPEN-OBS-002 #7838](https://github.com/Luminous-Dynamics/symthaea/issues/7838): paired target/control bundles, tests against held-out real backgrounds where licenses permit, instrument-specific acquisition semantics, and independent replication.
- A benchmark PASS does not establish extra-dimensional physics, intelligence, consciousness or a higher-dimensional being.


## Deterministic synthetic challenge corpus

A fixture generator can emit the same frozen synthetic corpus under four transparent baseline profiles:
- `matched-track`: sums evidence along candidate straight tracks and has a separate known-linear family threshold;
- `peak-threshold`: flags a case when any absolute grid value reaches 0.20;
- `always-reject`: negative control that must fail detection recall;
- `always-candidate`: negative control that must fail false-positive control.

Each corpus has 300 in-family linear positives, 300 noise-only negative controls, 80 stationary terrestrial-mimic controls, 60 low-amplitude curved held-out signals, 60 high-amplitude curved held-out signals, and small separate train/calibration cohorts.

Each generated observation and pristine noise background is represented by its SHA-256 digest, and each row binds instrument, scenario/injection, and detector-profile digests. The full manifest is also hashed by the evaluator. The generator validates the completed manifest before writing it.

Generate and evaluate the default matched-track corpus:

```sh
cargo run --manifest-path research/open-obs-001-evaluator/Cargo.toml \
  --bin open_obs_synthetic_demo -- --baseline matched-track \
  --output /tmp/open-obs-synthetic.tsv
cargo run --manifest-path research/open-obs-001-evaluator/Cargo.toml \
  --bin open-obs-evaluator -- /tmp/open-obs-synthetic.tsv \
  --source-revision "$(git rev-parse HEAD)" > /tmp/open-obs-report.tsv
```

Swap `matched-track` for `peak-threshold`, `always-reject`, or `always-candidate` to generate a separately profile-hashed manifest for a baseline control. Never combine several detector profiles into one holdout manifest.

Expected failure patterns are frozen rather than tuned away. The matched-track baseline is expected to flag stationary interference mimics and miss low-amplitude curved signals. Peak-threshold should recover the weak curved signals but still flag stationary mimics. Always-reject and always-candidate must fail different gates. Paired synthetic integration tests exercise target-only signal, the same stationary mimic in both panels, control-only transient, and incomplete control capture. These are fixture expectations to verify with Rust tests, not physical performance claims. The feature grid is not a telescope simulator and does not establish astrophysical relevance.

The v0.2 manifest now has typed columns for observation/background, instrument, scenario/injection and detector-profile digests. The evaluator validates digest syntax, uniqueness and profile consistency, but it does not yet reopen the original source files to independently recompute each hash or provide a signed scientific evidence attestation; that remains in the follow-on provenance work.


## Paired target/control bundle assessment

The `paired` module provides a strict two-panel v1 contract. A single `PairedObservationBundle` contains exactly one target and one control panel, each with a panel SHA-256, capture-completeness state, and detector-candidate boolean. The bundle binds a background-group identity and instrument-profile digest; the assessment additionally binds the detector-profile digest.

The status is one of:
- `TARGET_ASSOCIATED_CANDIDATE`: the complete target panel was flagged, and the complete control was not;
- `CANDIDATE_ALSO_IN_CONTROL`: both complete panels were flagged, so target-only association is not supported;
- `CONTROL_ONLY_CANDIDATE`: only the control was flagged;
- `NO_CANDIDATE`: neither panel was flagged;
- `UNKNOWN_INCOMPLETE_CAPTURE`: either panel lacks complete capture, preventing attribution.

The source/bundle digest excludes detector outcomes, while the assessment digest includes the detector profile, per-panel outcomes, and status. Changing a detector disposition should therefore not rewrite the declared source-bundle identity. Reversing input panel order does not change the canonical bundle digest.

This is **panel association only**. Even `TARGET_ASSOCIATED_CANDIDATE` does not establish extraterrestrial origin, novel physics, agency, or a higher-dimensional being. The module validates digest syntax and equality but does not yet reopen the source files to recompute the declared panel digests; issue #7838 tracks that independent-verification tranche.
