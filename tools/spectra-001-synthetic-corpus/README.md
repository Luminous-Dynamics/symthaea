# SPECTRA-001B deterministic synthetic corpus

This crate creates simulation-only calibration/evaluation campaigns for the
SPECTRA hidden-target protocol. It composes with the SPECTRA-001 T0 qualifier
and keeps candidate-facing evaluation observations separate from evaluator-only
target truth.

## Synthetic worlds

- Null: response rankings and all telemetry codes are independent of target.
- Signal injected: the target is promoted to rank one with a declared synthetic
  probability; telemetry remains target-independent.
- Known leakage: responses remain chance-generated while exactly one of nine
  synthetic channels exposes the target index: RF, low-frequency magnetic,
  acoustic, optical, vibration, thermal/airflow, conducted power, network
  timing, or procedural trial-ID metadata.
- Confounded: response rankings remain target-independent, but participant ID
  is coupled to the target assignment. The procedural attacker recovers this
  association, demonstrating a design/metadata confound rather than a physical
  transmission mechanism.
- Adversarial mutation: the response-seal timestamp follows target reveal and
  the existing T0 validator must reject the affected trials.

## Candidate/evaluator separation

PublicCampaignV1 includes an opaque caller-supplied campaign instance ID,
revealed calibration labels and evaluation rows with no target labels. It
contains no seed, world label, leakage label, hidden target ID or expected
outcome. The campaign ID must be assigned independently of target/world truth.

CampaignTruthV1 is a separate evaluator-only sidecar containing hidden labels
and generator parameters. It must never be passed to a candidate or
candidate-facing red-team model.

Candidate IDs are neutral. The procedural leakage scenario intentionally embeds
the target index in the trial ID as one adversarial feature. All other scenarios
use non-semantic trial IDs.

## Red-team baseline

A transparent per-channel lookup model learns feature-to-target mappings from
revealed calibration rows. It predicts evaluation targets from each admitted
channel independently. The report provides per-channel prediction counts,
accuracy and a configurable threshold flag.

This is a weak baseline, not an optimal attacker or calibrated hypothesis test.
The default 850-permille flag is a synthetic detection threshold only. A real
study would require preregistered classifier families, held-out seeds,
family-wise error control and independent validation.

## PRNG and determinism

The generator uses SplitMix64 with rejection-sampled bounded integers and
Fisher-Yates shuffling. It supports deterministic fixtures; it is not
cryptographic randomness, an external randomness beacon or a randomization
authority.

## Run

    cargo test --manifest-path tools/spectra-001-synthetic-corpus/Cargo.toml

## Claim ceiling

This crate tests synthetic code paths only. Detecting a generated telemetry
channel means only that the synthetic feature encodes the known target under
the declared generator. It is not evidence that the corresponding real-world
physical channel exists, is measurable or explains a human result.

No participant recruitment, physical instrument control, hardware actuation,
human exposure, ESP claim or new-field theory is included.

## Frozen reference campaign

The independent reference fixture is stored at
research/spectra-001/fixtures/corpus-known-answer-v0.1.json. It freezes exact
null and injected-signal scores, per-channel red-team accuracies, all nine
leakage-channel seeds, and the chronology-mutation disposition. The Rust unit
tests replay the exact null and signal scores; the leakage tests require
1,000-permille recovery on the channel deliberately wired to the target.

The current implementation covers W0 null, W1 injected response signal, W2
single-channel leakage, a narrow W3 participant-ID assignment confound, and W4
response-after-reveal integrity mutation. Broader session/operator/day/instrument
common-mode confounding, repeated-target imbalance, outcome-dependent stopping
and selective missingness remain follow-up work under issue 10046. No false-
positive rate or statistical power claim is made from one frozen seed.
