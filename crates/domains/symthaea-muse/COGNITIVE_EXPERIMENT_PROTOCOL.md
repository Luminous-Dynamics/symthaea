# First Symbolic Cognition Experiment Protocol

## Status

Pre-registration scaffold. Do not interpret descriptive summaries as
inferential evidence.

## Question

Can the Symthaea policy choose development and recapitulation alternatives that
preserve motif identity, increase instability during development, and produce a
clearer or more useful return than fixed and random-valid policies without
violating hard musical constraints?

## Arms

1. Fixed deterministic policy.
2. Random selection among theory-valid alternatives.
3. Existing hand-authored heuristic policy.
4. Symthaea active-inference policy.

The input digest must bind the subject pair, key, meter, orchestration, renderer,
soundfont, seed set, and hard theory constraints. Every arm for a trial must
carry the same digest.

## Evidence channels

### Structural

- hard constraint validity;
- obligation fulfilment;
- voice-leading violations;
- motif-return similarity;
- tonic return.

### Perceptual

- return recognition;
- perceived development instability;
- whether the recapitulation feels earned;
- blinded preference.

### Workflow

- keep;
- edit;
- reject;
- time to a committed version.

No channel is a substitute for another. A structurally valid score is not
necessarily preferred, and preference does not waive structural failures.

## Minimum descriptive success gate

The Symthaea arm must:

1. have at least eight paired frozen trials, with the same trial count in all
   four arms;
2. preserve every hard structural constraint, produce zero voice-leading
   violations, return to tonic in every Symthaea trial, and retain mean
   motif-return similarity of at least 0.95;
3. fulfil every registered formal obligation, with zero-obligation records
   treated as invalid rather than vacuously complete;
4. have complete paired observations for the channel being claimed;
5. exceed both fixed and random-valid baselines by at least 0.05 on a bounded
   rate, or reduce time to commitment by at least 10 seconds; and
6. remain within 0.02 of the heuristic on bounded rates, or within 5 seconds of
   the heuristic on time to commitment.

The heuristic is therefore part of the gate, not merely a reported comparator.
A tiny descriptive advantage or one favorable trial cannot establish success.
Passing this gate still does not establish statistical significance. A null or
negative result closes the experiment honestly and is not grounds for changing
the metric, exclusions, thresholds, or seed set after listening.

## Analysis discipline

- Freeze trial inputs and policy versions before listening data is collected.
- Keep raw records.
- Run `validate_experiment` before summarizing.
- Report every arm and every evidence channel.
- Separate descriptive summaries from statistical tests.
- Do not select seeds after hearing the outputs.
- Record exclusions and missing listener responses.
- Reject missing motif-return measurements, partial perceptual records,
  cross-arm completeness or listener-count mismatches, contradictory workflow
  dispositions, non-finite values, and fulfilled counts greater than registered
  obligations.
- Retain PieceRecipe, its reported reproduction gaps, and artifact digests for
  every rendered trial.
- Treat the gate as descriptive readiness only; pre-specify the inferential
  model and multiplicity correction before collecting listener data.

## Implementation map

- `symthaea_music_theory::profile_score`: symbolic observation.
- `symthaea_muse::closed_loop`: prediction-error evidence.
- `symthaea_muse::cognitive_selection`: alternative selection.
- `symthaea_muse::cognitive_experiment`: records, validation, summaries, gate.
- `PieceRecipe`: cognitive decision, selected alternative, artist response, and
  artifact provenance.

## V5 adaptive-calibration holdout

The adaptive predictor has a separate frozen question:

> Does context-sensitive calibration predict measured symbolic action effects
> more accurately than the unchanged hand-authored action prior on observations
> that were not yet visible to the model?

Each `AdaptiveHoldoutRecord` must be prequential: its calibration evidence is
captured before its own observed outcome is admitted to the model. The record
binds a unique trial ID, evaluation order, frozen-input digest, training-history
count, context, sufficient calibration evidence, and observed symbolic delta.

Run `validate_adaptive_holdout` before analysis. Reject duplicate trials or
orders, context mismatches, non-finite values, malformed digests, and evidence
sample counts greater than the training history declared at prediction time.

The frozen descriptive gate requires:

1. at least sixteen independent holdout records;
2. no detected prequential leakage or invalid evidence;
3. mean absolute error at least 0.02 lower than the hand-authored prior; and
4. no individual prediction channel worsening by more than 0.01 MAE.

This comparison must not be trained or thresholded after inspecting holdout
results. A pass establishes calibration improvement only. The four-arm Sonata
experiment remains necessary for any claim about musical usefulness, listener
preference, or artist workflow.

Implementation: `symthaea_muse::adaptive_experiment`.
## V6 policy boundary

V6 separates canonical theory validation, parameterized outcome prediction,
and musical utility policy. Preview learns symbolic outcomes from all valid
alternatives; explicit commit records artist disposition without silently
training preference. See `COGNITIVE_POLICY_BOUNDARY.md`.

V7 replaces the constructed Studio inference with a seeded pre-target temporal
session. This changes the policy input, not the standards for prediction or
musical-usefulness evidence.

## V7 temporal mechanism ablation

This experiment asks whether HDC/CfC temporal state changes the FEP trajectory,
not whether it improves music.

For every frozen Sonata input, run a paired control and treatment with identical
score, seed, FEP RNG, goals, observation windows, feedback, and network
architecture. The control sets temporal blend to zero; the treatment uses the
frozen positive blend. The raw symbolic observations must remain identical.

The descriptive gate requires sixteen unique pairs, valid trace evidence, no
configuration drift except temporal blend, mean per-channel temporal sensory
delta of at least `0.005`, and at least one divergent paired action. A pass is
mechanistic evidence only. A null result means the current HDC/CfC modulation
did not materially affect action selection under the frozen setup and should be
reported without tuning the threshold after observation.

Implementation: `symthaea_muse::cognitive_session_experiment`.
Detailed claim boundary: `TEMPORAL_COGNITION.md`.

## V8 confirmatory evidence path

V8 replaces hand-entered aggregate evidence with a frozen manifest, balanced
blinded schedules, individual anonymous response blocks, private codebook
compilation, and paired confirmatory inference.

The confirmatory set contains every fixture declared confirmatory in the
manifest, with related musical families prohibited from crossing the pilot
boundary. Listener endpoints require at least twelve included listeners per
fixture. The analysis plan must bind the exact manifest and schedule digests,
primary endpoints, alpha, complete fixture count, bootstrap count,
randomization count, and required number of passing endpoints before outcomes
are observed.

Each primary endpoint must clear superiority margins against fixed and
random-valid policies and a non-inferiority margin against the hand-authored
heuristic. Paired confidence intervals and one-sided sign-randomization tests
are computed at the fixture level, with Holm correction across all primary
comparisons.

See `EMPIRICAL_COGNITION_EVALUATION.md` and the modules
`experiment_manifest`, `blinded_study`, `study_evidence`, and
`confirmatory_analysis`.
