# Symthaea–Muse Cognitive Policy Boundary

Version: V6

## Purpose

V6 corrects a conceptual error in the earlier adaptive Sonata path: prediction
accuracy is not musical desirability. The system now has three explicit and
independently auditable authorities.

1. `symthaea-music-theory::validate_score` determines whether a completed score
   is admissible under the configured symbolic rules.
2. `AdaptiveOutcomeModel` predicts the measurable effect of one parameterized
   intervention.
3. `MusicalPolicyPreference` chooses which valid predicted effect best serves
   the active formal promise and frozen musical preference.

No candidate receives musical credit merely because the world model predicted
it accurately. Prediction error is computed after symbolic measurement and is
retained only as world-model evidence.

## Parameterized intervention evidence

An action label is not a sufficient learning key. Every V6 candidate records an
`InterventionDescriptor` containing:

- the symbolic action and concrete transformation strategy;
- source and target formal regions;
- the target obligation class;
- pitch and rhythmic transformation parameters;
- transformation strength;
- bounded baseline motif, tension, and density buckets;
- affected note count and score fraction.

The outcome model uses a conservative hierarchy:

1. exact parameterized intervention context;
2. transformation-strategy fallback;
3. symbolic-action fallback;
4. unchanged hand-authored prior.

Persisted V1 adaptive models migrate without inventing strategy or intervention
samples. Their action and coarse-context sufficient statistics are retained.

## Canonical theory validation

Every candidate carries a `TheoryValidationReport` from the theory crate. The
report includes versioned rule identifiers, severity, affected note indices,
time spans, and messages. The first configuration checks:

- score metadata and bounded note data;
- monophony in melodic, bass, and countermelodic voices;
- voice crossing;
- strong-beat consonance;
- parallel perfect motion;
- excessive melodic leaps;
- final tonic arrival.

Warnings remain visible. Any fatal issue makes the candidate ineligible. The
Preserve contract remains a separate eligibility requirement.

## Preview and commitment

`POST /api/cognitive/sonata-return` is a preview operation.

- It generates and validates all alternatives.
- It predicts every alternative separately.
- It selects through the frozen musical policy.
- It renders one recommendation.
- It updates the symbolic world model from every valid measured alternative.
- It does **not** record artist acceptance and does **not** train preference.

Artist disposition is recorded separately through:

`POST /api/cognitive/sonata-return/{id}/commit`

with `Accepted`, `Edited`, or `Rejected`. The narrow V6 endpoint can accept or
edit only the rendered alternative. It records evidence in the recipe but does
not yet fit a preference model from that evidence.

## Required evaluation separation

World-model evaluation and policy evaluation must never share one success
number.

### World model

Evaluate on frozen prequential or held-out interventions using:

- per-channel MAE;
- uncertainty calibration or a proper probabilistic score;
- action-only versus strategy versus exact-context ablations;
- unseen seeds and unseen thematic material;
- sample counts declared at prediction time.

### Musical policy

Evaluate only among theory-valid, Preserve-compliant candidates using:

- obligation fulfilment and residual deadline pressure;
- blinded artist or listener preference;
- time to committed version;
- edit distance after recommendation;
- regret against the best measured candidate in the same frozen batch.

A better world model does not establish a better policy. A preferred policy
does not establish calibrated prediction.

## Honest limitation

The Studio request still constructs a bounded `MusicInferenceResult` from the
request state. V6 therefore proves a clean theory/world-model/policy/product
boundary, not that a temporally evolving Symthaea FEP/HDC/LTC session selected
the intervention. Replacing that constructed result with a reproducible live
cognitive trajectory is the next research milestone.

## V7 relationship to temporal cognition

V7 changes where the symbolic proposal originates, not the V6 authority order.
The terminal action now comes from a seeded temporal HDC/CfC/FEP session, but it
still cannot bypass canonical theory validation, substitute prediction accuracy
for utility, or learn artist preference from preview behavior. Temporal-session
evidence is attached to the recipe before candidate validation and policy
ranking. See `TEMPORAL_COGNITION.md`.
