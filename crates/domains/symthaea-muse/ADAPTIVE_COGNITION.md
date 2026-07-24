# Adaptive Musical Cognition V5

## Purpose

V5 asks a narrower question than “does Symthaea compose better music?”:

> Can the system learn more accurate, context-sensitive predictions of the
> symbolic effects of its own theory-valid actions while preserving artist and
> theory authority?

The adaptive layer does not generate arbitrary notes, relax hard constraints,
or learn listener preference. It calibrates the predicted directional effects
of an already named symbolic action.

## Authority boundary

- `symthaea-music-theory` owns valid scores, Sonata plans, transformed motif
  evidence, and independent obligation verification.
- `cognitive_bridge` observes the piece, arbitrates competing promises, and
  proposes an inspectable action.
- `adaptive_prediction` estimates the likely symbolic outcome of that action in
  a declared context.
- `sonata_intervention` generates a small deterministic candidate family,
  verifies every candidate, and ranks only theory-valid Preserve-compliant
  alternatives.
- Studio exposes the result as a preview. Artist acceptance and musical quality
  remain separate from symbolic outcome calibration.

## Context model

A calibration context contains:

- symbolic action;
- formal section;
- style name;
- form name;
- meter;
- coarse texture band.

The model stores online sufficient statistics rather than an opaque latent
state. Exact-context estimates shrink toward action-level evidence, which in
turn shrinks toward the hand-authored prior. Sparse contexts therefore cannot
immediately overwrite the original expectation.

Every prediction records:

- model version;
- exact context;
- evidence source;
- exact-context and action-level sample counts;
- sufficient moments;
- original prior;
- calibrated prediction;
- per-channel uncertainty.

## Obligation arbitration

Multiple promises can request different actions at the same moment. V5 groups
compatible demands into action votes and orders them by:

1. overdue status;
2. priority-weighted urgency;
3. deadline;
4. stable action order.

The selected action records its driving and supporting promises. Lower-ranked
incompatible promises are recorded as deferred, not silently discarded.

## Transformation-aware thematic evidence

Literal pitch equality is not a sufficient model of musical return. The theory
crate now compares melodic regions under:

- literal return;
- transposition;
- inversion;
- augmentation;
- diminution;
- fragmentation;
- restoration.

Evidence retains pitch, interval, contour, rhythm, coverage, expected
transformation, and detected transformation. Sonata obligation verification
uses this score-side evidence rather than treating constructor progress as
proof of fulfilment.

## Narrow Studio path

`POST /api/cognitive/sonata-return` performs one bounded intervention:

1. compose a plan-bearing Sonata;
2. optionally perturb the primary return as a frozen negative control;
3. identify the formal return obligation;
4. produce and calibrate a typed cognitive proposal;
5. generate five deterministic return alternatives;
6. independently verify obligations and Preserve constraints;
7. rank valid candidates;
8. render and retain the selected preview with a complete recipe;
9. update symbolic-outcome calibration only after measurement.

This endpoint is intentionally separate from ordinary candidate generation. It
must earn broader product authority through the frozen experiments.

## Learning restrictions

The model may learn measured symbolic deltas. It must not infer or update from:

- unmeasured alternatives;
- non-finite evidence;
- listener identity;
- imported taste libraries;
- hidden manual edits;
- renderer measurements presented as symbolic measurements;
- an observation before its prediction has been recorded.

Model persistence is explicit and versioned at
`data/muse-adaptive-outcomes-v1.json`.

## Frozen evidence gates

Two independent gates remain necessary:

1. The four-arm Sonata experiment compares fixed, random-valid, heuristic, and
   Symthaea policies on structural, perceptual, and workflow outcomes.
2. The adaptive holdout experiment compares calibrated predictions with the
   unchanged hand-authored prior on frozen prequential records.

The adaptive gate requires at least sixteen holdout observations, rejects
sample-count leakage, requires a mean absolute error reduction of at least
0.02, and permits no outcome channel to regress by more than 0.01. Passing it
is evidence of improved calibration only—not improved music, consciousness, or
listener preference.

## Remaining limitations

- The online model is descriptive and does not establish causal effects.
- Contexts remain coarse and can alias musically distinct situations.
- Symbolic measurements remain proxies for perceived tension and familiarity.
- The Studio endpoint now separates preview from explicit accept/edit/reject
  commitment, but no preference model is trained from those decisions yet.
- Compilation, linting, and listening evaluation must be completed in the full
  workspace before V5 is treated as integrated.
## V6 policy boundary

V6 separates canonical theory validation, parameterized outcome prediction,
and musical utility policy. Preview learns symbolic outcomes from all valid
alternatives; explicit commit records artist disposition without silently
training preference. See `COGNITIVE_POLICY_BOUNDARY.md`.

V7 replaces the constructed Studio inference with a seeded pre-target
HDC/CfC/FEP session while preserving this authority boundary. Temporal
mechanism, prediction calibration, and musical usefulness remain separate
experiments. See `TEMPORAL_COGNITION.md`.
