# Symthaea–Muse Cognitive Integration

## Status

Active integration contract with additive implementations in:

- `src/cognitive_bridge.rs` — cognitive observation, action, and prediction;
- `src/closed_loop.rs` — score-side observation and prediction-error recording;
- `src/cognitive_selection.rs` — transparent selection among valid alternatives;
- `src/cognitive_experiment.rs` — frozen four-arm experiment records and gate;
- `src/cognitive_session.rs` — seeded temporal HDC/CfC/FEP session evidence;
- `src/cognitive_session_experiment.rs` — paired temporal-influence ablation;
- `src/studio_contract.rs` — explicit Preserve/Change translation;
- `src/piece_recipe.rs` — deterministic recipe, evidence, and artist response;
- `symthaea-music-theory::cognitive_analysis` — symbolic measurement channels;
- `symthaea-music-theory::sonata` — obligation-bearing long-form planning.

## Purpose

Muse already contains two valuable systems:

- a deterministic symbolic composition engine with explicit form, harmony,
  motifs, identity grammar, and performance realization;
- a Symthaea-driven live system with cognitive state, active inference,
  self-listening, and temporal dynamics.

The goal is not to merge them indiscriminately. The goal is to give Symthaea a
clear, testable responsibility in the symbolic product path.

## Division of responsibility

### Music theory

`Symthaea-music-theory` determines what is musically valid and represents exact
musical facts:

- pitch, spelling, meter, rhythm, harmony, and voice leading;
- motifs, form, cadence, and structural transformations;
- prospective compositional obligations;
- deterministic scores and analysis evidence.

### Symthaea

Symthaea determines what is meaningful to do next:

- remember what the piece has promised;
- estimate what the listener now expects;
- choose among valid compositional actions;
- predict the expected musical effect;
- compare prediction with score, render, and artist response;
- update future decisions.

### Muse

Muse bridges those layers and realizes the result:

- translate cognitive decisions into constrained edit contracts;
- ask the theory engine for valid alternatives;
- render score and performance;
- measure what survived the renderer;
- preserve decision traces and provenance.

## Canonical loop

1. Observe the symbolic score, current render, artist intent, active selection,
   and outstanding obligations.
2. Encode a `SymbolicMusicObservation` without discarding the exact score.
3. Run a seeded sequence of HDC/CfC observations and committed FEP cycles,
   then obtain the terminal `MusicInferenceResult`.
4. Convert the result into a `SymbolicActionProposal` with explicit scope and
   preserved invariants.
5. Realize alternatives through the theory engine.
6. Predict directional effects before rendering.
7. Measure symbolic and perceptual effects after rendering.
8. Record channel-specific prediction error.
9. Let the artist keep, edit, reject, or branch the proposal.
10. Store the complete `CognitiveDecisionTrace` with the piece version.

## Why the bridge is narrow

Symthaea must not compensate for missing theory. A cognitive state cannot make
an unresolved tendency tone valid, invent notation spelling, or replace
culture-specific rhythmic grammar.

The bridge therefore produces actions such as:

- develop motif;
- introduce contrast;
- increase harmonic instability;
- modulate to a related key;
- increase density;
- strengthen cadence;
- add a counterline;
- return opening material;
- thin texture.

Each action carries semantic invariants. A theory-aware caller remains
responsible for generating valid notes.

## Prospective memory

The theory crate's `ObligationLedger` is the common language for long-range
promises.

Examples:

- return the second subject in the tonic;
- restore the opening motif after fragmentation;
- resolve a raised fourth;
- reserve the brass entrance until the climax;
- arrive at the home key before the coda.

The cognitive observation records each pending obligation's identifier, kind,
priority, deadline, and overdue state. Action selection therefore uses typed
promise semantics: a motif-return obligation can request opening material, a
key-arrival obligation can request modulation, and a cadence obligation can
request cadential strengthening. Obligation pressure still affects urgency, but
it no longer makes an unrelated action merely happen faster.

## Prediction channels

The first bridge predicts four directional effects:

- tension;
- density;
- familiarity;
- tonal displacement.

These are deliberately separate. They must not be collapsed into a single
quality or consciousness score.

Future channels may include:

- motif recognition;
- cadence finality;
- orchestration transparency;
- rhythmic vitality;
- emotional-intent fit;
- renderer fidelity;
- artist preference.

Each channel requires a defined measurement method and uncertainty statement.

## First closed-loop experiment

### Question

Can Symthaea choose a development and recapitulation policy that preserves
motif identity, increases instability during development, and produces a
perceptually clear return?

### Arms

1. Fixed deterministic policy.
2. Random valid action selection.
3. Existing hand-authored heuristic policy.
4. Symthaea active-inference policy through `cognitive_bridge`.

### Frozen inputs

- one sonata subject pair;
- one home key and meter;
- one orchestration palette;
- fixed renderer and soundfont;
- fixed seed set;
- identical hard theory constraints.

### Measurements

Structural:

- motif identity at recapitulation;
- tonic return;
- obligation fulfilment;
- cadence validity;
- voice-leading violations;
- density and tension trajectories.

Perceptual:

- whether listeners recognize the return;
- whether development sounds meaningfully less stable;
- whether the recapitulation feels earned;
- whether the result is preferred in a blind comparison.

Workflow:

- keep rate;
- edit rate;
- rejection reason;
- time to a committed version.

### Success condition

The descriptive gate requires at least eight paired frozen trials per arm.
Symthaea must preserve every hard constraint, produce zero voice-leading
violations, return to tonic in every trial, retain mean motif-return similarity
of at least 0.95, fulfil every non-empty registered obligation set, and exceed
both fixed and random-valid baselines by a pre-registered
practical margin on at least one fully observed channel, and remain
non-inferior to the hand-authored heuristic. This is only a readiness gate;
inferential statistics, blinded-listener analysis, and multiplicity control
remain separate. A null or negative result is acceptable and should be
published.

## Determinism and provenance

Adaptive composition must remain reproducible.

A piece version influenced by Symthaea should record:

- exact symbolic input and initial `MusicalState`;
- cognitive observation;
- active-inference result;
- action proposal;
- preserved invariants;
- generated alternatives;
- selected alternative;
- predicted outcome;
- observed outcome;
- prediction error;
- artist response;
- model and policy versions;
- source revisions, renderer version, artifact digests, and environment digest;
- structured manual edit operations rather than edit identifiers alone.

`PieceRecipe::reproduction_gaps` must state which external identities or
digests are still absent. Hidden mutable cognitive state and convenient version
fallbacks must not become undocumented sources of score changes.

## Non-goals

This integration does not claim:

- that Symthaea is conscious;
- that IIT Phi measures musical quality;
- that active inference replaces music theory;
- that one scalar can judge a piece;
- that artist behavior may be learned without consent;
- that imported external-platform libraries may be used as training input.

## Implemented integration steps

1. `SymbolicActionProposal` translates into a validated Studio
   Preserve/Change contract and carries the typed obligation that drove it.
2. Sonata exposes exact section boundaries and pending formal obligations. A
   separate score-side verifier now resolves those obligations from observable
   motif, tonal-anchor, climax, and cadence evidence rather than constructor
   branch execution.
3. The obligation ledger reports priority-weighted deadline pressure, while the
   Muse bridge maps actionable obligation kinds to compatible symbolic actions.
4. Symbolic score-region measurements include sustained carry-in notes, count
   only true attacks as density, and evaluate vertical tension across sounding
   overlaps. Tonal displacement remains separate from the tension composite.
5. `closed_loop` compares baseline and candidate scores and writes versioned
   profiles, observed directional outcomes, and channel-specific prediction
   errors into the corresponding recipe decision.
6. `cognitive_selection` recommends only alternatives satisfying hard theory
   and the Studio Preserve contract. Its lexicographic ordering keeps formal
   obligations and channel errors visible rather than hiding them in one score.
7. `PieceRecipe` schema v3 retains the exact initial `MusicalState`, source and
   renderer evidence when available, explicit reproduction gaps, cognitive
   traces, selected alternatives, structured manual edits, symbolic evidence,
   and artist responses. Schema-v1 and schema-v2 recipes remain readable.
8. Studio keepers publish collision-resistant artifact directories through an
   atomic rename and update the JSONL index through locked atomic replacement.
   Legacy flat-file keepers remain readable.
9. `cognitive_experiment` rejects empty obligation sets, impossible fulfilment
   counts, non-finite or out-of-range values, partial listener outcomes,
   mismatched listener coverage, contradictory workflow states, and incomplete
   arm pairing. Its gate requires eight paired trials, complete structural
   return evidence, practical effect margins, complete channel coverage, and
   heuristic non-inferiority.

## V5 adaptive implementation

10. `cognitive_bridge` groups compatible obligation demands into transparent
    votes, selects one action deterministically, and records both supporting
    and explicitly deferred promises.
11. `adaptive_prediction` learns online sufficient statistics for symbolic
    action outcomes by section, style, form, meter, and texture. Exact-context
    and action-level estimates shrink toward the hand-authored prior and retain
    uncertainty and sufficient moments in every recipe decision.
12. `symthaea-music-theory::motif_return` provides transformation-aware pitch,
    interval, contour, rhythm, and coverage evidence for literal, transposed,
    inverted, augmented, diminished, fragmented, and restored returns.
13. `sonata_intervention` generates a fixed family of return candidates,
    independently verifies their formal promises, preserves the score skeleton
    outside the target region, and ranks only valid alternatives.
14. Studio exposes the narrow `/api/cognitive/sonata-return` preview path,
    persists the calibration model explicitly, and retains prediction,
    selection, measurement, preview, and reproduction evidence in `PieceRecipe`.
15. `adaptive_experiment` freezes a prequential holdout gate against the
    unchanged hand-authored prior and rejects evidence-count leakage.

## V6 policy boundary

V6 separates canonical theory validation, parameterized outcome prediction,
and musical utility policy. Preview learns symbolic outcomes from all valid
alternatives; explicit commit records artist disposition without silently
training preference. See `COGNITIVE_POLICY_BOUNDARY.md`.

## V7 temporal cognition

V7 replaces the Studio endpoint's constructed inference result with a seeded,
auditable pre-target trajectory through score-window HDC encoding, CfC temporal
evolution, and committed active-inference cycles. The target recapitulation is
not observed before the proposal. Every selected action becomes the cause
assigned to the following observation, enabling temporal-difference updates.
The recipe and endpoint retain the FEP seed, goals, complete trajectory,
learning statistics, and session fingerprint.

The paired temporal ablation holds the raw symbolic stream, seed, FEP RNG,
goals, windows, feedback, and architecture fixed while changing only whether
CfC output may influence the FEP observation. Its gate tests mechanistic
influence, not musical benefit. See `TEMPORAL_COGNITION.md`.

## Next implementation steps

1. Compile and run the complete V1–V7 series in the parent workspace and land a
   stabilization-only patch for any type, lint, or behavior failures.
2. Generate exact-replay fixtures on each supported CI platform and preserve
   platform identity with the evidence.
3. Run the frozen sixteen-pair temporal ablation without post-hoc threshold
   changes.
4. Run the V5 prequential calibration holdout and the four-arm Sonata study as
   separate evaluations.
5. Add renderer-derived observations only as a separate, uncertainty-labelled
   channel; do not silently replace the symbolic stream.
6. Expand temporal cognition beyond Sonata only after the mechanism, prediction,
   and usefulness questions each receive an honest result.

## V8 empirical evaluation

V8 freezes the first confirmatory evidence path rather than expanding cognitive
authority. It adds group-safe pilot/confirmatory manifests, committed private
randomization keys, arm-free public schedules, raw listener and artist blocks,
private evidence compilation, paired bootstrap intervals, sign-randomization
tests, Holm correction, and claim-safe reporting.

The V7 temporal mechanism, V5 adaptive prediction, and four-arm usefulness
questions remain independent. `cognitive_evidence_report` intentionally has no
single overall success field. See `EMPIRICAL_COGNITION_EVALUATION.md`.
