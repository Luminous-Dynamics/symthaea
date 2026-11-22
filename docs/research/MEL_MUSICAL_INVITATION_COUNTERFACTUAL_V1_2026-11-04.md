# MEL-MCTX-001 — Musical invitation counterfactual fixture v1

**Status:** synthetic symbolic fixtures and evaluator; no waveform perception or musician-study result claimed.

## Research question

Can a policy use collaborator intent and musical context to distinguish an invitation to take the floor from an acoustically similar pause, while remaining invariant to irrelevant changes such as instrument timbre?

## Six test types

1. Explicit-intent change must change action: keep the local cue, preceding harmonic context, phrase function, performer role and instrument fixed; change collaborator intent. Example: the same cadential rest invites a response in one condition but retains the floor in the other.
2. Preceding-harmonic-context change must change action: keep the local cue, phrase function, explicit collaborator-intent label, performer role and instrument fixed; change only the preceding harmonic progression/context. Example: an otherwise identical cadential rest follows a resolved dominant-to-tonic cadence in one member and an open half-cadence in the other. The annotation must specify why the valid next action differs.
3. Phrase-function change must change action: preserve the local cue, preceding harmonic context, collaborator intent, performer role and instrument; change only the annotated phrase function (for example, a phrase-internal breath versus a solo handoff).
4. Performer-role change must change action: preserve the local cue, harmonic context, phrase function, collaborator intent and instrument; change only the declared ensemble role.
5. Surface change must preserve action: keep cue, preceding harmonic context, phrase function, collaborator intent and performer role fixed; change only the surface instrument. The set includes both a musical continuation case and a technical-dropout case.
6. Ambiguous valid-action set: preserve a set of musically defensible actions, rather than forcing one canonical response.

## Evaluator

The Rust evaluator validates the family contract and scores a policy's predictions:
- per-member valid-action support;
- pairwise action change for context counterfactuals;
- pairwise action invariance for surface-only counterfactuals;
- set membership without a forced exact action for ambiguous cases.

The score is intentionally not an audio classifier score. The supplied v1 fixtures are symbolic and synthetic. A later corpus must bind exact source audio, renderer, sample rate, performance-time map, annotation provenance and permitted input prefix.



## Expanded symbolic corpus v1

The checked-in fixture now contains 11 families and 21 annotated members: three additional collaborator-intent pairs (pickup, motif completion and dramatic silence), one phrase-function pair, one performer-role pair, a technical-dropout surface-invariance pair, and the original cadence, harmonic-context, instrument-invariance and ambiguous-fermata cases. The evaluator still scores member support, context sensitivity, surface invariance and ambiguity coverage separately; no composite score is introduced. The fixture remains symbolic and synthetic: local-cue signatures are controlled labels, not evidence of waveform equivalence.

## Falsifiers

- Same local cue but different collaborator intent yields the same action in a context-change family.
- Same intent and musical role but instrument surface changes yield a different action in a surface-invariance family.
- A policy is forced to choose one label when the valid-action set explicitly contains more than one defensible action.
- A missing, corrupted or mismatched context field silently defaults to a single action instead of unresolved/abstain.

## Next dataset stage

Use these fixtures as unit tests only. Expand into multi-renderer symbolic renderings and musician-annotated real sessions. Use at least three independent musician annotations per real item, preserve disagreement, and split by performer, composition and ensemble. Keep synthetic and real cohorts separate.

No claim about subjective musical understanding, consciousness or real-world ensemble superiority is authorized by this v1.


## Aggregate metrics and frozen baseline controls

The Rust evaluator exposes `score_musical_context_counterfactual_set_v1`, with separate output fields for:
- member-level valid-action support;
- counterfactual sensitivity on intent-changing pairs;
- invariance on surface-only pairs;
- valid-action coverage on ambiguous cases;
- per-family scores and exact denominators.

There is deliberately no composite score. The checked-in `mctx-baselines-v1.json` includes:
1. `LOCAL-CUE-ONLY-001`: maps each local cue to one action and therefore fails both intent-changing and harmonic-context-changing pairs.
2. `INSTRUMENT-SURFACE-ONLY-001`: changes action from the instrument surface and therefore fails both irrelevant-surface invariance pairs.
3. `EXPLICIT-CONTEXT-RULE-SANITY-001`: a hand-written symbolic rule sanity check that uses explicit intent and preceding harmonic context and satisfies the toy fixture. This is not a learned model, an audio system, or a competitive result.

These controls demonstrate that the paired fixture design distinguishes the behaviours it claims to measure. They do not establish real-world model performance.


## Reproducible runner

Run the frozen policy controls and emit machine-readable aggregate JSON:

    cargo run -p symthaea-muse --example mel_context_counterfactual

The runner validates the counterfactual corpus, computes the three policy reports, and fails closed if any metric differs from the values frozen in `mctx-baselines-v1.json`. It emits member valid-action support rate, intent-change sensitivity, irrelevant-surface invariance, ambiguous valid-action coverage and per-family scores. No composite score is generated.
