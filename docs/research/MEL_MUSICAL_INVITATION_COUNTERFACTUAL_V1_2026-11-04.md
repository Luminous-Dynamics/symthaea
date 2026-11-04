# MEL-MCTX-001 — Musical invitation counterfactual fixture v1

**Status:** synthetic symbolic fixtures and evaluator; no waveform perception or musician-study result claimed.

## Research question

Can a policy use collaborator intent and musical context to distinguish an invitation to take the floor from an acoustically similar pause, while remaining invariant to irrelevant changes such as instrument timbre?

## Three test types

1. Context change must change action: keep the local cue, phrase function, performer role and instrument fixed; change collaborator intent. Example: the same cadential rest invites a response in one condition but retains the floor in the other.
2. Surface change must preserve action: keep cue, phrase function, collaborator intent and performer role fixed; change only the surface instrument.
3. Ambiguous valid-action set: preserve a set of musically defensible actions, rather than forcing one canonical response.

## Evaluator

The Rust evaluator validates the family contract and scores a policy's predictions:
- per-member valid-action support;
- pairwise action change for context counterfactuals;
- pairwise action invariance for surface-only counterfactuals;
- set membership without a forced exact action for ambiguous cases.

The score is intentionally not an audio classifier score. The supplied v1 fixtures are symbolic and synthetic. A later corpus must bind exact source audio, renderer, sample rate, performance-time map, annotation provenance and permitted input prefix.

## Falsifiers

- Same local cue but different collaborator intent yields the same action in a context-change family.
- Same intent and musical role but instrument surface changes yield a different action in a surface-invariance family.
- A policy is forced to choose one label when the valid-action set explicitly contains more than one defensible action.
- A missing, corrupted or mismatched context field silently defaults to a single action instead of unresolved/abstain.

## Next dataset stage

Use these fixtures as unit tests only. Expand into multi-renderer symbolic renderings and musician-annotated real sessions. Use at least three independent musician annotations per real item, preserve disagreement, and split by performer, composition and ensemble. Keep synthetic and real cohorts separate.

No claim about subjective musical understanding, consciousness or real-world ensemble superiority is authorized by this v1.
