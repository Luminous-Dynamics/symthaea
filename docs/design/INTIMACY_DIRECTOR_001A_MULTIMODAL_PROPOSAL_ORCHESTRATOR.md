# INTIMACY-DIRECTOR-001A — Multimodal Proposal Orchestrator

Status: source candidate; not qualified.

## Purpose

Compose an already-admitted `FantasyStyleProposalV1` into a small set of independently vetoable multimodal proposals without granting authority between modalities.

This tranche is intentionally a planner, not an executor.

## Inputs

- current style proposal from KAMA-DIALOGUE-001J;
- exact current session/world/scene/boundary epochs;
- runtime modality capabilities;
- optional read-only independent eligibility references for proximity and touch proposals;
- optional cue requests for music, lighting, expression, proximity, and touch.

## Outputs

`IntimacyDirectorPlanV1` carries:

- exact session/world/scene/boundary binding;
- semantic prosody intent projected from the style lattice;
- admitted optional modality proposals;
- independently recorded suppressed proposals and reasons.

No raw intimate transcript content is required.

## Independent modality admission

Nonphysical optional cues require their runtime capability.

Physical proposals require both the runtime proposal capability and an opaque reference to a separately governed eligibility decision. The director cannot mint that reference.

Therefore:

```text
dialogue/fantasy style != proximity authority
dialogue/fantasy style != touch authority
music/lighting/expression != physical authority
```

A missing or rejected physical proposal does not block the rest of the plan.

## Cross-modal pacing

The director honors the style planner's pacing semantics across modalities:

- normal `Maintain` may admit neutral/deescalating/escalating optional cues subject to independent modality gates;
- `Hold` / `must_not_increase_intensity` suppress escalating optional cues;
- `SlowDown` / `must_decrease_intensity` admits only explicitly deescalating optional cues;
- `ClarifyBeforeIncreasing` suppresses escalating optional cues while still permitting neutral or deescalating cues.

A stopped fantasy session emits no style proposal upstream, so there is nothing for this director to orchestrate.

## Staleness

The director rejects a style proposal unless its session ID/epoch, world ID, scene ID/epoch, and hard-boundary epoch exactly match current state.

That prevents a previously valid multimodal style proposal from silently surviving a scene, session, or boundary change.

## Prosody

The V1 prosody intent is semantic only. It projects tenderness, playfulness, directness, verbal-intensity ceiling, and pacing from the style lattice.

It does not claim those values are calibrated TTS parameters. A later voice/prosody layer must translate and evaluate them independently.

## Tests

The integration suite covers:

- stale session rejection;
- physical proposal requiring independent eligibility;
- missing physical eligibility not blocking music/lighting;
- read-only propagation of contact eligibility references;
- Hold suppressing escalation across modalities;
- SlowDown admitting only deescalating optional cues;
- uncertainty suppressing escalation;
- capability failure remaining modality-local;
- semantic prosody projection from style.

## Nonclaims

This tranche does not establish generation quality, prosody quality, actuator authority, contact consent, motor execution, physical safety, medical functionality, human-trial authorization, or engagement optimization.
