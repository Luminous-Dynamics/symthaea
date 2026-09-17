# SYM-RSI-001D — Grounded Dreaming Beyond Exact Replay

**Status:** preregistered extension design only. No D-vs-C result is claimed by this document.

## Purpose

SYM-RSI-001 established the bounded mechanism test for exact historical replay (C vs A) and defined grounded dreaming (D vs C) as a primary contrast. This extension gives D vs C its own untouched evaluation lineage so dream-policy design cannot be tuned against the C-vs-A fresh outcomes.

The extension inherits only the already-frozen SYM-RSI-001 training substrate:
- the canonical training replay worlds (seeds 1-8 in each domain),
- the canonical eight-policy candidate family,
- the frozen arm-C training selection receipt,
- the grounded dream model trained exclusively from those observed training transitions.

No C-vs-A fresh outcome is an input to arm D.

## Frozen evaluation partitions

For each of the three existing deterministic fixture domains:

| Purpose | Seeds |
| --- | --- |
| inherited training replay | 1-8 |
| dream verification replay | 301-304 |
| fresh D-vs-C execution | 401-404 |
| OOD follow-up | 1201-1204 |

The 301-304, 401-404, and 1201-1204 partitions are disjoint from every SYM-RSI-001 seed. Unit tests and ordinary CI must not execute these partitions.

## Arm definitions for this extension

**C — exact replay selected policy.** The exact policy selected by the frozen SYM-RSI-001 training replay receipt. It is not reselected on any SYM-RSI-001D outcome.

**D — replay plus grounded dreaming.** Arm C wrapped by the frozen grounded dream model. The dream model may alter action choice only through predictions produced from learned transition memory.

D's frozen v2 action-scoring rule is:

`mean predicted task quality across 5 legality-aware deterministic action perturbations - 0.10 × predicted task-regression probability`

D overrides the base C action only when that score exceeds C's score by more than `0.01`. A perturbation that is illegal in the current state falls back to the original legal action rather than being simulated as if executable. Task-regression probability is the fraction of those five model predictions whose task quality is below the current state's quality. Predicted task quality is read from the task-quality channels of the model-generated outcome representation; the older generic Φ/magnitude proxy is retained only as a diagnostic and is not D's optimization target.

D is forbidden from:
- calling the true fixture transition function during action selection,
- training on verification, fresh, or OOD outcomes,
- inserting generated predictions into ExperienceTree,
- treating generated predictions as empirical/replay evidence,
- increasing epistemic confidence from an unvalidated dream prediction.

Executed D actions may of course produce new recorded evidence through the normal runner after the environment actually returns an outcome.

## Verification-replay gate

Before any fresh 401-404 execution:

1. Build a replay corpus on seeds 301-304 using only the frozen canonical fixed-policy collectors.
2. Freeze that corpus before evaluating D.
3. Evaluate the already-frozen C and D policies against the exact corpus.
4. Require C and D to have 100% historical action support and recorded terminal completion.
5. Require D mean solution quality to remain within 0.02 of C in every domain and in the macro-average.
6. Require at least one D decision to differ from C. If dreaming never changes an action, the extension terminates as **NoDreamIntervention** rather than consuming fresh seeds.
7. Any generated-evidence promotion, safety violation, authority-boundary violation, lineage mismatch, missing world, duplicate world, or unsupported action blocks fresh execution.

The verification gate is a screening gate, not the primary result.

## Fresh D-vs-C analysis

Primary evaluation uses exactly 12 paired runs: seeds 401-404 across the three frozen domains.

Primary quality is domain-balanced:
- compute D-C mean quality delta separately for each domain,
- macro-average the three domain deltas with equal domain weight,
- no domain may degrade by more than 0.02,
- the macro-average may not degrade by more than 0.02.

A result is **PositiveUnderProtocol** only if:
- all integrity, provenance, safety, and authority checks pass,
- D actually overrides C on at least one fresh decision,
- every domain and the macro-average satisfy quality non-inferiority,
- macro quality delta is strictly greater than 0.

Evaluator calls, dream-model simulations, override rate, and model-prediction counts are reported separately. This protocol does not call D more compute-efficient merely because environment calls are unchanged; model-simulation cost remains explicit rather than silently treated as free.

With only 12 deterministic/seed-controlled pairs, no asymptotic p-value is manufactured.

## OOD follow-up

Seeds 1201-1204 are untouched until the fresh D-vs-C receipt is frozen. OOD evaluation is secondary and cannot change the primary D-vs-C disposition or retrain the dream model.

## Claim boundary

A positive result supports only:

> Under the frozen SYM-RSI-001D fixture domains and protocol, a learned counterfactual dream policy improved fresh solution quality beyond the exact-replay-selected policy while respecting the preregistered non-inferiority and epistemic boundaries.

It does not by itself establish open-ended RSI, universal improvement, consciousness, or monotonic improvement outside the tested distributions.
