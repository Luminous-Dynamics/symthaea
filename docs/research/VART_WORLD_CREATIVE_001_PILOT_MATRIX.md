# VART-WORLD-CREATIVE-001 — Noncanonical Pilot Matrix

This matrix is for plumbing validation only. Pilot outcomes MUST NOT be used to tune or justify confirmatory thresholds, scientific claims, or policy selection.

## Pilot cells

| Cell | Policy | Fixture | Purpose |
|---|---|---|---|
| P1 | FULL | ordinary | close the complete hypothesis -> outcome -> calibration chain |
| P2 | RANDOM_VALID | ordinary | verify same physically admitted candidate surface and random-selection provenance |
| P3 | HEURISTIC | ordinary | verify deterministic baseline isolation |
| P4 | FULL | PrettyTrap | ensure perceptual preference cannot override physical validity |
| P5 | RANDOM_VALID | PrettyTrap | ensure random baseline is still bounded by physical admission |
| P6 | FULL | MemoryTrap | verify committed/counterfactual/historical provenance separation across revisit |
| P7 | NO_EXPERIENCE | ordinary | verify ablation removes journey evidence without changing unrelated authority gates |
| P8 | NO_COUNTERFACTUALS | ordinary | verify ablation semantics and candidate/application closure |

## Required per-trial evidence

Every completed pilot trial should export, at minimum:

- source and environment identity;
- world/fixture identity and seed;
- policy/ablation identity;
- pre-revision world version/digest;
- `ExperienceEpisode` where the policy permits one;
- `RevisionHypothesis` with creation time/order before mutation;
- candidate set and physical-admission evidence;
- selected proposal and selection evidence;
- typed application receipt;
- post-revision world version/digest;
- revisit observation;
- `RevisionOutcome`;
- `CreativeTrial` hash closure;
- calibration-ledger append;
- drop/missing/abort counts.

## Pilot success means plumbing works, not that Symthaea is better

A successful pilot establishes only that the experimental machinery can faithfully compare policies later. It must not be summarized as evidence that FULL outperforms a baseline, improves worlds, learns a causal model, or generalizes.

Any threshold discovered by looking at pilot outcome values is exploratory and must be explicitly discarded or separately justified before the confirmatory freeze.
