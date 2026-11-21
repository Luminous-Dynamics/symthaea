# GGL Visual Quality-Diversity Benchmark v0.1

**Owner:** Symthaea CREATIVE-QD-001 (#7933)  
**Date:** 2026-11-19  
**Status:** pilot protocol + synthetic brief fixture; not confirmatory, not sealed, no system result claimed

## 1. Purpose and boundary

This is a visual-art extension of the existing creative quality-diversity programme. It does not create a second QD archive, renderer, critic, gallery, or reward system.

- **GGL** supplies typed programs, constraints, stable identities, and reversible edit plans.
- **Atelier** remains the generator and explore/mutate owner.
- **Canvas** remains the canonical scene/render owner.
- **Art Eye** supplies rendered-pixel measurements and perceptual features.
- **Gallery** stores opted-in artifacts and human feedback.
- **CREATIVE-QD-001 (#7933)** owns matched-budget QD selection and human-evaluation methodology.
- **SCI-GRAMMAR-001 (#8919)** supplies grammar-recovery equivalence classes, heterogeneous/no-common-grammar controls, held-out production compositions, and abstention principles.

The benchmark separates four things that must not be collapsed into one scalar: (1) hard validity, (2) stated-intent adherence, (3) descriptive visual features, and (4) human judgment.

## 2. Pilot brief corpus

`visual_qd_pilot_v0.1.json` contains 18 synthetic pilot briefs across six families:

1. geometric/ornamental;
2. icon/poster;
3. editorial illustration;
4. organic/branching;
5. abstract composition;
6. ambiguous/adversarial.

Each family has one generation, one edit, and one inverse-analysis task (the ambiguous family expects abstention or explicit conflict reporting). The manifest fixes an eight-candidate budget per system for a future matched comparison and declares the Canvas/Art Eye/Gallery owners.

**Scope limitation:** this is a machine-readable brief/contract pilot only. It does not yet include SVG/PNG stimuli, reference target programs, licensed external artwork, or human annotations. Those are separate preregistered corpus deliverables. The manifest is intentionally marked `not_materialized` and cannot be used as a confirmatory test set.

## 3. Experimental arms

For every brief, compare at equal candidate/tool budgets where task semantics permit:

1. Atelier single candidate;
2. Atelier best-of-8 with the selection rule disclosed;
3. VLM-only generation/editing;
4. VLM → typed GGL plan → deterministic Canvas transaction;
5. GGL-conditioned Atelier generation;
6. existing Creative-QD archive/selection at matched budget;
7. human artist workflow as a reference, not a machine leaderboard endpoint.

Record exact model/provider/version, prompt, seed, GGL/compiler revision, Atelier configuration, renderer/version, candidate count, selection rule, wall-clock time, tool calls, and output digests. Do not compare best-of-8 against single-candidate systems without disclosing the selection advantage.

## 4. Required metrics

### Hard validity (machine-checkable)

- parse/compile success;
- valid Canvas scene and valid SVG;
- declared primitive/element constraints;
- exact hard-constraint pass rate;
- supported/unsupported/opaque feature counts;
- deterministic same-seed output;
- edit target identification precision/recall where ground truth exists;
- protected-ID survival;
- valid-output Unintended Change Rate (UCR);
- stale-scene rejection and undo/redo integrity.

A candidate that violates a hard contract cannot become a pass by scoring well on appearance.

### Structural / program quality

- geometry residual and tolerance pass rate;
- relation-graph precision/recall;
- topology and group/identity preservation;
- compactness / MDL with reconstruction error reported as a Pareto frontier;
- grammar equivalence-class recovery on known synthetic sources;
- false-shared-grammar rate on heterogeneous controls;
- abstention and non-identifiability handling.

Never score exact source-string equality as the only grammar-recovery metric. Semantically equivalent programs can differ syntactically. For real artworks, the original construction process is generally not identifiable from pixels alone; return candidate equivalence classes and residuals, not one claimed historical authoring process.

### Descriptive visual features (not beauty scores)

Use existing Art Eye and existing SVG structural evaluation ideas. Report color, contrast, edge density, spatial layout, feature coverage, element influence, purity, coverage, compactness and locality separately. Adapt richer SVG leave-one-out metrics rather than presenting simple pixel ablation as novel. CLIP-like image-text similarity is only a diagnostic; it is insufficient to catch SVG count, color, and spatial-relation errors.

### Human evaluation

Blind, randomized, balanced pairwise judgments with ties allowed. Rate separately:

- brief adherence;
- structural correctness;
- visual coherence;
- novelty/originality;
- controllability/editability;
- overall preference (separate outcome, not a training reward by default).

Capture rater expertise, disagreement, time spent, correction count, and confidence. Analyze with mixed-effects or cluster-aware models across briefs and raters. Plan final sample size from pilot variance and a preregistered power simulation; do not treat 18 briefs as sufficient for a superiority claim.

## 5. Adversarial and negative controls

- contradictory edit: preserve every position while moving an object;
- impossible symmetry request while preserving unequal measured placements;
- non-identifiable image-to-program recovery;
- unknown or unsupported primitive;
- duplicate semantic IDs;
- malformed units or coordinate frame;
- non-finite / out-of-range geometry;
- empty or degenerate vector output;
- protected element removed, recolored, reordered, or geometrically changed;
- prompt-injection text inside SVG metadata or labels (treated as untrusted data, never instructions to the evaluator);
- randomized and human-designed ornament controls for claimed unusual mathematical encodings.

Correct behavior can be `abstain`, `request clarification`, or `return multiple equivalent programs`; forced confident output is not always success.

## 6. External comparison and SOTA position

- **VectorGym (EMNLP 2026)** evaluates Sketch2SVG, human-authored complex SVG edits, Text2SVG, and SVG captioning. Its GRPO-trained Qwen3-VL 8B baseline is reported by the authors as open-source SOTA across their multi-task benchmark and matches GPT-4o in their reported evaluation. This is a generation/understanding baseline, not a direct test of GGL's deterministic transaction layer. https://aclanthology.org/2026.emnlp-main.820/
- **Vector-Bench (July 2026)** tests 40 difficult SVG surgical repairs; the strongest of 34 evaluated endpoints reports 15.0% full-specification success and 43.7% mean repair progress. It explicitly scores protected objects and valid-output unintended changes. This is the most relevant external edit-preservation target, but it is not comparable to structured synthetic edit tests. https://arxiv.org/abs/2607.19056
- **VectorEdits (LREC 2026)** provides more than 270,000 SVG/instruction pairs for instruction-based editing. Dataset access and license terms must be reviewed before training or ingestion. https://aclanthology.org/2026.lrec-1.868/
- **SVG-Score (September 2026)** demonstrates that generic CLIP-like metrics can be insensitive to SVG errors such as incorrect counts, colors, and spatial relations, and proposes human-aligned evaluation. https://arxiv.org/abs/2609.03806

The intended advantage is not raw pixel generation alone. It is to test whether typed, executable plans improve exact edit specification, preservation, reproducibility, controllability, and interpretable inverse-grammar recovery when coupled to a strong VLM/Atelier generator. The VLM remains responsible for perceptual interpretation; GGL makes the resulting plan explicit and mechanically checkable. No superiority claim is made by this protocol.

## 7. Prospective targets (not results)

- 100% pass on all declared supported primitive/adapter fixtures;
- zero silent drops of unsupported semantics (reject or preserve opaque with provenance);
- 100% protected-ID preservation in structured transactions;
- ≥90% synthetic semantic-equivalence recovery on held-out combinations only after an independent evaluator is qualified;
- external Vector-Bench full-spec and UCR scores reported on the official benchmark, with all 40 tasks and exclusions visible;
- >60% blinded pairwise human preference with the 95% interval lower bound above 50%, only after a preregistered power plan and independent held-out replication.

Targets are aspirations. They are not observed GGL results.

## 8. Freeze and claim discipline

Before confirmatory evaluation, freeze the corpus, evaluator, render environment, task splits, seeds, model/provider versions, candidate budgets and scoring rules. Split by generating grammar, visual motif lineage, and source-artwork family rather than random crops/variants. Publish full denominators, failures, invalid outputs, abstentions, not-evaluable cases, exclusions, and digests.

This v0.1 pilot manifest is public and not sealed. It may be used to debug the protocol but not to claim generalization. A separate rights-cleared, professional-artist annotated and sequestered holdout is required for confirmatory claims.
