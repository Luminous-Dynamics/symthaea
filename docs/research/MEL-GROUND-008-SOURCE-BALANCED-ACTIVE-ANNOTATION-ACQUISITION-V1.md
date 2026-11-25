# MEL-GROUND-008 — Source-balanced active annotation acquisition v1

**Status:** deterministic planner implementation in progress; no musician annotation study  
**Date:** 25 November 2026  
**Implementation:** `tools/mel-temporal-grounding-qualifier/src/temporal_acquisition.rs`  
**Owner:** #13624  
**Consumes:** #13560/#13563 per-annotator model scores; #13617 inter-annotator reliability; #9151 sparse planner patterns; #9184 blinded metric calibration; #9509 rights and source registry

## 1. The problem: label quality is a budgeted resource

Temporal music annotation is expensive. A further listening pass can resolve a repeated-event count, clarify a gradual entrance, expose a false model absence, or simply repeat an already clear case. Uniformly annotating everything spends money on already-settled examples; adaptively annotating only model failures can overfocus on the current model's blind spots and destroy the representativeness of the test set.

The solution is not one active-learning queue. It is two independent acquisition streams with distinct statistical roles:

- **Active refinement:** deliberately samples high-value diagnostic examples to improve labels, identify model gaps and resolve task-policy ambiguity.
- **Probability audit:** samples source works and items under a frozen, known-probability design. This is the only stream eligible for unbiased performance/prevalence estimation under its sampling design.

The two streams must never be silently pooled.

## 2. Paradigm shift: uncertainty is not the same as model error

The planner classifies evidence into separate acquisition reasons:

| Band | Evidence | Next action | Interpretation |
|---|---|---|---|
| A — model failure | Low error-free coverage, high model-vs-annotator error, stable human panel | Review model output / add independent human label | Likely model miss |
| B — human disagreement | Present/Absent split, count disagreement, boundary dispersion | Add an independent annotator or adjudication under musician policy | Annotation uncertainty |
| C — coverage gap | Too few annotations or low prediction coverage / missing strata | Acquire coverage | Insufficient evidence |
| D — hidden repeat | Frozen known-answer repeat/control | Audit consistency | Process reliability |

The planner gives priority to **model error only when the current human panel is stable and prediction coverage is adequate**. When humans disagree, the next action is annotation—not a declaration that the model is wrong. When coverage is insufficient, the case is not treated as a reliable negative or a stable model miss.

The current v1 uses deterministic priority bands, not an opaque scalar utility. Every selected record carries the band and reason.

## 3. Two-stream design

### 3.1 Active refinement stream

Eligibility requires:
- authorization state is explicitly Authorized;
- a structurally valid authorization/use-purpose receipt digest is supplied;
- panel is below its maximum size;
- source work, recording, item, query, stratum, task family and purpose scope are present;
- cohort role is Development or ActiveRefinement;
- the source stratum has an active quota.

Priority bands are sorted in deterministic order:
1. ModelFailure only on a sufficiently stable panel with adequate model coverage.
2. HumanDisagreement for state/count/boundary disagreement.
3. CoverageGap for a low panel count or low prediction coverage.
4. HiddenRepeat control.

Within each band, the planner orders by the relevant severity field, then a deterministic seed-derived tie key. It first attempts per-stratum minimum quotas, then round-robins through priority bands and source strata. It enforces:
- total and active annotation budgets;
- minimum/maximum selected items per source stratum;
- per-source-work active cap;
- one candidate ID per output selection;
- no source work assigned to both Active and ConfirmatoryAudit roles.

An unmet quota is returned with requested and selected counts. No constraint is relaxed when the budget is exhausted.

### 3.2 Probability-audit stream

The first implementation uses a two-stage, source-first design, stratified by a frozen source-work-level stratum:

1. In stratum (h), let (W_h) be the number of eligible source works and (K_h) the preregistered source-work draw.
2. Select (K_h) source works uniformly without replacement.
3. For each selected work (w), let (n_w) be the number of eligible query items. Select one item uniformly from those (n_w) items.

For each work:
[
P(w	ext{ selected})=K_h/W_h
]

For each eligible item (i) in work (w):
[
P(i	ext{ selected})=\frac{K_h}{W_h}\times\frac{1}{n_w}
]

The output records all three values as exact rational fractions:
- source-work inclusion probability;
- item-given-source-work probability;
- item inclusion probability.

The sample sizes (K_h) are determined **before any random draw**, using the worst-case item cost in each stratum. Worst-case cost reserves are consumed in deterministic stratum order. This prevents a cheap item randomly selected in an early stratum from increasing the later stratum's sample size and changing its inclusion probabilities. Actual spend is tracked separately and must remain within the reserved audit budget.

This two-stage design gives each source work the same source-selection probability within its stratum, while query items in larger works have lower item-level inclusion probabilities. Confirmatory estimates must therefore use the recorded inclusion probabilities or a preregistered source-work-level estimand.

### 3.3 Estimation boundary

The selection planner only supplies a probability sample and exact inclusion probabilities. It does **not** yet implement final weighted estimators or standard errors.

For a finite item-level total, Horvitz–Thompson estimation uses:
[
\widehat{T}_{HT}=\sum_{i\in s}\frac{y_i}{\pi_i}
]

A normalized Hájek mean can be reported with its design-aware uncertainty, but it is a ratio estimator and should not be mislabeled as an exactly unbiased finite-sample mean. For an equally weighted source-work estimand, define each source-work summary first, then use the source-work selection probabilities.

A future estimator adapter must also account for stratification, the two-stage draw and finite-population corrections. Do not treat excerpts as independent works.

### 3.4 Hard split between the streams

The planner enforces source-work disjointness between active refinement and probability audit. An adaptive selection cannot enter the probability audit, even if the same item appears promising. The confirmatory audit membership is determined before outcomes are observed and must stay frozen.

An active selection is not a random test item. A non-selected item is *unreviewed*, not an absent event, not a negative model result and not proof that no annotation is needed.

## 4. Evidence and authorization binding

Each candidate binds:
- item, query, source-work and recording identities;
- source stratum and task family;
- purpose scope and cohort role;
- a rights/consent authorization receipt digest;
- the exact #13560 per-item model-grounding report digest;
- the exact #13617 inter-annotator reliability report digest;
- a frozen cohort-assignment manifest digest;
- current/max panel size, annotation cost, error/disagreement/count/coverage/boundary signals.

The planner hashes the normalized candidate manifest, normalized config and final selection receipt using the existing canonical evidence-digest implementation.

These digests are structural references. The planner does **not** authenticate the rights receipt, scorer reports, cohort assignment or musician identity. It never infers consent from a source URL or pseudonymous annotator ID.

## 5. Deterministic implementation controls

The Rust planner includes controls for:
- identical config/seed/candidate manifest ⇒ byte-identical selection receipt and selection records;
- model failure with stable panel ⇒ ModelFailure band;
- disagreement ⇒ HumanDisagreement band, not model failure;
- low coverage ⇒ CoverageGap band;
- hidden-repeat ⇒ separate control band;
- source-work cap and source-work stream disjointness;
- Rights Denied, Unknown and missing receipt ⇒ fail closed;
- duplicate item/query IDs, malformed digests and out-of-range metrics ⇒ reject;
- total/active/audit budgets and unmet quotas ⇒ report without relaxation;
- exact two-stage inclusion probabilities when works have different numbers of eligible query items;
- pre-reserved sample sizes remain constant when a randomly selected item has lower cost than another item from that work.

## 6. Competitive context and SOTA strategy

The planner is an **evaluation infrastructure** capability, not a music generator. Compare it against relevant data-acquisition designs and use the strongest grounding/revision systems as downstream models:

- **TAG-Bench**: 1,750 human-verified query/recording pairs over 149.5 hours; the paper reports 22.1% multi-target queries, best reported mIoU 31.2 and Recall@IoU≥0.7 21.5%; one-to-many count accuracy does not exceed 13.2% for the evaluated systems. That makes repeated-event coverage and source-disjoint audit sampling central evaluation strata. https://arxiv.org/abs/2609.01542
- **RefTAG**: reflection-aware draft→count/boundary diagnosis→revision; reports 38.50 TAG-Bench mIoU and 51.01 mean mIoU across three datasets. Its self-revision results make it important to sample high model error against reliable human references and to preserve the count/boundary error reasons. https://arxiv.org/abs/2610.16217
- **MusTBench v3**: music-specific temporal benchmark with five distinct tasks. Reproduce its official split and metrics separately. https://arxiv.org/abs/2605.29300
- **MusicGroundingBench**: exact piano MIDI-to-audio alignments; use for known-answer synthetic controls, not as a proxy for human boundary reliability. https://arxiv.org/abs/2608.29480
- **#9184** owns blinded human metric calibration and unseen-renderer generalization. Its probability sample must not be replaced by the adaptive stream.
- **#9151** already owns sparse budgeted probing for musical interaction measurements. Reuse its source/stratum round-robin and replay-receipt patterns; this planner is specific to annotation acquisition and two-stage sampling.

Commercial creator products such as Suno Studio 2.0 and ElevenLabs Music v2.5 demonstrate strong editing, generation and workflow UX. They are not annotation-acquisition baselines. Do not claim that the planner itself is a commercial product differentiator; its strategic value is to make the evidence behind musical claims more efficient and statistically defensible.

## 7. Prospective benchmarks and targets (not results)

The following are proposed pilot targets, not current results:

1. **Equal-budget comparison:** compare Active+Audit against uniform-random annotation at the same total budget, source-work caps, stratum minima and reviewer panel policy.
2. **Active discovery yield:** measure additional confirmed model failures, repeated-event omissions, false absences and resolved boundary/count disagreements per 100 annotation-cost units.
3. **Random-audit representativeness:** verify item inclusion probabilities exactly against exhaustive enumeration on finite synthetic populations and compare weighted estimators against known finite-population truth.
4. **No selection leakage:** zero source works shared between active refinement and frozen audit; zero confirmatory holdout leakage.
5. **Coverage:** report source-work effective sample size, strata coverage, source concentration, and unmet quotas.
6. **Cost benefit:** target at least 25% more confirmed actionable errors per equal annotation budget than uniform random selection in the pilot, while preserving audit estimation error within its preregistered margin.
7. **Reliability preservation:** active annotation must not reduce the number of independent raters per item or turn ambiguous/not-estimable labels into absent labels.
8. **Replay:** fixed candidate manifest, config and seed reproduce the exact receipt, active selections, probability sample and inclusion fractions.

A planner is not better because it finds more disagreements: it must find more actionable information per cost while preserving a separate probability sample and its intended estimand.

## 8. Sequencing

1. Finish exact-head qualification of the deterministic Rust planner.
2. Add a finite-population estimator adapter with exact HT totals and design-aware variance for this two-stage stratified sample.
3. Add a source-balanced uniform-random baseline and equal-budget pilot simulation.
4. Connect #13560/#13617 report receipts to actual candidates; authenticate through the owning evidence/rights services.
5. Run a musician pilot across separately governed event-family/source-tradition cohorts.
6. Freeze probability-audit design before confirmatory data collection; only then report weighted held-out grounding estimates.
7. Feed active-stream findings into model development, then evaluate the frozen model on the untouched probability audit.

## 9. Claim ceiling

This v1 is a deterministic annotation scheduling planner. Its synthetic fixture validates selection and inclusion-probability plumbing only. It does not verify real consent, authenticate external evidence, estimate model performance, demonstrate an annotation-cost reduction, or establish SOTA. Probability-audit estimates are defensible only under the recorded sampling design and its stated target population; active-stream observations must never be silently pooled into that denominator.
