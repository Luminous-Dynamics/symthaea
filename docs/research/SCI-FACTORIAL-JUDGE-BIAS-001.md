# SCI-FACTORIAL-JUDGE-BIAS-001 — Factorial measurement of interacting judge nuisance effects

**Status:** Rust plan verifier and descriptive analysis module, strict plan/receipt schemas, deterministic synthetic fixture and CI controls under development. No LLM judge is executed by this tool. No empirical Symthaea bias result is claimed.

**Tracking:** [Issue #9237](https://github.com/Luminous-Dynamics/symthaea/issues/9237).  
**Stack:** after SCI-METAMORPHIC-EVALUATOR-001 / [PR #9152](https://github.com/Luminous-Dynamics/symthaea/pull/9152).

## 1. Why another layer?

The existing metamorphic evaluator audit tests individual paired relations. That is necessary, but isolated tests cannot quantify interactions among several nuisance factors.

Recent research makes the gap concrete:

- **Am I More Pointwise or Pairwise?** (EMNLP Findings 2026) reports model-specific position bias in rubric score options and shows that criterion ordering can also shift criterion scores. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1444/)
- **Probe and Steer** (EMNLP Findings 2026) studies position bias as an internal model phenomenon and a possible target for localized steering. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1949/)
- **FairJudge** (ICML 2026) treats judging behavior as an adaptive, regularized policy and targets rubric adherence, non-semantic bias and cross-mode consistency. [PMLR](https://proceedings.mlr.press/v306/yang26n.html)
- **Explanation-Guided Metamorphic Testing** (ESEM 2026) combines explanation-guided mutation with semantic verification; its reported 2.30× verified-failure discovery advantage is specific to its evaluated data, models and budgets. [Paper](https://arxiv.org/abs/2608.07076)
- **Can We Trust LLM Judges?** (September 2026) reports capability-dependent leniency and evaluates a disagreement-weighted multi-judge calibration method. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/can-we-trust-llm-judges-a-study-of-capability-dependent-biases-and-multi-judge-ensemble-for-bias-calibration/?lang=ja)
- **Reliability without Validity** (June 2026 preprint) reports that repeatability and bias can coexist and warns that raw agreement can overstate chance-corrected agreement. [Preprint](https://www.alphaxiv.org/abs/2606.19544)

The proposed shift is from **one nuisance transformation at a time** to a preregistered experimental surface that measures main effects and pairwise interactions.

## 2. The full (2^5) design

Each semantic task/candidate block receives all 32 combinations of five binary factors. Each cell is repeated a preregistered number of times.

| Factor | Low level | High level | Primary outcome |
|---|---|---|---|
| Candidate order | A first | B first | A-minus-B score gap and A-choice rate |
| Score-option order | Canonical | Reversed | Selected semantic option and displayed position |
| Criterion order | Canonical | Reversed | Per-criterion score shifts |
| Provenance cue | Anonymous | Displayed | Score/choice shifts under system-identity cues |
| Presentation format | Structured | Prose | Score/choice shifts under equivalent formatting |

The v1 engineering fixture uses one block, 32 conditions per repeat and two repeats: 64 condition receipts. It plants a position/provenance choice interaction: candidate A retains the higher score, but the judge chooses candidate B when B is first and the provenance cue is hidden; showing provenance reverses that choice. This is an analyzer positive control, not an empirical behavior claim. A real study should use many independently sampled semantic blocks and multiple judge systems.

### Semantic identity must remain fixed

The block freezes candidate IDs, criterion IDs, score-option IDs and their canonical numeric values. Conditions change the presentation order/cue/format, not the underlying semantic identity.

The plan declares each condition's factors and binds its rendered-input digest. The receipt must echo that exact digest and the exact execution ordinal from the committed schedule. A receipt that omits or changes the planned ordinal is rejected. This records schedule agreement; it does not prove the runner actually executed in that order. Each selected criterion option must map to the same score value as its frozen option ID.

The transformation itself still needs independent semantic-validity review. A digest proves byte binding, not semantic equivalence. For provenance cues, only use domains in which system identity is a nuisance rather than a legitimate criterion. If source identity materially belongs in the decision, do not treat anonymization as a meaning-preserving transformation.

## 3. Plan, schedule and receipt contract

Files:
- `schema/judge-factorial-plan.schema.json`
- `schema/judge-factorial-receipt.schema.json`
- `schema/judge-factorial-report.schema.json`
- `src/judge_factorial.rs`
- `tools/judge_factorial_fixture.py`

The plan commits to:
- exact analyzer source revision (qualified plans require a clean analyzer worktree), judge source revision, prompt and rubric digests;
- semantic task-set digest and task-level semantic digests;
- candidate and criterion identities;
- canonical score-option IDs and numeric values;
- all 32 factor combinations per repeat;
- a unique condition ID derived from block, repeat and factor assignment;
- an input SHA-256 per condition;
- a committed execution-order permutation for every block.

The receipt echoes the exact scheduled ordinal for each condition. The analyzer rejects a receipt whose ordinal does not match the plan. This is schedule consistency, not proof of the runner's true execution sequence or randomization.

The verifier requires exactly 32 cells per repeat, with no duplicated or missing combination. The execution-order list must be a permutation of the full planned condition set.

The receipt contains exactly one record for every planned condition. A completed observation requires a judge receipt digest, decision, candidate scores and a score/option record for every criterion. Timed-out, blocked, invalid-receipt and not-started observations must have null outcomes and an empty criterion-score list. They are not silently omitted or scored as zero.

Hashes bind declared bytes; they do not prove that the raw rendered prompts were retained, that a random number generator produced the schedule, or that a judge actually executed. The receipt's execution ordinal only proves consistency with the committed schedule. The plan/receipt must be externally anchored before outcomes are observed for any future qualification.

## 4. Analysis

For each factor (f), the descriptive main effect for outcome (Y) is:

[
Delta_f = overline{Y}_{f=high} - overline{Y}_{f=low}
]

The current analyzer reports:
- candidate A minus candidate B score-gap effect;
- mean selected rubric-score effect;
- selected score-option display-position effect;
- candidate-A choice-rate effect, excluding ties and abstentions from the decisive-choice denominator;
- tie-rate and abstention-rate main effects, plus their pairwise interactions, with explicit counts in each factorial cell;
- per-criterion score changes under criterion ordering;
- completed/ineligible/planned counts at each factor level;
- fully balanced repeats used for effects and incomplete repeats excluded from effects;
- completed observations excluded by balanced-repeat filtering, separate from genuinely ineligible observations.

For each factor pair (a,b), the analyzer reports the raw difference-in-differences:

[
I_{ab} = overline{Y}_{11} - overline{Y}_{10}
        - overline{Y}_{01} + overline{Y}_{00}
]

There are ten pairwise factor interactions in a five-factor study. The report includes all four cell denominators, decisive-choice denominators within each interaction cell, and counts excluded observations across those cells. Main effects, criterion-order effects and interactions use only fully completed 32-cell block/repeat units. A repeat with even one timed-out/blocked/invalid/not-started condition is excluded as a whole from effect estimation; all its planned cells remain in the full denominator. The report separately counts ineligible observations and otherwise completed observations excluded because their repeat was incomplete.

This v1 analysis is deliberately descriptive. It does not calculate p-values, confidence intervals, randomization-inference tests, cluster-robust standard errors or multiplicity corrections. Missing outcomes remain explicit; effects computed from incomplete cells are not qualification evidence. A future inferential layer must respect task/judge clustering and the preregistered assignment mechanism.

## 5. Benchmark targets and competitive protocol

### Engineering qualification

- 100% acceptance of the clean synthetic 32-cell/repeated-cell fixture;
- 100% rejection of missing, duplicate, malformed, unbalanced or tampered factorial cells;
- exact condition/input digest binding;
- no duplicate or omitted execution-order entry;
- no score/option semantic mismatch;
- 100% receipt coverage for a complete run;
- explicit ineligible counts, with no imputation;
- byte-deterministic report for identical plan and receipt bytes.

These targets qualify the analyzer, not an LLM judge.

### Empirical comparison

Each plan pins one judge source revision, prompt and rubric. Therefore each judge model/version should receive its own plan and receipt; do not mix distinct judge revisions inside a single plan. A comparative study should use the same semantic task-set digest, task blocks, candidate identities, rubric semantics, factor assignments and repeat budget across those plans. Compare the resulting bias surfaces by judge, and preserve model/version as a top-level comparison stratum rather than pooling their observations as if they were one evaluator.

Compare under identical semantic tasks, judge prompt, rubric, model revision, repeat count and total token/inference budget:

1. one-pass baseline judge;
2. conventional bidirectional pairwise evaluation;
3. current judge-robustness / metamorphic controls;
4. FairJudge-style adaptive/debiased judge;
5. Probe-and-Steer where model weights/logits and the method's applicability assumptions are available;
6. calibrated multi-judge ensemble;
7. blinded, independently adjudicated human reference.

Report raw accuracy, Cohen's (kappa) or weighted (kappa), Krippendorff's (alpha) where appropriate, pairwise preference accuracy, calibration (Brier/ECE), abstention coverage and selective risk, candidate-position asymmetry, rubric-option-order effect, criterion-order effect, provenance/format effect, interaction effects, repeatability, token/call cost and human-adjudication cost.

Do not rank systems by raw agreement alone. Report chance-corrected agreement and validity against independent reference judgments alongside consistency and bias.

### Research success criteria

First reproduce the relevant published bias patterns on a matched subset. Then seek improvements under an equal judge-call budget:
- reduce nuisance-factor effects without reducing task accuracy;
- reduce candidate-choice reversals after pure position swaps;
- reduce score-option/criterion-order effects while preserving calibrated score meaning;
- reduce capability/provenance leniency;
- preserve or improve abstention calibration;
- obtain more valid decisions per dollar and per evaluator call.

The acceptable practical-equivalence margin for a nuisance effect must be preregistered per domain and endpoint. We should not declare an effect harmless merely because a small study fails to reject zero.

## 6. Next research stages

1. **Execution custody:** anchor plan before outcomes; store exact rendered-input bytes or independently verifiable encrypted artifacts; sign run receipts; prove execution order and model/prompt revisions.
2. **Semantic validity:** human/adjudicator review of transformation pairs, blinded to judge output; measure acceptance precision and mutation diversity.
3. **Inference:** matched block contrasts, randomization tests, hierarchical model for task/judge/provider clusters, multiplicity control and power analysis.
4. **Adaptive screening:** use a preregistered fractional-factorial design for cheap screening, then expand suspicious interactions to full factorial follow-up. Never choose follow-up cases based on hidden outcome access without a predeclared adaptive policy.
5. **Mitigation bake-off:** compare rubric randomization, bidirectional judging, FairJudge-style training, Probe-and-Steer, calibrated ensembles and abstaining judges against the same factorial surface.

## 7. Explicit claim boundary

The generated report always sets:
- `analyzer_worktree_clean` to the build-time checkout cleanliness value (and refuses `qualified_run` when false);
- `execution_authenticity_proven=false`;
- `randomization_proven=false`;
- `raw_prompt_custody_proven=false`;
- `semantic_equivalence_proven=false`;
- `causal_effect_identified=false`;
- `generalized_accuracy_estimated=false`;
- `external_anchor_required=true`;
- `scoring_eligible_declared=false`.

A committed schedule is not proof of randomness. A structured score receipt is not proof of judge execution. A synthetic positive control is not an empirical result. A factorial contrast is not a population effect without an adequate sampling and inference design.
