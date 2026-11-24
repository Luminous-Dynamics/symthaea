# Mathematical proof comprehension — evaluator calibration protocol v1

**Issue:** [#13476 — MATH-EVALUATOR-CALIBRATION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13476)  
**Parent:** [#13293 — MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Status:** prospective protocol; no evaluator-calibration experiment has been run.

## 1. Why evaluator quality is a separate research problem

A system can produce a plausible proof explanation, and a judge can rate it highly, even when the proof contains a missing premise or invalid inference. Mathematical evaluation therefore needs its own validity study.

Two useful public baselines make the issue concrete:

- [ProofBench / ProofGrader (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/764d4fc73ab69289b1e10f15089274f3-Abstract-Conference.html) contains 145 competition problems and 435 model-generated natural-language proofs with expert ratings. ProofGrader reports 0.926 MAE against expert scores under its scoring protocol.
- [QEDBench (ICML 2026)](https://proceedings.mlr.press/v306/gonzalez26a.html) studies alignment between automated judges and human experts on university-level proofs. It reports positive score inflation up to +0.28 for an evaluated judge and domain-specific performance failures, including graph theory.

These are adjacent baselines, not interchangeable tasks. We should reuse their scoring lessons and reproduce a comparable subset only when task, rubric, model and access conditions are pinned. A low grading MAE does not certify theorem truth.

## 2. Partition authority by task

The evaluator pipeline has four different authorities:

1. **Formal authority:** Lean kernel/type checking and exact source/environment identity for formal claims.
2. **Finite semantic authority:** a bounded, explicitly declared decision procedure for supported finite propositional tasks; atom-budget exhaustion returns unknown.
3. **Mathematical-review authority:** two independent reviewers and an adjudicator for interpretation, proof-plan role, and cases not settled by the formal/finite mechanisms.
4. **LLM evaluator:** a calibrated assistant for locating candidate defects, mapping claims to evidence, and drafting rubric assessments. It cannot set formal truth, eligibility or the hidden gold label.

The candidate model, evaluator model, human reviewers and checker must not share hidden gold material. The evaluator must not receive the hidden countermodel, upper/lower sandwich proof, answer key or gold classification when those are the objects being assessed.

## 3. Evaluation arms

On the same frozen task instances, compare:

- **Human gold process:** two independent mathematical annotators; a third adjudicator resolves disagreements and preserves unresolved cases.
- **Generic LLM judge:** proof/explanation plus task prompt and rubric, without structured proof-plan evidence.
- **Evidence-constrained LLM judge:** exact theorem/source/context identity, typed ProofPlanV1 nodes, candidate-visible evidence references, and explicit unknown/ill-formed outcomes.
- **ProofBench/ProofGrader-style evaluator:** where its public task assets and rubric permit a valid matched comparison.
- **Formal/finite checker:** only for claims within its declared formal or decidable scope; report this as a separate authority, not an LLM judge arm.

Blind judges and human reviewers to candidate author/system identity. Do not let a model evaluate its own output as the sole or authoritative judge. If the evaluator has access to the full proof source while another arm sees only a summary, label that as a separate context condition.

## 4. Required evaluator output

For each atomic proof claim or plan transition, record:

- disposition: accept, reject, unknown, or ill_formed;
- defect class, if any;
- first-invalid-step / first-unsupported-claim location when uniquely identifiable;
- exact source, declaration, goal, assumption and plan-node evidence references;
- whether the cited evidence entails the claim, contradicts it, or does not decide it;
- confidence and abstention reason;
- raw response digest, evaluator identity, prompt/version, token/tool budget and latency.

Do not score a citation as valid merely because it points to a relevant declaration. The cited evidence must support the exact claim and the correct direction of inference.

## 5. Adversarial strata

Freeze positive and negative controls for:

1. valid proof and valid alternative proof route;
2. invalid implication direction with a checked countermodel;
3. omitted side condition or hidden assumption;
4. statement strengthened beyond what the proof establishes;
5. statement weakened so that required corollaries no longer follow;
6. false statement versus true statement with a broken proof route;
7. valid but irrelevant dependency;
8. namespace collision or misleading short-name match;
9. opaque/axiom/admission boundary;
10. natural-language / AST translation mismatch;
11. unsupported explanation claim not present in source or formal proof;
12. unrelated successor-theorem failure that must not be blamed on the candidate statement.

Each item has one evaluator-side adjudication receipt. Controls must be balanced enough to expose both false acceptance and false rejection; a judge that rejects every proof is not reliable.

## 6. Metrics and prospective targets

Report each metric independently and stratify by theorem family/domain:

| Metric | Prospective target | Interpretation |
|---|---:|---|
| False acceptance on frozen negative controls | 0 observed | Include denominator and exact control coverage; zero observed is not a universal guarantee |
| Precision of defect flags | ≥95% | A reported defect is independently supported |
| Recall of seeded defects | ≥90% | Known planted failures are detected |
| First-invalid-step localization | ≥90% | Only where a unique location is defined |
| Evidence citation precision | ≥95% | Cited evidence entails or directly supports the stated claim |
| Positive judge-score bias | ≤0.10 rubric points | Mean signed score error versus adjudicated human rating |
| ProofBench-style score MAE | ≤0.926 on a genuinely matched subset | Stretch comparison only; require same rubric/tasks |
| Selective risk | Report full risk/coverage curve | Abstentions remain visible and are not silently dropped |
| Calibration | Report Brier score and ECE | Confidence should predict empirical correctness |
| Inter-rater reliability | Report by task/domain | Retain reviewer disagreement and adjudication rate |

These are prospective targets, not Symthaea results. Do not create a weighted composite that lets good style ratings compensate for accepting an invalid proof.

## 7. Analysis and contamination controls

- Split by theorem family and repository, not random paraphrase.
- Freeze source commits, Lean toolchain, import closure, prompts, model snapshots, decoding, candidate pool, budgets and judge context.
- Separate development, calibration and hidden evaluation sets.
- Tune thresholds only on the calibration split; evaluate once on the hidden theorem-family holdout.
- Report exact confusion matrices by defect class, positive score bias, false acceptance, false rejection, abstention, and confidence intervals.
- Record disagreements and adjudication effort. Distinct reviewer IDs alone do not establish reviewer independence.
- Use paired task-level analyses and cluster uncertainty by theorem family; repeated variants of one theorem are not independent samples.

## 8. Immediate pilot sequence

1. Select 30–50 development cases across the adversarial strata.
2. Have two independent mathematical reviewers annotate them without seeing model-produced judge scores.
3. Adjudicate disagreements and freeze gold receipts.
4. Run generic and evidence-constrained judges under matched contexts and budgets.
5. Add one theorem-family holdout before changing evaluator prompts or thresholds.
6. Publish failures and disagreements before connecting any evaluator score to the human transfer study.

## 9. Claim boundary

- LLM judge agreement is not theorem validity.
- A formal checker does not resolve informal/formal translation by itself.
- A valid dependency citation does not establish mathematical necessity.
- A proof-quality score does not establish delayed human learning.
- Finite truth-table completeness applies only to the supported propositional AST and atom budget.
- The evaluator can propose and localize a defect; hidden gold, formal checks and independent mathematical review determine the adjudicated outcome.
