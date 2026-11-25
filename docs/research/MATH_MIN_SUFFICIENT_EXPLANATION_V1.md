# Minimum sufficient mathematical explanation — protocol v1

**Issue:** [#13548 — MATH-MIN-SUFFICIENT-EXPLANATION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13548)  
**Programme:** [#13293 — MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Status:** prospective protocol. No Symthaea explanation-efficiency or human-transfer result is claimed.

## 1. Objective

Find the smallest evidence-grounded explanation that preserves the mathematical mechanism and supports human reconstruction and transfer. This is not a text-shortening contest.

Let E be an explanation and P an expert-reviewed ProofPlanV1. Minimize explanation cost L(E), subject to hard constraints:
- exact theorem statement, quantifiers, domain and hypotheses remain faithful;
- every critical plan action and warrant is covered;
- consumed side conditions and produced/discharged subgoals are represented;
- every mathematical claim has evidence or is explicitly unresolved;
- no unsupported mathematical claim is introduced;
- immediate reconstruction and delayed transfer are non-inferior to the strongest matched baseline.

L(E) includes token/word count, reading time, cognitive load, concept count and evidence-navigation cost. Report these separately; never treat token count as a complete quality measure.

## 2. Separate reader and prover products

Maintain separate tracks:
- **Reader-facing explanation:** proof plan, key ideas, side conditions, counterexamples and progressive disclosure.
- **Prover-facing context:** exact formal premises and tools permitted by the proving protocol.

FormalMATH reports that natural-language solution references can reduce formal proving performance in some settings. A pedagogically useful explanation must therefore not automatically become prover context.

## 3. Explanation tiers

- **E0 — formal source:** raw theorem/proof source with no generated explanation.
- **E1 — full evidence-grounded plan:** all critical plan actions, warrants, side conditions, subgoals and limitations.
- **E2 — sufficient core explanation:** smallest selected plan/evidence set that lets a reader reconstruct the main mechanism and validate critical transitions.
- **E3 — progressive disclosure:** compact core with optional expansion to evidence and technical detail.
- **E4 — compressed prover context:** separately measured formal context; not assumed equivalent to E2/E3.

Multiple explanations may be valid for one proof. Evaluation scores semantic coverage and transfer, not wording identity.

## 4. Explanation claim contract

Every fragment links to:
- ProofPlanV1 node/action/warrant;
- exact theorem and hypothesis scope;
- source declaration, elaborated type or proof artifact;
- consumed side conditions and created/discharged subgoals;
- epistemic class: formal fact, reviewed mathematical interpretation, bounded analogy or unknown;
- reviewer status and unresolved caveats.

A compressed explanation may omit tactic traces, but may not silently omit a condition required for a mathematical transition.

## 5. Experimental arms

At matched task, model, prompt and resource budgets compare:
1. raw source/proof;
2. generic model explanation;
3. expert-authored full explanation;
4. full ProofPlanV1-grounded explanation;
5. MSE compressed explanation;
6. progressive-disclosure explanation;
7. prover-facing structured context as a separate ablation.

Use blind human evaluation. Keep answer keys and expert annotations hidden from candidate generation.

## 6. Outcomes

Report separately:
- token/word count, reading time, concept count and evidence-navigation cost;
- theorem/assumption fidelity;
- critical plan-action coverage;
- warrant and side-condition recall;
- unsupported-claim and omission rates;
- immediate proof reconstruction;
- first-invalid-step localization;
- counterfactual prediction after assumption removal or mutation;
- delayed near-transfer and cross-family transfer;
- confidence calibration, cognitive load, reviewer minutes and adjudication;
- formal proof success in the separate prover-context experiment.

No weighted score may let compression compensate for false mathematics.

## 7. Prospective qualification targets

| Measure | Target |
|---|---:|
| Explanation token reduction vs strongest matched full explanation | ≥25% |
| Immediate comprehension non-inferiority margin | within 5 percentage points |
| Delayed cross-family transfer improvement vs strongest control | +10 percentage points (stretch target) |
| Critical side-condition coverage | ≥95% |
| Unsupported critical warrants / false theorem statements | 0 observed on frozen evaluation set |
| Claim evidence linkage | 100% resolvable or explicitly unknown |

These are prospective gates, not results. Human sample size must be computed from pilot variance, theorem-family clustering and the preregistered endpoint. Do not select N after seeing the treatment effect.

## 8. Competitive landscape

- [LeanLean (2026)](https://www.sri.inf.ethz.ch/publications/rognvaldsson2026leanlean): 64 repositories, up to 174k lines, and 48.3% average repository-level Lean code compression for its leading evaluated model under a 12-hour budget. This is code compression, not human explanation or learning.
- [ProofRank (NeurIPS 2026)](https://www.sri.inf.ethz.ch/publications/petrov2026proofrank): assesses conciseness, computational ease, cognitive simplicity, diversity and adaptivity. Useful proof-quality dimensions, but not a substitute for delayed learner transfer.
- [FormalMATH (EMNLP 2026 Findings)](https://aclanthology.org/2026.findings-emnlp.579/): reports that natural-language solution references can harm formal proof generation in some settings; this motivates separate reader/prover tracks.
- [QEDBench (ICML 2026)](https://proceedings.mlr.press/v306/gonzalez26a.html): demonstrates evaluator alignment gaps and score inflation, requiring calibrated independent evaluation.
- [Gao & Evans (2026)](https://link.springer.com/article/10.1007/s40753-026-00319-4): in a 143-student study, proof-presentation ratings were only moderately associated with delayed near-transfer proof construction, motivating direct measurement of retention and transfer.

## 9. Execution sequence

1. Freeze an expert-reviewed ProofPlanV1 corpus; provisional source annotations are not gold.
2. Create E1 explanations with explicit semantic coverage maps.
3. Generate E2 candidates by removing explanation/evidence fragments under hard coverage constraints.
4. Run structural checks for missing warrants, assumptions, side conditions and unresolved claims.
5. Blindly review E1/E2/E3 variants; adjudicate disagreements.
6. Pilot immediate reconstruction and counterfactual prediction.
7. Preregister delayed transfer with power analysis and ethics oversight.
8. Run the prover-context ablation separately.
9. Publish Pareto frontiers for explanation cost, fidelity, comprehension, transfer and reviewer effort.

## 10. Claim ceiling

- Shorter does not mean better understood.
- Formal proof correctness does not establish pedagogical sufficiency.
- Human explanation quality does not imply improved theorem-prover performance.
- LLM judge scores do not establish comprehension.
- No Symthaea MSE result or human-transfer outcome has been collected.
