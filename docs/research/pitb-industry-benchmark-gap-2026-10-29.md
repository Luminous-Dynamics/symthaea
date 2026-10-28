# PITB v0: industry benchmark gap and qualification roadmap

Date: 2026-10-29  
Status: research and preregistration guidance; **not an empirical model comparison**.

## Executive decision

PITB should not try to beat theorem-prover leaderboards by quoting a higher solve percentage on a different test set. Its distinctive claim should be narrower and more useful: *does transferring a proof insight cause more independently verified, semantically valid, held-out theorem solves per matched unit of compute, after correcting for dependency leakage and incomplete outcomes?*

The next paradigm shift is from **proof accepted** to **knowledge-transfer effect under adversarially audited, semantically tested, statistically qualified conditions**. A Lean kernel proves that a generated term establishes its formal proposition; it does not by itself prove that the proposition matches the intended task, that downstream results remain correct, or that the evaluation split is independent.

## What the current frontier says

These are useful external comparison points, but their denominators and tasks differ. None is a head-to-head result against PITB.

| Reference | Published result or scope | Lesson for PITB |
|---|---|---|
| [Goedel-Prover-V2 (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html) | Reports 90.4% MiniF2F pass@32 in self-correction mode for its 32B prover. | A strong solve-rate headline is already possible on a contest-style benchmark. PITB needs a distinct research-level transfer target and budget-matched baselines, not a claim of superiority from unlike percentages. |
| [Pythagoras-Prover](https://arxiv.org/abs/2606.12594) | Reports 93.0% MiniF2F-Test at pass@2048 and strong results under smaller model/compute configurations. | Report the full quality–compute frontier, including attempts and total spend—not only the best pass@k point. Structured statement variants can expose memorization sensitivity. |
| [miniF2F-Lean Revisited](https://proceedings.neurips.cc/paper_files/paper/2025/hash/72dad95a24fae750f8ab1cb3dab5e58d-Abstract-Conference.html) | The end-to-end natural-language-to-Lean pipeline reached 70% on corrected miniF2F-v2 versus 40% on the original set in the reported evaluation. The paper attributes substantial loss to statement/formalization mismatch. | Score the exact human-intended target, not just a proof of a potentially mismatched formal statement. |
| [T²: Benchmarking Testing in Automated Theorem Proving (ACL 2026 Industry)](https://aclanthology.org/2026.acl-industry.150/) | 2,206 problems from five Lean repositories, paired with an average of 41 successor theorems; the paper reports 38.9% best testing accuracy despite higher compilation rates. | Add semantic successor/integration tests alongside kernel acceptance; compilation and semantic correctness are different axes. |
| [TCSAlgBench (2026 preprint)](https://arxiv.org/abs/2609.35606) | 398 research-level TCS theorem challenges from 138 STOC/COLT 2026 papers; published comparisons use five runs and matched model-call opportunities, with highest reported five-run verifier-accepted coverage of 23.6% in one model comparison and 25.4% in a separate agent-workflow comparison. | PITB should be repeated, budget-matched, research-level and refreshable. Five-run coverage is not directly comparable to single-run or pass@k metrics. |
| [Formal Conjectures (2026)](https://arxiv.org/abs/2605.13171) | 2,615 Lean 4 statements, including 1,029 open research conjectures in the reported corpus. | Pair a frozen, reproducible comparison cohort with a rolling time-based holdout so progress is not measured only on static familiar material. |
| [TheoremBench (2026 preprint)](https://arxiv.org/abs/2606.09450) | Nearly 100 classical theorems, with a version exposing related subtheorems and metrics for theorem-level coverage and token efficiency. | Measure intermediate proof-structure coverage, not just final theorem completion, and report tokens spent per verified solve. |
| [Faults in Our Formal Benchmarking (2026 preprint)](https://arxiv.org/abs/2606.29493) | The audit reports 4,833 findings across five widely used Lean benchmarks/forks, including 398 mechanically certified issues. | Benchmark validity is a first-class measured property. Formal checking cannot compensate for vacuous or incorrectly formalized tasks and a weak harness. |
| [SorryDB snapshot / leaderboard](https://sorrydb.org/) | January 2026 snapshot of 1,000 open proof obligations intended to represent day-to-day Lean work; the leaderboard reports varying pass@1 and pass@32 results. | Maintain a practitioner-oriented secondary cohort; pin the exact snapshot and toolchain for every comparison. |

## The paradigm shift: a verified transfer experiment, not a solve-rate contest

PITB's treatment should be a frozen intervention—such as a retrieval/proof-insight strategy, memory abstraction, or cross-family decomposition—and the control should be the best validation-selected baseline under the same model-call, token, time, checker, tool and monetary budgets. The intervention must be selected before test outcomes are examined.

The primary outcome should be exact-target semantic-gate verified success. Each accepted claim should require:

1. the exact frozen theorem statement and toolchain to elaborate;
2. the proof to pass the trusted checker;
3. the intended statement to match its reviewed human/source specification;
4. relevant successor theorems or downstream invariants to pass their integration checks;
5. the run, checker, retrieval, semantic-gate and artifact receipts to verify against their digests.

Treat timeout, not attempted, tool unavailable, checker missing and invalid receipt as non-solves for the operational intention-to-treat outcome. A missing frozen eligible row or missing arm is not permission to silently shrink the denominator: reject the confirmatory analysis and report the missingness. A digest proves byte identity, not reviewer independence or semantic truth.

## Target gates

These are recommended qualification gates; they are not achieved results and must be frozen before evaluating the model.

| Dimension | Qualification target | Why it matters |
|---|---|---|
| Dataset/receipt integrity | 100% frozen eligible target/arm accounting; no silent deletion or post-outcome exclusion; every accepted solve has valid receipts. | Blocks denominator manipulation and untraceable claims. |
| Semantic correctness | 100% of accepted solves pass kernel, intended-statement alignment and registered downstream semantic tests. | A compile rate alone can overstate useful correctness. |
| Leakage resistance | Every frozen family maps to exactly one deterministic relationship component; no unresolved hard cross-split component can qualify; complete coverage has an independently reviewable evidence receipt. | Prevents related theorems from masquerading as independent evidence. |
| Reproducibility | Exact source/config/data/toolchain digests; deterministic report replay; independent rerun agrees on all count, digest and gate fields. | Makes the experiment auditable and falsifiable. |
| Type-I error | Worst-case simultaneous 95% upper confidence bound no greater than 5% across strict-null and preregistered weak-null heterogeneity scenarios. | Controls the false-claim rate across the registered stress grid, rather than checking a single convenient null. |
| Power | Simultaneous 95% lower confidence bound at least 80% for the preregistered +20 percentage-point effect in the hardest supported cluster/association scenarios. | A minimum cluster count alone does not establish adequate power. |
| Meaningful utility | Point estimate at least +10 percentage points and an informative 95% interval whose lower bound exceeds zero, with the exact registered component-cluster floor met. | Preserves the current practical-effect intent while requiring uncertainty resolution. |
| Compute efficiency | Report the Pareto frontier for verified solves, tokens, wall time, calls and cost; stretch target: at least 20% lower cost per semantically verified solve than the strongest matched baseline with no loss in the utility gate. | Avoids calling an expensive sampling increase an architectural improvement. |
| Missingness robustness | Fail closed on missing frozen arms/targets/checker receipts; zero confirmatory passes in the registered adversarial missingness controls. | An exploratory synthetic screen showed that outcome-dependent deletion can create large false-positive rates under an invalid complete-case comparator. |
| External validity | Replicate on a temporally later batch or independently maintained repository with a fresh split and the same scoring contract. | Separates true transfer from performance on one frozen catalogue. |

The existing 20-family minimum should be treated only as a provisional floor. The independent unit must be the audited connected component. If connected components collapse the eligible sample below the registered floor, the result is a pilot—not a reason to revert to family labels.

## Evaluation design that can beat the *methodology* of the field

### 1. Two complementary holdouts

Keep a pinned, versioned cohort for exact reruns; add an embargoed temporal cohort from later research releases. In addition, run a new-source repository cohort that was not used to build the insight library. A model must not choose an easier holdout after seeing scores.

### 2. Component-aware randomization and resampling

Construct the relationship graph from exact duplicates, shared source artifacts, formal dependencies, reused lemmas and equivalent statements. Include unresolved hard edges conservatively. Cluster bootstrap and power simulation must consume the same complete component manifest—not a family count reconstructed independently in each stage.

### 3. Semantic mutation controls

Create mutations that omit a required hypothesis, weaken the theorem, introduce a vacuous implication, exploit an unsafe axiom, or break a successor theorem while leaving the proof superficially plausible. Measure false acceptance and mutation kill rate as separate metrics. Every checker change needs negative controls demonstrating that the intended defect is detected.

### 4. Ablation and causal attribution

Under fixed budgets compare: no transferred insight; retrieval only; proof sketch only; decomposition only; memory/insight plus verifier; and the full treatment. Stratify by proof depth, dependency count, domain, family/component size and distance from the source insight. This tells us *why* transfer works rather than merely whether the aggregate score moved.

### 5. Sequential fresh-challenge evaluation

Use a pre-registered stopping rule for model comparisons and a sealed stream of newly released tasks. Keep the final test labels hidden from intervention selection. Publish every snapshot ID, stop decision, outcome denominator, failure category and confidence interval. New snapshots supplement—they do not silently rewrite—the frozen comparison set.

## Immediate engineering sequence

1. **Finish the complete component manifest contract.** Validate the builder against the pinned split and synthetic negative controls; keep its qualification status false while inventory coverage is incomplete.
2. **Wire the manifest into the frozen run config.** The config must pin its exact digest and reject any manifest that is incomplete, malformed or split-crossing.
3. **Use the same component IDs in power planning and test analysis.** Simulate and resample the same connected units, enforce the 20-component minimum, and keep original family/target counts separately for transparency.
4. **Add semantic successor tests to the existing outcome receipts.** Never infer semantic correctness from compilation alone.
5. **Requalify and rerun the registered grid.** Update schemas, fixtures, report replay and negative controls together. No production pass claim until exact-head Rust tests and the workflow actually run.
6. **Run matched baseline and treatment arms, then external replication.** Publish performance only with a fixed model/tool/checker version, matched budgets, component-aware intervals and the frozen snapshot hash.

## References

- Goedel-Prover-V2, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html
- Pythagoras-Prover, 2026: https://arxiv.org/abs/2606.12594
- miniF2F-Lean Revisited, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/72dad95a24fae750f8ab1cb3dab5e58d-Abstract-Conference.html
- T², ACL 2026 Industry: https://aclanthology.org/2026.acl-industry.150/
- TCSAlgBench, 2026 preprint: https://arxiv.org/abs/2609.35606
- Formal Conjectures, 2026 preprint: https://arxiv.org/abs/2605.13171
- TheoremBench, 2026 preprint: https://arxiv.org/abs/2606.09450
- Faults in Our Formal Benchmarking, 2026 preprint: https://arxiv.org/abs/2606.29493
- SorryDB snapshot: https://sorrydb.org/
