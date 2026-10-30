# PITB v0: industry benchmark gap and qualification roadmap

Date: 2026-10-30  
Status: research and preregistration guidance; **not an empirical model comparison**.

## Executive decision

PITB should not try to beat theorem-prover leaderboards by quoting a higher solve percentage on a different test set. Its distinctive claim should be narrower and more useful: *does transferring a proof insight cause more independently verified, semantically valid, held-out theorem solves per matched unit of compute, after correcting for dependency leakage and incomplete outcomes?*

The next paradigm shift is from **proof accepted** to **knowledge-transfer effect under adversarially audited, semantically tested, statistically qualified conditions**. A Lean kernel proves that a generated term establishes its formal proposition; it does not by itself prove that the proposition matches the intended task, that downstream results remain correct, or that the evaluation split is independent.

## What the current frontier says

These are useful external comparison points, but their denominators and tasks differ. None is a head-to-head result against PITB.

| Reference | Published result or scope | Lesson for PITB |
|---|---|---|
| [Goedel-Prover-V2 (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html) | Reports 90.4% MiniF2F pass@32 in self-correction mode for its 32B prover. | A strong solve-rate headline is already possible on a contest-style benchmark. PITB needs a distinct research-level transfer target and budget-matched baselines, not a claim of superiority from unlike percentages. |
| [Pythagoras-Prover](https://arxiv.org/abs/2606.12594) | Reports 89.8% MiniF2F-Test at pass@32 and 93.0% at pass@2048 for its 32B model; reports 85.0% pass@32 on MiniF2F-ALF. Its training uses the ALF perturbation family. | Report the full quality–compute frontier, including attempts and spend. ALF is a useful mutation stress test, but not a clean unseen-perturbation holdout for a model trained on ALF; publish transformation-family exposure explicitly. |
| [miniF2F-Lean Revisited](https://proceedings.neurips.cc/paper_files/paper/2025/hash/72dad95a24fae750f8ab1cb3dab5e58d-Abstract-Conference.html) | The end-to-end natural-language-to-Lean pipeline reached 70% on corrected miniF2F-v2 versus 40% on the original set in the reported evaluation. The paper attributes substantial loss to statement/formalization mismatch. | Score the exact human-intended target, not just a proof of a potentially mismatched formal statement. |
| [T²: Benchmarking Testing in Automated Theorem Proving (ACL 2026 Industry)](https://aclanthology.org/2026.acl-industry.150/) | 2,206 problems from five Lean repositories, paired with an average of 41 successor theorems; the paper reports 38.9% best testing accuracy despite higher compilation rates. | Add semantic successor/integration tests alongside kernel acceptance; compilation and semantic correctness are different axes. |
| [TCSAlgBench (2026 preprint)](https://arxiv.org/abs/2609.35606) | 398 research-level TCS theorem challenges from 138 STOC/COLT 2026 papers; published comparisons use five runs and matched model-call opportunities, with highest reported five-run verifier-accepted coverage of 23.6% in one model comparison and 25.4% in a separate agent-workflow comparison. | PITB should be repeated, budget-matched, research-level and refreshable. Five-run coverage is not directly comparable to single-run or pass@k metrics. |
| [Formal Conjectures (2026)](https://arxiv.org/abs/2605.13171) | The paper's reported snapshot contained 2,615 Lean 4 statements, including 1,029 open research conjectures. The live [project site](https://google-deepmind.github.io/formal-conjectures/) now lists 4,466 statements, including 1,613 open and 2,853 solved. | A benchmark's denominator changes over time. Pin exact repository commit, snapshot date and target IDs; never use a live landing-page count as the frozen evaluation denominator. Pair the reproducible cohort with a rolling time-based holdout. |
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

1. **Finish the evidence—not another schema.** The deterministic component builder, signed-receipt verifier, run-config v3 artifact pins, analyzer-side receipt re-verification, and exact component reconstruction are implemented. The latest verifier also rejects untrusted/unsigned edge reviewers and resolved-edge reviewers from the same organization.
2. **Secure real coverage evidence.** Provision a reviewer trust root independently of the experiment operator, complete the 372-family relationship inventory, and retain the dual-review/search logs. Synthetic keys and schema fixtures do not qualify the real inventory.
3. **Execute exact-head validation.** The focused workflow now invokes Rust tests for the analyzer and component builder/verifier. The current branch head has no reported status checks or PR-triggered workflow runs, so those tests remain unverified; do not substitute static inspection for a pass.
4. **Promote only from independently reconciled evidence.** The builder must emit a qualified graph only when signed coverage verifies and no hard component crosses splits. The analyzer separately enforces component membership, the 20-component floor, missing-row accounting and the statistical gates; the floor alone is not power.
5. **Add semantic successor tests to outcome receipts.** Every accepted target needs exact statement correspondence, a trusted proof check, declared axiom policy and meaningful downstream checks where available. Never infer semantic correctness from compilation alone.
6. **Run the matched-budget causal experiment, then replicate.** Compare typed insight with the validation-selected strongest baseline (including strong premise retrieval) under fixed model/checker/tool/call/token/time/cost budgets. Publish paired component-clustered intervals, the success–cost frontier and every failure/missingness category. Follow with a temporal or fresh-repository cohort.

The targets above are proposed qualification gates, not achieved results. PITB currently has no measured transfer-utility result and no claim of beating the prover state of the art.
 
## References

- Goedel-Prover-V2, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html
- Pythagoras-Prover, 2026: https://arxiv.org/abs/2606.12594
- miniF2F-Lean Revisited, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/72dad95a24fae750f8ab1cb3dab5e58d-Abstract-Conference.html
- T², ACL 2026 Industry: https://aclanthology.org/2026.acl-industry.150/
- TCSAlgBench, 2026 preprint: https://arxiv.org/abs/2609.35606
- Formal Conjectures paper snapshot: https://arxiv.org/abs/2605.13171
- Current Formal Conjectures live catalogue (a moving target; pin its source revision): https://google-deepmind.github.io/formal-conjectures/
- TheoremBench, 2026 preprint: https://arxiv.org/abs/2606.09450
- Faults in Our Formal Benchmarking, 2026 preprint: https://arxiv.org/abs/2606.29493
- SorryDB snapshot: https://sorrydb.org/

## Literature refresh: structure-aware retrieval, faithful proof decomposition, and repository transfer (2026-10-29)

The latest relevant work strengthens the case for PITB's central hypothesis—but also raises the bar for its baselines. PITB should not compete only against lexical or embedding similarity when newer systems retrieve through dependency graphs or align natural-language and formal proofs.

| Work | Relevant published result or scope | PITB implication and comparison boundary |
|---|---|---|
| [LeanGraphSearch (Oct. 15, 2026)](https://arxiv.org/abs/2610.22339) | Its project report shows MathlibQR Recall@100 rising from 82.7% for LeanSearch v2 Standard to 94.7% for LeanGraphSearch Standard (+12.0 points); MathlibMPR complete-route coverage rises from 39.1% to 42.8% (+3.7 points). The paper also reports downstream prover-loop gains in its own setup. | Include as a candidate for the strongest retrieval arm when PITB task format and source corpus are compatible. Its score is not a PITB result. Freeze one retriever for the complete arm; do not mix systems target-by-target. [Project results](https://pluslabnlp.github.io/LeanGraphSearch/). |
| [ProofBridge (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/16f5b8d916f13534d11c50adf60914e4-Abstract-Conference.html) | Cross-modal joint embeddings, retrieval-augmented fine-tuning and iterative repair; reports up to 3.28× Recall@1 over an all-MiniLM-L6-v2 baseline and +31.14% semantic correctness / +1.64% type correctness at pass@32 versus Kimina-Prover-RL-1.7B in its curated miniF2F-Test-PF evaluation. | Strong related work for cross-modal retrieval and statement-faithfulness measurements. Its model, curated task set and pass@32 budget differ, so use it as a design precedent or separate formalization track, not a numeric PITB comparator. |
| [ProofFlow (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5fce5198dbf92d5ff45f74504431e2ff-Abstract-Conference.html) | Uses a proof-step dependency DAG and lemma-by-lemma formalization; reports ProofScore 0.545 versus 0.279 for full-proof formalization and 0.046 for step-proof formalization on its 184-problem annotated benchmark. | Treat the insight as an executable dependency-preserving proof transformation, not merely a paragraph. Where gold step graphs exist, report structural coverage/fidelity separately from the primary exact-target solve endpoint. These ProofScore values are not comparable to PITB's solve rate. |
| [VeriSoftBench (2026)](https://arxiv.org/abs/2602.18307) | 500 Lean 4 proof obligations from real formal-methods repositories, preserving project-specific context and cross-file dependencies; Mathlib-tuned provers transfer poorly to this setting in the reported experiments. | A future independent-repository cohort is necessary to test whether insights transfer beyond one research-paper catalog and its local proof ecosystem. |
| [LeanLean (2026)](https://www.sri.inf.ethz.ch/publications/rognvaldsson2026leanlean) | Repository-scale proof compression over 64 real Lean repositories (up to 174k lines), with a 12-hour budget; the best reported model averages a 48.3% compression score. | Add an adjacent proof-refactoring/lemma-reuse track, preserving main-theorem validity and dependencies. Do not compare compression scores to exact-target solve rates. |
| [ProofOptimizer (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/60917970c4257e50aff60d1d7bde2edf-Abstract-Conference.html) | Reports proof-length reductions of 87% on miniF2F, 57% on PutnamBench and 50% on Seed-Prover's IMO 2025 proofs in its own setup. | Add proof concision/check time as a secondary engineering outcome only after semantic success is held constant; shorter code is not automatically a stronger proof or a better transfer result. |


## Additional frontier check — cross-prover portability (2026-10-30)

[ITPEval](https://arxiv.org/abs/2607.19407) adds an important adjacent frontier: formal statement and proof translation across Lean 4, Rocq, Isabelle, and HOL Light. It covers 1,560 source files and 6,848 theorems across a controlled tier and a real-library ecosystem tier. The paper reports up to 29.1% statement-translation pass@1 and 10.5% proof-translation pass@1 across its directed prover pairs; its Lean statement-equivalence check also shows that a subset of kernel-accepted translations can fail a stronger equivalence test. These figures are not PITB results and are not directly comparable to the v0 same-Lean transfer endpoint.

**V1 research opportunity — prover-neutral proof-insight IR:** represent the mathematical move in a kernel-independent intermediate form containing object sorts, assumptions, quantifier scope, invariant direction, derived obligations, and counterconditions. Then compile or instantiate that contract into each prover's native type/term language and independently check:
- statement and assumption correspondence;
- the obligations discharged by the move;
- native kernel acceptance and downstream semantic checks;
- portability loss, adapter effort, and proof cost;
- whether the same insight yields incremental utility over a native same-prover baseline.

Keep this as a separate portability cohort after PITB v0 is qualified. Lean, Rocq, Isabelle, and HOL Light differ in foundational logic, type systems, and library ecosystems; failure to translate may reflect foundation or library mismatch rather than failure of the underlying mathematical strategy. A cross-ITP result must therefore distinguish logical incompatibility, missing library support, translation failure, and genuine strategy non-transfer.

This gives PITB a longer-term route from a Lean-specific retrieval/transfer benchmark toward a testable theory of **prover-independent mathematical knowledge representation**. It is a research direction, not an implemented feature or current benchmark claim.

### Revised benchmark decision

1. **Primary:** paired, component-clustered change in exact-target semantic-gate success from typed insight versus the strongest validation-selected retrieval baseline, at identical budgets.
2. **Selectivity:** compatible-insight recall, incompatible-insight false-accept rate, abstention, and calibration. A system that declines every transfer does not pass.
3. **Mechanism:** proof-obligation transitions and dependency-preserving lemma structure; component ablations establish which contract fields cause gains.
4. **Efficiency:** full verified-solve success–cost frontier, plus proof-check time and proof size where comparable.
5. **Generalization:** sealed temporal challenges and an independently maintained repository cohort; never retroactively merge them into the frozen v0 denominator.

**Qualification rule:** do not claim to beat SOTA because PITB's score is higher on a different task set. A defensible superiority claim needs a matched-task or clearly labelled task-specific comparison, same budgets and checker, independent target semantics, uncertainty intervals, a completed mutation gate, and independent replication. PITB remains a proposed benchmark with no measured transfer-utility results.

### New calibration finding — component-size and estimand stress (2026-10-29)

A separate research-only NumPy screen now tests six synthetic connected-component size profiles (20–60 components), with component effects either independent of size or correlated with log component size. It runs 1,000 studies per cell and 299 whole-component bootstrap replicates per study. Coverage ranged from 90.7% to 95.3%; in the 20-component independent-effect weak-null case, the lower-bound rejection rate was 4.3% (Monte Carlo Wilson 95% interval 3.21–5.74%) and the primary +10pp utility gate falsely passed 3.8% of studies (2.78–5.17%). These simulations are conditional stress evidence only, not a result for real PITB data. See [the full method and table](pitb-component-size-stress-2026-10-29.md) and [the executable research script](../../scripts/research/pitb_component_size_stress_v0.py).

The key estimand finding: with treatment effects correlated with component size, a true target-weighted average effect of zero coexisted with a mean equal-component effect between -4.94 and -6.83 percentage points in the unequal-size profiles. Freeze target weighting as the primary operational estimand; report equal-component weighting only as a distinct diagnostic. The 20-component rule remains a structural floor, not a power or calibration certificate. No Rust test pass, independent statistical review or proof-transfer result is implied.

### Follow-up calibration — interval-method comparison (2026-10-29)

A second synthetic screen compared the whole-component percentile bootstrap with a weighted cluster-sandwich t interval over the same 12 component-size/effect-structure cells. Coverage ranged from 91.4–94.6% for the percentile bootstrap and 92.6–96.0% for the sandwich t method. Neither dominated: the sandwich t interval improved coverage in several unequal-size scenarios, but in the giant-component / size-independent cell its null lower-bound rejection rose to 4.2% (95% Monte Carlo Wilson interval 3.12–5.63%) and it produced narrower intervals. The 20-component size-correlated case also shows that nominal coverage remains fragile. Full details are in [the interval-method report](pitb-inference-method-comparison-2026-10-29.md) and executable [comparison script](../../scripts/research/pitb_inference_method_comparison_v0.py).

**Decision:** keep the current production interval provisional; do not promote the CR1-style t comparator based on this screen. Freeze the target-weighted estimand, add selective-missingness and alternative size-generating scenarios, and obtain independent statistical review. No production Rust change or proof-transfer result is implied.

### New robustness capability — partial identification of unknown paired outcomes (2026-10-29)

The branch now includes `pitb_missingness_bounds_v0.py`, a research-only diagnostic that returns sharp worst-case finite-cohort bounds for the paired binary outcome difference when a frozen target's control and/or treatment result is genuinely unknown. It requires the complete frozen target manifest and exactly one row per target; an absent row or malformed outcome status is rejected. `success`, `non_solve` and `unknown` are distinct so operational failures cannot be hidden by reclassifying them as missing evidence. The output includes a 25-point assumption grid for unknown outcomes, but always blocks confirmatory use when any unknown remains.

On the synthetic eight-target negative-control fixture, only three of eight target pairs had both outcomes observed. The complete-pair estimate was 0pp while the sharp full-cohort identification interval was -37.5pp to +50pp; this correctly demonstrates that incomplete evidence does not support a narrow claim. This calculation is a finite-cohort identified set, not a confidence interval, and it does not replace the existing fail-closed outcome/receipt gate.

Methodological grounding includes general partial-identification research ([Manski overview](https://www.sciencedirect.com/science/article/pii/S0888613X04001124)), systematic missing-outcome tipping-point analysis ([2025 paper](https://pubmed.ncbi.nlm.nih.gov/40140629/)), and 2026 cluster missing-outcome sensitivity methodology ([open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13108325/)). PITB is not a randomized clinical trial; these are methodological references only. The local authoring copy passed seven unit tests, and an exact check was run on the fixture and missing-row negative control. The newly added GitHub Actions step is still pending exact-head execution, so no CI pass is claimed.


## Literature and design refresh — 2026-10-30

The latest comparison strengthens the case for separating **proving**, **retrieving**, **semantically preserving**, and **constructively solving**. They are related capabilities, but their reported metrics are not interchangeable.

| Frontier | Current evidence | How PITB should respond |
|---|---|---|
| High-budget prover capability | Goedel-Prover-V2 reports 90.4% MiniF2F pass@32 for its 32B self-correction setup and 86 PutnamBench problems at pass@184 in the ICLR 2026 paper. | Treat this as a strong proof-generation reference, not a PITB baseline unless the exact same tasks, toolchain, compute budget and checker are rerun. Do not claim PITB superiority from a score on a different corpus. |
| Global premise retrieval | LeanSearch v2 reports nDCG@10 = 0.62 versus 0.53 for the next system in its Standard retrieval benchmark. On 69 research-level Mathlib theorems it recovers 46.1% of ground-truth premise groups within ten candidates; a fixed downstream prover loop reports 20% proof success versus 16% for the next retriever and 4% with no retrieval. | Arm I must remain a pinned, competitive global-premise retrieval comparator. Freeze retriever version, corpus/index digest, candidate budget and context policy. Measure both complete premise-set coverage and downstream semantic-gate success. |
| Semantic theorem correctness | T² evaluates dependent successor theorems over five Lean repositories: 2,206 problems, about 41 successors per target on average. Its best reported testing accuracy is 38.9%, materially below the high compilation success observed in the study. | Preserve the dual metric: trusted-kernel acceptance and dependent-successor semantic acceptance. A compiled proof is not automatically a semantically faithful theorem. |
| Long theorem structure | TheoremBench provides nearly 100 classical Lean theorems in plain-main and premised forms, plus theorem-level coverage and token-efficiency metrics. Its analysis finds a bias toward easier subtheorems and long, inefficient tactic traces. | Report proof-obligation coverage, solved-subgoal structure, theorem completion, tokens and verified cost per solve. Avoid counting many trivial subgoals as broad theorem competence. |
| Constructive problem solving | D-FPS separates solving from proving by representing an unknown answer as a metavariable plus proof obligation; it proposes RPE for semantic answer equivalence and supplies three benchmarks with over 1,000 problems. | Keep witness discovery as a separate track. The system must produce the witness and prove the specification, not merely prove a supplied proposition. |
| Benchmark validity | The ICML 2026 audit of five Lean benchmarks/forks reports 4,833 findings, including 398 mechanically certified issues such as counterexamples, vacuous statements and unsound axioms. | Make statement fidelity, dependency/axiom policy, mutation kill rate, and harness negative controls reported benchmark properties—not hidden assumptions. |

Primary sources: [Goedel-Prover-V2, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html); [LeanSearch v2](https://arxiv.org/abs/2605.13137); [T², ACL 2026 Industry](https://aclanthology.org/2026.acl-industry.150/); [TheoremBench](https://arxiv.org/abs/2606.09450); [D-FPS, ICML 2026](https://proceedings.mlr.press/v306/liu26gl.html); [Faults in Our Formal Benchmarking, ICML 2026](https://proceedings.mlr.press/v306/ammanamanchi26a.html).

### Strongest paradigm shift for PITB

The distinctive product should not be “another prover” or “another proof-summary retriever.” It should be an **auditable theorem-knowledge transfer system** where every proposed insight behaves like an executable, typed contract:

1. **Represent proof knowledge as a dependency-aware obligation graph.** Each insight records source/target object types, hypotheses, invariant direction, quantifier scope, intermediate obligations, evidence anchors, and known failure boundaries. It should be possible to see exactly which target obligation an insight claims to discharge.
2. **Make transfer selective and counterfactual.** Before using an insight, check its preconditions; when a premise or invariant is missing, abstain or explicitly re-plan. Evaluate compatible positives and adversarial near-miss negatives, including polarity, quantifier, domain, and hidden-premise mutations. A high similarity score must never substitute for compatibility.
3. **Measure causal utility, not attribution by anecdote.** Randomize or pair the same eligible targets across no-insight, retrieval-only, typed-insight, and full treatment under identical model/checker/tool/token/time/cost caps. The primary effect remains the difference in exact-target semantic-gate success over the frozen target denominator, with audited connected components as the inferential unit.
4. **Prove preservation downstream.** Retain exact target statement identity, checker and axiom receipts, two-reviewer statement correspondence, and successor/integration tests. Record raw proof compilation separately from semantically accepted proof outcomes.
5. **Treat witness synthesis as a different endpoint.** The constructive track should score witness validity, proof obligations, and RPE/formal answer equivalence. Do not blend these into a theorem-pass percentage.
6. **Optimize the full verified-solve frontier.** Report success against model calls, generated tokens, wall time, checker time, memory and spend. A shorter proof or a lower-cost run is useful only when the semantic and correctness gates remain intact.

### Concrete industry comparison rule

Use external results in three separate ways:

- **Contextual frontier:** publish MiniF2F, PutnamBench, TheoremBench, D-FPS, T² and retrieval results with their own task counts, model versions, pass@k settings, context and budgets. These are contextual references, not a single leaderboard.
- **Matched local baselines:** rerun the strongest compatible retriever (LeanSearch v2 or a documented equivalent), BM25/lexical, syntax/AST, dependency-graph retrieval, proof-only and no-retrieval arms using the exact PITB targets and caps. If an external service cannot expose deterministic versioning or budgets, label it contextual rather than confirmatory.
- **Independent replication:** after a clean source/relationship audit, replicate on a later release snapshot or a separately maintained Lean repository. This is the evidence needed to claim generalization beyond the one pinned catalogue.

### Immediate decision — finish the audit before another model experiment

The 30 October code addition makes the next human dependency explicit: `pitb_build_coverage_review_docket_v0.py` generates 744 pending review obligations (two independent passes for each of the 372 frozen families), and the CI workflow checks deterministic regeneration and complete family accounting. It does not discover dependencies, adjudicate edges, authenticate reviewers, or qualify the holdout. The real relationship inventory, independently governed trust root and signed coverage receipt remain absent.

Therefore the next meaningful milestone is not another retrieval arm, schema, or synthetic simulation. It is completing the two real full-universe review passes, preserving their evidence, adjudicating all candidates conservatively, and signing the exact edge payload with independently provisioned reviewer identities. Only after the resulting connected-component manifest has been independently reconciled should the current nine-arm matched-budget pilot proceed.

**Current status remains unchanged scientifically:** no real PITB transfer outcomes, no measured effect, no completed real leakage audit, and no claim to beat state of the art. The latest docket tooling and workflow are committed on the draft PR; no exact-head CI run has been observed.
