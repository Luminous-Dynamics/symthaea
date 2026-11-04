# CEB Engineering Quality Scorecard V0 — 2026-11-04

**Status:** research addendum / design-only  
**Related:** [CEB-CAUSE-001 #10366](https://github.com/Luminous-Dynamics/symthaea/issues/10366), [CEB-DEBUG-QUALITY-001 #10385](https://github.com/Luminous-Dynamics/symthaea/issues/10385), [CEB-CAUSE-INTEGRATION-001 #10387](https://github.com/Luminous-Dynamics/symthaea/issues/10387), [protocol PR #10367](https://github.com/Luminous-Dynamics/symthaea/pull/10367)  
**Claim ceiling:** this is a benchmark portfolio and experimental design. It is not a CEB score, an implementation claim, or evidence that Symthaea outperforms any published system.

## 1. Strategic correction: do not optimize one coding score

The current frontier makes a single "coding success rate" an inadequate research target.

- SWE-Bench Pro V2 full is near saturation on its refreshed 642-task public split: Scale reports Opus 5 + Claude Code at 99.40 ± 0.40, Gemini 4 Argon + mini-SWE-agent at 99.22 ± 0.45, Fable 5.1 at 99.10 ± 0.50, and GPT-6 Astra + Codex at 96.90 ± 1.10. The September 22 refresh removed 89 invalid tasks from 731 and added a pristine-image regrade plus a reference-pass / empty-patch-fail admission gate. It still has residual trusted-relay and verifier-execution risks. Public leaderboard success does not establish private-repository generalization.
- SWE-Bench Pro V2 HARD is a separate 51-task subset with a much wider spread: the November 2 snapshot reports Opus 5 at 98.0, Fable 5.1 at 92.2, Gemini 4 Argon at 92.16 and GPT-6 Astra at 90.2. Keep its sample size and uncertainty visible; it is a useful discriminative slice, not a replacement for private holdouts.
- SWE Atlas - Codebase QnA measures deep repository comprehension before editing: 124 tasks across 11 production repositories and four languages. Its published leader table reports Opus 5 at 63.17 ± 5.01, Fable 5.1 at 59.95 ± 4.85 and GPT-6 Astra at 59.14 ± 4.88. The page also notes only about 30% pass at its strictest rubric threshold. This is a separate scorecard from patch resolution.
- SWE Atlas - Refactoring uses 70 tasks across 10 production repositories and six languages. It covers decomposition, interface evolution, extraction and relocation. Its page reports that solutions are roughly twice the changed lines and 1.7x the changed files of SWE-Bench Pro, with leading performance below 50%. The grader checks preserved behavior and must-have maintainability, cleanup and anti-pattern rubrics.
- SWE-fficiency has 498 repository-level performance tasks across nine data-science, ML and HPC repositories. Published leading agents average less than 0.23x validated expert speedup. This exposes a large optimization gap even when code generation and issue resolution are strong.
- Precise Debugging Benchmark (PDB) separates bug-level recall from edit-level precision. Its April 2026 Microsoft Research summary reports >76% unit-test pass rates but <45% edit precision for the evaluated frontier systems. Passing tests can conceal broad, unnecessary or collateral edits.
- HiL-Bench contains 300 tasks across SWE and text-to-SQL, with 1,131 blockers and a 200-public / 100-held-out split. It reports ASK-F1 (question precision × blocker recall harmonic mean) and Pass@3. The best reported combined Pass@3 is 24% when agents must decide when to ask, compared with 75–91% when all information is provided upfront. The gap isolates selective-escalation judgment rather than raw task capability.
- AgentAbstain contains 263 near-identical act/abstain pairs in 42 environments and eight abstention scenarios. Its best reported paired accuracy is 59.5%. Pairing prevents always-act or always-refuse strategies from gaming the benchmark.
- SRE-Bench is a distinct source-free reverse-engineering track: 262 binary instances from 19 private programs and 1,572 deterministic task grades. The benchmark page updated October 2026 reports GPT-6-Astra fully solving 82.3% of graded instances in the public-facing setting, but only 181/262 instances were graded; in the unconstrained setting it reports 88.0% pass@1 and 99.2% pass@4. The strong conditional result must be reported beside its coverage and refusal denominator, not as an unconditional 262-task solve rate.
- Terminal-Bench 4.0 remains diagnostic only: Epoch AI labels the release flawed and reports publicly noted scoring defects in 30/66 tasks. Vals AI's October 29 Mini-SWE-agent results are third-party measurements, not official maintainer leaderboard results.
- CursorBench 4.0 measures ambiguous multi-file tasks from actual Cursor sessions. Its current table reports Opus 5.5 Max at 57.8%, $13.43, 218,363 tokens and 185 steps per task. Retain cost and trajectory measures, not just the score.

**Decision:** maintain a vector of independent capabilities. Do not average these benchmarks into an omnibus "engineering intelligence" score. They measure different constructs, use different task populations, harnesses, graders and resource budgets.

## 2. CEB capability portfolio

| Capability | External reference | CEB's independent measurement |
|---|---|---|
| Issue resolution | SWE-Bench Pro V2 Full / HARD; SWE-rebench | Independently accepted repair on fresh, sealed repositories |
| Repository comprehension | SWE Atlas Codebase QnA | Typed graph, runtime trace explanation, fact provenance, uncertainty and retrieval |
| Refactoring quality | SWE Atlas Refactoring | Behavior-preserving decomposition, extraction, relocation and interface evolution |
| Test-oracle quality | SWE Atlas Test Writing; PDB | Mutation sensitivity, independent oracle validity, bug recall and edit precision |
| Causal debugging | PDB + CEB-CAUSE | Hypothesis calibration, predicted intervention outcomes, experiment utility and verified repair |
| Performance engineering | SWE-fficiency | Candidate/expert speedup, semantic equivalence, resource/latency trade-offs |
| Help-seeking | HiL-Bench | Question precision, blocker recall, ask timing and successful conversion of answers into a solution |
| Safe abstention | AgentAbstain | Paired act/abstain accuracy, selective risk and false-action rate |
| Reverse engineering | SRE-Bench | Source-free semantic recovery and behavior prediction with explicit task coverage |
| Evaluator integrity | Epoch AI benchmark review / CEB-VER | False accepts, false rejects, leakage, mutation resistance and independent regrading |

Use released datasets and adapters only after license, terms, runner contracts, hidden-label custody and evaluator semantics are reviewed. A local synthetic pilot is not a reproduction of the published benchmark.

## 3. Paradigm shifts

### 3.1 From "tests pass" to a repair-quality envelope

Require every repair to satisfy all of:
- behavior contract / hidden acceptance;
- protected invariant preservation;
- relevant mutation controls;
- no new regressions;
- edit-scope justification;
- independently reproduced result.

For controlled injected bugs, report PDB-style bug recall and edit precision against an independently curated *admissible repair set*. For natural repository tasks, do not equate textual overlap with a reference patch to correctness; judge behavior, invariants and collateral effects.

A tiny patch that leaves the defect unresolved must not score well. A semantically necessary refactor must not be penalized just because it differs from one reference diff.

### 3.2 From "ask human" as a fallback to active information acquisition

A computational engineer should distinguish:
- information discoverable from source, tools, traces or controlled experiments;
- information that is missing but inferable only after a cheaper probe;
- information that exists only with a human or authority-holder;
- information that is contradictory or not safely resolvable.

Reuse the HiL-Bench and AgentAbstain pair design principles. A good question must target a real blocker; question spam must be penalized. A good abstention must withhold the side effect and communicate the missing information / approval.

### 3.3 From optimization by intuition to expert-normalized improvement

Compare candidate speedup to a validated expert baseline under the same workload and environment:

candidate_speedup / expert_speedup

Keep absolute performance, baseline variance, semantic equivalence, resource consumption and tail latency visible. Use paired randomized runs, fixed warm-up, stable hardware and noise thresholds. Do not claim an optimization from a single run.

### 3.4 From repository index to executable system comprehension

SWE Atlas QnA shows that repository understanding is a separate bottleneck from patch generation. Symthaea's repository twin should therefore answer questions with source/compiler/runtime evidence, trace the executed path, predict consequences and identify unknowns. A correct answer without evidence provenance is not sufficient for the internal CEB scorecard.

### 3.5 From one leaderboard to a quality/cost/risk frontier

For each capability, publish:
- correctness / quality;
- coverage and abstention;
- regression and false-accept rates;
- resource cost and latency;
- human review and escalation burden;
- transfer to sealed repositories/fault families;
- assurance tier and limitations of the evaluator.

Do not collapse these into one number. Report Pareto-dominant configurations and the cost of achieving a given quality/risk point.

## 4. Matched industry comparison design

For every CEB slice, compare:
1. deterministic or conventional tool baseline where applicable;
2. foundation model with minimal fixed harness;
3. the same model with a strong coding-agent harness;
4. the same model plus Symthaea repository twin / causal reasoning / evidence workflow;
5. Symthaea-native generation path where the capability is genuinely implemented without external model fallback.

Pin model revision, harness/scaffold, prompt, tools, task revision, execution image, toolchain, network, budgets and grader. Randomize task order. Test model × scaffold interactions when feasible. Attribute any improvement to the component actually changed.

The unit of efficiency should be **cost per independently accepted engineering outcome**, not cost per attempted patch. Also report human review minutes, experiment count and time-to-correct-result.

## 5. Statistical and benchmark-maintenance protocol

- Use paired task assignments and repeated runs for stochastic configurations. Three independent seeds are the minimum proposal; five are preferred when affordable.
- Report pass@k separately from pass-all-k / reliability. Pass@k measures whether a system can succeed at least once; it is not a reliability claim.
- Use paired confidence intervals and cluster bootstrap by repository and fault family. Variants derived from the same template are correlated; do not count them as independent observations.
- Pre-register primary outcomes, practical margins, exclusions, sample-size reasoning, missingness, stop rules and sealed confirmation data.
- Include positive and negative admission checks: reference solution must pass; empty patch must fail; known-bad, regression-inducing and evaluator-mutating controls must fail.
- Preserve all task dispositions in published denominators: pass, fail, refusal, abstention, unsupported, tool failure, timeout, ungraded and not attempted.
- Review rubric quality for prescriptiveness and inter-rater agreement. A rubric must describe observable behavior, not force an implementation identical to the reference patch.
- Retire saturated tasks and version the evaluator. Preserve old task/grader snapshots for longitudinal comparison; never silently replace them.
- Use human review for ambiguous task validity and grader disputes; do not use an uncalibrated LLM judge as the sole correctness oracle.

## 6. Proposed qualification gates

Prospective internal targets only; not achieved results or industry standards.

| Slice | Initial target | Sealed/transfer target |
|---|---:|---:|
| Natural issue resolution | +5 percentage points over strongest matched baseline | +10 pp over matched baseline |
| Controlled seeded-bug recall | >=0.90 | >=0.95 |
| Controlled edit precision | >=0.80 | >=0.90 |
| Repository graph syntax P/R | >=0.95 | >=0.97 |
| Supported semantic relation P/R | >=0.90 | >=0.95 |
| Causal experiment outcome calibration | Brier/log score baseline published | significant paired gain vs baseline |
| Performance candidate/expert speedup ratio | >=0.50 | >=0.80 |
| Selective escalation ASK-F1 | >=0.65 | >=0.75 |
| Act/abstain paired accuracy | >=0.75 | >=0.85 |
| High-impact false-action rate | <=1% | <=0.5% |
| Calibration ECE | <=0.05 with adequate holdout | <=0.03 |
| Cost per accepted outcome | >=20% reduction | >=30% reduction |
| Published evidence receipts | 100% | 100% |
| Pristine independent regrade | 100% of scored tasks | 100% of scored tasks |

These targets must be accompanied by confidence intervals, task counts, scenario-level floors and false-accept / false-reject analysis. No zero-risk claim follows from observing zero failures in a finite control set.

## 7. Immediate work plan

### No-runner tranche
- Update #10366/#10367 with this scorecard and explicit benchmark adapters.
- Define a compact non-executable pair/task authoring template for causal intervention and act/abstain twins.
- Define an admissible-repair annotation guide for controlled PDB-style seeded tasks.
- Define an expert-workload measurement template and metrology checklist for the performance slice.
- Write a preregistration template with paired/clustered analysis and missingness accounting.
- Audit the CEB task catalog so existing Rust/performance/system lanes are extended, not duplicated.

### Isolation-dependent tranche
- Execute candidate code, build scripts, tests and real workloads only after #10064 qualifies the isolation boundary.
- Establish independent trust roots and regrade failure domains under #10369.
- Reconcile generic accounting, task profile and DSSE attestation under #10370 before publishing signed results.

## 8. Sources and dated snapshots

- Scale SWE-Bench Pro V2 (refreshed September 22, 2026): https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full
- SWE-Bench Pro V2 HARD (November 2 snapshot mirror): https://isitsota.com/benchmarks/swe-bench-pro-hard
- Scale SWE Atlas Codebase QnA (updated July 28, 2026): https://labs.scale.com/leaderboard/sweatlas-qna
- Scale SWE Atlas Refactoring: https://labs.scale.com/leaderboard/sweatlas-refactoring
- Scale HiL-Bench: https://labs.scale.com/leaderboard/hil
- SWE-fficiency (ICML 2026): https://proceedings.mlr.press/v306/ma26l.html
- Precise Debugging Benchmark (April 2026): https://www.microsoft.com/en-us/research/publication/precise-debugging-benchmark-is-your-model-debugging-or-regenerating/
- AgentAbstain (NeurIPS 2026): https://agentabstain.github.io/
- Columbia DAPLab SRE-Bench (updated October 2026): https://daplab.cs.columbia.edu/benchmarks/sre-bench/
- Terminal-Bench 4.0 Epoch AI review (September 4, 2026): https://epoch.ai/benchmarks/terminal-bench-4/review
- Vals AI Terminal-Bench 4.0 (updated October 29, 2026): https://www.vals.ai/benchmarks/terminal-bench-4
- CursorBench 4.0: https://cursor.com/cursorbench
- NeurIPS Agentic Benchmark Checklist: https://proceedings.neurips.cc/paper_files/paper/2025/hash/f316275b44ee2de533102913828a8107-Abstract-Datasets_and_Benchmarks_Track.html

All leaderboard values are snapshots of specific tasks, dates, model configurations and harnesses. Re-fetch before public comparisons. CEB claims must never be presented as directly comparable unless task contracts, budgets, graders and coverage are matched.
