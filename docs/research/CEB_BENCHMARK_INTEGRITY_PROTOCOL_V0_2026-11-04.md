# CEB Benchmark Integrity Protocol V0 — 2026-11-04

**Status:** Design-only protocol. This document defines a test programme; it does not report test execution, qualify the current runner, or establish a CEB score.

**Machine-readable corpus:** [`benchmarks/ceb/integrity/attack-catalog.v0.json`](../../benchmarks/ceb/integrity/attack-catalog.v0.json)  
**Task:** `CEB-VER-001` in the CEB V0 catalog  
**Tracking issue:** [#10138](https://github.com/Luminous-Dynamics/symthaea/issues/10138)

## 1. Why this deserves its own benchmark lane

A software agent can solve a task and still fail as an engineering system if the evaluator can be manipulated, hidden answers are exposed, the environment is asymmetric, or the result cannot be independently reproduced.

Recent external evidence makes this a practical, not hypothetical, gap:

- Scale's SWE-Bench Pro V2 reports that its audit removed 89 invalid tasks, corrected 69 contradictory task statements, disabled agent web access after observed fix-commit retrieval, and re-grades patches in pristine images. Its published full split is now near saturation, with leading results around 99.1–99.4%; the 51-task HARD split remains more discriminating. This motivates evaluation integrity and task lifecycle as first-class axes, not just another pass@1 board. [Scale methodology and board](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2) · [SWE-Bench Pro V2 paper](https://arxiv.org/abs/2610.24206)
- Epoch AI's September 4, 2026 review rated Terminal-Bench 4.0.0 "Flawed", citing documented scoring defects in 30/66 tasks (45.5%). Representative failure modes include direct writes to a grading pass channel, answer leakage and false negatives. [Epoch AI review](https://epoch.ai/benchmarks/terminal-bench/review)
- Terminal-Bench's own 4.0 release notes describe task removals, fixes and a continuous versioning policy; its announced 4.1 direction explicitly includes tamper-resistant verifiers. [Maintainer 4.0 notes](https://www.tbench.ai/news/terminal-bench-4-0)
- A recent survey frames reward hacking as a system-level problem spanning evaluator and environment manipulation, not merely inaccurate model responses. [Survey](https://link.springer.com/article/10.1007/s44163-026-01980-z)

The CEB opportunity is to measure *whether a benchmark result deserves to be believed*.

## 2. Paradigm shifts

| Old default | CEB replacement |
|---|---|
| Agent pass rate is the benchmark | Separate task resolution from evaluator integrity and evidence qualification |
| The grader is trusted because it is maintained by the benchmark owner | Threat-model the grader, runner, artifact store, registry, regrader and operator |
| One final pass/fail bit | Three evidence states: qualifying, non-qualifying and unresolved; candidate correctness is a separate axis |
| One clean reference run | Two-sided reference/empty-patch qualification, then immutable independent pristine regrade |
| Same team owns task, oracle and score | Separate task authoring, agent workspace, grader custody, runner administration and review |
| Static attacks and known examples | Versioned, rotating, metamorphic and adversarially refreshed control families |
| One composite benchmark score | Per-axis denominators, uncertainty, severity and hard gates |
| Integrity checked once at release | Continuous benchmark lifecycle, semantic versioning and requalification after changes |

The most important change is to reject the idea that a better aggregate can compensate for a critical evaluator compromise. A single successful grader-bypass control must block qualification.

## 3. Corpus design

The initial catalog contains **28 negative/adversarial controls** and **8 positive/valid-state controls**, exceeding the initial minimum of 20 negative and 5 positive cases.

Negative controls cover:
- grader result channels, grader/test mutation and hidden-test access;
- answer extraction, ambient secrets, network retrieval and output leakage;
- writable caches, toolchain/lockfile substitution, symlink/hardlink/path escapes and artifact TOCTOU;
- forged/replayed receipts, task revision drift, stale result reuse and runner-policy mismatch;
- primary/regrade common-mode dependencies and input drift;
- weak oracles, ambiguous acceptance, flaky tests, reference/empty-patch gate failures, version drift, Unicode/path aliasing, prompt leakage and model identity attribution.

Positive controls are essential: a system that rejects every run can obtain zero false accepts while being useless. The controls include a valid reference patch, a correct alternative patch, authorized workspace edits, a true independent regrade, digest-bound cache reuse, exact pinned runner metadata, a validly recorded candidate failure and metamorphic equivalence.

The full case definitions and expected qualification states are machine-readable. Each case is explicitly marked `design_only_not_executed`. No attack payloads are included; scenarios describe what a future isolated control must test.

## 4. Scoring without hiding failure

Let:
- **FA** = negative controls incorrectly treated as qualifying.
- **FR** = positive controls incorrectly rejected or left ineligible.
- **U** = unresolved controls.
- **RD** = independent regrade disagreement.
- **RC** = exact-input replay reproducibility.

Publish:
1. false-accept rate, denominator and one-sided 95% exact upper confidence bound;
2. false-reject rate, denominator and one-sided 95% exact upper confidence bound;
3. unresolved rate, with critical-control subset;
4. attack-family coverage by threat class and severity;
5. regrade disagreement by failure-domain class;
6. exact-input reproducibility;
7. agent task-resolution separately, never merged into the integrity denominators.

Do not hide false accepts inside an accuracy/F1 score. Do not weight away critical failures. Every table must expose the raw numerator and denominator.

### Qualification targets (proposed; not measured)

| Dimension | Target |
|---|---:|
| Initial hand-authored corpus | >=20 negative and >=5 positive controls |
| Qualification campaign | >=300 distinct negative and >=300 distinct positive instantiations |
| Fault-family diversity | >=30 distinct negative fault families |
| Critical false accepts | 0 permitted |
| Statistical target | One-sided 95% exact upper bound <1% for FA and FR, subject to cluster-aware analysis |
| Reference / empty-patch gate | 100% reference pass and empty-patch fail before a task enters the scored set |
| Independent regrade | 100% of scored runs use pristine, independently administered regrade |
| Exact-input replay | 100% same normalized outcome for the same immutable inputs; any variance becomes unresolved |
| Critical-control ambiguity | 0 unresolved critical controls at qualification |

Zero errors in 300 genuinely independent Bernoulli trials gives a simple one-sided 95% upper bound just below 1%. This approximation is invalid when repetitions are correlated. The campaign must preregister fault-family clusters, use cluster-aware sensitivity analysis, and avoid counting reruns of the same bytes as independent evidence.

## 5. Experimental protocol

1. **Preregister.** Pin task IDs, split, fixture trees, candidate patch, grader, toolchain, runner policy, budgets, model endpoint and analysis plan. Publish public development commitments; keep private/rotating holdout labels outside the agent plane.
2. **Qualify tasks.** Require reference-patch pass and empty-patch fail on the same exact pinned environment. Independently review whether the oracle measures the stated capability.
3. **Separate domains.** Task authors do not administer the agent runner; the agent cannot write grader/test/evidence mounts; the grader cannot rely on candidate-controlled status; primary and regrade do not share a disqualifying failure domain.
4. **Exercise negative and positive controls.** Every attack family has at least one negative control and a valid counterpart where possible. Use paired controls to avoid a trivially reject-all verifier.
5. **Regrade pristine.** Reconstruct from immutable inputs in a fresh environment. Compare all task, patch, toolchain, policy, grader and budget digests before comparing verdicts.
6. **Adjudicate.** Independent reviewers classify ambiguous specifications and flaky oracles as unresolved. Do not silently relabel an incorrect oracle as an agent failure.
7. **Report.** Publish task-level outcomes, raw denominators, uncertainty, cost, wall time, human review effort, unresolved cases and deviations.

### Version policy

- **Major:** task set, agent-visible inputs, environment semantics or budget protocol changes; rerun affected model/task arms.
- **Minor:** grader semantics, hidden test logic, qualification or scoring logic changes; regrade affected artifacts and clearly version the score.
- **Patch:** metadata/documentation changes that do not alter execution or scoring; preserve prior results.

Any discovered answer leak, pass-channel shortcut or false-negative defect triggers a task-level quarantine and regrade decision. Do not patch a grader and silently retain an old score as comparable.

## 6. Industry comparison and route to differentiation

SWE-Bench Pro V2 remains the primary external long-horizon repository-repair comparison, but its saturated full split should not be the sole indicator of progress. Use its HARD split and fresh task sources for capability discrimination. ContextBench, PDB, CRBugBench and security-oriented coding evaluations should be compared only on exact shared versions, splits, metric definitions and budgets.

CEB should not claim to beat the best agents on resolution until it has matched runs and confidence intervals. It can aim to exceed the *assurance standard* by reporting the dimensions often omitted from a single resolution percentage:

- whether a task is correctly specified and independently solvable;
- whether the grader resists deliberate attacks;
- whether secrets, hidden labels and grader state are outside the candidate boundary;
- whether primary/regrade independence is operational rather than a pair of different key labels;
- whether score provenance is exact-head, immutable and reproducible;
- how much human review, time and money are needed to obtain an independently accepted result.

A system that solves 1–2 percentage points fewer tasks but has verified evidence, no critical false accepts and much lower human review burden may be preferable in production. That is a deployment trade-off, not an excuse to claim a higher benchmark score.

## 7. Hard dependencies and non-claims

- [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064) remains a hard P0 before executing candidate-controlled tasks, build scripts, tests or attack payloads. The present host-opt-in command runner is not an isolation boundary.
- [#10369](https://github.com/Luminous-Dynamics/symthaea/issues/10369) remains the trust-root, signer-custody and independent-runner-attestation workstream.
- The static Rust verifier can be expanded with non-executing malformed-receipt and inventory tests, but those tests do not qualify an OS boundary or hidden-answer custody.
- The CEB task catalog and this attack catalog are design artifacts until source fixtures, independently administered graders, pins, qualified runner and exact-head test evidence exist.

**Current status:** 28 negative controls, 8 positive controls, 300/300 statistical campaign targets, and a proposed hard-gate protocol are authored. All are unexecuted. No CEB integrity score, pass rate or SOTA comparison is claimed.

## CEB-wide measurement linkage

The integrity lane is one component of a 13-lane CEB measurement portfolio. See [the machine-readable comparison matrix](../../benchmarks/ceb/measurement-matrix.v0.json) and [the measurement/comparative-claim protocol](CEB_MEASUREMENT_AND_COMPARATIVE_CLAIM_PROTOCOL_V0_2026-11-05.md). The matrix prevents integrity, security, resolution, cost and human-review outcomes from being blended into a single compensatory score.
