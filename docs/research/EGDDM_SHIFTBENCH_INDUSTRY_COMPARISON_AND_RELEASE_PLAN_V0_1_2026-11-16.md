# EGDDM ShiftBench — Industry comparison and release plan v0.1

**Date:** 2026-11-16
**Status:** research plan; proposed targets are not measured results

## Executive position

EGDDM should not compete with SWE-Bench Pro or Terminal-Bench by trying to post a larger raw patch-resolution percentage on a different task distribution. Those benchmarks measure valuable task capability. EGDDM should complement them by measuring whether engineering agents preserve justified knowledge, causal correctness, evidence validity, and evaluator integrity as systems evolve and observations become incomplete.

The product thesis is a benchmark for **justified engineering performance over time**: solve the work, preserve earlier guarantees, know what changed, distinguish unknown from false, and prove that the measurement itself did not corrupt the result.

## Industry baseline as of 2026-11-16

| Benchmark / study | Published scope or finding | What EGDDM should add |
|---|---|---|
| [SWE-Bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2) | Refreshed public split: 642 tasks across 11 repositories. Its release review removed 89 invalid tasks, corrected 69 instruction/test contradictions, disabled web tools during the agent phase, regrades on pristine images, and applies reference-patch-pass / empty-patch-fail controls. | Extend these integrity practices over multiple shifts, branch histories, causal uncertainty and evidence continuity. |
| [Terminal-Bench 4.1](https://www.tbench.ai/news/terminal-bench-4-1) | Regraded 11,550 stored trials after verifier hardening; 162 changed scores (89 false negatives, 45 false positives, 28 measurement discrepancies). The saved-artifact regrade required no new inference spend. | Make verifier hardening, replay and two-sided controls an explicit benchmark dimension, not only periodic maintenance. |
| [SWE-Milestone](https://proceedings.mlr.press/v306/deng26d.html) | Evaluated 12 frontier models across four frameworks; results fell from above 80% on isolated tasks to at most 38% in continuous settings. | Measure not only task succession but whether evidence and authority remain valid at every intermediate transition. |
| [ChainSWE](https://aclanthology.org/2026.emnlp-main.371/) | 304 chronological issues across 54 Python projects; reported performance degradation reaches up to 70% as chains lengthen. | Add multi-shift causality, rollback, branch, evidence and authority-state semantics. |
| [SWE-Maintain](https://arxiv.org/abs/2610.14768) | 100 three-event episodes from 50 repositories; agent-produced histories lowered later-task resolution by 5.6 percentage points on average and by 14 points when accumulated tests had to remain passing. | Track inherited evidence debt and unsupported exposure, not merely final task success. |
| [VerifierBench](https://verifierbench.github.io/) | Reports 222 tasks, 4,993 hidden cases and 36 model–harness configurations; valid-solution over-rejection is a major issue. | Score false acceptance and false rejection together, including unnecessary abstention. |
| [Oracle-integrity audit](https://arxiv.org/abs/2610.16067) | Fault injection and replay found a state-mutating oracle defect; corrected labels changed measured success from 94.3% to 65.5% across 4,320 episodes. | Qualify the grader, state observer, cache and replay path as first-class benchmark components. |

### What these comparisons do and do not imply

Raw resolution rates across these benchmarks are not comparable: task units, environments, agent harnesses, budgets, hidden-test design and scoring rules differ. SWE-Bench Pro's current leaderboard is a useful static capability reference, while Terminal-Bench provides a useful terminal-workflow reference. SWE-Milestone, ChainSWE and SWE-Maintain establish the need for longitudinal testing. VerifierBench and oracle-integrity research establish that evaluators themselves can fail in both directions.

EGDDM should publish matched cross-benchmark results only where task, model, harness, budget, environment, repeat policy and adjudication protocol are controlled. Otherwise, position the work as a complementary measurement axis, not a claim of numerical SOTA superiority.

## Proposed benchmark architecture

1. **Capability:** task resolution and quality against mature external benchmarks and hidden tests.
2. **Continuity:** multi-shift repair, rollback/reinstatement, preservation of earlier passing guarantees, and cumulative evidence debt (D6.8).
3. **Adversarial integrity:** adaptive exploit-route migration, observer continuity, and legitimate-solve preservation (D6.9).
4. **Causal structure:** partial orders, concurrency versus commutativity, branch isolation and merge requalification (D6.10).
5. **Epistemic calibration:** observation completeness, admissible histories, underdetermination, compromise detection and justified abstention (D6.11).
6. **Measurement integrity:** state-mutating oracles, stale caches, replay mismatch, nondeterminism, observer coverage, false positives and false negatives (D6.12).

D6.7 remains the owner of independently adjudicated impact/common-cause provenance and measured check-cost distributions. These tracks should share provenance conventions and report schemas, but their gold labels and failure denominators must remain separate.

## How to outperform the state of the art

### 1. Target the missing evaluation dimension

Do not chase another leaderboard's pass-rate ceiling. Make the contribution a task family those leaderboards do not claim to measure: justified state and evidence maintenance under interacting change, uncertainty, adversarial pressure and evaluator faults.

### 2. Make success two-sided

A benchmark should require:
- Reference-valid solutions pass.
- Empty/no-op and known-invalid solutions fail.
- Semantically equivalent valid implementations remain accepted.
- Missing evidence produces underdetermination, not fabricated concurrency.
- Critical evidence failures cannot be averaged away by speed or patch success.
- The grader itself leaves measured state unchanged and agrees with independent replay.

### 3. Evaluate the full trajectory

Record every checkpoint, not just the terminal patch. Report critical unsupported exposure over time, false resurrection, recovery latency, preserved guarantees, unnecessary revalidation, and the cost of evidence acquisition. A final green state must not erase a long unsafe intermediate interval.

### 4. Build a reproducible, contamination-resistant leaderboard

- Pin exact model, agent harness, tool versions, runner image, network policy, environment, task split and resource limits.
- Separate public development controls from private rotating holdouts; use post-cutoff tasks and maintain contamination audits.
- Store candidate artifacts, action traces, pre/post state digests, verifier version and replay receipts.
- Grade on fresh/pristine state; independently replay state transitions.
- Repeat stochastic tasks and report clustered confidence intervals by repository, task family and seed.
- Publish both the raw task outcome and the evidence/measurement qualification status.

### 5. Expand beyond Python-only maintenance

After the software-engineering pilot, add Rust, Go, infrastructure-as-code, databases, CI/CD, distributed services and selected physical-system twins. Do not mix these into one leaderboard until task difficulty, evidence capture and adjudication are calibrated by domain.

## Proposed scorecard: a Pareto frontier, not a magic number

Publish separate axes:

- **Task capability:** accepted reference-valid outcomes / eligible tasks.
- **Evidence safety:** critical stale/false-resurrection/future-evidence/unauthorized-claim failures.
- **Causal quality:** false serialization, missing order, false concurrency and confluence accuracy.
- **Epistemic calibration:** identified relations, calibrated abstentions, false completeness and missed compromise.
- **Oracle integrity:** state mutations, replay mismatches, cache contamination, false acceptance and false rejection.
- **Efficiency:** wall-clock time, tokens, tool calls, independent check cost, revalidation churn and recovery latency.

Critical safety failures are hard gates. The leaderboard should expose the safe-performance frontier rather than combine dimensions into one weighted score.

## Proposed acceptance targets

| Area | Target for a qualified release |
|---|---:|
| Deterministic positive / negative / sham known-answer controls | 100% exact |
| Critical false-resurrection acceptance | 0 |
| Critical future-evidence leakage | 0 |
| Critical false-concurrency claims under incomplete observation | 0 |
| Critical branch receipt contamination | 0 |
| Oracle state mutation accepted as valid | 0 |
| Valid-solution false rejection on registered controls | 0 |
| Always-abstain / always-reject baselines passing | Never |

These are proposed release gates for finite registered controls, not a promise of zero failures on all real systems. Real-data rates need confidence intervals and external adjudication.

For a later matched-agent pilot, an aspirational target is +10 percentage points in shifted justified-recovery rate and at least 20% lower total evidence/verification cost than the strongest matched baseline, with no degradation in critical safety gates. Formal thresholds should follow pilot variance and power analysis.

## Corpus scale proposals

| Workstream | Proposed research target | Current status |
|---|---:|---|
| D6.8 multi-shift sequences | 240 matched clean/sequence pairs | Synthetic evaluator prototype; no agent corpus |
| D6.9 exploit migration | Three hardening epochs plus route-migration and condition-removed controls; expand after pilot | Synthetic evaluator prototype; no exploit execution / red-team trials |
| D6.10 partial orders | 360 DAG tasks after validity pilot | Synthetic known-answer fixture; no independent DAG adjudication |
| D6.11 identifiability | 384 tasks after validity pilot | Synthetic pairwise admissible-relation prototype; not a full-DAG possible-world solver |
| D6.12 oracle integrity | 480 audits after validity pilot | Synthetic state-digest/replay contract; no real benchmark fault injection |

These are separate task families and target counts; they are not a current 1,500-task benchmark. Pilot reliability, independent annotation agreement and power analysis determine whether the targets should change.

## Release gates

1. Compile, format, run tests and Clippy for each exact PR head; require exact-head CI evidence.
2. Validate every input/output schema with a standards-compliant Draft 2020-12 implementation.
3. Obtain independent annotations for impact sets, partial orders, admissible histories and oracle-fault labels.
4. Test positive, negative, sham, rollback, restoration, branch, observer-loss and condition-removed twins.
5. Run two independently implemented evaluator paths and compare reports on frozen fixtures.
6. Run blinded, matched agent trials with pinned budgets and repeated seeds.
7. Publish raw task data where safe, excluded-task reasons, failure denominators, confidence intervals, cost distributions, and the exact claim ceiling.
8. Obtain independent replication before claiming SOTA advantage.

## Present claim boundary

As of 2026-11-16, D6.8–D6.12 have draft evaluator/protocol work and synthetic known-answer fixtures. No Rust compilation/test/Clippy/CI, independent gold adjudication, real agent trials, measured cost study, production isolation proof or external replication has been established in this execution environment. The current contribution is a research and evaluator implementation programme, not a validated benchmark ranking.
## Industry update — temporal understanding, roadmap execution and project continuity (2026-11-16)

Recent work makes the competitive boundary more specific:

- [Long-Horizon Software Engineering Agents: From Tasks to Projects](https://longhorizonswe.github.io/) synthesizes the shift from one-off issues to projects that inherit agent-produced state. Its survey reports ChainSWE results of 69.1% in the reference-fix Oracle setting, 49.0% when later work inherits agents' own patches, and 47.4% when dialogue is also retained. The survey emphasizes retained functionality, active-requirement conformance, selective revision and interruption/handoff—not chain length alone.
- [SWE-Time](https://arxiv.org/abs/2611.12083), posted Nov 9, 2026, evaluates temporal repository understanding across 1,661 cases and 65 GitHub repositories. It covers retrospective regression analysis and prospective change-intent understanding; the paper reports 15 coding agents and a highest cited average of 61.9 for Claude Opus 4.8 within Claude Code.
- [RoadmapBench](https://arxiv.org/abs/2605.15846) evaluates 115 long-horizon version-upgrade tasks across 17 repositories and five programming languages. It reports a best task-resolution rate of 39.1% in its evaluated set and a median task involving 3,700 changed lines across 51 files.
- [SWE-fficiency](https://proceedings.mlr.press/v306/ma26l.html) tests optimization on 498 real-workload tasks across nine repositories. Its authors report that top agents achieve, on average, less than 0.23× the expert speedup, showing that real engineering quality includes performance and workload fidelity—not just passing tests.

### What this changes for EGDDM

1. **Temporal comprehension is no longer an empty niche.** SWE-Time explicitly evaluates repository evolution, and SWE-Milestone/ChainSWE cover continuous or dependent work. EGDDM must differentiate through evidence validity, causal identifiability, branch/authority continuity, adversarial changes and evaluator trust—not merely historical repository understanding.
2. **Project continuity requires selective revision.** Retained conversation alone does not preserve project capability. EGDDM should measure which prior guarantees remain valid, which became invalid, and whether agents repair only the affected obligations without stale-evidence reuse.
3. **Broader engineering matters.** RoadmapBench spans five languages; SWE-fficiency demonstrates the value of real workloads and measured performance. EGDDM should add Rust/Go/infrastructure/database tracks after the software pilot, with domain-specific gold adjudication and cost models.
4. **Report richer outcomes.** Alongside resolve rate, report retained functionality, active-requirement conformance, evidence support, graph-level consistency, justified abstention, evaluator integrity, cost and recovery latency. A safe-performance Pareto frontier is more informative than one aggregate rank.

These papers are complementary baselines, not evidence that EGDDM currently outperforms them. Matched trials remain necessary.
## Industry update — interruption, takeover and context compaction (2026-11-16)

Three recent benchmark lines make interruption and cross-agent transfer a concrete adjacent dimension:

- [Handoff Debt](https://arxiv.org/abs/2606.02875) studies 75 source tasks, 181 interruption points and 724 takeover runs per successor model. It reports 20–59% fewer median agent events and 42–63% fewer cumulative prompt tokens for context-bearing handoffs versus repository-only takeover, with solved-rate effects smaller and model-dependent.
- [HSC-Bench](https://arxiv.org/abs/2610.36523) reports 1,039 takeover tasks across six receiving models, separating recovery of repository evidence from continuation of inherited conversation, tool and file-work state.
- [ContinuityBench](https://openreview.net/pdf?id=3N3BzvoLbG) applies controlled interruptions to existing agent benchmarks. It reports average success falling from 41.7% to 28.0% after interruption; structured-summary handoff outperforms summary-plus-full-action-history in 11 of 18 settings. More context is not necessarily better.
- [CompactionBench](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/10/CompactionBench_preprint.pdf) analyzes 13.5 million GitHub Copilot sessions. The preprint reports that 7.8% of sessions invoke context compaction but account for 44.2% of total token use; compaction also reduces prefix-cache reuse.

### D6.14 differentiation

EGDDM D6.14 is tracked separately in [Issue #12470](https://github.com/Luminous-Dynamics/symthaea/issues/12470). Its target is not just successful task takeover. It tests whether the receiver inherits valid evidence, current authority, active obligations and unresolved side effects; rejects stale evidence; reconciles unknown non-idempotent operations before retry; and can verify artifact references without the sender's private transcript. This complements Handoff Debt, HSC-Bench and ContinuityBench rather than claiming to replace them.

The design objective is minimum sufficient, provenance-bearing state that preserves commitments and evidence while limiting rediscovery, over-steering, disclosure and serving cost.