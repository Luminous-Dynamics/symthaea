# AR-5B5 Anytime Evidence-Cost Attainment Protocol v0.1

**Status:** research specification; not implemented, compiled, or benchmarked  
**Tracking:** Issue #9469; implementation branch for PR #9429  
**Scope:** evaluation trajectories, target-attainment receipts, heterogeneous measured cost, and verification-aware comparison

## 1. Why this tranche exists

The current AR-5B5 comparison harness replay-verifies NSGA-II, MOEA/D-DE, and SMS-EMOA reports, then compares their final archives under matched problem, seed, objective definitions, reference data, and objective-evaluation count. This is a sound fixed-budget starting point, but it discards the path taken to reach the final archive.

A final score cannot answer whether a method improved early, spent most of its budget exploring dominated regions, reached a useful quality target and then stalled, or produced a late improvement that was expensive to verify.

COCO already establishes the fixed-target and anytime foundation: evaluate the nondominated archive as it grows, record the first function-evaluation count at which predeclared quality targets are reached, and preserve explicit failures for targets not reached. AR-5B5 should adopt this discipline rather than claim it as novel.

The proposed extension is an evidence-cost layer around that standard:

> Measure the first point at which a run reaches a preregistered Pareto-quality target with independently verifiable evidence, and retain the complete vector of resources spent to get there.

The primary question is not merely "how many evaluations to target?" It is "what quality was attained, with what evidence status, at what cumulative resource vector, and with what uncertainty across independent problem instances?"

This can make the programme materially different from a conventional evolutionary-optimization scoreboard, without misrepresenting COCO's prior work.

## 2. Two separate evaluation modes

Never merge these into one leaderboard.

### A. Mechanism-matched mode

Purpose: isolate the effect of the search/allocation mechanism.

Every method must receive the same:

- frozen candidate offers and offer-set evidence;
- problem, landscape template, objective order, scale anchors, constraints, reference point, and reference front;
- evaluator and verifier build identities;
- seed and paired landscape block;
- worker count, execution visibility rules, budget ceiling, and resource-meter contract.

A mismatch invalidates that paired comparison. This mode answers whether one allocator makes better choices under the same opportunities.

### B. Best-practical-system mode

Purpose: compare complete systems, including proposal generation, model calls, operators, tool use, candidate evaluation, retries, and verification.

Systems may create different proposals and candidate sets. Do not force artificial offer-set equality. Instead require common tasks, held-out tests, resource meters, verifier standard, and total resource ceilings. Every proposal, failure, retry, timeout, discarded candidate, evaluator action, and verifier action is charged.

This mode is where complete systems such as CodeEvolve- or ShinkaEvolve-style pipelines belong. Pin exact model weights/version, prompts, sampling parameters, tools, source revision, task set, and execution image. Published results are context, not matched experimental evidence.

## 3. Frozen target contract

A target plan is committed before any confirmatory run is inspected. It binds:

1. Benchmark suite, problem ID, instance/template ID, dimension, and version.
2. Ordered objective definitions, maximize/minimize direction, scale anchors, and feasibility rules.
3. Hypervolume reference point and archived reference-front digest.
4. Indicator version and target thresholds.
5. Maximum objective-evaluation budget and per-resource ceilings.
6. Candidate artifact, evaluator, and independent-verifier identities.
7. Execution model: sequential, batched, or asynchronous; worker count and accounting semantics.
8. Statistical analysis plan, independent cluster definition, practical-effect threshold, and multiplicity procedure.

The plan digest is included in every checkpoint and final receipt. No post-hoc choice of normalization, reference point, target, metric direction, or resource weighting is permitted.

### Initial target set

For the v0.1 comparison harness, use the existing shared indicators:

- Hypervolume: target is reached when HV is greater than or equal to a preregistered threshold.
- IGD+: target is reached when IGD+ is less than or equal to a preregistered threshold.
- Additive epsilon: target is reached when epsilon is less than or equal to a preregistered threshold.

Each target has a stable ID, indicator ID/version, threshold, comparison operator, and plan digest. Keep objective-evaluation targets separate from evidence-verification targets. A quality target is not an independently verified discovery merely because a deterministic optimizer's own report replays.

All target values and the reference front must be fixed for an entire comparison block. If the reference data changes, create a new benchmark version and rerun or explicitly publish a recalculation using retained archives.

## 4. Anytime archive and event semantics

The run must retain every evaluated candidate, not only the final population and final nondominated front.

For each evaluation retain:

- canonical run ID and evaluation ID;
- monotonically increasing evaluation ordinal;
- candidate ID and decision-vector digest;
- raw objective vector in native objective units;
- feasibility result and complete constraint-violation data;
- evidence artifact digest and verifier receipt identity/digest, when applicable;
- evaluation outcome (success, invalid, failed, timeout, or verifier-rejected);
- action/resource receipt references;
- completion time relative to the run's monotonic clock;
- provenance for the evaluator and task instance.

The archive at evaluation ordinal t is reconstructed from every eligible evaluation with ordinal less than or equal to t. Infeasible and invalid points remain in the event history and cost ledger, but cannot enter the feasible Pareto archive. Repeated or duplicate proposals remain charged. A retry is a new action and cannot overwrite the failed attempt.

For the sequential reference algorithms, evaluation ordinal is the order in which objective evaluations complete. Every objective evaluation increments the count, including failed or invalid evaluations if the declared evaluator contract charged the evaluation budget.

### Parallel and asynchronous runs

Do not pretend that worker-time equals elapsed time. For parallel runs, preserve dispatch sequence, completion sequence, batch/barrier identity, worker identity, and monotonic start/completion timestamps. At a target crossing, include resources already spent by all dispatched work and explicitly account for in-flight work according to the frozen execution contract.

The initial implementation may qualify only sequential runs. Parallel/asynchronous target-attainment qualification remains blocked until the ledger can charge pending and in-flight work without ambiguity. Do not silently apply sequential semantics to Coral's four-seat execution adapter.

## 5. Resource accounting: vector first, scalar only as a sensitivity view

Store integer raw counters with explicit units. At minimum preserve:

- host CPU time;
- summed worker-time;
- elapsed wall-clock time;
- accelerator time;
- peak host memory and peak accelerator memory;
- energy;
- input and output tokens;
- tool calls;
- evaluator calls;
- verifier calls;
- normalized charged budget, including its frozen executable cost-model identity.

Classify each meter before aggregation:

| Meter | Aggregation |
|---|---|
| CPU time | Sum |
| Worker-time | Sum; never label it elapsed latency |
| Accelerator time | Sum |
| Energy | Sum |
| Tokens and calls | Sum |
| Peak memory | Maximum, not sum |
| Elapsed wall time | Run-level monotonic span, not sum of action durations |
| Normalized charge | Sum only under the same frozen cost-model version |

Unknown required meters remain unknown and block strict resource qualification. Never replace an unknown with zero. Keep candidate-evaluation cost and proposal/search overhead separately visible, then reconcile both to total campaign spend. Include failed and timed-out work. A candidate's cost must include all retries associated with its canonical content identity; campaign overhead must not disappear simply because it cannot be attributed to one candidate.

Report the complete resource vector at every target crossing. A scalar weighted cost may be shown only as a declared sensitivity analysis over a frozen set of weights. It must not replace the vector Pareto frontier or the individual meter columns.

## 6. Target-attainment records

For every preregistered target and every run, emit exactly one outcome:

- Reached: first evaluation ordinal, cumulative resource vector, elapsed wall time, checkpoint digest, archive digest, evidence qualification state, and verifier receipt references.
- Not reached by budget: explicit censored outcome with final evaluation ordinal, final resource vector, and final checkpoint digest.
- Invalid run: explicit reason and charged resources; never silently omit it.

Do not report only successful runs. Report fixed-budget quality at each declared budget and fixed-target cost to each target. Retain the full anytime trajectory so indicators can be recalculated against a newer reference front without rerunning the optimizer, while preserving the original benchmark version for historical comparisons.

A target crossing must be based on the archive at that exact prefix, not the final archive projected backward. The target's first-attainment ordinal is the minimum ordinal whose checkpoint satisfies the frozen predicate.

## 7. Verification levels and claim ceilings

A digest string is a reference, not proof that the referenced bytes exist or are correct. Keep these levels distinct:

1. **Recorded:** the event and claimed result are present in the run ledger.
2. **Structurally valid:** schema, sequencing, provenance links, and receipt digests validate.
3. **Replay verified:** the declared deterministic algorithm/evaluator replay reproduces the recorded result.
4. **Artifact verified:** an independent verifier resolves the candidate/evidence bytes and verifies their digest and declared content.
5. **Held-out verified:** an independent evaluator verifies the candidate against the sealed test set or held-out physical/scientific regime.
6. **Confirmatory:** held-out verification and preregistered cluster-level statistical and practical-effect criteria pass.

The primary "cost to first independently verified held-out Pareto improvement" metric requires levels 4 and 5, plus the confirmatory conditions for a superiority claim. A replayable synthetic ZDT1 candidate can test archive mechanics; it cannot count as a scientific discovery.

A verifier must be independent of the proposer and must not be able to silently change evaluator truth, objective definitions, candidate bytes, or the archived reference set. Any verifier corruption control that is not detected at its predeclared sensitivity blocks superiority claims.

## 8. Statistical reporting

Use paired landscape/template blocks where the same independently authored instance is evaluated by all compared systems. Seeds nested within one landscape are repeated trials, not independent landscapes.

For each target report:

- per-run attainment or not-reached status;
- empirical distribution of evaluation count and each resource cost;
- empirical cumulative distribution of target-attainment cost;
- expected running time or other explicitly defined failure-penalized summary;
- paired effect estimates and confidence intervals at the independent landscape-cluster level;
- practical-effect threshold and multiplicity adjustment;
- uncertainty-aware Pareto membership and feasibility probability when repeated measurements support it.

For multi-objective run outcomes, retain empirical attainment information across independent runs. A median front alone conceals dispersion and multimodality. Report attainment surfaces/probabilities for two and three objectives where feasible, plus per-instance distributions and worst-family performance.

A pilot informs variance and power; it does not become the confirmatory test set. Use at least the current 100-independent-landscape confirmatory floor as a starting protocol, then increase sample size when the preregistered power calculation requires it. Never count seeds, generations, or candidates as independent landscape samples.

## 9. Benchmark ladder and competitive field

### Canonical mathematical suites

- ZDT: two-objective sanity, disconnected/convex/nonconvex geometry extensions.
- DTLZ: scalable objective counts and front geometry.
- WFG: non-separability, deception, multimodality, and mixed front shapes.
- Constrained DTLZ: feasibility and constraint handling.
- COCO BBOB-biobj: established black-box multi-objective fixed-target methodology.
- Expensive/noisy and multimodal families: measure surrogate error, evaluation cost, robustness, and diverse decision-space solutions.

Keep the 2–4 objective exact hypervolume path distinct from any many-objective approximation. Any approximation must have a versioned method, independent error validation, and a stated dimensionality regime.

### Algorithm reference set

Retain NSGA-II, NSGA-III, MOEA/D, SMS-EMOA, and RVEA as core references. Add decomposition and indicator variants when their configurations can be pinned. Add MAP-Elites/CVT-MAP-Elites only with independently validated behavior descriptors and archive metrics. Include novelty search/island evolution for discovery settings, not as substitutes for canonical continuous MOO baselines.

PlatEMO v4.16 is a useful external reference release (2026-07-19); its release notes report 360 algorithms and 630 problems, including new large-scale, constrained, expensive-constrained, and multimodal work. The current in-repo three-algorithm, two-problem surface is therefore a narrow protocol scaffold, not broad industry coverage.

### Engineering-search stratum

Add repository-level tasks inspired by FormulaCode and evolutionary coding agents. Freeze repository snapshot, compiler/toolchain, build environment, tests, task instructions, model weights/version, prompts, tools, sampling parameters, and all resource meters. Require independently executed correctness tests, regression tests, and held-out tasks. Separate algorithm-optimizer baselines from complete LLM/evolutionary systems.

CodeEvolve's 2026 Findings of EMNLP paper reports results on the AlphaEvolve benchmark suite and comparisons with other evolutionary coding frameworks. Treat those as claims to reproduce under pinned conditions, not results transferable to Symthaea's scientific-search workload.

## 10. Paradigm shift after the measurement layer: value-of-computation control

Once target-attainment is trustworthy, add a search controller that chooses among:

- propose another candidate;
- replicate a noisy evaluation;
- measure a currently uncertain objective;
- run an independent verifier;
- expand a promising region;
- transfer to a held-out regime;
- stop because the next action is unlikely to change the decision.

The controller should estimate the expected value of information or expected decision improvement per incremental resource vector, subject to hard feasibility and budget limits. Its training/evaluation target is not proposal count or immediate reward. It is reduction in expected cost to an independently verified, practically meaningful held-out Pareto improvement.

This becomes a separate benchmark track. Compare the controller with fixed schedules, uniform allocation, uncertainty sampling, hypervolume-contribution sampling, and random allocation. Use frozen evaluation tasks and report ablations. Do not let the controller read the held-out test set or evaluator-only semantic labels.

## 11. Acceptance tests for the first implementation

The first code tranche should add an anytime archive/target-attainment module and expose trace records from NSGA-II, MOEA/D-DE, and SMS-EMOA. It must include tests for:

1. Reconstructing the nondominated archive after every evaluation prefix.
2. A dominated candidate never reducing archive quality or inflating hypervolume.
3. A feasible new point changing the archive and monotonically improving fixed-reference hypervolume.
4. Correct first crossing for HV, IGD+, and additive epsilon targets.
5. Targets not reached by budget emitted as explicit outcomes.
6. Objective-evaluation counts including failed/invalid work according to the frozen execution contract.
7. Candidate duplicates and retries remaining in cost accounting.
8. Exact resource aggregation: sums for additive counters, maximum for peak memory, run-level elapsed time not summed.
9. Missing required meter blocks strict resource qualification.
10. A tampered evaluation, target plan, reference front, resource receipt, or checkpoint digest invalidating report verification.
11. Report replay being order-deterministic and exact-head/source-build bound.
12. A target crossed before the final budget retaining its original first-crossing ordinal and cumulative cost.
13. Parallel/in-flight inputs being rejected until the batch accounting model is implemented.
14. Cross-algorithm comparison rejecting any mismatch in target plan, objective contract, reference data, budget, or cost-meter version.

Exact-head acceptance should include rustfmt, focused tests, full crate tests, clippy, deterministic replay twice from clean state, and external cross-checks against a pinned PlatEMO/COCO configuration. A workflow marked skipped is not a pass.

## 12. Staged implementation order

1. Add evaluation-history records to each in-repo optimizer, without changing optimization decisions.
2. Add a generic archive-prefix reconstruction and target-attainment receipt module.
3. Replay-verify the history and receipt against the frozen run report.
4. Add sequential resource accumulation and exact not-reached/censoring semantics.
5. Integrate EventTape receipts and independent evidence-byte resolution.
6. Add parallel/batched accounting only after explicit dispatch/completion/in-flight semantics are covered.
7. Run exploratory multi-seed pilots; preregister confirmatory tasks and practical effects.
8. Cross-check external reference implementations before comparative claims.

This specification is not evidence that any optimizer has improved or that the implementation already exists. Its purpose is to freeze a testable contract before the next code tranche.

## References

- COCO, Performance Assessment: https://numbbo.github.io/coco-doc/perf-assessment/
- COCO, Biobjective Performance Assessment: https://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/
- COCO official data archive: https://coco-platform.org/data-archive/index.html
- PlatEMO v4.16 release notes: https://github.com/BIMK/PlatEMO/releases/tag/v4.16
- CodeEvolve, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.1345/
- Empirical attainment functions: https://eden.dei.uc.pt/~cmfonsec/fonseca-emo2011-reprint.pdf


## 13. Implementation status — first shared Rust tranche

The optimizer-agnostic sequential history/receipt foundation has now been committed:

- \`crates/domains/symthaea-architecture-engineering/src/pareto_anytime.rs\`
- exported from the architecture-engineering crate;
- hash-chained header and action events;
- candidate evaluation, failed attempt, verifier promotion/rejection, and auxiliary actions;
- candidate identity binds decision digests and prevents alias IDs for identical decision bytes;
- frozen algorithm/evaluator/verifier builds, execution environment, target plan, cost model, and required meters;
- archive-prefix HV/IGD+/epsilon analysis with evidence-level filters;
- explicit first-crossing and not-reached-by-budget outcomes;
- replay-reconstructed, digest-bound run receipts.

Implementation notes and claim limits: \`AR5B5_ANYTIME_IMPLEMENTATION_V0_1.md\`.

Optimizer instrumentation is deliberately not implied by this module: NSGA-II, MOEA/D-DE, and SMS-EMOA still need to emit the shared trace. That work is tracked in [Issue #9606](https://github.com/Luminous-Dynamics/symthaea/issues/9606).

**Validation boundary:** the code and tests are committed but have not been compiled, formatted, linted, or executed in this environment. The module records verifier-level assertions and receipt digests; it does not yet retrieve/verify candidate artifact bytes or authenticate verifier signatures. Therefore, no independently verified held-out discovery claim is yet supported.
