# Evidence-Qualified Capability: Research Roadmap v0.2

**Status:** Proposed research and benchmark targets. None of the targets below are achieved results, certifications, or claims of superiority.

## 1. Thesis: qualify the whole action, not just the answer

The unit of evaluation is the complete deployed system:

`model + agent/harness + tools + policy + data + runtime + verifier + operator workflow`

Conventional task success remains necessary, but it is not sufficient when a system can mutate records, issue credentials, alter infrastructure, or make engineering decisions. EQC therefore reports task capability separately from authority, input integrity, provenance, independent replay, failure containment, and resource cost. These dimensions are non-compensatory: a high task score cannot compensate for an unauthorized side effect or unverifiable result.

This complements rather than replaces domain benchmarks. It should be possible to wrap an existing SWE-bench, OSWorld, robotics, healthcare-workflow, or simulation task with an EQC episode contract and add qualification evidence without changing the original task score.

## 2. Paradigm shifts worth pursuing

### 2.1 From answer scoring to state-transition assurance

Score the before/after state and the permitted mutation set, not only the textual response or action trace. Every action has an explicit precondition, authority snapshot, idempotency key, expected state transition, and postcondition. Denials must prove zero unauthorized mutation.

### 2.2 From confidence scores to calibrated abstention

An agent must distinguish allow, deny, abstain, and unresolved. Evaluate selective risk: plot error rate against coverage as the abstention threshold changes. Reward a system for refusing when evidence is insufficient, but separately measure whether it can complete valid positive tasks. A system that always denies is safe only in a vacuous sense and must fail positive controls.

### 2.3 From one verifier to verifier diversity

Use at least two independently implemented checks for high-consequence tasks: e.g. a reference contract evaluator plus a state-diff oracle, or independent numerical methods for physics. Measure correlated failure, not merely agreement. A verifier that shares the same parser, code path, or assumptions as the system under test is not independent.

### 2.4 From static tests to mutation-tested benchmark validity

Benchmark tasks and graders need their own quality gates. Mutate the agent, inputs, policy, authority, timestamps, action order, environment and verifier. A task is valid only if known-bad variants fail and known-good variants pass. Preserve the mutation catalogue and per-mutant detection rate.

### 2.5 From reproducibility as a boolean to replay envelopes

Exact replay is appropriate for deterministic state transitions. Numerical and GPU simulation may require declared tolerances, uncertainty intervals, and distributional equivalence. Record the hardware, compiler, runtime, seeds, solver settings, tolerances, and all input digests. Never silently convert a tolerance miss into a pass.

### 2.6 From benchmark scores to capability-cost frontiers

Report qualified success against wall time, compute, memory, energy, tool calls, human intervention, and monetary cost. The primary operational measure is cost per independently verified qualified success, with latency and risk constraints shown separately. Do not hide trade-offs in one weighted score.

## 3. Benchmark portfolio and external comparison

| Area | External reference points | What they measure | EQC contribution |
|---|---|---|---|
| Repository engineering | SWE-bench Pro / Terminal-Bench | Real issue resolution and end-to-end terminal tasks | Pristine replay, policy compliance, dependency provenance, mutation-tested graders |
| Computer use | OSWorld and OSWorld 2.0 | Desktop workflows; OSWorld 2.0 emphasizes long-horizon professional workflows | State-transition invariants, mid-task state changes, correct escalation, no unintended side effects |
| Tool-use agents | WebArena, tau-bench, BFCL | Browser workflows, policy-bound API use, function-call correctness | Authority freshness, duplicate/replay safety, exact mutation accounting, independent state oracle |
| Embodied AI | ManiSkill3, Isaac Lab, MuJoCo-based tasks | Contact-rich manipulation, throughput, task success and sim-to-real workflows | Model credibility, sensor/actuator uncertainty, collision invariants, physical evidence and bounded operating envelopes |
| Simulation VVUQ | ASME V&V practices and NIST's 2026 UQ/metrology harmonization work | Verification, validation, uncertainty, risk-scaled model credibility | Machine-readable evidence passport, independent comparator diversity, uncertainty-aware qualification |
| Supply-chain integrity | SLSA provenance and in-toto attestations | Where/how artifacts were built and traceability | Bind source, environment, dependencies, policy, task, verifier and result into one receipt |

Never compare raw scores across different task sets, graders, step limits, hardware, model scaffolds or versions. Publish the exact benchmark version, date, task count, run count, system configuration, cost and confidence interval. Report official leaderboard values only from their maintainers and keep dated snapshots.

## 4. Proposed EQC scorecard and target gates

These are proposed engineering targets for a future held-out evaluation, not claims about current Symthaea or Mycelix performance.

| Metric | Initial qualification target | Stretch target |
|---|---:|---:|
| Unauthorized state mutations in adversarial episodes | 0 / 1,000 | 0 / 100,000 |
| Correct denial on invalid authority / wrong-subject cases | 100% on frozen suite | 100% on held-out mutation suite |
| Positive-task completion under valid authority | >= 90% | >= 97% |
| Provenance completeness on qualified episodes | 100% | 100% |
| Independent replay agreement | 100% deterministic tasks; declared tolerance for numerical tasks | Same, across a second clean runner |
| Known-bad mutant detection | >= 95% of curated mutants | >= 99%, with no critical mutant surviving |
| Unresolved outcomes silently treated as success | 0 | 0 |
| Reproducibility | Same source, data, environment and verifier digests | Independent organization reproduces a sample |
| Cost reporting | p50/p95 latency, CPU, memory, tool calls, operator time | Add energy and marginal infrastructure cost |

A 0-failure result is not proof of zero risk. Always publish the denominator and an uncertainty bound. For example, even 0 failures in 1,000 independent trials leaves a nontrivial upper confidence bound; increase the sample size and diversify the threat model before making stronger claims.

## 5. Symtropy physics qualification track

Build a benchmark suite around physical invariants and independent references, not a single aggregate simulator score.

1. **Collision correctness:** analytic primitive pairs, transformed OBB/SAT, GJK/EPA, degeneracy, near-touching and high-aspect-ratio cases. Include independent SAT or analytic oracles.
2. **Metamorphic invariance:** common rigid transforms, coordinate-frame changes, object-order permutations, and equivalent parameterizations must preserve physical classification within declared numerical tolerance.
3. **Adversarial geometry:** nearly parallel axes, tiny separations, extreme scales, large coordinates, ill-conditioned transforms, zero/near-zero extents and contact manifold transitions.
4. **Dynamics VVUQ:** conservation laws, constraint drift, energy/momentum residuals, convergence under timestep refinement, sensitivity to solver tolerances, and comparison to independent engines.
5. **Uncertainty propagation:** parameter distributions, sensor noise, contact/friction uncertainty and solver uncertainty must produce calibrated output intervals and bounded decision error.
6. **Performance:** throughput and p50/p95 latency across object counts, geometry complexity, contact density and hardware. Compare only with identical scenes, timestep, solver settings, precision, hardware, and correctness gates.

Initial acceptance: zero mismatches on the analytic/frozen deterministic corpus; all metamorphic invariants pass; every numerical comparator reports residuals and tolerances; no performance result is accepted unless correctness gates pass first. Expand to held-out and independently generated geometry before making SOTA claims.

## 6. Symthaea and agent qualification track

- Preserve task success and safe-abstention as separate curves.
- Add authority mutation between plan and execution, not only before task start.
- Add stale/missing/contradictory evidence and mid-run policy changes.
- Include tool-output injection, poisoned repository artifacts, duplicate delivery, timeout-after-commit and retry-after-unknown-outcome.
- Test long-horizon workflows with changing state and delayed observations.
- Measure task success, qualified success, false allow, false deny, abstention precision, recovery time, duplicate side effects, provenance completeness and cost.
- Keep public development fixtures separate from held-out evaluation fixtures. Rotate held-out tasks and publish task-set digests before the run.

## 7. Mycelix / Prism / Health integration track

- Treat identity, authority, consent, provenance and revocation as first-class inputs to the episode contract.
- Bind the exact Holochain DNA/zome, conductor, schema, policy, source and environment versions.
- Test concurrent revocation, stale authority snapshots, duplicate signals, partial network partitions, replayed challenges and unknown transaction outcomes.
- Prove that retries are idempotent and that unknown outcomes reconcile before another mutation.
- For health workflows, use synthetic data until a separate privacy, ethics, security and clinical governance process authorizes any real-data evaluation. No synthetic benchmark implies clinical validity or regulatory certification.
- For Prism ingestion, bind source attribution, dataset license, source commit, content digest and transformation version to every data product.

## 8. Experimental design and publication rules

- Pre-register hypotheses, task splits, exclusions, baselines, hardware and analysis.
- Run at least five independent seeds for stochastic systems; report mean, standard deviation and confidence intervals. Use paired comparisons where tasks are shared.
- Keep a public reproducibility set and a sealed/rotating challenge set.
- Use at least one independent implementation for critical reference results.
- Report all failed, timed-out, skipped and unresolved cases. A skipped workflow is not a pass.
- Use exact commit SHAs and immutable artifact digests in every result.
- Publish raw per-task receipts and aggregate summaries, subject to privacy and security constraints.
- Do not claim "better than SOTA" until the same versioned benchmark, task split, hardware, budgets and verifier have been used, and an independent party can reproduce the result.

## 9. Immediate implementation sequence

1. Complete the Rust reference evaluator's adversarial unit-test matrix and exact-head CI.
2. Add a receipt validator that enforces the JSON Schema plus cross-field invariants; schema validity alone is not semantic validity.
3. Build a separate state-transition oracle for a small local synthetic environment. Do not call the reference evaluator a system-under-test result.
4. Add mutation testing for authority, input digests, timestamps, replay state, action order and verifier outputs.
5. Extend Symtropy with metamorphic and adversarial geometry families, then independent dynamics comparators.
6. Add a common artifact/receipt format with SLSA/in-toto-compatible build provenance, keeping domain-specific evidence separate.
7. Publish baseline runs and cost curves before optimization; optimize the measured bottleneck, not the most impressive-looking metric.

## 10. Definition of a meaningful win

A meaningful win is not a higher number on a narrow public leaderboard. It is a reproducible improvement in qualified capability: more valid tasks completed, fewer unsafe or unauthorized actions, better-calibrated abstention, stronger independent replay, lower cost, and no hidden weakening of task validity or controls.
