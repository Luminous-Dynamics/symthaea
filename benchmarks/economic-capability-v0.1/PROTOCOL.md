# Economic Capability Benchmark v0.1

**Status:** protocol proposal; no benchmark runs or capability results are claimed  
**Snapshot:** 2026-11-28  
**Coordination:** [Symthaea #14200 — ECON-CAPABILITY-001](https://github.com/Luminous-Dynamics/symthaea/issues/14200)

## 1. Purpose

Measure whether an AI/automation workflow delivers more independently accepted useful work than a matched human or conventional workflow, under the same declared task, information, quality, safety, time and resource constraints.

The benchmark is about **task-level capability and economic outcomes**, not a model leaderboard and not a prediction that a whole occupation will disappear.

The primary question is:

> What amount of accepted work can this exact workflow deliver, at what fully burdened cost and resource intensity, with what human involvement, reliability, operating limits and evidence quality?

V0.1 is a cross-domain measurement and benchmark-composition layer. It does not create a new universal task ontology, occupational database, scheduler, robot ledger, evidence ledger, accounting system, safety authority or workforce policy.

## 2. The central paradigm shift

Do not infer:

`occupation -> automation percentage -> jobs replaced`

Measure:

`occupation crosswalk -> task bundle -> accepted work units -> matched workflow -> full resource cost -> human and service outcomes`

A job is a bundle of tasks, qualifications, decisions, relationships and responsibilities. Task automation exposure is not the same thing as role redesign, labour demand, employment change, wage effects or worker transition outcomes.

A plausible answer, a successful demonstration, a model-reported completion flag or a simulator metric does not establish accepted work.

## 3. Ownership and reuse

The benchmark composes existing owners. It must not make a new canonical source for their facts.

| Concern | Canonical work to consume |
|---|---|
| Task integrity, benchmark custody and engineering-agent evaluation | Symthaea CEB [#10115](https://github.com/Luminous-Dynamics/symthaea/pull/10115), [#10419](https://github.com/Luminous-Dynamics/symthaea/issues/10419) |
| Manufacturing output and economics | Symthaea [MFG-VALUE #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608) |
| Accepted productive work per robot-hour | Symthaea [ROB-OPERATIONAL-UTILITY #11607](https://github.com/Luminous-Dynamics/symthaea/issues/11607) |
| Agricultural service economics | Symtropy [AGRI-ROBOT-COST #1814](https://github.com/Luminous-Dynamics/symtropy/issues/1814) and Symthaea [AGRI platform kernel #7817](https://github.com/Luminous-Dynamics/symthaea/pull/7817) |
| Robot application maturity and task scenarios | Symtropy [ROBOTICS-DEMO-QUALIFICATION #5702](https://github.com/Luminous-Dynamics/symtropy/issues/5702) |
| Manufacturing and workforce scheduling | Symthaea [FACTORY-OPS #7677](https://github.com/Luminous-Dynamics/symthaea/issues/7677) |
| Matched cross-vendor engineering/operations comparisons | Symthaea [INDUSTRY-STACK-COMP #11774](https://github.com/Luminous-Dynamics/symthaea/issues/11774) |
| Human participation, care, training and automation shocks | Symthaea [SYM-FIN-SYS-013 #7042](https://github.com/Luminous-Dynamics/symthaea/issues/7042) |
| Resource-to-service and productive-capability dependencies | Symthaea [CIV-VALUE #5946](https://github.com/Luminous-Dynamics/symthaea/issues/5946), [CIV-BOOT #5774](https://github.com/Luminous-Dynamics/symthaea/issues/5774) |
| Financial research and investment workflows | Symthaea [SYM-FIN-EVAL #5399](https://github.com/Luminous-Dynamics/symthaea/issues/5399), [Financial Research Agent v1 #11233](https://github.com/Luminous-Dynamics/symthaea/pull/11233), [Quant Strategy v1 #11234](https://github.com/Luminous-Dynamics/symthaea/pull/11234) |

Mycelix remains authoritative for identity, consent, inventory, work orders, transactions, institutional records and provenance where applicable. Symtropy remains authoritative for its physical simulation/replay outputs and model limits. Existing domain owners retain acceptance, safety, measurement, scheduling and accounting semantics.

The cross-domain layer may join typed outputs and compute comparative projections. It may not upgrade one owner's output into another owner's evidence or authority.

## 4. Occupational mapping is a crosswalk, not the task ontology

Use version-pinned, rights-reviewed occupational/task classifications where they help locate coverage:

- South Africa: Department of Higher Education and Training Organising Framework for Occupations (OFO).
- International: European Commission ESCO.
- United States: O*NET.

Each mapping must preserve the exact source release, occupation/task identifier, language, region, source rights, mapping method, confidence and unmatched/ambiguous status.

Do not assume the task lists or wage structures are identical across jurisdictions. Do not infer a task from an occupation title when the source has no task-level evidence. Do not ingest private worker-surveillance data or infer individual worker performance.

## 5. Comparison arms

For each eligible task/work package, compare only arms that can be made meaningfully equivalent:

1. Qualified human-only workflow.
2. Conventional deterministic automation or the incumbent professional tool.
3. Symthaea contribution alone (for example retrieval, reasoning, proposal or verification).
4. Symtropy contribution alone (for example simulation, planning or replay).
5. Robotics/physical automation alone.
6. Composed Symthaea + Symtropy + robotics + professional-tool workflow.
7. No change, outsourcing, shared-service or no-automation alternative where legitimate.

Freeze the task definition, work mix, accepted-output oracle, information available at decision time, operating domain, resource budgets, staffing assumptions, deadlines and quality/safety constraints.

Allow native tools where they are required for a fair comparison, but record all transformations and information asymmetries. Mark a comparison `NotComparable` rather than silently removing a task or weakening its acceptance criterion.

Separate system-only performance from human-assisted performance. Count prompt engineering, workflow integration, supervision, corrections, exception handling, training, setup and maintenance in the appropriate resource ledger.

## 6. Task acceptance contract

V0.1 does not define another general task schema. It references the exact task/case ID and version from the existing benchmark owner, plus an economic measurement profile.

Every scored task must identify:

- exact task/work-package and version;
- the requested output and unit of work;
- an independent, deterministic or independently adjudicated acceptance oracle;
- quality, tolerance, timeliness and safety requirements;
- allowed inputs/tools, information frontier and prohibited effects;
- operational domain, site/asset/tool/robot/hardware profile where relevant;
- human qualifications and approval requirements;
- resource budget, time denominator and cost boundary;
- exact system/model/adapter/software/hardware versions;
- source evidence, calibration, output lineage and evaluator identities;
- unknown, unsupported, aborted, failed and indeterminate dispositions.

A task-completion flag alone does not count as acceptance. Physical output requires the relevant independent inspection/measurement. Engineering analysis requires the applicable solver-result and evidence checks. Digital work requires postconditions in the target application or service state.

## 7. Metric vector

Do not collapse the result into one weighted `automation_score`, `robot_score` or `economic_value_score`.

### 7.1 Work quality and delivery

- accepted output / requested output;
- first-pass acceptance;
- defect, rework, rejection and scrap rates;
- subgoal completion and long-horizon task success;
- p50/p90/p95 completion time and deadline adherence;
- failed, aborted, held, escalated, timed-out and indeterminate counts;
- robustness by task family, shift, site, operator, work mix and distribution shift.

### 7.2 Human work and agency

- direct operator/worker time;
- supervision, review, exception handling, correction and rescue time;
- setup, prompting, integration, calibration, training and coordination time;
- human judgement and authorization still required;
- work augmented, displaced, newly created or not assessed;
- training/skill formation and worker transition outcomes only where measured using appropriate consent and methods.

Reduced paid hours do not imply reduced useful activity. Care, maintenance, learning, community and unpaid work must not disappear merely because conventional payroll data under-observes them.

### 7.3 Full economics and resources

- fully burdened cost per accepted work unit;
- total human hours across operators, reviewers, supervisors, maintainers and integrators;
- capital, depreciation, financing and utilization;
- software, inference/compute, connectivity and data acquisition;
- energy, materials, consumables, tooling, floor-space and logistics;
- maintenance, calibration, repair, replacement, downtime and recovery;
- rework, scrap, damage, liability and failure costs where measured;
- currency, price basis, tax, FX source/date, cost-boundary ID and effective date.

Missing fields remain unknown or explicitly bounded; they are never silently set to zero. Do not transfer foreign wage or operating-cost estimates to South Africa without an evidenced local profile.

### 7.4 Safety, authority and trust

- critical safety and quality violations;
- unauthorized or out-of-domain actions;
- false acceptance of unsupported work;
- privacy/security/rights violations;
- human override, safe hold and recovery;
- traceability of every output counted toward the primary endpoint.

Safety, legal authority, quality, rights and privacy are non-compensable gates. High throughput cannot offset them.

## 8. Task-level capability disposition

Keep capability separate from economic value. Suggested statuses:

| Status | Meaning |
|---|---|
| `HumanOnly` | No qualified automation path in the declared profile |
| `Assistable` | Useful support; human performs the substantive task |
| `SupervisedExecution` | Bounded execution under human direction/approval |
| `BoundedAutonomy` | Execution without continuous human control inside a declared, monitored envelope |
| `QualifiedAutonomy` | Repeatable, independently accepted performance for the exact task/tool/embodiment/domain/version |
| `NotAssessed` | Evidence insufficient, unsupported domain or invalid comparison |

These dispositions are task-, profile- and version-specific. They do not rank workers or occupations.

Any occupation-level task-hour projection must expose its underlying task mappings, source weights, unassessed share and uncertainty. It is an exposure projection—not a net employment forecast. Employment, wages, demand, role redesign and worker transitions require separate models and external evidence.

## 9. First task families

Start with two digital and one bounded physical cohort. Add others only after the measurement contract is stable.

### A. Digital IT operations

`incident intake -> evidence retrieval -> diagnosis -> proposed remediation -> authorized change -> independent service-restoration check -> incident record`

Use a human operations baseline, existing deterministic runbooks/automation and a Symthaea-assisted workflow. The acceptance oracle is the verified service postcondition, not a plausible diagnosis or a closed ticket.

### B. Professional knowledge work / engineering

`source discovery -> requirements/assumptions -> calculation/model -> professional artifact -> independent review`

Start with evidence-bound research and a narrow engineering task that can use existing CEB and engineering adapters. Do not compare free-form writing scores to a verified engineering artifact.

### C. Bounded physical workcell

Select one low-risk, repeatable manipulation/assembly/inspection task with independent output inspection. Compare human, conventional automation, baseline robot policy and the relevant Luminous contribution. Reuse ROB-OPERATIONAL-UTILITY and Symtropy's scenario/recovery owners.

Agriculture and construction are follow-on applications, not assumed to be already field-qualified.

## 10. External benchmark crosswalk — do not merge scores

External benchmarks are versioned comparators with different task sets and scoring semantics.

| Benchmark | Appropriate use | Important limit |
|---|---|---|
| GDPval | Professional deliverable quality across occupations/industries | Does not by itself measure physical work, full deployment economics or net employment |
| OSWorld 2.0 | End-to-end computer-use workflows in real desktop/web applications | A specific release, task set, step budget and grading method are required |
| LADDER | Long-horizon manipulation, subgoal composition and physical execution | Embodiment/suite/simulation-vs-real differences matter |
| Finance Agent Benchmark | Financial analyst research/tool-use tasks | Edition scores are not interchangeable; not a live-trading or portfolio-return guarantee |
| CEB | Engineering/repository comprehension, review, security and correctness | A design-only task catalogue or schema pass is not agent performance |
| MFG-VALUE / ROB-OPERATIONAL-UTILITY | Production and accepted robot work | Domain-specific outcome metrics; not a universal occupational score |

The dated external snapshot is in [EXTERNAL_BENCHMARK_SNAPSHOT_2026-11-28.md](EXTERNAL_BENCHMARK_SNAPSHOT_2026-11-28.md). All external figures must be rechecked and version-pinned before any public report.

## 11. Qualification ladder

- **E0 — Contract integrity:** schemas, task identity, metric boundaries, comparator and oracle are structurally coherent.
- **E1 — Synthetic:** known-positive, negative, ambiguous and no-automation cases produce exact expected dispositions.
- **E2 — Reference:** independent calculations and acceptance oracles match known-answer references.
- **E3 — Blind matched comparison:** human, conventional and Luminous arms receive equivalent information and resource budgets; graders are blinded where practical.
- **E4 — Replay/shadow:** rights-cleared historical or real workflow data; no external effects by implication.
- **E5 — Prospective pilot:** pre-registered matched/crossover or otherwise justified design, independent acceptance and cost measurement.
- **E6 — Transfer:** replication across sites, shifts, operators, task mix or jurisdictions, with external validity analyzed.

A lower level never implies a higher one. Queued, skipped, unexecuted, synthetic, design-only or schema-only work is not a pass at a higher evidence level.

## 12. Initial pilot targets (prospective, not achieved)

For a narrowly scoped pilot, preregister one primary business endpoint and preserve all guardrails:

- 100% expected disposition on the frozen synthetic positive/negative/indeterminate panel.
- 100% lineage for output credited to the confirmatory endpoint.
- A prospective target of at least 10% improvement in the selected primary measure (for example, cost per accepted unit or accepted output per available work-hour) against a matched baseline, with uncertainty supporting improvement.
- No critical safety or quality regression; critical false acceptance ceiling is zero.
- A secondary target of at least 20% reduction in human intervention/recovery minutes where meaningful—not a mandate to remove professional oversight.
- Full reporting of labour, compute, energy, capital, maintenance, materials and failed-work costs.
- Preserve failures, timeouts, abstentions, holds, outages, non-comparable and missing-data cases in the disposition/cost report.

These are project pilot goals, not industry standards or achieved results. If the baseline is already near a ceiling, use a pre-registered non-inferiority or efficiency endpoint instead.

## 13. Validity and anti-gaming controls

- Freeze task, oracle, information frontier, resource budget, comparator versions, exclusions and analysis before scoring.
- Split by site, task family, asset, source family and time where observations are clustered; never randomly split correlated work orders/episodes into train and holdout.
- Count setup, prompting, engineering, integration, maintenance, training, review and failed attempts.
- A simulation result is not an independent physical observation.
- Uninspected, repeated, replayed, partially completed or rejected output is not accepted work.
- No weighted score may offset a safety, authority, quality, rights or privacy failure.
- Preserve timeouts, failures, abstentions, outages and indeterminate states.
- Qualify evaluator reliability and false-accept/false-reject behavior; LLM judges are not the sole authority.
- Publish the exact model, tool, hardware, runner, policy and adapter identities.
- Report sample size, cluster structure, uncertainty, cost and external-validity limits.

## 14. Next implementation slices

1. Freeze task-owner and metric ownership map; no new task ontology.
2. Add a thin economic-profile adapter over existing CEB and domain task contracts.
3. Select one digital IT-operations case and one bounded workcell case.
4. Freeze positive, negative, ambiguous and no-automation synthetic fixtures with independent deterministic oracles.
5. Define the full cost boundary and a local-currency cost profile.
6. Compose specialist receipts into a read-only comparative report.
7. Preregister a sealed matched cohort only after fixture and evaluator integrity pass.

## 15. Claim ceiling

A result applies only to the declared task, version, information, resource boundary, operating domain, acceptance method and cohort. It does not establish general occupational replacement, net employment effects, universal economic superiority, legal permission to practice a profession, safety certification or public-service sufficiency.
