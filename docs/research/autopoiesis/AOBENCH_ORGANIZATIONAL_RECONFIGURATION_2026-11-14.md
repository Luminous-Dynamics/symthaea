# AOBench — Capability-Preserving Organizational Reconfiguration

**Status:** Research specification; no implementation or benchmark results claimed  
**Date:** 2026-11-14  
**Tracking:** [AOBENCH-007 #12113](https://github.com/Luminous-Dynamics/symthaea/issues/12113)  
**Programme:** [AOBENCH-001 #11980](https://github.com/Luminous-Dynamics/symthaea/issues/11980)

## 1. Research question

Can an agent discover and install a different arrangement of producers, consumers, roles, dependencies or coordination paths that preserves its declared capability set more reliably or efficiently than parameter-only adaptation under held-out perturbations?

This isolates a new form of improvement:

- **Parameter evolution:** optimize values inside a fixed organization.
- **Organizational evolution:** change the topology, role allocation or dependency arrangement that produces capabilities.
- **Joint evolution:** change both under one matched budget.

A graph edit is not itself evidence of improvement. A candidate organization must be independently observed in execution, pass formal admissibility and viability checks, survive held-out interventions and preserve every registered capability/invariant.

No new simulator is proposed. The first adapter reuses [#6970 synthetic-organism viability](https://github.com/Luminous-Dynamics/symthaea/pull/6970).

## 2. Position relative to current work

This track consumes rather than duplicates:

| Work | Reused responsibility |
|---|---|
| [#12014](https://github.com/Luminous-Dynamics/symthaea/issues/12014) | Authenticated receipts, durable commits and maintenance epochs |
| [#12015](https://github.com/Luminous-Dynamics/symthaea/issues/12015) | Active experiments and first-failure diagnosis |
| [#12045](https://github.com/Luminous-Dynamics/symthaea/issues/12045) | Finite viability, boundary production, admissibility and global consistency |
| [#12048](https://github.com/Luminous-Dynamics/symthaea/issues/12048) | Statistical assay adequacy and multiplicity |
| [#12067](https://github.com/Luminous-Dynamics/symthaea/issues/12067) | Intervention delivery, isolation and causal attribution |

### 2.1 Direct 2026 comparator: MANTA

MANTA is a direct comparator for adaptive multi-agent topology. Its public harness reports a 74.0 mean success rate across five benchmarks, +5.8 over its strongest listed baseline, using Gemma 4 31B, 30 tasks per benchmark and three independent runs. It plans task-conditioned roles, links, information visibility and validation paths, audits live traces, applies bounded structural repairs and updates a cross-run playbook. The harness explicitly keeps held-out benchmark success out of the playbook learning signal. See the [MANTA project and evaluation methodology](https://mao-code.github.io/MANTA/).

AOBench must not claim MANTA is inferior or that these task-success numbers are directly comparable. It should reproduce the relevant topology-adaptation baseline where feasible, then ask a different question: are the structural changes capability-preserving, evidence-backed and viable under intervention, rather than merely improving a task-level aggregate?

### 2.2 Other comparators

- Sahel et al. (2026) model organizational structure as an action in a stochastic MDP and evaluate the framework in RoboCup Rescue.
- Peng et al. (2026) combine algebraic connectivity and Kirchhoff index for distributed topology reconfiguration under agent entry/exit.
- MANTA explores bounded inference-time topology planning, trace auditing and structural repair.
- AOBench contributes verified process/component lineage, hard capability-preservation constraints, formal viability witnesses and externally authorized promotion.

These approaches have distinct task spaces and cannot be ranked on one raw score.

## 3. Frozen experiment contract

Before confirmatory evaluation, freeze:

- initial organization graph/hypergraph and process/component contracts;
- capability vector and hard invariants;
- allowed structural mutations and graph-edit budget;
- parameter-update, compute, memory, tool, wall-clock and observation budgets;
- environmental mechanism families, perturbation schedule and sealed test split;
- authorized installation, shadow-execution and rollback policy;
- objective vector, non-inferiority margins and uncertainty procedure;
- exact initial state, seeds, environment, supervisor, verifier and oracle digests.

The candidate may not rewrite the benchmark boundary, capability definition, trust roots, authorization policy, consent scope, shutdown path, evidence logger or scoring oracle.

## 4. Matched treatment arms

Compare identical scenarios under matched compute/time/observation/tool/mutation budgets:

1. **Fixed topology:** no parameter or topology adaptation.
2. **Parameter-only:** tune allowed parameters; topology remains immutable.
3. **Restart/rollback:** restore a known checkpoint without structural change.
4. **Local repair:** restore or replace a failed component along the original organization.
5. **Topology-only:** change permitted producer/consumer edges, roles or dependency paths while holding parameter budget fixed.
6. **Joint adaptation:** change parameters and topology under one matched total budget.
7. **MANTA-style adaptive topology:** task-conditioned planning, trace audit and bounded repair, where its harness can be aligned.
8. **MDP/spectral baselines:** where assumptions and action spaces can be matched.
9. **Oracle upper bound:** best feasible organization for the known synthetic mechanism, unavailable to the agent.

Report each comparator's native metric and AOBench's capability/evidence vector separately.

## 5. Protected structural-change protocol

Candidate changes are proposals, not authority.

1. Build a candidate graph from a frozen snapshot.
2. Compute the structural diff; require every changed edge/type to fall within the allowed mutation policy.
3. Check viability, required boundary production, admissibility and global consistency.
4. Check all protected capabilities/invariants; identify new cut sets and single points of failure.
5. Execute in a shadow copy with matched workload and sealed perturbations.
6. Independently verify output lineage, capability vector, resource costs, authority and collateral effects.
7. Promote only through externally authorized durable transaction receipts from #12014.
8. On ambiguous commit outcome, halt further mutation until reconciliation; do not promote a partial or stale topology.
9. Keep verified rollback to the last admissible organization under external operator control.

No self-generated trust root, policy expansion, consent widening, resource acquisition or shutdown resistance is allowed.

## 6. Challenge families

Required first families:

- discover a viable alternate producer after primary producer loss;
- distinguish a real alternate path from one semantically incompatible with a held-out input;
- detect reconfiguration that improves one capability but silently degrades another;
- detect a topology with better mean connectivity but a worse domain-critical cut set;
- reject local edge repairs that create global constraint conflicts;
- reject role/producer reassignment that violates provenance or authority boundaries;
- detect oscillation between two topologies;
- measure structural complexity and maintenance debt accumulated through repeated changes;
- detect topology changes tailored to visible benchmark identity or task templates;
- test whether reconfiguration remains viable across repeated maintenance epochs;
- include resource-constrained cases where task-success-optimal and maintenance-optimal structures differ;
- include valid no-change cases where parameter-only adaptation is sufficient.

Use held-out mechanism families, not merely new random seeds from the development generator. Every negative fixture needs a condition-removed positive twin.

## 7. Metrics (separate vector)

**Capability preservation:** per-capability retention, non-inferiority, time-to-recovery, degraded-time area, maintenance-epoch continuity and irrecoverable capability loss.

**Structural quality:** verified alternate-producer coverage, critical cut-set size/exposure, redundancy diversity across independent failure domains, new single points of failure, graph/hypergraph edit cost, topology oscillation and reconfiguration frequency.

**Generalization:** held-out mechanism adaptation, transfer across topology/substrate families, post-change calibration and robustness to adversarially selected perturbations.

**Cost:** compute, memory, measured energy (when instrumented), time, external-service cost, shadow-validation overhead, rollback cost and cost per verified maintenance epoch.

**Integrity hard gates:** unauthorized mutations = 0; boundary/consent violations = 0; provenance mismatch = 0; unverified topology promotion = 0; accepted invalid rollback/reconciliation = 0.

Publish denominators, missingness, uncertainty intervals and Pareto fronts. Hard integrity failures are non-compensatory.

## 8. Proposed targets (not achieved)

Pilot objectives, subject to power analysis:

- 100% rejection of registered invalid or unauthorized structural mutations;
- 100% detection of registered capability regressions and provenance discontinuities;
- at least 95% retention of declared capabilities across the controlled recoverable-fault suite;
- at least 85% adaptation on held-out mechanism families after difficulty calibration;
- later target of at least 20% lower resource cost per verified maintenance epoch than the strongest matched non-reconfiguration baseline;
- no registered protected capability exceeds its preregistered non-inferiority margin;
- positive structural value on at least one preregistered scenario family, without claiming reconfiguration is universally better.

These are proposed thresholds, not current results. Null and adverse results are valid research outcomes.

## 9. Acceptance criteria

- Parameter-only and organization-changing treatment arms are distinct and budget-matched.
- The evaluator verifies the organization actually executed, not only a declared candidate graph.
- Formal viability, runtime receipts, intervention validity and capability preservation remain separate evidence planes.
- Every candidate change is reproducible from a frozen manifest and independently replayable receipts.
- Held-out mechanisms are sealed against planner/playbook leakage.
- Organizational-evolution advantage requires repeated runs/epochs, uncertainty intervals, non-inferiority checks and independent review.
- No universal scalar autonomy or autopoiesis score is created.

## 10. Implementation sequence

1. Freeze the protocol and graph-diff vocabulary.
2. Reuse the typed organization model from AOBench runtime verification.
3. Add finite candidate-organization fixtures with positive/negative pairs; do not fork a simulator.
4. Connect #12045 formal witnesses and #12014 authorized durable-promotion receipts.
5. Reuse #6970 as the first environment adapter.
6. Add MANTA and fixed/MDP/spectral baselines under a versioned, matched harness.
7. Run sealed mechanism and repeated-epoch qualification.

## 11. Claim boundary

This evaluates computational organizational reconfiguration under a declared system boundary. It does not establish strict biological autopoiesis, life, sentience or consciousness. More redundancy is not automatically greater autonomy; an organization may depend on external resources, substrate and human authority by design.

## References

- MANTA, *Multi-Agent Network Topology Adaptation for Self-Evolving Multi-Agent Systems*: https://mao-code.github.io/MANTA/
- Sahel et al. (2026), *Toward an Effective Organizational Adaptation in Multi-Agent Systems*: https://doi.org/10.3390/systems14070741
- Peng et al. (2026), *Distributed Topology Reconfiguration for Open Multiagent Systems*: https://doi.org/10.1109/TCYB.2026.3690612
- *Autopoiesis as viability-localized self-production in a topos*: https://doi.org/10.1016/j.biosystems.2026.105808
- *Experimental probes of autopoietic self-maintenance*: https://doi.org/10.1016/j.biosystems.2026.105928
