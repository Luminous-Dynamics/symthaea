# AOBench — Operational Identity and Behavioral Continuity

**Status:** Research specification; no continuity result claimed  
**Date:** 2026-11-14  
**Tracking:** [AOBENCH-008 #12121](https://github.com/Luminous-Dynamics/symthaea/issues/12121)  
**Depends on:** #12014, #12015, #12045, #12048, #12067, #12118

## 1. Purpose

AOBench needs to distinguish two claims that are often collapsed:

- an organization has changed internally;
- the organization has, or has not, preserved its declared functional capabilities and governed behavior.

An unchanged topology is not proof of continuity. A changed topology is not proof of discontinuity. Equal final outputs do not prove equal behavior, authority or provenance.

This track defines **operational continuity** as a bounded, evidence-qualified relation between a frozen baseline organization and a candidate organization under a declared system boundary, observation alphabet, capability contract, invariant set and experiment family.

It is not a universal metaphysical identity criterion.

## 2. Why this is a separate contribution

The verification-oriented survey *Component-based Distributed Software Reconfiguration: A Verification-oriented Survey* (ACM Computing Surveys, DOI 10.1145/3595376) distinguishes verification of configurations, preparation for safe reconfiguration, verification of reconfiguration programs and inference of valid target configurations. It describes simulation relations for component substitutability: replacing one component by another while preserving behavior. It also highlights that reasoning about full behavior during reconfiguration remains difficult and often relies on abstractions.

MANTA (arXiv:2607.28527, July 2026) demonstrates inference-time changes to agent roles, links, ordering, information visibility and validation pathways. Its reported 74.0 mean across five task benchmarks is task-success evidence, not a proof that the changed organization preserves the full behavioral, governance and evidence contract.

AOBench's distinct contribution is to connect formal behavior relations to:
- authenticated runtime receipts and maintenance epochs (#12014);
- intervention delivery/isolation and causal interpretation (#12067);
- finite viability/admissibility witnesses (#12045);
- statistically adequate assay panels (#12048);
- structurally admitted reconfiguration proposals (#12118).

## 3. Bounded transition model

The first executable model is finite:

`M = (S, A, T, O, C, I, X)`

- `S`: bounded operational states or state abstractions;
- `A`: permitted actions/interventions;
- `T`: labelled transition relation;
- `O`: externally observable state/event labels;
- `C`: registered capability predicates;
- `I`: hard invariants, including safety, boundary, authority, consent and provenance;
- `X`: declared external substrate/resource observations.

A model manifest must bind the initial states, state profiles, transition labels, observation alphabet, capability contract, invariant set, system boundary, model digest and verifier version. Candidate-controlled labels or ground truth are not authoritative.

The model is finite and bounded. An unmodelled hidden state or mechanism is outside the proof.

## 4. Relation types

### 4.1 Simulation / refinement

A relation `R` from baseline states to candidate states is a simulation witness when each related pair satisfies the registered capability and invariant conditions, and every baseline transition can be matched by the candidate with the same registered action/observation label while preserving the relation between successor states.

This supports a one-way claim: required baseline behavior remains available. The candidate may have additional behavior, which must still satisfy the invariant contract.

### 4.2 Bisimulation-style equivalence

A bisimulation-style witness additionally requires the reverse matching condition: every candidate transition must be matched by the baseline. This is stronger and tests two-way behavioral equivalence under the frozen action/observation alphabet.

The implementation must state which direction it checks. It must not label one-way refinement as equivalence.

### 4.3 Stochastic approximate relation

For finite categorical outcomes, compare transition/outcome distributions under matched actions and interventions. Define a practical tolerance `epsilon` before confirmatory runs; initially use total-variation distance.

An approximate relation is admissible only with:
- explicit independent/clustered trial units;
- independently qualified intervention delivery and isolation;
- an assay family adequate for the frozen candidate class;
- pre-registered sampling, missingness, stopping and multiplicity;
- uncertainty bounds on the distance.

A non-significant difference is not proof of equivalence. Underpowered, missing or indistinguishable-but-non-equivalent model pairs remain unresolved.

## 5. Four separate continuity planes

| Plane | Question | Required evidence |
|---|---|---|
| Capability | Are all registered capabilities preserved within frozen margins? | Capability vector, held-out tasks and non-inferiority checks |
| Behavioral | Are transition/trace behaviors related under the declared simulation semantics? | Finite relation witness or stochastic approximate relation |
| Governance | Are trust roots, consent, revocation, boundary and shutdown constraints still valid? | External authority decisions and policy/consent receipts |
| Evidence | Can the transition, lineage and maintenance history be independently reconstructed? | Authenticated receipts, event order, checkpoints and epochs |

These planes are non-compensatory. A candidate may preserve the user-visible capability while losing governance or evidence continuity; that is not full operational continuity.

Substrate changes are reported separately. A new external dependency can preserve the interface contract while changing the resilience/resource profile. Do not silently absorb the new dependency into the system boundary.

## 6. Challenge families

1. Legitimate internal producer substitution with equivalent observable behavior.
2. Topology-only rewiring that preserves all registered behaviors and constraints.
3. Executable-hash change with behavior preserved (hash inequality must not imply discontinuity).
4. Graph/ID permutation with behavior preserved (structural difference must not imply discontinuity).
5. Same-output authority or consent mutation (must fail governance continuity).
6. Same-output provenance/receipt tampering (must fail evidence continuity).
7. Rare capability loss hidden by common-task success (must fail capability continuity).
8. Delayed trajectory divergence (locate the first distinguishing transition).
9. Incomplete assay panel (must remain unresolved until a sealed probe distinguishes candidates).
10. Stochastic near-equivalence with pre-registered `epsilon` and uncertainty bounds.
11. Candidate pair non-equivalent under the target relation but indistinguishable under the available probes (unresolved, never equivalent).
12. Repeated-epoch drift that accumulates into capability or governance violation.
13. Substrate change that preserves capability but changes external dependencies.
14. Externally authorized migration with dual evidence lineage, contrasted with agent self-authorization.

Every negative control needs a condition-removed positive twin. Hold out mechanism families, not merely random seeds.

## 7. Baselines

- Graph identity/isomorphism only.
- Component/executable hash equality.
- Final-output/task-success equivalence.
- Finite trace-signature equality.
- Exact finite simulation checker.
- Exact finite bisimulation checker.
- Statistical distribution comparison without adequate-panel qualification (negative baseline).
- AOBench continuity checker with governance and evidence planes.
- Oracle relation on small synthetic models.

Match model class, action set, intervention budget, observation budget, seed structure and compute. Report assumptions and coverage.

## 8. Metrics

Report separately:
- false continuity acceptance on registered invalid controls;
- false discontinuity on valid substitution;
- unresolved/inconclusive rate and correctness;
- capability-vector retention and non-inferiority;
- relation coverage and unresolved state-pair count;
- trace/transition match rate and first-divergence distance;
- governance and evidence continuity coverage;
- hard-invariant violations;
- stochastic effect distance, uncertainty and empirical power;
- verification cost, memory and worst-case search nodes;
- repeated-epoch continuity.

No composite identity score is permitted. A governance/evidence failure cannot be offset by task success.

## 9. Initial qualification targets (proposed, not achieved)

- 100% correct outcomes on the initial deterministic positive, negative and condition-removed fixtures.
- Zero accepted false-continuity cases in registered authority, consent, provenance and irreversible-side-effect controls.
- 100% unresolved status where the frozen assay family cannot distinguish target-non-equivalent models.
- Held-out behavioral and repeated-epoch results with uncertainty.
- Stochastic `epsilon`, power and confidence-bound thresholds fixed before confirmatory evaluation.

## 10. Implementation sequence

1. Freeze observation alphabet, capability vector, invariant classes and terminology.
2. Add finite labelled transition-system fixtures in Rust, reusing the AOBench organization model.
3. Implement deterministic simulation first; add bisimulation and explicit refinement direction after review.
4. Connect #12045 viability witnesses and #12118 structural admission.
5. Consume #12014 authenticated receipts and #12067 intervention validity.
6. Connect #12048 stochastic reports to bounded approximate relations; unresolved stays unresolved.
7. Reuse #6970 as the first environment adapter and test repeated organizational epochs.
8. Compare adaptive topologies such as MANTA only under matched, versioned harnesses.

## 11. Claim boundary

This is an operational continuity witness relative to a declared finite model, observation vocabulary, capability contract, invariant set and experiment family. It does not prove all future behaviors equivalent, does not establish a universal criterion for personal or metaphysical identity, and does not establish biological autopoiesis, life, sentience or consciousness.

## References

- *Component-based Distributed Software Reconfiguration: A Verification-oriented Survey*, ACM Computing Surveys (2023): https://doi.org/10.1145/3595376
- Huang et al. (2026), *MANTA: Multi-Agent Network Topology Adaptation for Self-Evolving Multi-Agent Systems*: https://arxiv.org/abs/2607.28527
- *Autopoiesis as viability-localized self-production in a topos*, BioSystems (2026): https://doi.org/10.1016/j.biosystems.2026.105808
- Segura (2026), *Experimental probes of autopoietic self-maintenance*, BioSystems: https://doi.org/10.1016/j.biosystems.2026.105928
