# Open Mobility — Decision-Sensitive Measurement Triage v0.1

**Status:** research protocol draft  
**Tracking issue:** [MOBILITY-ASSIST-010 / #7904](https://github.com/Luminous-Dynamics/symthaea/issues/7904)  
**Implementation:** draft [PR #7907](https://github.com/Luminous-Dynamics/symthaea/pull/7907)  
**Interval comparator:** [MOBILITY-ASSIST-009 / PR #7892](https://github.com/Luminous-Dynamics/symthaea/pull/7892)  
**Action catalog:** [measurement-triage-actions-v1.json](./fixtures/measurement-triage-actions-v1.json)

## 1. Research question

Given a valid interval-aware comparison of wearable-robot architectures, which already-declared measurement proposals most directly address uncertainty that could change the dominance relation—without confusing a genuine trade-off with a missing measurement, or letting an attractive predicted score bypass feasibility and evidence requirements?

This is an adapter over existing comparison and active-experiment-selection contracts. It is not a new experiment scheduler or a physical measurement authority.

## 2. The critical distinction

The robust comparison layer uses three separate conditions:

- **Robust dominance:** one candidate is no worse throughout its declared envelope on every objective and strictly better on at least one.
- **Definite trade-off:** different objectives definitely favour different candidates. Narrowing measurement intervals alone does not guarantee a single winner; a user/task priority or hard constraint may be needed.
- **Additional evidence may change dominance:** neither candidate robustly dominates, no opposing-dimension trade-off is established, and at least one dominance direction remains logically possible inside the bounds.

Only the third condition feeds the dominance-sensitive measurement queue. Each pair's dimensions that are not separated throughout their intervals are published as **metric-pair cells**. These cells are diagnostic targets; they do not mean a measurement outcome is known in advance or that a test will resolve the comparison.

A wide interval that is irrelevant to a feasible decision is not automatically high priority. Conversely, measuring one target dimension may leave other dimensions unresolved. Coverage counts are not expected information gain (EIG), probabilities, or a promise of resolution.

## 3. Existing contracts to compose

- [MOBILITY-ASSIST-009 / #7890 / PR #7892](https://github.com/Luminous-Dynamics/symthaea/pull/7892): interval comparison, pairwise dominance receipts, task and condition identity.
- [Active experiment-selection benchmark #7756](https://github.com/Luminous-Dynamics/symthaea/issues/7756): simulation-only comparison of active experiment ranking against baselines.
- [Engineering epistemic-action semantics #6203](https://github.com/Luminous-Dynamics/symthaea/issues/6203): measurement and experiment proposals remain separate from action execution.
- [Calibration-chain / completeness contract #7592](https://github.com/Luminous-Dynamics/symthaea/issues/7592): physical evidence provenance requirements.
- [Shared active-learning semantics #6298](https://github.com/Luminous-Dynamics/symthaea/issues/6298): feasibility, confounding, costs, batch design and stopping rules.

The mobility adapter may consume a qualified EIG/active-inference score when the upstream model and likelihoods are appropriately qualified. It must preserve prediction status and calibration limitations, and must never fabricate outcome likelihoods.

## 4. Input contract

Each action catalog entry must declare:

- stable action ID and kind (`analysis`, `simulation`, `bench_measurement`, or `human_study`);
- exact task-protocol revision, matching the source comparison;
- all candidate IDs to be assessed under matched conditions;
- metrics the action is expected to measure for every listed candidate;
- finite, ordered, non-negative planning-cost bounds;
- explicit prerequisite statuses for protocol freeze, candidate coverage, feasibility, safety review, consent, calibration and evidence provenance.

For physical bench measurement, safety review and a relevant current calibration status must be explicitly satisfied. For human studies, consent/ethics and safety requirements must be explicitly satisfied. Unknown or blocked required gates exclude the action. Analysis/simulation may mark physical-only gates `not_applicable`; that designation is caller-declared, not independently authenticated.

A stale protocol, unknown/duplicate candidate, duplicate action ID, duplicate metric target, malformed cost, incomplete source pair matrix, or source report with missing/duplicate/invalid metric intervals rejects triage instead of quietly producing a partial result.

## 5. Deterministic action frontier

For a given source comparison:

1. Enumerate only pairwise comparisons whose dominance may change within the declared envelopes.
2. Materialize the ordered cells from their not-separated dimensions.
3. A declared not-separated dimension is a measurement target only when at least one candidate's echoed interval has width greater than the numerical comparison tolerance. Equal point-valued intervals contain no represented uncertainty to narrow and are not offered as measurement work.
4. An action covers a cell only if it names both candidates in the pair and names that metric as a target.
5. Keep blocked/unknown actions in the report with reasons but never include them in the eligible action frontier.
6. Among eligible actions that cover at least one cell, action A dominates action B only if A's covered-cell set is a superset of B's and A's conservative upper cost bound is no greater, with strict improvement in coverage or cost.
7. Return every non-dominated action, not one scalar winner. Actions that cover different cells stay available even when their coverage counts are equal.

The receipt binds the canonical serialized source comparison using BLAKE3. A digest establishes content identity of the supplied structure, not sensor authenticity, valid calibration, consent, truth or authorization.

## 6. Required report outputs

- source comparison ID, schema, canonical BLAKE3 digest, task-protocol revision and evidence level;
- count of potentially dominance-changing candidate pairs and unique metric-pair cells;
- action dispositions and explicit failed/unknown gate reasons;
- exact coverage cells and upper planning-cost bounds;
- the non-dominated action set and dominating action IDs;
- assumptions and claim ceiling.

The report must not imply a completed measurement, a selected actuator command, clinical benefit, or guaranteed decision resolution.

## 7. Software benchmark gates

These are proposed targets, not results or clinical standards.

| Gate | Target |
|---|---|
| Decision-state distinction | 100% correct fixtures for robust dominance, established trade-off and additional-evidence-may-change-dominance |
| Interval-width filter | Equal point-valued bounds are never proposed as a measurement target merely because their dimensions are not separated |  
| Source metric integrity | 100% rejection of source receipts with missing, duplicate, non-finite, reversed, negative or invalid normalized metric intervals |  
| Cell accounting | Every reported coverage set exactly matches named pairs × named metrics that belong to the source's decision-sensitive cells |
| Trade-off protection | Zero proposals claim that interval narrowing alone will resolve an established definite trade-off |
| Hard gates | 100% of blocked/unknown required prerequisites excluded from action frontier; unknown fails closed |
| Protocol/context | 100% rejection of stale protocol and unknown/duplicate candidates |
| Pareto action set | Exact agreement with hand-calculated coverage-set/cost dominance fixtures, preserving actions with different coverage |
| Determinism | Byte-identical report for 100 repeated evaluations of the same source report and action catalog |
| Provenance | Source report schema and canonical content digest present in every successful receipt |
| No invented EIG | No probability or expected-utility number generated from interval width or cell count alone |
| Claim ceiling | No code path authorizes or executes actions or promotes synthetic output into physical/clinical evidence |

## 8. Benchmark against baselines

Once the deterministic adapter is integrated with the existing experiment-selection benchmark, compare the *triage-plus-ranking* policy under equal simulated resource budgets against:

- widest-interval first;
- deterministic fixed-plan measurement order;
- seeded random action;
- qualified EIG only where outcome likelihoods are available and their calibration is independently checked.

Report separately:

- share of predeclared decision-sensitive cells covered by eligible actions;
- cost to resolve a predeclared synthetic decision;
- false-resolution and missed-critical-measurement rates;
- fraction of proposals blocked by feasibility/safety/consent/calibration/provenance gates;
- EIG prediction versus realized entropy reduction only for the EIG-enabled baseline;
- results by scenario family and repeated seed.

Use held-out scenario families and explicit negative controls. Synthetic cell coverage is not a real-world success rate. If the catalog lacks an eligible action, return an empty frontier with blockers rather than fabricate a useful test.

## 9. Initial hostile fixtures

The versioned action catalog includes a mass target, an energy target, a combined action, a cheaper duplicate-coverage action, a measurement aimed only at a definite trade-off, a physically blocked bench action, and a human study with unknown consent status.

Follow-on tests should add: stale calibration, unmatched task speed/payload/grade, incomplete candidate coverage, a wide but irrelevant dimension, a narrow metric that blocks dominance, shared calibration error, an infeasible high-predicted-EIG proposal, and discovery/confirmation contamination.

## 10. Research basis

- A review of 139 occupational-exoskeleton studies found substantial variation in experimental protocols and noted growing value of user-experience information as testing becomes more applied: https://doi.org/10.1016/j.apergo.2021.103582
- A 2026 lower-limb exoskeleton evaluation review reports a lack of standardized evaluation protocols and gaps in safety, stability, comfort and environmental adaptability: https://www.sciencedirect.com/science/article/pii/S0921889026003179
- A 2026 human-centred HILO perspective argues for better representation of individual capabilities, clinical expertise and personalized training: https://pubmed.ncbi.nlm.nih.gov/42657862/
- A September 2026 preprint explores multi-objective human-in-the-loop Bayesian optimization of exoskeleton controllers across metabolic cost and ordinal comfort feedback: https://arxiv.org/abs/2609.30695

The research implication is not “always choose the test with the highest predicted information.” It is “identify which uncertainty could alter an admissible decision, propose feasible evidence-gathering actions, and preserve the limits of the evidence and model.”

## 11. Claim ceiling and next steps

This protocol creates deterministic proposals over caller-supplied intervals, action targets, costs and gate statuses. It does not estimate or validate these inputs, predict physical measurement results, establish clinical efficacy, qualify hardware, infer consent, or authorize experiments.

Next: independently review the triage contract, reconcile it with the generic #7756 action-ranker semantics, then connect real physical proposals only through the calibration/provenance contract and appropriately reviewed human-study procedures.
