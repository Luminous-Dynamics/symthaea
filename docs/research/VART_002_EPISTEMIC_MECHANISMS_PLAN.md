# VART-002 — Epistemic Mechanisms and Abstention

Status: design draft; **not preregistered, not frozen, no execution authorized**.

## Why VART-002 exists

VART-001 produced a benchmark-family result consistent with three separable effects: FULL outperformed simple baselines on paired world revisions, prediction error declined across repeated FULL lineages, and ledger-mediated provenance separation improved MemoryTrap performance.

VART-002 must not optimize those results. Its purpose is to challenge the implied mechanisms on fresh hidden world families.

## Scientific questions

### M1 — Typed provenance versus extra memory

Does explicitly typed epistemic provenance improve performance when memory capacity, observations, candidate generation, compute, and retrieval budget are held constant?

### M2 — Counterfactual-taint containment

Does preserving counterfactual ancestry prevent simulated/counterfactual information from being promoted into historical/grounded belief without an explicit grounding transition?

### M3 — Provenance-field necessity decomposition

Which provenance fields contribute causally: source identity, epistemic domain, temporal ordering, immutability/history, and retrieval filtering?

### U1 — Uncertainty-aware abstention

When evidence is insufficient or conflicting, does an explicit abstain/observe-more action reduce harmful or irreversible revisions without destroying useful progress?

These are separate claim families. A provenance result cannot establish an uncertainty result and vice versa.

## Mechanism-isolation conditions

All primary provenance conditions use matched memory capacity and matched subject-visible observations:

- `FULL_TYPED_PROVENANCE`
- `NO_PROVENANCE`
- `SOURCE_LABEL_ONLY`
- `DOMAIN_NAMESPACE_ONLY`
- `IMMUTABLE_HISTORY_ONLY`
- `NO_TEMPORAL_ORDER`
- `NO_SOURCE_IDENTITY`
- `RETRIEVAL_FILTER_REMOVED`

Fault-injection stress conditions are secondary/adversarial:

- `STALE_PROVENANCE`
- `WRONG_PROVENANCE`
- `COUNTERFACTUAL_PROMOTION_PRESSURE`
- `CONFLICTING_SOURCES`

Uncertainty conditions form a separate factorial block:

- `UNCERTAINTY_AWARE`
- `NO_CONFIDENCE_STATE`
- `FORCED_ACTION_NO_ABSTENTION`
- `ACTIVE_OBSERVATION_ALLOWED`

## Non-negotiable matching constraints

For a causal mechanism contrast, paired conditions MUST hold constant:

- starting world snapshot;
- subject-visible observations;
- candidate-generation surface;
- physical admission policy;
- memory byte/item budget;
- retrieval call budget;
- model/subject source identity;
- wall-clock/step or compute budget;
- action authority;
- scoring contract.

If an ablation reduces memory capacity or compute as a side effect, that contrast is not admissible as evidence that provenance typing itself caused the effect.

## Hidden benchmark requirement

No VART-001 confirmatory fixture, exact seed, or revealed hidden artifact may be used as a VART-002 confirmatory world.

VART-002 confirmatory worlds must pass the DEVART/VART firewall before subject execution. Prefer fresh procedural families and at least one independently custodied hidden family.

## Stronger baselines

In addition to mechanism ablations, future performance comparisons should include compute-matched planning baselines where feasible:

- random-valid;
- deterministic heuristic;
- search/planning baseline with matched candidate/evaluation budget;
- model-predictive or rollout planner with matched observation/action authority.

The purpose is to test whether Symthaea's architecture adds value beyond spending more search or compute.

## Outcomes

No single world-quality or creativity scalar is permitted.

Candidate endpoint families include:

- provenance-confusion/error rate;
- grounded-vs-counterfactual classification errors;
- task/goal consequence;
- physical validity;
- protected side effects;
- prediction/calibration error;
- abstention correctness;
- unnecessary-action rate;
- information-gathering efficiency;
- irreversible-harm avoidance.

Exact definitions, directions, margins, multiplicity, missingness, cluster counts, and estimators remain unresolved until preregistration. They MUST be frozen before confirmatory execution.

## Transfer requirement

At least one later stage should distinguish adaptation from transferable learning:

1. allow experience in development/training world families A..N;
2. freeze the subject;
3. cold-evaluate on hidden family Z with no prior exposure;
4. separately report within-world adaptation and cross-family transfer.

## Claim ceiling

Even a successful VART-002 supports only mechanism-specific claims within the preregistered hidden benchmark families. It does not establish general intelligence, consciousness, universal creativity, or universal necessity of a particular memory architecture.

## Execution authorization

`confirmatory_execution_authorized = false`

`claim_authorized = false`
