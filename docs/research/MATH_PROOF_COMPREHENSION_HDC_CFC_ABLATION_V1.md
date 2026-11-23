# HDC / CfC ablation programme for mathematical proof comprehension

**Issue:** [#13440 — MATH-HDC-CFC-ABLATION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13440)
**Parent:** [#13293 — MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)
**Status:** prospective ablation design; no HDC/CfC performance result is claimed.

## Research question

Do hyperdimensional computing (HDC) and closed-form continuous-time (CfC) representations improve proof comprehension tasks beyond strong retrieval and symbolic baselines, under matched compute and data budgets?

Working hypothesis: HDC may bind theorem/assumption/warrant identities across renaming and compositional variation; CfC may model the sequence of proof-state transitions in a ProofPlanV1. Either hypothesis can fail. Mathematical truth, theorem identity and proof validity remain under exact symbolic/Lean authority.

## Mandatory baseline and ablation arms

| Arm | Representation / method | Purpose |
|---|---|---|
| A | BM25 / lexical retrieval | String-based lower baseline |
| B | Dense embedding retrieval | Semantic retrieval baseline |
| C | LeanSearch v2 | Strong semantic premise retrieval |
| D | LeanGraphSearch-style semantic seed + dependency PPR + reranking | Strong graph retrieval baseline |
| E | Type-directed / symbolic retrieval | Formal-structure baseline |
| F | HDC only | Test compositional symbolic binding |
| G | CfC only | Test proof-state transition modelling |
| H | HDC + CfC | Test hybrid complementarity |
| I | HDC + CfC + symbolic constraints / evidence filtering | Test constrained hybrid |

Reuse TheoremGraph / LeanGraph where licensing and version pins permit. Do not recreate their corpus-scale graph.

## Task suites

### 1. Role-conditioned dependency retrieval

Retrieve proof dependencies by role: declaration/type support, proof-value support, definitions/reducible computation, inductive/constructor/recursor machinery, and reviewed mathematical roles such as construction, key estimate, invariant, reduction or infrastructure. Evaluate Recall@k, nDCG and annotated complete-route coverage on theorem-family and repository holdouts.

### 2. Proof-plan transition reconstruction

Given a partial ProofPlanV1 state, reconstruct the next mathematical action, consumed warrants/evidence, required side conditions, generated subgoals, preserved invariants and valid alternative routes. Do not score tactic-name prediction as proof understanding. Gold labels require independent expert review and checked proof receipts.

### 3. Representation invariance and interventions

Include alpha-renaming, short-name collisions, reordered hypotheses/context, equivalent statement rewrites, generated-helper packaging changes, added/removed assumptions, reversed inference direction, route-only mutations preserving theorem truth, and altered domain/instance/definition. Measure semantic disposition, retrieval stability and plan-role preservation separately.

### 4. Local-to-global consistency

Use identical source/theorem/hypothesis/warrant preimages between local rejection and global proof planning. Evaluate paired receipts from independent post-hoc adjudication; report CIRR numerator/denominator, unresolved-outcome bounds and conservative Wilson envelope. Confirmatory intervals must account for theorem-family/model clustering.

## Experimental controls

- Split by theorem family and source repository, not random paraphrase.
- Freeze source revision, Lean toolchain, graph snapshot, task identity, evidence, prompts, model/config, context and resource budgets.
- Match training data, candidate pool, HDC dimension, CfC hidden state, memory, wall-clock, tool calls and hardware budget.
- Report no-training and trained variants; train/dev/test contamination boundaries are explicit.
- Separate reader-facing explanation and prover-facing context.
- Measure per-family metrics, abstentions, false acceptance/rejection, p50/p95 latency, peak memory/VRAM and throughput.

## Prospective targets

| Outcome | Target |
|---|---:|
| HDC role-binding advantage | +5 percentage points over strongest non-HDC matched baseline on held-out role-conditioned retrieval or plan-evidence binding |
| CfC transition advantage | +5 percentage points over matched non-temporal sequence baseline on held-out plan-transition reconstruction |
| Hybrid complementarity | Beat best individual component on at least two preregistered primary tasks; paired held-out confidence intervals exclude zero |
| Representation invariance | ≤1% semantic-disposition change under equivalence-preserving rename/reorder transforms |
| False formal-proof acceptance | Zero on seeded unsound/vacuous/missing-assumption controls |
| Efficiency | Publish p50/p95 latency, peak memory/VRAM and throughput; no accuracy claim without cost curve |

These are prospective research thresholds, not observed performance.

## Decision policy

If HDC/CfC do not beat matched baselines, retain the negative result and do not promote the component into the comprehension pipeline. HDC/CfC may rank candidates or model plan trajectories, but cannot authorize theorem identity, semantic entailment, kernel acceptance or reviewer status. The system must always preserve the exact source and theorem scope and must abstain on unsupported mathematical claims.

## Deliverables

1. Frozen task contract and family-level train/dev/test split.
2. Matched baseline and ablation runner.
3. Reproducible model/config/resource receipts.
4. Per-task/per-family results and uncertainty.
5. Error analysis separating identity failure, role confusion, missing side conditions, sequence errors and invalid inference.
6. Decision record: integrate, retain as experimental, or reject each component.

## 8. Additional mandatory transformation baseline (2026-11-24)

Use the expert-authored equivalent-reformulation controls from [MathAdv](https://arxiv.org/abs/2608.25449) as an adjacent baseline. Keep its equivalent presentation transformations separate from our semantic mutations: alpha-renaming/reordering/equivalent rewrites should preserve semantic disposition, while changed domains, removed assumptions, reversed warrants and changed definitions should trigger their appropriate changed/unknown dispositions. Report invariance and mutation sensitivity separately; a model that is invariant to every change is just as defective as one that changes under harmless renaming.
