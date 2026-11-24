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

## 9. Implementation-readiness audit — pin the real adapter boundary (2026-11-24)

The current main-branch code snapshot inspected for this audit is Luminous-Dynamics/symthaea@bc915c485402eaf955370eb09cffb6e4fad09e82. This is a source-code audit, not a benchmark execution.

### What exists and what does not

**HDC source-structure encoder exists.** src/hdc/code_encoder.rs defines CodeHDEncoder and encodes parsed code entities using role vectors, name vectors, annotations, ordered child positions and relation triples. It is a useful architectural starting point for compositional proof-plan encoding. It is not yet a Lean elaborated-AST / ProofPlanV1 adapter. Its encode_name path lowercases names and uses byte n-grams; raw variable identifiers therefore remain representation-sensitive. The alpha-renaming arm must use a binder-normalized typed representation (e.g. binder positions/indices) before HDC encoding, not expect the current character-name encoder to be invariant automatically.

**A CfC sequence network exists.** src/dynamics/cfc/network.rs exposes stateful forward(input, dt), forward_sequence(inputs, dts), state reset/state access, BPTT sequence training, and deterministic from_genesis initialization. This can support a proof-plan trajectory experiment, but no evidence yet shows that a proof-plan transition task has been wired into this network. The runner must assert inputs.len() == dts.len() before calling forward_sequence, because that forward path truncates to the shorter sequence; BPTT training performs explicit length checks.

**The inspected Broca structural generator is not the proof-plan adapter.** crates/domains/symthaea-broca/src/structural_generator.rs says that a specialized CfC head is for "real use", contains a mock two-node synthesis path, and derives its example HDC vector from value hash and source line rather than a parsed mathematical/Lean structure. Do not cite this file as evidence that mathematical proof-plan generation already uses HDC+CfC.

**The WGPU CfC backend is not a GPU result.** src/dynamics/cfc_gpu.rs reports is_gpu_available() == false, describes WGPU as a placeholder and resolves Auto to CPU. Any initial HDC/CfC result must report the CPU backend. No GPU speed or acceleration claim is permitted until a real WGPU kernel path and an independently measured hardware receipt exist.

### Required proof-specific adapters

1. **Formal AST → typed proof-state record.** Preserve source/theorem identity outside the vector representation. Parse elaborated goals, binder scopes, hypotheses, local context, action, side conditions, subgoals, invariant and exact evidence IDs. Never collapse theorem identity into approximate vector similarity.
2. **Typed proof-state record → HDC.** Use role vectors for goal, hypothesis, premise, warrant, action, side_condition, subgoal, invariant and evidence; relation vectors for uses, discharges, produces, preserves, specializes, rewrites and depends_on. Normalize bound-variable names before encoding. Keep the reversible/approximate HDC representation as a candidate-retrieval feature, not a truth authority.
3. **HDC state → fixed-dimension CfC input.** HDC's default HDC_DIMENSION is 16,384 while the default CfC network input dimension is 64. Specify and freeze a projection (including seed, matrix digest and precision) or a directly matched lower-dimensional HDC configuration. The projection, hidden state, parameter count and compute cost must be shared or explicitly matched across baselines.
4. **ProofPlanV1 trajectory → CfC.** Encode the current proof state and predict a semantic transition: action class, warrant/evidence consumed, side conditions and subgoal structure. Do not train only on tactic strings or token continuation. Use from_genesis for deterministic initialization and record the seed.
5. **Time semantics.** A proof plan is a sequence of discrete logical events; elapsed model wall-clock time is not automatically a mathematical time variable. Primary transition ablations should use a preregistered unit-step dt=1 or a clearly defined domain-specific event-time measure. Test irregular dt as a separate intervention. Do not feed hardware/network latency into the CfC time constant and call resulting gains mathematical reasoning.

### Stronger baseline factorial

In addition to retrieval arms A–I already listed above, the transition-reconstruction study must compare:

- static HDC embedding + symbolic retrieval;
- a parameter-matched MLP over the same per-state features (no recurrent state);
- GRU and LSTM sequence baselines with the same input features and parameter budget;
- a temporal convolution / fixed-window baseline;
- CfC with fixed unit dt;
- CfC with controlled irregular dt;
- HDC + each sequence model, including HDC+GRU and HDC+CfC;
- the symbolic/Lean-constrained hybrid.

The key CfC claim is not "continuous time is more biologically plausible." It is: **at matched inputs, training data, parameter count and compute, does explicit continuous-time state evolution improve held-out transition reconstruction or robustness to irregular event intervals?** A negative result is a valid result.

### Mandatory acceptance gates before a model run

- A compiler-checked adapter with deterministic fixtures for at least 25 proof-plan transitions across 5 held-out theorem families.
- Binder-renaming / alpha-normalization metamorphic controls and short-name collisions.
- Sequence-shuffle controls: the transition model should detect order changes when the proof plan is order-sensitive.
- Context permutation controls: exact logical disposition should remain stable under a semantics-preserving reorder.
- Missing-assumption and reversed-warrant controls must fail through symbolic/Lean authority even if HDC/CfC predicts a high-similarity state.
- No-training and trained conditions with family-level split; no random split of variants of the same theorem.
- Exact model, seed, parameter count, projection digest, dt schedule, tokens, tool calls, p50/p95 latency, CPU/GPU identity, peak RAM/VRAM and throughput receipts.

Until these adapters and gates exist, classify HDC/CfC as **available components under prospective evaluation**, not as an implemented proof-comprehension capability.
