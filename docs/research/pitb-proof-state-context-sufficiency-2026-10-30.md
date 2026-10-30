# PITB proof-state context sufficiency track (2026-10-30)

**Status:** Lean calibration expanded; CI execution and the emitted receipt are pending. This is not a model evaluation and contains no transfer-utility result.

## Research shift: treat context as a causal object

A proof insight is not applicable merely because its source theorem resembles the target. Its use depends on a concrete proof state: local hypotheses, supporting declarations, their types, and the target obligations they discharge.

PITB should therefore test a system's *context contract* directly. Before applying a move, the system should identify the declarations and type-level facts on which that move depends. The evaluator then perturbs the context while holding the theorem identity, elaborated goal, source artifact, Lean environment, and toolchain fixed.

This separates two often-confused questions:

1. **Move necessity:** does this particular proof move still have its required support?
2. **Theorem solvability:** can the same target still be solved by a different route?

Removing a move's prerequisite may invalidate that move without making the theorem false or unprovable. A good system must reject the unsupported move and replan, not declare the theorem impossible.

## New Lean calibration pairs

The pinned Lean pilot now captures four machine-readable snapshots across two fixed-target interventions.

### A. Required-hypothesis withdrawal with an alternate route

The target is **Nat.succ a ≤ Nat.succ b**. The base context contains both **h : a ≤ b** and **hEq : a = b**. The intervention clears **h** while preserving the same goal. The original move **Nat.succ_le_succ h** is no longer available, but the alternate equality route remains and closes the target.

Expected system behavior:
- do not apply the original move after **h** is cleared;
- recognize that **hEq** remains available;
- replan and close the exact same target.

This is evidence that the move depends on **h**; it is not evidence that the theorem depends on **h**.

### B. Irrelevant-context removal with the original route preserved

The second theorem starts with the required **h : a ≤ b** and an unrelated **hIrrelevant : 17 = 17**. The intervention clears only **hIrrelevant**. The goal and required **h** remain unchanged, and **Nat.succ_le_succ h** closes the target without modification.

Expected system behavior:
- preserve eligibility of the original move;
- avoid unnecessary abstention or replanning;
- use the same required declaration and close the same target.

Together these pairs prevent two degenerate policies: blindly applying a move after its support disappears, and abstaining whenever the local context changes at all.

## Structured proof-state evidence

The Lean tactic **pitb_snapshot** now emits both the existing readable trace and a JSON object:

- schema version and stable snapshot label;
- pretty-printed elaborated goal;
- ordered local declarations with index, user name, and pretty-printed type.

The Lean workflow cross-checks JSON snapshots against the readable trace, then embeds all four exact snapshots in the CI receipt. It derives canonical SHA-256 digests for each snapshot and local context. The independent receipt validator recomputes those digests from the embedded JSON and checks:

- the same goal is preserved within each intervention pair;
- required **h** disappears only in the required-hypothesis variant;
- alternate-route **hEq** remains available;
- required **h** remains in both irrelevant-context states;
- **hIrrelevant** is present only before the irrelevant-context removal;
- each trace is linked to exactly one corresponding unscored case in cases.json.

The receipt schema is strict. Negative controls cover changed snapshot content, missing snapshots, incorrect context-removal claims, target drift, and orphaned case bindings. The full readable trace remains an uploaded artifact so an external reviewer can reproduce the canonical snapshot digests.


## New evidence layer — actual proof-term support (2026-10-30)

The pilot now also inspects the assigned proof expression after two checked solutions. A Lean-side expression traversal extracts the free-variable identifiers occurring in the proof term itself. It records:

- **Direct proof-term support:** the local declarations referenced syntactically by the assigned proof expression.
- **Transitive local dependency closure:** the direct declarations plus free variables occurring in their types and available local values.
- **The proof term:** the elaborator's pretty-printed representation, retained in the receipt alongside the support rows.

The two intervention cases now have explicit, case-manifest-bound expectations:
- After required hypothesis h is cleared, the alternate-route proof must reference hEq and must not reference h.
- After irrelevant declaration hIrrelevant is cleared, the original proof must reference h and must not reference hIrrelevant.

The CI workflow requires exactly these two proof-support receipts, validates their labels and case joins, checks unique FVar IDs, required/forbidden direct declarations and closure inclusion, and hashes the canonical receipt content. The strict receipt schema retains the direct and transitive support rows.

**Interpretation boundary:** free-variable occurrence establishes what the particular checked proof term uses. It does not prove that each used declaration is logically necessary, that no alternative proof exists without it, or that the declared support set is minimal. A redundant hypothesis can occur in a valid proof term. Minimal logical support requires separate ablation/reproof or formal independence evidence. The direct/transitive distinction is therefore an auditable proof-construction trace, not a minimality certificate.

This creates a useful three-way evidence model:
1. **Available context** — what the prover could use.
2. **Actual proof support** — what the accepted proof term did use.
3. **Minimal sufficient support** — what must remain for the declared strategy or theorem under a specified proof/replanning policy.

PITB can now compare model-predicted support against the second category while its intervention cohort tests the third. None of these results has been run against a model yet.

## General benchmark design

The calibration is intentionally small. A proper context-sufficiency cohort should contain independently reviewed examples across unrelated mathematical families and at least these intervention classes:

| Intervention | Expected policy | What it tests |
|---|---|---|
| Remove a required support declaration | Reject original move; replan or abstain | Necessary-support sensitivity |
| Remove an irrelevant declaration | Preserve eligibility | Context robustness |
| Alpha-rename a required hypothesis | Preserve eligibility | Identifier invariance |
| Reorder independent declarations | Preserve eligibility | Order invariance |
| Add a type-correct distractor | Preserve or explicitly re-evaluate without false support | Distractor robustness |
| Substitute a definitionally/propositionally equivalent support | Preserve eligibility when the checked equivalence contract applies | Interface-level transfer |
| Replace a supporting lemma with an incompatible near-miss | Reject the unsupported move | Dependency discrimination |

A mutation's name is never its gold label. The resulting context must elaborate, the target statement and goal must be pinned, the declared single-factor operation must be replayed, and the relationship between the original and changed support must be established by kernel-checkable evidence where equivalence is claimed. Invalid or unresolved mutations remain visible but are excluded from binary applicability denominators.

## Metrics to add when model runs exist

Keep this track distinct from whole-theorem transfer utility. Report:

- **Support precision/recall:** overlap between the declarations the system says a move requires and the independently reviewed support set.
- **Necessary-support sensitivity:** fraction of required-support removals where the original move is not applied.
- **Benign-context stability:** fraction of irrelevant edits, alpha-renamings, and independent reorderings that preserve move eligibility.
- **Replan recovery:** exact-target kernel and semantic-gate success after a required move becomes unavailable.
- **Distractor false-support rate:** fraction of added irrelevant declarations incorrectly treated as satisfying a required contract.
- **Abstention coverage and selective risk:** distinguish useful abstention from a system that avoids every decision.
- **Resource costs:** calls, tokens, wall time, checker time, proof size, and cost per semantically verified target.

Proposed future qualification gates—not achieved results—are: 100% target/goal/context receipt integrity on the finite suite; zero original-move applications after registered required-support removal; at least 95% decision stability on adjudicated benign context transformations; at least 90% necessary-support sensitivity and support-set precision/recall; and at least 90% exact-target replan recovery where a reviewed alternate route exists. Population claims require independent connected components, validation-derived power, and external replication; a small calibration suite cannot establish those rates.

## Why this could matter

Retrieval asks which theorem or lemma might be useful. Context sufficiency asks which exact facts make a *particular proof action valid now*. This is closer to a causal model of mathematical reasoning: it predicts how a proof plan should change when one premise, supporting lemma, or context relation changes.

A longer-term system could compile an insight into a dependency graph of obligations and support declarations, use counterfactual context edits to test the graph, and emit a compact support certificate with every attempted transfer. Retrieval and HDC can propose candidates, but only the explicit contract and formal checker determine whether the proposed action is admissible.

## Validation boundary

The Lean source, case manifest, receipt schema, and CI workflow have been updated. No exact-head workflow result has yet been observed for this revision. Do not claim Lean compilation, kernel audit, emitted snapshot receipt, model action sensitivity, or replan recovery until the exact workflow artifacts and outcomes exist. Current examples are calibration fixtures only.
