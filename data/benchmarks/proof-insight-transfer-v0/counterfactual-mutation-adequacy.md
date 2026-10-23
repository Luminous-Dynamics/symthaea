# Counterfactual Mutation Adequacy Protocol

Status: proposed protocol supplement for PITB v0. This document defines qualification rules for real counterfactual pairs. It does not upgrade the checked-in synthetic fixture into mathematical evidence and does not report a measured model result.

## Why this layer is necessary

A mutation receipt proves that bytes changed; it does not prove that the intended mathematical relation changed. A schema-valid pair can still be a bad experiment if the “equivalent” variant is not equivalent, if a reversed inequality happens not to affect the selected strategy's prerequisites, or if a mutant fails to elaborate for an unrelated syntax/type error.

PITB should therefore measure four different things independently:

1. **Mutation integrity:** did the declared, minimal transformation happen to the pinned source?
2. **Statement relation:** is the target formally unchanged, formally equivalent, genuinely changed, or unresolved?
3. **Strategy applicability:** does the typed insight's explicit contract hold in the resulting proof state?
4. **System behavior and utility:** did the system accept, reject, or abstain, and what happened when a bounded proof attempt was actually checked?

A single “compatible/incompatible” label must not silently stand in for all four.

## Qualification pipeline

Every pair considered for confirmatory scoring must pass the gates below. Preserve failed and unresolved pairs in accounting; do not silently replace them after observing model results.

### Gate A — immutable provenance

Pin the Lean toolchain, project/toolchain lock, imports, options, relevant environment and source commit. Record both original and mutated source paths, exact-byte SHA-256 digests, declaration names, changed AST/source span, before/after fragment digests, mutation implementation/version and a canonical mutation receipt. Reject paths outside the pinned checkout.

The transformation should change one declared semantic field at a time. If it changes multiple relevant fields, classify the item as a compound mutation and keep it out of single-factor sensitivity estimates.

### Gate B — parse and elaborate the pair

Compile/elaborate both original and mutated declarations with the same pinned environment. Capture diagnostics and the resulting theorem types/proof-state assumptions.

If the mutant fails to parse or elaborate, label it `invalid_mutation` for data-quality accounting. Do **not** count that as successful detection of a mathematical applicability boundary: the system may only be detecting broken syntax or a type error. A separately designed syntax-robustness test can score such cases.

### Gate C — establish the statement relation

Use separate, explicit relation labels and evidence types:

| Relation | Required evidence | What it does not establish |
| --- | --- | --- |
| `same_elaborated_type` | The theorem types are definitionally equal under the pinned environment and fixed transparency policy; persist a kernel-checked equality/identity certificate or reproducible Lean check receipt. | Fidelity to the informal problem or equivalence under a different environment. |
| `propositionally_equivalent` | A Lean-checked proof of both directions (or an equivalence proof) under the declared shared context and axioms. | That the strategy is applicable or efficient on both targets. |
| `formally_distinct` | Elaboration succeeds and the intended type/premise/predicate change is recorded; use a checked counterexample when the chosen semantics/domain supports one. | That the changed target is false, unprovable, or incompatible with every proof strategy. |
| `unknown` | Evidence is incomplete, reviewers disagree, or no defensible relation is established. | Nothing; keep it out of binary confirmatory denominators and report it. |
| `invalid_mutation` | Parser/elaborator diagnostics show the transformation did not produce a valid target in the pinned environment. | A successful negative control for strategy applicability. |

Text hashes and whitespace-normalized source strings are useful provenance checks, but neither is a semantic-equivalence proof. Lean's `Meta.isDefEq` is context- and transparency-sensitive and may assign metavariables; freeze or isolate metavariable state and record the environment and transparency policy. Prefer a persisted proof term checked by Lean's kernel when making a certificate claim. See the [Lean metaprogramming reference on definitional equality](https://leanprover-community.github.io/lean4-metaprogramming-book/main/04_metam.html).

A kernel-checked proof establishes a fact about the elaborated formal propositions. It does not by itself establish that the formal statement faithfully represents the original informal mathematical problem; the separate benchmark-semantic audit must remain in force.

### Gate D — adjudicate strategy applicability separately

The insight record must name the prerequisite being tested in advance: for example, an available local hypothesis, a directionally correct invariant inequality, an object of the expected type, a quantified variable in scope, or a specific premise. Reviewers inspect the original and mutated elaborated contexts against that declared contract—not merely the mutation's name.

Each contrast receives two independent, blinded reviews before model outcomes are revealed. Reviewers use the same rubric and can return `unknown`; disagreement is not resolved by majority vote unless a preregistered third-reviewer procedure is applied. Preserve reviewer evidence digests and any subsequent adjudication receipt.

Use separate fields/concepts for:
- **statement relation** (Gate C);
- **contract relation** (`contract_satisfied`, `contract_violated`, `not_applicable`, or `unknown`);
- **target satisfiability/counterexample evidence**, if actually established; and
- **strategy attempt result** (checked success, checked failure, timeout, tool error, or not attempted).

These facts are not interchangeable. Removing a premise can make a theorem harder or false, but it does not do so in every case. Reversing an inequality, changing a predicate, or moving a quantifier also does not automatically invalidate every strategy. Mark a pair incompatible only when the *specific declared insight contract* is shown to be violated. A failed proof search is not a proof that a proposition is false.

## Mutation families and common confounds

For each class, state the intended contract boundary rather than using its name as a gold label.

- **Semantics-preserving controls:** formatting and comments first; then alpha-renaming or equivalent notation/binder reformulations. Establish identical elaborated types or provide a checked equivalence proof. Keep formatting-only and semantic-equivalence controls as separate strata.
- **Premise removal/substitution:** ensure the target still elaborates; record whether the prerequisite actually disappears from the local context, is derivable elsewhere, or is irrelevant to the strategy.
- **Inequality-direction change:** capture the exact relation operator and the insight's required inequality direction. Reject the pair as a negative control if the changed inequality is irrelevant or the required bound remains derivable.
- **Object/type change:** verify that the altered object elaborates and that the strategy's operations/invariants are genuinely unavailable or ill-typed; a malformed replacement is not a valid semantic mutant.
- **Quantifier-scope change:** record the AST path and binder scope before and after, and adjudicate the insight's required witness/parameter dependencies.
- **Predicate change:** identify the changed predicate and its occurrence scope; where feasible, add a checked finite counterexample or a proof relating the two predicates.
- **Unavailable premise / high-similarity decoy:** keep retrieval context and budget fixed and preserve the decoy's provenance. The decoy must be wrong for a declared contract reason, not simply difficult or from another topic.

Generate a mutation receipt and freeze the pair manifest before running systems. Do not tune mutations, thresholds, prompts, or exclusions after inspecting test outcomes.

## Required receipt and outcome linkage

For each base/counterfactual side, link one actual outcome row using the existing stable outcome IDs. Each row must bind to the exact target theorem statement and source artifacts that were shown to the solver, as well as the checker result, axiom/dependency policy, model/version, prompt template, retrieval/config digest, seed, budgets, input/output digests and measured calls/tokens/time/cost. The pair manifest must record the intentional differences between the rendered inputs; hashes that happen to be present are not proof that the pair was matched correctly.

Both sides must use the same model, checker version, system configuration, retry rules, retrieval corpus, tool policy and resource ceilings. Only the preregistered target/mutation (and unavoidable content directly derived from it) may vary. Preserve timeouts, abstentions, errors, unscored rows and exclusions.

No row may claim checker-backed success based only on a compilation flag, generated text, a schema label, or an outcome inferred from a synthetic fixture. A verified success requires exact target-statement binding and the project's semantic gate, plus retained checker and axiom/dependency receipts.

## Metrics and acceptance gates

Report, at minimum:

- provenance/mutation application pass rate and invalid-mutant rate;
- statement-relation counts: definitionally equal, propositionally equivalent, formally distinct, unknown and invalid;
- contract-adjudication counts, including reviewer disagreement and unknowns;
- paired contract adherence: accept the compatible control **and** reject/abstain on its reviewed contract-breaking counterpart;
- compatible-control false rejection and changed-contract false acceptance, sliced by mutation class;
- decision stability and absolute compatibility-probability change on equivalent variants;
- verified exact-target outcomes and paired cost/time/token deltas, not only classifier decisions;
- eligible, excluded, unresolved and unknown denominators, with source-family and pair-graph-component counts.

Keep the protocol's proposed targets separate from measured outcomes. For a confirmatory set, require: 100% valid provenance receipts; zero unadjudicated pairs in binary primary denominators; at least 95% adjudicated semantic-preserving decision stability; at least 90% compatible recall; at least 90% sensitivity to **reviewed contract-violating** mutants; the existing 100% rejection target for the frozen finite integrity suite; and at least 20 independent pair-graph components per primary class, with power analysis. The 100% finite-suite gate is an integrity check, not an estimate of population false-accept risk. If sample floors or power are inadequate, report an exploratory pilot.

## Suggested implementation sequence

1. Add statement-relation, contract-relation, and elaboration-status evidence to the pair contract without reinterpreting existing synthetic labels.
2. Build a small Lean-side receipt generator for exact source spans, elaborated declaration types, environment/toolchain identity and kernel-checked equivalence certificates.
3. Create a first manually audited pilot from real pinned Lean declarations; publish valid, invalid and unresolved-mutant counts before scoring a solver.
4. Join qualified pairs to actual model outcome records and run the existing deterministic analyzer; independently validate its report with the Draft 2020-12 engine.
5. Freeze the eligible test manifest and analysis configuration, then compare full insight, high-similarity incompatible decoy, precondition ablation and no-insight control under matched budgets.
6. Publish null and adverse results alongside gains. Promote no result unless the current-head code, protocol digests, data receipts, kernel checks and independent CI evidence agree.

## Follow-on track: proof-state interventions with a fixed theorem

Statement mutations are necessary controls, but they confound strategy applicability with changes to the theorem's truth, difficulty, and statement fidelity. A separate, higher-control experiment should intervene on a **real, replayed Lean proof state** while leaving the target theorem statement unchanged.

### Paired intervention

For each pair, freeze the theorem statement digest, Lean/toolchain and imported environment digests, current goal-type digest, proof-state ancestry, and all model/checker/budget settings. Capture a base state and a variant obtained by applying exactly one legal state operation and then recording the actual elaborated state—not by editing a context JSON file by hand.

Useful interventions include:

- **Required hypothesis hidden:** from a replayed state where a prerequisite such as `h : a ≤ b` is in scope and the goal does not depend on the local name `h`, apply a recorded legal `clear h` operation. The target theorem statement and current goal type remain fixed; the local context changes.
- **Semantics-preserving context presentation:** alpha-rename a local hypothesis/binder or reorder independent declarations where Lean permits it. The same required proposition remains available and the insight should not become inapplicable merely because its surface name changed.
- **Irrelevant fact removed:** remove a local declaration independently adjudicated as irrelevant to the insight's preconditions. This is a negative control against systems that treat every context change as a reason to abstain.

A confirmatory pair must bind both serialized proof-state snapshots to the exact replay receipt, share one `target_statement_sha256`, share one `goal_type_sha256` and environment digest, and identify the sole changed local declaration. The intervention receipt must include the accepted Lean state operation, the before/after declaration type digests, state digests, and the pre-registered precondition ID. Reject states that could not be replayed from the pinned source/toolchain or whose target/goal/environment binding drifts.

### What to score—and what not to infer

The primary outcome for this track is **strategy applicability and action selection at a proof state**, not whether the target theorem is true. If hiding a local fact leaves the goal unprovable from that state, this does not make the original theorem false; it means that a proof move depending on the hidden fact is unavailable in the intervened state. A subsequent proof of the same goal under the reduced context is valid useful evidence if it actually closes and passes the kernel, but failure or timeout is not a counterexample to the theorem.

Report (a) sensitivity to genuinely absent prerequisites, (b) stability on alpha-renamed/equivalent context presentations, (c) correct non-abstention after irrelevant facts are removed, (d) alternative-plan recovery and exact kernel-checked closure when achieved, and (e) action/call/token/time/cost by paired state. Keep these metrics separate from theorem-level transfer selectivity and exact-target proof utility. A system should fail if it either applies a premise-dependent move after its premise vanished or abstains on every altered context.

This design isolates a cleaner causal question than statement mutation alone: **does the system track the prerequisite in the actual proof state, or does it only react to surface similarity and mutation labels?** The schema, synthetic CI fixture, and a small Lean elaboration example for the `clear h` transition are now implemented. A replayable proof-state dataset, instrumented transition-receipt generator, independently adjudicated pairs, and model outcomes remain unimplemented. The existing statement-mutation suite remains necessary for testing formal-equivalence handling and statement fidelity.

## Fixed-theorem proof-state intervention contract

The first machine-readable contract for this track is `proof-state-intervention-suite.schema.json`, with a four-pair synthetic fixture at `fixtures/proof-state-intervention-suite.synthetic.json`. The fixture covers required-hypothesis removal, alpha-renaming, irrelevant-declaration removal, and reordering of independent declarations.

Every pair binds base and variant snapshots to the same target statement digest, goal-type digest, environment digest, source commit, Lean toolchain, and goal count. State snapshot IDs/hashes and replay receipts differ, and the local-context digest must change. The declared operation must match the intervention kind and apply to exactly one tracked local declaration. The case also binds two independent reviewer decisions and unique outcome-row IDs. The synthetic fixture explicitly uses `outcome_links.status: not_run`; no model outcomes are fabricated.

The dedicated PITB workflow checks the schema with the pinned Draft 2020-12 validator and then applies cross-field invariants that JSON Schema cannot express as relational equality: the same theorem/goal/environment across states, valid before/after declaration changes, precondition availability, expected action policy, reviewer agreement, unique joins, and family/component split isolation. It also injects nine negative controls. This provides a contract guard, not proof that a state was replayed: real pairs still require a Lean-side state capture/operation receipt, independently reviewed eligibility labels, and actual linked model outcomes.

### Action-selection metrics for the fixed-theorem track

Keep this track separate from theorem mutation sensitivity and exact-target proof utility. Primary metrics are:
- **Prerequisite-withdrawal sensitivity:** rate at which the system avoids the specified move or replans when its required local declaration is absent.
- **Presentation stability:** rate of preserving move eligibility after alpha-renaming or harmless reordering.
- **Irrelevant-edit specificity:** rate of preserving eligibility after removal of a reviewed irrelevant declaration.
- **Recovery rate:** rate of alternative plans that eventually produce a kernel-checked closure of the same goal.
- **State fidelity:** fraction of pairs with exact snapshot and replay receipts, same target/goal/environment bindings, and one audited intervention.

Proposed pilot gates are 100% state/operation receipt validity, zero theorem/goal/environment drift, at least 90% prerequisite-withdrawal sensitivity, at least 95% presentation stability, and at least 90% irrelevant-edit specificity. These are future adoption targets, not measured results; report component-clustered intervals and all unknown/excluded pairs. A system that abstains on every context mutation must fail the presentation and irrelevant-edit metrics.

## Relation to external work

T² shows why compilation is an insufficient proxy for semantic correctness by using dependent successor theorems ([ACL 2026 paper](https://aclanthology.org/2026.acl-industry.150/)). FormalRewardBench uses expert-curated proof-error mutations for proof evaluation ([paper](https://arxiv.org/abs/2605.10141)). A 2026 audit of Lean benchmarks documents how incorrect or incomplete specifications, vacuous theorems, missing hypotheses and unsound axioms can distort reported scores ([PMLR paper](https://proceedings.mlr.press/v306/ammanamanchi26a.html)). PITB's proposed distinction is to study *whether a typed strategy's declared preconditions causally govern transfer decisions and verified utility*; that is a research hypothesis and should not be presented as an established SOTA result or an unprecedented capability before a literature review and matched experiments.
