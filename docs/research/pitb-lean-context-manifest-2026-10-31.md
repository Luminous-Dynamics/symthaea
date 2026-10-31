# PITB Lean-to-support-lattice adapter (2026-10-31)

**Status:** adapter, schemas and synthetic negative controls committed. No real Lean export or actual role assignment has been generated yet.

## Why this is needed

The Lean exporter records the actual local context. The v3 support-landscape analyzer expects a mutable declaration universe, a fixed context, a full telescope order and a prerequisite graph. Manually transcribing those edges invites mistakes and permits the optimizer to choose its own graph.

This adapter joins the two without trusting the optimizer:

1. independently verifies the complete Lean evidence envelope using `pitb_validate_lean_context_evidence_v1.py`;
2. recomputes the local-context and exact target-statement digests;
3. requires a separately authored role plan assigning **every** captured FVarId exactly once to fixed or mutable;
4. verifies that role plan's pinned context and target digests match the Lean evidence;
5. derives the support-lattice dependency graph exclusively from declaration type references plus visible transparent-let value references;
6. emits the hidden nondependent have-value dependencies in a separate provenance graph that is **not** used for context-validity checks.

## Contracts

- Adapter: `scripts/research/pitb_lean_context_manifest_v0.py`
- Role-plan schema: `lean-pilot/lean-context-role-plan-v0.schema.json`
- Derived manifest schema: `lean-pilot/lean-context-manifest-v0.schema.json`

The role plan is intentionally not predicted by HDC/LTC or the context ranker. It is an explicit reviewed experimental-design decision, bound to the exact Lean context and target. The adapter rejects missing or unknown FVarIds, overlapping roles, stale context/target digests, unordered IDs, and incomplete declarations.

The derived manifest contains:
- `declaration_dependencies`: typing prerequisites for proof-state reconstruction;
- `assigned_value_provenance_dependencies`: all assigned-value edges, including nondependent have values;
- `declaration_order`: the exact Lean telescope order;
- exact source-evidence, role-plan, context, target, and toolchain digests;
- the target's typed closure and observed proof-term support closures.

The context validity relation remains: every retained declaration must have all its `declaration_dependencies` present in the selected mutable set plus fixed context. Hidden nondependent have-value provenance is never silently promoted into this relation.

Lean's API explicitly warns that a nondependent local declaration's stored value is not necessarily type-correct; such declarations should normally be treated like ordinary hypotheses unless the metaprogram deliberately opts into their hidden values. This is why the adapter never puts `nondep_value_fvar_ids` in the context-validity graph. The exported assigned-value graph is forensic/provenance data, not a proof-context prerequisite. [Lean.LocalContext API](https://lean-lang.org/doc/api/Lean/LocalContext.html).

**Digest qualification:** `target_statement_sha256` is currently the SHA-256 of the pinned toolchain's pretty-printed target string, not a universal canonical hash of Lean's elaborated Expr. It is safe as a within-toolchain receipt binding when accompanied by exact source, toolchain, and evidence digests; it must not be treated as cross-version semantic identity. A structured Expr serialization or independent kernel-level statement identity remains future work.

## Synthetic controls

The adapter self-test constructs an ordinary-let to nondependent-have chain. It confirms that:
- the ordinary let's prerequisite remains a typing dependency;
- the hidden have-value edge is retained as provenance only;
- the same context and target digests bind the role plan;
- the role partition covers every declaration exactly once;
- a fixed declaration that depends on a removed mutable declaration invalidates that subset.

Negative controls reject stale context/target pins, missing or unknown declarations, overlapping roles, and unsorted role IDs.

These are contract tests, not Lean measurements. The real integration begins only after the pinned Lean exporter builds and emits a receipt from an exact CI run.
