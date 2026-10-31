# PITB Lean context evidence v2: local-instance-aware proof states (2026-10-31)

**Status:** versioned Lean module, schema, independent verifier, context-manifest adapter and focused workflow committed. No Lean compilation or real proof-state capture is claimed in this authoring session.

## Why v2 is necessary

A Lean proof state is not only a local declaration list. Its metavariable declaration also carries a `LocalInstances` array used by typeclass synthesis. The kernel does not consume this registry directly—typeclass resolution is an elaboration service—but it can change which proof scripts and tactics elaborate. A context ablation that retains a local declaration but drops its local-instance registration is not the same experimental context.

The official API says `LocalInstance` stores the class name and concrete instance expression in `fvar`; `LocalInstances` is an array of these entries. The metavariable declaration stores its local instance registry. See [Lean.MetavarContext](https://api.cslib.io/docs/Lean/MetavarContext.html) and [Lean.Meta.Basic](https://lean-lang.org/doc/api/Lean/Meta/Basic.html). A filtered context must therefore project declarations and instance registry together. <Cite refs={["turn830286search0","turn830286search2"]}/>

V2 also records each declaration's binder information and auxiliary-declaration flag. In particular, `BinderInfo.instImplicit` and a registered `LocalInstance` are related but separate facts; PITB records both rather than inferring one from the other. <Cite refs={["turn114272search0","turn114272search1"]}/>

## Files

- Lean exporter: `lean-pilot/PITBCounterfactualPilot/ContextEvidenceV2.lean`
- Evidence schema: `lean-pilot/local-context-evidence-v2.schema.json`
- Independent verifier: `scripts/research/pitb_validate_lean_context_evidence_v2.py`
- Role-plan schema: `lean-pilot/lean-context-role-plan-v1.schema.json`
- Manifest schema: `lean-pilot/lean-context-manifest-v1.schema.json`
- Adapter: `scripts/research/pitb_lean_context_manifest_v1.py`
- Workflow: `.github/workflows/pitb-lean-context-evidence-v2.yml`

The exporter records ordered declarations, binder metadata, the metavariable's local-instance registry, proof support, typed dependency closure and assigned-value provenance. It rejects registry entries that are not direct local FVar expressions, are absent from the captured local context, or repeat a local FVarId.

## Context projection semantics

A support-lattice context is the pair:

`(retained LocalDecls, retained LocalInstances)`.

For selected mutable declarations S and fixed declarations F, the registry is projected by retaining exactly those captured local instances whose FVarId is in S ∪ F. The adapter emits the full declaration list and full registered-instance list; its `project_context` function returns a matched subset for runner use.

A registered local instance is **not** converted into an artificial type-dependency edge. It is separate elaboration state. A valid dependency closure does not imply that the search environment is identical unless its corresponding instance registry is projected too. The runner must bind each actual projected declaration+instance payload into its context receipt and budget identity before generating v3 support-landscape rows.

## Calibration

The calibration theorem adds an instance binder `[localDecidable : Decidable (0 = 0)]` to the earlier ordinary-let / opaque-have example. The CI assertion checks that the declaration appears in the local context with `BinderInfo.instImplicit`, that its FVarId appears exactly once in the `Decidable` registry, and that the adapter retains/removes the registry entry according to the role selection.

The synthetic verifier rejects unknown and duplicate local-instance FVarIds and a type record that disagrees with its corresponding declaration. The synthetic adapter asserts that projecting out the mutable instance also removes it from the registry while leaving fixed declarations intact.

## Interpretation limits

A local instance is an elaboration resource, not a kernel axiom. Its presence can change tactic success and proof-search behaviour even when the final proof term does not directly mention it. PITB therefore measures operational behavior under a frozen elaboration context; it does not claim every registered instance is logically necessary.

The target digest is still a pinned-toolchain pretty-printer digest, not a universal canonical Expr hash. Build metadata binds the exact toolchain file, Lake package file, root module and exporter source. Structured Expr serialization and authenticated CI/runner attestation remain future work for portable semantic identity and independent provenance.
