# PITB Lean context evidence layers (2026-10-31)

**Status:** source, schema, and exact-head workflow added. The local authoring environment has no Lean/Lake executable, so this module is not represented as compiled. The focused workflow must run against its exact proposed SHA before the export contract is called validated.

## The distinction

Lean LocalDecl values include ordered identity, type, and sometimes an assigned value. The API distinguishes ordinary transparent let declarations from nondependent have-bound local declarations. In particular, value? false returns ordinary visible let values, while value? true also exposes values normally hidden for nondependent declarations. See the official [Lean.LocalContext API](https://lean-lang.org/doc/api/Lean/LocalContext.html).

A context-minimization algorithm must not conflate these relations:

1. **Typing dependencies:** free variables in declaration types plus dependencies of transparent let values. These are the prerequisites used by the exporter’s typed-context closure.
2. **Assigned-value provenance:** all dependencies reachable through assigned values, including values hidden behind nondependent have declarations. This is useful for provenance and replay analysis, but it is not the same as a local-context well-formedness prerequisite.
3. **Proof-term support:** the free variables and global constants actually referenced by the elaborated proof term. This is construction evidence, not a theorem of logical necessity.

If a proof uses a have-bound local hypothesis, the proof term can refer to that local declaration without directly referring to the locals used to construct its hidden assigned value. Treating those hidden locals as typing prerequisites for the reconstructed context can overstate support. Ignoring assigned-value provenance entirely loses traceability. PITB now reports both closures.

## Implementation

- Lean module: lean-pilot/PITBCounterfactualPilot/ContextEvidence.lean
- Schema: lean-pilot/local-context-evidence-v1.schema.json
- Workflow: .github/workflows/pitb-lean-context-evidence.yml

The exporter records every local declaration in telescope order, its FVarId, type references, ordinary let-value references, nondependent assigned-value references, relevant global constants, and implementation-detail marker. It rejects unresolved metavariables and any free-variable reference that cannot be resolved in the captured local context.

The calibration theorem creates this chain:
- h : 0 = 0
- transparentValue : 0 = 0 := h (ordinary let)
- opaqueValue : 0 = 0 := transparentValue (have-bound / nondependent value)

The proof term uses opaqueValue. The expected typed dependency closure contains opaqueValue but does not follow the opaque assigned value through transparentValue to h. The separately reported assigned-value provenance closure does follow that chain. The CI step checks these relationships directly from Lean-emitted JSON and the strict schema.

The global-constant collector explicitly includes the structure name on projection expressions, which is not represented as an ordinary constant child node.

## Interpretation limits

- A transparent-let typing dependency is a reconstruction constraint, not proof of logical necessity across all alternative formalizations.
- A direct free-variable reference shows that a particular proof term uses that local declaration. It does not prove that every proof of the theorem requires it.
- Assigned-value provenance does not by itself say the underlying hypothesis is needed by a solver after context reconstruction.
- Pretty-printed expressions are inspection fields, not a canonical semantic encoding. The artifact is tied to exact source, Lean toolchain, elaboration process, and build log; a future schema should add a structured expression representation if consumers need cross-version semantic comparison.
- This calibration does not establish model transfer, general support minimization, or an empirical reliability landscape.

## Research implication

Support should become a **typed multi-relation graph**, not one undifferentiated dependency graph. Keep edges labelled type_dependency, transparent_value_dependency, opaque_assigned_value_provenance, proof_term_direct_use, and global_constant_reference. Only the first two govern context reconstruction in this exporter. The others support explanation, provenance, and model analysis.

Next: build the module under pinned Lean 4.34.1, inspect its exact artifact, then connect the emitted context rows to the v3 support-landscape trial schema through a separately reviewed host-side adapter. No receipt digest or pretty-printed name should be accepted as a substitute for validating the actual Lean-generated evidence bytes.
