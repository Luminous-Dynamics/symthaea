# Lean dependency extraction contract for MATH-COMPREHENSION-001

**Status:** Source-grounded implementation contract, v0.1  
**Issue:** [#13293](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**PR:** [#13294](https://github.com/Luminous-Dynamics/symthaea/pull/13294)  
**Reference implementation source:** Lean 4 commit `a9b5b8a25c4b932be8029426c2abd34685b078f0)

## 1. Research finding

Do not build the dependency extractor by parsing tactic scripts or matching declaration names in source text.

Lean already provides the primitives needed to extract constant references from elaborated declarations:

- `Lean.Expr.getUsedConstants` traverses expressions and returns referenced constants.
- `Lean.ConstantInfo.getUsedConstantsAsSet` gathers references from the declaration type and value, and includes special handling for inductives, constructors and recursors.
- `Lake.Check.runForUsedConsts` follows references in the declaration type and value, then explicitly follows inductive constructors, parent inductives and recursor rule constructors / right-hand sides.

These implementations are pinned here:
- [`Lean/Util/FoldConsts.lean`, lines 51–75](https://github.com/leanprover/lean4/blob/a9b5b8a25c4b932be8029426c2abd34685b078f0/src/Lean/Util/FoldConsts.lean)
- [`Lake/Check/Util.lean`, lines 14–30](https://github.com/leanprover/lean4/blob/a9b5b8a25c4b932be8029426c2abd34685b078f0/src/lake/Lake/Check/Util.lean)

Relevant source facts:
- `Expr.getUsedConstants` is implemented by folding constants in the elaborated expression.
- The fold traverses `forallE`, `lam`, `mdata`, `letE`, `app`, projection and constant nodes.
- `ConstantInfo.getUsedConstantsAsSet` includes type references and, where available, value references using `value? (allowOpaque := true)`.
- For inductive declarations, constructor names are included; constructors link back to their inductive; recursors include their generated declarations.
- `Lake.Check.runForUsedConsts` additionally traverses recursor rules and the constants in their right-hand sides.

This provides a trustworthy source-grounded starting point. It does **not** establish that every referenced constant is a mathematically essential lemma. Dependency extraction and mathematical role interpretation remain separate tasks.

## 2. Required graph semantics

Preserve separate edge classes rather than flattening all references into one undifferentiated relation:

| Edge | Evidence origin | Interpretation |
|---|---|---|
| `type_dependency` | Constant references in elaborated declaration type | Used to state the proposition or its assumptions |
| `proof_dependency` | Constant references in the elaborated value / proof term | Referenced by the formal proof object |
| `inductive_constructor` | Inductive metadata | Constructor belongs to the inductive type |
| `constructor_parent` | Constructor metadata | Constructor belongs to its parent inductive |
| `recursor_rule` | Recursor metadata | Recursor rule uses a constructor and right-hand-side expression |
| `imports_module` | Lake / module metadata | Declaration's environment depends on an imported module |
| `source_declared_at` | Source map / declaration range | Maps the elaborated declaration to its source location |

Do not infer a proof explanation from edge presence alone. A dependency can be an implementation detail, a typeclass instance, a coercion, a recursor, or a substantive mathematical lemma.

## 3. Extraction procedure

For each selected declaration:

1. Resolve the fully qualified declaration name in the pinned Lean environment.
2. Fetch its `ConstantInfo` from that environment.
3. Serialize the declaration kind, name, universe parameters, elaborated type, source range and environment identity.
4. Extract type constants from `ConstantInfo.type.getUsedConstants`.
5. Extract value constants when present using `value? (allowOpaque := true)`; preserve `none` as an explicit absence rather than an empty successful proof.
6. Add special inductive / constructor / recursor relations using the same semantics as `Lake.Check.runForUsedConsts`.
7. Traverse the closure with a visited set keyed by fully qualified `Name`; preserve cycles as graph edges rather than recursing forever.
8. Preserve imported and out-of-repository dependencies. Do not silently drop them.
9. Sort emitted nodes and edges canonically before digesting.
10. Bind every graph receipt to the exact Lean revision, Mathlib revision, Lake manifest, module closure and target declaration's elaborated type digest.

Capture universe parameters separately. `Expr.getUsedConstants` gives constant names from expressions; it is not by itself a complete serialization of universe-level dependencies, source spans, attributes, axioms or semantic theorem roles.

## 4. Two representations are required

### A. Formal dependency graph

Machine-extracted from Lean's elaborated environment. It answers:

- What constants occur in this declaration's type?
- What constants occur in its value?
- Which inductive constructors and recursor rules are involved?
- Which modules and toolchain artifacts define these objects?

This layer should be deterministic and independently reproducible.

### B. Mathematical role graph

A separate reviewed annotation layer. It answers:

- Which lemma supplies the key estimate?
- Which hypothesis is used in which direction?
- Which definition encodes the mathematical object?
- Which construction performs the critical transformation?
- Which imported result is merely infrastructure rather than substantive mathematics?

Role labels require proof inspection and mathematical review. They must not be inferred from lexical or HDC similarity.

## 5. Required adversarial tests

1. **Type-only dependency:** a constant appears in the theorem type but not in the proof value; it must be marked as a type dependency.
2. **Proof-only dependency:** a helper lemma appears in the proof value; it must be marked as a proof dependency.
3. **Opaque dependency:** an opaque declaration's value is included where Lean exposes it under `allowOpaque := true`; otherwise record the explicit unavailable status.
4. **Inductive closure:** constructors, parent inductive and recursor rules appear with correct edge types.
5. **Cycle handling:** mutually recursive declarations terminate extraction and preserve all graph edges.
6. **Namespace shadowing:** two declarations with similar short names remain distinct by fully qualified `Name`.
7. **Changed theorem statement:** changing a binder, hypothesis or conclusion changes the elaborated-type identity digest.
8. **Changed proof value:** changing a proof dependency changes the proof artifact / dependency digest even when the theorem type is unchanged.
9. **Imported dependency:** an imported declaration remains represented with module provenance.
10. **Axiom / sorry path:** an axiom is recorded as an axiom; it must never be relabelled as an ordinary proved lemma.
11. **Universe change:** a universe-parameter change must not be lost during serialization.
12. **Source-map ambiguity:** missing or ambiguous source spans must be recorded as unknown, not fabricated.

## 6. Minimum receipt fields

- Lean repository commit and exact Lean version;
- Mathlib revision and Lake manifest digest, when applicable;
- target declaration's fully qualified name and declaration kind;
- elaborated type serialization and SHA-256;
- proof value availability, serialization digest and SHA-256;
- universe parameters;
- source module, source range and source blob digest;
- direct type dependencies and direct proof dependencies;
- inductive / constructor / recursor metadata;
- transitive closure and closure digest;
- imported dependency classification;
- axiom / opaque / admission profile;
- extractor version and raw-output digest.

## 7. Pilot integration

Apply this extraction procedure to the six source-pinned Lean-core pilot tasks in `MATH_PROOF_COMPREHENSION_PILOT_V1.json`.

Expected result:
- retain the current manually annotated source-expression graph as a comparison baseline;
- generate a separate machine-extracted graph from the elaborated Lean environment;
- diff the two graphs and adjudicate missing or extra edges;
- do not promote the current seed to gold data until both extraction and expert role annotations are independently reviewed.

Only after this smoke test should we add 3–5 dependency-rich Mathlib theorems. The RH dependency slice is a later stage.

## 8. Claim boundary

- `getUsedConstants` reference extraction is not a natural-language explanation.
- A dependency edge is not proof that a lemma is mathematically essential.
- A source-level tactic token is not a formal dependency.
- A formal proof dependency graph is not a theorem-to-paper correspondence.
- A successful extraction is not a Lean kernel proof check.
- HDC/CfC retrieval is not proof authority.

The extractor should reuse Lean's own environment and dependency conventions, but the explanation and reviewer layers remain explicit and independently testable.

## 9. Toolchain identities are cohort-specific

A source-pin audit on 2026-11-23 found that the two pilot cohorts use different Lean revisions and must not share one environment identity:

- **Lean-core cohort:** exact source snapshot `leanprover/lean4@a9b5b8a25c4b932be8029426c2abd34685b078f0` (commit date 2026-11-22 18:20 UTC). Its repository `lean-toolchain` file says `build/release/stage1`, so reproducing this cohort requires building that exact Lean source snapshot; it is not a released toolchain name.
- **Mathlib cohort:** exact Mathlib snapshot `leanprover-community/mathlib4@6ebcce37335bcb6bca64596e5d57bb3035648030`. Its own `lean-toolchain` file says `leanprover/lean4:v4.37.0-rc1`, and that Lean tag resolves to commit `0b77e12952ad39077d80d8bd6b66d601f225c321` (2026-11-18 11:12 UTC).

The revisions `a9b5b8a…` and `0b77e12…` are not interchangeable: GitHub's compare endpoint reports a diverged history (87 commits ahead / 1 behind for the compared direction). The Mathlib seed's former `lean_core_revision: a9b5b8a…` value was therefore incorrect for that snapshot. It is corrected in `MATHLIB_PROOF_COMPREHENSION_PILOT_V1.json`, required by its schema, and asserted in the focused workflow.

The core pilot remains pinned to `a9b5b8a…`; the Mathlib pilot is now pinned to the actual toolchain `v4.37.0-rc1` / commit `0b77e12…`. This cohort split is part of the dataset identity, not a cosmetic metadata correction.

For the two revisions inspected, the source blob for `Lean/Util/FoldConsts.lean` was identical (`5429906b…`) and the source blob for `Lake/Check/Util.lean` was identical (`56ade87a…`). The `Lean/Declaration.lean` and `Lean/Environment.lean` blobs differed, so API inspection for each cohort must still be performed against its own pinned revision. The relevant APIs in the Mathlib cohort's exact Lean pin were verified:

- `Expr.getUsedConstants` folds over elaborated `Expr` constants; the projection case adds the projection's type name as well as recursively visiting its body.
- `ConstantInfo.value? (allowOpaque := true)` exposes values for definitions, theorems and opaque declarations; axioms, inductives, constructors and recursors do not receive a generic value from this accessor.
- `ConstantInfo.getUsedConstantsAsSet` unions type and value names, so it is useful as a closure cross-check but not as the source of separately typed `type_dependency` and `proof_dependency` edges.
- `Environment.find?` resolves fully qualified `Name` values in the imported environment. Preserve the full `Name`; do not key by short names.
- `Lake.Check.runForUsedConsts` visits type references, the declaration's own name, value references, and additional inductive/constructor/recursor relations. Its traversal is a useful compatibility baseline, but because it emits a flat stream of names it does not replace our typed-edge serializer.

A source-level API match is not a built or executed extractor. The Mathlib `lake build`, native extraction run, and byte-identical replay remain pending.