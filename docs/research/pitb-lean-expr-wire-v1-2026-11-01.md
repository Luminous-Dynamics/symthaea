# PITB research note: structural Lean expression identity v1

**Date:** 2026-11-01  
**Status:** versioned wire contract, synthetic fixture, and Rust verifier authored; JSON Schema and the synthetic fixture validated locally. The Rust verifier has not been compiled in this environment, and no Lean exporter has emitted a v1 capture. This is not a kernel receipt or a benchmark result.

## Why this is the next engineering frontier

PITB's current ContextEvidenceV2 exporter captures the right categories of evidence—ordered local declarations, binder metadata, auxiliary-declaration flags, assigned-value provenance, and LocalInstances—but its expressions are primarily represented as pretty-printed text. That is useful for inspection and same-toolchain debugging. It is weaker as a durable identity layer: pretty printing is a presentation function, not Lean's expression representation, and its output depends on formatting and name-resolution context.

Lean's Expr is the expression representation used by the elaborator and kernel. Its constructors include bound variables, local free variables, metavariables, sorts, constants with universe levels, applications, lambdas, dependent function types, let expressions, metadata, literals, and structure projections. Bound variables use De Bruijn indices; free variables refer into a LocalContext; kernel-facing expressions must not contain expression metavariables. Lean's Level also has universe metavariables, which must be resolved before evidence is eligible. The v1 contract therefore serializes expression structure instead of hashing ppExpr output.

## Three identities, not one overloaded hash

The wire format emits four domain-separated digests:

1. **Environment digest:** exact source/task identity, Lean binary and runtime-artifact manifest, imported OLean-closure manifest, toolchain file, Lake configuration and manifest, root module, exporter source, and clean-checkout assertion.
2. **Target expression digest:** exact structural expression transcript, scoped to the environment digest.
3. **Context digest:** ordered local declarations, binder metadata, value-role separation, and the ordered LocalInstances registry, scoped to the environment digest.
4. **Proof-state digest:** environment + target-expression digest + context digest.

This is an environment-scoped structural identity, not a universal theorem identifier. It intentionally does not normalize definitional equality: two definitionally equal expressions may have different structural hashes. A successful proof still has to be checked by Lean's kernel against the exact target in the pinned project. The hash binds the structure being replayed; it does not itself prove the theorem, prove source authenticity, or replace an independent checker.

The environment envelope requires two artifact manifests beyond a version string and an executable hash. The Lean runtime manifest is carried as explicit path-sorted artifact rows (executable/shared library plus SHA-256); the imported OLean manifest is carried as explicit module/path-sorted rows for the exact transitive imported module artifacts. The independent Rust verifier re-hashes those rows using distinct domain-separated transcripts and rejects missing executables, duplicate paths/modules, unsorted rows, path traversal, and hash mismatches. A Lake manifest pins dependency intent, but does not prove which compiled OLean bytes the process actually loaded. The capture harness must compute these manifests from the resolved build/import environment; it must not substitute package version labels or assume a cached artifact matches its source pin. Until both manifests are present and verified, a capture is not eligible for environment-identity qualification.

## Wire representation

The JSON Schema is lean-pilot/lean-expr-wire-v1.schema.json; the checked-in synthetic example is lean-pilot/fixtures/lean-expr-wire-v1.synthetic.json; the independent validator is the Rust binary verify_lean_expr_wire_v1.

| Wire tag | Lean form | Encoded payload |
|---|---|---|
| bvar | .bvar | De Bruijn index |
| fvar | .fvar | Ordered local-context index |
| sort | .sort | Structural universe level |
| const | .const | Recursive Lean name and ordered universe levels |
| app | .app | Function and argument |
| lam | .lam | Binder info, domain, body |
| forall_e | .forallE | Binder info, domain, body |
| let_e | .letE | Nondependent flag, domain, value, body |
| literal_nat / literal_string | .lit | Exact literal; natural numbers are canonical decimal strings |
| proj | .proj | Structure name, field index, structure expression |

Lean names are encoded by their recursive constructors (anonymous, str, num) rather than parsing a dotted pretty-printed name. Universe levels preserve zero, succ, max, imax, and named param constructors. Unresolved expression metavariables and universe metavariables are rejected. Expression metadata (mdata) is erased before encoding because Lean documents it as ignored by definitional equality; a capture shim must not smuggle metadata into a supposedly kernel-semantic digest.

FVarId values are generated identifiers, not stable local identities. The Lean shim must resolve each one against the captured LocalContext and emit a unique zero-based local_index. The ID itself is omitted from the stable wire. This makes local references unambiguous without making the digest depend on volatile name-generator state. The ordered local user names remain in the context transcript because tactics and elaboration tools can address hypotheses by those names.

Declaration values retain two roles:

- transparent_value is a typing-visible dependency for ordinary transparent lets.
- nondep_value_provenance records the assignment behind a nondependent have, but is not automatically a typing prerequisite.

The synthetic fixture includes both an ordinary transparent let referencing an earlier local and a nondependent have whose assigned value references an earlier proof. The Rust verifier checks their separate edge counts (three typing edges versus one provenance edge) and rejects a declaration that mixes those two fields.

The validator rejects a declaration that mixes those two fields. It derives separate direct-edge counts for typing dependencies and assigned-value provenance. It also rejects self/later telescope references, out-of-context FVars, out-of-scope De Bruijn indices, duplicate local-instance indices, unsafe source paths, unresolved metavariables, and digest mismatches.

## Canonical transcript and hash domains

The JSON document is the interchange envelope; hashes are not computed over arbitrary JSON text. The Rust verifier reconstructs a typed binary transcript with:

- a fixed serializer/version prefix and distinct ENVIRONMENT, TARGET, CONTEXT, and PROOF-STATE domains;
- single-byte constructor tags;
- unsigned 64-bit big-endian lengths and indices;
- UTF-8 strings preceded by their byte length;
- exact array order (including local declarations, universe arguments, and local instances);
- natural-number and Name.num components encoded as canonical decimal strings.

This avoids relying on JSON object-key ordering or floating-point JSON number semantics. The wire schema caps indices to the interoperable integer range; arbitrary Lean naturals are strings and do not pass through floating point.

Conceptually:

~~~text
E = SHA256(prefix || ENVIRONMENT || encode(environment))
T = SHA256(prefix || TARGET || E || encode(target_expr))
C = SHA256(prefix || CONTEXT || E || encode(local_context))
S = SHA256(prefix || PROOF_STATE || E || T || C)
~~~

Every digest is recomputed by the Rust verifier; supplied digest strings are claims, not trusted input. A serializer change must use a new serializer_id and schema version. No cross-version or cross-environment digest comparison is valid without a separate equivalence study.

## Required negative controls

The Rust self-test and focused workflow cover:

- a target FVar outside the captured context;
- an unbound De Bruijn index;
- a declaration referring to itself or a later telescope slot;
- removal of a local instance (the context digest must change);
- Lean-version drift (the environment and all dependent identity digests must change);
- expression and universe metavariables;
- unknown top-level fields;
- digest recomputation from the fixed synthetic fixture.

The next Lean-shim golden tests must additionally show that alpha-renaming a lambda/forall binder leaves its structural expression digest unchanged, while changing binder info, universe parameters, a projection index, local-context order, a declaration value role, or a registered local instance changes the appropriate digest. Inserting or removing mdata must not change the expression digest after the documented erasure.

## Immediate integration sequence

1. Compile the Rust verifier and run its self-test from the isolated PITB validation crate; retain the exact output and artifact hashes.
2. Implement a version-local Lean capture shim beginning with the existing 4.34.1 pilot. It must recursively traverse actual Expr, Level, and Name constructors, resolve FVars to LocalContext indices, preserve LocalInstances, reject unresolved metavariables, and emit source/build metadata.
3. Add independent compatibility shims for the SorryDB smoke toolchains (4.24.0, 4.26.0, and 4.26.0-rc2). Do not run a 4.34.1-compiled exporter by changing a version string.
4. Compare v1 structural target/context digests with the existing v2 pretty-text evidence as a diagnostic cross-check, not as a substitute for kernel checking.
5. Capture the actual clap-lean sorry state, verify the real Fact (Nat.Prime p) registry entry through the Lean API, and then run the first exhaustive two-to-four-declaration support lattice.
6. Promote v1 identity into SorryDB attempt receipts only after independent replay confirms the target declaration type, sorry-only patch scope, axioms, and clean project build.

## Benchmark consequence

This work does not directly increase proof pass rate. It creates a stronger experimental substrate for deciding whether PITB's context projection, typed insight transfer, and repair policies actually improve proof completion. The first measurable claim remains causal: under the same task, Lean environment, model, checker, and total budget, does an evidence-aware arm increase exact-target verified completions or lower cost per verified success?

The SorryDB_2601 cohort remains a separate real-project benchmark from the 372-family research-math transfer split. Its 1,000 tasks span 78 projects and 14 Lean toolchain tags. Report preflight readiness separately from proof-search success, preserve every task/arm denominator, cluster uncertainty by repository, and reserve a later temporal cohort for forward validation. Do not compare raw task percentages from different datasets or different attempt budgets.

## Sources

- Lean Expr representation, constructors, De Bruijn indices, FVars, metavariable restrictions, and metadata behavior: https://lean-lang.org/doc/api/Lean/Expr.html and https://github.com/leanprover/lean4/blob/master/src/Lean/Expr.lean
- Lean universe-level constructors and metavariables: https://lean-lang.org/doc/api/Lean/Level.html
- Lean recursive Name representation: https://github.com/leanprover/lean4/blob/master/src/Init/Prelude.lean
- PITB current v2 evidence contract: lean-pilot/PITBCounterfactualPilot/ContextEvidenceV2.lean
- SorryDB_2601 task and source pins: sorrydb-track/smoke-cohort-candidates.v0.json
