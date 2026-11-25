# Independent Lean axiom-dependency audit (research protocol)

**Status:** bounded source implementation and pinned two-module reproduction committed; exact-head execution pending. This is not a verified replacement for Lean's kernel or a repository-wide trust proof.

## 1. Why this exists

A proof term accepted by Lean is only as trustworthy as the full trusted computing base and its axiom dependencies. `#print axioms` is useful, but it must not be promoted to the sole authority for trust closure without testing the exact Lean version and import pattern.

Lean issue [#15226](https://github.com/leanprover/lean4/issues/15226) documents an under-reporting case in `Lean.collectAxioms` / `#print axioms` involving imported inductives and a shared exported-axiom cache. The report gives a fresh dependency walk as a workaround: disable exporting, walk the current environment from each root, and use only a visited set rather than memoizing partially computed per-constant closures. The affected versions described in the issue include Lean 4.30 through 4.36 nightly. The Symthaea pilot pins Lean `4.37.0-rc1`; whether the same false-negative reproduces in that exact toolchain is an empirical question for this workflow, not an assumption.

## 2. Minimal imported-inductive reproduction

The current source pair is:

- [Module A](../../scripts/research/lean-proof-extractor/axiom-audit-imported-inductive/A.lean): defines `pick := Classical.choice ...` and `S9`, whose field type depends on `pick`; records native axiom output at the defining module.
- [Module B](../../scripts/research/lean-proof-extractor/axiom-audit-imported-inductive/B.lean): imports A and the `And.comm` proof-action module, records native `#print axioms` output, and runs a separate fresh per-root dependency walk.

The audited roots are intentionally divided into three dependency classes:

| Roots | Expected fresh dependency set |
|---|---|
| `S9`, `S9.mk`, `usesS9` | exactly `{Classical.choice}` |
| `auditDirectAxiom`, `auditTransitiveAxiom`, `usesTransitiveAxiom` | exactly `{auditDirectAxiom}` |
| `Even`, `Odd`, `Even.succ`, `Odd.succ`, `And.comm`, `and_comm_split_iff`, `and_comm_forward`, `and_comm_reverse`, `and_comm_plan_replay` | empty |

The same exact expected set is required for each root. An unresolved constant is retained by name and makes the exact-set assertion fail. It must never silently become an empty dependency result.

## 3. Fresh traversal contract

For every root, independently:

1. Obtain the environment with `env.setExporting false`.
2. Start a new DFS using `env.checked.get.find?` for constant information.
3. Use a per-root visited-name set solely to break cycles. Do not cache partial dependency sets per constant and do not read the exported-axiom result extension.
4. Traverse type and value expressions for axioms, definitions, theorems and opaque constants; traverse types for constructors and recursors; for inductives, traverse the inductive type and every constructor; quotient primitives terminate the walk.
5. Record direct axioms by declaration name. Preserve unresolved names and fail closed.
6. Sort the final set deterministically before comparison and receipt generation.

The current implementation lives inside Module B for easy reproduction. It is deliberately small enough to review. It is not yet a separately audited library or independently verified algorithm.

## 4. CI receipt and evidence separation

The focused workflow pins both source hashes, the Lean toolchain, Lean core revision, Mathlib revision, and the compiled `A.olean` / proof-module `.olean` digests. It uploads the native diagnostic output and the independent-walk receipt separately.

Native `#print axioms` is recorded as a comparison channel only. A native “does not depend on any axioms” line is not by itself accepted as evidence of a complete trust closure.

The expected receipt has fifteen roots: three roots with exactly `Classical.choice`, three roots with exactly the custom `auditDirectAxiom`, and nine roots with an empty set. Each root is audited twice in opposite root orders; the log must contain identical exact sets. The run must fail if any root is unresolved, contains an additional axiom, or omits the expected one.

## 5. Acceptance targets

These are prospective targets, not results:

- 100% detection of all seeded direct and transitive dependencies.
- Zero false negatives on imported-inductive, constructor-cycle and mutual-inductive controls.
- Exact deterministic closure under permutations of audit-root order.
- Explicit treatment of `sorryAx`, custom axioms, `Classical.choice`, unsafe/partial declarations, opaque constants, quotient primitives and allowlisted foundation axioms.
- Every unresolved or inaccessible declaration fails closed.
- Independent agreement with at least one separately implemented oracle on the frozen corpus.

## 6. What is not claimed

The prototype does not prove the completeness of its own traversal, replace kernel checking, or establish repository-wide soundness. Its current control covers one imported inductive/constructor path, one downstream theorem, `And.comm`, and four local proof-action declarations. Mutual/nested inductives, all imported-module serialization edge cases, custom axiom allowlists, unsafe runtime effects, plugins, external oracles, and whole-project transitive trust closure need additional cases and review.

The research question is not “can we print an empty axiom list?” It is: **can an independently reviewable, deterministic audit trace every trust dependency from a theorem root through imported declarations, and can seeded omissions prove that the auditor fails closed?**