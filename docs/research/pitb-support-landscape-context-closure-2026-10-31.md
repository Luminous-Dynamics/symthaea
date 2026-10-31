# PITB support-landscape context-closure amendment (2026-10-31)

This amendment clarifies the v1 support-landscape validity contract.

The declaration order and prerequisite graph cover the union of mutable and fixed declarations, not only the mutable subset. For a requested mutable subset S and retained fixed declarations F, the reconstructed context contains P = S ∪ F. It is well formed only if every retained declaration remains dependency-closed:

\[
\operatorname{valid}(S) \iff \forall x \in S \cup F,\ D(x) \subseteq S \cup F
\]

The full ordered telescope must contain every declaration in S ∪ F exactly once, and every direct prerequisite must precede the declaration that refers to it. This catches a case where a fixed declaration depends on a mutable declaration: removing that mutable prerequisite invalidates the reconstructed context even though the fixed declaration itself is not ablated.

The v1 analyzer now checks this closure for all retained declarations. Its synthetic control uses a fixed declaration whose type depends on mutable base; the context retaining the fixed declaration but removing base is classified as invalid_context with no prover attempt, rather than a mathematical non-solve or evidence of unnecessary support.

This is still a structural dependency manifest. In real Lean data, it must be generated from the exact elaborated local context (including declaration types and available let-values), then bound to the exact goal/environment/context receipts. The synthetic control is not a Lean result.
