# ETH-027 — semantic metamorphic law catalogue v1

**Status:** normative research catalogue for test generation. Each law is scoped to an output contract and preconditions. A passing finite metamorphic test is evidence for that fixture, not a proof for all inputs.

## 1. Semantic contracts are different observers

Let a source problem be F(X,H), with visible variables X and hidden variables H. The system has several semantic observers:

- SAT observes whether any satisfying assignment exists.
- #SAT observes the number of full satisfying assignments.
- WMC observes the exact sum over full satisfying assignments and a full literal-weight map.
- PMC observes the set of visible assignments with at least one hidden witness.
- PWMC observes the exact weight sum over that distinct visible support and a visible-only weight map.
- AMC observes the full-model algebraic sum in the declared coefficient domain.

A transformation is not generically equivalent. It carries a relation and a precondition set. The same transformation can preserve one observer and intentionally change another.

## 2. Metamorphic laws

### L1 — Clause order and duplicate idempotence

Permuting clauses, permuting literals inside a clause, repeating a literal inside a clause, or repeating an identical clause preserves the Boolean CNF relation. Therefore all five exact counting contracts preserve their result when the declared variables, partition, domain and weights remain unchanged.

### L2 — Consistent variable renaming/permutation

A bijection of variable identities preserves all contracts when applied consistently to source literals, variable universe, visible/hidden partition and weight-map keys. The certificate must bind the bijection. Reordering visible variables may reorder serialized assignment keys, so comparison must use the explicit variable map rather than raw key strings.

### L3 — Add an unconstrained hidden variable

Let H' = H ∪ {z}, where z does not occur in F.

- #SAT doubles.
- WMC is multiplied by w_z(false)+w_z(true).
- PMC and PWMC are unchanged because existential projection absorbs the extra hidden witness dimension.
- Full AMC is multiplied by the corresponding coefficient-domain sum of the two z literal weights.

This is a particularly useful projection metamorphism: witness counts change, support does not.

### L4 — Add an unconstrained visible variable

Let X' = X ∪ {z}, with z absent from F.

- #SAT doubles.
- WMC is multiplied by w_z(false)+w_z(true).
- PMC doubles because each supported visible assignment extends to two visible assignments.
- PWMC is multiplied by the visible weight sum w_z(false)+w_z(true).

This is not an invariance relation. It is an exact predicted-delta relation.

### L5 — Change the projection boundary

Moving a variable from H to X changes the question. A variable previously existentially eliminated becomes part of the output key and may multiply projected support and PWMC. The partition must therefore be included in the source contract digest, not inferred from a compiled representation.

### L6 — Add a redundant/subsumed clause

Adding a clause already implied by F preserves the Boolean relation and therefore every contract over the same universe/partition/weights. The obligation is formula equivalence, not merely equisatisfiability.

### L7 — Unique definitional extension

If a fresh auxiliary z is constrained by a definition z ↔ g(X), each assignment to X has exactly one z extension. Full #SAT and projected support over original variables are preserved. Weighted preservation over an expanded full universe requires neutral auxiliary weights or an explicit weight transport; arbitrary auxiliary weights invalidate a generic WMC-preservation claim.

### L8 — Existential witness duplication

Adding a fresh unconstrained hidden variable duplicates each hidden witness but leaves visible support invariant. This is the same algebraic effect as L3, but should be generated as an adversarial projection test: a naive hidden-witness sum must fail by the exact factor of two.

### L9 — Shannon partition

For a selected variable x:

F = (x AND F[x=true]) OR (NOT x AND F[x=false]).

The two branches are disjoint because they assign opposite values to x. This can discharge an OR determinism obligation if the child translations are sound. The AND branches must also satisfy decomposability where a d-DNNF target requires it. A syntactic Shannon split alone is not a certificate of source-to-circuit correctness.

### L10 — Smoothness completion

For a deterministic OR whose children do not mention the same variables, a child's missing variable z must be completed with its tautological support (z OR NOT z). Under weighted evaluation its contribution is multiplied by w_z(false)+w_z(true). Omitting that factor changes the weighted value even though Boolean support may still look correct.

### L11 — Weight-map substitution

Changing only a weight map preserves Boolean support but generally changes WMC/PWMC/AMC. A certified support representation can be reused, but each result needs the new domain, weight-map digest and independent arithmetic replay. Reusing the support proof does not reuse the old scalar result.

### L12 — Exact cancellation

A weighted or algebraic result can equal exact zero while Boolean support is nonempty. Therefore satisfiability/support and arithmetic value are separate result fields. value == 0 is never an UNSAT test in a signed or complex coefficient domain.

## 3. Contract × transformation matrix

| Transformation | SAT | #SAT | WMC | PMC | PWMC | AMC |
|---|---|---|---|---|---|---|
| Clause permutation/duplication | same | same | same | same | same | same |
| Consistent variable bijection + weight transport | same | same | same | same | same | same |
| Add free hidden variable | same | ×2 | ×(w0+w1) | same | same | ×(w0+w1) |
| Add free visible variable | same | ×2 | ×(w0+w1) | ×2 | ×(w0+w1) | ×(w0+w1) |
| Move hidden to visible | same | same | depends on contract/weight scope | may change | may change | same full AMC if weights unchanged |
| Redundant implied clause | same | same | same | same | same | same |
| Unique definitional extension | same | same under unique extension | only with neutral/transported auxiliary weights | same over old visible variables | same over old visible variables with visible weights | only with neutral/transported auxiliary weights |

Here w0+w1 denotes the two literal weights for the added variable in the appropriate exact coefficient domain. “Same” assumes all stated preconditions, especially identical declared variable universes and correct transport of weights.

## 4. Executable v1 coverage

The current PWMC micro-suite executes eight laws:

1. L3 free-hidden addition: full models double while support and PWMC are unchanged.
2. L2 variable-order permutation with literal-index remapping.
3. L1 clause/literal duplication and permutation.
4. L6 redundant/subsumed clause addition.
5. L4 free-visible extension with predicted exact PWMC multiplication.
6. L5 projection-boundary change with a predicted support/value change.
7. A unique hidden definitional extension z <-> d, preserving the projected support and PWMC on the fixed fixture.
8. L11 visible weight-map substitution, preserving support while changing the exact scalar (-2/5 to 17/24).

The unique-definitional-extension test covers projected support/PWMC only; it does not establish general full WMC preservation or prove arbitrary auxiliary-weight transport. L8, L9, L10 and L12 are not all fully generated as independent fixtures. L9/L10 require circuit-level structural witnesses; they should not be faked by CNF-only tests.

## 5. Research significance

A metamorphic benchmark should score both invariance accuracy and delta accuracy. This prevents a system from passing by returning the same constant answer for every transformed input. Each law should eventually have positive cases, negative controls that violate a precondition, and a minimal counterexample retained as a fixture.
