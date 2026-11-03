# ETH-027 — query-relative semantic compilation contract v1

**Status:** proposed executable wire contract and reference-test tranche. This document does not claim a Rust implementation, byte-level canonicalizer, SCPOG adapter, or production proof checker.

## 1. Equivalence is query-relative

Let F(V,H) be a Boolean formula over a declared variable universe, partitioned into visible variables V and hidden variables H. A transformation is valid only relative to the exact semantic question being answered.

| Relation | Establishes | Does not establish |
|---|---|---|
| satisfiability_equivalence | Both inputs are satisfiable or both unsatisfiable | Model identity, model count, projected support, weighted or algebraic values |
| full_model_equivalence | Identical satisfying assignments over the same declared universe | Preservation after changing variable identities, weights or domain |
| projected_support_equivalence | Identical visible assignments for which at least one hidden witness exists | Hidden-witness multiplicity or full model count |
| model_bijection | A bijection between source and target models under an explicit variable map | Weighted-value equality without weight transport |
| weighted_value_preservation | The same exact value for a declared domain and weight-map digest | Other weight maps or coefficient domains |
| support_to_scalar_evaluation | Exact evaluation of a previously established support representation | Source-to-support correctness |
| representation_only | Encoding/serialization change with no new semantic claim | Any unlisted semantic property |

A SAT-preserving rewrite is not automatically #SAT-preserving. A model-bijective Tseitin extension does not automatically preserve WMC if auxiliary-variable weights are not neutral or transported. A projected-support rewrite can change the number of hidden witnesses and still be correct for PMC/PWMC.

## 2. Five contracts

For F over variables V and H:

- #SAT counts satisfying assignments over V ∪ H.
- WMC sums the product of literal weights over all satisfying assignments over V ∪ H, initially over exact Q.
- PMC counts visible assignments v for which some hidden assignment h satisfies F(v,h).
- PWMC sums the visible literal-weight product once for each v for which some h satisfies F(v,h), initially over exact Q.
- Complex AMC sums per-model products over all assignments, initially over exact Q(i).

For non-projected contracts in v1, H is empty. The contract and variable partition must be explicit; no implicit quantifier or semiring coercion is permitted.

## 3. Two independent semantic rails

For projected contracts, first establish:

\[
\psi(V) \equiv \exists H\,F(V,H).
\]

Only then count or weight the distinct visible assignments represented by ψ. Hidden-witness addition is not existential projection.

Example: F(d,h)=d, with visible d and hidden h. There are two full satisfying assignments, but projected support is just {d=true}. If w_d(true)=2/3, PWMC is 2/3, not 4/3.

Store support and arithmetic separately. Signed or complex exact cancellation can produce arithmetic value zero with nonempty support. Zero must never be interpreted as UNSAT.

## 4. Scope-correct arithmetic

For a deterministic, decomposable circuit:

- Decomposable AND multiplies child values because child variable scopes are disjoint.
- Deterministic OR adds child values because child model sets are disjoint.
- A non-smooth OR completes each child over variables missing from its scope using the product of (w_x(false)+w_x(true)).
- The root completes variables in the declared universe omitted from the root scope.

These rules are valid only when the structural preconditions and exact coefficient semantics are established. ETH-027's finite Python prototype uses exhaustive checks for small cases; those checks are not a scalable determinism proof.

## 5. Evidence chain and obligation states

Each envelope binds source bytes, parser profile, output contract, variable partition, domain, ordered representation stages, exact result and independent replay.

Each stage identifies input/output representation digests, an explicit preservation relation and obligations. An obligation is one of:

- proof_checked: requires a proof artifact digest and checker identity/binary digest;
- tested_only: requires test artifact digest, suite ID and case count;
- unsupported: guarantee not provided;
- failed: attempted and rejected.

A test must never be promoted to a proof. A digest is an identity commitment, not a correctness argument. A relation whose obligation failed or is unsupported cannot be presented as established.

Cross-field checks must enforce exact visible/hidden partitioning, contract/quantifier/domain compatibility, weight-map scope, digest-chain continuity, proof-evidence requirements and replay binding. A SAT-only relation cannot discharge any exact-counting or projected-counting obligation.

## 6. SCPOG boundary

Use SAT 2025 SCPOG / Skolem POG for the source-CNF-to-projected-Boolean-support obligation. Its certificate does not automatically prove arbitrary Q or Q(i) weighted values.

The composition is:

1. Pin the upstream compiler, proof generator and checker.
2. Check that the proof establishes the exact projected support for the declared visible variables and the required circuit properties.
3. Import the resulting support representation under a strict parser and bind its digest.
4. Evaluate it with exact Q or Q(i) arithmetic under a specific weight-map digest.
5. Replay the arithmetic with structurally independent code and bind replay to the same support, weights and result.

Keep projected SCPOG and ordinary CPOG profiles distinct. Keep the arithmetic proof obligation distinct from the Boolean projection proof.

## 7. Canonical bytes

The schema reserves symthaea-json-struct-v1. It is not a production canonicalizer yet. Before relying on its digests, implement fixed UTF-8/field ordering, canonical ordering for set-valued arrays, duplicate-key rejection, stable integer encodings and reduced rationals with positive denominators. Q(i) values must canonicalize real and imaginary rationals independently.

A manually authored fixture with replay status not_run must not invent a checker binary digest. A replay marked matched or mismatched must identify the checker engine and binary digest.

## 8. Acceptance

For each supported contract/gate combination: exact agreement with exhaustive small oracles; zero false accepts on malformed and mutated artifacts; canonical round-trip identity; explicit proof-vs-test status; no hidden-witness multiplicity in PMC/PWMC; exact cancellation preservation; and recorded solver, proof, checker, arithmetic and end-to-end costs.

This is a proposed v1 wire contract. It does not claim that the canonicalizer, Rust evaluator, SCPOG adapter or formal Symthaea proof checker has been implemented.
