# ETH-027 — semantic aggregation algebra and compositional certificate DAG v1

**Status:** research architecture proposal. It defines an implementable semantic model and falsifiable experiments; it is not a formalized theorem or implemented checker.

## 1. The key semantic distinction is the aggregation operator

Let F(X,H) be a Boolean relation over visible variables X and hidden variables H. For each visible assignment x, define its witness fibre:

W_F(x) = { h | F(x,h) }.

The projected support is:

S_F = { x | W_F(x) is nonempty }.

The output contract chooses how the hidden fibre is reduced:

- **Existential projection:** exists h. F(x,h); the hidden fibre becomes one Boolean support bit.
- **Additive projection:** sum a weight over every hidden witness; witness multiplicity remains observable.
- Other future contracts may use max, min, probability, or a domain-specific aggregate, but that aggregate must be explicit in the contract.

This gives distinct formulas:

PMC(F;X) = |S_F|

PWMC(F;X,w_X) = sum over x in S_F of the product of visible literal weights.

WMC(F;w_X,w_H) = sum over x and satisfying h of the visible and hidden literal-weight products.

The first two collapse every nonempty hidden witness fibre to one Boolean fact before counting/weighting visible assignments. WMC sums the fibre with multiplicity and hidden literal weights. They are not interchangeable implementations of the same scalar query.

The compiler should therefore carry a typed aggregation plan:

| Contract | Hidden-variable reduction | Final reduction |
|---|---|---|
| #SAT | none | sum over full assignments in N |
| WMC | weighted sum in Q over hidden states | weighted sum over visible states |
| PMC | Boolean OR over hidden states | count supported visible states in N |
| PWMC | Boolean OR over hidden states | weighted sum over supported visible states in Q |
| Complex AMC | weighted sum in Q(i) over all states | Q(i) scalar |

This typed plan prevents a particularly dangerous bug: replacing existential projection with additive elimination and multiplying by hidden-witness multiplicity.

## 2. A certificate is a typed DAG, not merely a linear log

Represent every immutable input/output representation as a node. A transformation is a directed edge with:

- source and target representation digests;
- operation and tool identity/version;
- semantic contract/context (variable universe, visible/hidden partition, coefficient domain, weight-map identity where relevant);
- a typed preservation relation;
- proof obligations and evidence digests;
- resource/provenance receipt.

The certificate structure is a DAG because one certified projected-support node can feed many downstream exact evaluations under different visible weight maps. A second solver or representation can also feed an independently checked comparison node. The DAG shares evidence rather than duplicating a full source-to-support proof for every scalar query.

A simplified graph is:

source CNF -> parsed semantic problem -> compiled projected support

compiled projected support -> Q evaluation under weights A -> replay A

compiled projected support -> Q evaluation under weights B -> replay B

compiled projected support -> PMC cardinality -> replay cardinality

The projection proof is reusable only while the source digest, parser profile, projection partition and compiled support digest remain identical. A different partition or target support digest requires a new support certificate.

## 3. Preservation relations and safe composition

An edge may claim one of the following (or a more specialized relation):

- SatisfiabilityEquivalent: only SAT/UNSAT is preserved.
- FullModelEquivalent: the same full assignments under an explicit variable identity map.
- ProjectedSupportEquivalent(X): the same support over the exact visible-variable set X.
- ModelBijection(map): a bijection between source and target model sets.
- WeightedValuePreserving(domain, weight-map-transport): exact scalar equality for a specific coefficient domain and explicit weight transport.
- SupportToScalarEvaluation(domain, weight-map-digest): evaluates an already established support representation; it is not a source-equivalence claim.
- RepresentationOnly: changes encoding/serialization, with no stronger semantic claim.

Composition rules must be explicit:

1. Full-model equivalence composes transitively only when the intermediate variable map and universe agree.
2. Projected-support equivalence composes only when both edges use the same visible-variable identities and quantifier boundary, or provide a proved partition transport.
3. Model bijections compose by function composition; the composed map and its inverse obligations must be bound.
4. Weighted-value preservation composes only over the same exact coefficient domain with composed, digest-bound weight transport. A claim for one weight map cannot be silently generalized to another.
5. Satisfiability equivalence never upgrades to model equivalence, projected support equivalence, model bijection or weighted-value preservation.
6. SupportToScalarEvaluation consumes a support-certified node and one weight-map digest. Its output is a new scalar result node with its own replay receipt.

The validator must reject any attempted composition whose intermediate representation digest or semantic context does not match exactly.

## 4. Witness fibres make the preservation conditions measurable

A useful extension is to attach a fibre profile to transformations when practical:

- bijection: every target model has exactly one source preimage;
- constant_k_to_1: every target model has exactly k source preimages;
- variable_fibre: per-target multiplicity is supplied as an exact function/certificate;
- existential_surjection: every target support assignment has at least one source witness, with no assertion that witness counts are preserved.

This can explain why a transformation preserves one contract but not another. A unique definitional extension is a bijection on model sets and preserves #SAT. It preserves WMC only when the new-variable weights are neutral or the weight map is explicitly transported. An unconstrained hidden extension is 2-to-1 and does not preserve #SAT/WMC, but it preserves projected support and PWMC.

Fibre profiles are explanatory metadata until backed by a checked proof. A producer-supplied integer k is not evidence.

## 5. SCPOG proof-generation bottleneck: a focused research opportunity

SCPOG already uses an important architecture: the Skolem POG represents the projected d-DNNF and gives a hidden-variable witness for every supported visible assignment. The checker can then validate reverse implication and structural conditions without requiring a giant producer-emitted clause sequence for every input clause.

The published evaluation identifies forward-implication proof generation as the major weakness. The next work should be a controlled proof-generation portfolio rather than a replacement checker:

1. Baseline A: reproduce the upstream SCPOG generator and checker unchanged.
2. Baseline B: retain the same Skolem POG but generate the forward implication proof using an independent SAT proof-producing check of F(X,H) AND NOT psi(X), with an independently validated CNF encoding of NOT psi.
3. Candidate C: reuse D4's component/decomposition trace to produce local implication lemmas and compose them through the same target root.
4. Candidate D: hash-cons repeated forward lemmas, share proof sub-DAGs, and eliminate duplicate/subsumed derivation work where the proof calculus permits it.
5. Candidate E: compare eager full Skolem witness materialization with lazy, component-local witness generation, while retaining exact Skolem semantics and a checker that validates the declared representation.

These are hypotheses. A SAT refutation of F AND NOT psi is a valid way to establish the forward direction only when the target formula encoding and its equivalence to NOT psi is itself correctly bound and checked. It does not establish the reverse projected implication; the Skolem witness/checking obligation remains.

Every candidate must be compared on the same inputs and hardware, recording proof generation time/bytes, checker time/RSS, compiled DAG size and complete verified wall time.

## 6. A new primary benchmark: transformation coverage

In addition to final solver answers, measure:

- Contract coverage: fraction of transformation stages with a valid, checked obligation for the requested contract.
- Compositionality coverage: fraction of multi-stage chains that compose without falling back to a monolithic proof.
- Evidence reuse: number of downstream weight maps/evaluations supported by one unchanged projected-support proof.
- Proof amortization: projection-proof bytes and time divided by the number of verified downstream queries.
- Semantic mutation score: fraction of injected contract/precondition/evidence-binding bugs rejected.
- Unsupported surface: fraction of requests rejected explicitly rather than silently evaluated under a weaker contract.

A system that produces more raw counts but cannot identify which compiler transformations were justified should lose to a slower system with checked semantic coverage. Conversely, certificate overhead cannot be ignored: optimize the verified-coverage-versus-time curve.

## 7. Falsifiable acceptance criteria

1. A valid typed certificate DAG composes across at least three transformations with a checked relation at every edge.
2. Replacing any edge with SAT-only evidence cannot validate #SAT/WMC/PMC/PWMC/AMC.
3. A single certified projection is reused for 100 distinct weight maps; each scalar is independently replayed and keyed by the exact map digest.
4. A partition change invalidates the support proof cache.
5. A unique definitional extension preserves model count; an unconstrained hidden extension doubles model count but preserves projected support.
6. A generated proof mutation, target digest substitution, stale source digest or altered weight-map digest is rejected.
7. Each SCPOG forward-proof candidate is compared against the unchanged upstream baseline with identical compiler output and checker obligations.

Targets such as >=10% lower verified end-to-end cost or >=10% more verified held-out instances remain aspirational until measured on matched hardware/resources.

## 8. Scope and claim ceiling

This document is a proposed research architecture. The typed certificate DAG, fibre-profile checker, SAT-proof alternative and SCPOG proof-generation portfolio have not yet been implemented or formally verified. Existing finite PWMC tests validate selected small examples, not these general claims.
