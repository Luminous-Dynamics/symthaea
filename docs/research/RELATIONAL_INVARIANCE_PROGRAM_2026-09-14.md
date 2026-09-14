# REL — Relational Invariance Program

**Status:** frozen research contract for REL-000 through REL-003.  This document does not change production cognition.

## Purpose

REL asks one narrow question before any architectural or consciousness claim:

> Which transformations of a Symthaea internal representation change only its description, and which change admitted cognitive content?

The first tranche treats that as a representation-specific algebra/geometry question.  It does **not** assume that Symthaea has a field-theoretic gauge symmetry.

## Terminology

The following terms are intentionally distinct.

### Algebra automorphism

For a declared representation algebra `(H, B, ...)`, a transformation `g` is an automorphism only when it preserves the admitted operations, e.g.

`g(B(x, y)) = B(g(x), g(y))`.

### Metric isometry

For a declared similarity/distance `S`, a transformation `g` is an isometry when

`S(g(x), g(y)) = S(x, y)`.

An isometry need not be an algebra automorphism.

### Representation covariance

A coordinate/frame change is covariant when states **and the operators that act on them** transform consistently.  For an operator `rho`, a frame transformation `g` induces

`rho' = g rho g^-1`.

Covariance is weaker than the claim that `g` commutes with a fixed `rho`.

### Representational redundancy

A redundancy requires more than covariance: multiple internal descriptions must induce the same declared observables or computation.  REL reserves the word **gauge** for a demonstrated redundancy rather than using it as a synonym for symmetry, isometry, or covariance.

## Representation-specific contracts

There is no universal `G_Sym` assumed across Symthaea.

### BinaryHV

Production binding is XOR and the metric is Hamming-derived similarity.  Coordinate permutations are candidate full automorphisms/isometries because a common permutation preserves both XOR coordinatewise structure and Hamming matches.

A generic invertible linear transformation over `F_2` may preserve XOR algebra while changing Hamming geometry.  Such transformations are therefore negative controls for the stronger full-algebra-plus-metric claim.

### ContinuousHV

Production binding is Hadamard/element-wise multiplication and similarity is cosine.  Coordinate permutations are candidate full automorphisms/isometries.

Generic orthogonal transforms preserve cosine geometry but do not, in general, preserve Hadamard binding:

`U(x .* y) != (Ux) .* (Uy)`.

Orthogonal mixing is therefore a negative control for the invalid inference "metric isometry implies full HDC automorphism".

### Sequence operator

Cyclic permutation `rho` carries position/sequence semantics.  REL distinguishes:

1. **fixed-operator invariance:** `g rho = rho g`; from
2. **covariant frame change:** `rho' = g rho g^-1`.

The second can hold even when the first does not.  Tests must not collapse these claims.

## REL-001 — automorphism theorem surface

The first theorem tranche must establish, with deterministic fixtures:

- BinaryHV common-coordinate permutations preserve binding and similarity;
- ContinuousHV common-coordinate permutations preserve binding and similarity to narrow floating-point tolerance;
- sequence encodings are tested separately for fixed-operator commutation and covariant conjugation.

## REL-002 — independent oracle boundary

Transformation/oracle code belongs outside production HDC methods.  It may call the public production operations being tested, but it must not add a new production representation or modify cognition paths.

A positive result supports only the declared identity for the tested representation and fixture family.

## REL-003 — negative controls

The first tranche must include transformations intentionally preserving only part of the structure.

Required examples:

- a continuous orthogonal mixing transform that preserves cosine while breaking Hadamard-binding covariance for a deterministic non-degenerate fixture;
- a binary XOR-linear transform that preserves XOR composition while changing Hamming similarity for a deterministic non-degenerate fixture.

These controls prevent a trivially insensitive test surface from being mistaken for relational invariance.

## Evidence and interpretation rules

1. Deterministic fixtures/seeds are frozen before interpreting results.
2. Exact algebraic identities use exact equality where the representation permits it.
3. Floating-point tests use narrow, stated tolerances and report the quantity compared.
4. Negative controls must mechanically falsify the stronger invalid claim for their fixture.
5. Nulls, unexpectedly small symmetry groups, and failed candidate transformations count as results.
6. No claim about improved reasoning, memory, efficiency, intelligence, or consciousness follows from REL-001 through REL-003.
7. Quotient/reduced representations are downstream experiments; equivariance/covariance is retained when reduction would discard task-relevant information.

## Downstream gates

- REL-004: temporal-origin/reference-frame invariance, explicitly separated from causal/version authority.
- REL-005: multi-agent frame transforms and loop closure.
- REL-006: persistent relational object identity.
- REL-007: raw vs covariant/equivariant vs reduced representation ablation.
- REL-008: self/world-model bridge only after earlier measurable results.
- REL-009: independent replication package after a non-null result.

## Issue map

- #3012 — taxonomy and parent program
- #3013 — REL-001/002/003 implementation
- #3014 — REL-004/005
- #3015 — REL-006/007/008
- #3016 — literature/provenance boundary
- #3017 — preregistered metrics
- #3018 — replication package
- #3019 — exact-subject qualification
