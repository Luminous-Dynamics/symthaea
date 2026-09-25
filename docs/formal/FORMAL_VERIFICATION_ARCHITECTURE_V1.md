# Symthaea Formal Verification Architecture v1

Status: architecture contract only

Tracking issue: #5712 (`SYM-FV-000`)

## Purpose

This document freezes the verification roles, evidence classes, trust boundaries, and claim ceilings for formal-methods work across Symthaea and its direct integration boundaries.

The objective is not to maximize the number of proof tools. The objective is to preserve one semantic implementation while obtaining multiple independently useful evidence views whose authority never exceeds what they actually establish.

## Governing rule

```text
one semantic system
+ multiple explicit verification views
+ exact refinement/crosswalk evidence
!= multiple independently evolving specifications
```

No proof result, model-checker result, translation artifact, crosswalk, test, or qualification receipt gains runtime authority merely by existing.

## Verification planes

### 1. Mathematical specification — Lean 4 + Mathlib

Lean owns high-level mathematical definitions and theorems where deductive proof is the appropriate evidence class. Initial targets include HDC algebra and, later, carefully stated LTC/ODE properties.

Lean proofs remain subject to `symthaea-proof-audit` for theorem-statement conformance and declared axiom policy. A clean kernel proof of the wrong theorem is not accepted evidence for the intended specification.

### 2. Rust-to-Lean source-semantics refinement — Charon + Aeneas

Aeneas may translate selected supported safe-Rust source semantics into Lean. The first target is the scalar `BinaryHV` binding path.

Aeneas is a source-semantics bridge, not a verified Rust compiler. Translation success does not establish correctness of rustc, LLVM, machine code, unsafe code, concurrency, foreign libraries, SIMD instructions, or any source construct outside the exact supported/extracted boundary.

Generated external/unmodeled definitions are part of the evidence boundary and MUST be inventoried rather than silently trusted.

### 3. Deductive implementation verification — Verus

Verus may be used for selected low-level Rust kernels where SMT-backed contracts provide useful implementation-level assurance: bounded arithmetic, array/bitmap invariants, state transitions, replay windows, and similarly narrow execution kernels.

Verus is not the default proof language for abstract mathematics and does not replace Lean. It also does not establish rustc/LLVM correctness.

Creusot/Why3 is not part of the v1 standard toolchain. It may be evaluated later only when a concrete important Rust proof is materially blocked in Verus and Creusot can demonstrate a distinct advantage.

### 4. Distributed/temporal and structural formal models — TLA+/Alloy

TLA+/TLC and Alloy are appropriate for bounded temporal/concurrent and structural state-space properties. Mycelix already owns substantial canonical work in this plane; Symthaea should consume or interoperate with that evidence rather than create competing models for the same semantics.

Quint is not a second canonical formal specification language in v1. It may be evaluated as a bounded model-based-testing/developer bridge where it can drive actual Rust implementations without introducing a parallel normative specification.

### 5. Qualification and provenance

Exact-head qualification binds what was actually checked:

- exact source/model/spec identities;
- exact translator/verifier/prover identities;
- options and configurations;
- retained generated artifacts where required;
- theorem/property identity;
- negative controls and mutation controls where applicable;
- evidence-class claim ceiling;
- postflight immutability.

A workflow definition, queued run, stale ancestor run, generated artifact, or local test result is not automatically a qualification PASS.

## Evidence classes

### `AbstractFormalTheorem`

A theorem follows from the stated formal definitions and assumptions under the admitted Lean axiom policy.

Does not imply the production Rust implements that theorem.

### `ExtractedSourceRefinement`

A selected supported Rust source subject, as translated through the exact Charon/Aeneas pipeline, satisfies a stated formal relationship/specification.

Does not imply native binary correctness or coverage of excluded source/runtime behavior.

### `DeductiveImplementationProof`

A selected executable Rust subject satisfies the exact contracts established by the implementation verifier under its trust boundary.

Does not imply whole-program, compiler, dependency, environment, distributed, or deployment correctness.

### `BoundedModelSafety`

No specified safety violation was found inside the exact finite model/configuration/state-space qualification profile.

Does not imply an unbounded theorem or liveness.

### `TemporalModelEvidence`

A named temporal/reachability/liveness-related property is established under the exact model, fairness assumptions, bounds, checker, and configuration recorded by the receipt.

Its scope MUST be stated explicitly.

### `BoundedTraceConformance`

An implementation and formal model agree under an explicit projection/refinement relation over the exact retained bounded traces.

Does not imply full semantic refinement.

### `RuntimeQualification`

The exact runtime subject exhibited the claimed bounded behavior under the qualification environment/profile.

Does not strengthen the underlying mathematics or formal theorem and does not generalize outside its qualification profile.

## Mandatory non-equivalences

The following distinctions are constitutional for formal evidence:

```text
AbstractFormalTheorem != implementation theorem

Aeneas extraction != proof
Aeneas extraction != compiler verification
Aeneas extraction != native binary verification

generated Lean != Lean proof
Lean proof != intended-theorem conformance
Lean proof != axiom-policy acceptance

Verus PASS != rustc correctness
Verus PASS != LLVM correctness
Verus PASS != distributed-protocol correctness

TLC bounded PASS != unbounded proof
Alloy UNSAT != theorem outside the declared scope
model reachability != production reachability

crosswalk != behavioral refinement
bounded trace conformance != full refinement
formal-model refinement != Holochain/runtime refinement

cryptographic authentication != authorization
formal verification != permission for external effects
proof receipt != runtime authority
```

## Formal artifact trust chain

For Rust-to-Lean work, the minimum evidence chain is:

```text
exact Rust source
    -> pinned Rust toolchain
    -> pinned Charon
    -> retained translation identity / LLBC digest
    -> pinned Aeneas
    -> generated Lean digest
    -> external-model census
    -> pinned Lean + Mathlib
    -> checked theorem
    -> #print axioms
    -> symthaea-proof-audit
    -> exact qualification receipt
```

A later receipt schema should bind at least:

- source Git commit/blob/path;
- Rust toolchain;
- Charon revision/options;
- translation/LLBC digest where available;
- Aeneas revision/options;
- generated Lean file digests;
- external/unmodeled type/function census + digests;
- hand-written specification/proof digests;
- Lean version;
- Mathlib revision;
- theorem identity and statement digest;
- `#print axioms` result;
- `symthaea-proof-audit` policy/result;
- qualification workflow/runner/config identity;
- evidence class and claim ceiling.

## HDC program

The first formal implementation program is intentionally small.

### `SYM-FV-001A`

Attempt exact Charon/Aeneas extraction of the current production scalar binary HDC binding function without product behavior change.

A failed extraction must be classified as an extraction/tooling boundary, not as a failed HDC theorem.

### Conditional `SYM-FV-001B`

Create a tiny verification-friendly production kernel only if `001A` proves the current crate boundary is not practically extractable.

The kernel MUST be used by production `BinaryHV`; a copied verification-only XOR implementation is forbidden as refinement evidence.

### `SYM-FV-002`

Define abstract Lean HDC binary binding semantics and prove the basic algebraic laws under the intended representation.

Candidate laws include identity, self-inverse, commutativity, associativity, and exact bitwise characterization.

### `SYM-FV-003`

Prove the exact extracted scalar Rust function refines the abstract Lean binding operation.

### `SYM-FV-004`

Extend the relationship to Hamming distance/similarity semantics where the production representation and formal definition admit an exact statement.

### `SYM-FV-005`

Establish a separate optimized-implementation relation between production SIMD/runtime kernels and the verified scalar reference. Unless those optimized paths are directly verified, the claim remains conformance/refinement testing rather than Lean verification of machine instructions.

## LTC/FEP program

Do not begin with a blanket "FEP convergence" claim.

The staged mathematical program is:

```text
LTC vector-field definition
-> explicit domain and positive-tau assumptions
-> continuity / Lipschitz obligations
-> local existence and uniqueness
-> boundedness under stated assumptions
-> candidate Lyapunov/free-energy functional
-> monotonicity theorem if justified
-> asymptotic/convergence theorem only if justified
-> separate numerical/Rust refinement obligations
```

Required distinction:

```text
FEP-inspired architecture
!= free-energy monotonicity theorem
!= convergence theorem
```

Floating-point Euler/RK4 execution is not definitionally identical to the idealized real-valued ODE and requires a separate numerical-analysis/refinement argument.

## Distributed-system relationship

Symthaea should not duplicate Mycelix's existing constitutional TLA+/Alloy stack. Cross-repository consumers should bind qualified semantic/model artifacts and explicit crosswalk/refinement evidence.

For Holochain-backed systems, avoid assuming one global consensus state. Relevant properties include deterministic validation, unresolved dependencies, partitioned views, fork/conflict handling, causal histories, reconciliation, freshness, and application-defined authority/finality.

## Tool admission policy

A new formal tool is admitted to the standard stack only when all of the following are true:

1. it establishes an evidence class not already covered adequately;
2. it has a concrete high-value production subject;
3. its trust boundary and claim ceiling can be made explicit;
4. its outputs can be bound into the qualification/evidence system;
5. maintaining it does not create an independently evolving semantic specification.

This policy intentionally prevents a formal-methods tool zoo.

## Immediate dependency order

```text
SYM-FV-000  verification constitution
    |
    v
SYM-FV-001A exact Aeneas extraction probe
    |
    +-- if required --> SYM-FV-001B production kernel seam
    |
    v
SYM-FV-002 abstract HDC specification
    |
    v
SYM-FV-003 extracted-source refinement proof
    |
    +--> SYM-FV-004 Hamming semantics
    +--> SYM-FV-005 optimized implementation relation
    +--> formal evidence receipt schema

Only after this pipeline is operational:

LTC mathematical foundation -> existence/uniqueness -> stability -> stronger FEP claims
```

## Nonclaims of this document

This architecture document changes no production runtime behavior and establishes no positive proof about HDC, LTC, FEP, SIMD, cryptography, distributed protocols, or external effects.

It defines only how later evidence may be constructed, classified, combined, and limited.
