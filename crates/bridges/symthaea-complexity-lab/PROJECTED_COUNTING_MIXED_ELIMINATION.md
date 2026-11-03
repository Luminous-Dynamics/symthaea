# ETH-021: projected counting through mixed existential/counting elimination

**Status:** Finite Python semantic reference committed. Full parser, proof certificate, Rust and Lean work remain open.

## 1. Output semantics are not interchangeable

For visible variables X and hidden variables Y, projected model counting is:

\[
\operatorname{PMC}(F,X)=
|\{x\in D_X:\exists y\in D_Y,\ F(x,y)\}|.
\]

The number of hidden witnesses for a visible assignment is irrelevant after the existential quantifier. If one visible assignment has 100 satisfying hidden witnesses, it contributes one projected assignment, not 100.

Ordinary #SAT instead counts full satisfying assignments. Ordinary WMC weights full satisfying assignments. Projected WMC applies the documented weight to each distinct supported projection. The projection set and quantifier boundary therefore belong in the output contract and proof certificate.

## 2. Safe initial execution strategy

Use an explicit two-phase plan:

1. Combine constraints and eliminate every hidden variable using Boolean conjunction/disjunction. This computes the support predicate over the still-visible variables.
2. Accumulate the exact count or projected weight over distinct visible assignments satisfying that predicate.

For PWMC in this reference, projected unary weights are exact rationals; negative weights are included. This is a deliberately narrow semantics fixture, not a claim to implement every official PWMC input format.

Do not sum hidden witnesses in natural-number or rational arithmetic and then call the result projected count. That computes a different quantity.

## 3. Executed finite reference

File: reference/projected_counting.py
Results: reference/projected_counting_results.json

Seed 0xE7021:
- 160 random finite Boolean-factor networks;
- 480 exact direct-projection versus mixed-elimination comparisons;
- three hidden-variable orders per instance: forward, reverse and random;
- exact rational visible weights, including negative values;
- a witness-multiplicity control where 2 full satisfying assignments map to 1 projected assignment.

The reference performs exhaustive enumeration over finite instances. It is not a scalable solver, official competition parser, Rust implementation or formal proof.

## 4. Required proof and certificate obligations

A future proof-carrying plan must include:
- exact projected/hidden partition of variables;
- hidden-only elimination order;
- evidence that hidden variables are eliminated with Boolean existential semantics;
- exact support predicate over the projected variables;
- projected weight application exactly once per supported projection;
- completeness and disjointness of the final projected assignment domain.

Mutation controls must reject:
- a projected variable included in the existential phase;
- a hidden variable omitted from the phase;
- duplicated or missing projected variables;
- a hidden witness multiplicity leaking into the projected accumulator;
- projected weights multiplied once per hidden witness;
- an incomplete or duplicated visible assignment enumeration.

## 5. Research sequence

1. Map the exact MC2026 PMC and PWMC formats and parser conventions.
2. Freeze #SAT, WMC, PMC and PWMC as different output contracts.
3. Port the two-phase mixed elimination to the existing factor table engine.
4. Add replayable certificates and an independent checker.
5. Prove the finite existential-elimination and projected-accumulation rules in Lean.
6. Add weighted/projected benchmark suites and compare against Ganak and other tools with matching formats and exact outputs.
7. Only then test more advanced projected decomposition and variable-order planning.

## 6. Claim ceiling

This reference validates the basic distinction between hidden witnesses and distinct visible assignments on a small corpus. It does not show an official competition result, a general projected-counting algorithm, arbitrary PWMC semantics, Rust/Lean qualification or better-than-SOTA performance.
