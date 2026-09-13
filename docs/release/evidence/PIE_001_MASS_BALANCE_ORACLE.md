# PIE-001 independent mass-balance oracle evidence note

Status: independent reference arithmetic for PIE-001. This is not production Rust evidence and not a lunar/Mars process qualification claim.

## Scope

`scripts/pie-mass-balance-oracle.py` is standard-library Python and imports no Symthaea code. It freezes conservative interval semantics for one declared process basis before the production PIE implementation is written.

Mass residual sign convention:

`residual = output mass - input mass`

All matter crossing the process boundary is expected to be represented as a material input or output. Electricity, heat, power and process time are non-material services and do not enter the mass balance.

## Classifications

- `ExactBalanced`: every stream is a point value and total input/output agree within the explicit tolerance policy.
- `ExactUnbalanced`: every stream is a point value and the totals disagree beyond tolerance.
- `PossibleWithUncertainty`: at least one stream is uncertain and the input/output intervals overlap once the explicit tolerance is applied. This is deliberately **not** evidence that conservation is established.
- `ImpossibleWithinBounds`: input/output intervals are disjoint beyond the declared tolerance.

An uncertain case can never be upgraded to `ExactBalanced` merely because its intervals are broad.

## Tolerance

The effective tolerance is:

`max(absolute_tolerance_kg, relative_fraction * max(total_input_upper, total_output_upper))`

Tolerance is an explicit study policy and must never be inferred from desired closure.

## Executed synthetic fixtures

The oracle self-test was executed locally on 2026-09-11 and returned `ok`.

Covered fixtures:

1. exact 10 kg -> 10 kg closure;
2. exact 10 kg -> 9.9 kg with loose vs tight absolute tolerance;
3. 9–11 kg input vs 10–12 kg output => `PossibleWithUncertainty`;
4. 9–10 kg input vs 11–12 kg output => `ImpossibleWithinBounds`;
5. product + by-product + waste output streams summed together;
6. multiple material inputs summed together;
7. widening an exact failed case can weaken it to `PossibleWithUncertainty` but cannot strengthen it to `ExactBalanced`;
8. relative-tolerance fixture;
9. negative, reversed, NaN, empty-stream and negative-tolerance inputs fail closed.

## Deliberate non-claims

This oracle does not model elemental/species conservation, stoichiometry, thermodynamics, reaction kinetics, yield, purity, process feasibility, equipment feasibility, resource abundance, economics or hardware authority.

Element/species closure belongs after PIE-003 introduces evidence-bearing composition and impurity models.

Parent issue: #1606
Master program: #1604
