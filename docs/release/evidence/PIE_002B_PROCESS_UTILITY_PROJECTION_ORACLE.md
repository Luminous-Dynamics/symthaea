# PIE-002B Process Utility Projection Oracle

Status: independent reference semantics; synthetic fixtures only.

## Purpose

`scripts/pie-process-utility-projection-oracle.py` freezes a fail-closed projection from generic process utility declarations into an electrical accounting basis without importing Symthaea production code.

The projection deliberately refuses to treat all utility records as one additive quantity.

## Core theorem

On one declared process basis:

- electrical energy entries are additive and may be summed with checked interval arithmetic;
- peak electrical power is not safely additive or reducible by `max()` without an explicit concurrency/coincidence model;
- process-time entries are not safely additive or selectable without explicit sequencing semantics;
- thermal-energy quantity alone is not a thermal-feasibility claim because required temperature is unbound;
- cooling-energy quantity alone is not a heat-rejection feasibility claim because sink temperature, rejection path and timing are unbound.

Therefore multiple peak-power or process-time declarations are `Ambiguous`, not silently combined. Missing electrical energy, peak power or process time is `Incomplete`, not zero.

## Status semantics

`Complete` means the electrical projection has:

1. at least one electrical-energy declaration, summed with checked interval addition;
2. exactly one process-basis peak-power declaration;
3. exactly one strictly-positive process-time declaration.

`Ambiguous` means at least one non-additive electrical basis field has multiple declarations whose relationship is not specified.

`Incomplete` means the basis is missing one or more required electrical fields and contains no multiplicity ambiguity.

Thermal/cooling unresolved reasons are tracked separately and do not rewrite an otherwise complete electrical basis.

## Explicit unresolved reasons

- `MissingElectricalEnergy`
- `MissingPeakPower`
- `MultiplePeakPower`
- `MissingProcessTime`
- `MultipleProcessTime`
- `ThermalTemperatureUnbound`
- `CoolingRejectionUnbound`

## Synthetic fixtures

The self-test covers:

- two electrical-energy intervals aggregating to `[50,70] J` while one peak-power and one process-time interval preserve a `Complete` electrical projection;
- missing process time -> `Incomplete`;
- duplicate peak power -> `Ambiguous` and no selected peak value;
- duplicate process time -> `Ambiguous` and no selected duration;
- thermal and cooling quantities remain visible while their missing quality/rejection semantics remain explicit;
- no electrical energy -> `Incomplete`;
- aggregate floating-point overflow -> fail closed;
- negative/reversed ranges, zero-inclusive process time, and blank process ID -> fail closed.

## Claim boundary

This oracle does not prove electrical supply feasibility, thermal feasibility, cooling feasibility, storage dispatch, equipment capability, process performance, or control authority. It only decides whether generic utility declarations can be projected into an unambiguous electrical accounting basis and which non-electrical semantics remain unresolved.

Tracks #2660, #1610, and master program #1604.
