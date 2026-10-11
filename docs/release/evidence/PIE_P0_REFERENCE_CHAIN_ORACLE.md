# PIE Phase-0 synthetic reference-chain audit oracle

## Purpose

Freeze a small end-to-end acceptance harness that composes conservative Phase-0 semantics across resource access, process capacity, bulk mass, constituent balance, material grade, utilities, circularity, maintenance/spares, and import dependency closure.

The oracle is `scripts/pie-reference-chain-oracle.py` and imports no Symthaea code.

This is deliberately **Moon-shaped synthetic data**, not a lunar process-performance claim.

## Executed reference behavior

The final self-test was executed locally on 2026-09-11 and returned `ok` before this reference was committed.

The baseline synthetic chain is physically feasible under its declared arithmetic fixtures but remains explicitly `ImportDependent` because one blocker has no local production route.

Failure/uncertainty injections prove:

1. removing remote transport makes the chain `Impossible`;
2. bulk mass closure cannot hide a tracked-constituent imbalance;
3. unknown composition weakens an otherwise exact grade to `Possible`;
4. allocating more recycled mass than was actually recovered makes circularity `Impossible`;
5. removing a required maintenance spare makes lifecycle feasibility `Impossible`;
6. removing an imported critical blocker makes the whole chain `Unavailable` unless an explicit local alternative exists;
7. enough total energy does not compensate for insufficient peak power;
8. widening resource-recovery uncertainty weakens `Guaranteed` resource access to `Possible`;
9. introducing a local alternative for the blocker changes closure to `LocallyClosed` without modifying unrelated physical gates.

## Gate aggregation

The harness uses conservative three-state physical gate semantics:

- `Guaranteed`: the full admissible range satisfies the requirement;
- `Possible`: some admissible states satisfy it and some do not;
- `Impossible`: the admissible range cannot satisfy it.

`overall` is the weakest physical gate unless the declared critical dependency is unavailable, in which case the chain is `Impossible`.

Dependency closure is reported separately as `LocallyClosed`, `ImportDependent`, or `Unavailable` rather than being hidden inside one weighted score.

## Important limitations

This reference intentionally duplicates only a minimal subset of the individual Phase-0 oracle semantics so it can test cross-gate composition from clean `main`. The individual independent oracles remain the normative detailed references for their domains.

The harness does not model real lunar/Martian compositions, chemical reactions, true thermodynamics, process kinetics, real equipment reliability, scheduling, multi-period inventory, route optimization, economics, human labor, autonomous plant control, or qualification.

A later production integration harness should consume shared typed PIE records and compare its frozen fixtures against the detailed independent oracles rather than expanding this synthetic script into a production simulator.

## Non-claims

Passing this synthetic chain does not establish that a lunar or Martian industrial chain is feasible. It establishes only that the Phase-0 bookkeeping rules can be composed without obvious contradiction.

Tracks #1698, Phase-0 exit gates #1647, and master #1604.
