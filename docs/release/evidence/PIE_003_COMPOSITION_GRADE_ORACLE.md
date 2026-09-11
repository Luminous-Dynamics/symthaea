# PIE-003A composition / grade / constituent-balance oracle

## Purpose

Freeze implementation-independent reference semantics for PIE-003 before adding production Rust types.

The oracle is `scripts/pie-composition-grade-oracle.py` and imports no Symthaea code.

## Semantics

- composition is represented as bounded mass-fraction intervals plus an explicit unknown-fraction interval;
- a profile is admissible only when those bounds can contain a total mass fraction of exactly 1.0;
- duplicate constituent identifiers fail closed;
- missing constituent information never silently satisfies a grade requirement;
- grade assessment is conservative:
  - `Pass`: every admissible composition satisfies the declared constraints;
  - `Fail`: at least one declared constraint cannot be satisfied;
  - `Indeterminate`: some admissible compositions pass and some fail;
- widening uncertainty cannot strengthen a grade conclusion;
- exact-mass lot mixing preserves constituent masses;
- constituent-level balance uses the same exact/possible/impossible uncertainty discipline as PIE-001;
- exact constituent closure is reported only when both stream masses and composition fractions are point-valued;
- total mass closure can coexist with constituent imbalance, and the latter remains visible.

## Executed synthetic fixtures

The candidate oracle self-test was executed locally on 2026-09-11 and returned `ok` before the checked-in reference was created. The checked-in source was then fetched back from GitHub and reviewed against that candidate.

Fixtures cover:

1. valid exact 60/40 two-constituent composition;
2. physically impossible fraction totals fail closed;
3. exact grade pass;
4. impossible grade fail;
5. widened composition uncertainty -> `Indeterminate`, never stronger;
6. missing required constituent -> `Fail`;
7. exact 6 kg A + 4 kg B mixing -> 60/40 composition;
8. exact constituent closure for a synthetic separation process;
9. bulk mass closes while constituent A fails closure;
10. uncertain constituent balance -> `PossibleWithUncertainty`;
11. explicit unknown fraction can block a grade;
12. duplicate/nonfinite/out-of-range fractions fail closed.

## Important limitation

Constituent intervals are treated as independent bounds. This is a conservative screening representation and can over-approximate the physically reachable joint composition set when constituents are correlated. A later correlated-composition model may tighten bounds but must not retroactively convert uncertainty into stronger evidence without explicit justification.

## Non-claims

This tranche does not establish:

- lunar or Martian resource composition;
- process yield or product purity;
- reaction stoichiometry or thermodynamic feasibility;
- kinetics, scale-up, equipment performance, or economics;
- qualification of any material grade;
- manufacturing or hardware authority.

Tracks #1611 and master program #1604.
