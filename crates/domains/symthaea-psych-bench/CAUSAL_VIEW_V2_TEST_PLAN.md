# Causal view V2 regression plan

The V2 reporting layer must preserve these invariants:

1. `Observed + causal Qualified -> CausallySupported`.
2. `Observed + causal NotDemonstrated -> final Observed`, while the causal lineage records `NotDemonstrated`.
3. `Observed + causal Contradicted -> final Observed`, while the causal lineage records `Contradicted`.
4. `Observed + causal Inconclusive -> final Observed`, while the causal lineage records `Inconclusive`.
5. A positive causal lineage cannot promote from a base weaker than `Observed`.
6. A direct lineage cannot author `CausallySupported` or `FunctionallySupported`.
7. A causal lineage cannot author `FunctionallySupported`.
8. Duplicate method lineages fail closed.
9. The base scalar Butlin report remains immutable.
10. The public causal resolver requires the opaque cryptographically verified promotion token.

V1 remains the compatibility and authority-verification path. V2 changes only cross-lineage resolution semantics.
