## Multi-order reduction tournament (MON-039 v1.1 extension)

For fault universes larger than the exhaustive limit, the original reducer uses one deterministic deletion order and reports one_minimal when every single deletion has been tested and rejected.

A bounded tournament evaluates the same trial, initial fault set, scenario, recovery boundary, code revision and exact failure-signature oracle under up to eight explicit deletion permutations. Every order must be an exact permutation of the initial fault IDs. Duplicate or malformed orders block.

Each order produces its own reduction trace, subset-evaluation count, final fault set and minimality statement. The tournament selects the smallest complete result observed across tested orders, with deterministic lexicographic tie-breaking.

Guarantees:
- <=12 faults: each order uses exhaustive subset search; global_candidate_minimum remains scoped to the complete finite submitted fault universe and complete replay table.
- >12 faults: each completed ordering can establish only one_minimal. The tournament reports best_observed_one_minimal_across_tested_orders. It does not claim global minimality, a unique root cause or causation outside the tested intervention universe.
- Missing/incomplete subset evidence in any declared ordering keeps the tournament evidence incomplete. The best complete observed candidate may still be reported descriptively, but the tournament remains indeterminate.
- The original Minimize API and semantics remain unchanged; the new MinimizeInOrder and MinimizeTournament APIs are additive.

The deterministic order generator emits at most eight unique permutations (reverse, ascending, then rotations). It uses no ambient random state. A future experiment can preregister other explicit orders; the permutation bytes are digest-bound in the result.
