# D4v2 source-faithful dynamic projected-BCE trace v1

**Claim ceiling:** independent executable reference model of the protected-triple/watch/queue/assignment/active-clause subset, plus exhaustive projected-support oracle. It is not a D4 binary replay, full solver implementation, formal proof or benchmark.

## Pinned implementation

The reference model is based on the following upstream Git objects from crillab/d4v2 main:

- CnfManagerDynBlockedCl.cpp: 303817529debd95735a32ca9200791c3a854d992
- CnfManagerDynBlockedCl.hpp: d92fe5e5d25b26f981ff657230ff0a9cb8ad04bf
- CnfManagerDyn.cpp: 3f19bc0664ff90d167ee00b1b6321a0015fe1ff8
- CnfManagerDyn.hpp: 8c014829c7ccae545b1afb98c10d21b425e1b4b7
- ClauseInfo.hpp: c8d0718fdad78aba0c87701e2e19b5fd2acd7f75
- DataOccurrence.hpp: 0a74b2c9db6c8eb0863d4d43d1b13f9c0df5bc29
- ProblemManager.hpp: f5c2bbfd3a0951685a7055d1a9b4153395563bc1

The source is GPL-3.0-or-later; this reference implementation is an independent model and does not copy upstream source.

## State machine

Each protected record binds:
- one hidden/non-selected blocker literal;
- one target clause;
- the complete set of opposing clauses whose resolvents are non-tautological;
- a sentinel watch on the first candidate in that set.

When a watched clause becomes inactive, D4 either relocates the sentinel to another active opposing clause or marks the target blocked when no active opposing clause remains. The sentinel stays parked on the inactive clause that triggered the transition. Backtracking restores clause activity and assignments, not the watch move.

The executable checks the active-target/sentinel invariant:
if a target is active and its blocker variable is unassigned, its current sentinel must be active.

## Fixture

Hidden h=1; selected/visible a=2,b=3,c=4.

- C1 = (h OR a)
- C2 = (NOT h OR b)
- C3 = (NOT h OR NOT a OR c)

C3 is initially blocked. C1 watches C2 via blocker h. C2 watches C1 via blocker NOT h. At b=true, C2 is satisfied; its watch triggers and C1 becomes dynamically blocked. On backtrack, C1/C2 reactivate but watches remain parked. The b=false sibling does not inherit the b=true deletion.

| Scope | Source full models | Processed full models | Projected support | PWMC |
|---|---:|---:|---:|---:|
| Root | 7 | 8 | 6 | 5/6 |
| b=true | 5 | 8 | 4 | 1/2 |
| b=false | 2 | 2 | 2 | 1/3 |

The two branch supports union to the root support. Full-model count is not an invariant of projected BCE; visible support and PWMC are.

## Reproduction

Run python3 verify_d4v2_dynamic_bce_trace.py. It verifies the certificate and writes d4v2_dynamic_bce_trace_results.v1.json. The runner computes its own SHA-256 and Git blob SHA-1.

## Deliberate exclusions

This first model does not reproduce packed occurrence arrays, clause literal watcher ordering, binary-clause optimizations, parser/selected-variable import semantics, all D4 solver assignment events, or runtime performance. The mapping of D4 selected variables to projected/visible support must be confirmed per frontend. The next gate is differential trace comparison with a pinned unchanged D4v2 build, then HOL4 refinement.