# ETH-027 — projected blocked-clause elimination receipt v1

**Claim ceiling:** static BCE precondition verification plus exhaustive finite fixture oracle. This is not a formal proof checker, not a dynamic D4-style BCE trace verifier, and not a general preprocessing compiler.

## Prior art and research question

Dynamic Blocked Clause Elimination for Projected Model Counting is prior art (Lagniez, Marquis and Biere, SAT 2024; DOI: 10.4230/LIPIcs.SAT.2024.21). BCE itself is not a new Symthaea algorithm. This experiment tests a digest-bound, independently checkable receipt for one projected-safe deletion step, as a possible building block for a compositional preprocessing trace.

For clause C and literal l in C, C is blocked by l if every other clause D containing the complementary literal not-l has a tautological resolvent:

(C without l) union (D without not-l).

The fixture restricts l to a declared hidden/existential variable. The declared theorem profile is projected-support preservation, not full-model preservation.

## Teaching fixture

Variables: visible a, hidden h.

Source:
- c1 = (h OR a)
- c2 = (NOT h OR NOT a)

Delete c1 using blocking literal h. The only opposing clause is c2. Its resolvent is (a OR NOT a), with tautology witness a.

Target:
- c2 = (NOT h OR NOT a)

| Quantity | Source | Target |
|---|---:|---:|
| Full satisfying assignments | 2 | 3 |
| Projected visible support | {a=0, a=1} | {a=0, a=1} |
| PMC | 2 | 2 |
| PWMC with visible weights 1/3 and 2/3 | 1 | 1 |

The deletion changes full model count but preserves visible support. The checker verifies the deliberately narrow claim.

## Receipt checks

The reference verifier checks a duplicate-free visible/hidden partition, well-formed source/target clauses, canonical source and target SHA-256 digests, exact single-clause deletion, a hidden blocking literal, complete opposing-clause coverage, recomputed resolvents, and explicit tautology witnesses. It rejects a full-model-equivalence overclaim.

It then exhaustively enumerates the two-variable teaching fixture and checks visible support and exact visible-weighted value. Nine negative controls exercise malformed partitions, incomplete opposing-clause coverage, false witnesses, target edits, theorem overclaims, wrong blockers, and stale source/target digests.

## Reproduction

Run python3 verify_projected_bce_step.py from this directory. It writes projected_bce_step_results.v1.json.

## Limitations

- Not an implementation of dynamic BCE in D4 and not a new BCE algorithm.
- Not a formal proof of the BCE theorem in Lean, HOL4 or CakeML.
- Does not certify a sequence of preprocessing steps or partial-assignment contexts.
- Does not prove full #SAT/WMC preservation; this fixture explicitly falsifies that stronger claim.
- The exhaustive oracle is a tiny regression oracle, not a scalable verifier.

## Next research gate

Extend the receipt for dynamic inprocessing: bind parent formula digest, partial-assignment digest, conditioned formula digest, blocking clause/literal, complete opposing-clause set in that context, and resulting formula digest. Verify conditioning and transformation order, then compose receipts only when each target digest equals the next source digest. Differentially replay against unchanged D4 dynamic BCE. Only then call it a dynamic preprocessing trace certificate.
