# PIE-009L partially observed black-start oracle evidence

Date: 2026-09-12

## Scope

This note records the independent synthetic reference semantics in `scripts/pie-009l-partially-observed-black-start-oracle.py`.

The oracle is intentionally structural and non-probabilistic. It does not claim lunar or Martian sensor reliability, electrical-network behavior, mission safety, hardware qualification, or optimal recovery performance.

## Independent execution

The final candidate self-test was executed locally with Python 3 on 2026-09-12 and returned:

`ok`

No Symthaea module is imported by the oracle.

## Frozen semantics

The reference establishes:

- the planner receives an explicit `BeliefState`, not the hidden true world;
- no probability distribution is invented over hidden states;
- an action is admissible only if it is safe in every hidden world still compatible with the belief;
- unresolved equipment health or topology therefore triggers diagnosis, waiting, or `Blocked` rather than a guessed start;
- failed sensors return `Unknown` and do not reduce uncertainty;
- contradictory observations fail closed instead of fabricating a new hidden state;
- observations older than the current topology version fail closed;
- planned actions carry an expected topology version;
- topology-version drift makes stale start/topology actions inadmissible;
- the planner emits one action and then requires re-observation/replanning rather than relying on an open-loop future sequence.

## Executed synthetic fixtures

The final self-test demonstrates:

1. uncertain generator health produces `probe_generator`, not `start_generator`;
2. a healthy generator measurement collapses the belief and permits generator start;
3. a failed generator sensor returns `Unknown`, leaves the belief unchanged, and cannot create false certainty;
4. uncertain intertie state produces `probe_tie` before water restoration;
5. a known-open intertie produces `close_tie` rather than a water start;
6. a known-closed intertie permits water start;
7. topology-version drift invalidates an otherwise structurally valid water-start action;
8. stale observations fail closed;
9. observations contradicting every admissible hidden world fail closed;
10. observationally identical belief states produce identical deterministic actions regardless of hidden scenario labeling/order.

## Important limitations

This oracle does not model startup energy, startup consumables, power-flow transients, protection/relay behavior, continuous state estimation, probabilistic sensor error, or physical sensor dynamics. Those belong in separate evidence lines. PIE-009J remains the startup-energy/islanding reference, PIE-009K remains the structural criticality reference, PIE-009H remains the non-anticipative policy reference, and PIE-009I remains the communications/authority reference.

The hidden-world set is finite and synthetic. A later production implementation may use richer symbolic or constraint-based uncertainty while preserving the same fail-closed theorem: **never execute an action that is unsafe in any still-admissible world**.

## Promotion boundary

The intended promotion path is:

`independent belief-state oracle -> production partial-observation semantics -> cross-check with PIE-009J/K/H/I -> synthetic black-start recovery under sensor/topology faults -> evidence-bounded Moon/Mars campaigns`

Tracks #1968, #1924, #1932, #1847, #1852, #1647 and master #1604.
