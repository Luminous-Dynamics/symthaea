# Sailcraft COLREG / runtime-assurance action admissibility v0.1

**Owner:** SAIL-COLREG-RTA-001 #12386  
**Parent:** MARINE-SAIL-AUTONOMY-001 #12153 and #12158  
**Status:** synthetic software contract; no physical or regulatory claim  
**Version:** 0.1.0

## Purpose

This module is a small domain adapter for one question:

> Does a proposed sailing maneuver remain admissible under the exact evidence, model, ODD, external encounter/rule assessment, and pre-existing authority ceiling supplied for this evaluation?

It is not a COLREG interpreter, radar/AIS pipeline, target-motion predictor, sailboat dynamics solver, autopilot, runtime controller, or operational authorization service.

The module consumes external assessment results. This is deliberate: COLREG applicability and encounter classification remain with the dedicated maritime benchmark/interpretation owner; vessel ODD remains with the ODD owner; generic runtime-assurance and authority monotonicity remain with SE-RTA.

## Input contract

Every required evidence reference binds:
- a source SHA-256 digest;
- an explicit evidence state;
- own-vessel identity and exact configuration digest;
- target-track identity and encounter/scenario identity;
- a common qualified timebase identity;
- observation age and maximum permitted age.

The evaluator requires evidence references for the route candidate, encounter/rule assessment, ODD assessment, model-feasibility assessment and the authority ceiling. Each reference must match the same own-vessel, target-track, configuration, scenario and timebase scope; a digest from another rig, vessel, target track or encounter is not reusable. Human readiness evidence is additionally required when rule applicability is ambiguous.

A digest identifies an artifact; it does not authenticate its publisher, prove the underlying physical measurement, or establish independence.

### External assessments

- **Rule assessment:** `ConsistentWithCandidate`, `ConflictsWithCandidate`, `Ambiguous`, or `Unknown`. The module does not derive this value or interpret Rule 12/13.
- **ODD assessment:** within the declared vessel domain, outside it, or unknown.
- **Model assessment:** feasible, infeasible, outside the model's applicability domain, or unknown.
- **Authority ceiling:** an explicit set of maneuver classes already admitted by an independent authority source. The candidate must be a member; this module cannot add one.
- **Human oversight:** a readiness assessment backed by its own current evidence. Ambiguity routes to a ready human or returns `FallbackRequired`.

## Dispositions

| Disposition | Meaning |
|---|---|
| `AdmissibleWithinDeclaredModel` | All supplied evidence and assessments admit the candidate within the declared synthetic contract |
| `RejectedModelInfeasible` | The named qualified model reports the maneuver infeasible |
| `RejectedRuleConflict` | The external encounter/rule assessment conflicts with the candidate |
| `AmbiguousRequiresHuman` | Rule applicability is ambiguous and a ready, qualified human is available for adjudication |
| `NotEvaluatedMissingEvidence` | Required evidence is missing, unverified, malformed, or on a different timebase |
| `NotEvaluatedStaleEvidence` | Required evidence exceeds its declared freshness bound |
| `ConflictedInput` | A required evidence source reports a conflict |
| `OutOfScope` | The candidate is outside the vessel ODD or model applicability domain |
| `FallbackRequired` | A required qualification/authority reference is revoked, or ambiguity has no ready qualified human |
| `RejectedAuthorityCeiling` | Candidate maneuver class is not in the independently supplied authority set |

These are analysis dispositions, not executable commands or grants of authority.

## Precedence

The deterministic P0 evaluator checks:
1. required evidence integrity, state, freshness, and common timebase;
2. candidate bearing validity;
3. vessel ODD;
4. model feasibility/applicability;
5. externally supplied encounter/rule disposition;
6. pre-existing authority ceiling;
7. admissibility within the declared model.

An unsupported or unknown prerequisite is never interpreted as “clear.” An ambiguous rule assessment is not converted into a guessed maneuver. A model-feasible action cannot override the ODD, rule assessment or authority ceiling.

## Authority boundary

The module returns only a disposition and a reason. It does not return a helm angle, sail-sheet setting, reef/furl command, waypoint change, communications message or mode transition.

For every disposition:
- `actuator_command_emitted == false`;
- `authority_expanded == false`.

A result of `AdmissibleWithinDeclaredModel` is only a software admission result relative to the supplied synthetic references. It is not a navigation safety assertion.

## Frozen tests

Twenty-seven Rust unit tests cover:
- nominal evidence-qualified candidate;
- missing, stale, conflicted and revoked evidence;
- common-timebase mismatch and freshness expiry;
- ODD outside/unknown;
- model infeasible/out-of-domain/unknown;
- external rule conflict/ambiguity/unknown;
- ready, unavailable, stale or missing human oversight;
- authority-ceiling rejection;
- malformed and non-finite course bearings;
- invariant that no disposition emits commands or expands authority.

## Comparison programme

The next benchmark phase should compare, on identical encounter and vessel-model inputs:
- rule/encounter state machine + polar/VMG planner;
- COLREG state/action constraints + classical MPC;
- reward-shaped SAC-PopArt;
- reward shaping + MPC shield;
- temporal-logic falsification-driven RL;
- Symthaea planner + this independent action-admissibility layer;
- Symthaea HDC/LTC + active sensing + counterexample-guided updates.

A current comparator is Deng et al., *COLREGs-compliant unmanned sailboat navigation via safe reinforcement learning*, Ocean Engineering 368 (2026), article 128187, published 15 November 2026, DOI [10.1016/j.oceaneng.2026.128187](https://doi.org/10.1016/j.oceaneng.2026.128187). It combines wind-dependent Rule 12, risk-aware radar/TCPA/DCPA, SAC-PopArt, reward shaping and model-predictive shielding in a simulated 4-DOF MMG sailboat. Its simulation results are a comparator, not physical or regulatory qualification.

Report collision/near-miss, rule applicability/violation, minimum separation, sailing feasibility, false-safe rate, abstention, shield intervention quality, temporal robustness, p50/p95/p99 monitor latency, ODD/shift resilience and prior-skill retention separately. Do not collapse them to one reward or route-time score.

## Claim ceiling

A passing test suite establishes only deterministic behavior of this synthetic input contract. It does not establish COLREG compliance, collision-avoidance safety, vessel dynamics validity, seaworthiness, MASS Code compliance, class/flag approval, autonomous navigation competence, or permission to operate.
