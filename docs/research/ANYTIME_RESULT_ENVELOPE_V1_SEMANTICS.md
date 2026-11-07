# Evidence-qualified anytime result envelope v1 — normative semantics

Status: research contract / implementation target. The JSON Schema is structural validation only. This document specifies the additional semantic checks required before a result may be admitted by the Luminous platform.

## 1. Claim dimensions

Do not collapse these statements:

- **Feasible incumbent:** the exact witness is complete and an independent domain validator accepts it against the original request.
- **Search completeness:** the declared finite search was exhausted, or a typed stopping reason and unresolved frontier remain.
- **Bound:** an independently checked lower or upper bound applies to the same exact request and objective.
- **Optimality:** a valid incumbent and valid opposing bound/proof establish the declared optimum under the declared tie policy.
- **Infeasibility:** exhaustive accepted search or an independent infeasibility certificate establishes no feasible witness exists for that exact request.

A witness validator does not establish source truth, authority to actuate, policy approval, operational safety, or optimality. A search-completion claim does not establish model fidelity. A digest is an identity commitment, not a correctness proof.

## 2. Immutable binding tuple

Every incumbent, bound, completion certificate, infeasibility certificate and optimality certificate must refer to one immutable tuple:

`(schema_version, contract_digest, request_digest, objective_digest, source_data_digest)`

The objective digest binds the objective sense, exact arithmetic domain, canonical encoding, units and tie policy. If no objective exists, use a separately versioned feasibility-result contract; do not smuggle a null objective into an optimization certificate.

Evidence for a different request, source-data revision, contract version, objective, or units cannot be reused. A mismatch is a hard rejection, not a warning.

## 3. Exact objective comparison

Only compare objective values using the declared exact domain:

- integer: canonical signed decimal integer, parsed with checked arbitrary precision or checked target-width arithmetic;
- rational: canonical numerator/denominator with positive denominator, reduced form, nonzero denominator;
- exact decimal: canonical coefficient and scale or a formally specified decimal representation;
- vector: each component's domain, order and unit must be declared;
- problem-defined: requires a separately versioned comparator and semantics identifier.

Never parse exact objective values through binary floating point.

For minimization, a feasible incumbent is an upper bound (U); an admissible lower bound (L) must satisfy (L le U).

For maximization, a feasible incumbent is a lower bound (L); an admissible upper bound (U) must satisfy (L le U).

The bound direction is part of the evidence. A numeric bound without the right relaxation theorem, model binding and checker is not an admissible bound.

An objective gap may only be reported when both endpoints are independently accepted for the same immutable tuple. Gap display conventions must define the denominator and zero-crossing behavior. A solver's heuristic estimate is not a certified bound.

## 4. Incumbent checkpoint stream

For a fixed request and fixed objective:

- Minimize: accepted incumbent values are non-increasing.
- Maximize: accepted incumbent values are non-decreasing.
- Timestamps are strictly increasing, or checkpoints carry an explicit deterministic sequence index that is strictly increasing.
- Every checkpoint contains a complete witness and its own independent validator receipt.
- A later partial, stale or rejected witness cannot overwrite an earlier accepted incumbent.
- A candidate may be retained after timeout, cancellation, worker crash or proof-generation failure, but the result remains feasible-unproven unless stronger evidence is admitted.
- Time-to-first-candidate, time-to-first independently verified candidate, time-to-first bound and time-to-proof are distinct measurements.

A worse candidate may be stored as a diagnostic attempt, but it must not be represented as a new incumbent in the monotone accepted-incumbent sequence. Keep rejected candidate attempts outside that sequence.

## 5. Disposition invariants

| Disposition | Required claim | Forbidden inference |
|---|---|---|
| exact_complete | Complete declared search + accepted incumbent + accepted bound/optimality evidence + tie policy resolved | Not a real-world safety/authority claim |
| feasible_unproven | At least one complete, independently accepted incumbent; search is incomplete/unknown | Not optimal, not exhaustive |
| proven_infeasible | No admitted incumbent; either accepted exhaustive completion or independent infeasibility certificate | Timeout is not infeasibility |
| bound_only | At least one independently accepted bound, no admitted incumbent | Not a feasible plan |
| limit_without_candidate | No admitted incumbent; an explicit resource/deadline stop | Not infeasible |
| unknown | Evidence insufficient to strengthen another disposition | No implicit success |
| input_rejected / unsupported / error | Explicit non-solving disposition | Must not map to UNSAT or infeasible |

For optimization v1, exact_complete needs a declared objective and an admitted optimum. Feasibility-only exhaustive search must use a separate, explicit contract version/disposition.

A proven_infeasible result cannot coexist with any accepted incumbent for the same immutable request. An exact_complete result cannot coexist with unresolved search obligations or an unaccepted completion checker.

## 6. Completion and pruning

Exhaustive completion requires:
- a precise finite search domain and canonical request binding;
- a completion artifact or trusted finite-domain procedure;
- independent acceptance of that evidence;
- completed stopping reason;
- zero unresolved frontier items (where the algorithm exposes a frontier);
- a valid accounting of all pruned subproblems.

Every pruned node must be covered by an admissible bound or a proved infeasibility condition. Equal-bound candidates must remain open when tie multiplicity or canonical tie selection could change the required output. Strict pruning rules must be tied to the declared objective sense and tie policy.

Do not accept a producer's "queue empty" boolean as an exhaustive proof. The completion checker must validate the tree/log/finite enumeration boundary.

## 7. State transitions

A checkpoint stream may strengthen its disposition only when new evidence is added:

- limit_without_candidate -> feasible_unproven: a complete witness is independently accepted.
- bound_only -> feasible_unproven: a complete witness is independently accepted, retaining the bound if still applicable.
- feasible_unproven -> exact_complete: completion and optimality evidence are independently accepted and the tie policy is discharged.
- unknown -> any stronger status: only when the additional required evidence is admitted.
- Any nonterminal status -> error/input_rejected only if a later integrity failure invalidates the stream; retain prior artifacts and record a revocation/rejection event rather than silently rewriting history.

Forbidden transitions:
- timeout/resource exhaustion -> proven_infeasible;
- solver-reported UNSAT without an accepted proof/completion path -> proven_infeasible;
- feasible_unproven -> exact_complete because the objective gap is small or zero in unverified telemetry;
- proven_infeasible after an accepted incumbent for the same immutable request;
- exact_complete after request/objective/source digest changes.

Stream transitions must be versioned separately from the single-result v1 JSON Schema.

## 8. Cross-project adapters

- Mycelix #9425: keep the proof-packet model, incumbent and independent lower-bound certificate native to Mycelix; adapter binds them to the shared request/objective digest. The shared envelope must not rebuild that optimization model.
- Symtropy #2608: preserve its actual clean/complete board/route candidate and domain validator; a clean route is not a global optimum.
- Sol-Atlas #417: preserve exact request/schedule bytes and the standalone schedule-auditor result; a feasible energy plan is distinct from service-floor qualification and optimality.
- Symthaea #10810: SAT decision outcomes stay separate; this optimization envelope is not a replacement for SAT/UNSAT/UNKNOWN proof semantics.

## 9. Known-answer fixtures

The accompanying `ANYTIME_RESULT_ENVELOPE_V1.examples.json` provides:
- feasible incumbent retained after a deadline, with optimality unproven;
- exhaustively completed search with accepted optimum evidence;
- independent infeasibility proof despite incomplete search.

The independent Rust semantic verifier issue is [#10887](https://github.com/Luminous-Dynamics/symthaea/issues/10887). Its negative corpus must include request mismatch, objective mismatch, wrong bound direction, incumbent regression, stale proof, unresolved frontier, timeout-as-infeasible and tie-policy violations.

## 10. Claim ceiling

The schema/fixtures and this document specify a contract; they are not a deployed verifier, solver result, formal proof, or SOTA performance result. Structural schema validation does not validate mathematics. Production admission requires an independently implemented verifier and independently checked domain evidence.
