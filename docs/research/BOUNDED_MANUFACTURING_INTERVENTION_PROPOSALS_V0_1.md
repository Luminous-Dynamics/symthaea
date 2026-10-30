# Bounded Manufacturing Intervention Proposals v0.1

**Status:** research interface only. Symthaea does not authorize or execute physical manufacturing actions.

## Role separation

Symthaea can propose what an operator or governed process gateway might do. It cannot grant authority to perform that action. Symtropy owns manufacturing route/test-plan/metrology constraints and the authority/evidence lifecycle. An independent policy gateway must be the only future machine-facing boundary.

This separation is deliberate. High information gain, a high-confidence prediction, or a model's notion of feasibility must never become physical authority.

## Proposal envelope

Each Symthaea proposal should be a versioned immutable record binding:
- product, serial, component/lot genealogy, equipment, station and operation;
- exact input snapshot digest and capture time;
- model version, training/evaluation provenance, declared domain and OOD status;
- predicted effect with calibrated uncertainty and relevant finite discrepancy scenarios;
- proposed parameter values, units, hard allowable envelope, rate/slew limits and duration;
- required preconditions, interlocks, stop conditions and max attempts;
- simulation and validation receipts, known failure modes and alternatives;
- proposal ID, revision, nonce request and expiry;
- explicit statement: `PROPOSED_NOT_AUTHORIZED_NOT_EXECUTABLE`.

The proposal digest must bind all the above fields. Any change to context, product, recipe, model, parameter or route creates a new proposal digest and invalidates earlier authorization.

## Mandatory abstention

Do not produce an actionable proposal when:
- the reference population does not cover the current lot/station/process regime;
- model OOD or model-discrepancy checks fail;
- input evidence is missing, stale, contradictory or untraceable;
- metrology is not current or a required acceptance/test limit is unknown;
- the proposed parameter is outside the supplied approved envelope;
- uncertainty includes unacceptable risk or no safe alternative is established;
- the required postcondition cannot be measured and independently verified.

A model cannot relax an acceptance limit, widen a recipe envelope or override a failed interlock.

## Interface with Symtropy

Symthaea sends an immutable proposal to Symtropy. Symtropy re-checks current product/lot, route, recipe/test-plan, calibration, constraints, policy revision and action validity. A separately governed authority signs approval over the exact proposal/context digest and one-use nonce. The independent policy gateway verifies signature, revocation, expiry, nonce freshness and interlocks immediately before execution.

After execution, machine telemetry and independent postcondition measurements are returned. Symthaea may analyze the outcome, but cannot mark an action authorized, executed or closed. Closure is a Symtropy evidence-state transition after all required postconditions and recovery obligations are satisfied.

## Benchmarks

Test proposals against:
- supported vs OOD lot, supplier, geometry, station, operator and operating regime;
- stale/mismatched snapshot, route, recipe and model versions;
- high predicted benefit but invalid parameter range;
- high EIG but high misspecification amplification risk;
- missing calibration, expired authority, nonce replay and duplicate action;
- failed interlock, controller/setpoint mismatch and partial execution;
- abort, failed postcondition, missing measurement and recovery requirement.

Report proposal validity, OOD detection, calibrated effect intervals, constraint-violation prediction, invalid proposal rate, abstention, false-safe proposal rate, postcondition-verification coverage, and cost/time to a verified decision. The system should optimize verified beneficial outcomes per cost, not actions issued.

Proposed safety gate: zero out-of-envelope, stale-context or unauthorized proposals accepted by Symtropy in the frozen negative-control suite; zero false CLOSED states without independent postcondition evidence.

## Research context

An October 2026 systematic review of 235 manufacturing QA digital-twin studies proposes an Embodiment-Authority-Assurance (EAA) framing and reports no reviewed study assembling the complete assured intervention chain. This makes authorization and post-action verification an important research frontier, not a reason to bypass industrial control governance. https://doi.org/10.1186/s44147-026-01295-x

NISTIR 8620 (2026) identifies interoperability, VVUQ, cybersecurity and workforce readiness as open manufacturing digital-twin barriers. https://doi.org/10.6028/NIST.IR.8620

## Non-goals

No PLC or robot commands, no direct process write access, no authorization signing, no safety interlock bypass, no autonomous product release. Those require separately validated and approved systems.
