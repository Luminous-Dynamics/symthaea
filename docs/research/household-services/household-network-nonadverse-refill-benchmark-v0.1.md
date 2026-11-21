# Household network-nonadverse backup refill benchmark v0.1

**Status:** design and synthetic challenge corpus; no physical feeder experiment  
**Owner:** [WATER-REFILL-COORD-001 #10517](https://github.com/Luminous-Dynamics/symthaea/issues/10517)  
**Parent:** [HOME-UTILITY-COORD-001 #10342](https://github.com/Luminous-Dynamics/symthaea/issues/10342)  
**Compose:** household continuity #10445/#10446, WaterLoop comparator #10483/#10485, source-to-use qualification #10500, arbitration evaluator #10516, lifecycle/reliability and metrology #10341  
**Relevant deterministic replay work:** [Symtropy #3543 — branch-stable exogenous randomness for counterfactual replay](https://github.com/Luminous-Dynamics/symtropy/issues/3543)

## 1. Paradigm: resilience without exporting the cost to neighbors

The first-order WaterLoop objective cannot be "fill the largest tank as fast as possible." In a constrained network, multiple household buffers can create correlated demand at exactly the moment a reservoir, tower or pumping system is trying to recover.

Johannesburg Water reported on 29 October 2026 that high demand and excessive draw-off were limiting reservoir and tower recovery, producing low pressure, intermittent supply and prolonged outages. The utility described targeted throttling and network flow/pressure reconfiguration as part of its response. This makes refill externalities a credible research question. It does **not** establish that JoJo or any competitor's current installation is causing these conditions.

Source: https://www.johannesburgwater.co.za/blog/2026/10/29/johannesburg-water-implements-urgent-interventions-to-stabilise-severely-constrained-systems/

The paradigm shift is from **household autonomy by inventory** to **household resilience under a shared-source constraint**. A backup system should improve the household's critical service while remaining within its approved connection envelope and not worsening coincident shared-network draw unnecessarily.

## 2. Four comparator policies

Freeze the same household cohort, initial reserves, demand traces, source-pressure trace, pump curve, tank usable capacity and outage/restoration timeline across all policies.

| Policy | Definition | Role |
|---|---|---|
| P0 — Immediate synchronized refill | All buffers refill as soon as supply is detected | Adversarial but operationally common reference; not assumed to describe any specific vendor |
| P1 — Local envelope-limited refill | Each local controller enforces its approved inlet pressure/flow envelope and rate limit | Minimum safety-aware comparator |
| P2 — Decentralized staggered refill | Each device chooses a bounded refill window from local reserve urgency and replayable exogenous jitter; local controller can refuse | Candidate network-nonadverse design |
| P3 — Utility-signal-aware refill | Uses an authorized utility signal or service window, if one exists and its semantics are documented | Optional upper-bound comparator; never assume access or permission |

P2 must not depend on a central household coordinator, cloud account, shared private identity graph or neighbor telemetry. A device may compute a bounded delay locally, preserve a locally approved reserve floor, and disclose a minimal auditable receipt. Loss of network or coordinator communication falls back to the local safe policy.

P3 is only a benchmark arm when a competent utility has authorized the interface and its signal scope, freshness, priority, failure behavior and security have been documented. A public outage dashboard or consumer status feed is not a refill authorization signal.

## 3. Bounded multi-household reference rig

Start with a synthetic feeder and then a controlled physical rig reviewed by qualified plumbing/engineering professionals:
- shared source/reservoir surrogate with controllable upstream pressure and flow envelope;
- one measured common upstream boundary and separately metered household branches;
- 8-, 16- and 32-household cohort sizes as proposed scale points;
- heterogeneous usable reserves, household demand, elevation/fixture needs, tank sizes and pump curves;
- municipal interruption, partial restoration, targeted throttling, pressure rebound/collapse and source return traces;
- a declared critical reserve/service floor and use-qualified water class for each household;
- independently calibrated upstream and branch pressure/flow instrumentation.

The household controller remains local. The bench harness may orchestrate disturbances and capture outcomes, but it cannot issue field actuation commands. Physical valves/pumps must remain under approved local controls and safety interlocks.

## 4. Non-compensatory refill policy

A candidate refill request is eligible only when:
1. source and local connection state are known and within the approved installation envelope;
2. the required flow/pressure is present and measurement uncertainty does not cross the stop threshold;
3. the household's protected service reserve remains within its declared floor under the proposed delay;
4. the action is within current local authorization and installation policy;
5. local safety interlocks and controller capability revision accept the request.

A lower network peak cannot compensate for a household falling below its protected reserve floor. Conversely, a household's desire to refill cannot authorize a draw that violates the approved inlet envelope or local authority constraints.

Unknown pressure/flow state means no optimistic refill. Use hysteresis and bounded dwell time to avoid pump/valve chatter, but never invent a universal pressure threshold; use the approved exact installation envelope.

## 5. Replayability and fairness

The staggered policy must be reproducible. Use branch-stable exogenous randomness so counterfactual policies see the same underlying random variates and household demand/outage traces. Do not derive random refill priority from a mutable branch-local sequence whose consumption changes when policy logic changes. See Symtropy #3543 for the related replay contract.

Fairness is not one weighted score. Report:
- each household's critical-floor breach minutes and interval-valued cumulative service deficit;
- distribution of refill start delay and refill completion time;
- worst household and upper-tail delay/deficit, not just the mean;
- how many times each household is delayed or receives a priority slot across repeated traces;
- reserve-urgency ordering and any authorized user override;
- whether device identity, clock skew, ordering or seed changes create systematic advantage.

A max-min or leximin allocation may be explored after hard service/safety gates. Do not claim equity merely because the cohort average improved.

## 6. Benchmark outcomes

Record independently at both the shared upstream boundary and every household branch:
- coincident peak refill draw (L/min) and peak-to-average ratio;
- upstream pressure deviation and time below the approved envelope;
- per-branch inlet flow/pressure and any local limiter/refusal;
- household critical reserve-floor breach minutes;
- service deficit by household, quantile and worst case;
- refill delay, total completion time and reserve urgency;
- pump/valve/controller energy and cycling;
- leak/overflow/rejected refill volume;
- local manual operation, override and controller refusal;
- coordinator/network dependency and recovery after comms/power loss;
- measurement uncertainty, calibration state and missing observations.

Do not infer shared-network relief from a household inlet meter alone. An upstream boundary measurement is required for a network-level peak claim.

## 7. Proposed target ladder

These are research aims, not industry facts, physical findings or regulatory thresholds:
- **Peak draw:** at least 20% reduction in cohort coincident peak refill flow versus P0 under matched traces.
- **Protected household service:** zero additional critical reserve-floor breach minutes attributable to staggering; no candidate can pass if it achieves lower peak only by leaving a household below its declared floor.
- **Envelope compliance:** zero refill requests when local pressure/flow is below or Unknown relative to the approved installation envelope.
- **Service non-inferiority:** no household's cumulative qualified-service deficit may worsen beyond a preregistered measurement/uncertainty margin; report the margin per endpoint, not one scalar.
- **Fairness:** no persistent household identity/order advantage under repeated branch-stable counterfactual seeds; bound maximum delay and report its distribution.
- **Offline resilience:** local limiter, stop conditions, manual controls and safe fallback continue when WAN/coordinator is unavailable, where hardware supports those functions.

Before any physical claim, preregister the non-inferiority margins, number of independent source/pressure traces and household branches from a variance-component pilot under #10341. Repeated cycles on one branch are not independent households.

## 8. Adversarial challenge corpus

The companion JSON includes RFL-001…014 plus positive controls RFL-POS-001…003. Cases include synchronized restoration, low pressure, misleading flow state, heterogeneous reserve urgency, targeted throttling, comms loss, pressure oscillation, identity/clock bias, unstable restoration, override, local refusal, common-mode power loss and unverified network-relief claims.

All cases are synthetic design fixtures. They do not constitute a feeder simulation result, product certification, installation approval, utility agreement or claim that an incumbent performs poorly.

## 9. Claim ceiling

No household refill controller, neighborhood coordination system, utility integration, physical network relief, water savings, resilience superiority, municipal permission, potable quality or safety approval is established by this document. A future physical rig requires qualified review, approved test boundaries, calibrated instrumentation and independent observation.
