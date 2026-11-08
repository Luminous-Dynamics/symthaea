# Marine service-continuity evaluator — V0.1

**Status:** typed Rust implementation and unit tests added on the draft branch; exact-head Cargo execution has not yet been verified in this environment. No physical vessel or reliability result is claimed.  
**Implementation issue:** [MARINE-SERVICE-CONTINUITY-EVALUATOR-001 #11126](https://github.com/Luminous-Dynamics/symthaea/issues/11126)  
**Benchmark parent:** [MARINE-SERVICE-CONTINUITY-001 #11115](https://github.com/Luminous-Dynamics/symthaea/pull/11115)

## 1. Why this module exists

The benchmark changes the unit of success from “component operational” or “vessel still running” to **essential service delivered against a declared floor**. Potable water, safe electrical distribution, propulsion, sanitation, refrigeration and communications remain separate service contracts. A surplus in one service must not compensate for a breach in another.

The evaluator lives in the existing, dependency-free `symthaea-operations-research` crate. It does not create a second maintenance system, service ontology, vessel model or fault simulator.

## 2. Metrics

For each service, the evaluator recomputes from the supplied timeline:

- minimum observed delivery rate;
- seconds below the owner-declared minimum floor;
- number and start time of contiguous floor-breach episodes;
- integrated shortfall in that service's own rate-unit-hours;
- first fault, repair start/completion, spare arrival, passing post-test and configuration-update timestamps;
- accepted-restoration timestamp;
- **Time to Accepted Service Restoration (TASR)** = accepted-restoration time minus first declared fault-observation time;
- rejected release attempts.

The shortfall integral is not summed across services with unlike units or criticality. A dashboard may display service-specific values side by side, but must not convert them into an opaque compensating score.

## 3. What qualifies as accepted restoration

An `authorized_release` event cannot raise the delivered rate unless all relevant conditions are met:

1. A matching maintenance task exists for the same service and component.
2. The declared maintenance procedure is approved.
3. Qualified personnel are recorded as present.
4. The spare is compatible and onboard, or its later arrival is evidenced before release.
5. Every required task test is listed as completed.
6. A repair-completed event follows the latest declared fault.
7. A passing post-test follows that repair and is later than any failed post-test.
8. A configuration-update event follows the passing post-test.
9. The task records configuration update and authorized release.
10. The release event itself reports a rate at or above the service floor.

An attempted release that fails any gate is counted as rejected and cannot promote the rate. A safety-gate failure is a separate global disposition: it produces `unsafe_not_authorized`, blocks restoration and is not misreported as an ordinary service-flow breach.

Conflicting service rates at the same timestamp, orphan service references, non-finite/negative rates, duplicate IDs and out-of-window events fail closed. Input event order is normalized before evaluation.

## 4. Synthetic golden cases

The unit tests encode the six synthetic situations from the parent corpus:

| Case | Expected independent result |
|---|---|
| Buffer holds potable-water service above its floor | Zero breach episodes; accepted release at 19,200 s; TASR 15,600 s from fault at 3,600 s |
| Water reserve exhausts at 86,400 s | Floor breach begins at 86,400 s; no accepted restoration |
| Common controller failure defeats redundant pumps | Floor breach begins at 1,800 s |
| Repair completes but post-test fails | Release rejected; no accepted restoration |
| Spare has the wrong revision | Release rejected; no accepted restoration |
| Safety gate fails | Unsafe/not-authorized disposition; not counted as a service-flow breach |

Additional controls cover simultaneous conflicting rates, orphan service, late event, release without a matching task, and event-order permutation.

These fixtures are synthetic contract tests, not observed failure rates or measured equipment data.

## 5. Industry comparison and claim ceiling

This is not a claim to invent planned maintenance or condition monitoring. ISO 23323:2021 already specifies minimum functions for software-based planned-maintenance systems, including maintenance scheduling, recording, reporting, analysing and optimizing onboard equipment maintenance. ISO 19030-1:2016 defines hull/propeller performance indicators for comparing a vessel with itself over time.

The differentiation hypothesis is the cross-system, mission-level acceptance chain: service floors + common dependencies + component/revision identity + spares + competent work + post-repair tests + configuration update + authorized restoration. It must be tested against real, permissioned, matched evidence before any superiority claim.

## 6. Explicit limitations

- The module accepts typed, normalized structs; it does not parse the JSON corpus or ingest XLSX files.
- An evidence reference is checked for presence only. This code does not authenticate receipts, verify sensor calibration, check timestamps against a signed source, or prove that an event happened.
- The task booleans and spare state are inputs. A production evidence adapter must bind them to revision-specific work orders, inventory identities, competence records, test reports and approval receipts.
- No hazard analysis, seaworthiness determination, statutory/class approval, potable-water certification, failure-probability estimate, MTBF, availability prediction or real-world MTTR is produced.
- No physical demonstrator has been run. Software correctness cannot establish vessel capability.

## 7. Next evidence gates

1. Compile and run the tests on the exact PR head; record the actual Rust toolchain and all workflow outcomes.
2. Add a separate JSON adapter/independent verifier that reads the frozen V0.1 schema and corpus without trusting producer-supplied expected fields.
3. Add multi-service and repeated-fault episodes, task-to-component dependency identity, and explicit event-time provenance.
4. Admit public shipyard source workbooks with raw hashes and independently reproduce their source objectives before any optimizer comparison.
5. Feed only measured, calibrated, rights-cleared physical service traces from the staged LOS-18 test programme into the same contract.

Skipped, queued, missing or stale CI evidence is not PASS.
