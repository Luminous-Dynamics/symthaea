# Household service evaluation contract v0.1

**Status:** deterministic software contract and synthetic test harness. Not a physical qualification method, product specification, safety case or certification.

## 1. Separation of concerns

- Fixture integrity: tools/household-bench-fixtures checks the declared corpus structure and frozen adversarial cases.
- Evaluation: tools/household-service-evaluator performs bounded calculations and emits fail-closed dispositions from typed input evidence.
- Physical qualification: remains with the relevant product engineering, safety and metrology owners. It requires calibrated instruments, topology audit, independent outcome assays and production-representative hardware.
- Household validity: requires field exposure and household evidence; software tests cannot promote evidence to this level.

## 2. Water balance contract

The signed equation is:

    R = V_in - ΔV_tank - V_delivered - V_purge_overflow

ΔV_tank > 0 means tank storage increased during the observation window. Each measurement carries its own conservative absolute error bound (u_in, u_tank, u_delivered, u_purge). The evaluator derives the residual bound rather than accepting a caller-supplied aggregate:

    U_R = u_in + u_tank + u_delivered + u_purge
    residual interval = [R - U_R, R + U_R]

This is worst-case interval propagation. It assumes neither measurement independence nor a covariance model and therefore cannot silently shrink because a caller omitted a term. The measurement protocol must justify each individual bound. Zero inside the interval means only that the interval does not exclude zero. It is not proof of correct topology or physical closure.

All four terms must use the same observation window, distinct stream IDs and distinct declared boundary IDs. Calibration record and topology-audit digests are references only. Their bytes, signer authority and physical meaning must be independently checked.

Known-answer negative control from WCONT-008:

    R = 480 - 120 - 520 - 40 = -200 L
    The fixture provides meter_uncertainty_l = 5 L but does not say whether
    that bound is per meter or already aggregated. The integration negative
    control conservatively assigns 5 L to each of the four typed meter terms;
    the evaluator derives U_R = 4 × 5 L = 20 L:

    interval = [-220 L, -180 L]

The interval still excludes zero, so the declared values contradict closure under this conservative interpretation. The uncertainty-scope ambiguity remains a fixture-schema issue to resolve before any physical use. This fixture is synthetic; it does not size or qualify a tank, pump, meter or installation.

## 3. Deliverable reserve contract

The calculation accepts only usable volume at the service-delivery boundary and a bounded positive demand rate:

    hours ∈ [V_usable_lower / demand_upper,
             V_usable_upper / demand_lower]

It returns no duration if the pump/electrical path is unavailable, delivery flow/pressure is not qualified, measured pressure is below the declared service minimum, or the water is not independently qualified for the named end use. Gross tank inventory, level-sensor estimates, municipal restoration notices or a pump command are not substitutes for these inputs.

This is a conditional arithmetic range, not an outage forecast. It must not be used for potable/cooking claims without separate product- and use-specific qualification.

## 4. Repair outcome contract

The evaluator distinguishes:
- Blocked: diagnostic, part, compatibility or required offline-access prerequisite absent;
- HeldForRequalification: required calibration, functional or safety evidence incomplete;
- RecurrenceObserved: fault recurred after initial repair;
- InitiallyRestoredPendingFollowUp: initial gates passed, but the requested follow-up has not elapsed;
- NoRecurrenceObservedDuringFollowUp: no recurrence observed over the exact recorded interval.

No state supports a lifetime reliability claim. Longer-term claims require a preregistered duty profile, production-representative sample, failure/censoring model, exposure tracking and appropriate confidence bounds.

## 5. Benchmark protocol implications

For each physical campaign, freeze before testing:
- functional unit, workload and service envelope;
- exact comparator product revisions and configurations;
- observation window and resource boundaries;
- endpoint acceptance rules and independent assay method;
- uncertainty budget and calibration state;
- device, batch, operator, site and day identifiers;
- all retries, interventions, maintenance and right-censored outcomes;
- a locked confirmation subset not used for tuning.

Report quality-qualified service throughput, first-pass yield, rework, resource intensity, downtime, uncertainty and failures separately. Do not collapse quality, safety, energy, water, affordability and repairability into a single weighted score.

For InstantClean, compose existing request/system-window accounting rather than duplicating it. For WaterLoop, do not equate stored litres with deliverable service. For HomeRenew, do not equate immediate restoration with durable repair.

## 6. Claim ceiling

Passing this Rust test suite establishes only the deterministic behavior of the software on its declared inputs. It does not establish:
- correct or independent physical metering boundaries;
- water quality, potable safety, hygiene or backflow protection;
- pump sizing, hydraulic performance, pressure stability or plumbing compliance;
- actual household demand or outage duration;
- appliance cleaning performance or energy/water superiority;
- safe repair, spare-parts availability, regulatory compliance or lifetime reliability;
- production variation or real-household validity.


## 7. Batch evaluation contract

The offline batch evaluator accepts only:
- water corpus ID HOME-UTILITY-COORD-001-WATER-CONTINUITY-SYNTHETIC-001;
- repair corpus ID HOME-REPAIR-QUAL-001-SYNTHETIC-REPAIR-TASKS-001;
- schema version 0.1 and status synthetic_test_fixture_only;
- the exact ten known water scenario IDs and exact ten known repair task IDs.

Missing, duplicate, unknown, additional, or malformed cases fail the batch. A fixture's expected_disposition is copied into the receipt only as source metadata; evaluator output is selected independently from the scenario inputs. Deterministic metrics are derived from explicit synthetic assumptions and labelled accordingly.

The batch report is generated without a timestamp and orders receipts by frozen corpus order. It includes corpus IDs, counts, per-case dispositions, derived metrics, evidence notes, unresolved claims and the claim ceiling. Re-running the same evaluator version against identical fixture bytes must produce byte-identical JSON.

The report is a software-conformance artifact, not an installation result. Cases without measured pressure, calibrated outlet flow, usable delivery-boundary reserve, qualified end-use water, executed repair actions, post-repair checks or physical follow-up stay unresolved, blocked or held. The batch evaluator does not infer missing evidence from the expected fixture disposition.

CI uploads the generated JSON for traceability. An uploaded artifact proves only that the declared deterministic software path produced a report from the declared synthetic fixture inputs.
