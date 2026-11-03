# Outcome-qualified service resource accounting

## Purpose

The service module reports resources per unique service item that receives every required endpoint PASS in one completed physical run. It makes failed cycles, interrupted cycles and rewashes visible in the resource numerator instead of allowing them to disappear from an efficiency calculation.

This is a ledger calculation, not an assay, controller, cleaning model, certification or statistical guarantee.

## Release denominator

For one stable service-request ID, service-item ID i, run r, and caller-declared required endpoint set E, define:

Q(i,r) = AND over e in E of [D(i,e,r) = PASS]

where each decision must be derived from an independent outcome assay bound to that physical run. An absent record is unknown and does not satisfy the conjunction. FAIL and INDETERMINATE do not qualify.

An item is counted once when a single non-interrupted attempt qualifies it. If an item fails, is interrupted, or lacks an endpoint result, it may appear in a later attempt under the same service-item ID. Every attempt's resources remain in the total. An item may not be attempted again after it has already qualified; a later use request must receive a new service-item ID.

The service-level resource ratio is:

R_per_verified_item = sum(resources for all attempts) / number of unique items that qualified

The report also identifies `first_pass_verified_item_ids` and `first_pass_verified_item_count`: an item is first-pass only if it passes every required endpoint on its first physical attempt. Passing after a failure, interruption or incomplete first attempt remains a verified success but is not a first-pass success. This distinction feeds system-window first-pass yield.

The report also includes `rewash_attempt_count`: the number of physical attempts containing at least one item that has appeared in an earlier attempt within the same service request. This differs from `reattempted_item_visits`, which counts each item-level revisit. One rewash attempt can include several reattempted items; several rewash attempts can revisit one item repeatedly.

If no item qualifies, the per-item ratio is absent rather than zero or infinity. Report total resources and unqualified IDs separately.

## Resource boundaries

Each attempt records the following independently:

- appliance electricity at the declared wall-meter boundary (kWh);
- upstream hot-water energy (kWh);
- fresh water (L);
- actual detergent delivered (g);
- actual rinse-aid delivered (g);
- active operator time (s).

Do not add upstream hot-water energy to appliance wall energy if the wall meter already includes the water heater. Include preheat, purge and standby overhead in the appropriate attempt/service window. If overhead allocation across attempts is used, document the allocation rule and keep a service-window total as the controlling result.

Attempt timestamps use one monotonic clock. The service makespan is maximum finish minus minimum start, so parallel attempts are not summed as if they ran serially. The current module does not calculate per-item arrival latency or queueing; those remain scheduler/service-trace outputs.

## Data and evidence requirements

- One physical run ID per attempt; rewashes require new run IDs. Every item/endpoint decision must carry the same run ID as its attempt.
- Each attempt carries separate resource and outcome archive references plus SHA-256 digests. Each outcome carries its item, endpoint, criterion, method, protocol and source-record reference.
- These references/digests support traceability only: they do not authenticate the file, prove the contents or establish that an assay is independent/valid. Verify the archive and endpoint record through the upstream evidence pipeline before building the ledger.
- Sequence numbers are strictly increasing in the order supplied to the accounting function.
- Every attempt in one accounting call must carry the same nonblank `service_request_id`; mixed request IDs are rejected. Use one accounting call per requested service.
- Item IDs identify individual items within that service request, not a reusable item category such as glass or plate.
- Every outcome record carries a `run_id` that must exactly match the containing attempt, plus an item and endpoint identity. Criterion, method, protocol and evidence-reference binding must be validated upstream against the qualified outcome-evidence record.
- Only IndependentOutcomeAssay is accepted. Turbidity, conductivity, camera confidence, temperature, commanded dosing and model predictions must not be submitted as outcome decisions.
- The module checks that a required endpoint has a PASS record, that its run ID matches the attempt, and that criterion/method/protocol/evidence references are nonblank. It does not verify their contents, authenticate the assay, validate its uncertainty, prove physical independence or verify scientific quality. Perform the existing outcome qualification and run/derived-record lineage audits before constructing an accepted ledger.
- An interrupted run never releases an item, even where partial observations look favourable.

## Illustrative arithmetic (not measured performance)

Suppose one cup is attempted twice under one service-request ID. The first attempt fails soil removal and consumes 0.04 kWh appliance electricity, 0.8 L fresh water and 0.5 g rinse aid. The second attempt passes all required endpoints and consumes 0.05 kWh, 0.7 L and another 0.5 g rinse aid.

The service report therefore has one service-request ID, one unique requested item, one verified successful item, two physical runs, one reattempted item visit, 0.09 kWh appliance electricity, 1.5 L fresh water and 1.0 g rinse aid per verified successful item. Upstream hot-water energy, detergent and active operator time remain separately reported.

This is the intended accounting behaviour; the numbers are explanatory fixtures, not an InstantClean result.

## Deliberate limitations

- A caller can incorrectly label an assay as independent; this module cannot detect that.
- A PASS does not imply sanitization, disinfection or sterilization.
- A one-item success denominator does not imply independence across physical runs.
- The function does not infer actual delivered resources from commanded values.
- Missing/invalid endpoint results remain unqualified; the module does not impute outcomes.
- The ledger does not compare alternatives or select a Pareto-optimal process.

## Tests authored

The module includes 14 source-authored tests for service-request isolation, rinse-aid accounting, complete endpoint qualification, failed-attempt rewash accounting, missing/indeterminate endpoint rejection, per-item outcomes within a multi-item run, interrupted-attempt resource inclusion, proxy-evidence rejection, run-ID mismatch, archive reference/digest validation, evidence-reference validation, duplicate outcomes/run IDs, invalid resource and sequence values, and reattempt after qualification.

They are source-authored tests only until Cargo runs them on the exact commit.