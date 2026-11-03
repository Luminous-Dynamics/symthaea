# Outcome-qualified service resource accounting

## Purpose

The service module reports resources per unique service item that receives every required endpoint PASS in one completed physical run. It makes failed cycles, interrupted cycles and rewashes visible in the resource numerator instead of allowing them to disappear from an efficiency calculation.

This is a ledger calculation, not an assay, controller, cleaning model, certification or statistical guarantee.

## Release denominator

For a service-item ID i, run r, and caller-declared required endpoint set E, define:

Q(i,r) = AND over e in E of [D(i,e,r) = PASS]

where each decision must be derived from an independent outcome assay bound to that physical run. An absent record is unknown and does not satisfy the conjunction. FAIL and INDETERMINATE do not qualify.

An item is counted once when a single non-interrupted attempt qualifies it. If an item fails, is interrupted, or lacks an endpoint result, it may appear in a later attempt under the same service-item ID. Every attempt's resources remain in the total. An item may not be attempted again after it has already qualified; a later use request must receive a new service-item ID.

The service-level resource ratio is:

R_per_verified_item = sum(resources for all attempts) / number of unique items that qualified

If no item qualifies, the per-item ratio is absent rather than zero or infinity. Report total resources and unqualified IDs separately.

## Resource boundaries

Each attempt records the following independently:

- appliance electricity at the declared wall-meter boundary (kWh);
- upstream hot-water energy (kWh);
- fresh water (L);
- actual detergent delivered (g);
- active operator time (s).

Do not add upstream hot-water energy to appliance wall energy if the wall meter already includes the water heater. Include preheat, purge and standby overhead in the appropriate attempt/service window. If overhead allocation across attempts is used, document the allocation rule and keep a service-window total as the controlling result.

Attempt timestamps use one monotonic clock. The service makespan is maximum finish minus minimum start, so parallel attempts are not summed as if they ran serially. The current module does not calculate per-item arrival latency or queueing; those remain scheduler/service-trace outputs.

## Data and evidence requirements

- One physical run ID per attempt; rewashes require new run IDs.
- Sequence numbers are strictly increasing in the order supplied to the accounting function.
- Item IDs identify one requested service, not a reusable item category such as glass or plate.
- Every outcome record is bound externally to the exact run, item, criterion, method and protocol.
- Only IndependentOutcomeAssay is accepted. Turbidity, conductivity, camera confidence, temperature, commanded dosing and model predictions must not be submitted as outcome decisions.
- The module checks that a required endpoint has a PASS record; it does not authenticate the assay, validate its uncertainty, prove physical independence or verify the scientific quality of the evidence. Perform the existing outcome qualification and run/derived-record lineage audits before constructing an accepted ledger.
- An interrupted run never releases an item, even where partial observations look favourable.

## Illustrative arithmetic (not measured performance)

Suppose one cup is attempted twice. The first attempt fails soil removal and consumes 0.04 kWh appliance electricity and 0.8 L fresh water. The second attempt passes all required endpoints and consumes 0.05 kWh and 0.7 L.

The service report therefore has one unique requested item, one verified successful item, two physical runs, one reattempted item visit, 0.09 kWh appliance electricity and 1.5 L fresh water per verified successful item. Upstream hot-water energy, detergent and active operator time remain separately reported.

This is the intended accounting behaviour; the numbers are explanatory fixtures, not an InstantClean result.

## Deliberate limitations

- A caller can incorrectly label an assay as independent; this module cannot detect that.
- A PASS does not imply sanitization, disinfection or sterilization.
- A one-item success denominator does not imply independence across physical runs.
- The function does not infer actual delivered resources from commanded values.
- Missing/invalid endpoint results remain unqualified; the module does not impute outcomes.
- The ledger does not compare alternatives or select a Pareto-optimal process.

## Tests authored

The module includes tests for complete endpoint qualification, failed-attempt rewash accounting, missing/indeterminate endpoint rejection, per-item outcomes within a multi-item run, interrupted-attempt resource inclusion, proxy-evidence rejection, duplicate outcomes/run IDs, invalid resource and sequence values, and reattempt after qualification.

They are source-authored tests only until Cargo runs them on the exact commit.