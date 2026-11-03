# InstantClean system-window accounting

**Status:** research accounting contract; not a certification, hygiene claim, or physical performance result.

## Why a second accounting level exists

The service ledger accounts one stable service request, including all physical attempts, failures and re-washes. Its report is item-outcome aware, but a multi-compartment device may process several service requests concurrently. Summing request durations would overstate elapsed time; summing per-request resource reports can double-count shared standby, preheat, fill, purge, or an aggregate meter duplicated by a branch meter.

This module therefore separates three things:

1. **Service qualification:** `service::account_service` determines which requested items passed every required independent endpoint in their physical attempt history. It also records which items passed on their first attempt.
2. **System-window operation:** `system_window::account_system_window` computes releases, request latency, queue wait, carry-in, right-censored requests and concurrency over one half-open wall-clock window.
3. **System resource measurement:** resource rows measured at authoritative system boundaries provide the window's energy, water, chemical and operator-time totals. Request-level resource totals are not summed again.

The system window uses one declared monotonic clock domain. Every request, release event and resource-meter interval must use that clock. The interval is `[start_at_s, end_at_s)`; events exactly at the end belong to the next window. Meter instrument IDs identify independently calibrated measurement streams, not a multi-channel data-acquisition chassis; distinct channels need distinct stream identities in this schema.

Each scored window also freezes the machine-unit ID, whole-system configuration SHA-256 and utility-topology ID. Each request carries a compartment/cell ID and a compartment-configuration SHA-256. Reusing one cell ID for two configuration digests within one window is rejected; a configuration change is a different experimental condition and must not be pooled as if it were the same cell. Each window also freezes one `service_class_id` and one `protocol_id`; every request must match both. A window for lightly soiled glassware therefore cannot be mixed with a normally soiled plate workload or a different independent-assay protocol. Mixed workloads require separate windows or a separately specified workload-mix benchmark with fixed, declared proportions.

## Definitions

- **Accepted request:** one stable service-request identity with one or more requested item IDs.
- **Started request:** accepted work that has entered physical service. A request can remain queued and unresolved with no start timestamp and no service report.
- **Completed request:** has a start, a close timestamp and an upstream service-accounting report.
- **Cancelled request:** a terminal cancellation timestamp, either while still queued (no physical attempt/report) or after at least one physical attempt (report retained). A cancellation is not an unresolved or completed request.
- **Unresolved request:** not closed or cancelled at the window boundary. It is right-censored at `end_at_s`; it cannot be silently dropped from the denominator or counted as a complete request.
- **Verified item release:** an item with an explicit release event whose identity is present in the upstream service report's independently verified item IDs. A timer, turbidity, model prediction or process proxy cannot authorize release.
- **Carry-in request:** accepted before the observation window and still queued or active at its start. It is reported separately to expose the boundary condition.
- **Maximum concurrency:** peak number of started service requests active at the same instant. Service intervals are half-open; a request ending at the same timestamp another starts does not create overlap.
- **Active request seconds:** sum of each started request's intersection with the observation window. This can exceed elapsed wall time under concurrency and is not the window duration.

## Reported metrics

- Quality-qualified throughput: independently verified item releases divided by elapsed wall-clock window hours.
- Request count, accepted/started/completed/cancelled counts, carry-in requests, and unresolved requests at window end.
- Requested items in the accepted-in-window cohort, items in cancelled accepted requests, and items in unresolved accepted requests that have not yet achieved a complete independent-assay qualification. An already-qualified item in a still-open multi-item request is not counted as pending qualification.
- Time to first independently verified release.
- Request latency p50/p95 over completed requests accepted during the window, and queue-wait p50/p95 over requests accepted in-window that actually started (including requests still in service at window end). Both use nearest-rank percentiles. Right-censored request latency is reported separately and is not imputed into percentiles.
- First-pass yield: first-attempt verified items divided by requested items in completed requests accepted during the window. The service ledger reports first-pass IDs; an item first qualified after a re-wash is not first-pass.
- Physical attempt count, rewash-attempt count, requests with any rewash attempt, item-attempt visits and reattempted item visits.
- Unqualified items reported by the service ledgers, active request seconds and maximum concurrency.
- Per-compartment request counts, cancellations, independently verified releases, active request seconds, peak within-cell concurrency and the configuration digest.
- System-level resource totals by canonical unit, plus window resource totals per verified release when at least one item was released. The field is a window-intensity ratio, not the full lifecycle resource cost for one service request.

A release count is event-based. An item qualified in a service report but not released in the scored window is not counted as a release in that window. A request accepted before the window can still contribute a release during it; this is why carry-in is surfaced.

## Resource boundary policy

The system-window resource rows are authoritative for the window numerator. Do not add the request-ledger totals to them. Otherwise an appliance main-feed meter plus a per-request allocation of the same energy can be counted twice. The report aggregates the underlying service reports separately: physical attempts, rewash attempts, item-attempt visits and item reattempt visits remain visible without adding their resource vectors again.

Each row contains a stable `measurement_boundary_id`. Overlapping intervals from different instruments on the same declared boundary are rejected. A parent meter can declare that it covers downstream boundary IDs; overlapping parent and child rows are rejected. This catches explicit aggregate-plus-branch duplication, but a false boundary map can still evade the structural check. The physical wiring/plumbing map and every meter's placement must be independently audited.

Distinct simultaneous branch-meter intervals may be summed only where their physical coverage is genuinely disjoint and that fact is supported by the independently audited topology. Boundary IDs are declarations, not evidence of physical separation.

For `ActiveOperatorTime`, use a pseudonymous per-person boundary ID. Two different people may legitimately accumulate person-seconds over overlapping intervals; two logs for the same person/boundary cannot overlap. Report person-seconds, not merely team wall-clock seconds.

A zero resource measurement must be represented by an explicit row. A missing resource dimension is not interpreted as zero. Resource rows need canonical units, interval bounds, method, evidence reference and digest; all instrumented dimensions also need instrument and calibration IDs.

**Window resource totals per release are period-intensity metrics, not full lifecycle resource costs for a particular item.** Carry-in work may have consumed resources before the window, and unresolved work may consume resources but produce no release before it ends. For fair comparisons, use a predeclared warm-up / scored / drain protocol, publish carry-in and unresolved counts, run sufficiently long steady-state windows, and separately retain each complete service-request ledger for lifecycle resources per verified item.

## Experimental protocol: one cell, concurrent cells, commercial racks

For each condition:

1. Freeze machine unit, whole-system configuration digest, utility-topology ID, compartment IDs/configuration digests, programme/firmware, water batch/hardness/inlet temperature, detergent and rinse-aid lots, soil family/age, loading recipe, required independent endpoints, resource meter map and clock source.
2. Register an arrival trace and service-request item set before dispatch. Retain every accepted, queued, started, cancelled, failed, interrupted and unresolved request. Cancellation time and reason are distinct from right-censoring at the window boundary. A queued cancellation has no physical attempt/report; a cancellation after start retains the service report and any releases independently qualified before cancellation. Releases at or after the cancellation timestamp are rejected.
3. Randomize service order within declared blocks. Capture actual start, item-level release, close, rewash and abort timestamps against the same monotonic clock.
4. Measure system-boundary wall electricity, upstream hot-water energy, fresh water, delivered detergent/rinse aid and operator person-seconds. Allocate shared preheat, standby, purge and fill overhead once at a documented boundary.
5. Use the exact per-item independent soil-removal, final-rinse-residue and cross-transfer assays. Keep microbiological efficacy and sanitization separate.
6. Score a fixed wall-clock window. Report carry-in, unresolved, queue, throughput, latency and all resources. Do not sum overlapping request durations to form the denominator.
7. Repeat with one compartment active, two compartments simultaneous, all compartments active, injected isolation/valve/filter/sensor faults, and an external operator/site where feasible.

Primary system-level results are verified items/hour, first-pass yield, physical attempt count, physical rewash-attempt count, number of requests containing a rewash, reattempted item visits, time-to-first-release, p50/p95 request latency, maximum concurrency, resource totals per window and window resource totals per verified release. Report per-compartment request/release counts, active time and concurrency, plus exact machine/configuration/topology IDs. Also report per-request lifecycle resources from complete service ledgers; the two denominators answer different questions.

## Benchmark standards and scope separation

### Household full-load

Use the corrected IEC 60436:2025 fifth edition (corrigendum 2026-04) for household/similar electric dishwasher performance. It includes a new reference machine and detergent, model unit-variation assessment and multi-compartment methodology. It is neither a safety standard nor a minimum-performance requirement.

Official source: https://webstore.iec.ch/en/publication/80459

### Commercial undercounter / hood-type

IEC 63136:2019 applies to manually loaded, electrically heated undercounter one-tank and one-tank hood-type commercial machines, measuring performance by washing plates. Its scope excludes transport-system machines (rack conveyors and flight-type), utensil washers and undercounter water-change machines. Use it for its stated machine class, not as a universal method for every commercial washer.

Official source: https://webstore.iec.ch/en/publication/59666

The UK Energy Technology List's high-temperature undercounter requirements provide an additional acquisition screen: energy at or below 0.4 kWh/cycle, water at or below 2.5 L/cycle, ready-to-use power at or below 0.4 kW, cleaning performance at least 90%, and a hygiene/sanitation pass. It refers to BS EN IEC 63136:2019 for energy and cleaning, and EN 17735:2022 or NSF/ANSI 3-2023 for hygiene. These are ETL eligibility thresholds, not the entire commercial SOTA frontier.

Official methodology and requirements: https://etl.energysecurity.gov.uk/products/professional-foodservice-equipment/undercounter-dishwashers

### Commercial rack-conveyor / flight-type

Do not claim IEC 63136 comparability for conveyor or flight-type machines because those are outside its scope. Use exact rack/flight geometry, loading recipe, rack or flight rate, item count and mix per rack, initial-fill/ready-state overhead, cycle/transport energy and water, and independent item release assays. Use applicable commercial hygiene requirements separately.

### Commercial hygiene

EN 17735:2022 specifies hygiene requirements and testing for professional commercial dishwashing machines used on food-contact washware, with operational, care and maintenance guidance. NSF/ANSI 3-2023 establishes minimum public-health and sanitation requirements for commercial warewashing equipment, including stationary rack and conveyor machines. Confirm exact jurisdiction, edition and certification scope before product claims; a high rinse temperature or a passed cleaning assay alone is not a sanitization certification.

Sources:
- BSI EN 17735:2022: https://knowledge.bsigroup.com/products/commercial-dishwashing-machines-hygiene-requirements-and-testing
- NSF food-equipment standards: https://www.nsf.org/nsf-standards/standards-portfolio/food-equipment-standards

## Acceptance tests

Source-authored Rust tests cover concurrent versus sequential service, carry-in requests, queued and in-flight cancellation, post-cancellation release rejection, right-censored queued and started requests, independent-assay release qualification, duplicate releases, clock mismatch, cross-instrument boundary overlap, declared parent/branch duplication, parallel operator person-time, missing required resource dimensions, first-pass yield, rewash aggregation, per-compartment statistics and configuration mismatch, and inconsistent upstream service reports. The module currently contains 23 tests; none are executed until the exact PR head is compiled and run.

All tests, formatting and clippy must run on the exact pull-request head before any pass claim. The validator does not authenticate referenced evidence bytes, prove assay independence, establish the truth of boundary IDs, infer omitted requests, or establish physical cleaning/hygiene performance.
