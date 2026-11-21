# Luminous Aqua — Repairability and Serviceability Evidence Protocol v0.1

**Status:** evidence contract and timed-trial preregistration; no repair trials have been executed  
**Workstream:** [AQUA-SERVICEABILITY-RECEIPTS-001 #12844](https://github.com/Luminous-Dynamics/symthaea/issues/12844)  
**Parent:** [AQUA-CIRCULAR-SERVICE-001 #12727](https://github.com/Luminous-Dynamics/symthaea/issues/12727)  
**Implementation:** `symthaea-aqua-bench::serviceability`

## 1. Paradigm shift: a repair is not complete when the part is replaced

A product is serviceable only when a defined fault can be diagnosed, accessed, repaired, checked and safely returned to its intended service. A replaceable cartridge hidden behind destructive finishes is not a fully repairable product.

Aqua treats maintenance as a first-class phase of the bathing service. The benchmark therefore distinguishes:

1. **Access result:** whether the target part could be reached and removed without destructive finish removal.
2. **Repair result:** whether the specified repair/replacement was completed and the post-repair functional check passed.
3. **Return-to-service result:** whether all required independent device, water-quality and metrology gates were completed and an authorized return-to-service receipt exists.

A job may succeed at one layer and fail at another. These states must not be collapsed into a marketing repairability score.

## 2. Standards context and limitations

### ISO 20887:2020

ISO 20887:2020 provides design-for-disassembly and adaptability principles and guidance for buildings, civil engineering works and their constituent parts. ISO records that it was reviewed and confirmed in 2025 and remains current. It explicitly does not prescribe universal numeric disassembly or adaptability performance levels.

Use ISO 20887 to structure AQS room/service-spine design decisions: reversible connections, access, adaptability, component separation and avoiding unnecessary removal of surrounding construction. It is not a pass/fail repairability certification for a shower fixture.

Official source: https://www.iso.org/standard/69370.html

### EN 45554:2020

EN 45554:2020 provides generic assessment methods for repair, reuse and upgrade of **energy-related products**. Its concepts include disassembly depth, fasteners/connectors, tools and support availability. Do not assume the standard's direct legal or product-group applicability to passive plumbing fixtures. Use relevant concepts only as methodological inspiration unless the responsible standards authority confirms applicability.

Official national standards catalogue: https://www.nen.nl/en/nen-en-45554-2020-en-270440

The Aqua protocol is a product-specific repair-trial method, not conformity assessment under either standard.

## 3. Frozen AQS-1 fault/job manifest

The initial manifest should include, at minimum:

| Job ID | Routine? | Starting fault / task | Primary access question | Required return-to-service gates |
|---|---|---|---|---|
| AQUA-SVC-01 | Yes | Shower nozzle/cassette blockage | Can cassette be removed and cleaned/replaced without finish damage? | Device function; wet-path hygiene if opened/contaminated |
| AQUA-SVC-02 | Yes | Inlet strainer blockage | Can strainer be isolated, removed and reinstalled from dry service side? | Leak/function; source-quality gate if required |
| AQUA-SVC-03 | Yes | Thermostatic cartridge replacement | Can cartridge be isolated and replaced without destructive wall access? | Independent device/safety authority; temperature/function checks |
| AQUA-SVC-04 | Yes | Seal, hose or handpiece replacement | Can wear part be replaced using documented tools and stocked parts? | Leak and functional check |
| AQUA-SVC-05 | Yes | Drain trap / cleanout obstruction | Can cleanout be reached and restored without removing permanent finishes? | Drain function; hygiene gate if wet path is opened |
| AQUA-SVC-06 | Conditional | DWHR cleanout / isolation / bypass | Can the exchanger be isolated, cleaned and returned without opening unrelated wet paths? | Leak/function; hygiene and resource/metrology gates as applicable |
| AQUA-SVC-07 | Conditional | Sensor replacement | Can sensor be replaced without modifying the calibrated installation boundary? | Mycelix metrology identity, calibration and uncertainty requalification |
| AQUA-SVC-08 | Yes | Dry-side isolation valve replacement | Can isolation and union access support replacement without destructive finishes? | Independent device authority; leak/function and local plumbing review |

The manifest is frozen before comparative trials. A job can be marked not applicable only with a documented architecture reason established before results are known. Failed, aborted and unserviceable jobs remain in the data set.

## 4. Observation contract

Disassembly depth is defined as the count of ordered disassembly actions from the initial service access to the target component, using the frozen job instructions. It is not the number of parts removed. Record these separately: (a) disassembly depth, (b) distinct components physically removed, and (c) fastener/connector separations. This avoids treating one large removable assembly and several nested access steps as equivalent jobs.

Each repair run binds to:
- exact product variant, specimen and hardware/configuration digest;
- fault manifest, job profile, method/protocol and replicate identifiers;
- technician block and skill class;
- access outcome and disassembly depth;
- tools used and unique tool identifiers;
- spare-part identity and whether the part was available before the job;
- technician hands-on minutes and total service downtime;
- water, electricity and waste observations, if independently metered;
- post-repair functional check;
- required requalification flags;
- owner-specific return-to-service evidence.

Unknown values are explicit. Missing water, electricity, waste or environmental-impact data remain unknown, not zero. The Aqua Rust evaluator preserves resource observations but does not total them; Symtropy owns canonical resource closure and lifecycle cost.

### Access outcomes

- **DrySideWithoutDestructiveFinish:** reachable from the designated dry-side service access with no destructive finish removal.
- **WetSideWithoutDestructiveFinish:** reachable from the wet side without destructive finish removal; the opened wet path may still trigger hygiene/requalification.
- **DestructiveFinishRemovalRequired:** tile, wall, fixed panel, waterproofing or other permanent finish must be cut/broken/removed.
- **NotRepairable:** the defined repair cannot be completed using the documented process.
- **Unknown:** evidence is insufficient to classify access.

Unknown is never counted as non-destructive success.

### Qualified return-to-service

A job counts as qualified return only when:
- post-repair function is independently recorded as passed;
- return status is QualifiedReady;
- an independent device-authority receipt is present;
- if water-quality requalification is required, a Mycelix water-quality receipt is present;
- if metrology requalification is required, a Mycelix metrology receipt is present.

A replacement part installed without those gates is not counted as returned to service.

## 5. Matched incumbent versus AQS trials

Compare:
1. one exact incumbent shower/basin installation and its documented repair path;
2. AQS-1 direct-flow module with dry-side service spine;
3. AQS-1 plus optional DWHR service path only for the jobs that actually exercise DWHR.

Pair runs by exact job profile, fault-manifest digest, replicate, technician block, technician skill and protocol. Product, specimen and configuration identity are expected to differ. Randomize job order within practical blocks. Use the same technician for a matched pair where practical; rotate technicians across designs to expose operator effects.

A seeded fault must be repeatable and must not create an uncontrolled hazard. Physical fault injection and service work require a separately reviewed, isolated/de-energized or otherwise professionally controlled setup. Wet-path opening, electrical work and pressure-bearing component replacement must follow the competent service procedure.

## 6. Metrics — report a vector

The initial Rust evaluator reports:
- observation count;
- routine-job count;
- qualified, non-destructive routine-repair count and fraction;
- unknown-access count;
- destructive/unrepairable access count;
- qualified return-to-service count;
- post-repair functional-pass count;
- unique tool count;
- median technician minutes;
- median total downtime minutes.

It does not calculate:
- lifecycle cost;
- water/energy/material/waste totals;
- embodied carbon or circularity impact;
- hygiene or metrology validity;
- repair probability over a product lifetime.

Those require their canonical owner evidence and methods.

### Initial targets

- **100% of declared routine wear-part jobs** should be accessible without destructive finish removal in the AQS-1 reference design.
- Establish incumbent baselines before setting percentage claims for repair time, unique tools, spare availability or downtime.
- Report first-pass function success separately from qualified return-to-service.
- Report failed, pending and unknown outcomes, and do not count them as successes.
- Any lifecycle-cost or environmental claim must use a complete, source-classified functional unit and uncertainty analysis from the responsible owner.

The 100% access target is a design objective, not an achieved result or legal threshold.

## 7. Statistical and anti-gaming rules

- Freeze job profiles, difficulty, instructions, access criteria and acceptance tests before comparative trials.
- Do not compare a simple cartridge swap against an incumbent wall-opening job unless the same repair objective is being achieved.
- Report between-specimen and between-technician variation.
- Repeated jobs on one specimen are not independent product specimens.
- Keep pilot and confirmation runs separate.
- Keep routine and non-routine jobs separate.
- Keep dry-side and wet-side access outcomes separate.
- No single composite score may hide destructive access, failed function, missing requalification or parts unavailable at the start.
- Publish all failed, aborted, repeated and excluded jobs with reasons.
- Use a fixed observation window for spare availability; supplier promises and actual stock status are distinct evidence.

## 8. Software implementation and test boundary

The dependency-free Rust module in `crates/domains/symthaea-aqua-bench/src/serviceability.rs`:
- validates job/run identities, digests, tool identifiers and measured time/resource fields;
- requires an independent device-authority receipt for QualifiedReady;
- requires Mycelix water-quality or metrology evidence when requalification is flagged;
- counts only non-destructive access plus passed function plus qualified return as a successful routine repair;
- keeps unknown access in the denominator;
- reports an undefined fraction when no routine jobs are present;
- validates matched incumbent/AQS job context.

The serviceability cohort now also distinguishes spare availability as AvailableBeforeJob / NotAvailableBeforeJob / Unknown, requires an exact spare-part identity before counting a part as available, reports unique spare-part count, and reports median disassembly depth separately. Unknown spare availability is excluded from the known-availability fraction but remains explicitly counted; a zero known denominator yields an undefined fraction. The authored tests exercise fail-closed behavior, return-to-service evidence, repair denominator handling, spare availability and disassembly-depth reporting. They are not a CI pass until the exact PR head reports successful execution.

## 9. Claim ceiling

This is an evidence contract and timed-trial protocol. It does not establish that AQS-1 is repairable, lower-cost, circular, lower-carbon, safe to service or more durable. It authorizes no procurement, construction, occupied service or physical test outside a separately reviewed procedure.


## 10. Extended cohort measures

The Rust evaluator now records three distinct spare-availability states rather than a boolean:
- AvailableBeforeJob (requires an exact part ID);
- NotAvailableBeforeJob;
- Unknown.

For routine jobs it reports each count separately and computes an availability fraction only over the known Available + NotAvailable subset. Unknown observations do not silently become unavailable or available. If no known observations exist, the fraction is undefined.

The cohort also reports unique spare-part IDs and median disassembly depth, distinct components removed and fastener/connector separations separately from technician time and downtime. These are descriptive serviceability measures; they are not lifecycle cost, repair-probability, circularity or environmental-impact calculations.

## 11. Partial repair records versus qualified return

A repair trial may legitimately be pending while some requalification evidence has already been collected. For example, a wet-path intervention may have a Mycelix water-quality receipt but still await the independent device-authority return-to-service receipt. Such an observation remains valid as a pending job record; it is not a qualified return and cannot count as a successful repair.

The Rust evaluator therefore distinguishes:
- **record validity:** the run and its partial observations are well-formed;
- **qualified return:** the function check passed, return state is QualifiedReady, independent device-authority evidence exists, and every flagged Mycelix water-quality/metrology gate has its required receipt.

The public qualified-return predicate validates the evidence record itself; it does not rely solely on a caller-supplied QualifiedReady enum. Known AvailableBeforeJob and NotAvailableBeforeJob spare states both require an exact spare-part identity. Unknown state may omit the ID but is reported separately.
