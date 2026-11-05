# Water-use qualification: source-to-use evidence contract v0.1

**Status:** design specification; no-runner tranche  
**Owner:** WATER-QUALITY-EVIDENCE-001 (#10500)  
**Parents:** WATERLOOP-APPL-001 (#10323), HOME-SERVICE-CONTINUITY-001 (#10445)  
**Compose:** HOME-SVC-METROLOGY-001 (#10341), HOME-SVC-RECEIPT-001 (#10403), lifecycle exposure/censoring (#4920), WaterLoop comparator work (#10483)  
**Evidence class:** design-only; no water sample or household system has been qualified.

## 1. Problem and paradigm

A water-quality decision is not a property of the noun "water". It is a relation:

`source + system + batch/stream + treatment state + sample evidence + intended use + physical outlet + time`

The question is not "Is the tank safe?" It is:

> Does the current, source-bound evidence support this declared use at this exact delivery boundary at this time, under the applicable standard and system-specific monitoring programme?

This moves the product from **sensor-led water status** to **source-to-use, evidence-qualified delivery**. It also prevents one passing sample from being projected onto a different source, treatment path, batch, tank, outlet, use class or future period.

This is a domain adapter and semantic contract. It is not another laboratory information system, generic provenance store, standards database, treatment engine, water controller or safety authority.

## 2. Compose existing infrastructure; do not fork it

- #10403 owns exact-byte / typed evidence receipts. Reuse its source and evaluator lineage seam when the dependency is available; do not invent another digest envelope.
- #10341 owns metrology identity, calibration and evidence strata. Reuse its instrument/method and uncertainty semantics.
- #4920 and #3697 own generic exposure, lifecycle, reliability and censoring semantics.
- #10445 owns service envelopes, user consent, boundary-level delivery and unknown-state handling.
- #10483 owns WaterLoop's distinct backup, reuse and leak-detection comparator functions.

This contract adds water-specific identity, quality-use compatibility and source-to-use invalidation only. It must not create a competing sensor catalogue, lab database or quality-limit repository.

## 3. Qualification record

The normative draft shape is in [`water-use-qualification-v1.schema.json`](water-use-qualification-v1.schema.json). A record binds:

| Dimension | Required identity / content | What it prevents |
|---|---|---|
| Source | source class, source ID, supply/system identity, source lineage | treating municipal, borehole, tanker, rainwater and greywater as interchangeable |
| Batch / stream | batch or stream ID, collection window, storage vessel, blend lineage | carrying one batch result to another batch or mixture |
| Treatment | train ID/revision, process state, media/service state, alarms, bypass/fault state | assuming treatment label equals actual process performance |
| Use | explicit drinking or non-potable use class | promoting a reuse classification into potable status |
| Boundary | exact sample point and claimed delivery boundary | claiming a result at a different outlet |
| Sample | sample ID, collection time, method, laboratory/provider, custody, parameters and analytical qualifiers | untraceable or method-incompatible claims |
| Governing profile | standard edition, system-specific risk assessment and monitoring programme revisions | applying stale or wrong-system criteria |
| Decision | state, reason, evidence IDs, decision time and monitoring status | replacing uncertainty with a pass |
| Events | source switches, mixing, treatment faults, storage events, non-compliance and corrective actions | reusing evidence after a material change |

A valid observation remains attached to its exact source ID, stream/batch ID, sample point and time. A decision can be reused only where the declared profile establishes that the evidence applies to the same source/system, stream, treatment state, use class, boundary and monitoring period.

## 4. Qualification states and service-credit mapping

| Water qualification state | Meaning | Household service treatment |
|---|---|---|
| `QualifiedForDeclaredUse` | Current evidence and applicable profile support this exact use and boundary | May be credited only for the declared use and interval |
| `KnownNonCompliance` | Evidence identifies a failed applicable requirement | No qualified service credit for the affected use |
| `UnknownEvidenceInsufficient` | Missing, incomplete, contradictory or insufficient evidence | No confirmed potable credit; physical flow is still reported |
| `ExpiredOrMonitoringOverdue` | The governing monitoring programme requires evidence that is overdue | No qualified credit until the programme's requalification rule is satisfied |
| `SourceOrSamplePointMismatch` | Evidence does not bind to the declared source, system, batch or boundary | No qualified credit |
| `TreatmentStateUnqualified` | Relevant fault, bypass, maintenance or operating condition is not qualified | No qualified credit for affected downstream use |
| `CorrectiveActionPending` | A non-compliant event is under response; restoration evidence is incomplete | Remain restricted / unknown until the prescribed closure criteria pass |

The general service evaluator's `Qualified / KnownFailure / Unknown` classification is a terminal projection, not a replacement for the water-specific reason. Map known non-compliance to known failure; map missing, stale, mismatched or unresolved evidence to unknown; preserve the water-specific reason and the raw delivery observation.

**Critical rule:** an unknown result is not proof of contamination, but it is also not proof of drinking-water quality. Do not conflate the epistemic state with the physical state.

## 5. Edition-aware, risk-based quality profile

The 2025 South African compulsory service standards refer to SANS 241 "as may be updated". The current-edition crosswalk is recorded in the parent household-continuity profile. Rand Water's 2026 migration briefing describes SANS 241:2025 as integrating risk assessment, monitoring, verification, non-compliance response and corrective action; it distinguishes mandatory process-risk parameters from system-specific risk-defined parameters and describes separate monitoring contexts for water supply systems, borehole systems, mobile tankers and static tanks.

Implementation rules:

1. Store the exact standard identifier/edition and the exact applicable risk-assessment and monitoring-programme revisions.
2. Keep mandatory process-risk parameters separate from system-specific risk-defined parameters.
3. Do not hardcode numeric SANS limits in this project schema or copy them from explanatory presentations. Use the current controlled SANS publication and qualified programme.
4. Do not invent one universal sample TTL. Determine due/overdue state from the relevant standard, source/system risk assessment, sampling programme and event invalidators.
5. A process sensor may supply operational telemetry but does not automatically replace a required laboratory or qualified verification method.
6. A non-compliant result starts a recorded incident / corrective-action path. Do not restore Qualified solely because a timer elapsed or an alarm disappeared.
7. Confirm applicability of each standard regime for the actual installation with qualified South African water-quality professionals / authorities. The migration briefing is explanatory evidence, not the standard itself.

For greywater and other reuse streams, keep the declared non-potable use and its applicable requirements distinct from drinking-water qualification. ISO/CD 25446:2026 is a committee draft under development, not a final standard or product certification.

## 6. Source-to-use graph and invalidation

Represent the evidence chain as a directed, inspectable chain:

`source -> collection -> storage / blend -> treatment -> sample point -> delivery boundary -> declared use`

A node or edge may be absent or unknown; never silently substitute an adjacent node. Every transition is an event with identity, timestamp and affected downstream claims.

| Event | Required consequence |
|---|---|
| Source switch | New source lineage; old result does not transfer by default |
| Batch blend / mixed source | Create a new derived batch with all parents; no "best source wins" |
| Treatment bypass / fault | Mark affected downstream use unqualified until the defined requalification rule passes |
| Media replacement / maintenance | Record configuration transition and any required requalification; do not assume automatic pass or automatic failure |
| Material storage/stagnation event | Apply the source/system risk programme's rule; absent a rule, state remains unresolved rather than inventing a TTL |
| Sample point / outlet change | Reject transfer unless a validated boundary relationship and programme support it |
| Monitoring becomes overdue | Expire the claim under the current monitoring programme |
| Non-compliant result | Open incident; retain the affected source/batch/boundary scope; require documented correction and verification |
| Contradictory observations | Preserve both observations and their lineage; adjudication must be explicit and independent |
| Corrective action completed | Requalify only against declared closure criteria and fresh required evidence |

A downstream result does not prove upstream supply compliance. An upstream result does not prove downstream storage or fixture quality. Sample points must be selected for the actual system and the claim being made.

## 7. Adversarial challenge corpus

The proposed deterministic WQ-001…WQ-014 matrix is design-only and should be maintained as a frozen manifest with expected dispositions held outside any candidate policy workspace.

| ID | Challenge | Required disposition |
|---|---|---|
| WQ-001 | Source switched after last valid sample | Mismatch / Unknown |
| WQ-002 | Monitoring programme overdue | Expired / Unknown |
| WQ-003 | Raw-source sample claimed for treated outlet | Sample-point mismatch |
| WQ-004 | Qualified batch blended with unqualified source | New derived batch; Unknown until qualified |
| WQ-005 | Storage residence event crosses a risk-defined limit | Apply profile; otherwise Unknown |
| WQ-006 | Treatment bypass or relevant process fault | TreatmentStateUnqualified |
| WQ-007 | Wrong supply-system / borehole risk profile revision | Mismatch / Unknown |
| WQ-008 | Required method, result qualifier, reporting limit or unit missing | Unknown |
| WQ-009 | Contradictory laboratory results with no resolved adjudication | Unknown |
| WQ-010 | Tanker/static-tank sample cannot bind to delivered batch | Mismatch / Unknown |
| WQ-011 | Non-compliant result with incomplete corrective action | CorrectiveActionPending |
| WQ-012 | Current evidence matches source, system, sample point, use and programme | Qualified only within declared scope |
| WQ-013 | Non-potable greywater result relabelled as drinking water | Hard rejection |
| WQ-014 | One household outlet's result reused for another outlet without supported transfer | Sample-point mismatch |

Add negative controls for digest changes, replay from old profile revisions, time-zone/clock ambiguity, censored samples, unit conversions, parameter aliases and duplicate sample IDs without counting them as independent physical evidence.

The positive cases are essential. A validator that rejects everything is not a qualified solution.

## 8. Outcomes and proposed gates

Keep these outputs separate:
- false acceptance of known-noncompliant, stale, mismatched and out-of-scope evidence;
- false rejection / unnecessary service restriction;
- qualified litres by use class, source and delivery boundary;
- physical litres delivered but not quality-qualified;
- duration by qualification state;
- time from a source/process event to detection and state update;
- time from corrective action to verified restoration;
- sample-point and source coverage versus the declared programme;
- evidence completeness, missingness, contradiction and adjudication burden;
- sampling / analysis cost and turnaround;
- household service deficit attributable to quality restriction.

Proposed software acceptance: every negative control must be rejected or remain explicitly unresolved; no negative control may receive a potable pass. Positive controls must be accepted only when every required field and transition condition is satisfied.

For a future physical false-acceptance statement, preregister independent trials. Under an IID Bernoulli approximation, zero false accepts in 299 independent trials gives a one-sided 95% upper bound near 1%. Repeated samples sharing a source, batch, site, instrument, operator or collection event are clustered and cannot be counted as independent without a justified dependence model. This is a planning relationship, not a result.

## 9. Execution order

1. Review the schema and state semantics with the water-quality/metrology owners.
2. Freeze WQ-001…014 and an independent expected-disposition oracle.
3. Add a standalone Rust offline validator that composes with the existing household evidence/receipt path; do not add production Python or a duplicate receipt envelope.
4. Add mutation/negative tests for every invalidator and positive known-answer tests.
5. Replay receipts independently from raw source bytes and preserve exact profile / evaluator identities.
6. Only after qualified partner review, design a contained bench with independent sample collection, correct sample points, laboratory methods and safety boundaries.
7. Progress E1 repeatability, E2 independent lab reproducibility, E3 source/system/production variation, then E4 household validity. Do not leap between strata.

## 10. Claim ceiling

This package can qualify schema and software interpretation only. It cannot declare a sample safe, prove treatment efficacy, establish compliance with SANS 241, validate a municipal or private supply, authorize plumbing, or establish WaterLoop superiority. Those require the exact current standard, qualified interpretation, appropriate physical sampling and the relevant independent evidence.


## 11. Industry baseline and competitive target ladder

The current comparator ledger in [WaterLoop competitive evidence PR #10485](https://github.com/Luminous-Dynamics/symthaea/pull/10485) already identifies established vendor capability families:
- **Backup delivery:** JoJo municipal-backup tank + pump + automatic filling / switching.
- **Leak observability and isolation:** Moen Flo and Phyn Plus flow/pressure monitoring, leak detection and automatic shutoff.
- **Greywater reuse:** Hydraloop H300/H350 (Class R / Laundry & Bathing listing) and INTEWA AQUALOOP (Class C / Laundry & Bathing listings for larger systems).

These are different functions and reference classes, not one league table. Public product pages and certification listings are not independent challenge results. Do not claim that WaterLoop has invented tank backup, leak sensing/shutoff, or greywater treatment. Do not claim that no incumbent offers an integrated combination without a fresh, broader market search.

### How the candidate can be better

The plausible differentiated capability is not another isolated tank, sensor, or treatment vessel. It is a **verifiable source-to-use service contract** that joins:
- what water source and batch actually entered the system;
- which treatment and storage state applied;
- which samples and risk programme support the claimed use;
- what service was delivered at the actual user/fixture boundary;
- what changed, became stale, failed, or required corrective action;
- whether the system remained locally understandable and operable without cloud/account dependencies.

That is a research hypothesis. It must be compared against the strongest matched incumbents and any integrated solutions discovered during landscape expansion.

### Proposed benchmark target ladder

| Tier | Proposed target | Evidence required |
|---|---|---|
| Deterministic semantic conformance | 100% of WQ negative controls rejected or explicitly Unknown / restricted; zero wrong-use or stale-evidence potable passes | Frozen known-answer WQ corpus, independent evaluator, mutation tests |
| Positive-control utility | All valid positive controls accepted when the full exact-scope evidence is present; report false refusals and causes | Positive controls across each declared source/system/use class |
| Lineage completeness | 100% of credited potable litres bind to source/system, stream or batch, treatment state, sample point, boundary, use class, current SANS edition and monitoring-programme revision | Receipt replay and event-history audit |
| Invalidation latency | Every declared source/treatment/boundary/monitoring invalidator changes the downstream qualification state before further service is credited; report event-to-state latency separately | Deterministic event replay, then instrumented fault injection |
| Independent false-acceptance bound | Stretch objective: zero false potable acceptances in at least 299 independent, externally adjudicated trials, giving an approximately 1% one-sided 95% upper bound under IID Bernoulli assumptions | Independent physical samples / batches / sites; explicit clustering correction |
| Operational utility | Minimize unjustified restriction and report litres withheld solely because evidence is missing, stale or mis-scoped | Matched control, user/service impact and evidence turnaround |
| Offline resilience | Local service, manual fallback, local evidence visibility and safe recovery remain characterized across NET-001…NET-008 | Exact SKU/firmware/network profile and safe, approved disconnect tests |
| Deployment evidence | Progress E1 repeatability → E2 independent-lab reproducibility → E3 source/system/unit variation → E4 household use; no shortcut | Independent sampling, calibration, site and household protocols |

The 299-trial calculation is a statistical planning bound, not a blanket guarantee and not a target that synthetic fixtures can satisfy. Repeated samples from one batch, tank, laboratory, site or operator are correlated and must not be counted as independent trials without a justified model.

### Competitive evaluation rule

For each incumbent function, match the same source, use class, delivery boundary, service demand, initial reserve, maintenance state and disturbance profile. Compare:
- qualified service actually delivered (not storage nameplate volume);
- quality qualification / false-acceptance and unresolved-evidence behaviour;
- flow/pressure at the use point;
- loss events, isolation latency and false interruption;
- mains displacement only for the reuse arm;
- local/offline operation, recovery and human maintenance;
- total installed and lifecycle cost.

The target is to exceed the best matched operational baseline on a declared decision-relevant outcome without violating any service, quality, safety, access or maintenance gate. An improvement on one function cannot compensate for failure in another. Do not produce a scalar "WaterLoop score."

**External comparison references:** JoJo municipal backup https://www.jojo.co.za/water-storage-solutions/municipal-backup-water-storage-solutions/ ; Moen Flo https://flo.moen.com/pages/flo-smart-water-monitor ; Phyn https://www.phyn.com/ ; Hydraloop H300 https://www.hydraloop.com/products/h300 ; NSF/ANSI 350 listings https://info.nsf.org/Certified/Wastewater/Listings.asp?Company=C0427542&Standard=350 ; INTEWA AQUALOOP https://www.intewa.com/en/greywater-recycling/.


### Timestamp normalization for deterministic evaluation

The current Rust adapter requires every timestamp used in temporal comparisons to be normalized to UTC RFC3339 with a trailing Z. Offset-bearing values (for example +02:00) must be converted by the producer before evaluation; the adapter rejects ambiguous/non-normalized offsets rather than comparing timestamp strings lexicographically. This is an adapter constraint, not a claim that the source system's original timezone metadata should be discarded: retain the original timestamp and timezone in the upstream receipt when present.

The boundary record includes both boundary_id and sample_point_id. They identify different concepts: the service outlet and the exact sampling location. same_physical_point requires them to denote the same physical point; validated_transfer requires an explicit transfer-validation reference.


### Sample source and stream binding

Every sample observation carries both source_id and stream_id. A supporting sample must match the qualification record's exact source and stream/batch. A sample from a different batch at the same physical tap is not a supporting sample; neither matching outlet geometry nor a similar result can repair that identity mismatch. A validated boundary transfer is a separate relation and does not erase source/stream lineage.


## 12. Sample time must fall within the declared stream window

In addition to exact source, stream/batch and sample-point binding, every supporting sample's collection timestamp must lie within the declared stream observation interval:
- collection before `stream.observed_from` is rejected;
- collection after a non-null `stream.observed_until` is rejected;
- timestamps are normalized to UTC before temporal comparison;
- a matching ID or sample point cannot override a temporal mismatch.

This prevents an old batch observation or a future sample from being attached to a different claimed service interval. It does not define a universal freshness period: after the stream window check, evidence currentness still follows the applicable standard, risk assessment, monitoring programme, storage/treatment events and declared validity interval.

The Rust adapter includes WQ-015 / WQ-016 regression controls for samples immediately outside the stream window. Software validation of the latest commit remains pending until its exact-head workflow completes.


### Complementary international management guidance published in 2026

ISO 18998:2026, *Water reuse in urban areas — Guidelines for decentralized water reuse system — Management of a decentralized water reuse system*, was published in January 2026. Its stated scope spans the decentralized system components, including source water, reclaimed water, treatment, storage, distribution, operation/maintenance, monitoring and emergency response. ISO explicitly says it does **not** provide the monitoring parameters or regulatory values for a decentralized reuse system. Treat it as a useful management-system comparator, not as a substitute for South African law, SANS requirements, source-specific qualification, or numerical reuse-quality limits.

The separate ISO 25446 building-greywater guidance has progressed to the DIS/FDIS-preparation stage but remains under development; keep its status distinct from published ISO 18998:2026.

References: https://www.iso.org/standard/85697.html ; https://www.iso.org/standard/90421.html
