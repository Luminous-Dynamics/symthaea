# WATER-LEAK-001 — Data acquisition playbook for Johannesburg

**Status:** Research plan; no utility records have been obtained and no partner access is implied.  
**Reviewed:** 2026-10-12  
**Parent:** [WATER-LEAK-001 #7413](https://github.com/Luminous-Dynamics/symthaea/issues/7413)  
**Satellite gate:** [WATER-SAT-001 #7415](https://github.com/Luminous-Dynamics/symthaea/issues/7415)

## Decision

The most promising unique data strategy is to build a **repair-centred, multi-modal event cohort**, not to hunt for an enormous anonymous set of “leak” points.

For each eligible incident, link (under an approved data agreement) a pre-repair observation, a documented independent inspection, the actual repair outcome, and post-repair observations. The repair becomes a time-localized intervention. This lets us ask whether a satellite or sensor anomaly changes after a verified repair, while retaining matched controls for rain, ground disturbance, pressure changes and unrelated maintenance.

This is not a substitute for ground truth: post-repair signal change is corroboration, not proof on its own. A utility work order alone is not a confirmed leak, a complaint is not a confirmed leak, and an unreported location is not a confirmed no-leak case.

The work has two distinct tracks:

1. **Satellite feasibility** needs timestamped, spatially qualified field outcomes joined to imagery, pipe/zone context where permitted, and data-use rights.
2. **Acoustic/sensor feasibility** can start earlier using controlled data and public benchmarks, but those results cannot establish satellite performance or Johannesburg transfer.

## Source ladder

| Priority | Source | What it could supply | Main limitation / gate |
|---|---|---|---|
| 1 | Johannesburg Water research, leak-detection, noise-logger, pressure-controller and repair workflows | The highest-value local linkage: candidate alert → blind inspection → work order → completed repair → independent post-repair check | Requires authorized cooperation, agreed purpose and secure handling; current device rollout counts must be reconfirmed |
| 2 | University of Johannesburg / city-water research collaboration | A route to learn what the prior satellite initiative actually measured, which imagery and labels exist, and whether a non-duplicative reproducibility study is possible | Public publication abstract is not a dataset, validation report or licence |
| 3 | CGIS and named municipal data custodians | Minimum necessary pipe/zone geometry, material/diameter where recorded, PRVs/reservoir context and coordinates with accuracy metadata | A spatial-layer viewer is not permission to export, train on or redistribute data; request access from the actual owner |
| 4 | Local plumbers, leak-detection contractors, facilities teams, property managers and insurers | Prospective repair-verified events outside a municipal data-sharing bottleneck; exposed mains, campuses, industrial facilities and building plumbing may be easier first sites | Different physical profile from buried municipal mains; consent, coverage bias and evidence quality must be explicit |
| 5 | Public sensor/acoustic benchmarks | A reproducible code and baseline test path while local data agreements are pending | Useful for sensors/DSP only; not a Johannesburg dataset and not satellite ground truth |
| 6 | Community reports, call-centre narratives and public complaints | Leads for where and when to investigate; helps study coverage and reporting bias | Weak labels only unless linked to an actual independently recorded inspection and outcome |
| 7 | Rainfall, land cover, soil-moisture and land-disturbance products | Confounder covariates and matched-control construction | Never a leak label; these variables can themselves explain satellite anomalies |

## Highest-value local route: make the repair process the data generator

Johannesburg Water's public material describes 15 technical teams using listening sticks, ground microphones and correlators, with detections referred for repair. Its July 2026 rollout announcement reported 225 installed noise loggers, a further 125 planned by October, and smart pressure controllers in installation. The dates matter: these are published programme/rollout figures, not a verified current fleet inventory. See the [Active Leak Detection programme](https://www.johannesburgwater.co.za/about-us/campaigns/active-leak-detection-programme/) and [23 July 2026 rollout announcement](https://www.johannesburgwater.co.za/blog/2026/07/23/johannesburg-water-accelerates-smart-technology-rollout-to-reduce-water-losses/).

Johannesburg Water's [Innovation & Technology / RDI unit](https://www.johannesburgwater.co.za/about-us/innovationtechnology/) explicitly describes collaboration among stakeholders in government, academia and industry and mentions independent, unbiased studies. This makes it the most relevant institutional entry point for a bounded feasibility request, rather than asking operations teams for an indiscriminate data dump.

### Ask for a minimum viable extract

Start with a **joinability assessment**, not permission for a production training corpus. Ask whether the data steward can provide a small, de-identified sample or run a secure in-environment join that reports only allowed aggregate counts.

Request field definitions and completeness for:

- Pseudonymous event ID, pipe/zone/asset-segment IDs, and coordinate reference system plus horizontal uncertainty. Keep true customer identity and residential addresses out of the research corpus.
- Event times with timezone and semantics: first reported, first detected, inspection, repair start, repair completion and post-repair check. Preserve missing values; do not invent exact leak onset times.
- Observation method and source: noise logger, listening stick, ground microphone, correlator, visual inspection, physical test or repair confirmation; distinguish an alarm from a human decision.
- Outcome with evidence class: independently confirmed physical leak, independently inspected no-leak, inconclusive, or unknown.
- Work-order status transitions, repair type and repair verification, not merely ticket creation or closure.
- Available acoustic/pressure/flow samples or summary features around the event, with sensor ID, calibration/currentness, time synchronization, firmware/configuration identity and quality flags.
- Pipe material/diameter, pressure zone/PRV, resurfacing/excavation or other ground disturbance, and available rainfall/land-cover context where licensable.
- Data owner, permitted purposes, retention, access controls, attribution, redistribution and commercial-use terms for every source.

The first question is whether the join is feasible and how many records survive the eligibility rules. A tiny, reliable, inspectable cohort beats a large export whose outcomes cannot be trusted.

### The repair-response packet

For each eligible event, make a bundle of linked evidence records rather than one ambiguous label:

1. **Before:** candidate detector/satellite observation(s), exact acquisition/recording times, raw-source digest, preprocessing version and known confounders.
2. **Inspection:** the independent observation method, time, operator role or verification source, uncertainty, and result including inconclusive/no-access.
3. **Intervention:** repair action and completion time, supported by an authorized work-order/evidence reference.
4. **After:** repeat observation under comparable conditions, calibration and operating context; record whether the earlier signal changed, persisted or could not be compared.
5. **Controls:** sample inspected no-leak locations and matched nearby/temporal controls, including non-leak moisture sources, rain, irrigation, resurfacing, construction, sewer/wastewater and pressure/demand changes.

Do not require every event to have every modality. Missingness is explicit and the usable cohort is modality-specific. A repair can alter surface water, pressure and acoustic noise at once; pre/post signal reduction is supportive evidence, not independent confirmation unless the verification protocol says so.

## A distinctive study design: intervention-centred satellite time series

Instead of asking whether a single image classifies a pipe pixel as “leak” or “no leak,” identify verified repair events with adequate imagery on both sides of the repair and test for a predeclared, persistent anomaly change. Compare that change with matched non-leak controls and unrelated repair/construction events in the same season/land-cover conditions.

This design may obtain useful signal with fewer labels, but it has strict eligibility gates:

- Event location uncertainty must be compatible with the imagery footprint and candidate area; report a range, do not snap an uncertain point to the nearest pipe and call it exact.
- Acquisition dates must bracket the intervention closely enough for the target phenomenon; do not imply a satellite scene provides continuous observation.
- Rainfall, irrigation, vegetation, roadworks, surface sealing, sewer leaks, reservoir/PRV operations and unrelated repairs must be recorded or matched where possible.
- Every pre/post pair shares a single event group and must stay inside one evaluation split. Nearby pixels, multiple acquisitions of the same event and derivative patches cannot be randomized across train/test.
- Report the number of events excluded for poor geometry, missing before/after imagery, unknown repair time, missing permission and confounding. A filtered sample is not the whole utility network.
- The model cannot be trained on the held-out test events or have thresholds tuned against them.

A useful first output may be a **go/no-go feasibility result**, not a model: “X candidate repairs have adequate image timing and coordinate uncertainty; Y have independent positive confirmation; Z controls are inspectable; licensing allows/blocks the intended use.”

## Public data to use now, without waiting on utility access

### 1. New acoustic-data release (published 13 February 2026)

The [Zenodo dataset](https://zenodo.org/records/18631450) describes 1,000 acoustic clips, with 500 leak clips, 386 no-leak clips and 114 environmental-noise clips. The publisher says most environmental-noise examples came from a separate public site. However, the Zenodo Rights/License field is blank in the page inspected for this review. **Do not assume the audio is cleared for commercial training or redistribution.** Contact the dataset owners for a clear licence, clip provenance, consent/collection terms, pipe and sensor metadata, and permitted derivative uses before ingesting it into the shared corpus. Until then, treat it as a research lead, not approved training data.

### 2. Laboratory sensor benchmark (2023)

The [Aghashahi, Sela and Banks benchmark](https://doi.org/10.1016/j.dib.2023.109148) describes 280 measurements from accelerometers, hydrophones and dynamic-pressure sensors, spanning four leak types plus no-leak cases, looped/branched configurations and six noise/demand conditions. It is suitable for reproducing sensor/DSP baselines and examining negative cases. Confirm the exact deposited data licence and preserve original splits/measurement identity before any reuse. It is not satellite ground truth.

### 3. BattLeDIM SCADA and leak-event benchmark

The [BattLeDIM dataset on Zenodo](https://zenodo.org/records/4017659) includes historical flow, pressure, tank-level and demand measurements, leak reports and a network model for the L-Town evaluation. It is a useful benchmark for time-series leakage detection and repair-time alignment. The network is a benchmark constructed from a real utility context, not Johannesburg's actual network; results must be labelled as external benchmark results. Confirm the applicable licence and terms on the deposit before redistribution.

### Public data ingestion rule

Do not download directly into a mixed-purpose corpus before rights review. A quarantine manifest should record DOI/source URL, owner, licence text as observed, licence evidence digest, intended purposes, original checksum, file inventory, consent/provenance notes, and approval status. Missing/ambiguous licence means `rights_status=blocked_pending_review`, not “open data.” The 2026 acoustic record is a practical test case for this policy.

## If municipal access is slow: create a verified local cohort

A second unique route is a voluntary **Verified Repair Atlas** with a small group of licensed plumbers, industrial facilities, campuses or property/facilities managers. The value proposition is simple: participants provide a permissioned, minimized repair/inspection record; in return, they receive a local leak-evidence summary or an anonymized benchmark report. No participant must disclose customer identity, and exact asset coordinates remain access-controlled.

Required safeguards:

- Explicit opt-in and permitted-use agreement before data collection.
- Human inspection result and independent repair/after-check evidence; a photo, complaint or contractor invoice alone does not automatically establish a physical leak.
- Pseudonymous event and asset IDs; coarse/public location only where re-identification risk is acceptable.
- Balanced collection of leak positives, genuinely inspected negatives, no-access, inconclusive and sensor-fault cases—not just successful detections.
- Standardized sensor mounting, calibration, timestamping and protocol; preserve the device, pipe material, pressure and distance context.
- No claim that a household/building plumbing cohort transfers to buried municipal mains without a separate transfer experiment.

This creates genuinely local acoustic and repair evidence even before satellite feasibility is known. If the goal remains satellite detection, it is useful only if the events also have adequate coordinates/time and rights for imagery matching.

## Independent catalogue cross-check: OData

The STAC collector should not be the sole test of catalogue availability. The companion [OData inventory utility](https://github.com/Luminous-Dynamics/symthaea/blob/research/water-leak-001-johannesburg-transfer/tools/water_leak_satellite/odata_inventory.py) queries the Copernicus Products endpoint independently for Sentinel-1 GRD and Sentinel-2 L2A product families. It supports month, quarter, and year partitions; the live workflow defaults to quarters, using a 30-second request timeout and at most two bounded retries for transient network failures and HTTP 408/429/5xx responses. Successful pages record the number of request attempts, and the schema-v2 manifest binds the retry policy and time partition to the saved query evidence. The verifier checks that policy and interval coverage against the original scope.

The earlier OData cross-check timed out at the former 45-second broad-window setting. That failure did not mean zero catalogue items were available. A new metadata-only refresh is queued at [workflow run 38576061525](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38576061525); wait for its completed exact-run artifact and reverify it before drawing counts or coverage conclusions. Compare OData and STAC results, but do not assume identical product semantics or item counts. A product listing or footprint intersection does not establish resolution, cloud-free pixels at a pipe, leak truth, or commercial-use rights. Raster retrieval remains separate and gated on an approved study area, permissioned repair/inspection outcomes, and usage rights.

## Environmental covariates: valuable, but never labels

The 2026 review of Sentinel-1 soil-moisture retrieval says that vegetation, surface roughness, spatial/temporal constraints and sensor/model limitations affect retrieval quality. Use public rainfall, soil-moisture, land-cover, roadworks and vegetation series to test alternative explanations for radar anomalies, not to label leaks. See [Rahmati et al., *Remote Sensing of Environment* (2026)](https://doi.org/10.1016/j.rse.2025.115146).

Similarly, a customer report or water outage may help prioritize sampling but cannot become a positive physical-leak label. The absence of reports cannot become a negative label.

## Concrete next sequence

1. **Now:** run the 57-case offline suite on the exact branch checkout; keep the result distinct from hosted Actions status.
2. **Data inventory:** produce a source-by-source rights register; begin with the 2026 Zenodo acoustic release and the 2023 lab benchmark, keeping unlicensed records quarantined.
3. **Local entry point:** send a bounded research-data feasibility enquiry to Johannesburg Water's RDI function and the University of Johannesburg authors of the 2025 satellite initiative. Ask for the prior validation design and whether a secure aggregate join can be performed; do not request unrestricted raw data as the first move.
4. **Public geodata:** use the City's CGIS task request and contact the named custodian for any needed spatial layer, recording exact access and licence conditions rather than assuming the map viewer grants reuse rights. [CGIS request route](https://eservices.joburg.org.za/corporate-geo-informatics/overview).
5. **Go/no-go:** only freeze a satellite benchmark if independent repair outcomes, image timing, location uncertainty and data rights jointly support it. Otherwise stop at the documented gap and proceed on acoustic/utility-sensor methods without relabelling those results as satellite evidence.

## Claim ceiling

This playbook establishes no access to utility records, no consent, no dataset licence for the unlicensed Zenodo acoustic release, no actual Johannesburg imagery coverage, and no water-loss reduction or detector performance. It proposes a reproducible way to acquire and qualify evidence while preserving those distinctions.
