# WATER-LEAK-001 — Low-cost leak detection: Johannesburg transfer plan

**Status:** Research/design proposal. No prototype performance or field result is claimed.  
**Research snapshot:** 2026-10-12  
**Tracking issue:** [WATER-LEAK-001 #7413](https://github.com/Luminous-Dynamics/symthaea/issues/7413)  
**Composition:** [MFG-PORTFOLIO-001 #5752](https://github.com/Luminous-Dynamics/symthaea/issues/5752), [WATER-MFG-000 #5966](https://github.com/Luminous-Dynamics/symthaea/issues/5966), [WATER-MFG-001A #5997](https://github.com/Luminous-Dynamics/symthaea/issues/5997), [CIV-SERVICE-001 #5949](https://github.com/Luminous-Dynamics/symthaea/issues/5949)

## Decision

Do not begin by training an AI model on invented, public, or weakly labelled audio. Build a small, safe, repeatable test rig; collect controlled measurements; freeze a labelled benchmark; compare simple signal processing with Symthaea; only then attempt a permitted field pilot.

The first municipal-oriented prototype should be a **portable pipe-contact acoustic/vibration instrument**. A separate **stand-off acoustic profile** should evaluate the AquaPing reference for buildings and exposed plumbing. These are different physical sensing problems and their results must not be merged into one headline accuracy number.

## What current research establishes—and does not establish

### Johannesburg already uses acoustic detection

Johannesburg Water's [Active Leak Detection programme](https://www.johannesburgwater.co.za/about-us/campaigns/active-leak-detection-programme/) describes 15 technical teams surveying over 10,500 km of infrastructure annually, using listening sticks, ground microphones and correlators. The page reports annual averages of 1,456 burst pipes, 5,227 leaking meters, 341 leaking valves and 217 leaking hydrants detected, and approximately 8,100 ML saved. Treat these as the utility's published programme figures; they are not an independent evaluation of a new detector.

In a [23 July 2026 announcement](https://www.johannesburgwater.co.za/blog/2026/07/23/johannesburg-water-accelerates-smart-technology-rollout-to-reduce-water-losses/), Johannesburg Water said 225 noise loggers had been installed and a further 125 were planned by October 2026; it also described installation of smart pressure controllers and hydraulic flow controls. The announcement is a dated rollout plan, not proof that every planned device was installed by October. Verify the latest installed counts before any partnership proposal.

**Implication:** our value proposition must be incremental and measurable: lower total cost, maintainability, better detection in a documented difficult case, lower false-alarm burden, useful operation under weak connectivity, better handling of pipe-material differences, or improved alert-to-repair evidence. "AI can detect leaks" is not a differentiator by itself.

### AquaPing is a useful reference, but not a direct municipal-main solution

- The [AquaPing repository](https://github.com/microphonon/acoustic-leak-detector) publishes its source as CC0-1.0. Re-check the licence on each artifact before reuse and retain its origin in our provenance notes.
- Its [project page](https://www.crowdsupply.com/microphonon/aquaping) describes a stand-off acoustic sensor for pressurized plumbing, with local signal processing, configurable environmental training, and no audio streaming. The page displayed US$99 as the device price at research time; this is not a South African landed-cost quote.
- The associated 2025 paper, [*Ultralow-power standoff acoustic leak detection*](https://arxiv.org/abs/2511.00348), reports a simulated 0.15 L/min leak detected at more than 10 m under the authors' test conditions and power in the 20–200 µW range depending on noise/latency. It also states that leak geometry, ambient noise, acoustic paths and barriers affect detection, that smooth pinholes can be much quieter than jagged leaks, and that the device indicates a possible leak rather than measuring leak flow rate. These are paper-specific results, not independently reproduced Johannesburg performance.
- A [review of acoustic pipeline methods](https://doi.org/10.1016/j.autcon.2022.104226) identifies false alarms and weak-leak detection among unresolved challenges. Research on [acoustic sensors for plastic pipes](https://www.sciencedirect.com/science/article/abs/pii/S0022460X04005358) explains that signals in plastic pipes are more strongly attenuated and can be narrow-band/low-frequency, making detection and correlation harder than in metal pipes.

**Implication:** copy neither the AquaPing form factor nor its distance claim into the buried-main design without a specific transfer experiment. Use it as one open baseline for exposed/building plumbing, and examine pipe-coupled accelerometer/piezo sensing for the municipal profile.

### Data rights are part of the design

The [DWS water-quality data explorer](https://www.dws.gov.za/iwqs/wms/data/000key.asp) concerns water-quality observations, not acoustic leak telemetry. DWS guidance states that some supplied datasets are for academic, research or personal use and may not be sold or redistributed. Do not place restricted DWS data into a commercial training corpus or public product without written permission. Start with our own controlled bench data and separately verify licence/consent for every external dataset.

## Proposed architecture

### 1. Pipe-contact profile — first priority

A portable sensor is clamped or otherwise coupled consistently to an accessible valve, hydrant, exposed pipe or other approved contact point. Candidate transducers include a contact piezoelectric sensor and a small accelerometer. Capture a short, timestamped measurement using a suitable high-impedance/low-noise analogue front end or data-acquisition unit. The first prototype can be tethered; radio, battery optimisation and fixed mounting are follow-on requirements.

The first release reports **anomaly evidence and uncertainty**, not "leak confirmed." Do not require a pressure sensor in the first cut; add pressure, flow and pump-state context only where these are available and their clocks/calibration can be trusted.

### 2. Stand-off profile — separate experiment

Reproduce or adapt the CC0 AquaPing hardware/firmware only after checking individual file licences and required build tools. Evaluate it on exposed plumbing, a controlled wall fixture, a pump room or similar use case with safe access. Do not claim buried-pipe suitability from the published wall/free-space results.

### 3. Signal and inference stack

1. Validate recording integrity, clipping, sample rate, sensor attachment, timestamp and calibration state.
2. Compute deterministic baseline features: windowed RMS/energy, power spectral density or band energy, spectral flatness/entropy, persistence, and local noise-floor change. Keep the frequency bands and window sizes configurable and freeze them before held-out evaluation.
3. Establish simple threshold and statistical-change baselines.
4. Add a supervised model only after independently labelled recordings exist. Compare all models on the same frozen splits.
5. Integrate Symthaea only if it improves predeclared field-relevant metrics over the simpler baseline at acceptable CPU, memory, latency and power cost. HDC/CfC is a candidate, not a required assumption.
6. Return a compact event: sensor/site identifier, time range, feature-summary reference, configuration/calibration identity, confidence or score with calibration status, and recommended next inspection. Keep raw recordings local or in an access-controlled evidence store unless there is a justified need to retain them.

### 4. Mycelix evidence and operations

Mycelix may coordinate authenticated device identity, data provenance, calibration/currentness, alert acknowledgement, human confirmation, work-order state, repair observations and post-repair checks. It must not:
- turn a model score into a confirmed leak without independent evidence;
- treat a dispatched work order as a repair;
- infer litres saved from a detector score;
- store every raw waveform on a distributed ledger;
- authorize a valve, pump, pressure-control or grid action.

The water-quality and public-health authority remains independent. A leak detector gives no evidence that water is potable or safe.

## Test-rig outline

Use a low-pressure, recirculating water loop with a replaceable pipe section, controlled flow path, collection/measurement container, leak outlets of documented geometry, a pump, pressure gauge, and safe electrical separation/spill protection. Start at modest pressure; the fixture must have an independently reviewed pressure envelope and a safe shutdown. Do not use live municipal infrastructure, mains electrical enclosures, or unapproved public pipework for the prototype.

Test at least:
- steel/copper-like metal pipe and separately PVC/HDPE plastic pipe;
- multiple sensor positions and repeatable mounting/coupling pressure;
- no leak; known leak-orifice configurations; normal use/flow changes; pump operation; valve changes; tapping/handling; traffic-like background vibration; and adjacent plumbing noise;
- multiple pressure/flow conditions within the fixture's safe envelope;
- detached or degraded sensor, clipping, stale calibration, missing timestamps and out-of-profile conditions.

Measure actual flow through a controlled outlet independently when safe and feasible; the acoustic score itself is not a flow-rate measurement. Keep positive/negative fixture truth independent from the algorithm developer where possible.

## Benchmark contract

Every recording/session should bind to:
- immutable recording ID and file hash;
- device, sensor, analogue front end and firmware identifiers;
- sample rate, bit depth, gain/range and any filtering;
- pipe material, outside diameter/wall thickness where known, mounting location and coupling method;
- test-session ID, date/time, leak geometry, leak/no-leak label and independent truth method;
- pressure/flow values with units and calibration status where measured;
- pump/valve/ambient-noise conditions;
- train/threshold-selection/held-out partition;
- model/configuration hash and decision threshold.

Split by whole physical configuration and session/day, not random adjacent windows from the same waveform. Preserve unknown/unverified labels as unknown rather than silently forcing them into positive or negative classes.

Report denominators and uncertainty with at least:
- sensitivity/recall and missed-leak rate;
- precision and false alarms per sensor-day or monitored segment-day;
- calibration/reliability of any published confidence;
- localization error only if a localization method is implemented;
- detection latency, device power/runtime, installation/retest time and total per-site cost;
- baseline comparison (existing manual/DSP procedure), conventional DSP and Symthaea, separately.

Before opening the held-out set, freeze threshold-selection rules and decision thresholds against the actual operating trade-off. Avoid a universal target selected after seeing the test data. Keep synthetic, simulation, controlled bench and live-field results as separate evidence classes.

## Staged gates

| Gate | Exit requirement | What a pass does not establish |
|---|---|---|
| G0 — rights and scope | Hardware/software licence review; data rights and test-fixture safety review; declared target pipe profile | Performance |
| G1 — acquisition | Repeatable sensor recordings with metadata, hashes, signal-integrity checks and known-answer fixtures | Leak detection |
| G2 — baseline | Frozen feature extraction and simple baseline; positive and negative cases are repeatable | Field validity |
| G3 — held-out bench | Preregistered thresholds, configuration/session holdout, uncertainty and error analysis; Symthaea compared against baseline | Johannesburg transfer or cost savings |
| G4 — permitted silent field pilot | Written site permission; technician verification; no actuator path; record alert and follow-up outcomes | City-wide or all-pipe coverage |
| G5 — utility comparison | Compared with existing survey practice and existing logger options; cost per confirmed leak and alert-to-repair times measured | City-wide savings without scaled evidence |

## Provisional delivery plan

These are planning estimates for a small team with hardware access, not commitments:

- **Weeks 1–2:** verify requirements, licences, transducer/DAQ availability and local supplier quotes; write a safe fixture drawing and test-record schema.
- **Weeks 2–6:** build the tethered contact-sensor bench, data capture and deterministic DSP baseline.
- **Weeks 6–12:** collect repeatable positive/negative cases, freeze a benchmark, test cross-session and cross-material generalisation, and decide whether Symthaea adds enough value.
- **Months 3–6:** seek permission for a supervised, non-actuating field feasibility study. Access, procurement and utility review may extend this.
- **Months 6–18+:** only after pilot evidence, harden low-power hardware and maintenance workflows and integrate into utility operations.

A small bench device can be attempted within weeks; a credible field-tested product requires months. City-scale rollout depends on utility partnership, procurement, installation, repair capacity and evidence—not model development alone.

## Immediate next tasks

1. Obtain three local quotations for contact transducer + low-noise front end/DAQ, microcontroller/edge compute, enclosure and safe low-pressure fixture components; compare to a minimal borrowed/USB-DAQ setup before designing a PCB.
2. Create a canonical, versioned known-answer test matrix and independent stdlib-only validator before production Rust adapters. Minimum cases must include plastic-pipe attenuation, weak leak, pump/valve/noise negatives, sensor fault, stale calibration, and out-of-profile operation.
3. Make no commercial cost or accuracy claim until the bill of materials, lawful datasets, exact denominators and held-out evidence are reviewed.


## Satellite-assisted screening and satellite-connected telemetry

### Terminology: two complementary technologies

"Satellite telemetry" usually means transmitting measurements from a remote device via satellite. The primary idea here—using imagery to find leaks—is more accurately called **satellite remote sensing**. We should investigate both:

1. **Satellite remote sensing:** satellites observe indirect surface/subsurface moisture, vegetation or temperature anomalies to rank suspect areas for ground inspection.
2. **Satellite-connected telemetry:** low-power pressure, flow, acoustic or reservoir sensors send measurements over a satellite IoT link where cellular or utility communications are unreliable. This can monitor a known network point but cannot locate an uninstrumented buried leak by itself.

For dense urban Johannesburg, test remote sensing as a wide-area screening layer and ground sensors as the localization/confirmation layer. Satellite IoT connectivity is a separate later option for remote bulk mains, reservoirs or pump stations lacking reliable terrestrial communications.

### A local research lead to follow before duplicating work

A peer-reviewed 2025 chapter by Telukdarie and Nkadimeng documents a satellite-imagery leak-detection research initiative involving the City of Johannesburg's water department and the University of Johannesburg. Public metadata says SAR-based monitoring was proposed; it does not establish city-scale verified performance or tell us which parts are deployed now. Contact/coordinate with this research stream and ask for methodology, trial boundaries, leak labels, sensor/satellite sources, validation counts, false-positive causes, current deployment status and data-access terms before claiming novelty or building a competing pipeline.

Source: https://pure.uj.ac.za/en/publications/satellite-imagery-for-water-leak-detection-a-joint-research-initi/
DOI: https://doi.org/10.1108/978-1-83662-642-820251005

### What each data source is actually good for

| Source | Useful role | Critical limit |
|---|---|---|
| Sentinel-1 C-band SAR (open Copernicus data) | Repeat backscatter/change features; all-weather/night acquisition; cheap city-wide research baseline | Standard Interferometric Wide Swath imagery has about 5 × 20 m spatial resolution and roughly 6-day revisit at the equator for a two-satellite constellation. Moisture sensitivity is indirect; urban buildings, surface roughness, soil cover and mixed pixels complicate attribution. C-band results cannot simply inherit L-band claims. |
| Sentinel-2 multispectral optical imagery (open Copernicus data) | Vegetation/moisture-index changes, wet patches and land-cover context; 10 m bands and nominal 5-day revisit | Clouds, illumination, vegetation changes, irrigation, parks and mixed urban pixels create gaps/false alarms. It cannot directly see buried water through concrete or asphalt. |
| Landsat thermal infrared | Coarse surface-temperature anomalies over broad areas or large assets | Thermal bands have 100 m ground sampling; this is not a street-level pipe locator. Clouds, emissivity, shade, time of day and water/air temperature contrast matter. |
| NASA SMAP soil-moisture products | Regional rainfall/soil-moisture context, not individual pipes | Approximate 9 km products are far too coarse for individual urban leaks. Never treat SMAP pixels as pipe-level evidence. |
| Higher-resolution commercial L-band SAR analytics (e.g. ASTERRA Recover) | Potential partner/vendor comparator, designed specifically to rank likely treated-water leak locations | Proprietary costs and algorithms; vendor-published performance and “minimum detectable leak” claims are not independent guarantees. Require transparent denominators, local trial evidence, data rights, POI-to-confirmed-leak yield and a paid/pilot quote before procurement. |
| Drone/airborne thermal or hyperspectral survey | Follow-up over small suspect sites when surface moisture/temperature signals may be observable | More detailed but limited area, flight/weather/permission constraints, staffing and cost. It is still a screening measurement, not proof of a leaking pipe. |

References:
- ESA Sentinel-1 mission characteristics: https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1/Facts_and_figures
- ESA Sentinel-2 mission characteristics: https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/
- Open Copernicus data access: https://dataspace.copernicus.eu/explore-data
- USGS Landsat 9 thermal resolution: https://www.usgs.gov/landsat-missions/landsat-9
- NASA SMAP spatial resolution/context: https://science.nasa.gov/missions/smap/soil-moisture-mission-begins-science-operations/
- ASTERRA description of its satellite SAR technology: https://asterra.io/about/
- Arabi & Grau (2025), radar-multispectral fusion: https://doi.org/10.1016/j.rsase.2025.101594
- Ali et al. (2025), L-band SAR and machine learning, 6 ALOS-2 images; 80% reported accuracy, with generalization limitations: https://doi.org/10.5194/isprs-archives-XLVIII-G-2025-147-2025
- Optical leak-detection research with false positives due to rainfall, irrigation, natural water flow and vegetation: https://doi.org/10.3390/s22031057
- ASTERRA is a vendor/source for its product claims; treat the site's claimed leak thresholds and cumulative outcomes as vendor-reported, not independently validated Johannesburg performance.

### Johannesburg-specific pilot targeting and data stewardship

#### Do not conflate non-revenue water with physical leakage

Johannesburg Water's public reports distinguish non-revenue water (NRW) from physical/real losses. NRW can include physical losses (leaking mains, service connections, reservoirs), commercial/apparent losses (metering or billing issues), and authorized unbilled consumption. A satellite/moisture or acoustic pipeline product primarily targets a subset of physical-loss candidates. It cannot, by itself, detect or explain the full NRW metric, including commercial losses.

- Johannesburg Water Annual Report 2023/24 reports NRW of 46.18%; its breakdown identifies 24.83% as physical losses, 9.66% as commercial losses, and 11.69% as authorized unbilled use. These are historical 2023/24 figures and must not be represented as the 2024/25 breakdown: https://www.johannesburgwater.co.za/wp-content/uploads/2025/02/FINAL-JOHANNESBURG-WATER-ANNUAL-REPORT-23_2024-20-02-2025.pdf
- City of Johannesburg Annual Integrated Report 2025 reports audited NRW of 44.8% for 2024/25 versus 46.2% in 2023/24. This is the broader NRW KPI, not a satellite-detectable physical leak percentage: https://joburg.org.za/work_/Documents/Annual-Integrated-Report-2025/City-of-Johannesburg-Annual-Intergrated-Report-2025.pdf
- Johannesburg Water Business Plan 2025/26 also describes NRW as including real losses, commercial losses and authorized unbilled consumption: https://www.johannesburgwater.co.za/wp-content/uploads/2025/07/JHB-Water_Business-Plan-2025_For-Upload_FINAL_070725.pdf

Never estimate water saved by multiplying an anomaly count or a model's conditional hit-rate by total NRW. Quantify savings from measured leak flow/validated repair outcomes where those data are available and water accounting permits the attribution.

#### Candidate assets are a hypothesis, not a target list yet

The City of Johannesburg Annual Integrated Report 2025 describes 42 leaking reservoirs having been identified, with 23 considered critical, and discusses an inventory of 730 zonal pressure-reducing valves (PRVs), including many previously vandalized or inoperative. It reports 211 PRVs (29%) had been returned to operational status during 2024/25 and 26 smart pressure controllers were added that year, bringing the report's stated installed total to 51. Those numbers are dated annual-report descriptions, not a verified current asset register or evidence that any listed asset remains faulty in October 2026.

These facts justify asking the data owner whether a suitable, currently valid pilot site exists; they do not justify declaring a leak location. Consider two candidate profiles, chosen only after data and permission review:

1. **Known, physically confirmed leak/repair segment:** best for a retrospective sensor/image study because event time, spatial uncertainty and outcome can potentially be aligned.
2. **Authorized reservoir or PRV/pressure-zone site:** potentially useful for a prospective multimodal pilot because asset location and operational context may be known. Use pressure/flow/controller telemetry and technician inspection as primary evidence; satellite surface anomalies are only supplementary and may be too coarse to resolve a small asset.

A pilot should choose the area where labels, repeat imagery, field access and operational comparator are strongest—not simply the place with the most alarming reported count.

#### Data-access route

The City of Johannesburg Corporate Geo-Informatics (CGIS) site describes a Spatial Data Inventory with dataset names, descriptions, formats, dates, data owners/contacts and usage limitations; digital data requests follow its GIS/task-request process. Use it to identify the right custodian and request permission instead of assuming water-main GIS is public or available for commercial use.

- City CGIS maps / Spatial Data Inventory entry point: https://eservices.joburg.org.za/sites/cgismaps/Pages/default.aspx
- City CGIS overview and task-request information: https://eservices.joburg.org.za/corporate-geo-informatics/overview

Request the minimum necessary, permissioned layers: pipe centerlines and spatial accuracy; material/diameter/age where recorded; pressure zones, PRVs and reservoirs; timestamped confirmed leak/repair work orders; inspected no-leak outcomes; maintenance/inspection method; and any available flow/pressure measurements. Record owner, licence/allowed purpose, permitted retention, spatial precision, timestamp quality, suppression rules and whether derived model outputs may be commercialized or redistributed.

The 2025 University of Johannesburg / Johannesburg Water satellite research initiative is a priority partner-discovery lead. Ask what area, sensor/wavelength, leak records, validation design and operating status it actually covered before running an overlapping study:
https://pure.uj.ac.za/en/publications/satellite-imagery-for-water-leak-detection-a-joint-research-initi/

### Why fusion is more credible than satellite-only detection

A wet or thermally unusual area is not automatically a leak. Confounders include rainfall, irrigation, groundwater seepage, stormwater, leaking sewers, broken drains, ponds, changes in vegetation, slope/runoff, soil composition, excavation, surface materials, shadow, traffic, and land-cover change. Satellite anomalies can also miss leaks that stay contained underground, are small/short-lived, occur under impermeable surfaces, or produce a signal below the sensor's spatial resolution.

The proposed output is a **ranked area/pipe-segment inspection queue with explicit uncertainty and contributing evidence**, not a map that asserts every flagged coordinate is a leak.

### Candidate open-data experiment

Begin with a small, historical, retrospective pilot, not an all-city model:

1. Agree on a permitted spatial boundary and evaluation time window with a data owner. Obtain a versioned GIS water-main network and historical confirmed-leak/repair records with timestamps, pipe class/material if known, and investigation outcomes. Do not infer leak coordinates from vague service complaints or label every unreported location as a true non-leak.
2. Use free Copernicus Sentinel-1 scenes as the primary open SAR baseline; add Sentinel-2 when valid cloud-free imagery exists. Add time-matched rainfall/weather, land cover, terrain/slope, drainage/irrigation context, network topology and repair history where permitted. Record acquisition dates, processing versions and per-pixel quality masks.
3. Estimate temporal anomaly features against a local seasonal baseline; explicitly mask water bodies, invalid/layover areas where applicable, fresh excavation, heavy cloud for optical scenes, and pixels with poor observation quality. Keep rainfall/irrigation controls as features and evaluation strata, not informal caveats.
4. Compare three methods on the same frozen cases: a non-AI baseline, conventional statistical/ML baseline, and optional Symthaea-assisted fusion. Avoid a complex model until enough independently labelled outcomes exist.
5. Split evaluation by geography and time—for example, hold out whole zones and later periods—not random adjacent pixels from the same scene. Make sure the same leak event and its correlated image neighbourhood cannot appear in both train and test.
6. If utility rights permit, benchmark commercial L-band analytics on the same study area and compare every method against the same field-confirmed outcomes. Do not assume Sentinel-1's C-band will reproduce commercial L-band results.
7. Only proceed to a prospective pilot if the retrospective study shows a useful signal and the field owner approves the protocol. A blind crew confirms or rejects points using standard acoustic/correlation practice; record all investigated points, not just successful discoveries.

### Metrics that determine whether satellite screening adds value

Measure:
- precision/yield among the top N points a field team can realistically inspect;
- confirmed leaks per 100 inspections and per crew-day versus current practice and existing logger/acoustic workflows;
- recall on a timestamped set of confirmed leaks that should have left an observable signal;
- false-positive causes and false-positive rate by land cover, weather/rainfall, pipe type and urban setting;
- spatial error between the satellite point-of-interest and the field-confirmed leak, distinguishing a broad anomaly zone from actual leak localization;
- cost per confirmed leak, analyst time, data/licensing cost, time from imagery acquisition to action, and repair verification;
- abstention/unknown rates when cloud, mixed pixels, outdated network maps or weak ground labels prevent a reliable score.

Preregister the acceptance threshold based on the utility's cost per inspection and crew capacity. Do not set a universal accuracy target after looking at the test outcomes. Class imbalance makes overall accuracy insufficient on its own.

### Staged plan

- **2–4 weeks:** scope the Johannesburg/UJ workstream, obtain data rights and map layers, make a Copernicus acquisition inventory for a small permitted area, and test preprocessing/repeatability.
- **4–8 additional weeks:** produce a frozen retrospective benchmark if adequate historical leak labels and valid imagery exist; compare no-model, conventional ML and optional Symthaea. If the labels or image archive are insufficient, report that as a no-go/blocker rather than manufacture cases.
- **3–6 months or longer:** prospective silent-mode field pilot, requiring written access, technician verification, operational baseline and outcome tracking. Procurement and municipal coordination may dominate the timeline.

### Satellite-connected telemetry: a second lane

Where remote trunk mains, reservoirs or pump stations lack reliable cellular connectivity, investigate low-power satellite IoT for **known instrumented points**: pressure trends, flow/level measurements, pump health, acoustic features and heartbeat/battery health. Prefer sending compact features and alerts rather than continuous raw audio. Compare satellite modem/device cost, antenna visibility, power budget, message latency, subscription costs and coverage against cellular/LoRaWAN/store-and-forward alternatives. Satellite messaging can transport sensor readings; it does not make an uninstrumented pipe observable.

This lane should share device identity, calibration, freshness and provenance semantics with the Mycelix operations plane. Sensor messages and Mycelix work-order events remain operational facts, not proof of water quality or leak confirmation.

## 2026 evidence update: radar source selection and independent comparison

Research reviewed on 2026-10-12 adds two decision-relevant sources:

### Arabi & Grau (2026): paved-surface pipeline leaks using L-band radar

Arabi and Grau, *Automated detection of pipe water leaks under paved surfaces with machine learning analysis of satellite radar data*, Journal of Environmental Management 403 (1 April 2026), 129209. DOI: https://doi.org/10.1016/j.jenvman.2026.129209

- The study investigates anomalies from ALOS-2 PALSAR-2 **L-band polarimetric SAR** in a paved, arid, sparsely vegetated setting, using leak records supplied by the City of Phoenix.
- The abstract reports that its best unsupervised anomaly-detection model identified about 81% of actual leaks and reduced field inspection effort by 93%; the abstract also describes false-positive trade-offs as thresholds change.
- Do not reuse these figures as expected Johannesburg performance. Before cross-study comparison, obtain the paper's complete evaluation definitions and denominators (what counts as a leak found, a false positive, and an inspection reduction), sample counts, selection procedure, geography and all thresholding details. The reported operating point must be read together with false alarms and the crew's true inspection capacity.
- This is strong support for evaluating L-band/PALSAR-2 as a candidate modality for buried pipes, not evidence that Sentinel-1 C-band alone will work equally well.

Source records:
- PubMed bibliographic record and abstract: https://pubmed.ncbi.nlm.nih.gov/41785796/
- ScienceDirect article: https://www.sciencedirect.com/science/article/pii/S0301479726006699
- JAXA PALSAR-2 ScanSAR L2.2 products: https://www.eorc.jaxa.jp/ALOS/en/dataset/palsar2_l22_e.htm

### Independent programme evaluation: California Energy Commission (2021)

The California Energy Commission published the final report *Demonstrating Innovative Water Leakage Reduction Strategies* (CEC-500-2021-036). Its project compared satellite imagery leak pre-location, continuous acoustic monitoring and flow-sensitive pressure-reducing valve systems. The official abstract says both satellite imagery and acoustic monitoring located subsurface leaks and suggests utilities consider combined deployment according to network, budget and operating conditions.

Source: https://www.energy.ca.gov/publications/2021/demonstrating-innovative-water-leakage-reduction-strategies-correlating

The official report's Table 33 gives the underlying denominators: satellite imagery (Utilis/SILD) supplied 504 points of interest (POIs), of which 146 were investigated (29%); crews found 117 leaks, or 80% of the investigated POIs. Continuous acoustic monitoring (Echologics/CCAM) supplied 77 POIs, of which 49 were investigated (64%); crews found 20 leaks, or 41% of investigated POIs. Table 15 separately reports **leaks per POI provided** as 23% for satellite imagery versus 26% for acoustic monitoring. Table 15 also notes that the acoustic provider filtered its larger stream of candidate signals before supplying a smaller recommended set, whereas satellite imagery supplied substantially more POIs. Therefore, the denominator is influenced by each vendor's detection/triage policy and this is not a matched randomized quality comparison. The satellite results covered 45 miles of pipe investigated, at 2.6 leaks per mile, and averaged 5.6 labor-hours per leak found; acoustic monitoring covered 5 miles, at 3.8 leaks per mile, and averaged 4.9 labor-hours per leak found. The report also gives 3.8 labor-hours per investigated satellite POI versus 1–3 hours per acoustic POI.

The 80% number is a conditional yield among investigated satellite POIs, not recall across all candidates: 358 of 504 satellite POIs were not investigated. The report states that approximately 40 POIs were generated per monthly flyover and the provider supplied an ordered list based on crew availability; it does not establish that the 146 investigated satellite points were a random sample. Thus, the evidence suggests broad satellite pre-location can expand coverage, but not that it wins on every metric: in this study it had higher yield per investigated POI but lower leak yield per POI provided, fewer leaks per mile investigated, and more estimated labor per leak found. Satellite and acoustic results are **different operational measures**, not a single winner score. Private-side irrigation leaks, shadow effects, POI area size, local geology/land cover, crew selection and field-label uncertainty all affect interpretation. The study is not a randomized head-to-head performance estimate and does not establish 93% inspection savings in Johannesburg. Any comparison must retain all POI denominators, selection process, per-mile yield, labor burden, false-positive categories and how leak/no-leak ground truth was established.

Source: official report PDF, printed page 108 / Table 33: https://www.energy.ca.gov/sites/default/files/2021-06/CEC-500-2021-036.pdf

### Revised source-selection decision

1. **Do not make Sentinel-1 C-band the only candidate.** Keep it as the low-cost, open-data baseline, since open repeat coverage is valuable and scene inventory can be reproduced.
2. **Investigate L-band feasibility in parallel.** Check historical ALOS-2 PALSAR-2 observations and rights for the selected Johannesburg pilot area. JAXA's ScanSAR L2.2 archive describes free products subject to each distribution platform's terms; commercial rights must be confirmed on the actual item/platform. Data currently documented by JAXA covers historical periods through July 2025 in the stated L2.2 archive, with subsequent data released progressively; do not assume all latest observations are in the archive. JAXA's separate [open/free release page](https://www.eorc.jaxa.jp/ALOS/en/dataset/alos_open_and_free_e.htm) also documents event-specific ALOS-2/ALOS-4 releases restricted to non-commercial use and directs commercial-use requests to JAXA. Treat this as evidence that terms can differ by product/release—not as proof that every item in the baseline L2.2 archive is non-commercial.
3. **Seek access to the local research design before duplication.** The UJ/Joburg Water chapter establishes that satellite-based monitoring/SAR was proposed as a joint research initiative, but its public abstract does not confirm current deployment status or give a reproducible validation set. Ask for target geography, radar source/wavelength, labeled leak records, sample size, validation method, false-positive distribution, current operational status and permitted sharing.
4. **Keep commercial product use gated.** The full rights chain must cover source imagery, derived products, historical leak labels, model training, and distribution of outputs. "Free to download" or "public for research" is not enough for unrestricted commercial use.

**Interpret zero-result inventories narrowly.** The [official Copernicus Data Space STAC documentation](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) states that STAC is complementary to the Open Data Protocol (OData), and currently covers a limited set of collections. Therefore, zero Sentinel-1/Sentinel-2 items from this workflow means only that no items were returned for the exact endpoint, collection, AOI and time-window query at that retrieval time. It does **not** establish that no relevant acquisitions exist anywhere in the broader Copernicus catalogue. Before declaring acquisition coverage unavailable, cross-check the identical AOI/time window and product family through the [official CDSE OData catalogue](https://documentation.dataspace.copernicus.eu/APIs/OData.html), retain the query and retrieval timestamp, and record any mismatch as a catalogue-access gap rather than a global no-data conclusion.



### Independent OData catalogue cross-check

The [OData inventory utility](https://github.com/Luminous-Dynamics/symthaea/blob/research/water-leak-001-johannesburg-transfer/tools/water_leak_satellite/odata_inventory.py) provides an independent metadata-only cross-check via the official Copernicus Data Space Products API. It uses the documented `Collection/Name`, `productType`, `OData.CSC.Intersects`, `ContentDate/Start`, `$select`, `$orderby`, `$top` and `@odata.nextLink` interfaces. It supports calendar-month, calendar-quarter and calendar-year partitions; the live workflow now defaults to calendar quarters after the earlier 45-second broad-window request timed out. Each request has a 30-second timeout and at most two retries for bounded transient network failures or HTTP 408/429/5xx responses. Successful page receipts record request-attempt counts; OData inventory schema v2 binds the chosen partition and retry policy into the manifest, and the independent verifier checks those values along with AOI/filter hashes, page counts and item summaries. Exhausted retries, a repeated pagination URL or incomplete pagination fails the inventory instead of yielding a zero-result record.

OData and STAC are independent inventories, and discrepancies should be investigated rather than silently merged. A product record or intersecting footprint is still not proof of useful resolution, acquisition suitability, a leak or commercial rights. The earlier OData cross-check timed out and did not produce a valid completed inventory. A refreshed metadata-only workflow is queued at [run 38576061525](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38576061525); no result from that new run is claimed yet.

### Reproducible catalogue inventory utility

The feature branch now includes a standard-library-only catalogue metadata collector:
- `tools/water_leak_satellite/catalog_inventory.py`
- `tests/test_water_leak_satellite_catalog_inventory.py`

Run from the repository root to perform only catalogue reconnaissance:

```sh
python3 -m unittest discover -s tests -p 'test_water_leak_satellite_catalog_inventory.py' -v
python3 tools/water_leak_satellite/catalog_inventory.py \
  --bbox 27.7,-26.5,28.4,-25.8 \
  --start 2023-01-01 --end 2026-10-12 \
  --out artifacts/water-leak/satellite-catalog-inventory.json
```

The bounding box is a coarse Johannesburg reconnaissance extent, not an approved evaluation area. The utility queries Copernicus STAC for Sentinel-1 GRD and Sentinel-2 L2A, writes item metadata plus per-item and per-page SHA-256 hashes, records the explicit query and retrieval time, follows only HTTPS pagination links on the official STAC host, deduplicates repeated item IDs and fails closed when pagination is ambiguous or incomplete. It limits each request to 100 items, matching the Copernicus STAC item-search cap, and atomically publishes the JSON file so interrupted writes do not leave a partial inventory. See the [official STAC API documentation](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) and the [Copernicus support note on the 100-item item-search limit](https://forum.dataspace.copernicus.eu/t/waf-rate-limit-exceeded-status-429/4857). The query uses the STAC Fields Extension to exclude the bulky `assets` object during this first-pass inventory, while retaining geometry/bbox, selected acquisition properties and pagination links. Asset metadata returned by the service is flagged explicitly so an omitted `assets` field is not misread as evidence that a scene lacks data assets. The hash for each item therefore binds the field-projected item representation returned by the server, and the page hash binds the raw page response; neither claims to hash excluded asset payload metadata. The output stores compact item summaries, omits asset hrefs, and redacts query strings from stored self/pagination URIs while recording a separate hash of the full URI. This reduces payload size and avoids retaining signed download URLs or opaque tokens in artifacts. Later asset-level review/download is a separate stage. Each item summary carries its own SHA-256, and the full JSON inventory carries a manifest digest. A separate no-network verifier checks those stored hashes/counts/redaction invariants, and the workflow prints the manifest digest alongside the summary. This detects inconsistency relative to stored digests only; it does not authenticate the provider or establish leak evidence. This verifies the manifest, per-item summary hashes, digest shapes, declared counts and URL-redaction contract, but does not authenticate the data provider or establish scene suitability. It does not download raster imagery or claim that scenes are suitable for detecting leaks.

The offline suite now contains **57 tests**: 13 STAC catalogue-inventory tests, 14 ground-truth-contract tests, 11 STAC inventory-integrity tests, 4 strict-JSON parsing tests and 15 OData inventory tests. The strict JSON decoder rejects duplicate objAs of 2026-10-14, the prior hosted STAC inventory run [38294607122](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38294607122) produced an auditable metadata artifact for the coarse Johannesburg reconnaissance bbox: 313 Sentinel-1 products across four pages and 2,161 Sentinel-2 products across 22 pages, with four intersecting MGRS tiles. Of the Sentinel-2 catalogue entries, 1,271 report tile-level cloud cover at or below 20%; this is a screening attribute, not a guarantee of clear pixels along a pipe. The STAC inventory manifest SHA-256 `917ec77a38aa278846645afca028221ba9b5d7f5b45f7595779fdf85bd580c74` was independently recomputed and matched. The OData job in that run timed out at a 45-second request setting, so it does not support a zero-result conclusion.

The OData implementation and workflow have since been updated to query calendar quarters by default, use a 30-second timeout with at most two bounded transient retries, and record the attempts in a versioned manifest. The dedicated refresh is [queued at run 38576061525](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38576061525); the offline inventory tests are also queued at the current branch head. Treat the old STAC counts as catalogue evidence for a coarse bounding box only. No rasters have been downloaded, no approved utility AOI or segment-level repair sample has been established, and no leak localization or model performance has been demonstrated.

As of this revision, the offline suite contains **60 tests**: 13 STAC catalogue-inventory tests, 14 ground-truth-contract tests, 11 STAC inventory-integrity tests, 4 strict-JSON parsing tests and 18 OData inventory tests. The new retry regressions cover transient timeout recovery, bounded exhaustion without zero-result conversion, and retry-limit validation; quarter-partition tests cover exact half-open date coverage. The latest exact-head hosted run is still queued, so no PASS is claimed for these latest changes.

The live inventory is isolated in `.github/workflows/water-leak-satellite-inventory.yml` and runs only through a deliberate trigger-file push or workflow dispatch. Its default AOI remains coarse reconnaissance, not an approved evaluation area. The first goal is to qualify the catalogues and rights; only then should the project seek a permissioned repair-centred sample and retrieve selected raster assets.



As of 2026-10-12, I retrieved the official STAC collection metadata pages for `sentinel-1-grd` and `sentinel-2-l2a`. That confirms the collection identifiers, product descriptions and endpoints exist; it does not establish any scene intersection, acquisition continuity or data suitability for the reconnaissance area. Direct item-search URLs were inaccessible through the research browser, and the working container could not resolve external DNS. Do not claim a live AOI inventory or model/test PASS beyond the locally reproduced offline suite until a completed hosted run and its artifact are independently inspected.

### Data acquisition playbook

The complementary source ladder, repair-response packet, prospective local cohort and public benchmark rights gates are specified in [WATER_LEAK_001_DATA_ACQUISITION_PLAYBOOK.md](WATER_LEAK_001_DATA_ACQUISITION_PLAYBOOK.md). The key priority is repair-centred event linking: pre-repair observations + independent inspection + verified repair + post-repair observations + matched controls. This does not replace the strict ground-truth contract below.

### Ground-truth label and data-rights contract

A new strict record schema and standard-library-only validator define a safe handoff for any future labelled dataset:
- Schema: `docs/research/water_leak_ground_truth.schema.json`
- Validator: `tools/water_leak_satellite/ground_truth.py`
- Offline tests: `tests/test_water_leak_satellite_ground_truth.py`

The contract separates four outcomes: `confirmed_leak`, `inspected_no_leak`, `inconclusive`, and `unknown`. An acoustic localization, complaint, or work-order entry alone cannot become a confirmed positive; a negative requires an actual documented inspection, not the absence of a leak record. The validator also requires coordinate uncertainty, timezone-aware observation time, permission evidence and explicit permitted purposes. Commercial training fails closed unless the rights record explicitly allows it. A spatiotemporal split group, zone or pipe-segment ID cannot cross evaluation partitions, and held-out test observations must occur after all training/threshold-validation timestamps. These are guardrails for spatial/temporal leakage, not a substitute for an independently frozen split manifest.

Running this validator only establishes consistency with the declared data/label contract. It does not prove the underlying field claim is true, resolve biased sampling, prove representativeness, or qualify a model. Raw personal information and residential addresses must not be stored in the public corpus; use permissioned source references and pseudonymous asset IDs. Coordinate detail should be access-controlled in the utility's authorized store when necessary.

### Near-term go/no-go decision

The immediate scientific question is not "can an algorithm draw a hotspot map?" It is whether repeatable satellite-derived anomalies overlap with independently confirmed leaks often enough to reduce real inspection burden in the selected area.

Do not start model training until a feasibility packet contains:
- actual catalog results for Sentinel-1 and Sentinel-2 covering the exact agreed area/time window;
- an ALOS-2 PALSAR-2 availability and rights assessment for the same area;
- historical confirmed leak/repair records with usable timestamps, spatial accuracy and outcome labels;
- image-to-event matching tolerance justified by imagery resolution and the field-label accuracy;
- a record of missingness, acquisition geometry, weather/land cover, and confounders;
- a predeclared top-N inspection budget and baseline practice for evaluation.

If leak labels or spatial accuracy are not sufficient, stop at the data-gap finding and seek a utility/university partner; synthetic examples must not be presented as remote-sensing evidence.

### Satellite-specific claim ceiling

A satellite anomaly establishes at most a remotely sensed surface/subsurface indicator consistent with excess moisture under a stated sensor/model profile. It does not by itself establish a water-main leak, potable water identity, leak flow rate, precise pipe location, water savings, pipe condition, or repair necessity. Every operational leak claim requires independent field verification under a recorded method.

## Claim ceiling

This document is a research and design plan. It establishes no functioning detector, current local build price, sensitivity, specificity, battery life, localization precision, Johannesburg transfer performance, water saved, commercial rights to third-party data, utility acceptance, drinking-water safety or operation authority.
