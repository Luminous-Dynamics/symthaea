# Flagship Audio Competitive Challenge: ABYSS-Class Headphones and Arendal-Class Loudspeakers

**Status:** research and qualification plan; not a product specification freeze, test result, or superiority claim.  
**Snapshot:** 2026-10-17.  
**Tracker:** [FLAGSHIP-AUDIO-001 #7848](https://github.com/Luminous-Dynamics/symthaea/issues/7848).  
**Portfolio:** [OPEN-AUDIO-000 #7826](https://github.com/Luminous-Dynamics/symthaea/issues/7826); [product-line roadmap PR #7827](https://github.com/Luminous-Dynamics/symthaea/pull/7827).

## Executive proposition

We should attempt to make headphones that can compete with the ABYSS AB-1266 Phi TC, and a loudspeaker system that can compete with the Arendal Sound 1528 Tower 8. But we must define “better” as a multidimensional, independently tested result—not as a marketing adjective or one frequency-response plot.

The paradigm shift is to publish an **open, repeatable qualification system** around every physical product:

- acoustics tuned with declared targets and validated fixtures;
- robustness across real fit, reseating, off-axis angles, rooms, builds, and component lots;
- distortion, compression, output headroom, electrical load and thermal limits measured at stated conditions;
- listening preference tested separately from measurement accuracy;
- repairability, comfort, lifetime cost and source completeness treated as first-class product qualities;
- all prototypes, analysis versions, negative results and uncertainty linked to an exact hardware revision.

We may beat a flagship in one dimension and lose in another. We must report that honestly. The actual design has to be built and measured before a claim of sonic superiority is justified.

## 1. Reference products and limits of the published claims

### Headphone reference: ABYSS AB-1266 Phi TC

The ABYSS product page lists nominal impedance **50 Ω**, sensitivity **88 dB/mW**, frequency response **5 Hz–45 kHz**, average THD **0.03%**, and mass **640 g**. These are manufacturer-published values; they are not automatically comparable with measurements we later make unless the definitions, fixtures, drive levels, weighting and included accessories match. Source: [ABYSS AB1266 TC product page](https://abyss-headphones.com/en-gb/products/abyss-ab-1266-phi-headphone).

The flagship challenge is not simply to copy its driver architecture. It is to determine where a new open headphone can deliver equal-or-better independently measured acoustic performance with a substantially more wearable, repairable and accessible design. Lower mass alone cannot prove better sound, and a prettier response trace alone cannot prove better comfort or preference.

### Loudspeaker reference: Arendal Sound 1528 Tower 8

The manufacturer lists a 3-way system with a 28 mm tweeter, 5-inch midrange and four 8-inch woofers; **89.5 dB / 2.83 V / 1 m** sensitivity; anechoic **-6 dB response from 26 Hz to 20 kHz**; typical in-room bass response to **16 Hz**; frequency linearity of **42 Hz–20 kHz ±2 dB**; and stated maximum output **118 dB peak / 112 dB RMS** under its cited full-bandwidth conditions. Cabinet mass is listed at **79 kg**, excluding the Puck+ accessories. Sources: [official technical page](https://arendalsound.com/product/1528-tower-8/tech/) and [manufacturer product handbook](https://arendalsound.com/wp-content/uploads/2025/02/2025_02_Product_Handbook_Arendal_1528_Speakers_8_v2.5_WEB.pdf).

These figures are a useful reference for the scale of the challenge, not a substitute for independently measured results under one agreed protocol. Arendal’s 118 dB peak / 112 dB RMS specification must not be equated directly with an AES75 result unless signal, bandwidth, distance, environment and pass criteria are comparable.

### Independent measurement reconnaissance (not a substitute for our common protocol)

- **ABYSS AB-1266 Phi TC:** [Hi-Fi News' lab report](https://www.hifinews.com/content/abyss-ab-1266-phi-tc-headphones-lab-report) reports ten reseated artificial-ear measurements, 98.4 dB SPL at 1 V / 1 kHz, impedance extrema of about 42.0–42.1 Ω over its stated band, and distortion below 0.1% at 100 Hz and 0.1% at 1 kHz for its 90 dB SPL test. Its interpretation also notes reticent/curtailed treble with a perceived-response depression around 5 kHz. The report's mass is 646 g including cable and 1/4-inch connector, so it is not directly interchangeable with the manufacturer's 640 g figure without matching weigh-in conditions.
- **Fit is a first-order variable:** [Kuulokenurkka's AB1266 Phi TC review](https://kuulokenurkka.com/en/review-abyss-ab1266-phi-tc/) reports IEC 711-type measurements and describes bass response changing materially with the ear-pad/head gap. This is one reviewer/fixture's observation, not a population study; it is enough to require a fit/seal sensitivity experiment rather than assume repeatability.
- **Arendal measurement gap:** [Spinorama's public measurement catalogue](https://www.spinorama.org/) surfaced an entry for the Arendal Sound **1610 Tower 8**, an adjacent model, but this research pass did not establish an apples-to-apples independent standardized measurement package for the exact **1528 Tower 8**. The 1610 may be a useful additional comparative design point; it must not be substituted as evidence for the 1528. The exact 1528 should be bought or borrowed and measured under the same declared protocol, or the missing comparison must remain explicit.

Use these reports to formulate test hypotheses, not to declare either product good or bad from isolated values. Their fixtures, response compensation, samples, test signals and reporting formats differ, so values cannot be pooled as if they were one laboratory experiment.

### Required competitor selection

Before confirmatory measurements, freeze a shortlist that includes:

- ABYSS AB-1266 Phi TC for the reference-class headphone comparison, plus at least two relevant alternatives spanning strong measured performance and a meaningfully different price/fit trade-off;
- Arendal 1528 Tower 8 for the reference-class loudspeaker comparison, plus at least two relevant alternatives with different design approaches, including at least one strong measurement-led competitor;
- exact model, revision, condition, pad/cushion state, firmware, accessory set, purchase/loan provenance and test date.

Do not select competitors after looking at our prototype results. If budget only allows one sample per competitor, disclose that the sample does not establish population or manufacturing-lot variation. Borrowing or independent-lab collaboration is preferable to silently narrowing the claim.

## 2. Paradigm shifts worth pursuing

### A. Personal audio: co-design the driver, acoustic load and fit

**Shift from “one driver + one tuned cup” to a coupled human-interface acoustic system.** Driver selection, ear-cup volume, pad seal, opening geometry, damping, headband force and ear clearance should be explored as linked variables. HRTF/ear-shape personalization can be a later opt-in layer, but it must not obscure the passive baseline or imply that a single compensation curve fits everyone.

Practical experiments:
1. Measure driver impedance, sensitivity, distortion and pair-to-pair variation before cup design is frozen.
2. Quantify fixture reseating, pad seal and geometry repeatability first.
3. Use deliberately isolated experiments: change one controllable parameter at a time or use a preregistered factorial design.
4. Separate tuning data from held-out confirmatory placements and units.
5. Test comfort and fit across head sizes, glasses, hairstyles and different ear shapes; publish the distribution rather than a single “fits everyone” statement.

### B. Headphone ownership: comfort, weight and repair are performance

A headphone that measures well but causes hot spots, slips, traps heat or is difficult to service is not a complete flagship. Use a low-mass frame, replaceable suspension and pads, robust standard fasteners, serviceable cable sockets and parts with explicit compatibility/revision IDs. Instrumented pressure measurements can support the work, but cannot replace wearer-reported comfort.

### C. Loudspeakers: optimize directivity and output, not just the on-axis curve

**Shift from “flat at one microphone point” to “controlled radiation into a listening region.”** Crossover integration, waveguide geometry, baffle diffraction, cabinet edge behavior, vertical lobing, power response, compression and room interaction are one system problem.

Our higher-end path should investigate a modular 3-way active reference system with independently powered/limited ways, a controlled-directivity mid/high section, separately optimized bass/subwoofer modules and documented DSP. This is a research direction, not a chosen driver topology. Keep a passive S1 two-way as the affordable, simpler first speaker. Active DSP cannot repair every off-axis defect or eliminate every room null, so publish raw and corrected results side by side.

### D. Room integration: publish the listening area, not one magic seat

Measure multiple positions and show pre/post correction. Optimize an agreed listening area with capped boosts, sensible low-frequency headroom and regularization. The default must remain usable offline, without a cloud account. A correction profile is a versioned artifact linked to the measured speakers, microphone calibration, room positions and software revision.

### E. Open hardware as the product’s trust architecture

Every sold unit/revision should link editable CAD/PCB sources, BOM and alternates, firmware source and build steps, test profiles, calibration identity, raw captures, analysis version, repair guide and known limitations. Make a product’s performance independently falsifiable. Open files without repeatable tests are incomplete; a good test report without buildable source is not a complete open-hardware release.

## 3. Provisional flagship headphone challenge (H-FLAGSHIP)

These are **internal aspiration targets** for design exploration, not measured results or standard requirements. Freeze the test method before tuning against the target. Any threshold that proves ill-posed under fixture repeatability must be revised *before* confirmatory data are inspected.

| Dimension | Provisional target / gate | Notes |
|---|---|---|
| Mass | **≤450 g** for the complete headphone with pads, excluding detachable cable | An ambitious ergonomic target against ABYSS’s listed 640 g; weigh both products on the same defined basis. Do not let lower mass compromise clamping stability or durability. |
| Frequency response | **≤2.0 dB log-frequency RMS deviation** from a frozen target over 100 Hz–8 kHz on a validated fixture | Declare normalization, frequency weighting, smoothing and target. Also publish raw curves and reseat distributions; target matching is not universal preference. |
| Channel matching | Median L/R mismatch **≤1.0 dB** over 100 Hz–8 kHz; 95th-percentile mismatch **≤1.5 dB** across declared reseats/builds | The fixture’s own repeatability must be reported alongside these goals. |
| Distortion | Aspiration: THD **≤1% at 94 dB SPL** from 100 Hz–10 kHz; low-frequency THD target to be separately frozen after feasibility runs | Report frequency-by-frequency curves at multiple safe, declared fixture levels. Do not infer this from a maker’s single aggregate THD value. |
| Wearability | At least **80% of a preregistered pilot cohort** rates a 2-hour session acceptable without a stop due to discomfort | Exploratory human comfort study only; record cohort, head/ear fit, glasses, pressure hotspots, heat, stability and survey instrument. This is not a medical or universal-fit claim. |
| Build consistency | At least 10 units across two builds/lots for release-candidate characterization, where feasible | Report spread and rejects, not just the median. If this is initially unaffordable, label the sample size and claim limits. |
| Service | Pads, cable, suspension/headband and each designated wear part replaceable using documented tools without destructive disassembly | Time the repairs with an independent builder; record broken fasteners/parts and post-repair acoustic changes. |
| Openness | 100% of first-party mechanical design source, BOM, drawings, assembly guide, licenses/provenance and measurement manifest included | A mesh export alone does not satisfy the open-source contract. |

Additional mandatory plots: sensitivity and impedance versus frequency; THD by frequency and level; intermodulation distortion when instrumentation permits; response differences under reseating and different seal conditions; mass and clamp-force distribution; build and repair time.

**Important:** do not claim that H-FLAGSHIP beats ABYSS merely because it is lighter or meets the target curve. A defensible claim requires same-fixture measurements and appropriately controlled listening with representative samples.

## 4. Provisional flagship loudspeaker challenge (S-FLAGSHIP)

Treat the loudspeaker and its subwoofer/room processor as both a complete system and separable components. Benchmark the S1 passive monitor in parallel, but do not pretend a nearfield two-way and a 79 kg high-output floorstander occupy the same design envelope.

| Dimension | Provisional target / gate | Notes |
|---|---|---|
| Listening-window accuracy | **≤1.5 dB log-frequency RMS deviation** from a frozen target, 100 Hz–10 kHz | CTA-2034-B-style measurement where applicable; publish the exact procedure, normalization, smoothing and angular weighting. A project metric is not CTA compliance. |
| Crossover/directivity | No abrupt **>3 dB** discontinuity between adjacent declared angular averages around the crossover regions, 500 Hz–10 kHz | Define the angle set and bands first. Also inspect vertical lobing and early-reflection/power-response behavior; this single metric cannot summarize directivity. |
| Harmonic distortion | Aspiration: **≤1% THD at 96 dB SPL / 1 m** from 100 Hz–10 kHz, excluding explicitly documented measurement-floor limitations | Declare room/ground-plane/anechoic conditions and verify the noise floor. Publish distortion and compression versus frequency and output. |
| Compression | **≤1 dB compression** over the designated operating range during controlled level sweeps, with the test range preregistered | Test thermally stabilized and cold-start behavior; inspect individual driver bands, not only full-system output. |
| Maximum linear output | Aspirational **≥115 dB maximum linear SPL at 1 m per speaker** using AES75-2023 when applicable | This is a challenging test target, not a safety target for human listening. State signal, band, distance, environmental corrections and failure criterion. Do not compare directly with non-equivalent marketing figures. |
| Bass system | A modular system target of **20 Hz at −6 dB anechoic** with the designated subwoofer, plus disclosed maximum linear output by frequency | Large low-frequency output costs size, displacement, power and distortion. In-room “16 Hz” depends on room gain; publish anechoic and multi-position in-room data separately. |
| Listening-area consistency | Report response and correction limits across at least a **3-position** pilot listening region, then expand the region for qualification | Do not overfit one seat or use aggressive boost to conceal nulls. Publish raw and corrected curves. |
| Buildability/service | Replaceable driver, crossover/amplifier/DSP module where designed, documented seals/fasteners, spare-part identifiers and post-repair test | An active flagship can be modular without requiring a proprietary central controller or cloud login. |
| Manufacturing variation | Measure at least 3 units for alpha, 10+ across two builds/lots for release candidate where feasible | Disclose sample-size limits and per-unit serial/revision linkage. |

Use **ANSI/CTA-2034-B** for residential loudspeaker response, directivity and output measurement methodology where applicable: [CTA official scope](https://www.cta.tech/standards/ansicta-2034-b/). Use **AES75-2023** for maximum-linear-output claims only when following its actual procedure and signal: [AES75 resources](https://aes.org/standards/AES75/). Obtain the licensed standard texts and perform the actual required procedures before claiming compliance. These project thresholds do not replace either standard.

A 20 Hz anechoic target plus high output is a substantial size/cost challenge. If feasibility fails, preserve the modular system concept and report the trade-off rather than quietly reducing bandwidth or changing the claimed measurement conditions.

## 5. Competitive protocol: how to earn the word “better”

### Laboratory protocol

1. **Freeze the preregistration:** exact products, revisions, target curve, response bands, smoothing, angular sample grid, levels, measurement chain, calibration, fixture, processing code, acceptance gates and exclusions.
2. **Calibrate and run known-answer controls:** verify microphones, coupler/fixture, turntable/angle geometry, amplifier output, loads and analysis pipeline before the product samples.
3. **Capture raw data:** store un-smoothed captures, fixture/calibration IDs, ambient conditions, exact sample serials, run order, positioning and failures. Publish processed data only alongside raw files and processing revision.
4. **Repeat placements:** headphones require multiple reseats and, when possible, multiple fixture/headforms; speakers require horizontal and vertical angular response plus low-frequency/bass measurements appropriate to their enclosure.
5. **Separate training from confirmation:** prototype tuning uses a development set; final target gates are evaluated on held-out placements/units after the design is frozen.
6. **Independent replication:** a tester who did not tune the product rebuilds the analysis from the release manifest and checks a subset of measurements.
7. **Publish uncertainty and nulls:** include failed units, rejected runs with reasons, confidence intervals where meaningful, measurement limits and disagreements between labs.

### Listening protocol

Listening preference is a separate question from response accuracy.

- Randomize sample identity and presentation order; conceal brand/price where practical.
- Match headphone playback level to **within 0.2 dB** at a declared reference procedure and speaker comparisons to **within 0.1 dB** at the listening position, verifying over the chosen content/band.
- Use the same source, program excerpts, switch path and room/placement policy. Report any unavoidable difference.
- Use a preregistered paired-comparison/preference protocol and an exploratory pilot to estimate variance before powering the confirmatory study. Report participant count, hearing/experience inclusion policy, effect sizes and uncertainty, not only a winning percentage.
- Human sessions stay at safe, controlled listening levels. Do not use maximum-SPL exposure as a preference test.
- If participants disagree, publish preference segments and dispersion instead of averaging the disagreement away.

### Decision rule

A product can claim:
- **measurement parity** only for stated metrics and conditions;
- **measured improvement** when the difference exceeds test uncertainty and repeats independently;
- **listening preference** only when the randomized controlled result supports it for the tested cohort;
- **ergonomic advantage** only for declared mass/fit/comfort measures and cohort;
- **open/repairability advantage** only when source, parts, and documented repairs can be independently inspected.

Avoid a single composite “best sound” score. Keep acoustic error, distortion, compression, output, directivity, room variation, listening preference, comfort, repairability, cost, reliability and source completeness as separate dimensions, then use a documented Pareto frontier to show trade-offs. If a design loses at output or cost, do not hide that behind an acoustic average.

## 6. Design-space exploration for Symthaea EAC

The existing response-analysis, Pareto, uncertainty-aware Pareto and experiment-selection work is well suited to this programme, but the optimizer needs a real-world measurement contract.

Candidate objectives and constraints:
- minimize weighted target deviation across declared listening-window / headphone bands;
- minimize off-axis discontinuity and variation across the declared listening area;
- minimize THD and dynamic compression at multiple output levels;
- maximize AES75-compliant maximum linear output or a clearly defined headphone output envelope;
- minimize mass, enclosure volume, part cost, energy and repair time;
- maximize robustness to driver-lot variation, seal changes, tolerances and component substitutions;
- constrain impedance/current demand, temperature and safety limits;
- penalize results that depend on unrealistic EQ boost, unverified material properties or a single fragile build.

Do **not** treat simulated response as measurement evidence. Build an explicit chain: model hypothesis → chosen experiment → physical capture → uncertainty record → model update → held-out confirmation. Count deferred validation cost so the policy cannot “win” by recommending cheap preliminary measurements that merely postpone expensive truth.

The next highest-value EAC experiment is a fixture/repeatability campaign and baseline measurements before algorithmically optimizing speaker or headphone targets. An optimizer trained on invalid or non-repeatable data only makes the wrong answer more precise.

## 7. Milestones and stop conditions

### F0 — reference definition
- Verify exact product references, target segment and sample/loan plan.
- Define test profiles, safety envelope, calibration, raw-data schema and immutable sample manifest.
- Publish BOM cost and test-cost ledger structure; do not invent prices or availability.

### F1 — feasibility alpha
- Obtain actual candidate drivers and measure at least two alternatives; no unmeasured final driver selection.
- Establish coupler/fixture repeatability and measurement floor.
- Run basic speaker driver/cabinet/crossover prototypes and angular tests.
- Freeze initial user-facing targets only after feasibility data, before confirmatory runs.

### F2 — robust prototype
- Repeat results across units, reseats, angles, and realistic component tolerance.
- Separate development and holdout data; publish source for analysis/optimization.
- Demonstrate repair and remeasurement after part replacement.

### F3 — independent validation
- Independent lab/builder reproduces key data.
- Controlled listening/preference study uses safe levels and blinded, level-matched presentation.
- Compare against preselected retail references; publish all limitations and failures.

### F4 — release candidate
- Native source package, licences/provenance, BOM/alternates, calibration/measurement manifest, repair guide, safety/regulatory plan, spare-part plan and revision policy complete.
- Multi-unit variation characterized to the extent feasible and limitations disclosed.
- Any standards claim is backed by the actual applicable procedure, not a similar-looking plot.

**Stop / revise the design** if the result depends on one lucky reseat, a hidden DSP boost, a cherry-picked unit, an invalid analysis pipeline, unsafe test levels, a fragile part with no replacement path or a measurement uncertainty larger than the claimed improvement.

## 8. References

- [ABYSS AB1266 TC — manufacturer specifications](https://abyss-headphones.com/en-gb/products/abyss-ab-1266-phi-headphone).
- [Arendal Sound 1528 Tower 8 — official technical specifications](https://arendalsound.com/product/1528-tower-8/tech/).
- [Arendal Sound 1528 product handbook (PDF)](https://arendalsound.com/wp-content/uploads/2025/02/2025_02_Product_Handbook_Arendal_1528_Speakers_8_v2.5_WEB.pdf).
- [Hi-Fi News AB-1266 Phi TC lab report](https://www.hifinews.com/content/abyss-ab-1266-phi-tc-headphones-lab-report).
- [Kuulokenurkka AB1266 Phi TC measurement review](https://kuulokenurkka.com/en/review-abyss-ab1266-phi-tc/).
- [Spinorama open loudspeaker measurement catalogue](https://www.spinorama.org/).
- [IEC 60268-7:2025 — headphones and earphones](https://webstore.iec.ch/en/publication/86573).
- [ANSI/CTA-2034-B — residential loudspeaker measurement scope](https://www.cta.tech/standards/ansicta-2034-b/).
- [AES75-2023 — maximum linear sound level using Music-Noise](https://aes.org/standards/AES75/).
- [IEC 62368-1 — audio/video and ICT equipment safety](https://webstore.iec.ch/en/publication/69308).

## Claim boundary

At document creation time, no Luminous headphone or flagship loudspeaker has been built, measured or independently compared to these references. Every target above is an aspiration or proposed procedure. This document deliberately establishes a falsifiable challenge; it does not establish success.
