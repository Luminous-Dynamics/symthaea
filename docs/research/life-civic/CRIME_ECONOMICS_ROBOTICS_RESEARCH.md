# Crime-Economics and Robotics: Research-to-Evaluation Plan

**Status:** Research proposal and evaluation design; not an empirical result  
**Version:** 0.1.3  
**Relationship:** Life/civic domain mastery and the Symthaea-to-Mycelix boundary  
**Authority:** Effects disabled; all scenario records are synthetic reference fixtures

## Objective

Test whether combinations of reliable access to necessities, high-quality lawful opportunities, safer environments, accountable property protection, and well-secured service robots can reduce specific forms of preventable harm at acceptable social cost.

The target is not a universal theory that every offense is economically rational. Offenses differ in motivation, setting, victim relationship, opportunity, and situational constraints. An economic model is a bounded lens for selected mechanisms—not a person classifier, a guilt estimator, or a substitute for legal findings.

Operational success should be described as a measured reduction in a clearly defined harm outcome against a credible counterfactual, with the uncertainty and side effects reported. It must not be inferred from a plausible simulation, a lower count of police reports alone, or a robot detecting more events.

## Working hypotheses (not established conclusions)

- **H1 — Need reliability:** Stable, accessible access to essentials may reduce some need-driven property offenses. Test whether the relevant need actually constrained the event; do not infer motive from poverty or neighborhood.
- **H2 — Quality of lawful opportunity:** Accessible, dignified work and productive participation may change incentives more than nominal job availability alone. Evaluate job quality, stability, autonomy, safety, and future prospects, not only employment counts.
- **H3 — Opportunity reduction:** Specific environmental and asset-protection changes may reduce selected property offenses. Measure both intended-area outcomes and displacement/diffusion; do not generalize camera or patrol findings to autonomous robots without direct testing.
- **H4 — Automation distribution:** Robotics may lower costs and expand productive capacity, but badly governed automation may also displace income, concentrate control, or create new barriers. Measure who receives the gains and who bears the transition costs.
- **H5 — Accountable coordination:** Mycelix can preserve evidence lineage, participant consent, appeals, and service commitments; Symthaea can compare intervention options and uncertainty. Neither registry inclusion nor model confidence establishes truth, legal authority, or guilt.
- **H6 — Safe robotics:** Service robots may support maintenance, repair, logistics, accessibility, and bounded property protection. Their control systems, remote access, sensors, actuators, and recovery procedures become part of the threat model; physical intervention requires a separately authorized safety case.

These hypotheses should be revised when data contradict them. Do not optimize for the lowest observed offense count if reporting falls, victimization rises, or harms shift to less visible groups.

## Economic model boundary

At the community or intervention level, compare the whole-life costs and outcomes of candidate policies:

- reliable essential-service access;
- paid, high-quality work, training, and cooperative production;
- environmental design and proportionate property protection;
- repair, emergency response, and victim support;
- robotic capital, maintenance, energy, insurance, security, and decommissioning;
- displacement, unequal access, privacy loss, and error-correction costs.

A useful intervention accounting identity is:

`Net social value = avoided harm + added lawful opportunity + service gains
                    - intervention cost - transferred harms - rights/privacy costs`

This is an accounting frame, not a validated numerical law. Monetized benefits must not erase non-monetizable rights, dignity, injury, or distributional harms. Report disaggregated outcomes alongside any aggregate cost-effectiveness estimate.

Do not assign an individual a criminal propensity, risk score, or eligibility decision from economic status, location, associations, reputation, inferred need, or a Symthaea prediction. Use aggregate and opt-in intervention data with privacy safeguards. A specific investigation must rely on incident-specific evidence, applicable law, due process, and human accountability.

## Evidence we have—and do not have

- NIJ-sponsored research reports that the employment–crime relationship depends on context. A 2006 study found a modest relationship in urban young adults, not rural young adults; another study abstract reported that employment quality appeared more influential than income or job stability alone. These findings motivate stratified tests; they do not establish a universal causal effect or predict any person's behavior.
- A 2024 Campbell systematic review of summer programmes for disadvantaged or “at risk” young people (ages 10–25; eligible studies were in high-income countries) found a near-zero, statistically non-significant pooled estimate for the probability of a post-programme arrest/arraignment/conviction/incarceration outcome (log odds ratio −0.05, 95% CI −0.15 to +0.05). A separate pooled estimate for the number of arrests/convictions around programme months was also near zero (standardised mean difference −0.01, 95% CI −0.03 to 0.00). The review reports mixed findings, small/limited evidence for several outcomes and important variation across studies. These are not transferable coefficients for stable year-round quality work, cooperative production, adults, or South Africa. They support testing the model with zero assumed work effect; they do not prove that work programmes have no effect in every context. [Muir, Orlando & Newton (2024), *Campbell Systematic Reviews*](https://doi.org/10.1002/cl2.1406).
- A systematic review of CCTV research found modest and setting-dependent effects, with stronger results in some property-crime contexts and where surveillance was actively monitored or paired with other interventions. That evidence is not evidence that autonomous security robots work, nor is it justification for pervasive surveillance.
- The World Bank has described randomized evaluations of public-works and youth-employment interventions aimed at economic welfare and violence-related outcomes. The reviewed program designs and settings vary; do not represent the whole portfolio as one confirmed effect.
- NIST's September 2026 SP 800-82 Revision 4 is an **initial public draft**, not a final standard. Its guidance is relevant to OT security principles, but this project must pin status, date, version, scope, and later supersession before claiming requirements.

## Minimum viable research design

### Stage 0 — synthetic mechanism model

Create a deterministic, effects-disabled simulation with explicit assumptions and independent expected-result checks. Vary need reliability, quality of lawful opportunity, local crime opportunity, robot cost, false alarms, reporting rates, displacement, cyber compromise, and benefit distribution. The simulator demonstrates only that the model is implemented as specified.

### Stage 1 — evidence and baseline

For one defined offense category and bounded place/service, document source provenance, jurisdiction, period, operational definitions, reporting channels, missingness, demographic/distributional risks, and baseline costs. Separate reported incidents, victimization surveys, verified losses, response time, and outcomes.

### Stage 2 — comparison and pilot

Pre-register a comparison strategy before collecting outcomes: randomized assignment where ethical and feasible, otherwise a credible quasi-experimental design with explicit confounders and sensitivity analysis. Do not deny anyone essential services or legal protections for an experiment. Use staged deployment and independent review.

### Stage 3 — independent evaluation

Report point estimates and uncertainty; absolute counts and denominators; comparison-group trends; adverse events; false-positive burden; reporting changes; geographic or temporal displacement; accessibility; who receives economic gains; privacy/security incidents; and full lifecycle cost. Publish null and adverse results.

### Stage 4 — conditional scaling

Scale only the intervention and context actually supported by evidence. Any new jurisdiction, offense type, robot model, sensor, autonomy level, data source, or operational authority is a new qualification question.

## Minimum outcome set

- harm: victimization, severity, recurrence, and independently confirmed losses where available;
- economic pathway: essential-service reliability, quality and accessibility of lawful opportunities, income volatility where lawfully and consensually measured, and distribution of benefits;
- service quality: response/repair time, successful resolution, downtime, accessibility, and service cost per beneficiary;
- protection quality: prevention-specific outcomes, false alarms, false accusations, missed events, reversals, and appeal outcomes;
- system effects: displaced incidents, changed reporting, unequal impacts, coercion, mission creep, and privacy intrusion;
- robotics assurance: incident logs, maintenance failures, remote-access security, override operation, safe stopping, recovery, and service continuity;
- evidence quality: denominator completeness, source freshness, provenance independence, uncertainty, model/version identity, and reproducibility.

Do not declare success on arrests, stops, alerts, robot patrol-hours, or dashboard activity alone.

## Red-team cases

The accompanying machine-readable scenario suite tests common overclaiming failures. Every fixture requires human review and keeps external effects disabled. The CI workflow validates document shape and safety invariants only; it does not run a causal model, qualify reasoning accuracy, prove legal compliance, or establish a real-world crime reduction.

Required adversarial controls include:
1. Need-driven theft is assumed from income or geography without incident evidence.
2. Any job is counted as a protective intervention despite unsafe or exploitative conditions.
3. Automation increases output but removes local livelihoods or concentrates ownership.
4. Security robots appear to reduce losses while incidents move next door or become less likely to be reported.
5. A false-positive alert is treated as proof or used to deny housing, work, benefits, or movement.
6. A signed event or reputation receipt is mistaken for truth or guilt.
7. A compromised robot or remote-control path can endanger people or interrupt essential infrastructure.
8. Interpersonal, coercive, ideological, or mental-health-related harm is modeled as purely economic.
9. A small, selected sample or short observation window is presented as universal proof.
10. A simulation, source citation, queued workflow, or skipped check is called an empirical pass.

## Standards and source treatment

Store identifiers and public metadata where licensing permits. Never republish protected normative text without permission. Distinguish standard publication from legal adoption, operational authorization, and demonstrated conformance. Because this is research rather than a legal profile, jurisdiction-specific human review remains necessary before any deployment.


## Stage 0 implementation: synthetic Rust evaluator

The branch now contains an isolated, dependency-free Rust crate at `tools/crime-economics-sim/`. It is a model-mechanics prototype, not an empirically calibrated criminology model.

The initial design, including the robot-effectiveness sensitivity extension, is:
- 1,000 synthetic household-equivalents represented only as five aggregate access cohorts;
- 52 weekly steps per replication;
- 32 deterministic random seeds per cell;
- five policy arms: baseline, essentials, quality work, robot protection, and combined support;
- eight scenarios: reference, high displacement, reporting suppression, concentrated automation, high robot compromise, unequal access, low robot effectiveness, and zero assumed quality-work effect;
- 1,280 output rows, with no person-level records or scores.

Event generation separates need-related property events, opportunity-related property events, and non-economic harm. The output uses `simulated_harm_events_total` and `reported_harm_events` labels to avoid implying that generated incidents are observed or adjudicated crimes. Robot protection affects only the modeled property-event channel, can be compromised, can generate false alarms and safety incidents, and may displace events into a neighboring ring. Report probability is a separate scenario parameter. Access gains, distribution concentration, privacy burden and annualized costs are explicit illustrative parameters/proxies, not measured outcomes or monetary valuations.

The standalone verifier reads the simulator's emitted TSV as a separate executable and checks row completeness, event accounting, policy-only fields, and adversarial cross-scenario properties. Required negative controls include:
- lower reporting probability changes reports but leaves underlying generated harm unchanged;
- higher displacement leaves focal-site outcomes unchanged and cannot reduce modeled displacement;
- non-economic harm remains visible across all policy arms;
- uneven essential-service access worsens the modeled access gap;
- weakening the robot-effectiveness assumption cannot improve robot-enabled policy outcomes and cannot perturb non-robot arms;
- setting the assumed quality-work effect to zero makes the quality-work-only arm match baseline modeled outcomes apart from fictional cost, while non-work arms remain invariant.

These controls validate programmed invariants, not the realism of the assumptions. The simulator uses an explicit seeded pseudorandom generator, common random numbers for paired comparisons, and no third-party dependencies. Its cost units and privacy/concentration proxies are intentionally not combined into a single social-value score. The research must not choose a policy based on the synthetic means.

The path-scoped workflow runs Rust unit tests, generates the 1,280-row artifact twice and byte-compares the outputs, invokes the independent verifier, records the exact source commit and a SHA-256 digest, and uploads the evidence artifact for seven days. Passing this workflow would establish that the prototype and artifact satisfy their coded contracts at that exact head. It would not establish model calibration, a causal effect, a robot safety case, or real-world crime reduction.

## Missing economic-equilibrium layer

Although this crate is named a crime-economics evaluator, version 0.3.0 remains an aggregate intervention-sensitivity prototype. It **does not calculate an offender payoff, a lawful-versus-unlawful utility crossover, or a quantitative threshold for economic irrationality**. Its effects on event rates are explicitly hard-coded test assumptions. The model can check whether software preserves specified relationships; it cannot currently test the main economic thesis on its own terms.

A next phase should use a separate aggregate contract for lawful-alternative viability—reliable access to essentials, attainable high-quality work or cooperative production, lawful access time/transaction costs, and uncertainty in those alternatives. It should define observable, behavior-specific outcomes before parameter selection, use literature to establish defensible ranges rather than transplant a single effect size, and preserve subgroup uncertainty without assigning individual propensities. Keep hypothetical illicit payoff variables abstract and non-operational; do not model detailed methods, target selection, or evasion strategies. The result should remain a population/service-level policy model, not a person classifier.

## Lawful-alternative viability: first executable contract

The branch now includes a separate Rust module (`src/viability.rs`), output generator (`assess_viability`), and independent artifact verifier (`verify_viability`). The current reference matrix contains ten synthetic scenarios across five anonymous aggregate cohorts (50 rows). It does not feed back into the crime-event simulator and does not claim to estimate crime.

The contract separates essential-service access, lawful-opportunity coverage, fair-pay adequacy, safe-work coverage, work stability, social protection, and access time. Confidence/range uncertainty, stale/conflicted/unavailable evidence, and conservative cohort-level coverage gaps remain visible. No single weighted score can compensate for a cohort failing a minimum condition. The fixtures explicitly challenge high averages hiding a low essentials floor, job quantity hiding poor quality, excessive travel/access friction, uncertainty crossing thresholds, stale/conflicting/missing evidence, and large cohort gaps.

The gate status names are deliberately prefixed `fixture_` because all numbers and thresholds are synthetic. A `fixture_pass` means only that the chosen test profile satisfies a contract; it does not mean lawful alternatives are genuinely adequate in a real community. The gate thresholds are not taken from Stats SA or ILO as normative values.

Before those source notes, the manifest now includes a seven-measure dictionary with each numerator, denominator, required data and non-substitution rule. Lawful-opportunity coverage is not a vacancy count: it measures the share of a pre-registered target population with at least one attainable option meeting declared pay, safety, stability and access criteria. The synthetic low/high bounds are illustrative ranges, not statistical confidence intervals. Any real pilot must pre-register its source, reference period, denominator, missingness treatment and uncertainty method.

For the future empirical data map, two sources have different roles:
- Statistics South Africa's QLFS Q2 2026 reports a national official unemployment rate of 33.6% and provides downloadable series and metadata. It is a descriptive labour-market baseline—not a crime coefficient, intervention effect, or local service-area estimate. [Official Q2 2026 results](https://www.statssa.gov.za/?PPN=P0211&SCH=74512&page_id=1856).
- The latest Statistics South Africa GPSJS 2025/26 Victims of Crime Release (published 1 September 2026) is useful for separating underlying survey indicators from police-reporting processes. It estimates 1.549 million housebreaking incidents versus 1.493 million in 2024/25, while affected households fell from 1.123 million to 1.078 million and the share reporting some/all incidents rose from 43.4% to 48.2%. For personal theft, the share reporting some/all incidents rose from 31.0% to 37.3%; 41.7% of people experiencing street robbery reported some/all incidents. These are descriptive national estimates with distinct denominators, not a causal intervention effect or a Johannesburg baseline. Stats SA's technical notes report response rates of 85.57% nationally, 76.97% for Gauteng, 73.72% for the City of Johannesburg, and 67.54% for the City of Tshwane; four geographic areas were below the cited 80% minimum-acceptable response rate. Treat these rates as quality diagnostics, not as significance tests or proof that every lower-response-area estimate is invalid. The 80% minimum-acceptable response-rate threshold is from Stats SA 019:2017 V2.00; it is not a statistical significance level. Table 41 reports household housebreaking experience in the past 12 months at 43.1% (CV 3%) for 2024/25 and 46.0% (CV 3%) for 2025/26. These are prevalence estimates, distinct from the 43.4% to 48.2% share reporting incidents to police. Their CVs do not provide the precision of the reporting-share difference. Neither response-rate thresholds nor CVs alone establish statistical significance. A city response rate is not itself an estimate-precision statistic. Before making statistical-significance claims, preserve indicator definitions and denominators, survey design and weights, reference periods, geographic fit, and estimate-specific standard errors/confidence intervals/coefficient of variation or other precision information where available. [Official key findings](https://www.statssa.gov.za/?PPN=P0341&SCH=74591&page_id=1856) · [Full release and technical notes, Table 39](https://www.statssa.gov.za/publications/P0341/P03412026.pdf).
- The ILO's decent-work indicator framework supports choosing a multidimensional measurement set including earnings, job safety, work stability/security, and social protection. It does not justify the synthetic numeric thresholds or supply an automatic causal link to crime. [ILO indicator concepts](https://www.ilo.org/publications/decent-work-indicators-concepts-and-definitions).

The CI workflow checks deterministic generation, row/schema completeness and an independently recomputed set of non-compensatory gates. Even a passing synthetic verifier cannot establish the gate thresholds are normatively correct, that an area has sufficient opportunity, or that crime has fallen. The next empirical stage must pre-register a specific intervention and counterfactual, use aggregate and privacy-governed data, and report adverse as well as favorable effects.

## Immediate model limitations

The implementation intentionally omits a calibrated empirical event process, household/person dynamics, network interactions, offense-specific victimization weights, confidence intervals across real locations, real prices, legal authority, individual identity, and any actual robot control. The latest national GPSJS reporting and victimization statistics remain context-only and are not used as synthetic event parameters. The 15-entry parameter register in `tools/crime-economics-sim/manifest.json` now explicitly labels those inputs as synthetic assumptions, with units and limitations; it is a reviewed declaration, not a parser-generated export or external-source provenance record. The next scientific increment is to propose source-grounded alternative ranges where credible evidence exists, preserve uncertainty and context limits, and then design local data governance and a pre-registered evaluation. It should not be an automated individual risk model.

## References

1. National Institute of Justice, **Labor Force Participation, Labor Markets, and Crime: Final Report** (2006): https://nij.ojp.gov/library/publications/labor-force-participation-labor-markets-and-crime-final-report
2. National Institute of Justice, **The Meaning of Work: Conceptualizing the Deterrent Effect of Employment on Crime among Young Adults** (2006): https://nij.ojp.gov/index.php/library/publications/meaning-work-conceptualizing-deterrent-effect-employment-crime-among-young
3. Welsh & Farrington, **CCTV surveillance for crime prevention: A 40-year systematic review with meta-analysis**: https://www.crimrxiv.com/pub/przahuaw/release/1
4. World Bank, **What are we learning about the impacts of public works programs on employment and violence?**: https://blogs.worldbank.org/en/impactevaluations/what-are-we-learning-about-impacts-public-works-programs-employment-and-violence-early-findings
5. NIST, **SP 800-82 Rev. 4 Initial Public Draft** (published 2026-09-21; comments due 2026-11-30): https://csrc.nist.gov/pubs/sp/800/82/r4/ipd
6. UNODC, **Crime Prevention & Criminal Justice Module 2: Summary**: https://www.unodc.org/e4j/en/crime-prevention-criminal-justice/module-2/key-issues/summary.html
7. Weisburd et al., **Can increasing preventive patrol in large geographic areas reduce crime? A systematic review and meta-analysis** (2024): https://doi.org/10.1111/1745-9133.12665. The review found small average reductions for broad-area preventive patrol overall and larger estimated reductions for property and violent crime subsets; only four studies measured displacement, with uncertain/inconsistent results. This is evidence about preventive patrol, not autonomous robots, and should not be transposed as a robot-effect estimate.
8. Muir, Orlando & Newton, **Impact of summer programmes on the outcomes of disadvantaged or ‘at risk’ young people: A systematic review** (2024): https://doi.org/10.1002/cl2.1406. The review included six summer employment programmes within high-income-country studies; evidence for offending outcomes was limited and heterogeneous, with no clear pooled effect for several criminal-justice outcomes. It does not estimate effects for all quality work, cooperative production, or South African contexts.
8. ISO, **ISO 10218-1:2025 — Robotics — Safety requirements — Part 1: Industrial robots**: https://www.iso.org/standard/73933.html
9. ISO, **ISO 10218-2:2025 — Robotics — Safety requirements — Part 2: Industrial robot applications and robot cells**: https://www.iso.org/standard/73934.html. The public catalogue describes industrial robot applications/cells and lists public-access service robots and law-enforcement uses among exclusions. This is why industrial robot standards must not be treated as a universal safety qualification for public-facing service or security robots.


**Evidence claim ceiling:** these sources support a research agenda and bounded hypotheses, not a claim that robotics makes most crime economically irrational. No pilot result, causal effect size, or Symthaea reasoning score is asserted.
