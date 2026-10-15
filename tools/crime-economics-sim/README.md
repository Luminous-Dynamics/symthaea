# Symthaea Crime-Economics Synthetic Evaluator

This is a small isolated Rust research tool. It is not calibrated to observed crime data and is not an individual-risk model. It emits aggregate simulated counts only. Event labels describe model categories; they are not observed or adjudicated crimes, and no person-level risk is estimated.

## Reproduce locally

From the repository root:

    cargo fmt --manifest-path tools/crime-economics-sim/Cargo.toml -- --check
    cargo test --offline --manifest-path tools/crime-economics-sim/Cargo.toml --locked
    cargo run --offline --quiet --manifest-path tools/crime-economics-sim/Cargo.toml --locked --bin simulate > crime-economics.tsv
    cargo run --offline --quiet --manifest-path tools/crime-economics-sim/Cargo.toml --locked --bin verify -- crime-economics.tsv
    cargo run --offline --quiet --manifest-path tools/crime-economics-sim/Cargo.toml --locked --bin assess_viability > lawful-alternative-viability.tsv
    cargo run --offline --quiet --manifest-path tools/crime-economics-sim/Cargo.toml --locked --bin verify_viability -- lawful-alternative-viability.tsv

The output is a tab-separated artifact with 8 scenarios × 5 policy arms × 32 deterministic Monte Carlo replications (1,280 rows). The verifier separately parses the emitted artifact and checks frozen schema identity, row completeness, event-accounting identities, reporting/displacement negative controls, non-economic-harm preservation, robot-only fields, distributional-access, concentrated-automation, high-robot-compromise, and low-robot-effectiveness adversarial cases.

## Parameter register

`tools/crime-economics-sim/manifest.json` has a 15-entry parameter register. Every entry is marked `synthetic_assumption` and `not_empirical`, with units and limitations. It records the cohort weights, baseline access levels, event-probability assumptions, work/essentials effects, robot protection/compromise/displacement/reporting assumptions, fictional cost units, and unvalidated governance proxies.

The register is a human-reviewable statement of intended inputs, not a parser-generated export of Rust constants. Changes to model constants and their corresponding registered assumptions must be reviewed together. The CI workflow validates the registry's shape and required claim labels; it does not prove numerical correspondence, empirical validity, or external-source provenance.

## Lawful-alternative viability contract

A separate aggregate-only fixture contract now emits `lawful-alternative-viability.tsv`: **10 scenarios × 5 anonymous cohorts = 50 rows**. Run it with `assess_viability`; validate the artifact with the independent `verify_viability` binary.

It does not calculate crime rates or an “irrationality score.” It asks whether minimum conditions for a viable lawful alternative are met in each aggregate cohort. It keeps seven dimensions distinct: essential-service access, lawful-opportunity coverage, fair-pay adequacy, work safety, work stability, social-protection coverage, and access time. It also checks uncertainty, source freshness/state, and conservative cross-cohort coverage gaps.

The manifest contains a measurement dictionary for each dimension: its numerator, denominator, required data and a non-substitution rule. Most importantly, lawful-opportunity coverage is defined against the entire pre-registered target population and requires at least one attainable option that already meets declared pay, safety, stability and access criteria; it is not a vacancy count or placement rate. Fair pay, safety, stability and social protection remain separate gates. The low/high fixture bounds are **not statistical confidence intervals**; a real pilot would need to pre-register each source, reference period, denominator, missingness treatment and uncertainty method.

The result is non-compensatory:
- **`fixture_pass`** means every synthetic gate passes for this fixture only.
- **`fixture_fail`** means at least one required floor/cap/gap clearly fails.
- **`fixture_indeterminate`** means evidence is missing, stale, conflicted, too uncertain, or straddles a gate.

A high mean cannot hide a cohort below the essentials floor; opportunity coverage cannot compensate for unsafe or inadequately paid work; a service cannot pass when lawful alternatives remain too costly to reach; and wide uncertainty does not silently become success. These labels qualify neither a real service nor a real population.

All thresholds—including 90% essential access, 80% opportunity coverage and pay adequacy, 90% safe work, 75% stability and protection, 60 minutes of access time, 15 percentage-point maximum coverage gap, 20 percentage-point maximum interval width, 20-minute access-time interval width, and a 180-day source-age limit—are **synthetic test thresholds**, not legal rules, scientific constants, or universally valid targets. The independent verifier duplicates the gate logic rather than importing the assessment function.

### Evidence context, kept separate from the fixture values

- Statistics South Africa reported a 33.6% official unemployment rate in Q2 2026. The QLFS provides a current national labour-market context and downloadable metadata/data; it does **not** estimate how unemployment causes crime, and this national statistic is not an input coefficient in the synthetic model. See [Stats SA QLFS Q2 2026](https://www.statssa.gov.za/?PPN=P0211&SCH=74512&page_id=1856).
- The latest [Stats SA GPSJS 2025/26 Victims of Crime Release](https://www.statssa.gov.za/?PPN=P0341&SCH=74591&page_id=1856), published 1 September 2026, reports an estimated 1.549 million housebreaking incidents (versus 1.493 million in 2024/25), while households experiencing housebreaking fell from 1.123 million to 1.078 million and the share reporting some/all incidents rose from 43.4% to 48.2%. These are different indicators and denominators, not a single “crime went up/down” result. The survey is national descriptive context, not a Johannesburg baseline or causal coefficient; the model does not use it to calibrate its synthetic event parameters. Stats SA's technical notes report response rates of 85.57% nationally, 76.97% for Gauteng, 73.72% for the City of Johannesburg, and 67.54% for the City of Tshwane; four geographic areas were below the cited 80% minimum-acceptable response rate. These figures are quality diagnostics, not a significance test or proof that every lower-response-area estimate is invalid. The cited 80% minimum-acceptable level comes from Stats SA 019:2017 V2.00; it is a response-rate quality benchmark, not a significance level such as alpha = 0.05. An empirical comparison must use estimate-specific precision information (such as standard errors, confidence intervals or coefficient of variation) where available, and preserve survey design/weights, indicator denominators, time periods and geographic fit. Table 41 adds a useful caution: household housebreaking experience over the preceding 12 months was 43.1% (CV 3%) in 2024/25 and 46.0% (CV 3%) in 2025/26. These are prevalence estimates, not the separate 43.4% to 48.2% police-reporting proportions, so their CVs cannot be reused as precision for the reporting-rate difference. Neither response-rate benchmarks nor CV values alone establish statistical significance. A local response rate is not itself an estimate-precision statistic. [Full release and technical notes, Table 39](https://www.statssa.gov.za/publications/P0341/P03412026.pdf).
- The ILO's decent-work indicator framework covers dimensions such as adequate earnings, safety, stability/security, and social protection. It helps choose what to measure, but it does not supply the synthetic thresholds used here. See [ILO Decent Work Indicators](https://www.ilo.org/publications/decent-work-indicators-concepts-and-definitions) and the [ILO measurement framework](https://www.ilo.org/resource/ilo-framework-measurement-decent-work).

Neither source is used as a crime-effect estimate. The next empirical step would require a service-area baseline, data-governance review, specific local outcomes, measurement uncertainty and a pre-registered evaluation. Avoid person-level profiling; if survey microdata is ever considered, it requires separate permissions and a privacy review.

## Scenarios

- reference: ordinary illustrative assumptions.
- high_displacement: a larger share of prevented focal-area property incidents is assumed to move to neighboring areas.
- reporting_suppression: only reporting probability changes; the verifier requires underlying harm to remain identical and requires at least one strict reduction in reported incidents across matched seed/policy cells, preventing an inert suppression scenario from passing.
- concentrated_automation: a deliberately adverse livelihood shock is applied to robot-enabled arms; combined support partly mitigates the shock.
- high_robot_compromise: compromised robot weeks are more common; the verifier checks that exposure increases without altering non-robot arms.
- unequal_essential_access: the same essentials program is delivered unevenly; the verifier must detect the worsening access gap.
- low_robot_effectiveness: the robot protection-effect assumption is scaled to one quarter of the reference scenario; the verifier checks that weaker assumed protection cannot improve the modeled protection outcomes.
- zero_quality_work_effect: the assumed crime-channel effects of quality-work support are set to zero. The verifier requires the quality-work-only arm to match baseline outcomes aside from the fictional cost ledger, and requires non-work arms to remain invariant.

Policy arms are baseline, essentials, quality_work, robot_protection, and combined.

## Important units and assumptions

- The 1,000 synthetic household-equivalents, 52-week window, cohort weights, event rates, program effects, reporting probabilities, robot protection rate, false-alarm rate, compromise rates, and scenario shocks are synthetic assumptions, not estimated parameters.
- annualized_cost_units are fictional accounting units, not dollars or a cost-effectiveness finding.
- privacy_burden_points and benefit_concentration_bps are explicit proxies, not validated privacy or inequality measures.
- Need-related and opportunity-related property events are modeled separately from non-economic harm. The model does not assume economic interventions remove non-economic harm.
- displaced_property_events are included in total simulated harm events but are separate from incidents remaining at the protected site.
- reported_harm_events depend on the scenario's reporting probability. A decline in reports does not necessarily indicate a decline in underlying harm.
- false_alarms and robot_safety_incidents are separate from crime-event and report counts.
- Seeds provide deterministic reproducibility and common random numbers across comparable scenario arms. They do not create empirical validity.

## What this prototype does not model

The current version does **not** calculate offender expected utility, an economic-payoff equilibrium, or a threshold at which crime becomes economically irrational. It directly applies synthetic changes to aggregate event probabilities to exercise evaluation controls.

The quality-work assumption is explicitly stress-tested at zero. A 2024 systematic review found the offending-outcome evidence for summer employment programmes limited and heterogeneous, with no clear pooled impact on several criminal-justice outcomes; those high-income-country findings do not estimate the effect of year-round quality work or cooperative production in South Africa. See [Muir et al. (2024), Campbell Systematic Reviews](https://doi.org/10.1002/cl2.1406). It therefore cannot answer the original question quantitatively and should not be used to rank policies by simulated means.

A later model increment should first define an **aggregate lawful-alternative viability** measure: reliable access to essentials, quality and accessibility of lawful work/production, time and transaction costs for lawful alternatives, and the uncertainty of those options. It should estimate only population- or service-level relationships supported by transparent evidence, with ranges and sensitivity analysis—not infer criminal propensity about individuals or optimize methods for committing or evading crime.

## Claim ceiling

Passing the Rust unit tests establishes only that the implementation satisfies selected coded invariants. Passing the independent artifact verifier establishes only that the emitted TSV satisfies the frozen structural and cross-scenario checks. Neither establishes:
- that the modeled relationships match real behavior;
- that any intervention causally reduces crime;
- that most crime is economically motivated or can be made economically irrational;
- that costs, rights impacts, robot safety, or cybersecurity have been appropriately valued;
- that the model generalizes to Johannesburg, South Africa, or any other place;
- that Symthaea reasoning is accurate or ready for operational use.

Before any empirical pilot, specify a local problem, obtain privacy and community review, pre-register outcomes and a comparison strategy, collect valid baselines, measure victimization as well as reports, measure displacement and unequal effects, and qualify all robots under a separate safety and cyber-security process. The tool has no network access and no service side effects.
