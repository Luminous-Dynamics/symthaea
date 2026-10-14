# Symthaea Crime-Economics Synthetic Evaluator

This is a small isolated Rust research tool. It is not calibrated to observed crime data and is not an individual-risk model. It emits aggregate simulated counts only. Event labels describe model categories; they are not observed or adjudicated crimes, and no person-level risk is estimated.

## Reproduce locally

From the repository root:

    cargo fmt --manifest-path tools/crime-economics-sim/Cargo.toml -- --check
    cargo test --offline --manifest-path tools/crime-economics-sim/Cargo.toml --locked
    cargo run --offline --quiet --manifest-path tools/crime-economics-sim/Cargo.toml --locked --bin simulate > crime-economics.tsv
    cargo run --offline --quiet --manifest-path tools/crime-economics-sim/Cargo.toml --locked --bin verify -- crime-economics.tsv

The output is a tab-separated artifact with 8 scenarios × 5 policy arms × 32 deterministic Monte Carlo replications (1,280 rows). The verifier separately parses the emitted artifact and checks frozen schema identity, row completeness, event-accounting identities, reporting/displacement negative controls, non-economic-harm preservation, robot-only fields, distributional-access, concentrated-automation, high-robot-compromise, and low-robot-effectiveness adversarial cases.

## Parameter register

`tools/crime-economics-sim/manifest.json` has a 15-entry parameter register. Every entry is marked `synthetic_assumption` and `not_empirical`, with units and limitations. It records the cohort weights, baseline access levels, event-probability assumptions, work/essentials effects, robot protection/compromise/displacement/reporting assumptions, fictional cost units, and unvalidated governance proxies.

The register is a human-reviewable statement of intended inputs, not a parser-generated export of Rust constants. Changes to model constants and their corresponding registered assumptions must be reviewed together. The CI workflow validates the registry's shape and required claim labels; it does not prove numerical correspondence, empirical validity, or external-source provenance.

## Scenarios

- reference: ordinary illustrative assumptions.
- high_displacement: a larger share of prevented focal-area property incidents is assumed to move to neighboring areas.
- reporting_suppression: only reporting probability changes; the verifier requires underlying harm to remain identical.
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
