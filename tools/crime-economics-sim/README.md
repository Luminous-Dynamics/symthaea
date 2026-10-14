# Symthaea Crime-Economics Synthetic Evaluator

This is a small isolated Rust research tool. It is not calibrated to observed crime data and is not an individual-risk model. It emits aggregate simulated counts only.

## Reproduce locally

From the repository root:

    cargo fmt --manifest-path tools/crime-economics-sim/Cargo.toml
    cargo test --manifest-path tools/crime-economics-sim/Cargo.toml
    cargo run --quiet --manifest-path tools/crime-economics-sim/Cargo.toml --bin simulate > crime-economics.tsv
    cargo run --quiet --manifest-path tools/crime-economics-sim/Cargo.toml --bin verify -- crime-economics.tsv

The output is a tab-separated artifact with 6 scenarios × 5 policy arms × 32 deterministic Monte Carlo replications (960 rows). The verifier separately parses the emitted artifact and checks frozen schema identity, row completeness, event-accounting identities, reporting/displacement negative controls, non-economic-harm preservation, robot-only fields, distributional-access, concentrated-automation, and high-robot-compromise adversarial cases.

## Scenarios

- reference: ordinary illustrative assumptions.
- high_displacement: a larger share of prevented focal-area property incidents is assumed to move to neighboring areas.
- reporting_suppression: only reporting probability changes; the verifier requires underlying harm to remain identical.
- concentrated_automation: a deliberately adverse livelihood shock is applied to robot-enabled arms; combined support partly mitigates the shock.
- high_robot_compromise: compromised robot weeks are more common; the verifier checks that exposure increases without altering non-robot arms.
- unequal_essential_access: the same essentials program is delivered unevenly; the verifier must detect the worsening access gap.

Policy arms are baseline, essentials, quality_work, robot_protection, and combined.

## Important units and assumptions

- The 1,000 synthetic household-equivalents, 52-week window, cohort weights, event rates, program effects, reporting probabilities, robot protection rate, false-alarm rate, compromise rates, and scenario shocks are synthetic assumptions, not estimated parameters.
- annualized_cost_units are fictional accounting units, not dollars or a cost-effectiveness finding.
- privacy_burden_points and benefit_concentration_bps are explicit proxies, not validated privacy or inequality measures.
- Need-related and opportunity-related property events are modeled separately from non-economic harm. The model does not assume economic interventions remove non-economic harm.
- displaced_property_events are included in total synthetic crime events but are separate from incidents remaining at the protected site.
- reported_crime_reports depend on the scenario's reporting probability. A decline in reports does not necessarily indicate a decline in underlying harm.
- false_alarms and robot_safety_incidents are separate from crime-event and report counts.
- Seeds provide deterministic reproducibility and common random numbers across comparable scenario arms. They do not create empirical validity.

## Claim ceiling

Passing the Rust unit tests establishes only that the implementation satisfies selected coded invariants. Passing the independent artifact verifier establishes only that the emitted TSV satisfies the frozen structural and cross-scenario checks. Neither establishes:
- that the modeled relationships match real behavior;
- that any intervention causally reduces crime;
- that most crime is economically motivated or can be made economically irrational;
- that costs, rights impacts, robot safety, or cybersecurity have been appropriately valued;
- that the model generalizes to Johannesburg, South Africa, or any other place;
- that Symthaea reasoning is accurate or ready for operational use.

Before any empirical pilot, specify a local problem, obtain privacy and community review, pre-register outcomes and a comparison strategy, collect valid baselines, measure victimization as well as reports, measure displacement and unequal effects, and qualify all robots under a separate safety and cyber-security process. The tool has no network access and no service side effects.
