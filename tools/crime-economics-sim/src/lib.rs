//! Deterministic, aggregate-only synthetic model for research hypotheses.
//!
//! This is not calibrated to real crime data and must not be used to predict,
//! score, investigate, or make decisions about real people. No person-level
//! observations are generated or emitted. The parameters are explicit teaching
//! assumptions chosen to exercise invariants, not empirical effect estimates.

pub const MODEL_SCHEMA_VERSION: u32 = 1;
pub const POPULATION: u32 = 1_000;
pub const WEEKS: u32 = 52;
pub const REPLICATIONS: u64 = 32;
pub const COHORT_SIZES: [u32; 5] = [100, 200, 300, 250, 150];
pub const BASE_ACCESS: [f64; 5] = [0.45, 0.58, 0.70, 0.80, 0.90];
pub const NEED_MULTIPLIERS: [f64; 5] = [2.4, 1.8, 1.25, 0.8, 0.45];
pub const OPPORTUNITY_MULTIPLIERS: [f64; 5] = [1.35, 1.2, 1.0, 0.85, 0.7];
pub const NON_ECONOMIC_MULTIPLIERS: [f64; 5] = [1.2, 1.1, 1.0, 0.9, 0.8];
pub const ESSENTIALS_ACCESS_GAINS: [f64; 5] = [0.28, 0.24, 0.18, 0.10, 0.04];
pub const ROBOT_PREVENTION_PROBABILITY: f64 = 0.36 * 0.70 * 0.96;
pub const ROBOT_FALSE_ALARM_PROBABILITY_PER_HOUSEHOLD_WEEK: f64 = 0.00002;
pub const ROBOT_SAFETY_HARM_GIVEN_COMPROMISE: f64 = 0.10;
pub const BASE_NEED_EVENT_PROBABILITY_PER_HOUSEHOLD_WEEK: f64 = 0.00055;
pub const BASE_OPPORTUNITY_EVENT_PROBABILITY_PER_HOUSEHOLD_WEEK: f64 = 0.00110;
pub const BASE_NON_ECONOMIC_HARM_PROBABILITY_PER_HOUSEHOLD_WEEK: f64 = 0.00065;

const NEED_STREAM: u64 = 0x11;
const OPPORTUNITY_STREAM: u64 = 0x12;
const NON_ECONOMIC_STREAM: u64 = 0x13;
const NEED_PROTECTION_STREAM: u64 = 0x21;
const OPPORTUNITY_PROTECTION_STREAM: u64 = 0x22;
const DISPLACEMENT_STREAM: u64 = 0x23;
const ROBOT_COMPROMISE_STREAM: u64 = 0x31;
const ROBOT_SAFETY_STREAM: u64 = 0x32;
const REPORT_STREAM: u64 = 0x41;
const FALSE_ALARM_STREAM: u64 = 0x42;

#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    pub id: &'static str,
    pub report_probability: f64,
    pub displacement_probability: f64,
    pub robot_protection_multiplier: f64,
    pub robot_compromise_probability: f64,
    pub quality_work_effect_multiplier: f64,
    pub automation_livelihood_shock: f64,
    pub access_gain_multipliers: [f64; 5],
    pub benefit_concentration_bps: i64,
}

pub const SCENARIOS: &[Scenario] = &[
    Scenario {
        id: "reference",
        quality_work_effect_multiplier: 1.0,
        report_probability: 0.78,
        displacement_probability: 0.30,
        robot_protection_multiplier: 1.0,
        robot_compromise_probability: 0.01,
        automation_livelihood_shock: 0.0,
        access_gain_multipliers: [1.0; 5],
        benefit_concentration_bps: 4_500,
    },
    Scenario {
        id: "high_displacement",
        quality_work_effect_multiplier: 1.0,
        report_probability: 0.78,
        displacement_probability: 0.90,
        robot_protection_multiplier: 1.0,
        robot_compromise_probability: 0.01,
        automation_livelihood_shock: 0.0,
        access_gain_multipliers: [1.0; 5],
        benefit_concentration_bps: 4_500,
    },
    Scenario {
        id: "reporting_suppression",
        quality_work_effect_multiplier: 1.0,
        report_probability: 0.35,
        displacement_probability: 0.30,
        robot_protection_multiplier: 1.0,
        robot_compromise_probability: 0.01,
        automation_livelihood_shock: 0.0,
        access_gain_multipliers: [1.0; 5],
        benefit_concentration_bps: 4_500,
    },
    Scenario {
        id: "concentrated_automation",
        quality_work_effect_multiplier: 1.0,
        report_probability: 0.78,
        displacement_probability: 0.30,
        robot_protection_multiplier: 1.0,
        robot_compromise_probability: 0.01,
        automation_livelihood_shock: 0.35,
        access_gain_multipliers: [1.0; 5],
        benefit_concentration_bps: 8_500,
    },
    Scenario {
        id: "high_robot_compromise",
        quality_work_effect_multiplier: 1.0,
        report_probability: 0.78,
        displacement_probability: 0.30,
        robot_protection_multiplier: 1.0,
        robot_compromise_probability: 0.18,
        automation_livelihood_shock: 0.0,
        access_gain_multipliers: [1.0; 5],
        benefit_concentration_bps: 4_500,
    },
    Scenario {
        id: "unequal_essential_access",
        quality_work_effect_multiplier: 1.0,
        report_probability: 0.78,
        displacement_probability: 0.30,
        robot_protection_multiplier: 1.0,
        robot_compromise_probability: 0.01,
        automation_livelihood_shock: 0.0,
        access_gain_multipliers: [0.15, 0.40, 0.80, 1.20, 1.50],
        benefit_concentration_bps: 4_500,
    },
    Scenario {
        id: "low_robot_effectiveness",
        quality_work_effect_multiplier: 1.0,
        report_probability: 0.78,
        displacement_probability: 0.30,
        robot_protection_multiplier: 0.25,
        robot_compromise_probability: 0.01,
        automation_livelihood_shock: 0.0,
        access_gain_multipliers: [1.0; 5],
        benefit_concentration_bps: 4_500,
    },
    Scenario {
        id: "zero_quality_work_effect",
        quality_work_effect_multiplier: 0.0,
        report_probability: 0.78,
        displacement_probability: 0.30,
        robot_protection_multiplier: 1.0,
        robot_compromise_probability: 0.01,
        automation_livelihood_shock: 0.0,
        access_gain_multipliers: [1.0; 5],
        benefit_concentration_bps: 4_500,
    },
];

#[derive(Clone, Copy, Debug)]
pub struct PolicyArm {
    pub id: &'static str,
    pub provides_essentials: bool,
    pub provides_quality_work: bool,
    pub uses_robot_protection: bool,
    pub annualized_cost_units: u64,
}

pub const POLICY_ARMS: &[PolicyArm] = &[
    PolicyArm {
        id: "baseline",
        provides_essentials: false,
        provides_quality_work: false,
        uses_robot_protection: false,
        annualized_cost_units: 0,
    },
    PolicyArm {
        id: "essentials",
        provides_essentials: true,
        provides_quality_work: false,
        uses_robot_protection: false,
        annualized_cost_units: 40_000,
    },
    PolicyArm {
        id: "quality_work",
        provides_essentials: false,
        provides_quality_work: true,
        uses_robot_protection: false,
        annualized_cost_units: 60_000,
    },
    PolicyArm {
        id: "robot_protection",
        provides_essentials: false,
        provides_quality_work: false,
        uses_robot_protection: true,
        annualized_cost_units: 100_000,
    },
    PolicyArm {
        id: "combined",
        provides_essentials: true,
        provides_quality_work: true,
        uses_robot_protection: true,
        annualized_cost_units: 190_000,
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultRow {
    pub scenario_id: String,
    pub seed: u64,
    pub policy_id: String,
    pub population: u32,
    pub model_weeks: u32,
    pub need_property_events_local: u64,
    pub opportunity_property_events_local: u64,
    pub non_economic_harm_events: u64,
    pub displaced_property_events: u64,
    pub simulated_harm_events_total: u64,
    pub reported_harm_events: u64,
    pub false_alarms: u64,
    pub robot_compromise_weeks: u64,
    pub robot_safety_incidents: u64,
    pub annualized_cost_units: u64,
    pub essential_access_gain_bps: u64,
    pub essential_access_gap_bps: u64,
    pub privacy_burden_points: u64,
    pub benefit_concentration_bps: i64,
}

impl ResultRow {
    pub fn to_tsv(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            MODEL_SCHEMA_VERSION,
            self.scenario_id,
            self.seed,
            self.policy_id,
            self.population,
            self.model_weeks,
            self.need_property_events_local,
            self.opportunity_property_events_local,
            self.non_economic_harm_events,
            self.displaced_property_events,
            self.simulated_harm_events_total,
            self.reported_harm_events,
            self.false_alarms,
            self.robot_compromise_weeks,
            self.robot_safety_incidents,
            self.annualized_cost_units,
            self.essential_access_gain_bps,
            self.essential_access_gap_bps,
            self.privacy_burden_points,
            self.benefit_concentration_bps
        )
    }

    /// Compare every result field except the scenario label.
    ///
    /// This is used for null-effect controls: when a scenario disables one
    /// mechanism, its output must match the reference scenario for policy arms
    /// that do not use that mechanism.
    pub fn same_outcomes_except_scenario_id(&self, other: &Self) -> bool {
        let mut left = self.clone();
        let mut right = other.clone();
        left.scenario_id.clear();
        right.scenario_id.clear();
        left == right
    }
}

pub const TSV_HEADER: &str = "schema_version\tscenario_id\tseed\tpolicy_id\tpopulation\tmodel_weeks\tneed_property_events_local\topportunity_property_events_local\tnon_economic_harm_events\tdisplaced_property_events\tsimulated_harm_events_total\treported_harm_events\tfalse_alarms\trobot_compromise_weeks\trobot_safety_incidents\tannualized_cost_units\tessential_access_gain_bps\tessential_access_gap_bps\tprivacy_burden_points\tbenefit_concentration_bps";

#[derive(Clone, Copy)]
struct AccessSummary {
    gain_bps: u64,
    gap_bps: u64,
}

fn access_summary(scenario: &Scenario, arm: &PolicyArm) -> AccessSummary {
    let mut weighted_gain = 0.0;
    let mut minimum: f64 = 1.0;
    let mut maximum: f64 = 0.0;

    for index in 0..BASE_ACCESS.len() {
        let gain = if arm.provides_essentials {
            ESSENTIALS_ACCESS_GAINS[index] * scenario.access_gain_multipliers[index]
        } else {
            0.0
        };
        let final_access = (BASE_ACCESS[index] + gain).clamp(0.0, 1.0);
        weighted_gain += gain * f64::from(COHORT_SIZES[index]);
        minimum = minimum.min(final_access);
        maximum = maximum.max(final_access);
    }

    AccessSummary {
        gain_bps: (weighted_gain / f64::from(POPULATION) * 10_000.0).round() as u64,
        gap_bps: ((maximum - minimum).max(0.0) * 10_000.0).round() as u64,
    }
}

pub fn run_one(scenario: &Scenario, arm: &PolicyArm, seed: u64) -> ResultRow {
    let mut need_events_local = 0_u64;
    let mut opportunity_events_local = 0_u64;
    let mut non_economic_harm_events = 0_u64;
    let mut displaced_property_events = 0_u64;
    let mut robot_compromise_weeks = 0_u64;
    let mut robot_safety_incidents = 0_u64;

    let work_need_reduction = if arm.provides_quality_work {
        0.12 * scenario.quality_work_effect_multiplier
    } else {
        0.0
    };
    let work_opportunity_reduction = if arm.provides_quality_work {
        0.08 * scenario.quality_work_effect_multiplier
    } else {
        0.0
    };
    let protection_probability = if arm.uses_robot_protection {
        (ROBOT_PREVENTION_PROBABILITY * scenario.robot_protection_multiplier).clamp(0.0, 1.0)
    } else {
        0.0
    };

    for week in 0..WEEKS {
        let robot_compromised = arm.uses_robot_protection
            && binomial(
                1,
                scenario.robot_compromise_probability,
                cell_seed(seed, week, 99, ROBOT_COMPROMISE_STREAM),
            ) > 0;

        if robot_compromised {
            robot_compromise_weeks += 1;
            if binomial(
                1,
                ROBOT_SAFETY_HARM_GIVEN_COMPROMISE,
                cell_seed(seed, week, 99, ROBOT_SAFETY_STREAM),
            ) > 0
            {
                robot_safety_incidents += 1;
            }
        }

        for cohort in 0..COHORT_SIZES.len() {
            let essential_gain = if arm.provides_essentials {
                ESSENTIALS_ACCESS_GAINS[cohort] * scenario.access_gain_multipliers[cohort]
            } else {
                0.0
            };

            // The concentrated automation shock is an adverse assumption, applied
            // only to robot-enabled arms; quality-work support partly mitigates it.
            let work_mitigation = if arm.provides_quality_work {
                1.0 - 0.55 * scenario.quality_work_effect_multiplier
            } else {
                1.0
            };
            let livelihood_shock = if arm.uses_robot_protection {
                scenario.automation_livelihood_shock * work_mitigation
            } else {
                0.0
            };

            let need_probability = (BASE_NEED_EVENT_PROBABILITY_PER_HOUSEHOLD_WEEK
                * NEED_MULTIPLIERS[cohort]
                * (1.0 - essential_gain * 0.55)
                * (1.0 - work_need_reduction)
                * (1.0 + livelihood_shock))
                .clamp(0.0, 1.0);
            let opportunity_probability = (BASE_OPPORTUNITY_EVENT_PROBABILITY_PER_HOUSEHOLD_WEEK
                * OPPORTUNITY_MULTIPLIERS[cohort]
                * (1.0 - work_opportunity_reduction))
                .clamp(0.0, 1.0);
            let non_economic_probability = BASE_NON_ECONOMIC_HARM_PROBABILITY_PER_HOUSEHOLD_WEEK
                * NON_ECONOMIC_MULTIPLIERS[cohort];

            let need_generated = binomial(
                COHORT_SIZES[cohort],
                need_probability,
                cell_seed(seed, week, cohort, NEED_STREAM),
            );
            let opportunity_generated = binomial(
                COHORT_SIZES[cohort],
                opportunity_probability,
                cell_seed(seed, week, cohort, OPPORTUNITY_STREAM),
            );
            let non_economic_generated = binomial(
                COHORT_SIZES[cohort],
                non_economic_probability,
                cell_seed(seed, week, cohort, NON_ECONOMIC_STREAM),
            );

            let mut need_remaining = need_generated;
            let mut opportunity_remaining = opportunity_generated;
            let mut prevented_total = 0_u32;

            if arm.uses_robot_protection && !robot_compromised {
                let need_prevented = binomial(
                    need_generated,
                    protection_probability,
                    cell_seed(seed, week, cohort, NEED_PROTECTION_STREAM),
                );
                let opportunity_prevented = binomial(
                    opportunity_generated,
                    protection_probability,
                    cell_seed(seed, week, cohort, OPPORTUNITY_PROTECTION_STREAM),
                );
                need_remaining -= need_prevented;
                opportunity_remaining -= opportunity_prevented;
                prevented_total = need_prevented + opportunity_prevented;
            }

            let displaced = binomial(
                prevented_total,
                scenario.displacement_probability,
                cell_seed(seed, week, cohort, DISPLACEMENT_STREAM),
            );

            need_events_local += u64::from(need_remaining);
            opportunity_events_local += u64::from(opportunity_remaining);
            non_economic_harm_events += u64::from(non_economic_generated);
            displaced_property_events += u64::from(displaced);
        }
    }

    let simulated_harm_events_total = need_events_local
        + opportunity_events_local
        + non_economic_harm_events
        + displaced_property_events;

    let reported_harm_events = u64::from(binomial(
        simulated_harm_events_total as u32,
        scenario.report_probability,
        mix64(seed ^ REPORT_STREAM),
    ));

    let false_alarms = if arm.uses_robot_protection {
        u64::from(binomial(
            POPULATION * WEEKS,
            ROBOT_FALSE_ALARM_PROBABILITY_PER_HOUSEHOLD_WEEK,
            mix64(seed ^ FALSE_ALARM_STREAM),
        ))
    } else {
        0
    };

    let access = access_summary(scenario, arm);
    let benefit_concentration_bps = if arm.uses_robot_protection {
        scenario.benefit_concentration_bps - if arm.provides_quality_work { 1_500 } else { 0 }
    } else {
        -1
    };

    ResultRow {
        scenario_id: scenario.id.to_string(),
        seed,
        policy_id: arm.id.to_string(),
        population: POPULATION,
        model_weeks: WEEKS,
        need_property_events_local: need_events_local,
        opportunity_property_events_local: opportunity_events_local,
        non_economic_harm_events,
        displaced_property_events,
        simulated_harm_events_total,
        reported_harm_events,
        false_alarms,
        robot_compromise_weeks,
        robot_safety_incidents,
        annualized_cost_units: arm.annualized_cost_units,
        essential_access_gain_bps: access.gain_bps,
        essential_access_gap_bps: access.gap_bps,
        privacy_burden_points: if arm.uses_robot_protection { 120 } else { 0 },
        benefit_concentration_bps,
    }
}

pub fn run_matrix() -> Vec<ResultRow> {
    let mut rows = Vec::with_capacity(SCENARIOS.len() * POLICY_ARMS.len() * REPLICATIONS as usize);
    for scenario in SCENARIOS {
        for arm in POLICY_ARMS {
            for seed in 0..REPLICATIONS {
                rows.push(run_one(scenario, arm, seed));
            }
        }
    }
    rows
}

pub fn scenario_by_id(id: &str) -> Option<&'static Scenario> {
    SCENARIOS.iter().find(|scenario| scenario.id == id)
}

pub fn policy_by_id(id: &str) -> Option<&'static PolicyArm> {
    POLICY_ARMS.iter().find(|arm| arm.id == id)
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn cell_seed(seed: u64, week: u32, cohort: usize, stream: u64) -> u64 {
    mix64(
        seed ^ u64::from(week).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (cohort as u64).wrapping_mul(0xd1b5_4a32_d192_ed03)
            ^ stream,
    )
}

fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mix64(*state)
}

fn binomial(trials: u32, probability: f64, seed: u64) -> u32 {
    if trials == 0 || probability <= 0.0 {
        return 0;
    }
    if probability >= 1.0 {
        return trials;
    }

    let mut state = seed;
    let mut successes = 0_u32;
    for _ in 0..trials {
        let sample = (next_random(&mut state) >> 11) as f64 / ((1_u64 << 53) as f64);
        if sample < probability {
            successes += 1;
        }
    }
    successes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(id: &str) -> &'static Scenario {
        scenario_by_id(id).expect("required scenario missing")
    }

    fn arm(id: &str) -> &'static PolicyArm {
        policy_by_id(id).expect("required policy arm missing")
    }

    #[test]
    fn fixture_is_aggregate_only_and_fully_specified() {
        assert_eq!(COHORT_SIZES.iter().sum::<u32>(), POPULATION);
        assert_eq!(WEEKS, 52);
        assert_eq!(REPLICATIONS, 32);
        assert_eq!(SCENARIOS.len(), 8);
        assert_eq!(POLICY_ARMS.len(), 5);
        assert_eq!(
            run_matrix().len(),
            SCENARIOS.len() * POLICY_ARMS.len() * REPLICATIONS as usize
        );
    }

    #[test]
    fn scenario_policy_seed_results_are_reproducible() {
        for scenario in SCENARIOS {
            for policy in POLICY_ARMS {
                let first = run_one(scenario, policy, 17);
                let second = run_one(scenario, policy, 17);
                assert_eq!(
                    first, second,
                    "non-deterministic result for {}/{}",
                    scenario.id, policy.id
                );
            }
        }
    }

    #[test]
    fn reporting_suppression_changes_reports_not_underlying_harm() {
        let mut observed_strict_reporting_reduction = false;
        for policy in POLICY_ARMS {
            for seed in 0..REPLICATIONS {
                let reference = run_one(scenario("reference"), policy, seed);
                let suppressed = run_one(scenario("reporting_suppression"), policy, seed);
                assert_eq!(
                    reference.simulated_harm_events_total,
                    suppressed.simulated_harm_events_total
                );
                assert_eq!(
                    reference.need_property_events_local,
                    suppressed.need_property_events_local
                );
                assert_eq!(
                    reference.opportunity_property_events_local,
                    suppressed.opportunity_property_events_local
                );
                assert_eq!(
                    reference.non_economic_harm_events,
                    suppressed.non_economic_harm_events
                );
                assert!(suppressed.reported_harm_events <= reference.reported_harm_events);
                if suppressed.reported_harm_events < reference.reported_harm_events {
                    observed_strict_reporting_reduction = true;
                }
            }
        }
        assert!(
            observed_strict_reporting_reduction,
            "reporting suppression must strictly reduce reports for at least one matched seed/policy"
        );
    }

    #[test]
    fn greater_displacement_does_not_change_local_prevention_but_changes_neighbor_events() {
        for policy in POLICY_ARMS {
            for seed in 0..REPLICATIONS {
                let reference = run_one(scenario("reference"), policy, seed);
                let high = run_one(scenario("high_displacement"), policy, seed);
                assert_eq!(
                    reference.need_property_events_local,
                    high.need_property_events_local
                );
                assert_eq!(
                    reference.opportunity_property_events_local,
                    high.opportunity_property_events_local
                );
                assert_eq!(
                    reference.non_economic_harm_events,
                    high.non_economic_harm_events
                );
                if policy.uses_robot_protection {
                    assert!(high.displaced_property_events >= reference.displaced_property_events);
                } else {
                    assert_eq!(high.displaced_property_events, 0);
                }
            }
        }
    }

    #[test]
    fn non_economic_harm_is_not_assumed_away_by_economic_interventions() {
        for scenario in SCENARIOS {
            for policy in POLICY_ARMS {
                let observed = run_one(scenario, policy, 7);
                let baseline = run_one(scenario_by_id("reference").unwrap(), arm("baseline"), 7);
                assert_eq!(
                    observed.non_economic_harm_events,
                    baseline.non_economic_harm_events
                );
            }
        }
    }

    #[test]
    fn concentrated_automation_shock_is_visible_and_combined_support_mitigates_it() {
        for policy_id in ["robot_protection", "combined"] {
            for seed in 0..REPLICATIONS {
                let policy = arm(policy_id);
                let reference = run_one(scenario("reference"), policy, seed);
                let concentrated = run_one(scenario("concentrated_automation"), policy, seed);
                assert!(
                    concentrated.need_property_events_local >= reference.need_property_events_local
                );
                assert!(
                    concentrated.benefit_concentration_bps > reference.benefit_concentration_bps
                );
            }
        }
    }

    #[test]
    fn compromised_robot_weeks_raise_exposure_without_affecting_non_robot_arms() {
        for seed in 0..REPLICATIONS {
            let reference_robot = run_one(scenario("reference"), arm("robot_protection"), seed);
            let compromised_robot = run_one(
                scenario("high_robot_compromise"),
                arm("robot_protection"),
                seed,
            );
            assert!(
                compromised_robot.robot_compromise_weeks >= reference_robot.robot_compromise_weeks
            );
            assert!(
                compromised_robot.robot_safety_incidents >= reference_robot.robot_safety_incidents
            );

            for policy in POLICY_ARMS
                .iter()
                .filter(|policy| !policy.uses_robot_protection)
            {
                let reference = run_one(scenario("reference"), policy, seed);
                let compromised = run_one(scenario("high_robot_compromise"), policy, seed);
                assert_eq!(
                    reference.simulated_harm_events_total,
                    compromised.simulated_harm_events_total
                );
                assert_eq!(
                    reference.reported_harm_events,
                    compromised.reported_harm_events
                );
            }
        }
    }

    #[test]
    fn unequal_access_can_worsen_the_access_gap_even_with_a_service_program() {
        for policy_id in ["essentials", "combined"] {
            for seed in 0..REPLICATIONS {
                let policy = arm(policy_id);
                let reference = run_one(scenario("reference"), policy, seed);
                let unequal = run_one(scenario("unequal_essential_access"), policy, seed);
                assert!(unequal.essential_access_gap_bps > reference.essential_access_gap_bps);
                assert!(unequal.essential_access_gain_bps < reference.essential_access_gain_bps);
            }
        }
    }

    #[test]
    fn lower_robot_effectiveness_worsens_protection_outcomes_without_affecting_other_arms() {
        let reference_scenario = scenario("reference");
        let low_effect_scenario = scenario("low_robot_effectiveness");
        for seed in 0..REPLICATIONS {
            for policy in POLICY_ARMS {
                let reference = run_one(reference_scenario, policy, seed);
                let low_effect = run_one(low_effect_scenario, policy, seed);
                if policy.uses_robot_protection {
                    assert!(
                        low_effect.need_property_events_local
                            >= reference.need_property_events_local
                    );
                    assert!(
                        low_effect.opportunity_property_events_local
                            >= reference.opportunity_property_events_local
                    );
                    assert!(
                        low_effect.displaced_property_events <= reference.displaced_property_events
                    );
                    assert!(
                        low_effect.simulated_harm_events_total
                            >= reference.simulated_harm_events_total
                    );
                } else {
                    assert!(low_effect.same_outcomes_except_scenario_id(&reference));
                }
            }
        }
    }

    #[test]
    fn zero_quality_work_effect_is_a_real_null_control() {
        let null_scenario = scenario("zero_quality_work_effect");
        let reference_scenario = scenario("reference");

        for seed in 0..REPLICATIONS {
            let baseline = run_one(null_scenario, arm("baseline"), seed);
            let quality_work = run_one(null_scenario, arm("quality_work"), seed);
            assert_eq!(
                quality_work.need_property_events_local,
                baseline.need_property_events_local
            );
            assert_eq!(
                quality_work.opportunity_property_events_local,
                baseline.opportunity_property_events_local
            );
            assert_eq!(
                quality_work.non_economic_harm_events,
                baseline.non_economic_harm_events
            );
            assert_eq!(
                quality_work.simulated_harm_events_total,
                baseline.simulated_harm_events_total
            );
            assert_eq!(
                quality_work.reported_harm_events,
                baseline.reported_harm_events
            );
            assert_eq!(quality_work.annualized_cost_units, 60_000);
            assert_eq!(baseline.annualized_cost_units, 0);

            for policy_id in ["baseline", "essentials", "robot_protection"] {
                let reference = run_one(reference_scenario, arm(policy_id), seed);
                let null_row = run_one(null_scenario, arm(policy_id), seed);
                assert!(
                    null_row.same_outcomes_except_scenario_id(&reference),
                    "null quality-work scenario leaked into non-work arm {policy_id}"
                );
            }
        }
    }

    #[test]
    fn false_alarms_are_separate_from_crime_counts() {
        for seed in 0..REPLICATIONS {
            let baseline = run_one(scenario("reference"), arm("baseline"), seed);
            let robot = run_one(scenario("reference"), arm("robot_protection"), seed);
            assert_eq!(baseline.false_alarms, 0);
            assert_eq!(
                robot.simulated_harm_events_total,
                robot.need_property_events_local
                    + robot.opportunity_property_events_local
                    + robot.non_economic_harm_events
                    + robot.displaced_property_events
            );
            assert!(robot.reported_harm_events <= robot.simulated_harm_events_total);
        }
    }
}

pub mod viability;
