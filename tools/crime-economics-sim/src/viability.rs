//! Aggregate lawful-alternative viability fixture and non-compensatory gate contract.
//!
//! All values here are synthetic fixtures. This module does not estimate crime,
//! classify people, measure a real labor market, or make legal judgments. Cohort
//! labels are anonymous aggregate cells and do not encode individual attributes.
//! The gates are illustrative policy-contract values, not universal standards.

pub const VIABILITY_SCHEMA_VERSION: u32 = 1;
pub const COHORT_IDS: [&str; 5] = [
    "cohort_01",
    "cohort_02",
    "cohort_03",
    "cohort_04",
    "cohort_05",
];
pub const VIABILITY_SCENARIOS: [&str; 10] = [
    "baseline",
    "high_mean_low_floor",
    "opportunity_without_quality",
    "high_access_friction",
    "threshold_straddle",
    "stale_source_data",
    "conflicted_sources",
    "missing_data",
    "wide_uncertainty",
    "opportunity_gap_hidden_by_mean",
];

pub const MIN_ESSENTIALS_ACCESS: f64 = 0.90;
pub const MIN_LAWFUL_OPPORTUNITY_COVERAGE: f64 = 0.80;
pub const MIN_FAIR_PAY_ADEQUACY: f64 = 0.80;
pub const MIN_SAFE_WORK: f64 = 0.90;
pub const MIN_WORK_STABILITY: f64 = 0.75;
pub const MIN_SOCIAL_PROTECTION: f64 = 0.75;
pub const MAX_LAWFUL_ACCESS_TIME_MINUTES: f64 = 60.0;
pub const MAX_COHORT_COVERAGE_GAP: f64 = 0.15;
pub const MAX_INTERVAL_WIDTH: f64 = 0.20;
pub const MAX_TIME_INTERVAL_WIDTH_MINUTES: f64 = 20.0;
pub const MAX_EVIDENCE_AGE_DAYS: u32 = 180;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub low: f64,
    pub high: f64,
}

const fn interval(low: f64, high: f64) -> Interval {
    Interval { low, high }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohortProfile {
    pub essentials_access: Option<Interval>,
    pub lawful_opportunity_coverage: Option<Interval>,
    pub fair_pay_adequacy: Option<Interval>,
    pub safe_work: Option<Interval>,
    pub work_stability: Option<Interval>,
    pub social_protection_coverage: Option<Interval>,
    pub lawful_access_time_minutes: Option<Interval>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scenario {
    pub id: &'static str,
    pub evidence_status: &'static str,
    pub evidence_age_days: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ViabilityRow {
    pub scenario_id: String,
    pub cohort_id: String,
    pub evidence_class: String,
    pub evidence_status: String,
    pub evidence_age_days: u32,
    pub profile: CohortProfile,
    pub cohort_gate_status: String,
    pub scenario_gate_status: String,
}

pub static SCENARIOS: [Scenario; 10] = [
    Scenario {
        id: "baseline",
        evidence_status: "usable",
        evidence_age_days: 30,
    },
    Scenario {
        id: "high_mean_low_floor",
        evidence_status: "usable",
        evidence_age_days: 30,
    },
    Scenario {
        id: "opportunity_without_quality",
        evidence_status: "usable",
        evidence_age_days: 30,
    },
    Scenario {
        id: "high_access_friction",
        evidence_status: "usable",
        evidence_age_days: 30,
    },
    Scenario {
        id: "threshold_straddle",
        evidence_status: "usable",
        evidence_age_days: 30,
    },
    Scenario {
        id: "stale_source_data",
        evidence_status: "stale",
        evidence_age_days: 270,
    },
    Scenario {
        id: "conflicted_sources",
        evidence_status: "conflicted",
        evidence_age_days: 30,
    },
    Scenario {
        id: "missing_data",
        evidence_status: "unavailable",
        evidence_age_days: 30,
    },
    Scenario {
        id: "wide_uncertainty",
        evidence_status: "usable",
        evidence_age_days: 30,
    },
    Scenario {
        id: "opportunity_gap_hidden_by_mean",
        evidence_status: "usable",
        evidence_age_days: 30,
    },
];

const fn baseline_profile(cohort: usize) -> CohortProfile {
    match cohort {
        0 => CohortProfile {
            essentials_access: Some(interval(0.94, 0.96)),
            lawful_opportunity_coverage: Some(interval(0.84, 0.86)),
            fair_pay_adequacy: Some(interval(0.82, 0.85)),
            safe_work: Some(interval(0.93, 0.96)),
            work_stability: Some(interval(0.80, 0.84)),
            social_protection_coverage: Some(interval(0.80, 0.84)),
            lawful_access_time_minutes: Some(interval(30.0, 40.0)),
        },
        1 => CohortProfile {
            essentials_access: Some(interval(0.92, 0.94)),
            lawful_opportunity_coverage: Some(interval(0.83, 0.85)),
            fair_pay_adequacy: Some(interval(0.81, 0.84)),
            safe_work: Some(interval(0.92, 0.95)),
            work_stability: Some(interval(0.78, 0.82)),
            social_protection_coverage: Some(interval(0.78, 0.82)),
            lawful_access_time_minutes: Some(interval(35.0, 45.0)),
        },
        2 => CohortProfile {
            essentials_access: Some(interval(0.93, 0.95)),
            lawful_opportunity_coverage: Some(interval(0.85, 0.87)),
            fair_pay_adequacy: Some(interval(0.83, 0.86)),
            safe_work: Some(interval(0.94, 0.97)),
            work_stability: Some(interval(0.82, 0.86)),
            social_protection_coverage: Some(interval(0.82, 0.86)),
            lawful_access_time_minutes: Some(interval(25.0, 35.0)),
        },
        3 => CohortProfile {
            essentials_access: Some(interval(0.95, 0.97)),
            lawful_opportunity_coverage: Some(interval(0.86, 0.88)),
            fair_pay_adequacy: Some(interval(0.84, 0.87)),
            safe_work: Some(interval(0.95, 0.98)),
            work_stability: Some(interval(0.84, 0.88)),
            social_protection_coverage: Some(interval(0.84, 0.88)),
            lawful_access_time_minutes: Some(interval(20.0, 30.0)),
        },
        _ => CohortProfile {
            essentials_access: Some(interval(0.91, 0.93)),
            lawful_opportunity_coverage: Some(interval(0.82, 0.84)),
            fair_pay_adequacy: Some(interval(0.80, 0.83)),
            safe_work: Some(interval(0.91, 0.94)),
            work_stability: Some(interval(0.76, 0.80)),
            social_protection_coverage: Some(interval(0.76, 0.80)),
            lawful_access_time_minutes: Some(interval(40.0, 50.0)),
        },
    }
}

fn profile_for(scenario_id: &str, cohort: usize) -> CohortProfile {
    let mut profile = baseline_profile(cohort);
    match scenario_id {
        "baseline" | "stale_source_data" | "conflicted_sources" => profile,
        "high_mean_low_floor" => {
            profile.essentials_access = Some(if cohort < 4 {
                [
                    interval(0.97, 0.99),
                    interval(0.96, 0.99),
                    interval(0.96, 0.98),
                    interval(0.95, 0.98),
                ][cohort]
            } else {
                interval(0.68, 0.72)
            });
            profile
        }
        "opportunity_without_quality" => {
            profile.lawful_opportunity_coverage = Some(interval(0.94, 0.97));
            profile.fair_pay_adequacy = Some(interval(0.45, 0.52));
            profile.safe_work = Some(interval(0.60, 0.68));
            profile.work_stability = Some(interval(0.50, 0.58));
            profile.social_protection_coverage = Some(interval(0.30, 0.40));
            profile
        }
        "high_access_friction" => {
            profile.lawful_access_time_minutes = Some(interval(90.0, 110.0));
            profile
        }
        "threshold_straddle" => {
            profile.essentials_access = Some(interval(0.88, 0.94));
            profile.lawful_opportunity_coverage = Some(interval(0.77, 0.84));
            profile.fair_pay_adequacy = Some(interval(0.76, 0.84));
            profile.safe_work = Some(interval(0.87, 0.93));
            profile.work_stability = Some(interval(0.72, 0.78));
            profile.social_protection_coverage = Some(interval(0.72, 0.78));
            profile.lawful_access_time_minutes = Some(interval(50.0, 70.0));
            profile
        }
        "missing_data" => CohortProfile {
            essentials_access: None,
            lawful_opportunity_coverage: None,
            fair_pay_adequacy: None,
            safe_work: None,
            work_stability: None,
            social_protection_coverage: None,
            lawful_access_time_minutes: None,
        },
        "wide_uncertainty" => {
            profile.essentials_access = Some(interval(0.70, 0.98));
            profile.lawful_opportunity_coverage = Some(interval(0.60, 0.96));
            profile.fair_pay_adequacy = Some(interval(0.60, 0.96));
            profile.safe_work = Some(interval(0.72, 0.97));
            profile.work_stability = Some(interval(0.60, 0.90));
            profile.social_protection_coverage = Some(interval(0.58, 0.92));
            profile.lawful_access_time_minutes = Some(interval(35.0, 85.0));
            profile
        }
        "opportunity_gap_hidden_by_mean" => {
            profile.lawful_opportunity_coverage = Some(match cohort {
                0 => interval(0.96, 0.98),
                1 => interval(0.94, 0.97),
                2 => interval(0.93, 0.96),
                3 => interval(0.92, 0.95),
                _ => interval(0.80, 0.82),
            });
            profile
        }
        _ => profile,
    }
}

fn interval_pass_or_uncertain(interval: Option<Interval>, floor: f64) -> &'static str {
    match interval {
        None => "missing",
        Some(value) if !valid_fraction(value) => "invalid",
        Some(value) if value.high < floor => "fail",
        Some(value) if value.low >= floor && value.high - value.low <= MAX_INTERVAL_WIDTH => "pass",
        Some(_) => "uncertain",
    }
}

fn lower_is_better_pass_or_uncertain(interval: Option<Interval>, ceiling: f64) -> &'static str {
    match interval {
        None => "missing",
        Some(value) if !valid_time(value) => "invalid",
        Some(value) if value.low > ceiling => "fail",
        Some(value)
            if value.high <= ceiling
                && value.high - value.low <= MAX_TIME_INTERVAL_WIDTH_MINUTES =>
        {
            "pass"
        }
        Some(_) => "uncertain",
    }
}

fn valid_fraction(value: Interval) -> bool {
    value.low.is_finite()
        && value.high.is_finite()
        && value.low >= 0.0
        && value.high <= 1.0
        && value.low <= value.high
}

fn valid_time(value: Interval) -> bool {
    value.low.is_finite() && value.high.is_finite() && value.low >= 0.0 && value.low <= value.high
}

pub fn cohort_status(scenario: &Scenario, profile: CohortProfile) -> &'static str {
    if scenario.evidence_status != "usable" || scenario.evidence_age_days > MAX_EVIDENCE_AGE_DAYS {
        return "fixture_indeterminate";
    }
    let checks = [
        interval_pass_or_uncertain(profile.essentials_access, MIN_ESSENTIALS_ACCESS),
        interval_pass_or_uncertain(
            profile.lawful_opportunity_coverage,
            MIN_LAWFUL_OPPORTUNITY_COVERAGE,
        ),
        interval_pass_or_uncertain(profile.fair_pay_adequacy, MIN_FAIR_PAY_ADEQUACY),
        interval_pass_or_uncertain(profile.safe_work, MIN_SAFE_WORK),
        interval_pass_or_uncertain(profile.work_stability, MIN_WORK_STABILITY),
        interval_pass_or_uncertain(profile.social_protection_coverage, MIN_SOCIAL_PROTECTION),
        lower_is_better_pass_or_uncertain(
            profile.lawful_access_time_minutes,
            MAX_LAWFUL_ACCESS_TIME_MINUTES,
        ),
    ];
    if checks.contains(&"invalid") {
        return "fixture_indeterminate";
    }
    if checks.contains(&"fail") {
        return "fixture_fail";
    }
    if checks.contains(&"missing") || checks.contains(&"uncertain") {
        return "fixture_indeterminate";
    }
    "fixture_pass"
}

fn scenario_gap(
    profile_selector: impl Fn(usize) -> CohortProfile,
    selector: impl Fn(CohortProfile) -> Option<Interval>,
) -> Option<f64> {
    let mut min_low = f64::INFINITY;
    let mut max_high = f64::NEG_INFINITY;
    for cohort in 0..COHORT_IDS.len() {
        let interval = selector(profile_selector(cohort))?;
        if !valid_fraction(interval) {
            return None;
        }
        min_low = min_low.min(interval.low);
        max_high = max_high.max(interval.high);
    }
    Some(max_high - min_low)
}

pub fn scenario_status(scenario: &Scenario) -> &'static str {
    if scenario.evidence_status != "usable" || scenario.evidence_age_days > MAX_EVIDENCE_AGE_DAYS {
        return "fixture_indeterminate";
    }
    let mut any_indeterminate = false;
    for cohort in 0..COHORT_IDS.len() {
        match cohort_status(scenario, profile_for(scenario.id, cohort)) {
            "fixture_fail" => return "fixture_fail",
            "fixture_indeterminate" => any_indeterminate = true,
            _ => {}
        }
    }
    if any_indeterminate {
        return "fixture_indeterminate";
    }
    let essentials_gap = scenario_gap(
        |cohort| profile_for(scenario.id, cohort),
        |profile| profile.essentials_access,
    );
    let opportunity_gap = scenario_gap(
        |cohort| profile_for(scenario.id, cohort),
        |profile| profile.lawful_opportunity_coverage,
    );
    match (essentials_gap, opportunity_gap) {
        (Some(essential), Some(opportunity))
            if essential <= MAX_COHORT_COVERAGE_GAP && opportunity <= MAX_COHORT_COVERAGE_GAP =>
        {
            "fixture_pass"
        }
        (Some(_), Some(_)) => "fixture_fail",
        _ => "fixture_indeterminate",
    }
}

pub fn scenarios() -> impl Iterator<Item = &'static Scenario> {
    SCENARIOS.iter()
}

pub fn rows() -> Vec<ViabilityRow> {
    let mut result = Vec::with_capacity(SCENARIOS.len() * COHORT_IDS.len());
    for scenario in &SCENARIOS {
        let scenario_gate_status = scenario_status(scenario).to_string();
        for (index, cohort_id) in COHORT_IDS.iter().enumerate() {
            result.push(ViabilityRow {
                scenario_id: scenario.id.to_string(),
                cohort_id: (*cohort_id).to_string(),
                evidence_class: "synthetic_fixture".to_string(),
                evidence_status: scenario.evidence_status.to_string(),
                evidence_age_days: scenario.evidence_age_days,
                profile: profile_for(scenario.id, index),
                cohort_gate_status: cohort_status(scenario, profile_for(scenario.id, index))
                    .to_string(),
                scenario_gate_status: scenario_gate_status.clone(),
            });
        }
    }
    result
}

pub const TSV_HEADER: &str = "schema_version\tscenario_id\tcohort_id\tevidence_class\tevidence_status\tevidence_age_days\tessentials_access_low\tessentials_access_high\tlawful_opportunity_coverage_low\tlawful_opportunity_coverage_high\tfair_pay_adequacy_low\tfair_pay_adequacy_high\tsafe_work_low\tsafe_work_high\twork_stability_low\twork_stability_high\tsocial_protection_coverage_low\tsocial_protection_coverage_high\tlawful_access_time_minutes_low\tlawful_access_time_minutes_high\tcohort_gate_status\tscenario_gate_status";

fn fmt_interval(interval: Option<Interval>) -> (String, String) {
    match interval {
        Some(value) => (format!("{:.3}", value.low), format!("{:.3}", value.high)),
        None => (String::new(), String::new()),
    }
}

impl ViabilityRow {
    pub fn to_tsv(&self) -> String {
        let metrics = [
            fmt_interval(self.profile.essentials_access),
            fmt_interval(self.profile.lawful_opportunity_coverage),
            fmt_interval(self.profile.fair_pay_adequacy),
            fmt_interval(self.profile.safe_work),
            fmt_interval(self.profile.work_stability),
            fmt_interval(self.profile.social_protection_coverage),
            fmt_interval(self.profile.lawful_access_time_minutes),
        ];
        let mut fields = vec![
            VIABILITY_SCHEMA_VERSION.to_string(),
            self.scenario_id.clone(),
            self.cohort_id.clone(),
            self.evidence_class.clone(),
            self.evidence_status.clone(),
            self.evidence_age_days.to_string(),
        ];
        for (low, high) in metrics {
            fields.push(low);
            fields.push(high);
        }
        fields.push(self.cohort_gate_status.clone());
        fields.push(self.scenario_gate_status.clone());
        fields.join("\t")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(id: &str) -> &'static Scenario {
        SCENARIOS
            .iter()
            .find(|scenario| scenario.id == id)
            .expect("fixture scenario missing")
    }

    #[test]
    fn fixture_rows_are_complete_and_aggregate_only() {
        assert_eq!(rows().len(), 50);
        assert_eq!(COHORT_IDS.len(), 5);
        assert_eq!(SCENARIOS.len(), 10);
        assert!(
            rows()
                .iter()
                .all(|row| row.evidence_class == "synthetic_fixture")
        );
    }

    #[test]
    fn baseline_passes_all_non_compensatory_gates() {
        assert_eq!(scenario_status(scenario("baseline")), "fixture_pass");
    }

    #[test]
    fn high_average_cannot_hide_a_low_floor_cohort() {
        let mean = (0..COHORT_IDS.len())
            .map(|index| {
                let range = profile_for("high_mean_low_floor", index)
                    .essentials_access
                    .unwrap();
                (range.low + range.high) / 2.0
            })
            .sum::<f64>()
            / COHORT_IDS.len() as f64;
        assert!(mean > MIN_ESSENTIALS_ACCESS);
        assert_eq!(
            cohort_status(
                scenario("high_mean_low_floor"),
                profile_for("high_mean_low_floor", 4)
            ),
            "fixture_fail"
        );
        assert_eq!(
            scenario_status(scenario("high_mean_low_floor")),
            "fixture_fail"
        );
    }

    #[test]
    fn opportunity_quantity_cannot_compensate_for_poor_work_quality() {
        let p = profile_for("opportunity_without_quality", 0);
        assert!(p.lawful_opportunity_coverage.unwrap().low >= MIN_LAWFUL_OPPORTUNITY_COVERAGE);
        assert!(p.fair_pay_adequacy.unwrap().high < MIN_FAIR_PAY_ADEQUACY);
        assert_eq!(
            scenario_status(scenario("opportunity_without_quality")),
            "fixture_fail"
        );
    }

    #[test]
    fn access_friction_is_an_independent_veto() {
        assert_eq!(
            scenario_status(scenario("high_access_friction")),
            "fixture_fail"
        );
    }

    #[test]
    fn uncertain_stale_conflicted_and_missing_data_do_not_pass() {
        for id in [
            "threshold_straddle",
            "stale_source_data",
            "conflicted_sources",
            "missing_data",
            "wide_uncertainty",
        ] {
            assert_eq!(
                scenario_status(scenario(id)),
                "fixture_indeterminate",
                "scenario {id}"
            );
        }
    }

    #[test]
    fn cohort_coverage_gap_cannot_be_hidden_by_a_mean() {
        assert_eq!(
            scenario_status(scenario("opportunity_gap_hidden_by_mean")),
            "fixture_fail"
        );
    }
}
