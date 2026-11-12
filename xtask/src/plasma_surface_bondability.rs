// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline, fail-closed evaluator for the synthetic plasma surface bondability
//! fixture. It validates data identity, complete-system accounting, bond
//! acceptance over declared treatment-to-bond delays, and substrate-property
//! gates. It does not predict plasma chemistry or physical bond strength.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const EXPECTED_EVIDENCE_CLASS: &str = "SYNTHETIC";
const MIN_REPLICATES_PER_DELAY: usize = 3;
const ENERGY_RELATIVE_TOLERANCE: f64 = 1.0e-9;
const FRACTION_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ArmDisposition {
    Pass,
    Blocked,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ComparisonDisposition {
    Comparable,
    NotComparable,
    Blocked,
}

#[derive(Debug, Clone, Deserialize)]
struct Fixture {
    schema_version: String,
    fixture_id: String,
    evidence_class: String,
    synthetic_only: bool,
    luminous_physical_reproduction: bool,
    claim_ceiling: String,
    service_profile: ServiceProfile,
    energy_boundary: EnergyBoundary,
    arms: Vec<Arm>,
    comparisons: Vec<Comparison>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceProfile {
    service_profile_id: String,
    substrate_family: String,
    substrate_grade: String,
    substrate_lot: String,
    adhesive_system: String,
    joint_geometry: String,
    test_method: String,
    acceptance_threshold_mpa: f64,
    minimum_acceptance_fraction: f64,
    bondability_delays_hours: Vec<u64>,
    storage_profile_id: String,
    required_property_gates: Vec<PropertyGate>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PropertyGate {
    property: String,
    unit: String,
    minimum: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnergyBoundary {
    energy_boundary_contract_id: String,
    basis: String,
    required_subsystems: Vec<String>,
    missing_load_policy: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Arm {
    arm_id: String,
    process_class: String,
    service_profile_id: String,
    energy_boundary_contract_id: String,
    complete_system_energy_kwh_per_m2: f64,
    subsystem_energy_kwh_per_m2: BTreeMap<String, f64>,
    contact_angle_deg: f64,
    substrate_functional_property_retention: f64,
    bond_strength_mpa_by_delay: BTreeMap<String, Vec<f64>>,
    expected_disposition: ArmDisposition,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Comparison {
    comparison_id: String,
    baseline_arm_id: String,
    candidate_arm_id: String,
    expected_disposition: ComparisonDisposition,
}

#[derive(Debug, Clone, Serialize)]
pub struct DelayResult {
    pub delay_hours: u64,
    pub accepted_replicates: usize,
    pub total_replicates: usize,
    pub acceptance_fraction_ppm: u64,
    pub meets_acceptance_fraction: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArmResult {
    pub arm_id: String,
    pub disposition: ArmDisposition,
    pub reason: String,
    pub complete_system_energy_kwh_per_m2: Option<f64>,
    pub contact_angle_deg_diagnostic_only: Option<f64>,
    pub substrate_functional_property_retention: Option<f64>,
    pub bondability_window_hours: Option<u64>,
    pub delay_results: Vec<DelayResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonResult {
    pub comparison_id: String,
    pub disposition: ComparisonDisposition,
    pub reason: String,
    pub baseline_arm_id: String,
    pub candidate_arm_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditReport {
    pub schema_version: String,
    pub fixture_id: String,
    pub evidence_class: String,
    pub synthetic_only: bool,
    pub luminous_physical_reproduction: bool,
    pub claim_ceiling: String,
    pub arm_results: Vec<ArmResult>,
    pub comparison_results: Vec<ComparisonResult>,
    pub all_expected_dispositions_matched: bool,
}

pub fn run_audit(fixture_path: &Path) -> Result<AuditReport> {
    let bytes = fs::read(fixture_path)
        .with_context(|| format!("read bondability fixture {}", fixture_path.display()))?;
    let fixture: Fixture = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse bondability fixture {}", fixture_path.display()))?;
    evaluate_fixture(fixture)
}

fn evaluate_fixture(fixture: Fixture) -> Result<AuditReport> {
    ensure!(fixture.schema_version == "0.1.0", "unsupported schema version");
    ensure!(!fixture.fixture_id.trim().is_empty(), "empty fixture ID");
    ensure!(
        fixture.evidence_class == EXPECTED_EVIDENCE_CLASS,
        "fixture evidence class must remain SYNTHETIC"
    );
    ensure!(fixture.synthetic_only, "fixture must be marked synthetic-only");
    ensure!(
        !fixture.luminous_physical_reproduction,
        "synthetic fixture cannot claim Luminous physical reproduction"
    );
    ensure!(!fixture.claim_ceiling.trim().is_empty(), "empty claim ceiling");

    validate_service_profile(&fixture.service_profile)?;
    validate_energy_boundary(&fixture.energy_boundary)?;

    let mut arm_ids = BTreeSet::new();
    let mut arm_results = BTreeMap::<String, ArmResult>::new();
    let mut arms = BTreeMap::<String, Arm>::new();

    for arm in fixture.arms {
        ensure!(!arm.arm_id.trim().is_empty(), "empty arm ID");
        ensure!(arm_ids.insert(arm.arm_id.clone()), "duplicate arm ID: {}", arm.arm_id);

        let result = evaluate_arm(&arm, &fixture.service_profile, &fixture.energy_boundary);
        ensure!(
            result.disposition == arm.expected_disposition,
            "arm {} expected {:?}, evaluator returned {:?}: {}",
            arm.arm_id,
            arm.expected_disposition,
            result.disposition,
            result.reason
        );
        arms.insert(arm.arm_id.clone(), arm);
        arm_results.insert(result.arm_id.clone(), result);
    }

    let mut comparison_ids = BTreeSet::new();
    let mut comparison_results = Vec::with_capacity(fixture.comparisons.len());
    for comparison in &fixture.comparisons {
        ensure!(
            !comparison.comparison_id.trim().is_empty(),
            "empty comparison ID"
        );
        ensure!(
            comparison_ids.insert(comparison.comparison_id.clone()),
            "duplicate comparison ID: {}",
            comparison.comparison_id
        );
        let result = evaluate_comparison(&comparison, &arms, &arm_results);
        ensure!(
            result.disposition == comparison.expected_disposition,
            "comparison {} expected {:?}, evaluator returned {:?}: {}",
            comparison.comparison_id,
            comparison.expected_disposition,
            result.disposition,
            result.reason
        );
        comparison_results.push(result);
    }

    let mut ordered_arm_results: Vec<_> = arm_results.into_values().collect();
    ordered_arm_results.sort_by(|a, b| a.arm_id.cmp(&b.arm_id));
    comparison_results.sort_by(|a, b| a.comparison_id.cmp(&b.comparison_id));

    Ok(AuditReport {
        schema_version: fixture.schema_version,
        fixture_id: fixture.fixture_id,
        evidence_class: fixture.evidence_class,
        synthetic_only: fixture.synthetic_only,
        luminous_physical_reproduction: fixture.luminous_physical_reproduction,
        claim_ceiling: fixture.claim_ceiling,
        arm_results: ordered_arm_results,
        comparison_results,
        all_expected_dispositions_matched: true,
    })
}

fn validate_service_profile(profile: &ServiceProfile) -> Result<()> {
    for (name, value) in [
        ("service profile ID", profile.service_profile_id.as_str()),
        ("substrate family", profile.substrate_family.as_str()),
        ("substrate grade", profile.substrate_grade.as_str()),
        ("substrate lot", profile.substrate_lot.as_str()),
        ("adhesive system", profile.adhesive_system.as_str()),
        ("joint geometry", profile.joint_geometry.as_str()),
        ("test method", profile.test_method.as_str()),
        ("storage profile", profile.storage_profile_id.as_str()),
    ] {
        ensure!(!value.trim().is_empty(), "empty {name}");
    }
    ensure!(
        profile.acceptance_threshold_mpa.is_finite()
            && profile.acceptance_threshold_mpa > 0.0,
        "bond acceptance threshold must be finite and positive"
    );
    ensure!(
        profile.minimum_acceptance_fraction.is_finite()
            && profile.minimum_acceptance_fraction > 0.0
            && profile.minimum_acceptance_fraction <= 1.0,
        "minimum acceptance fraction must be within (0, 1]"
    );
    ensure!(
        !profile.bondability_delays_hours.is_empty(),
        "no bondability delay points declared"
    );
    let mut delays = profile.bondability_delays_hours.clone();
    delays.sort_unstable();
    delays.dedup();
    ensure!(
        delays.len() == profile.bondability_delays_hours.len(),
        "duplicate bondability delay"
    );
    ensure!(
        profile.bondability_delays_hours.iter().copied().eq(delays.iter().copied()),
        "bondability delays must be strictly increasing"
    );
    ensure!(
        profile.bondability_delays_hours.first() == Some(&0),
        "bondability series must include the immediate (0 h) point"
    );
    ensure!(
        !profile.required_property_gates.is_empty(),
        "at least one substrate-preservation gate is required"
    );
    for gate in &profile.required_property_gates {
        ensure!(!gate.property.trim().is_empty(), "empty property gate name");
        ensure!(!gate.unit.trim().is_empty(), "empty property gate unit");
        ensure!(
            gate.minimum.is_finite(),
            "property gate minimum must be finite"
        );
    }
    Ok(())
}

fn validate_energy_boundary(boundary: &EnergyBoundary) -> Result<()> {
    ensure!(
        !boundary.energy_boundary_contract_id.trim().is_empty(),
        "empty energy-boundary contract ID"
    );
    ensure!(
        boundary.basis == "kWh_per_m2_treated",
        "unsupported energy basis: {}",
        boundary.basis
    );
    ensure!(
        boundary.missing_load_policy == "UNKNOWN_NOT_ZERO",
        "missing-load policy must be UNKNOWN_NOT_ZERO"
    );
    ensure!(
        !boundary.required_subsystems.is_empty(),
        "no required energy subsystems declared"
    );
    let mut seen = BTreeSet::new();
    for subsystem in &boundary.required_subsystems {
        ensure!(!subsystem.trim().is_empty(), "empty subsystem name");
        ensure!(
            seen.insert(subsystem),
            "duplicate required subsystem: {subsystem}"
        );
    }
    Ok(())
}

fn evaluate_arm(
    arm: &Arm,
    profile: &ServiceProfile,
    boundary: &EnergyBoundary,
) -> ArmResult {
    let blocked = |reason: &str| arm_result(arm, ArmDisposition::Blocked, reason, None, Vec::new());
    let rejected =
        |reason: &str| arm_result(arm, ArmDisposition::Rejected, reason, None, Vec::new());

    if arm.service_profile_id != profile.service_profile_id {
        return rejected("arm service-profile identity does not match the frozen fixture profile");
    }
    if arm.energy_boundary_contract_id != boundary.energy_boundary_contract_id {
        return rejected("arm energy-boundary identity does not match the frozen fixture boundary");
    }
    if !arm.complete_system_energy_kwh_per_m2.is_finite()
        || arm.complete_system_energy_kwh_per_m2 < 0.0
    {
        return rejected("complete-system energy must be finite and non-negative");
    }
    if !arm.contact_angle_deg.is_finite()
        || !(0.0..=180.0).contains(&arm.contact_angle_deg)
    {
        return rejected("contact-angle diagnostic must be finite and within [0, 180]");
    }
    if !arm.substrate_functional_property_retention.is_finite()
        || arm.substrate_functional_property_retention < 0.0
    {
        return rejected("substrate-property retention must be finite and non-negative");
    }

    let required: BTreeSet<_> = boundary.required_subsystems.iter().collect();
    let observed: BTreeSet<_> = arm.subsystem_energy_kwh_per_m2.keys().collect();
    if required != observed {
        if required.is_superset(&observed) {
            return blocked("one or more required subsystem energy observations are missing");
        }
        return rejected("energy record contains an undeclared subsystem");
    }

    let mut energy_sum = 0.0;
    for (subsystem, energy) in &arm.subsystem_energy_kwh_per_m2 {
        if !energy.is_finite() || *energy < 0.0 {
            return rejected(&format!("invalid energy for subsystem {subsystem}"));
        }
        energy_sum += energy;
    }
    if !energy_sum.is_finite() {
        return rejected("subsystem energy sum overflowed");
    }
    let tolerance = ENERGY_RELATIVE_TOLERANCE
        * arm.complete_system_energy_kwh_per_m2.abs().max(1.0);
    if (energy_sum - arm.complete_system_energy_kwh_per_m2).abs() > tolerance {
        return rejected("subsystem energy sum does not match complete-system energy");
    }

    for gate in &profile.required_property_gates {
        if gate.property != "substrate_functional_property_retention" {
            return blocked("fixture does not provide a measurement for a required property gate");
        }
        if arm.substrate_functional_property_retention + FRACTION_EPSILON < gate.minimum {
            return rejected("substrate functional-property retention violates the frozen gate");
        }
    }

    let expected_delay_keys: BTreeSet<_> = profile
        .bondability_delays_hours
        .iter()
        .map(u64::to_string)
        .collect();
    let observed_delay_keys: BTreeSet<_> = arm.bond_strength_mpa_by_delay.keys().cloned().collect();
    if expected_delay_keys != observed_delay_keys {
        if expected_delay_keys.is_superset(&observed_delay_keys) {
            return blocked("one or more declared treatment-to-bond delay groups are missing");
        }
        return rejected("bondability data contains an undeclared delay group");
    }

    let mut delay_results = Vec::with_capacity(profile.bondability_delays_hours.len());
    let mut window = None;
    let mut continuous_window = true;

    for delay_hours in &profile.bondability_delays_hours {
        let key = delay_hours.to_string();
        let Some(values) = arm.bond_strength_mpa_by_delay.get(&key) else {
            return blocked("missing bond-strength replicate group");
        };
        if values.len() < MIN_REPLICATES_PER_DELAY {
            return blocked("too few bond-strength replicates at a declared delay");
        }
        if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return rejected("bond-strength replicates must be finite and non-negative");
        }

        let accepted = values
            .iter()
            .filter(|value| **value >= profile.acceptance_threshold_mpa)
            .count();
        let fraction = accepted as f64 / values.len() as f64;
        let meets = fraction + FRACTION_EPSILON >= profile.minimum_acceptance_fraction;
        delay_results.push(DelayResult {
            delay_hours: *delay_hours,
            accepted_replicates: accepted,
            total_replicates: values.len(),
            acceptance_fraction_ppm: (fraction * 1_000_000.0).round() as u64,
            meets_acceptance_fraction: meets,
        });

        if continuous_window && meets {
            window = Some(*delay_hours);
        } else {
            continuous_window = false;
        }
    }

    arm_result(
        arm,
        ArmDisposition::Pass,
        "identities, energy closure, replicates and substrate-preservation gates are complete",
        window,
        delay_results,
    )
}

fn arm_result(
    arm: &Arm,
    disposition: ArmDisposition,
    reason: &str,
    bondability_window_hours: Option<u64>,
    delay_results: Vec<DelayResult>,
) -> ArmResult {
    ArmResult {
        arm_id: arm.arm_id.clone(),
        disposition,
        reason: reason.to_owned(),
        complete_system_energy_kwh_per_m2: Some(arm.complete_system_energy_kwh_per_m2),
        contact_angle_deg_diagnostic_only: Some(arm.contact_angle_deg),
        substrate_functional_property_retention: Some(arm.substrate_functional_property_retention),
        bondability_window_hours,
        delay_results,
    }
}

fn evaluate_comparison(
    comparison: &Comparison,
    arms: &BTreeMap<String, Arm>,
    results: &BTreeMap<String, ArmResult>,
) -> ComparisonResult {
    let result = |disposition, reason: &str| ComparisonResult {
        comparison_id: comparison.comparison_id.clone(),
        disposition,
        reason: reason.to_owned(),
        baseline_arm_id: comparison.baseline_arm_id.clone(),
        candidate_arm_id: comparison.candidate_arm_id.clone(),
    };

    let (Some(baseline), Some(candidate), Some(baseline_result), Some(candidate_result)) = (
        arms.get(&comparison.baseline_arm_id),
        arms.get(&comparison.candidate_arm_id),
        results.get(&comparison.baseline_arm_id),
        results.get(&comparison.candidate_arm_id),
    ) else {
        return result(ComparisonDisposition::Blocked, "one or both referenced arms are missing");
    };

    if baseline_result.disposition != ArmDisposition::Pass
        || candidate_result.disposition != ArmDisposition::Pass
    {
        return result(
            ComparisonDisposition::Blocked,
            "one or both arms did not pass their own evidence and preservation gates",
        );
    }
    if baseline.service_profile_id != candidate.service_profile_id
        || baseline.energy_boundary_contract_id != candidate.energy_boundary_contract_id
    {
        return result(
            ComparisonDisposition::NotComparable,
            "service-profile or complete energy-boundary identity differs",
        );
    }
    result(
        ComparisonDisposition::Comparable,
        "matched service and complete energy-boundary identities; no scalar winner is inferred",
    )
}

pub fn print_report(report: &AuditReport, as_json: bool) -> Result<()> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }
    println!("Fixture: {}", report.fixture_id);
    println!("Evidence: {}", report.evidence_class);
    println!("Claim ceiling: {}", report.claim_ceiling);
    for arm in &report.arm_results {
        println!(
            "  {:?} {} bondability_window_h={:?} reason={}",
            arm.disposition, arm.arm_id, arm.bondability_window_hours, arm.reason
        );
    }
    for comparison in &report.comparison_results {
        println!(
            "  {:?} {} baseline={} candidate={} reason={}",
            comparison.disposition,
            comparison.comparison_id,
            comparison.baseline_arm_id,
            comparison.candidate_arm_id,
            comparison.reason
        );
    }
    println!(
        "Expected dispositions matched: {}",
        report.all_expected_dispositions_matched
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Fixture {
        serde_json::from_str(include_str!(
            "../../data/research/plasma_surface_bondability_fixture_v0_1.json"
        ))
        .expect("synthetic bondability fixture parses")
    }

    fn result_for<'a>(report: &'a AuditReport, arm_id: &str) -> &'a ArmResult {
        report
            .arm_results
            .iter()
            .find(|result| result.arm_id == arm_id)
            .expect("arm result exists")
    }

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let report = evaluate_fixture(fixture()).unwrap();
        assert!(report.synthetic_only);
        assert!(!report.luminous_physical_reproduction);
        assert!(report.all_expected_dispositions_matched);
        assert_eq!(report.arm_results.len(), 5);
        assert_eq!(report.comparison_results.len(), 3);
    }

    #[test]
    fn argon_candidate_retains_synthetic_bondability_window_through_168_hours() {
        let report = evaluate_fixture(fixture()).unwrap();
        assert_eq!(
            result_for(&report, "modular_argon_candidate").bondability_window_hours,
            Some(168)
        );
    }

    #[test]
    fn untreated_control_has_no_qualifying_bondability_window() {
        let report = evaluate_fixture(fixture()).unwrap();
        assert_eq!(
            result_for(&report, "untreated_control").bondability_window_hours,
            None
        );
    }

    #[test]
    fn over_treatment_is_rejected_by_substrate_preservation_gate() {
        let report = evaluate_fixture(fixture()).unwrap();
        assert_eq!(
            result_for(&report, "over_treatment_damage_control").disposition,
            ArmDisposition::Rejected
        );
    }

    #[test]
    fn changing_contact_angle_cannot_change_bondability_results_or_comparability() {
        let mut input = fixture();
        let before = evaluate_fixture(input.clone()).unwrap();
        for arm in &mut input.arms {
            arm.contact_angle_deg = 1.0 + (180.0 - arm.contact_angle_deg);
        }
        let after = evaluate_fixture(input).unwrap();
        for arm_id in [
            "untreated_control",
            "corona_reference",
            "modular_argon_candidate",
            "nitrogen_low_angle_control",
        ] {
            assert_eq!(
                result_for(&before, arm_id).bondability_window_hours,
                result_for(&after, arm_id).bondability_window_hours
            );
            assert_eq!(
                result_for(&before, arm_id).delay_results.len(),
                result_for(&after, arm_id).delay_results.len()
            );
        }
        assert_eq!(
            before.comparison_results
                .iter()
                .map(|x| x.disposition)
                .collect::<Vec<_>>(),
            after.comparison_results
                .iter()
                .map(|x| x.disposition)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn missing_auxiliary_or_compressor_load_is_blocked_not_zeroed() {
        let mut input = fixture();
        input.arms[2]
            .subsystem_energy_kwh_per_m2
            .remove("process_gas_or_compressed_air");
        let result = evaluate_arm(
            &input.arms[2],
            &input.service_profile,
            &input.energy_boundary,
        );
        assert_eq!(result.disposition, ArmDisposition::Blocked);
    }

    #[test]
    fn subsystem_total_mismatch_is_rejected() {
        let mut input = fixture();
        *input.arms[2]
            .subsystem_energy_kwh_per_m2
            .get_mut("cooling")
            .unwrap() += 0.01;
        let result = evaluate_arm(
            &input.arms[2],
            &input.service_profile,
            &input.energy_boundary,
        );
        assert_eq!(result.disposition, ArmDisposition::Rejected);
    }

    #[test]
    fn too_few_replicates_blocks_an_arm() {
        let mut input = fixture();
        input.arms[2].bond_strength_mpa_by_delay.insert("24".to_owned(), vec![7.0, 7.1]);
        let result = evaluate_arm(
            &input.arms[2],
            &input.service_profile,
            &input.energy_boundary,
        );
        assert_eq!(result.disposition, ArmDisposition::Blocked);
    }

    #[test]
    fn energy_boundary_mismatch_is_not_comparable() {
        let mut input = fixture();
        let baseline = input.arms[1].clone();
        let mut candidate = input.arms[2].clone();
        candidate.energy_boundary_contract_id = "SYNTHETIC_DIFFERENT_BOUNDARY".to_owned();
        let mut arms = BTreeMap::new();
        arms.insert(baseline.arm_id.clone(), baseline.clone());
        arms.insert(candidate.arm_id.clone(), candidate.clone());
        let mut results = BTreeMap::new();
        results.insert(
            baseline.arm_id.clone(),
            evaluate_arm(&baseline, &input.service_profile, &input.energy_boundary),
        );
        results.insert(
            candidate.arm_id.clone(),
            ArmResult {
                disposition: ArmDisposition::Pass,
                ..evaluate_arm(&input.arms[2], &input.service_profile, &input.energy_boundary)
            },
        );
        let pair = Comparison {
            comparison_id: "boundary-mismatch".to_owned(),
            baseline_arm_id: baseline.arm_id,
            candidate_arm_id: candidate.arm_id,
            expected_disposition: ComparisonDisposition::NotComparable,
        };
        assert_eq!(
            evaluate_comparison(&pair, &arms, &results).disposition,
            ComparisonDisposition::NotComparable
        );
    }

    #[test]
    fn service_profile_mismatch_is_not_comparable() {
        let input = fixture();
        let baseline = input.arms[1].clone();
        let mut candidate = input.arms[2].clone();
        candidate.service_profile_id = "SYNTHETIC_DIFFERENT_PROFILE".to_owned();
        let mut arms = BTreeMap::new();
        arms.insert(baseline.arm_id.clone(), baseline.clone());
        arms.insert(candidate.arm_id.clone(), candidate.clone());
        let mut results = BTreeMap::new();
        results.insert(
            baseline.arm_id.clone(),
            evaluate_arm(&baseline, &input.service_profile, &input.energy_boundary),
        );
        results.insert(
            candidate.arm_id.clone(),
            ArmResult {
                disposition: ArmDisposition::Pass,
                ..evaluate_arm(&input.arms[2], &input.service_profile, &input.energy_boundary)
            },
        );
        let pair = Comparison {
            comparison_id: "service-mismatch".to_owned(),
            baseline_arm_id: baseline.arm_id,
            candidate_arm_id: candidate.arm_id,
            expected_disposition: ComparisonDisposition::NotComparable,
        };
        assert_eq!(
            evaluate_comparison(&pair, &arms, &results).disposition,
            ComparisonDisposition::NotComparable
        );
    }
}
