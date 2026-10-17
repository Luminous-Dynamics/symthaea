// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prints a synthetic screening example. It performs no network or hardware I/O.

use agri_platform_bench::{
    assess, Assessment, AssessmentStatus, EvidenceDomain, EvidenceRecord, PlatformClass,
    PlatformProfile, ScenarioProfile, ServiceUnit, TaskKind, CONTRACT_VERSION,
};

const FIXTURE_REF: &str = "inline-fixture:synthetic-agri-platform-bench-fixture-v1";
const FIXTURE_SHA256: &str =
    "db5822bd2544c966f3d8d5348e55551aef8e4d4f48cc562edf6a0deada73c3bf";

fn scenario_evidence() -> Vec<EvidenceRecord> {
    agri_platform_bench::SCENARIO_EVIDENCE_DOMAINS
        .into_iter()
        .map(|domain| EvidenceRecord::verified(domain, FIXTURE_REF, FIXTURE_SHA256))
        .collect()
}

fn platform_evidence() -> Vec<EvidenceRecord> {
    agri_platform_bench::PLATFORM_EVIDENCE_DOMAINS
        .into_iter()
        .map(|domain| EvidenceRecord::verified(domain, FIXTURE_REF, FIXTURE_SHA256))
        .collect()
}

fn demo_scenario() -> ScenarioProfile {
    ScenarioProfile {
        id: "synthetic-highveld-vegetable-scouting".into(),
        task: TaskKind::FieldScouting,
        required_daily_capacity: 1_000,
        site_slope_permille: 100,
        available_row_clearance_cm: 120,
        available_turning_radius_cm: 150,
        site_roughness_mm: 40,
        gnss_available: true,
        // Deliberately demonstrates that "no robot" remains eligible.
        max_upfront_budget_cents: 0,
        max_annualized_cost_cents: 500_000,
        annual_operating_hours: 100,
        require_local_repair: true,
        max_repair_lead_days: 14,
        evidence: scenario_evidence(),
    }
}

fn candidate(id: &str, class: PlatformClass) -> PlatformProfile {
    let manual = class == PlatformClass::ManualCurrentPractice;
    PlatformProfile {
        id: id.into(),
        class,
        supported_tasks: vec![TaskKind::FieldScouting],
        service_unit: ServiceUnit::SquareMetersPerDay,
        max_slope_permille: 300,
        width_cm: if manual { 0 } else { 80 },
        turning_radius_cm: if manual { 0 } else { 100 },
        max_roughness_mm: 100,
        requires_gnss: !manual,
        gnss_denied_capable: false,
        daily_capacity_units: 2_000,
        acquisition_cost_cents: if manual { 0 } else { 100_000 },
        operating_cost_cents_per_hour: if manual { 100 } else { 500 },
        supervision_minutes_per_hour: if manual { 60 } else { 10 },
        maintenance_minutes_per_100_hours: if manual { 0 } else { 60 },
        useful_life_years: if manual { 1 } else { 5 },
        local_repair_available: true,
        repair_lead_days: 7,
        bounded_safety_case_verified: true,
        human_override_available: true,
        evidence: platform_evidence(),
    }
}

fn status_name(status: AssessmentStatus) -> &'static str {
    match status {
        AssessmentStatus::Feasible => "FEASIBLE",
        AssessmentStatus::Provisional => "PROVISIONAL",
        AssessmentStatus::Rejected => "REJECTED",
        AssessmentStatus::Blocked => "BLOCKED",
    }
}

fn print_assessment(result: Assessment) {
    let (cost, supervision, maintenance, repair) = match result.metrics {
        Some(metrics) => (
            metrics.annualized_total_cost_cents.to_string(),
            metrics.annual_supervision_minutes.to_string(),
            metrics.annual_maintenance_minutes.to_string(),
            metrics.repair_lead_days.to_string(),
        ),
        None => ("UNKNOWN".into(), "UNKNOWN".into(), "UNKNOWN".into(), "UNKNOWN".into()),
    };
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}",
        result.candidate_id,
        status_name(result.status),
        format!("{:?}", result.blockers),
        cost,
        supervision,
        maintenance
    );
    println!("repair_lead_days={repair}");
}

fn main() {
    let scenario = demo_scenario();
    println!("contract={CONTRACT_VERSION}");
    println!("scenario={} (SYNTHETIC; NOT A SITE ASSESSMENT)", scenario.id);
    println!("candidate\tstatus\tblockers\tannualized_cost_cents\tsupervision_minutes_per_year\tmaintenance_minutes_per_year");
    print_assessment(assess(
        &scenario,
        &candidate("manual-current-practice", PlatformClass::ManualCurrentPractice),
    ));
    print_assessment(assess(
        &scenario,
        &candidate("candidate-modular-ugv", PlatformClass::ModularGroundRobot),
    ));
}
