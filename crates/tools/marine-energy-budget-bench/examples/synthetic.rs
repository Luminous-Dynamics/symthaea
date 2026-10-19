use marine_energy_budget_bench::{
    compare, ComparisonDisposition, EnergyCase, EvidenceOrigin, MissionSpec,
};

fn synthetic_case(
    case_id: &str,
    propulsion_wh: f64,
    auxiliary_wh: f64,
) -> EnergyCase {
    EnergyCase {
        case_id: case_id.to_owned(),
        mission: MissionSpec {
            mission_id: "synthetic-auv-survey-v1".to_owned(),
            conditions_digest: "conditions-sha256:synthetic-001".to_owned(),
            route_distance_m: 10_000.0,
            payload_kg: Some(2_000.0),
            target_speed_mps: 1.0,
            arrival_policy_id: "target-speed-fixed-v1".to_owned(),
        },
        energy_boundary_id: "battery-electrical-input-v1".to_owned(),
        configuration_digest: Some(format!("config:{case_id}")),
        evidence_ref: Some(format!("synthetic-evidence:{case_id}")),
        evidence_origin: Some(EvidenceOrigin::Synthetic),
        uncertainty_method_id: Some("conservative-absolute-sum-v1".to_owned()),
        propulsion_input_wh: propulsion_wh,
        propulsion_uncertainty_wh: 10.0,
        auxiliary_input_wh: Some(auxiliary_wh),
        auxiliary_uncertainty_wh: Some(5.0),
    }
}

fn main() {
    let baseline = synthetic_case("baseline", 1_000.0, 100.0);
    let candidate = synthetic_case("candidate", 900.0, 100.0);
    let report = compare(&baseline, &candidate);
    assert_eq!(
        report.disposition,
        ComparisonDisposition::RobustSavingWithinDeclaredBounds
    );

    let baseline_metrics = report.baseline_metrics.expect("synthetic baseline metrics");
    let candidate_metrics = report.candidate_metrics.expect("synthetic candidate metrics");
    let saving = report.nominal_saving_fraction.expect("nominal saving");
    let bounds = report.saving_fraction_interval.expect("conservative bounds");

    // Stable CSV demonstration; all values are synthetic and only test accounting.
    println!("schema,disposition,baseline_total_wh,candidate_total_wh,nominal_saving_fraction,conservative_saving_lower,conservative_saving_upper,baseline_wh_per_km,candidate_wh_per_km");
    println!(
        "marine-energy-budget-v0,{:?},{:.3},{:.3},{:.9},{:.9},{:.9},{:.3},{:.3}",
        report.disposition,
        baseline_metrics.total_input_wh,
        candidate_metrics.total_input_wh,
        saving,
        bounds.lower,
        bounds.upper,
        baseline_metrics.wh_per_km,
        candidate_metrics.wh_per_km,
    );
}
