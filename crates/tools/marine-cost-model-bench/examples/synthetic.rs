use marine_cost_model_bench::{
    compare, CostBand, CostCase, CostEstimate, CostLine, ComparisonDisposition,
    EvidenceOrigin, ReplacementEvent,
};

fn estimate(value: f64, id: &str) -> CostEstimate {
    CostEstimate {
        amount: CostBand::exact(value),
        evidence_ref: Some(format!("synthetic:{id}")),
        origin: Some(EvidenceOrigin::Synthetic),
    }
}

fn synthetic_case(
    case_id: &str,
    non_recurring: f64,
    unit_build: f64,
    annual_operating: f64,
    replacement: f64,
    residual: f64,
) -> CostCase {
    CostCase {
        case_id: case_id.to_owned(),
        vessel_family: case_id.to_owned(),
        service_contract_id: "synthetic-utility-work-v1".to_owned(),
        currency_basis_id: "ZAR-2026-SA".to_owned(),
        cost_scope_id: "yard-gate-plus-service-v1".to_owned(),
        non_recurring_program_cost: Some(estimate(non_recurring, &format!("{case_id}-nre"))),
        unit_build_costs: Some(vec![
            CostLine {
                line_id: "hull-and-fabrication".to_owned(),
                estimate: estimate(unit_build * 0.7, &format!("{case_id}-hull")),
            },
            CostLine {
                line_id: "propulsion-and-outfit".to_owned(),
                estimate: estimate(unit_build * 0.3, &format!("{case_id}-equipment")),
            },
        ]),
        annual_operating_cost: Some(estimate(annual_operating, &format!("{case_id}-opex"))),
        replacement_plan: Some(if replacement == 0.0 {
            Vec::new()
        } else {
            vec![ReplacementEvent {
                year: 3,
                component_id: "battery-module".to_owned(),
                estimate: estimate(replacement, &format!("{case_id}-replacement")),
            }]
        }),
        terminal_residual_value: Some(estimate(residual, &format!("{case_id}-residual"))),
        service_life_years: 5,
        annual_service_units: 200.0,
        discount_rate: 0.05,
    }
}

fn main() {
    let baseline = synthetic_case("baseline", 100_000.0, 70_000.0, 18_000.0, 12_000.0, 10_000.0);
    let candidate = synthetic_case("candidate", 150_000.0, 64_000.0, 14_000.0, 15_000.0, 8_000.0);
    let report = compare(&baseline, &candidate, 20);
    assert!(report.baseline.is_some() && report.candidate.is_some());

    let disposition = match report.disposition {
        ComparisonDisposition::CandidateRobustlyCheaperWithinDeclaredBounds => "CANDIDATE_ROBUSTLY_CHEAPER",
        ComparisonDisposition::BaselineRobustlyCheaperWithinDeclaredBounds => "BASELINE_ROBUSTLY_CHEAPER",
        ComparisonDisposition::Indeterminate => "INDETERMINATE",
        ComparisonDisposition::Incomparable => "INCOMPARABLE",
        ComparisonDisposition::Invalid => "INVALID",
    };
    let baseline = report.baseline.expect("validated synthetic baseline");
    let candidate = report.candidate.expect("validated synthetic candidate");

    println!("schema,disposition,planned_volume,baseline_build_per_unit_nominal,candidate_build_per_unit_nominal,baseline_lifecycle_per_service_nominal,candidate_lifecycle_per_service_nominal");
    println!(
        "marine-cost-model-v0,{disposition},20,{:.2},{:.2},{:.4},{:.4}",
        baseline.amortized_build_cost_per_unit.nominal,
        candidate.amortized_build_cost_per_unit.nominal,
        baseline.cost_per_service_unit.nominal,
        candidate.cost_per_service_unit.nominal,
    );
    println!("All values are synthetic accounting fixtures; not market quotes or performance evidence.");
    if let Some(crossover) = report.nominal_break_even {
        println!(
            "nominal_break_even_units={:.3}; candidate_cheaper_above_crossover={}",
            crossover.vessel_units,
            crossover.candidate_cheaper_above_crossover
        );
    }
    if !report.findings.is_empty() {
        for finding in report.findings {
            eprintln!("finding: {finding}");
        }
        std::process::exit(2);
    }
}
