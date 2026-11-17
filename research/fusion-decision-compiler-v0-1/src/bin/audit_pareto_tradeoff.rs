use fusion_decision_compiler_contract::pareto_tradeoff::{
    CandidateEligibility, CandidateObjectiveVector, DominanceRelation, ObjectiveComparisonProfile,
    ParetoFrontStatus, ScalarPreference, ScalarTradeoffDisposition, SourcedScalar, compare_pair,
    extract_pareto_front,
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn scalar(value: f64, evidence_id: &str) -> Option<SourcedScalar> {
    Some(SourcedScalar {
        value,
        evidence_ref: format!("synthetic-evidence-{evidence_id}"),
    })
}

fn profile() -> ObjectiveComparisonProfile {
    ObjectiveComparisonProfile {
        profile_id: "synthetic-fusion-service-vector-v1".to_string(),
        profile_sha256: sha('a'),
        service_boundary_ref: "synthetic-plant-terminal-net-electric-v1".to_string(),
        lifetime_horizon_ref: "synthetic-30-year-life-v1".to_string(),
        cost_basis_ref: "synthetic-usd-2026-real-discount-6pct-v1".to_string(),
        ebi_normalization_ref: "synthetic-qeng-ebi-normalization-v1".to_string(),
        uncertainty_basis_ref: "synthetic-net-export-95pct-interval-v1".to_string(),
    }
}

fn candidate(id: &str, values: [f64; 7]) -> CandidateObjectiveVector {
    let [
        energy_mwh,
        whole_life_cost,
        availability,
        planned_downtime,
        unplanned_downtime,
        ebi,
        uncertainty_width_mwe,
    ] = values;
    let profile = profile();
    CandidateObjectiveVector {
        candidate_id: id.to_string(),
        candidate_revision: format!("{id}-synthetic-r1"),
        profile_id: profile.profile_id,
        profile_sha256: profile.profile_sha256,
        objective_vector_receipt_ref: format!("synthetic-vector-receipt-{id}"),
        objective_vector_receipt_sha256: sha('b'),
        eligibility: CandidateEligibility::Eligible,
        lifetime_net_electric_energy_mwh: scalar(energy_mwh, &format!("{id}-energy")),
        whole_life_cost: scalar(whole_life_cost, &format!("{id}-cost")),
        operational_availability: scalar(availability, &format!("{id}-availability")),
        planned_downtime_hours: scalar(planned_downtime, &format!("{id}-planned")),
        unplanned_downtime_hours: scalar(unplanned_downtime, &format!("{id}-unplanned")),
        engineering_burden_index: scalar(ebi, &format!("{id}-ebi")),
        net_export_bound_width_mwe: scalar(uncertainty_width_mwe, &format!("{id}-uncertainty")),
        ebi_normalization_ref: profile.ebi_normalization_ref,
        uncertainty_basis_ref: profile.uncertainty_basis_ref,
    }
}

fn main() {
    let profile = profile();
    let higher_qeng = candidate(
        "higher-qeng",
        [8_000.0, 12_000.0, 0.95, 500.0, 300.0, 0.8, 25.0],
    );
    let higher_service = candidate(
        "higher-service",
        12_000.0,
        8_000.0,
        0.90,
        650.0,
        350.0,
        0.9,
        15.0,
    );

    let pair = compare_pair(&profile, &higher_qeng, &higher_service);
    println!("contract_version={}", pair.contract_version);
    println!("higher_qeng_scalar_preference={:?}", pair.scalar_preference);
    println!("qeng_a={:.6}", pair.first_q_eng.unwrap().q_eng);
    println!("qeng_b={:.6}", pair.second_q_eng.unwrap().q_eng);
    println!("pareto_dominance={:?}", pair.dominance);
    println!("scalar_tradeoff={:?}", pair.scalar_tradeoff);
    println!(
        "scalar_winner_worse_dimensions={:?}",
        pair.scalar_winner_worse_dimensions
    );

    if pair.scalar_preference != ScalarPreference::First
        || pair.dominance != DominanceRelation::NonDominatedTradeoff
        || pair.scalar_tradeoff
            != ScalarTradeoffDisposition::ScalarPreferenceHidesOmittedObjectiveTradeoff
    {
        eprintln!("Q_ENG trade-off fixture did not produce the expected incomparable preference");
        std::process::exit(1);
    }

    let strict_dominator = candidate("dominant", 13_000.0, 7_000.0, 0.97, 250.0, 150.0, 0.6, 10.0);
    let frontier = extract_pareto_front(&profile, &[higher_qeng, higher_service, strict_dominator]);
    println!("pareto_front={:?}", frontier.frontier_candidate_ids);
    println!(
        "non_dominated_statuses={}",
        frontier
            .candidates
            .iter()
            .filter(|item| item.status == ParetoFrontStatus::NonDominated)
            .count()
    );
    println!(
        "SCOPE: synthetic Q_ENG/Pareto comparator fixtures only; not a reproduced Q_ENG paper result or reactor ranking"
    );

    if frontier.frontier_candidate_ids != vec!["dominant".to_string()] {
        eprintln!("synthetic Pareto frontier did not match the known-answer fixture");
        std::process::exit(1);
    }
}
