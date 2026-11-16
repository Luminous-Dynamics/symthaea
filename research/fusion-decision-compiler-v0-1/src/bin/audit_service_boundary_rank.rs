use fusion_decision_compiler_contract::{
    EvidenceClass,
    lifecycle_service_ledger::{
        AvailabilityEvent, LifecycleEvidenceClass, LifecycleServicePlan,
        OutageCategory, PowerOperatingState, PowerStateSegment, ServiceImpact,
        evaluate_lifecycle_service_budget,
    },
    plant_service_gate::{
        PowerBalanceProfile, PowerBalanceRun, PowerObservable, PowerTermEvidence,
        evaluate_plant_power_balance,
    },
    service_boundary_rank::{
        CandidateServiceInput, ServiceBoundaryRankingProfile,
        ServiceRankingMetric, compare_service_boundary_rankings,
    },
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn power_profile() -> PowerBalanceProfile {
    PowerBalanceProfile {
        schema_ref: "synthetic-systems-code-net-power-v1".to_string(),
        schema_sha256: sha('a'),
        service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
        unit_ref: "MWe".to_string(),
        averaging_window_ref: "steady-state-60s-v1".to_string(),
        sign_convention_ref: "gross-and-load-positive-net-export-positive-v1".to_string(),
        absolute_balance_tolerance_mw: 0.001,
        required_net_export_reserve_mw: 0.0,
        reserve_threshold_inclusive: true,
    }
}

fn candidate(
    candidate_id: &str,
    q: f64,
    gross: f64,
    variable: f64,
    fixed: f64,
    availability: f64,
    outage_power: f64,
    cost: f64,
    digest_character: char,
) -> CandidateServiceInput {
    let profile = power_profile();
    let output_sha256 = sha(digest_character);
    let total_recirculated = variable + fixed;
    let net_electric = gross - total_recirculated;
    let run_id = format!("solver-run-{candidate_id}");
    let candidate_revision = format!("design-{candidate_id}-r1");
    let power_report_ref = format!("power-report-{candidate_id}");

    let power_run = PowerBalanceRun {
        run_id: run_id.clone(),
        solver_id: "synthetic-systems-code".to_string(),
        solver_revision: "synthetic-revision-r1".to_string(),
        candidate_revision: candidate_revision.clone(),
        output_sha256: output_sha256.clone(),
        schema_ref: profile.schema_ref.clone(),
        schema_sha256: profile.schema_sha256.clone(),
        service_boundary_ref: profile.service_boundary_ref.clone(),
        unit_ref: profile.unit_ref.clone(),
        averaging_window_ref: profile.averaging_window_ref.clone(),
        sign_convention_ref: profile.sign_convention_ref.clone(),
        receipt_ref: format!("synthetic-power-run-receipt-{candidate_id}"),
        terms: vec![
            PowerTermEvidence {
                observable: PowerObservable::GrossElectric,
                value_mw: gross,
                source_locator: "synthetic.output.gross".to_string(),
                evidence_ref: format!("gross-evidence-{candidate_id}"),
            },
            PowerTermEvidence {
                observable: PowerObservable::VariableRecirculated,
                value_mw: variable,
                source_locator: "synthetic.output.variable-recirculated".to_string(),
                evidence_ref: format!("variable-evidence-{candidate_id}"),
            },
            PowerTermEvidence {
                observable: PowerObservable::FixedRecirculated,
                value_mw: fixed,
                source_locator: "synthetic.output.fixed-recirculated".to_string(),
                evidence_ref: format!("fixed-evidence-{candidate_id}"),
            },
            PowerTermEvidence {
                observable: PowerObservable::TotalRecirculated,
                value_mw: total_recirculated,
                source_locator: "synthetic.output.total-recirculated".to_string(),
                evidence_ref: format!("total-evidence-{candidate_id}"),
            },
            PowerTermEvidence {
                observable: PowerObservable::NetElectric,
                value_mw: net_electric,
                source_locator: "synthetic.output.net-electric".to_string(),
                evidence_ref: format!("net-evidence-{candidate_id}"),
            },
        ],
    };
    let power_report = evaluate_plant_power_balance(&profile, &power_run);

    let horizon_seconds = 10_000_u64;
    let outage_seconds = ((1.0 - availability) * horizon_seconds as f64).round() as u64;
    let outage_start = horizon_seconds - outage_seconds;
    let lifecycle_plan = LifecycleServicePlan {
        campaign_id: format!("synthetic-lifecycle-{candidate_id}"),
        horizon_ref: "synthetic-10000-second-horizon-v1".to_string(),
        horizon_start_second: 0,
        horizon_end_second: horizon_seconds,
        service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
        unit_ref: "MWe".to_string(),
        sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
        events: vec![AvailabilityEvent {
            event_id: format!("planned-outage-{candidate_id}"),
            category: OutageCategory::Planned,
            impact: ServiceImpact::FullOutage,
            subsystem_id: "synthetic-plant".to_string(),
            failure_mode_id: "synthetic-maintenance".to_string(),
            start_second: outage_start,
            end_second: horizon_seconds,
            evidence_ref: format!("synthetic-outage-evidence-{candidate_id}"),
            evidence_class: LifecycleEvidenceClass::Synthetic,
        }],
        power_segments: vec![
            PowerStateSegment {
                segment_id: format!("service-{candidate_id}"),
                start_second: 0,
                end_second: outage_start,
                net_power_mwe: net_electric,
                state: PowerOperatingState::PowerGeneration,
                active_event_ids: vec![],
                power_balance_report_ref: power_report_ref.clone(),
                output_sha256: output_sha256.clone(),
                evidence_ref: format!("synthetic-service-state-{candidate_id}"),
                evidence_class: LifecycleEvidenceClass::Synthetic,
            },
            PowerStateSegment {
                segment_id: format!("outage-{candidate_id}"),
                start_second: outage_start,
                end_second: horizon_seconds,
                net_power_mwe: outage_power,
                state: PowerOperatingState::PlannedMaintenance,
                active_event_ids: vec![format!("planned-outage-{candidate_id}")],
                power_balance_report_ref: power_report_ref.clone(),
                output_sha256: output_sha256.clone(),
                evidence_ref: format!("synthetic-outage-state-{candidate_id}"),
                evidence_class: LifecycleEvidenceClass::Synthetic,
            },
        ],
    };
    let lifecycle_report = evaluate_lifecycle_service_budget(&lifecycle_plan);

    CandidateServiceInput {
        candidate_id: candidate_id.to_string(),
        candidate_revision,
        geometry_sha256: sha(digest_character),
        power_profile: profile,
        power_run,
        power_report,
        power_report_ref,
        plasma_gain_q: q,
        plasma_gain_run_id: run_id,
        plasma_gain_output_sha256: output_sha256,
        plasma_gain_evidence_ref: format!("synthetic-q-evidence-{candidate_id}"),
        lifecycle_report_ref: format!("synthetic-lifecycle-report-{candidate_id}"),
        lifecycle_report_sha256: sha(digest_character.to_ascii_uppercase()),
        lifecycle_report,
        whole_life_cost: cost,
        cost_currency_ref: "MUSD_2026".to_string(),
        cost_scope_ref: "discounted-whole-life-cost-v1".to_string(),
        cost_evidence_ref: format!("synthetic-cost-evidence-{candidate_id}"),
        evidence_class: EvidenceClass::SyntheticFixture,
    }
}

fn main() {
    let profile = ServiceBoundaryRankingProfile {
        service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
        unit_ref: "MWe".to_string(),
        power_sign_convention_ref: "gross-and-load-positive-net-export-positive-v1".to_string(),
        lifecycle_sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
        lifecycle_horizon_ref: "synthetic-10000-second-horizon-v1".to_string(),
        lifecycle_horizon_seconds: 10_000,
        cost_currency_ref: "MUSD_2026".to_string(),
        cost_scope_ref: "discounted-whole-life-cost-v1".to_string(),
        required_net_export_reserve_mwe: 0.0,
        numerical_tolerance_mwh: 0.001,
    };

    let candidates = vec![
        // Highest Q/gross, but recirculation exceeds generation: net importer.
        candidate("A", 9.9, 210.0, 175.0, 55.0, 0.90, -5.0, 4_000.0, 'a'),
        // Lower Q, lower lifetime MWh, lower cost.
        candidate("B", 6.5, 180.0, 100.0, 30.0, 0.85, -7.0, 4_200.0, 'b'),
        // Intermediate Q, highest lifetime MWh, higher cost.
        candidate("C", 7.2, 195.0, 120.0, 25.0, 0.95, -8.0, 4_900.0, 'c'),
    ];

    match compare_service_boundary_rankings(&profile, &candidates) {
        Ok(report) => {
            println!("contract_version={}", report.contract_version);
            for ranking in &report.rankings {
                println!(
                    "ranking={:?},order={}",
                    ranking.metric,
                    ranking.ordered_candidate_ids.join(",")
                );
            }
            for finding in &report.findings {
                println!(
                    "candidate={},Q={:.3},gross={:.3},net={:.3},availability={:.4},lifetime_net_mwh={:.6},cost_musd_2026={:.3},service={:?},evidence={:?}",
                    finding.candidate_id,
                    finding.plasma_gain_q,
                    finding.gross_electric_mwe,
                    finding.net_electric_mwe,
                    finding.availability_fraction,
                    finding.lifetime_net_energy_mwh,
                    finding.whole_life_cost,
                    finding.net_service_disposition,
                    finding.evidence_class
                );
            }
            for reversal in &report.pairwise_rank_reversals {
                let winners = reversal
                    .metric_winners
                    .iter()
                    .map(|winner| format!("{:?}:{}", winner.metric, winner.winner_candidate_id))
                    .collect::<Vec<_>>()
                    .join(",");
                println!(
                    "rank_reversal=({},{}),winners={}",
                    reversal.first_candidate_id,
                    reversal.second_candidate_id,
                    winners
                );
            }
            println!(
                "qualified_cost_energy_pareto={}",
                report
                    .qualified_cost_energy_pareto_front
                    .iter()
                    .map(|point| point.candidate_id.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            println!(
                "SCOPE: synthetic service-boundary rank-reversal benchmark; not an industry design result or physical validation"
            );

            if report.rankings[0].ordered_candidate_ids != vec!["A", "C", "B"]
                || report.rankings[1].ordered_candidate_ids != vec!["A", "C", "B"]
                || report.qualified_cost_energy_pareto_front
                    .iter()
                    .map(|point| point.candidate_id.as_str())
                    .collect::<Vec<_>>()
                    != vec!["B", "C"]
            {
                eprintln!("service-boundary rank-reversal fixture did not match its known-answer result");
                std::process::exit(1);
            }
        }
        Err(errors) => {
            eprintln!("service-boundary comparison refused: {errors:?}");
            std::process::exit(1);
        }
    }
}
