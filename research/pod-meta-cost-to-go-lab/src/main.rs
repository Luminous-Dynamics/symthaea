use pod_meta_cost_to_go_lab::{
    BudgetLedger, CompletedCalibrationExample, EffectDisposition, ForecastId, ForecastJournal,
    ForecastOutcomeReceipt, ForecastReceipt, FrontierCostCoefficients, JointCostEnvelope,
    LinearProofFrontierCostModel, ProofFrontierFeatures, ReconciliationOutcome,
    RemainingCostForecast, ReservationId, ResourceVector, ShiftStatus,
    SplitVectorConformalCalibrator, StateEvidence, TerminalOutcome, assess_completion_feasibility,
    assess_joint_resource_fit,
};

fn tokens(value: u128) -> ResourceVector {
    ResourceVector::new(value, 0, 0, 0, 0, 0)
}

fn evidence(id: ReservationId, effect: EffectDisposition) -> StateEvidence {
    StateEvidence {
        action_id: id.0,
        authoritative: true,
        state_digest: [1; 32],
        effect,
    }
}

fn print_case(name: &str, status: &str, ledger: &BudgetLedger) {
    println!(
        "CASE\t{name}\t{status}\t{}\t{}\t{}\t{}",
        ledger.spent.tokens,
        ledger.reserved.tokens,
        ledger.reservation_overruns,
        ledger.hard_budget_violations
    );
}

fn main() {
    println!(
        "record_type\tcase\tstatus\tspent_tokens\treserved_tokens\treservation_overruns\thard_budget_violations"
    );

    let mut safe = BudgetLedger::new(tokens(100));
    safe.reserve(ReservationId(1), tokens(60)).unwrap();
    print_case("reserve-within-cap", "accepted", &safe);

    let mut rejected = BudgetLedger::new(tokens(100));
    rejected.reserve(ReservationId(1), tokens(60)).unwrap();
    let result = rejected.reserve(ReservationId(2), tokens(50));
    assert!(result.is_err());
    print_case("overcommit-rejected", "rejected", &rejected);

    let mut unknown = BudgetLedger::new(tokens(100));
    unknown.reserve(ReservationId(3), tokens(40)).unwrap();
    unknown.mark_outcome_unknown(ReservationId(3)).unwrap();
    assert_eq!(unknown.reserved, tokens(40));
    print_case("unknown-retains-reserve", "held", &unknown);

    let mut released = BudgetLedger::new(tokens(100));
    released.reserve(ReservationId(4), tokens(40)).unwrap();
    released.mark_outcome_unknown(ReservationId(4)).unwrap();
    assert_eq!(
        released
            .reconcile(
                ReservationId(4),
                evidence(ReservationId(4), EffectDisposition::NotApplied)
            )
            .unwrap(),
        ReconciliationOutcome::Released
    );
    print_case("authoritative-not-applied", "released", &released);

    let mut settled = BudgetLedger::new(tokens(100));
    settled.reserve(ReservationId(5), tokens(30)).unwrap();
    settled
        .reconcile(
            ReservationId(5),
            evidence(
                ReservationId(5),
                EffectDisposition::Applied {
                    measured_usage: tokens(25),
                },
            ),
        )
        .unwrap();
    assert_eq!(settled.spent, tokens(25));
    print_case("measured-settlement", "settled", &settled);

    let mut forecast_overrun = BudgetLedger::new(tokens(100));
    forecast_overrun
        .reserve(ReservationId(6), tokens(30))
        .unwrap();
    forecast_overrun
        .reconcile(
            ReservationId(6),
            evidence(
                ReservationId(6),
                EffectDisposition::Applied {
                    measured_usage: tokens(40),
                },
            ),
        )
        .unwrap();
    print_case("reservation-overrun-visible", "recorded", &forecast_overrun);

    let mut cap_overrun = BudgetLedger::new(tokens(100));
    cap_overrun.reserve(ReservationId(7), tokens(90)).unwrap();
    cap_overrun
        .reconcile(
            ReservationId(7),
            evidence(
                ReservationId(7),
                EffectDisposition::Applied {
                    measured_usage: tokens(110),
                },
            ),
        )
        .unwrap();
    assert_eq!(cap_overrun.hard_budget_violations, 1);
    print_case(
        "hard-cap-overrun-visible",
        "violation-recorded",
        &cap_overrun,
    );

    let mut shifted_forecast = RemainingCostForecast {
        pre_state_digest: [1; 32],
        frontier_digest: [2; 32],
        model_digest: [3; 32],
        calibration_digest: [4; 32],
        completion_probability: 0.95,
        remaining_q50: tokens(30),
        remaining_q90: tokens(60),
        remaining_q95: tokens(75),
        mandatory_verification_reserve: tokens(20),
        support_count: 128,
        shift_status: ShiftStatus::UnsupportedShift,
    };
    let clear_budget = BudgetLedger::new(tokens(100));
    let shifted_decision = assess_completion_feasibility(shifted_forecast, &clear_budget, 0.90, 32);
    assert!(matches!(
        shifted_decision,
        pod_meta_cost_to_go_lab::AdmissionDecision::Abstain(_)
    ));
    shifted_forecast.shift_status = ShiftStatus::InDomain;
    assert!(matches!(
        assess_completion_feasibility(shifted_forecast, &clear_budget, 0.90, 32),
        pod_meta_cost_to_go_lab::AdmissionDecision::Admit
    ));
    println!("CASE\tshift-aware-feasibility\tshift-abstains-in-domain-admits\t0\t0\t0\t0");
    println!(
        "SURVIVAL_HEADER\tcohort\tepisodes\tcompleted\tright_censored\tcompeting\tcompletion_cif\tmedian_net_work\tq90_net_work"
    );
    let censoring_observations = [
        pod_meta_cost_to_go_lab::TimeToProofObservation::new(
            21,
            1,
            pod_meta_cost_to_go_lab::TerminalOutcome::SufficientProof,
        )
        .unwrap(),
        pod_meta_cost_to_go_lab::TimeToProofObservation::new(
            21,
            2,
            pod_meta_cost_to_go_lab::TerminalOutcome::RightCensored(
                pod_meta_cost_to_go_lab::CensorReason::BudgetExhausted,
            ),
        )
        .unwrap(),
        pod_meta_cost_to_go_lab::TimeToProofObservation::new(
            21,
            3,
            pod_meta_cost_to_go_lab::TerminalOutcome::SufficientProof,
        )
        .unwrap(),
    ];
    let censoring_summary =
        pod_meta_cost_to_go_lab::estimate_time_to_proof(&censoring_observations, Some(21)).unwrap();
    println!(
        "SURVIVAL_ROW\tbudget-censored-cohort\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}",
        censoring_summary.episodes,
        censoring_summary.completed,
        censoring_summary.right_censored,
        censoring_summary.competing_terminal,
        censoring_summary.completion_cumulative_incidence,
        censoring_summary
            .median_net_work_units
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_owned()),
        censoring_summary
            .q90_net_work_units
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_owned())
    );

    let competing_observations = [
        pod_meta_cost_to_go_lab::TimeToProofObservation::new(
            22,
            1,
            pod_meta_cost_to_go_lab::TerminalOutcome::SufficientProof,
        )
        .unwrap(),
        pod_meta_cost_to_go_lab::TimeToProofObservation::new(
            22,
            2,
            pod_meta_cost_to_go_lab::TerminalOutcome::CompetingTerminal(
                pod_meta_cost_to_go_lab::CompetingReason::PermissionRevoked,
            ),
        )
        .unwrap(),
        pod_meta_cost_to_go_lab::TimeToProofObservation::new(
            22,
            3,
            pod_meta_cost_to_go_lab::TerminalOutcome::RightCensored(
                pod_meta_cost_to_go_lab::CensorReason::ActionOutcomeUnknown,
            ),
        )
        .unwrap(),
    ];
    let competing_summary =
        pod_meta_cost_to_go_lab::estimate_time_to_proof(&competing_observations, Some(22)).unwrap();
    println!(
        "SURVIVAL_ROW\tcompeting-terminal-cohort\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}",
        competing_summary.episodes,
        competing_summary.completed,
        competing_summary.right_censored,
        competing_summary.competing_terminal,
        competing_summary.completion_cumulative_incidence,
        competing_summary
            .median_net_work_units
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_owned()),
        competing_summary
            .q90_net_work_units
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_owned())
    );
    let mut journal = ForecastJournal::default();
    let journal_forecast = RemainingCostForecast {
        pre_state_digest: [1; 32],
        frontier_digest: [2; 32],
        model_digest: [3; 32],
        calibration_digest: [4; 32],
        completion_probability: 0.95,
        remaining_q50: tokens(20),
        remaining_q90: tokens(40),
        remaining_q95: tokens(50),
        mandatory_verification_reserve: tokens(5),
        support_count: 128,
        shift_status: ShiftStatus::InDomain,
    };
    journal
        .append_forecast(ForecastReceipt {
            id: ForecastId(101),
            run_id: 31,
            decision_sequence: 1,
            task_family: 21,
            operation_family: 3,
            forecast: journal_forecast,
        })
        .unwrap();
    journal
        .append_outcome(ForecastOutcomeReceipt {
            forecast_id: ForecastId(101),
            observed_sequence: 2,
            observed_work: tokens(18),
            terminal: TerminalOutcome::SufficientProof,
            evidence_digest: [5; 32],
            meter_digest: [6; 32],
            verifier_digest: Some([7; 32]),
        })
        .unwrap();
    journal
        .append_forecast(ForecastReceipt {
            id: ForecastId(102),
            run_id: 31,
            decision_sequence: 3,
            task_family: 22,
            operation_family: 4,
            forecast: journal_forecast,
        })
        .unwrap();
    journal
        .append_outcome(ForecastOutcomeReceipt {
            forecast_id: ForecastId(102),
            observed_sequence: 4,
            observed_work: tokens(12),
            terminal: TerminalOutcome::RightCensored(
                pod_meta_cost_to_go_lab::CensorReason::BudgetExhausted,
            ),
            evidence_digest: [8; 32],
            meter_digest: [9; 32],
            verifier_digest: None,
        })
        .unwrap();
    assert_eq!(journal.forecast_count(), 2);
    assert_eq!(journal.outcome_count(), 2);
    println!(
        "FORECAST_JOURNAL_ROW\timmutable-pre-action-receipt\t{}\t{}",
        journal.forecast_count(),
        journal.outcome_count()
    );
    println!(
        "FORECAST_JOURNAL_ROW\tseparate-verifier-and-censor-outcomes\t{}\t{}",
        journal
            .outcome(ForecastId(101))
            .is_some_and(|outcome| outcome.verifier_digest.is_some()),
        journal
            .outcome(ForecastId(102))
            .is_some_and(|outcome| matches!(outcome.terminal, TerminalOutcome::RightCensored(_)))
    );
    let unit_cost = ResourceVector::new(3, 3, 3, 3, 3, 3);
    let model_digest = [21; 32];
    let population_digest = [23; 32];
    let frontier_model = LinearProofFrontierCostModel {
        model_digest,
        point_cost: FrontierCostCoefficients {
            intercept: ResourceVector::new(10, 10, 10, 10, 10, 10),
            per_unresolved_obligation: unit_cost,
            ..FrontierCostCoefficients::default()
        },
        uncertainty_scale: FrontierCostCoefficients {
            intercept: ResourceVector::new(2, 2, 2, 2, 2, 2),
            ..FrontierCostCoefficients::default()
        },
    };
    let frontier_features = ProofFrontierFeatures {
        unresolved_obligations: 4,
        mandatory_obligations: 2,
        maximum_dependency_depth: 3,
        dependency_edges: 3,
        admissible_alternatives: 4,
        missing_evidence_requirements: 1,
        distinct_operation_families: 2,
        distinct_evidence_families: 2,
        mandatory_verifier_operations: 1,
    };
    let frontier_prediction = frontier_model
        .predict(
            frontier_features,
            [22; 32],
            population_digest,
            ShiftStatus::InDomain,
        )
        .unwrap();
    let calibration_examples = (1..=9_u128)
        .map(|score| {
            let point = frontier_prediction.point_cost;
            let scale = frontier_prediction.uncertainty_scale;
            let residual = ResourceVector::new(
                score * scale.tokens,
                score * scale.tool_calls,
                score * scale.cpu_ms,
                score * scale.gpu_ms,
                score * scale.wall_ms,
                score * scale.micro_usd,
            );
            CompletedCalibrationExample {
                episode_id: score as u64,
                prediction: frontier_prediction,
                observed_remaining: point.checked_add(residual).unwrap(),
                completion_verifier_digest: [24; 32],
            }
        })
        .collect::<Vec<_>>();
    let vector_calibrator = SplitVectorConformalCalibrator::fit(
        &calibration_examples,
        0.10,
        model_digest,
        [25; 32],
        population_digest,
    )
    .unwrap();
    let joint_envelope = vector_calibrator
        .predict_upper(frontier_prediction)
        .unwrap();
    let JointCostEnvelope::Bounded(joint_bound) = joint_envelope else {
        panic!("nine exchangeable completed cases support a finite 90% conformal rank");
    };
    let joint_fit = assess_joint_resource_fit(
        &joint_envelope,
        tokens(5),
        &BudgetLedger::new(ResourceVector::new(1000, 1000, 1000, 1000, 1000, 1000)),
    );
    assert_eq!(
        joint_fit,
        pod_meta_cost_to_go_lab::JointBudgetDecision::Fits
    );
    println!(
        "VECTOR_CONFORMAL_ROW\tjoint-max-score-envelope\t{:.2}\t{}\t{}\t{}",
        joint_bound.target_coverage,
        joint_bound.calibration_count,
        joint_bound.conformal_threshold,
        joint_bound.upper.tokens
    );

    let shifted_prediction = frontier_model
        .predict(
            frontier_features,
            [22; 32],
            population_digest,
            ShiftStatus::SuspectedShift,
        )
        .unwrap();
    let shifted_envelope = vector_calibrator.predict_upper(shifted_prediction).unwrap();
    assert!(matches!(
        shifted_envelope,
        JointCostEnvelope::Unbounded { .. }
    ));
    println!("VECTOR_CONFORMAL_ROW\tshift-abstention\tunbounded");

    let small_calibrator = SplitVectorConformalCalibrator::fit(
        &calibration_examples[..8],
        0.10,
        model_digest,
        [26; 32],
        population_digest,
    )
    .unwrap();
    assert!(matches!(
        small_calibrator.predict_upper(frontier_prediction).unwrap(),
        JointCostEnvelope::Unbounded { .. }
    ));
    println!("VECTOR_CONFORMAL_ROW\tfinite-sample-rank-overflow\tunbounded");
}
