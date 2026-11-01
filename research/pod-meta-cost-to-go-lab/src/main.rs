use pod_meta_cost_to_go_lab::{
    BudgetLedger, EffectDisposition, ReconciliationOutcome, RemainingCostForecast, ReservationId,
    ResourceVector, ShiftStatus, StateEvidence, assess_completion_feasibility,
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
}
