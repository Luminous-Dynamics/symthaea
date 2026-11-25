use symthaea_semiconductor::benchmark::{
    CA_OCRC_FEEDBACK_SCHEMA_V1, CA_OCRC_ROUND_INPUT_SCHEMA_V1,
    CaOcrcEvidenceDispositionV1, CaOcrcFeedbackArrivalV1, CaOcrcPolicyV1,
    CaOcrcPredictionSetKindV1, CaOcrcRoundInputV1,
    advance_ca_ocrc_round_v1, create_ca_ocrc_engine_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn feedback(origin: u64, delivered: u64, loss: f64) -> CaOcrcFeedbackArrivalV1 {
    CaOcrcFeedbackArrivalV1 {
        schema: CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
        origin_round_id: origin,
        delivered_round_id: delivered,
        actual_delay: delivered - origin,
        loss: Some(loss),
        evidence_disposition: Some(CaOcrcEvidenceDispositionV1::Qualified),
        evidence_receipt_sha256: Some(digest('c')),
    }
}

fn input(
    round: u64,
    draw: f64,
    arrivals: Vec<CaOcrcFeedbackArrivalV1>,
) -> CaOcrcRoundInputV1 {
    CaOcrcRoundInputV1 {
        schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
        round_id: round,
        prediction_set_kind: CaOcrcPredictionSetKindV1::NonTrivial,
        proxy_delay_uniform_draw: Some(draw),
        model_artifact_sha256: digest('a'),
        score_policy_sha256: digest('b'),
        feedback_arrivals: arrivals,
    }
}

/// Independent scalar oracle for the paper's bounded-delay update:
/// r_(t+1) = r_t + eta * sum_i (loss_i - alpha) / F(actual_delay_i).
#[test]
fn multi_arrival_round_matches_independent_scalar_ipw_oracle() {
    let policy = CaOcrcPolicyV1 {
        schema: "luminous.semi-bench.ca-ocrc-policy.v1".into(),
        policy_id: "ca-ocrc-independent-oracle".into(),
        capacity: 2,
        maximum_feedback_delay: 2,
        target_risk: 0.20,
        learning_rate: 0.10,
        score_bound: 1.0,
        proxy_delay_survival: vec![1.0, 0.50, 0.25],
    };
    let mut engine = create_ca_ocrc_engine_v1(policy, 0.40).unwrap();

    let r1 = advance_ca_ocrc_round_v1(&mut engine, &input(1, 0.90, vec![])).unwrap();
    let r2 = advance_ca_ocrc_round_v1(&mut engine, &input(2, 0.90, vec![])).unwrap();
    assert!(r1.admitted_current_round && r2.admitted_current_round);

    // Round 3 admits before it sees feedback. K=2 is full, so the new
    // prediction is denied; both tracked outcomes then arrive in the same round.
    let arrivals = vec![
        feedback(1, 3, 1.0), // age 2; F(2)=0.25; weight 4
        feedback(2, 3, 0.0), // age 1; F(1)=0.50; weight 2
    ];
    let r3 = advance_ca_ocrc_round_v1(&mut engine, &input(3, 0.90, arrivals)).unwrap();

    let eta = 0.10_f64;
    let alpha = 0.20_f64;
    let first = eta * (1.0 - alpha) / 0.25;
    let second = eta * (0.0 - alpha) / 0.50;
    let oracle_delta = first + second;
    let oracle_threshold = 0.40 + oracle_delta;

    assert!(!r3.admitted_current_round);
    assert_eq!(r3.capacity_occupancy_after, 0);
    assert_eq!(r3.feedback.len(), 2);
    assert!((r3.feedback_threshold_contribution - oracle_delta).abs() < 1.0e-12);
    assert!((r3.threshold_after - oracle_threshold).abs() < 1.0e-12);
    assert!((engine.threshold - oracle_threshold).abs() < 1.0e-12);
}

/// A transport permutation must not alter the mathematical update or receipt.
#[test]
fn independent_oracle_is_invariant_to_feedback_container_order() {
    fn run(reverse: bool) -> (f64, String) {
        let policy = CaOcrcPolicyV1 {
            schema: "luminous.semi-bench.ca-ocrc-policy.v1".into(),
            policy_id: "ca-ocrc-independent-oracle".into(),
            capacity: 2,
            maximum_feedback_delay: 2,
            target_risk: 0.20,
            learning_rate: 0.10,
            score_bound: 1.0,
            proxy_delay_survival: vec![1.0, 0.50, 0.25],
        };
        let mut engine = create_ca_ocrc_engine_v1(policy, 0.40).unwrap();
        advance_ca_ocrc_round_v1(&mut engine, &input(1, 0.90, vec![])).unwrap();
        advance_ca_ocrc_round_v1(&mut engine, &input(2, 0.90, vec![])).unwrap();
        let mut arrivals = vec![feedback(1, 3, 1.0), feedback(2, 3, 0.0)];
        if reverse {
            arrivals.reverse();
        }
        let result = advance_ca_ocrc_round_v1(
            &mut engine,
            &input(3, 0.90, arrivals),
        )
        .unwrap();
        (result.threshold_after, result.receipt_sha256)
    }
    assert_eq!(run(false), run(true));
}
