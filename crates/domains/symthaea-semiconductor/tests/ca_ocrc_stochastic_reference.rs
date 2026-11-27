#![forbid(unsafe_code)]

use symthaea_semiconductor::benchmark::{
    CA_OCRC_FEEDBACK_SCHEMA_V1, CA_OCRC_POLICY_SCHEMA_V1, CA_OCRC_ROUND_INPUT_SCHEMA_V1,
    CaoOcrcDelayAssumptionV1, CaoOcrcEngineV1, CaoOcrcFeedbackInputV1, CaoOcrcPolicyV1,
    CaoOcrcPopulationScopeV1, CaoOcrcRoundInputV1, CaoOcrcStochasticAssumptionStatusV1,
    LabelConsumptionEligibilityReportV1,
    LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1, advance_ca_ocrc_round_v1,
    compute_ca_ocrc_stochastic_risk_bound_v1, create_ca_ocrc_engine_v1,
    optimize_ca_ocrc_stochastic_proxy_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn policy(
    capacity: usize,
    proxy_max: u64,
    proxy_survival: Vec<f64>,
    true_delay_survival: Vec<f64>,
) -> CaoOcrcPolicyV1 {
    CaoOcrcPolicyV1 {
        schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
        policy_id: "ca-ocrc-iid-stochastic-reference-v1".into(),
        revision: 1,
        target_risk: 0.10,
        learning_rate: 0.05,
        score_upper_bound: 100.0,
        initial_threshold: 0.5,
        tracking_capacity: capacity,
        maximum_feedback_delay_rounds: (true_delay_survival.len() - 2) as u64,
        maximum_proxy_delay_rounds: proxy_max,
        delay_assumption: CaoOcrcDelayAssumptionV1::IidStochastic,
        feedback_delay_survival: Some(true_delay_survival),
        feedback_delay_law_manifest_sha256: Some(digest('9')),
        population_scope: CaoOcrcPopulationScopeV1::QualifiedMeasuredStreamOnly,
        proxy_survival,
    }
}

fn engine(policy: CaoOcrcPolicyV1) -> CaoOcrcEngineV1 {
    create_ca_ocrc_engine_v1("stochastic-reference-engine", "fab-clock", policy).unwrap()
}

fn round(index: u64, threshold: f64, proxy_draw: f64) -> CaoOcrcRoundInputV1 {
    CaoOcrcRoundInputV1 {
        schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
        round_index: index,
        sample_id: format!("sample-{index}"),
        measurement_event_id: format!("event-{index}"),
        event_time_unix_seconds: index as i64 * 10,
        prediction_at_unix_seconds: index as i64 * 10 + 1,
        clock_domain_id: "fab-clock".into(),
        global_model_artifact_sha256: digest('a'),
        interval_policy_sha256: digest('b'),
        upstream_prediction_receipt_sha256: digest('c'),
        threshold_used: threshold,
        score_location: 10.0,
        trivial_set: None,
        interval_lower: Some(10.0 - threshold),
        interval_upper: Some(10.0 + threshold),
        proxy_uniform_draw: Some(proxy_draw),
    }
}

fn independent_stochastic_bound(
    policy: &CaoOcrcPolicyV1,
    horizon: u64,
    proxy_survival: &[f64],
    proxy_max: usize,
) -> f64 {
    let g = policy.feedback_delay_survival.as_ref().unwrap();
    let q = 1.0 - g[proxy_max + 1];
    let mut ell = 0.0;
    let mut mean_delay = 0.0;
    let mut q_fg = 0.0;
    for age in 0..=proxy_max {
        let mass = g[age] - g[age + 1];
        mean_delay += age as f64 * mass / q;
        q_fg += mass / (q * q * proxy_survival[age]);
        if age > 0 {
            ell += proxy_survival[age] * g[age];
        }
    }
    let k = policy.tracking_capacity as f64;
    let alpha_bar = policy.target_risk.max(1.0 - policy.target_risk);
    let r_fg = alpha_bar
        * (mean_delay + 2.0
            + k * alpha_bar.powi(2) / (2.0 * (1.0 - alpha_bar)) * q_fg);
    let capacity_term = if proxy_max >= policy.tracking_capacity {
        let x = ell / k;
        let h = x - 1.0 - x.ln();
        (-k * h).exp()
    } else {
        0.0
    };
    policy.score_upper_bound / (policy.learning_rate * horizon as f64)
        + r_fg / horizon as f64
        + capacity_term
}

fn independent_kappa(g: &[f64], proxy_max: usize, age: usize) -> f64 {
    let mut min_over_start = f64::INFINITY;
    for start in 1..=age {
        let mut max_over_end: f64 = 0.0;
        for end in age..=proxy_max {
            let numerator = g[start] - g[end + 1];
            let denominator: f64 = (start..=end).map(|i| g[i]).sum();
            max_over_end = max_over_end.max(numerator / denominator);
        }
        min_over_start = min_over_start.min(max_over_end);
    }
    min_over_start
}

fn feedback(origin: u64, delivery: u64, value: f64, label_id: &str) -> CaoOcrcFeedbackInputV1 {
    let payload = digest('d');
    let event_id = format!("event-{origin}");
    let mut report = LabelConsumptionEligibilityReportV1 {
        schema: "luminous.semi-bench.label-consumption-report.v1".into(),
        label_id: label_id.into(),
        label_revision: 1,
        measurement_event_id: event_id.clone(),
        calibration_certificate_id: "cert-stochastic-001".into(),
        frontier: LabelEvidenceFrontierV1::CurrentAtK1,
        eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        downstream_consumption_allowed: true,
        reasons: Vec::new(),
        label_sha256: payload.clone(),
        measurement_sha256: digest('e'),
        qualified_corrected_value: value,
        qualified_standard_uncertainty_upper_bound: Some(0.1),
        qualification_report_sha256: digest('f'),
        requalification_report_sha256: digest('1'),
        requalification_frontier_sha256: digest('2'),
        source_authenticity_proven: false,
        training_snapshot_mutated: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    report.report_sha256 =
        format!("{:x}", sha2::Sha256::digest(serde_json::to_vec(&canonical).unwrap()));

    CaoOcrcFeedbackInputV1 {
        schema: CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
        delivery_round: delivery,
        originating_round: origin,
        sample_id: format!("sample-{origin}"),
        label_id: label_id.into(),
        label_revision: 1,
        measurement_event_id: event_id,
        label_payload_sha256: payload,
        corrected_measurement_value: value,
        label_available_at_unix_seconds: origin as i64 * 10 + 2,
        eligibility_report_available_at_unix_seconds: delivery as i64 * 10 + 2,
        delivered_at_unix_seconds: delivery as i64 * 10 + 3,
        clock_domain_id: "fab-clock".into(),
        idempotency_key: format!("{label_id}:1"),
        eligibility_report: report,
    }
}

#[test]
fn iid_feedback_uses_qg_times_proxy_survival_in_the_ipw_weight() {
    // True delay PMF is [0.2, 0.5, 0.3], so G=[1,0.8,0.3,0].
    // Proxy horizon is one, q_G=P(d<=1)=0.7. F=[1,0.5].
    let mut controller = engine(policy(2, 1, vec![1.0, 0.5], vec![1.0, 0.8, 0.3, 0.0]));
    let first_threshold = controller.threshold;
    let first = advance_ca_ocrc_round_v1(
        &mut controller,
        &round(1, first_threshold, 0.75), // proxy delay 1
        &[],
    )
    .unwrap();
    assert_eq!(first.selected_proxy_delay_rounds, 1);
    assert!((first.q_g - 0.7).abs() < 1.0e-12);

    let second_threshold = controller.threshold;
    let second = advance_ca_ocrc_round_v1(
        &mut controller,
        &round(2, second_threshold, 0.1), // proxy delay 0
        &[feedback(1, 2, 11.0, "label-1")], // actual delay 1, outside interval
    )
    .unwrap();

    let receipt = &second.received_feedback_receipts[0];
    assert_eq!(receipt.actual_delay_rounds, Some(1));
    assert_eq!(receipt.proxy_survival_probability, Some(0.5));
    assert_eq!(receipt.feedback_observation_probability, Some(0.7));
    assert!((receipt.inverse_probability_weight.unwrap() - 1.0 / (0.7 * 0.5)).abs() < 1.0e-12);
    let expected = 0.5 + 0.05 * (1.0 - 0.10) / (0.7 * 0.5);
    assert!((controller.threshold - expected).abs() < 1.0e-12);
}

#[test]
fn outcomes_beyond_proxy_horizon_are_untracked_but_within_true_support() {
    let mut controller = engine(policy(2, 1, vec![1.0, 0.5], vec![1.0, 0.8, 0.3, 0.0]));
    let first_threshold = controller.threshold;
    advance_ca_ocrc_round_v1(
        &mut controller,
        &round(1, first_threshold, 0.75), // proxy expiry at round 2
        &[],
    )
    .unwrap();
    let second_threshold = controller.threshold;
    advance_ca_ocrc_round_v1(
        &mut controller,
        &round(2, second_threshold, 0.75),
        &[],
    )
    .unwrap();
    let third_threshold = controller.threshold;
    let third = advance_ca_ocrc_round_v1(
        &mut controller,
        &round(3, third_threshold, 0.1),
        &[feedback(1, 3, 11.0, "late-but-in-support")], // actual delay 2 <= d_max, > dbar_max
    )
    .unwrap();

    assert_eq!(
        third.received_feedback_receipts[0].disposition,
        symthaea_semiconductor::benchmark::CaoOcrcFeedbackDispositionV1::AlreadyExpired
    );
    assert_eq!(controller.guarantee_status, symthaea_semiconductor::benchmark::CaoOcrcGuaranteeStatusV1::ConditionalOnDeclaredPaperAssumptions);
}

#[test]
fn stochastic_theorem_bound_matches_independent_scalar_formula() {
    let policy = policy(2, 1, vec![1.0, 0.5], vec![1.0, 0.8, 0.3, 0.0]);
    let bound = compute_ca_ocrc_stochastic_risk_bound_v1(&policy, 100).unwrap();
    let q = 0.7;
    let alpha_bar: f64 = 0.9;
    let ell = 0.5 * 0.8;
    let expected_delay = 0.5 / q;
    let q_fg = 0.2 / (q * q * 1.0) + 0.5 / (q * q * 0.5);
    let range = alpha_bar
        * (expected_delay + 2.0 + (2.0 * alpha_bar * alpha_bar / (2.0 * (1.0 - alpha_bar))) * q_fg);
    let expected_bound = 100.0 / (0.05 * 100.0) + range / 100.0;
    assert!((bound.q_g - q).abs() < 1.0e-12);
    assert!((bound.stochastic_tracking_load - ell).abs() < 1.0e-12);
    assert!((bound.expected_conditional_delay - expected_delay).abs() < 1.0e-12);
    assert!((bound.q_fg - q_fg).abs() < 1.0e-12);
    assert!((bound.range_term_r_fg - range).abs() < 1.0e-12);
    assert_eq!(bound.capacity_saturation_term, 0.0); // dbar_max < K
    assert!((bound.bound - expected_bound).abs() < 1.0e-10);
    assert_eq!(
        bound.theorem_assumption_status,
        CaoOcrcStochasticAssumptionStatusV1::DeclaredByCallerNotIndependentlyVerified
    );
    assert_eq!(bound.feedback_delay_law_manifest_sha256, digest('9'));
}

#[test]
fn optimizer_returns_monotone_survival_and_supported_proxy_horizon() {
    let base = policy(1, 2, vec![1.0, 1.0, 1.0], vec![1.0, 0.8, 0.3, 0.0]);
    let result = optimize_ca_ocrc_stochastic_proxy_v1(&base, 100).unwrap();
    assert!(result.q_g > 0.0);
    assert!(result.selected_proxy_delay_rounds <= result.maximum_feedback_delay_rounds);
    assert_eq!(result.proxy_survival[0], 1.0);
    assert!(result.proxy_survival.iter().all(|v| v.is_finite() && *v > 0.0 && *v <= 1.0));
    assert!(result.proxy_survival.windows(2).all(|w| w[0] >= w[1]));
    assert!(result.stochastic_tracking_load <= result.capacity as f64 + 1.0e-12);
    assert!(result.risk_bound.is_finite());
}

#[test]
fn iid_delay_profile_requires_a_content_addressed_delay_law_manifest() {
    let mut policy = policy(2, 1, vec![1.0, 0.5], vec![1.0, 0.8, 0.3, 0.0]);
    policy.feedback_delay_law_manifest_sha256 = None;
    assert!(
        create_ca_ocrc_engine_v1("missing-g-manifest", "fab-clock", policy).is_err()
    );
}

#[test]
fn stochastic_optimizer_is_no_worse_than_independent_dense_theta_grid() {
    let base = policy(1, 2, vec![1.0, 1.0, 1.0], vec![1.0, 0.8, 0.3, 0.0]);
    let optimized = optimize_ca_ocrc_stochastic_proxy_v1(&base, 100).unwrap();
    let g = base.feedback_delay_survival.as_ref().unwrap();
    let mut grid_best = f64::INFINITY;

    for proxy_max in 0..=base.maximum_feedback_delay_rounds as usize {
        if g[proxy_max] - g[proxy_max + 1] <= 0.0 {
            continue;
        }
        if proxy_max < base.tracking_capacity {
            let candidate = vec![1.0; proxy_max + 1];
            grid_best = grid_best.min(independent_stochastic_bound(
                &base,
                100,
                &candidate,
                proxy_max,
            ));
            continue;
        }

        let kappa: Vec<f64> = (0..=proxy_max)
            .map(|age| {
                if age == 0 {
                    0.0
                } else {
                    independent_kappa(g, proxy_max, age)
                }
            })
            .collect();
        let theta_max = (1..=proxy_max)
            .map(|age| 1.0 / kappa[age].sqrt())
            .fold(0.0_f64, f64::max);
        for step in 1..=20_000 {
            let theta = theta_max * step as f64 / 20_000.0;
            let mut candidate = vec![1.0];
            candidate.extend((1..=proxy_max).map(|age| (theta * kappa[age].sqrt()).min(1.0)));
            if candidate.iter().any(|value| *value <= 0.0) {
                continue;
            }
            let bound = independent_stochastic_bound(&base, 100, &candidate, proxy_max);
            grid_best = grid_best.min(bound);
        }
    }

    assert!(optimized.risk_bound <= grid_best + 2.0e-3);
}

#[test]
fn stochastic_mode_rejects_zero_qg_and_arbitrary_bound_calculator() {
    let zero_q = policy(1, 0, vec![1.0], vec![1.0, 1.0, 0.5, 0.0]);
    assert!(create_ca_ocrc_engine_v1("bad-q", "fab-clock", zero_q).is_err());

    let stochastic = policy(2, 1, vec![1.0, 0.5], vec![1.0, 0.8, 0.3, 0.0]);
    assert!(symthaea_semiconductor::benchmark::compute_ca_ocrc_risk_bound_v1(&stochastic, 100).is_err());
}
