#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use symthaea_semiconductor::benchmark::{
    CA_OCRC_FEEDBACK_SCHEMA_V1, CA_OCRC_POLICY_SCHEMA_V1, CA_OCRC_PREDICTION_SCHEMA_V1,
    CA_OCRC_ROUND_INPUT_SCHEMA_V1, CaOcrcEngineV1, CaOcrcFeedbackDispositionV1,
    CaOcrcLabelFeedbackV1, CaOcrcPolicyV1, CaOcrcPredictionInputV1, CaOcrcPredictionSetV1,
    CaOcrcRoundInputV1, CaOcrcSetKindV1, LabelConsumptionEligibilityReportV1,
    LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1, create_ca_ocrc_engine_v1,
    step_ca_ocrc_round_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn sha256_json<T: Serialize + ?Sized>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn policy() -> CaOcrcPolicyV1 {
    CaOcrcPolicyV1 {
        schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
        policy_id: "ca-ocrc-independent-oracle.v1".into(),
        revision: 1,
        target_risk_alpha: 0.10,
        learning_rate_eta: 0.05,
        bounded_score_upper: 1.0,
        maximum_actual_delay_steps: 2,
        tracking_capacity: 2,
        // dbar=-1: 0.25, dbar=0: 0.25, dbar=1: 0, dbar=2: 0.5.
        // Thus F(0)=0.75, F(1)=F(2)=0.5, ell_F=1 <= K=2.
        proxy_delay_pmf: vec![0.25, 0.25, 0.0, 0.5],
        actual_delay_contract_sha256: digest('f'),
        bounded_actual_delay_assumption_declared: true,
    }
}

fn engine() -> CaOcrcEngineV1 {
    create_ca_ocrc_engine_v1(
        "ca-ocrc-reference-engine",
        "fab-clock",
        &digest('a'),
        &digest('b'),
        0.5,
        policy(),
    )
    .unwrap()
}

fn prediction(round: u64, sample: &str, draw: f64) -> CaOcrcPredictionInputV1 {
    let time = 1_800_000_010 + round as i64 * 60;
    CaOcrcPredictionInputV1 {
        schema: CA_OCRC_PREDICTION_SCHEMA_V1.into(),
        round,
        sample_id: sample.into(),
        measurement_event_id: format!("event-{sample}"),
        event_time_unix_seconds: time - 10,
        prediction_at_unix_seconds: time - 5,
        round_time_unix_seconds: time,
        clock_domain_id: "fab-clock".into(),
        global_model_artifact_sha256: digest('a'),
        interval_method_sha256: digest('b'),
        threshold_used: 0.5,
        prediction_set: CaOcrcPredictionSetV1 {
            kind: CaOcrcSetKindV1::Interval,
            lower: Some(90.0),
            upper: Some(110.0),
            measurand_id: "cd".into(),
            unit: "nm".into(),
        },
        physical_measurement_selected: true,
        proxy_uniform_draw: Some(draw),
    }
}

fn feedback(round: u64, sample: &str, label: &str) -> CaOcrcLabelFeedbackV1 {
    let payload = digest('c');
    let time = 1_800_000_010 + round as i64 * 60;
    let mut report = LabelConsumptionEligibilityReportV1 {
        schema: "luminous.semi-bench.label-consumption-report.v1".into(),
        label_id: label.into(),
        label_revision: 1,
        measurement_event_id: format!("event-{sample}"),
        calibration_certificate_id: "cal-cert-reference".into(),
        frontier: LabelEvidenceFrontierV1::CurrentAtK1,
        eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        downstream_consumption_allowed: true,
        reasons: Vec::new(),
        label_sha256: payload.clone(),
        measurement_sha256: digest('d'),
        qualified_corrected_value: 100.0,
        qualified_standard_uncertainty_upper_bound: Some(0.1),
        qualification_report_sha256: digest('e'),
        requalification_report_sha256: digest('f'),
        requalification_frontier_sha256: digest('0'),
        source_authenticity_proven: false,
        training_snapshot_mutated: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    report.report_sha256 = sha256_json(&canonical);
    CaOcrcLabelFeedbackV1 {
        schema: CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
        feedback_round: round,
        sample_id: sample.into(),
        label_id: label.into(),
        label_revision: 1,
        measurement_event_id: format!("event-{sample}"),
        measurand_id: "cd".into(),
        unit: "nm".into(),
        label_payload_sha256: payload,
        label_value: 100.0,
        label_available_at_unix_seconds: time,
        eligibility_report_available_at_unix_seconds: time,
        delivered_at_unix_seconds: time,
        clock_domain_id: "fab-clock".into(),
        eligibility_report: report,
    }
}

fn round(
    prediction: CaOcrcPredictionInputV1,
    feedbacks: Vec<CaOcrcLabelFeedbackV1>,
) -> CaOcrcRoundInputV1 {
    CaOcrcRoundInputV1 {
        schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
        prediction,
        feedbacks,
    }
}

/// Independent sampler: the proxy PMF is accumulated directly rather than
/// calling the production proxy-delay map.
fn oracle_proxy_delay(pmf: &[f64], uniform: f64) -> i64 {
    let mut cumulative = 0.0;
    for (index, probability) in pmf.iter().enumerate() {
        cumulative += probability;
        if uniform < cumulative {
            return index as i64 - 1;
        }
    }
    pmf.len() as i64 - 2
}

fn oracle_survival(pmf: &[f64], actual_delay: usize) -> f64 {
    pmf[actual_delay + 1..].iter().sum()
}

/// Independent scalar CA-OCRC state machine for this frozen deterministic
/// stream. It implements the paper's strict expiry and IPW equation directly.
fn reference_replay() -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let p = policy();
    let mut threshold = 0.5;
    let mut tracked: BTreeMap<u64, i64> = BTreeMap::new();
    let mut thresholds = Vec::new();
    let mut occupancies = Vec::new();
    let mut update_counts = Vec::new();

    let streams = [
        (0_u64, "s0", 0.9_f64, Vec::<(&str, &str)>::new()),
        (1_u64, "s1", 0.3_f64, vec![("s1", "l1")]),
        (2_u64, "s2", 0.9_f64, vec![("s0", "l0")]),
        (3_u64, "s3", 0.1_f64, vec![("s2", "l2")]),
    ];

    for (round_index, _sample, draw, outcomes) in streams {
        // 1. Expire only when i + dbar_i < t; equality remains tracked.
        let expired: Vec<u64> = tracked
            .iter()
            .filter(|(origin, proxy)| **origin + (**proxy as u64) < round_index)
            .map(|(origin, _)| *origin)
            .collect();
        for origin in expired {
            tracked.remove(&origin);
        }

        // 2. Independently sample and admit the current nontrivial round.
        let proxy = oracle_proxy_delay(&p.proxy_delay_pmf, draw);
        if proxy >= 0 && tracked.len() < p.tracking_capacity {
            tracked.insert(round_index, proxy);
        }

        // 3. Receive feedback after admission; compute inverse-survival terms.
        let mut update_sum = 0.0;
        let mut accepted = 0_usize;
        for (sample, _label) in outcomes {
            let origin = match sample {
                "s0" => 0_u64,
                "s1" => 1_u64,
                "s2" => 2_u64,
                _ => unreachable!(),
            };
            if let Some(proxy_delay) = tracked.get(&origin).copied() {
                let actual_delay = (round_index - origin) as usize;
                assert!(actual_delay as i64 <= proxy_delay);
                let loss = 0.0; // y=100 lies inside the frozen [90,110] interval.
                update_sum += (loss - p.target_risk_alpha)
                    / oracle_survival(&p.proxy_delay_pmf, actual_delay);
                tracked.remove(&origin);
                accepted += 1;
            }
        }

        // 4. No trivial empty/full set in this trace; phi_t=0.
        threshold += p.learning_rate_eta * update_sum;
        thresholds.push(threshold);
        occupancies.push(tracked.len());
        update_counts.push(accepted);
    }

    (thresholds, occupancies, update_counts)
}

#[test]
fn production_round_engine_matches_independent_ca_ocrc_state_machine() {
    let mut engine = engine();
    let mut rounds = [
        round(prediction(0, "s0", 0.9), Vec::new()),
        round(prediction(1, "s1", 0.3), vec![feedback(1, "s1", "l1")]),
        round(prediction(2, "s2", 0.9), vec![feedback(2, "s0", "l0")]),
        round(prediction(3, "s3", 0.1), vec![feedback(3, "s2", "l2")]),
    ];

    let (expected_thresholds, expected_occupancies, expected_updates) = reference_replay();
    let mut actual_thresholds = Vec::new();
    let mut actual_occupancies = Vec::new();
    let mut actual_updates = Vec::new();

    for input in &mut rounds {
        // CA-OCRC freezes the threshold in each prediction receipt. The
        // next prediction must use the threshold produced by the prior round.
        input.prediction.threshold_used = engine.state.threshold;
        let report = step_ca_ocrc_round_v1(&mut engine, input).unwrap();
        actual_thresholds.push(report.threshold_after);
        actual_occupancies.push(report.tracking_occupancy_after);
        actual_updates.push(
            report
                .feedback_results
                .iter()
                .filter(|result| result.disposition == CaOcrcFeedbackDispositionV1::Updated)
                .count(),
        );
    }

    for (actual, expected) in actual_thresholds.iter().zip(&expected_thresholds) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    assert_eq!(actual_occupancies, expected_occupancies);
    assert_eq!(actual_updates, expected_updates);
    assert!(engine.state.theorem_assumptions_intact);
}

#[test]
fn paper_survival_vector_and_load_match_independent_pmf_calculation() {
    let p = policy();
    let expected = [
        p.proxy_delay_pmf[1..].iter().sum::<f64>(),
        p.proxy_delay_pmf[2..].iter().sum::<f64>(),
        p.proxy_delay_pmf[3..].iter().sum::<f64>(),
    ];
    let loaded = symthaea_semiconductor::benchmark::proxy_survival_vector(&p).unwrap();
    assert_eq!(loaded, expected);
    assert!((loaded[1] + loaded[2] - 1.0).abs() < 1.0e-12);
}

#[test]
fn optimal_bernoulli_proxy_is_checked_against_independent_grid_search() {
    let (pmf, p_star, objective) =
        symthaea_semiconductor::benchmark::optimal_arbitrary_delay_proxy_policy_v1(3, 2, 500)
            .unwrap();
    let mut best_grid = f64::INFINITY;
    let mut best_p = 0.0;
    for step in 1..=200_000 {
        let p = step as f64 / 200_000.0 * (2.0 / 3.0);
        let ratio = 3.0 * p / 2.0;
        let h = if ratio == 0.0 {
            f64::INFINITY
        } else {
            ratio - 1.0 - ratio.ln()
        };
        let value = 3.0 / (p * 500.0) + (-2.0 * h).exp();
        if value < best_grid {
            best_grid = value;
            best_p = p;
        }
    }
    assert!((p_star - best_p).abs() < 1.0e-4);
    assert!((objective - best_grid).abs() < 1.0e-7);
    assert!((pmf[0] + pmf[4] - 1.0).abs() < 1.0e-12);
    assert!((pmf[4] - p_star).abs() < 1.0e-12);
}
