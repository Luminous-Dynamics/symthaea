//! Deterministic smoke benchmarks for calibrated residual diagnostics and fusion.
//!
//! These tests intentionally live outside the production module so the benchmark harness stays
//! separate from the score API. They compare methods at matched clean-calibration budgets, print
//! rates with Wilson intervals, and encode a few qualitative smoke assertions. They are not a
//! substitute for the preregistered, held-out SOTA comparison documented in the research note.

use symthaea_physics_bridge::adequacy_calibration::{
    ConformalAdequacyCalibrator, ResidualDiagnosticScores, minimum_component_p_fusion_score,
    standardized_residual_diagnostics, standardized_rms_residual, sum_log_component_p_fusion_score,
};
use symthaea_physics_bridge::feature_shift::{
    FeatureShiftScorer, ShiftTriageDecision, classify_shift_evidence,
};

#[derive(Debug)]
struct DeterministicNormalRng {
    state: u64,
    spare: Option<f64>,
}

impl DeterministicNormalRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_uniform(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let bits = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((bits >> 11) as f64 / ((1_u64 << 53) as f64)).max(1e-15)
    }

    fn next_normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.next_uniform().ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.next_uniform();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    fn normal_batch(&mut self, len: usize) -> Vec<f64> {
        (0..len).map(|_| self.next_normal()).collect()
    }

    fn student_t4_batch(&mut self, len: usize) -> Vec<f64> {
        (0..len)
            .map(|_| {
                let numerator = self.next_normal();
                let chi_squared = (0..4)
                    .map(|_| {
                        let value = self.next_normal();
                        value * value
                    })
                    .sum::<f64>();
                numerator * (2.0 / chi_squared).sqrt()
            })
            .collect()
    }

    fn ar1_batch(&mut self, len: usize, rho: f64) -> Vec<f64> {
        let mut values = vec![0.0; len];
        if len == 0 {
            return values;
        }
        values[0] = self.next_normal();
        let innovation_scale = (1.0 - rho * rho).sqrt();
        for index in 1..len {
            values[index] = rho * values[index - 1] + innovation_scale * self.next_normal();
        }
        values
    }
}

fn score_residual_batch(
    residuals: &[f64],
    zeros: &[f64],
    ones: &[f64],
) -> ResidualDiagnosticScores {
    standardized_residual_diagnostics(residuals, zeros, ones)
        .expect("synthetic batch uses finite, matching residual inputs")
}

fn calibrated_alarm(calibrator: &ConformalAdequacyCalibrator, score: f64) -> bool {
    calibrator
        .audit_p_value(score, true)
        .expect("synthetic audit is independent and finite")
        <= 0.05
}

fn wilson_interval(successes: usize, trials: usize) -> (f64, f64) {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let denominator = 1.0 + z * z / n;
    let center = (p + z * z / (2.0 * n)) / denominator;
    let half_width = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt()) / denominator;
    (center - half_width, center + half_width)
}

fn correlated_feature_batch(
    rng: &mut DeterministicNormalRng,
    len: usize,
    rho: f64,
) -> Vec<Vec<f64>> {
    let innovation_scale = (1.0 - rho * rho).sqrt();
    (0..len)
        .map(|_| {
            let first = rng.next_normal();
            let independent = rng.next_normal();
            vec![first, rho * first + innovation_scale * independent]
        })
        .collect()
}

/// Covariate-shift counterexample: the conditional predictor can remain exact (zero residual)
/// while the input feature correlation moves well outside its fixed reference distribution.
#[test]
fn feature_shift_detects_input_drift_when_residual_monitor_is_silent() {
    const BATCH_SIZE: usize = 64;
    const REFERENCE_ROWS: usize = 512;
    const CALIBRATION_BATCHES: usize = 199;
    const AUDIT_BATCHES: usize = 1_000;
    const ALPHA: f64 = 0.05;

    let mut rng = DeterministicNormalRng::new(0x6DA2_91F4_B783_C05E);
    let reference = correlated_feature_batch(&mut rng, REFERENCE_ROWS, 0.0);
    let scorer = FeatureShiftScorer::fit(&reference, 1e-8)
        .expect("the reference has finite nonconstant two-feature rows");

    let calibration_scores = (0..CALIBRATION_BATCHES)
        .map(|_| {
            scorer
                .score(&correlated_feature_batch(&mut rng, BATCH_SIZE, 0.0))
                .expect("clean calibration batch matches the fixed feature profile")
                .aggregate
        })
        .collect::<Vec<_>>();
    let feature_calibrator = ConformalAdequacyCalibrator::new(&calibration_scores, ALPHA).unwrap();
    let residual_calibration = vec![0.0; CALIBRATION_BATCHES];
    let residual_calibrator =
        ConformalAdequacyCalibrator::new(&residual_calibration, ALPHA).unwrap();

    let mut clean_feature_alarms = 0_usize;
    let mut shifted_feature_alarms = 0_usize;
    let mut clean_residual_alarms = 0_usize;
    let mut shifted_residual_alarms = 0_usize;

    let mut feature_only_triage = 0_usize;
    for _ in 0..AUDIT_BATCHES {
        let clean = correlated_feature_batch(&mut rng, BATCH_SIZE, 0.0);
        let shifted = correlated_feature_batch(&mut rng, BATCH_SIZE, 0.8);

        let clean_feature_result = scorer.assess(&feature_calibrator, &clean, true).unwrap();
        let shifted_feature_result = scorer.assess(&feature_calibrator, &shifted, true).unwrap();
        clean_feature_alarms += (clean_feature_result.p_value <= ALPHA) as usize;
        shifted_feature_alarms += (shifted_feature_result.p_value <= ALPHA) as usize;

        // The predictor observes both input dimensions and remains exact in either regime.
        // Residual-only monitoring therefore gets zero in both the clean and shifted batches.
        let zero_residuals = [0.0_f64; 8];
        let unit_scales = [1.0_f64; 8];
        let clean_residual_score =
            standardized_rms_residual(&zero_residuals, &zero_residuals, &unit_scales).unwrap();
        let shifted_residual_score =
            standardized_rms_residual(&zero_residuals, &zero_residuals, &unit_scales).unwrap();
        let clean_model_result = residual_calibrator
            .assess_audit_score(clean_residual_score, true)
            .unwrap();
        let shifted_model_result = residual_calibrator
            .assess_audit_score(shifted_residual_score, true)
            .unwrap();
        clean_residual_alarms += (clean_model_result.p_value <= ALPHA) as usize;
        shifted_residual_alarms += (shifted_model_result.p_value <= ALPHA) as usize;

        if classify_shift_evidence(
            shifted_model_result.assessment,
            shifted_feature_result.assessment,
        ) == ShiftTriageDecision::FeatureDistributionShiftOnly
        {
            feature_only_triage += 1;
        }
    }

    let clean_rate = clean_feature_alarms as f64 / AUDIT_BATCHES as f64;
    let shifted_rate = shifted_feature_alarms as f64 / AUDIT_BATCHES as f64;
    let (clean_lower, clean_upper) = wilson_interval(clean_feature_alarms, AUDIT_BATCHES);
    let (shift_lower, shift_upper) = wilson_interval(shifted_feature_alarms, AUDIT_BATCHES);
    println!(
        "covariate-correlation shift: clean feature alarms={clean_rate:.4} \
         [{clean_lower:.4},{clean_upper:.4}], shifted alarms={shifted_rate:.4} \
         [{shift_lower:.4},{shift_upper:.4}], residual alarms clean={clean_residual_alarms}, \
         shifted={shifted_residual_alarms}"
    );

    assert!(
        clean_rate <= 0.09,
        "clean feature-shift alarm rate was {clean_rate:.4}"
    );
    assert!(
        shifted_rate >= 0.80,
        "feature-shift detection was too weak: {shifted_rate:.4}"
    );
    assert_eq!(clean_residual_alarms, 0);
    assert_eq!(shifted_residual_alarms, 0);
    assert_eq!(feature_only_triage, shifted_feature_alarms);
}

/// Verify calibration with two null families beyond a single Gaussian/noise regime:
/// matched heavy-tailed Student-t residuals, and heteroscedastic raw residuals divided by the
/// correct declared predictive scale. Each replication has its own calibration reference and
/// one independent audit, so the empirical false-alarm rate is not conditional on one lucky
/// fixed reference sample.
#[test]
fn matched_heavy_tail_and_standardized_heteroscedastic_null_smoke() {
    const BATCH_SIZE: usize = 64;
    const CALIBRATION_BATCHES: usize = 99;
    const REPLICATIONS: usize = 1_200;
    const ALPHA: f64 = 0.05;

    let mut rng = DeterministicNormalRng::new(0xA42D_7C19_5E83_B601);
    let zeros = vec![0.0; BATCH_SIZE];
    let ones = vec![1.0; BATCH_SIZE];
    let mut heavy_tail_alarms = [0_usize; 2];
    let mut heteroscedastic_alarms = [0_usize; 2];

    for _ in 0..REPLICATIONS {
        let heavy_calibration = (0..CALIBRATION_BATCHES)
            .map(|_| score_residual_batch(&rng.student_t4_batch(BATCH_SIZE), &zeros, &ones))
            .collect::<Vec<_>>();
        let heavy_rms = heavy_calibration
            .iter()
            .map(|score| score.rms_standardized_residual)
            .collect::<Vec<_>>();
        let heavy_max = heavy_calibration
            .iter()
            .map(|score| score.maximum())
            .collect::<Vec<_>>();
        let heavy_rms_calibrator = ConformalAdequacyCalibrator::new(&heavy_rms, ALPHA).unwrap();
        let heavy_max_calibrator = ConformalAdequacyCalibrator::new(&heavy_max, ALPHA).unwrap();
        let heavy_audit = score_residual_batch(&rng.student_t4_batch(BATCH_SIZE), &zeros, &ones);
        heavy_tail_alarms[0] +=
            calibrated_alarm(&heavy_rms_calibrator, heavy_audit.rms_standardized_residual) as usize;
        heavy_tail_alarms[1] +=
            calibrated_alarm(&heavy_max_calibrator, heavy_audit.maximum()) as usize;

        let gaussian_calibration = (0..CALIBRATION_BATCHES)
            .map(|_| score_residual_batch(&rng.normal_batch(BATCH_SIZE), &zeros, &ones))
            .collect::<Vec<_>>();
        let gaussian_rms = gaussian_calibration
            .iter()
            .map(|score| score.rms_standardized_residual)
            .collect::<Vec<_>>();
        let gaussian_max = gaussian_calibration
            .iter()
            .map(|score| score.maximum())
            .collect::<Vec<_>>();
        let gaussian_rms_calibrator =
            ConformalAdequacyCalibrator::new(&gaussian_rms, ALPHA).unwrap();
        let gaussian_max_calibrator =
            ConformalAdequacyCalibrator::new(&gaussian_max, ALPHA).unwrap();

        let latent = rng.normal_batch(BATCH_SIZE);
        let scales = (0..BATCH_SIZE)
            .map(|index| 0.5 + (index % 8) as f64 * 0.25)
            .collect::<Vec<_>>();
        let heteroscedastic_observed = latent
            .iter()
            .zip(scales.iter())
            .map(|(noise, scale)| noise * scale)
            .collect::<Vec<_>>();
        let heteroscedastic =
            standardized_residual_diagnostics(&heteroscedastic_observed, &zeros, &scales)
                .expect("known predictive scales standardize the null residuals");
        heteroscedastic_alarms[0] += calibrated_alarm(
            &gaussian_rms_calibrator,
            heteroscedastic.rms_standardized_residual,
        ) as usize;
        heteroscedastic_alarms[1] +=
            calibrated_alarm(&gaussian_max_calibrator, heteroscedastic.maximum()) as usize;
    }

    for (name, counts) in [
        ("matched Student-t(df=4) null", heavy_tail_alarms),
        ("standardized heteroscedastic null", heteroscedastic_alarms),
    ] {
        print!("{name}; ");
        for (index, method) in ["RMS", "raw-max"].iter().enumerate() {
            let (lower, upper) = wilson_interval(counts[index], REPLICATIONS);
            print!(
                "{}={:.4} [{:.4},{:.4}]{}",
                method,
                counts[index] as f64 / REPLICATIONS as f64,
                lower,
                upper,
                if index == 1 { "" } else { "; " }
            );
        }
        println!();
    }

    for counts in [heavy_tail_alarms, heteroscedastic_alarms] {
        assert!(
            counts
                .iter()
                .all(|count| (*count as f64 / REPLICATIONS as f64) <= 0.08),
            "matched clean-null false-alarm rate exceeded the smoke envelope: {counts:?}"
        );
        assert!(
            counts
                .iter()
                .all(|count| (*count as f64 / REPLICATIONS as f64) >= 0.02),
            "matched clean-null false-alarm rate was unexpectedly conservative: {counts:?}"
        );
    }
}

/// Reproduces the score-fusion pilot inside Rust with a fixed seed and equal clean-calibration
/// budget (198 batches per policy). This is a deterministic smoke benchmark, not a claim of
/// uniform power or a substitute for a large held-out evaluation.
#[test]
fn diagnostic_fusion_rust_benchmark_smoke() {
    const BATCH_SIZE: usize = 64;
    const SPLIT_BATCHES: usize = 99;
    const REPLICATIONS: usize = 1_200;
    const ALPHA: f64 = 0.05;

    let mut rng = DeterministicNormalRng::new(0x9BF1_D3A7_254E_8C61);
    let zeros = vec![0.0; BATCH_SIZE];
    let ones = vec![1.0; BATCH_SIZE];
    // Split A normalizes the four components; split B calibrates aggregate fusion scores.
    // Single-score baselines use A+B, so every policy sees 198 clean calibration batches.
    let mut alarms = [[0_usize; 4]; 4];

    for replication in 0..REPLICATIONS {
        let calibration_a = (0..SPLIT_BATCHES)
            .map(|_| score_residual_batch(&rng.normal_batch(BATCH_SIZE), &zeros, &ones))
            .collect::<Vec<_>>();
        let calibration_b = (0..SPLIT_BATCHES)
            .map(|_| score_residual_batch(&rng.normal_batch(BATCH_SIZE), &zeros, &ones))
            .collect::<Vec<_>>();
        let calibration_all = calibration_a
            .iter()
            .chain(calibration_b.iter())
            .copied()
            .collect::<Vec<_>>();

        let rms_reference = calibration_all
            .iter()
            .map(|scores| scores.rms_standardized_residual)
            .collect::<Vec<_>>();
        let max_reference = calibration_all
            .iter()
            .map(|scores| scores.maximum())
            .collect::<Vec<_>>();
        let rms_calibrator = ConformalAdequacyCalibrator::new(&rms_reference, ALPHA).unwrap();
        let max_calibrator = ConformalAdequacyCalibrator::new(&max_reference, ALPHA).unwrap();

        let component_calibrators = (0..4)
            .map(|component| {
                let scores = calibration_a
                    .iter()
                    .map(|diagnostic| diagnostic.as_array()[component])
                    .collect::<Vec<_>>();
                ConformalAdequacyCalibrator::new(&scores, ALPHA).unwrap()
            })
            .collect::<Vec<_>>();

        let min_reference = calibration_b
            .iter()
            .map(|diagnostic| {
                minimum_component_p_fusion_score(
                    &component_calibrators,
                    &diagnostic.as_array(),
                    true,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let sum_reference = calibration_b
            .iter()
            .map(|diagnostic| {
                sum_log_component_p_fusion_score(
                    &component_calibrators,
                    &diagnostic.as_array(),
                    true,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let min_calibrator = ConformalAdequacyCalibrator::new(&min_reference, ALPHA).unwrap();
        let sum_calibrator = ConformalAdequacyCalibrator::new(&sum_reference, ALPHA).unwrap();

        let clean = rng.normal_batch(BATCH_SIZE);
        let mut localized = rng.normal_batch(BATCH_SIZE);
        localized[(replication * 17) % BATCH_SIZE] += 6.0;
        let biased = rng
            .normal_batch(BATCH_SIZE)
            .into_iter()
            .map(|value| value + 0.4)
            .collect::<Vec<_>>();
        let autocorrelated = rng.ar1_batch(BATCH_SIZE, 0.55);
        let scenarios = [
            ("clean-null", clean),
            ("single +6 sigma spike", localized),
            ("+0.4 sigma mean shift", biased),
            ("AR(1) rho=0.55", autocorrelated),
        ];

        for (scenario_index, (_, residuals)) in scenarios.iter().enumerate() {
            let diagnostic = score_residual_batch(residuals, &zeros, &ones);
            let components = diagnostic.as_array();
            let min_score =
                minimum_component_p_fusion_score(&component_calibrators, &components, true)
                    .unwrap();
            let sum_score =
                sum_log_component_p_fusion_score(&component_calibrators, &components, true)
                    .unwrap();
            let decisions = [
                calibrated_alarm(&rms_calibrator, diagnostic.rms_standardized_residual),
                calibrated_alarm(&max_calibrator, diagnostic.maximum()),
                calibrated_alarm(&min_calibrator, min_score),
                calibrated_alarm(&sum_calibrator, sum_score),
            ];
            for (method_index, alarm) in decisions.into_iter().enumerate() {
                if alarm {
                    alarms[scenario_index][method_index] += 1;
                }
            }
        }
    }

    let method_names = ["RMS", "raw-max", "min-p fusion", "sum-log fusion"];
    for scenario_index in 0..4 {
        let scenario_name = [
            "clean-null",
            "single +6 sigma spike",
            "+0.4 sigma mean shift",
            "AR(1) rho=0.55",
        ][scenario_index];
        print!("scenario={scenario_name}; ");
        for method_index in 0..4 {
            let count = alarms[scenario_index][method_index];
            let rate = count as f64 / REPLICATIONS as f64;
            let (lower, upper) = wilson_interval(count, REPLICATIONS);
            print!(
                "{}={:.4} [{:.4},{:.4}]{}",
                method_names[method_index],
                rate,
                lower,
                upper,
                if method_index == 3 { "" } else { "; " }
            );
        }
        println!();
    }

    // The clean-null gate catches major calibration mistakes; it is not a proof of validity.
    assert!(
        alarms[0]
            .iter()
            .all(|count| (*count as f64 / REPLICATIONS as f64) < 0.08),
        "clean-null alarm rate exceeded the smoke-test envelope: {:?}",
        alarms[0]
    );

    let rms_bias = alarms[2][0] as f64 / REPLICATIONS as f64;
    let max_bias = alarms[2][1] as f64 / REPLICATIONS as f64;
    assert!(
        max_bias > rms_bias + 0.30,
        "raw maximum should expose this bias alternative more often: \
         rms={rms_bias:.4}, max={max_bias:.4}"
    );
    let rms_ar = alarms[3][0] as f64 / REPLICATIONS as f64;
    let max_ar = alarms[3][1] as f64 / REPLICATIONS as f64;
    assert!(
        max_ar > rms_ar + 0.50,
        "raw maximum should expose this autocorrelation alternative more often: \
         rms={rms_ar:.4}, max={max_ar:.4}"
    );
    let rms_spike = alarms[1][0] as f64 / REPLICATIONS as f64;
    let max_spike = alarms[1][1] as f64 / REPLICATIONS as f64;
    assert!(
        max_spike + 0.25 < rms_spike,
        "raw maximum trade-off on the spike alternative changed: \
         rms={rms_spike:.4}, max={max_spike:.4}"
    );
    let sum_spike = alarms[1][3] as f64 / REPLICATIONS as f64;
    assert!(
        sum_spike > rms_spike + 0.05,
        "sum-log fusion should expose this spike alternative more often: \
         rms={rms_spike:.4}, sum-log={sum_spike:.4}"
    );
}
