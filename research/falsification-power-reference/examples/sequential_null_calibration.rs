use std::f64::consts::TAU;

use falsification_power_reference::{
    FiniteReferenceConformalEProcess, MixtureAlternative, MixtureDiscrepancy,
    PredictableGaussianEProcess, PredictableGaussianMixture,
};

const ALPHA: f64 = 0.05;
const PATHS: u64 = 10_000;
const HORIZON: u64 = 50;
const MASTER_SEED: u64 = 0x5345_5155_454E_5449;

fn uniform_open(state: &mut u64) -> f64 {
    let mut value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    let bits = value.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
    (bits as f64 + 0.5) / 9_007_199_254_740_992.0
}

fn standard_normal(state: &mut u64) -> f64 {
    let first = uniform_open(state);
    let second = uniform_open(state);
    (-2.0 * first.ln()).sqrt() * (TAU * second).cos()
}

fn wilson_interval(successes: u64, trials: u64) -> (f64, f64) {
    let n = trials as f64;
    let observed = successes as f64 / n;
    let z = 1.959_963_984_540_054;
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / n;
    let center = (observed + z_squared / (2.0 * n)) / denominator;
    let radius = z
        * (observed * (1.0 - observed) / n
            + z_squared / (4.0 * n * n))
            .sqrt()
        / denominator;
    ((center - radius).max(0.0), (center + radius).min(1.0))
}

fn variance_at(step: u64) -> f64 {
    // A predictable, action-dependent precision schedule fixed in advance.
    0.65 + 0.15 * (step % 5) as f64 / 4.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut single_crossings = 0_u64;
    let mut mixture_crossings = 0_u64;
    let mut conformal_crossings = 0_u64;
    let mut leaked_crossings = 0_u64;

    for path in 0..PATHS {
        let mut rng = MASTER_SEED
            .wrapping_add(path.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .max(1);
        let path_id = format!("null-path-{path:05}");
        let mut previous_residual = 0.0_f64;
        let mut single_crossed = false;
        let mut mixture_crossed = false;
        let mut conformal_crossed = false;
        let mut leaked_crossed = false;

        let mut single =
            PredictableGaussianEProcess::new("registered-break-a", ALPHA)?;
        let mut mixture = PredictableGaussianMixture::new(
            vec![
                MixtureAlternative {
                    id: "registered-break-a".to_owned(),
                    weight: 0.5,
                },
                MixtureAlternative {
                    id: "registered-break-b".to_owned(),
                    weight: 0.5,
                },
            ],
            ALPHA,
        )?;
        let mut conformal = ConformalEProcess_PLACEHOLDER::new("simulator-null-rotation", ALPHA)?;
        let mut intentionally_leaky =
            PredictableGaussianEProcess::new("invalid-posthoc-break", ALPHA)?;

        for step in 0..HORIZON {
            let observation_id = format!("{path_id}-obs-{step:03}");
            let decision_digest = format!("frozen-null-policy-v1-{path_id}-{step:03}");
            let sigma = variance_at(step);

            // Both valid policies choose their residual predictions using only
            // prior observations, then commit before the current outcome exists.
            let history_signal = previous_residual.tanh();
            let single_discrepancy = 0.30 + 0.12 * history_signal;
            let mixture_a = 0.32 + 0.08 * history_signal;
            let mixture_b = -0.28 + 0.06 * history_signal;

            single.commit_prediction(
                observation_id.clone(),
                decision_digest.clone(),
                single_discrepancy,
                sigma,
            )?;
            mixture.commit_prediction(
                observation_id.clone(),
                decision_digest.clone(),
                sigma,
                &[
                    MixtureDiscrepancy {
                        alternative_id: "registered-break-a".to_owned(),
                        expected_discrepancy: mixture_a,
                    },
                    MixtureDiscrepancy {
                        alternative_id: "registered-break-b".to_owned(),
                        expected_discrepancy: mixture_b,
                    },
                ],
            )?;

            // A positive score is selected from history before the next
            // observation. The same score profile is applied to the observed
            // residual and five fresh simulator draws under the candidate null.
            let conformal_strength = 0.30 + 0.10 * history_signal;
            let reference_batch_id = format!("simulator-null-{path_id}-{step:03}");
            conformal.commit_score(
                observation_id.clone(),
                decision_digest.clone(),
                "positive-score-exp-v1",
                reference_batch_id.clone(),
                5,
            )?;

            // Under the simple conditional null, residual/sigma is standard
            // normal. Each sigma_t is known before the current residual.
            let standardized_observation = standard_normal(&mut rng);
            let observed_residual = sigma * standardized_observation;
            let observed_score = (conformal_strength * standardized_observation).exp();
            let mut reference_scores = Vec::with_capacity(5);
            for _ in 0..5 {
                let reference_residual = standard_normal(&mut rng);
                reference_scores.push((conformal_strength * reference_residual).exp());
            }
            let conformal_update = conformal.observe(
                &observation_id,
                &decision_digest,
                "positive-score-exp-v1",
                &reference_batch_id,
                observed_score,
                &reference_scores,
            )?;
            let single_update = single.observe(
                &observation_id,
                &decision_digest,
                observed_residual,
            )?;
            let mixture_update = mixture.observe(
                &observation_id,
                &decision_digest,
                observed_residual,
            )?;

            // Deliberately invalid: this prediction is chosen AFTER seeing the
            // outcome. The local API cannot detect out-of-band caller leakage.
            let leaked_discrepancy = 0.5 * sigma * observed_residual.signum();
            intentionally_leaky.commit_prediction(
                observation_id,
                format!("INVALID-posthoc-{path_id}-{step:03}"),
                leaked_discrepancy,
                sigma,
            )?;
            let leaked_update = intentionally_leaky.observe(
                &format!("{path_id}-obs-{step:03}"),
                &format!("INVALID-posthoc-{path_id}-{step:03}"),
                observed_residual,
            )?;

            single_crossed |= single_update.crossed_alpha;
            mixture_crossed |= mixture_update.crossed_alpha;
            conformal_crossed |= conformal_update.update.crossed_alpha;
            leaked_crossed |= leaked_update.crossed_alpha;
            previous_residual = observed_residual;
        }

        if single_crossed {
            single_crossings += 1;
        }
        if mixture_crossed {
            mixture_crossings += 1;
        }
        if conformal_crossed {
            conformal_crossings += 1;
        }
        if leaked_crossed {
            leaked_crossings += 1;
        }
    }

    let (single_lower, single_upper) = wilson_interval(single_crossings, PATHS);
    let (mixture_lower, mixture_upper) = wilson_interval(mixture_crossings, PATHS);
    let (conformal_lower, conformal_upper) = wilson_interval(conformal_crossings, PATHS);
    let (leaked_lower, leaked_upper) = wilson_interval(leaked_crossings, PATHS);

    println!(
        "{{\"seed\":{},\"paths\":{},\"horizon\":{},\"alpha\":{},\"predictable_single\":{{\"crossings\":{},\"rate\":{},\"wilson95\":[{},{}]}},\"predictable_fixed_mixture\":{{\"crossings\":{},\"rate\":{},\"wilson95\":[{},{}]}},\"predictable_finite_reference_conformal\":{{\"crossings\":{},\"rate\":{},\"wilson95\":[{},{}]}},\"invalid_posthoc_leak\":{{\"crossings\":{},\"rate\":{},\"wilson95\":[{},{}]}}}}",
        MASTER_SEED,
        PATHS,
        HORIZON,
        ALPHA,
        single_crossings,
        single_crossings as f64 / PATHS as f64,
        single_lower,
        single_upper,
        mixture_crossings,
        mixture_crossings as f64 / PATHS as f64,
        mixture_lower,
        mixture_upper,
        conformal_crossings,
        conformal_crossings as f64 / PATHS as f64,
        conformal_lower,
        conformal_upper,
        leaked_crossings,
        leaked_crossings as f64 / PATHS as f64,
        leaked_lower,
        leaked_upper,
    );

    // This is a regression sanity gate, not a formal proof or a substitute for
    // a calibrated campaign. The tolerance allows Monte Carlo variation.
    if single_upper > ALPHA + 0.02
        || mixture_upper > ALPHA + 0.02
        || conformal_upper > ALPHA + 0.02
    {
        return Err(std::io::Error::other(
            "predictable null crossing rate exceeded the calibration tolerance",
        )
        .into());
    }
    if leaked_lower < 0.50 {
        return Err(std::io::Error::other(
            "post-hoc leakage control did not show material false promotion",
        )
        .into());
    }
    Ok(())
}
