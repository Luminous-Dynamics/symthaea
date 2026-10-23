use super::*;

/// Small deterministic PRNG local to the synthetic stress test; no crate dependency or
/// platform-specific random source is involved. This is fixture machinery, not a production
/// randomness primitive.
struct CastingShiftRng {
    state: u64,
}

impl CastingShiftRng {
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn unit_interval(&mut self) -> f64 {
        // Midpoints of 2^53 buckets avoid exact zero and one.
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
}

/// Covariate-dependent exponential score. The conditional score law remains fixed under
/// source/target change; only the marginal covariate distribution shifts.
fn casting_shift_residual_score(x: f64, rng: &mut CastingShiftRng) -> f64 {
    (0.2 + 2.0 * x) * -(1.0 - rng.unit_interval()).ln()
}

#[test]
fn monte_carlo_weighting_repairs_target_coverage_but_selection_needs_its_own_shift() {
    const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
    const REPLICATES: usize = 8_000;
    const CALIBRATION_N: usize = 128;
    const ALPHA: f64 = 0.20;
    const SELECTION_THRESHOLD: f64 = 0.75;
    const TARGET_RATIO_SCALE: f64 = 2.0;
    const SELECTED_TARGET_MASS: f64 =
        1.0 - SELECTION_THRESHOLD * SELECTION_THRESHOLD;

    let mut rng = CastingShiftRng { state: SEED };
    let mut unweighted_target_covered = 0usize;
    let mut target_weighted_covered = 0usize;
    let mut selected_unweighted_covered = 0usize;
    let mut selected_target_weighted_covered = 0usize;
    let mut selected_conditioned_weighted_covered = 0usize;
    let mut selected_count = 0usize;

    let mut unweighted_refusals = 0usize;
    let mut target_weighted_refusals = 0usize;
    let mut selected_weighted_refusals = 0usize;
    let mut target_weighted_successes = 0usize;
    let mut selected_weighted_successes = 0usize;

    let mut unweighted_multiplier_sum = 0.0;
    let mut target_weighted_multiplier_sum = 0.0;
    let mut selected_weighted_multiplier_sum = 0.0;
    let mut target_weighted_ess_sum = 0.0;
    let mut selected_weighted_ess_sum = 0.0;

    for _ in 0..REPLICATES {
        let mut source_calibration = Vec::with_capacity(CALIBRATION_N);
        let mut target_weighted_calibration = Vec::with_capacity(CALIBRATION_N);
        let mut selected_weighted_calibration = Vec::with_capacity(CALIBRATION_N);

        for _ in 0..CALIBRATION_N {
            // Source X ~ Uniform(0,1); target X ~ Beta(2,1), so the exact density ratio is 2x.
            let x = rng.unit_interval();
            let score = casting_shift_residual_score(x, &mut rng);
            let observation = PredictionCalibrationSample {
                predicted_mean: 0.0,
                predictive_std_dev: 1.0,
                observed_value: score,
            };
            source_calibration.push(observation);
            target_weighted_calibration.push(WeightedPredictionCalibrationSample {
                sample: observation,
                importance_weight: TARGET_RATIO_SCALE * x,
            });

            // For the target distribution conditioned on X >= threshold, the exact ratio
            // against the uniform source is 2x / P_target(X >= threshold), and zero outside.
            let selected_weight = if x >= SELECTION_THRESHOLD {
                TARGET_RATIO_SCALE * x / SELECTED_TARGET_MASS
            } else {
                0.0
            };
            selected_weighted_calibration.push(WeightedPredictionCalibrationSample {
                sample: observation,
                importance_weight: selected_weight,
            });
        }

        let unweighted = match fit_split_conformal_multiplier(&source_calibration, ALPHA) {
            Ok(result) => {
                unweighted_multiplier_sum += result.multiplier;
                result.multiplier
            }
            Err(_) => {
                unweighted_refusals += 1;
                continue;
            }
        };

        // Inverse-CDF sampling yields the target density f_t(x)=2x.
        let target_x = rng.unit_interval().sqrt();
        let target_score = casting_shift_residual_score(target_x, &mut rng);
        if target_score <= unweighted {
            unweighted_target_covered += 1;
        }

        let target_weighted = match fit_weighted_split_conformal_multiplier(
            &target_weighted_calibration,
            TARGET_RATIO_SCALE * target_x,
            ALPHA,
        ) {
            Ok(result) => {
                target_weighted_successes += 1;
                target_weighted_multiplier_sum += result.multiplier;
                target_weighted_ess_sum += result.weighted_effective_sample_size;
                Some(result.multiplier)
            }
            Err(_) => {
                target_weighted_refusals += 1;
                None
            }
        };

        if let Some(multiplier) = target_weighted {
            if target_score <= multiplier {
                target_weighted_covered += 1;
            }
        }

        if target_x >= SELECTION_THRESHOLD {
            selected_count += 1;
            if target_score <= unweighted {
                selected_unweighted_covered += 1;
            }
            if let Some(multiplier) = target_weighted {
                if target_score <= multiplier {
                    selected_target_weighted_covered += 1;
                }
            }

            let selected_test_weight = TARGET_RATIO_SCALE * target_x / SELECTED_TARGET_MASS;
            match fit_weighted_split_conformal_multiplier(
                &selected_weighted_calibration,
                selected_test_weight,
                ALPHA,
            ) {
                Ok(result) => {
                    selected_weighted_successes += 1;
                    selected_weighted_multiplier_sum += result.multiplier;
                    selected_weighted_ess_sum += result.weighted_effective_sample_size;
                    if target_score <= result.multiplier {
                        selected_conditioned_weighted_covered += 1;
                    }
                }
                Err(_) => selected_weighted_refusals += 1,
            }
        }
    }

    // Refusals are measured and treated as failures, never silently dropped from denominators.
    assert_eq!(unweighted_refusals, 0);
    assert_eq!(target_weighted_refusals, 0);
    assert_eq!(selected_weighted_refusals, 0);
    assert_eq!(target_weighted_successes, REPLICATES);
    assert_eq!(selected_weighted_successes, selected_count);
    assert!(selected_count > REPLICATES / 3);

    let unweighted_target_coverage = unweighted_target_covered as f64 / REPLICATES as f64;
    let target_weighted_coverage =
        target_weighted_covered as f64 / target_weighted_successes as f64;
    let selected_unweighted_coverage =
        selected_unweighted_covered as f64 / selected_count as f64;
    let selected_target_weighted_coverage =
        selected_target_weighted_covered as f64 / selected_count as f64;
    let selected_conditioned_weighted_coverage =
        selected_conditioned_weighted_covered as f64 / selected_weighted_successes as f64;
    let mean_target_ess = target_weighted_ess_sum / target_weighted_successes as f64;
    let mean_selected_ess = selected_weighted_ess_sum / selected_weighted_successes as f64;

    // Broad seed-fixed Monte Carlo envelopes are regression checks, not deployment guarantees.
    assert!(
        (0.68..0.79).contains(&unweighted_target_coverage),
        "unweighted target coverage outside stress envelope: {unweighted_target_coverage}"
    );
    assert!(
        (0.77..0.85).contains(&target_weighted_coverage),
        "target-weighted coverage outside stress envelope: {target_weighted_coverage}"
    );
    assert!(
        (0.60..0.70).contains(&selected_unweighted_coverage),
        "unweighted selected coverage outside stress envelope: {selected_unweighted_coverage}"
    );
    assert!(
        (0.68..0.79).contains(&selected_target_weighted_coverage),
        "target-weighted selected coverage outside stress envelope: {selected_target_weighted_coverage}"
    );
    assert!(
        (0.77..0.87).contains(&selected_conditioned_weighted_coverage),
        "selection-conditioned coverage outside stress envelope: {selected_conditioned_weighted_coverage}"
    );
    assert!(
        target_weighted_coverage - unweighted_target_coverage > 0.04,
        "known-ratio weighting did not materially improve target-marginal coverage"
    );
    assert!(
        selected_conditioned_weighted_coverage - selected_target_weighted_coverage > 0.06,
        "selection-conditioned weights did not materially improve selected-subset coverage"
    );
    assert!(
        mean_selected_ess < mean_target_ess,
        "selected-support conditioning should expose the ESS cost"
    );

    println!(
        "CASTING-002K seed={SEED:#018x} reps={REPLICATES} calibration_n={CALIBRATION_N} alpha={ALPHA:.2} selection_x>={SELECTION_THRESHOLD:.2}\n\
         coverage target: unweighted={unweighted_target_coverage:.4}, weighted={target_weighted_coverage:.4}\n\
         coverage selected: unweighted={selected_unweighted_coverage:.4}, target-weighted={selected_target_weighted_coverage:.4}, selection-weighted={selected_conditioned_weighted_coverage:.4}\n\
         mean multiplier (interval-width proxy): unweighted={:.4}, target-weighted={:.4}, selection-weighted={:.4}\n\
         mean weighted ESS: target={mean_target_ess:.2}, selected={mean_selected_ess:.2}; refusals target={target_weighted_refusals}, selected={selected_weighted_refusals}",
        unweighted_multiplier_sum / REPLICATES as f64,
        target_weighted_multiplier_sum / target_weighted_successes as f64,
        selected_weighted_multiplier_sum / selected_weighted_successes as f64,
    );
}
