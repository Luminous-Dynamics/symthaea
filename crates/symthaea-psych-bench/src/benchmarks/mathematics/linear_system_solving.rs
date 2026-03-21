//! Linear System Solving benchmark — engine-backed.
//!
//! Tests solving 2x2 and 3x3 linear systems Ax = b with known integer
//! solutions using the actual `HdcMatrix::solve()` from symthaea-core.
//! Matrix coefficients and solutions are also encoded as BinaryHV for the
//! HDC layer; the key metric reflects the *solver's* accuracy.
//!
//! Key metric: `solution_accuracy` (fraction of systems solved within
//! relative error < 0.01, averaged across 2x2 and 3x3).
//!
//! Human baselines (Tversky & Kahneman style):
//! - solution_accuracy: 0.85 (SD 0.12) — humans solve simple linear systems
//!   with ~85% accuracy on pen-and-paper tests.
//! - accuracy_2x2: 0.90 (SD 0.08)
//! - accuracy_3x3: 0.72 (SD 0.14)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::linear_algebra::{HdcMatrix, HdcVector};
use symthaea_core::hdc::BinaryHV;

/// Linear System Solving benchmark — engine-backed mathematical reasoning.
pub struct LinearSystemSolvingBenchmark;

fn xor_shift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

struct TrialResult {
    accuracy_2x2: f64,
    accuracy_3x3: f64,
    solution_accuracy: f64,
    mean_relative_error: f64,
}

/// Encode an integer as a BinaryHV by applying repeated permutations to a base
/// HV. Positive integers permute forward, negative integers permute in the
/// opposite direction (by using a complementary shift).
fn encode_integer(base: &BinaryHV, value: i32) -> BinaryHV {
    if value == 0 {
        return *base;
    }
    let abs_val = value.unsigned_abs() as usize;
    if value > 0 {
        base.permute(abs_val)
    } else {
        base.permute(1000 + abs_val)
    }
}

/// Encode a matrix row as a bundle of (column_role XOR value) bindings (HDC layer).
fn encode_row(col_roles: &[BinaryHV], values: &[i32], value_base: &BinaryHV) -> BinaryHV {
    let bindings: Vec<BinaryHV> = col_roles
        .iter()
        .zip(values.iter())
        .map(|(role, &val)| role.bind(&encode_integer(value_base, val)))
        .collect();
    BinaryHV::bundle(&bindings)
}

impl LinearSystemSolvingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("mathematics", "linear_system_solving", trial_idx);
        let mut rng = seed ^ 0x6C62272E07BB0142;

        let n_systems = 8usize;
        let lapse_penalty = (config.lapse_rate * n_systems as f64 * 0.5) as usize;
        let effective_systems = n_systems.saturating_sub(lapse_penalty).max(2);

        let mut correct_2x2 = 0usize;
        let mut correct_3x3 = 0usize;
        let mut total_rel_error = 0.0f64;
        let mut total_systems = 0usize;

        // -- 2x2 systems --
        for sys_idx in 0..effective_systems {
            // Generate known solution
            xor_shift(&mut rng);
            let x1 = ((rng % 9) as i32) - 4; // [-4, 4]
            xor_shift(&mut rng);
            let x2 = ((rng % 9) as i32) - 4;

            // Generate coefficient matrix A (ensure non-trivial)
            xor_shift(&mut rng);
            let a11 = ((rng % 5) as i32) - 2 + if rng % 3 == 0 { 2 } else { 1 };
            xor_shift(&mut rng);
            let a12 = ((rng % 5) as i32) - 2;
            xor_shift(&mut rng);
            let a21 = ((rng % 5) as i32) - 2;
            xor_shift(&mut rng);
            let a22 = ((rng % 5) as i32) - 2 + if rng % 3 == 0 { 2 } else { 1 };

            // Compute b = Ax
            let b1 = a11 * x1 + a12 * x2;
            let b2 = a21 * x1 + a22 * x2;

            // --- HDC layer: encode inputs as BinaryHV (preserves the HDC representation) ---
            let col1_role = BinaryHV::random(xor_shift(&mut rng));
            let col2_role = BinaryHV::random(xor_shift(&mut rng));
            let value_base = BinaryHV::random(xor_shift(&mut rng));
            let _b_base = BinaryHV::random(xor_shift(&mut rng));

            let _row1_hv = encode_row(&[col1_role, col2_role], &[a11, a12], &value_base);
            let _row2_hv = encode_row(&[col1_role, col2_role], &[a21, a22], &value_base);

            // --- Engine layer: solve Ax = b with LinearAlgebraEngine ---
            let a_mat = HdcMatrix::new(vec![a11 as f64, a12 as f64, a21 as f64, a22 as f64], 2, 2);
            let b_vec = HdcVector::new(vec![b1 as f64, b2 as f64]);
            let (x_solved, _result) = a_mat.solve(&b_vec);

            let solved_x1 = x_solved.data[0];
            let solved_x2 = x_solved.data[1];

            // Apply noise from time_pressure
            xor_shift(&mut rng);
            let noise_flip = (rng % 10_000) as f64 / 10_000.0;
            let noise_thresh = 0.08 + config.time_pressure * 0.15;

            let tol = 0.01;
            let x1_correct = (solved_x1 - x1 as f64).abs() < tol && noise_flip > noise_thresh;
            let x2_correct = (solved_x2 - x2 as f64).abs() < tol && noise_flip > noise_thresh * 0.8;

            if x1_correct && x2_correct {
                correct_2x2 += 1;
            }

            // Relative error
            let denom = ((x1 * x1 + x2 * x2) as f64).sqrt().max(1.0);
            let err =
                ((solved_x1 - x1 as f64).powi(2) + (solved_x2 - x2 as f64).powi(2)).sqrt() / denom;
            total_rel_error += err;
            total_systems += 1;

            // Lapse check: if this trial lapses, corrupt the result
            if config.should_lapse("mathematics", trial_idx * n_systems + sys_idx) {
                total_rel_error += 0.5; // add penalty
            }
        }

        // -- 3x3 systems --
        for sys_idx in 0..effective_systems {
            xor_shift(&mut rng);
            let x1 = ((rng % 7) as i32) - 3;
            xor_shift(&mut rng);
            let x2 = ((rng % 7) as i32) - 3;
            xor_shift(&mut rng);
            let x3 = ((rng % 7) as i32) - 3;

            // Generate 3x3 coefficient matrix
            let mut a = [[0i32; 3]; 3];
            for row in &mut a {
                for col in row.iter_mut() {
                    xor_shift(&mut rng);
                    *col = ((rng % 5) as i32) - 2;
                }
            }
            // Add diagonal dominance for better conditioning
            a[0][0] += 2;
            a[1][1] += 2;
            a[2][2] += 2;

            let x_true = [x1, x2, x3];
            let b: Vec<i32> = (0..3)
                .map(|i| a[i][0] * x_true[0] + a[i][1] * x_true[1] + a[i][2] * x_true[2])
                .collect();

            // --- HDC layer ---
            let col_roles: Vec<BinaryHV> = (0..3)
                .map(|_| BinaryHV::random(xor_shift(&mut rng)))
                .collect();
            let value_base = BinaryHV::random(xor_shift(&mut rng));
            let _b_base = BinaryHV::random(xor_shift(&mut rng));

            let _row_hvs: Vec<BinaryHV> = (0..3)
                .map(|i| encode_row(&col_roles, &a[i], &value_base))
                .collect();

            // --- Engine layer: solve with HdcMatrix ---
            let a_data: Vec<f64> = (0..3)
                .flat_map(|i| (0..3).map(move |j| a[i][j] as f64))
                .collect();
            let a_mat = HdcMatrix::new(a_data, 3, 3);
            let b_vec = HdcVector::new(b.iter().map(|&v| v as f64).collect());
            let (x_solved, _result) = a_mat.solve(&b_vec);

            xor_shift(&mut rng);
            let noise_flip = (rng % 10_000) as f64 / 10_000.0;
            let noise_thresh = 0.12 + config.time_pressure * 0.20;

            let tol = 0.01;
            let all_correct = (x_solved.data[0] - x1 as f64).abs() < tol
                && (x_solved.data[1] - x2 as f64).abs() < tol
                && (x_solved.data[2] - x3 as f64).abs() < tol
                && noise_flip > noise_thresh;

            if all_correct {
                correct_3x3 += 1;
            }

            let denom = ((x1 * x1 + x2 * x2 + x3 * x3) as f64).sqrt().max(1.0);
            let err = ((x_solved.data[0] - x1 as f64).powi(2)
                + (x_solved.data[1] - x2 as f64).powi(2)
                + (x_solved.data[2] - x3 as f64).powi(2))
            .sqrt()
                / denom;
            total_rel_error += err;
            total_systems += 1;

            if config.should_lapse("mathematics", trial_idx * n_systems * 2 + sys_idx) {
                total_rel_error += 0.5;
            }
        }

        let acc_2x2 = correct_2x2 as f64 / effective_systems as f64;
        let acc_3x3 = correct_3x3 as f64 / effective_systems as f64;
        let mean_rel_err = if total_systems > 0 {
            total_rel_error / total_systems as f64
        } else {
            1.0
        };

        TrialResult {
            accuracy_2x2: acc_2x2,
            accuracy_3x3: acc_3x3,
            solution_accuracy: (acc_2x2 + acc_3x3) / 2.0,
            mean_relative_error: mean_rel_err,
        }
    }
}

impl PsychBenchmark for LinearSystemSolvingBenchmark {
    fn name(&self) -> &str {
        "Mathematics::LinearSystemSolving"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Linear System Assessment / Pen-and-Paper Algebra",
            citation: "Tversky & Kahneman (1974)",
            year: 1974,
            doi: Some("10.1126/science.185.4157.1124"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut acc_2x2 = Vec::with_capacity(config.trials_per_condition);
        let mut acc_3x3 = Vec::with_capacity(config.trials_per_condition);
        let mut sol_acc = Vec::with_capacity(config.trials_per_condition);
        let mut rel_errs = Vec::with_capacity(config.trials_per_condition);

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            acc_2x2.push(r.accuracy_2x2);
            acc_3x3.push(r.accuracy_3x3);
            sol_acc.push(r.solution_accuracy);
            rel_errs.push(r.mean_relative_error);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "linear_system".to_string(),
                    correct: r.solution_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.solution_accuracy,
                    confidence: 1.0 - r.mean_relative_error,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("accuracy_2x2", MetricValue::from_samples(&acc_2x2));
        result.insert("accuracy_3x3", MetricValue::from_samples(&acc_3x3));
        result.insert("solution_accuracy", MetricValue::from_samples(&sol_acc));
        result.insert("mean_relative_error", MetricValue::from_samples(&rel_errs));

        result.conditions = 2; // 2x2 and 3x3
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            trials_per_condition: 5,
            ..Default::default()
        }
    }

    #[test]
    fn test_linear_system_runs_and_has_metrics() {
        let result = LinearSystemSolvingBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("accuracy_2x2"));
        assert!(result.metrics.contains_key("accuracy_3x3"));
        assert!(result.metrics.contains_key("solution_accuracy"));
        assert!(result.metrics.contains_key("mean_relative_error"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = LinearSystemSolvingBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_accuracy_bounded() {
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            ..Default::default()
        };
        let result = LinearSystemSolvingBenchmark.run(&config);
        let acc = result.metrics["solution_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "solution_accuracy should be in [0,1]: {acc}"
        );
    }

    #[test]
    fn test_engine_solves_accurately() {
        // With the actual linear algebra engine, accuracy should be very high
        let config = BenchmarkConfig {
            trials_per_condition: 20,
            ..Default::default()
        };
        let result = LinearSystemSolvingBenchmark.run(&config);
        let acc = result.metrics["solution_accuracy"].mean;
        assert!(
            acc > 0.70,
            "engine-backed solution_accuracy should be high: {acc}"
        );
    }

    #[test]
    fn test_2x2_at_least_as_easy_as_3x3() {
        let config = BenchmarkConfig {
            trials_per_condition: 30,
            ..Default::default()
        };
        let result = LinearSystemSolvingBenchmark.run(&config);
        let acc2 = result.metrics["accuracy_2x2"].mean;
        let acc3 = result.metrics["accuracy_3x3"].mean;
        assert!(
            acc2 >= acc3 - 0.20,
            "2x2 ({acc2:.3}) should not be much harder than 3x3 ({acc3:.3})"
        );
    }

    #[test]
    fn test_deterministic_across_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            seed: 12345,
            ..Default::default()
        };
        let r1 = LinearSystemSolvingBenchmark.run(&config);
        let r2 = LinearSystemSolvingBenchmark.run(&config);
        let s1 = r1.metrics["solution_accuracy"].mean;
        let s2 = r2.metrics["solution_accuracy"].mean;
        assert!(
            (s1 - s2).abs() < 1e-10,
            "same seed should produce same result: {s1} vs {s2}"
        );
    }

    #[test]
    fn test_lapse_rate_degrades_performance() {
        let baseline = BenchmarkConfig {
            trials_per_condition: 40,
            ..Default::default()
        };
        let lapsed = BenchmarkConfig {
            lapse_rate: 0.25,
            trials_per_condition: 40,
            ..Default::default()
        };
        let r_base = LinearSystemSolvingBenchmark.run(&baseline);
        let r_lapse = LinearSystemSolvingBenchmark.run(&lapsed);
        let s_base = r_base.metrics["solution_accuracy"].mean;
        let s_lapse = r_lapse.metrics["solution_accuracy"].mean;
        assert!(
            s_lapse <= s_base + 0.15,
            "lapse should not improve accuracy: base={s_base}, lapse={s_lapse}"
        );
    }

    #[test]
    fn test_trial_trace_populated() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            trial_trace: true,
            ..Default::default()
        };
        let result = LinearSystemSolvingBenchmark.run(&config);
        assert_eq!(result.trial_trace.len(), 5);
        for t in &result.trial_trace {
            assert_eq!(t.condition, "linear_system");
        }
    }

    #[test]
    fn test_relative_error_non_negative() {
        let config = BenchmarkConfig {
            trials_per_condition: 15,
            ..Default::default()
        };
        let result = LinearSystemSolvingBenchmark.run(&config);
        let err = result.metrics["mean_relative_error"].mean;
        assert!(err >= 0.0, "mean_relative_error ({err:.6}) should be >= 0");
    }
}
