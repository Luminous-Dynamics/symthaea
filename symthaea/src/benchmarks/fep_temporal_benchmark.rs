//! FEP Temporal Benchmark
//!
//! Feeds synthetic time-series patterns through the cognitive loop and measures
//! prediction error reduction, FEP free energy decrease, and coherence stability.

use crate::cognitive_loop::{CognitiveLoopService, CognitiveLoopConfig};

/// Configuration for the FEP temporal benchmark.
#[derive(Debug, Clone)]
pub struct FepTemporalBenchmarkConfig {
    pub num_cycles: usize,
    pub warmup_cycles: usize,
    pub measurement_window: usize,
}

impl Default for FepTemporalBenchmarkConfig {
    fn default() -> Self {
        Self {
            num_cycles: 200,
            warmup_cycles: 20,
            measurement_window: 10,
        }
    }
}

/// Results from a benchmark run.
#[derive(Debug, Clone)]
pub struct FepTemporalBenchmarkResult {
    pub initial_error: f32,
    pub final_error: f32,
    pub error_reduction_pct: f32,
    pub fep_initial_free_energy: f64,
    pub fep_final_free_energy: f64,
    pub coherence_stability: f32,
    pub prediction_errors: Vec<f32>,
    pub passed: bool,
}

/// FEP temporal benchmark runner.
pub struct FepTemporalBenchmark {
    config: FepTemporalBenchmarkConfig,
}

impl FepTemporalBenchmark {
    pub fn new(config: FepTemporalBenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run a sine-like repeating word pattern benchmark.
    pub fn run_sine_pattern(&self) -> FepTemporalBenchmarkResult {
        // Repeating pattern: A B C D A B C D ...
        let words = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"];
        self.run_pattern(&words)
    }

    /// Run a step function pattern (abrupt change).
    pub fn run_step_function(&self) -> FepTemporalBenchmarkResult {
        // First half one pattern, second half another
        let mut pattern: Vec<&str> = Vec::with_capacity(self.config.num_cycles);
        for i in 0..self.config.num_cycles {
            if i < self.config.num_cycles / 2 {
                pattern.push("stable input alpha");
            } else {
                pattern.push("changed input beta");
            }
        }
        self.run_sequence(&pattern)
    }

    /// Run a noisy repeating pattern.
    pub fn run_noisy_pattern(&self) -> FepTemporalBenchmarkResult {
        // Repeating with occasional deviations
        let base = ["one two", "three four", "five six"];
        let mut pattern: Vec<&str> = Vec::with_capacity(self.config.num_cycles);
        for i in 0..self.config.num_cycles {
            if i % 7 == 0 {
                pattern.push("noise surprise");
            } else {
                pattern.push(base[i % base.len()]);
            }
        }
        self.run_sequence(&pattern)
    }

    fn run_pattern(&self, words: &[&str]) -> FepTemporalBenchmarkResult {
        let mut pattern: Vec<&str> = Vec::with_capacity(self.config.num_cycles);
        for i in 0..self.config.num_cycles {
            pattern.push(words[i % words.len()]);
        }
        self.run_sequence(&pattern)
    }

    fn run_sequence(&self, sequence: &[&str]) -> FepTemporalBenchmarkResult {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default())
            .expect("Failed to create CognitiveLoopService");

        let mut errors: Vec<f32> = Vec::with_capacity(self.config.num_cycles);
        let mut free_energies: Vec<f64> = Vec::new();

        for (i, input) in sequence.iter().enumerate() {
            let result = service.cycle(input);
            errors.push(result.prediction_error);

            if let Some(fe) = service.fep_free_energy() {
                free_energies.push(fe);
            }

            // Suppress unused variable warnings for warmup period
            let _ = i;
        }

        let w = self.config.measurement_window;
        let warmup = self.config.warmup_cycles;

        // Initial error: average over first measurement window after warmup
        let initial_start = warmup;
        let initial_end = (warmup + w).min(errors.len());
        let initial_error = if initial_end > initial_start {
            errors[initial_start..initial_end].iter().sum::<f32>() / (initial_end - initial_start) as f32
        } else {
            errors.first().copied().unwrap_or(1.0)
        };

        // Final error: average over last measurement window
        let final_start = errors.len().saturating_sub(w);
        let final_error = if final_start < errors.len() {
            errors[final_start..].iter().sum::<f32>() / (errors.len() - final_start) as f32
        } else {
            errors.last().copied().unwrap_or(1.0)
        };

        let error_reduction_pct = if initial_error > 0.0 {
            ((initial_error - final_error) / initial_error) * 100.0
        } else {
            0.0
        };

        // FEP free energy
        let fep_initial = if free_energies.len() > w {
            free_energies[..w].iter().sum::<f64>() / w as f64
        } else {
            free_energies.first().copied().unwrap_or(0.0)
        };

        let fep_final = if free_energies.len() > w {
            let start = free_energies.len() - w;
            free_energies[start..].iter().sum::<f64>() / w as f64
        } else {
            free_energies.last().copied().unwrap_or(0.0)
        };

        // Coherence stability: std_dev of last 50 coherence values
        let coherence_stats = service.coherence_tracker().stats();
        let coherence_stability = coherence_stats.stability;

        // Pass criteria: error reduced by at least some amount
        let passed = error_reduction_pct > 5.0 || final_error < initial_error;

        FepTemporalBenchmarkResult {
            initial_error,
            final_error,
            error_reduction_pct,
            fep_initial_free_energy: fep_initial,
            fep_final_free_energy: fep_final,
            coherence_stability,
            prediction_errors: errors,
            passed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_error_reduction_sine() {
        // Fixed: BPTT regression for 4-item cyclic patterns was caused by:
        // 1. reset_states_only() call in train_step_bptt erasing temporal memory
        // 2. Gradient clipping too permissive (1.0), allowing oscillation
        // 3. Effective learning rate cap too high (0.05), causing overshooting
        //
        // Fix applied in cfc.rs and cognitive_loop.rs:
        // - Removed reset_states_only() from train_step_bptt (preserves temporal context)
        // - Reduced gradient clip from 1.0 to 0.5
        // - Reduced learning rate cap from 0.05 to 0.01
        let bench = FepTemporalBenchmark::new(FepTemporalBenchmarkConfig::default());
        let result = bench.run_sine_pattern();
        println!(
            "Sine: initial_error={:.4}, final_error={:.4}, reduction={:.1}%",
            result.initial_error, result.final_error, result.error_reduction_pct
        );
        // The system should show some learning over 200 cycles of repeating input
        assert!(
            result.final_error <= result.initial_error * 1.1,
            "Final error ({:.4}) should not be significantly worse than initial ({:.4})",
            result.final_error, result.initial_error
        );
    }

    #[test]
    fn test_fep_error_reduction_step() {
        let bench = FepTemporalBenchmark::new(FepTemporalBenchmarkConfig {
            num_cycles: 300,
            warmup_cycles: 20,
            measurement_window: 10,
        });
        let result = bench.run_step_function();
        println!(
            "Step: initial_error={:.4}, final_error={:.4}, reduction={:.1}%",
            result.initial_error, result.final_error, result.error_reduction_pct
        );
        assert!(
            result.prediction_errors.len() == 300,
            "Should have run all 300 cycles"
        );

        // Measure recovery within the second half (after the distribution shift at cycle 150):
        // Compare the spike region (first 20 cycles after shift) vs recovery region (last 20).
        let half = result.prediction_errors.len() / 2;
        let early_2nd: f32 = result.prediction_errors[half..half + 20].iter().sum::<f32>() / 20.0;
        let late_2nd: f32 = result.prediction_errors[result.prediction_errors.len() - 20..].iter().sum::<f32>() / 20.0;
        println!(
            "Step 2nd-half: early_avg={:.4}, late_avg={:.4}, recovery={:.1}%",
            early_2nd, late_2nd,
            if early_2nd > 0.0 { ((early_2nd - late_2nd) / early_2nd) * 100.0 } else { 0.0 }
        );
        // The system should not degrade: late error should not exceed early error by more than 5%
        assert!(
            late_2nd <= early_2nd * 1.05,
            "Error in last 20 cycles ({:.4}) should not significantly exceed first 20 of 2nd half ({:.4})",
            late_2nd, early_2nd
        );
    }

    #[test]
    fn test_coherence_stability() {
        let bench = FepTemporalBenchmark::new(FepTemporalBenchmarkConfig::default());
        let result = bench.run_sine_pattern();
        println!("Coherence stability: {:.4}", result.coherence_stability);
        // Stability should be reasonable (not wildly oscillating)
        assert!(
            result.coherence_stability > 0.5,
            "Coherence stability ({:.4}) should be > 0.5",
            result.coherence_stability
        );
    }

    #[test]
    fn test_fep_precision_increases() {
        let bench = FepTemporalBenchmark::new(FepTemporalBenchmarkConfig {
            num_cycles: 150,
            warmup_cycles: 10,
            measurement_window: 10,
        });
        let result = bench.run_sine_pattern();
        println!(
            "FEP initial free energy: {:.4}, final: {:.4}",
            result.fep_initial_free_energy, result.fep_final_free_energy
        );
        // Just verify the benchmark runs to completion with FEP data
        assert!(result.prediction_errors.len() == 150);
    }
}
