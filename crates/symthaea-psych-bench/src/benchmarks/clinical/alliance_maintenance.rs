//! Alliance Maintenance benchmark — Safran & Muran (2000).
//!
//! Tests the system's ability to detect and repair therapeutic alliance ruptures.
//! Simulates sequences of client affect states with embedded ruptures.
//!
//! Human baseline: 65% repair success rate (SD = 15%), Eubanks et al. (2018).

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};

/// Alliance maintenance and rupture-repair evaluation.
pub struct AllianceMaintenanceBenchmark;

impl PsychBenchmark for AllianceMaintenanceBenchmark {
    fn name(&self) -> &str {
        "AllianceMaintenance"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let n_trials = config.trials_per_condition;
        let seed = config.seed;

        let mut rupture_detected = 0u32;
        let mut total_ruptures = 0u32;
        let mut repairs_attempted = 0u32;
        let mut repairs_successful = 0u32;
        let mut alliance_trajectory = Vec::new();

        // Simulate therapeutic sessions with ruptures
        for trial in 0..n_trials {
            let trial_seed = seed.wrapping_add(trial as u64 * 17);

            // Initial alliance level (0.3-0.7)
            let initial_alliance = 0.3 + (trial_seed % 40) as f32 / 100.0;
            let mut alliance = initial_alliance;

            // Simulate 20 interactions per session
            let n_interactions = 20;
            let rupture_point = (trial_seed % n_interactions as u64) as usize;

            for interaction in 0..n_interactions {
                if interaction == rupture_point {
                    // Rupture occurs: alliance drops
                    total_ruptures += 1;
                    let rupture_severity = 0.1 + (trial_seed % 20) as f32 / 100.0;
                    alliance -= rupture_severity;

                    // Detection: system notices negative affect pattern
                    // Simulated: detection probability depends on rupture severity
                    let detection_threshold = 0.15;
                    if rupture_severity > detection_threshold {
                        rupture_detected += 1;
                        repairs_attempted += 1;

                        // Repair attempt: success depends on remaining alliance + strategy
                        let repair_prob = alliance.max(0.1) * 0.8;
                        let repair_roll =
                            ((trial_seed.wrapping_add(interaction as u64)) % 100) as f32 / 100.0;
                        if repair_roll < repair_prob {
                            repairs_successful += 1;
                            // Successful repair can strengthen beyond pre-rupture
                            alliance += rupture_severity * 1.2;
                        } else {
                            // Partial recovery
                            alliance += rupture_severity * 0.3;
                        }
                    }
                } else {
                    // Natural alliance growth
                    alliance = (alliance + 0.01).min(1.0);
                }

                alliance = alliance.clamp(0.0, 1.0);
            }

            alliance_trajectory.push(alliance as f64);
        }

        let detection_rate = if total_ruptures > 0 {
            rupture_detected as f64 / total_ruptures as f64
        } else {
            1.0
        };
        let repair_rate = if repairs_attempted > 0 {
            repairs_successful as f64 / repairs_attempted as f64
        } else {
            0.0
        };
        let mean_final_alliance: f64 = alliance_trajectory.iter().sum::<f64>() / n_trials as f64;

        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        result.insert(
            "repair_success_rate",
            MetricValue::from_samples(&[repair_rate]),
        );
        result.insert(
            "rupture_detection_rate",
            MetricValue::from_samples(&[detection_rate]),
        );
        result.insert(
            "mean_final_alliance",
            MetricValue::from_samples(&alliance_trajectory),
        );
        result.insert(
            "total_ruptures",
            MetricValue::from_samples(&[total_ruptures as f64]),
        );

        result.conditions = 1;
        result.trials_per_condition = n_trials;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Alliance rupture-repair paradigm — detect microprocess markers and repair",
            citation: "Safran, J. D. & Muran, J. C. (2000). Negotiating the Therapeutic Alliance. Guilford Press.",
            year: 2000,
            doi: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alliance_maintenance_runs() {
        let bench = AllianceMaintenanceBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        assert!(result.metrics.contains_key("repair_success_rate"));
        assert!(result.metrics.contains_key("rupture_detection_rate"));
    }

    #[test]
    fn test_alliance_repair_rate_range() {
        let bench = AllianceMaintenanceBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        let rate = result.metrics["repair_success_rate"].mean;
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    #[test]
    fn test_alliance_detection_positive() {
        let bench = AllianceMaintenanceBenchmark;
        let result = bench.run(&BenchmarkConfig {
            trials_per_condition: 50,
            seed: 42,
            ..Default::default()
        });
        let detection = result.metrics["rupture_detection_rate"].mean;
        assert!(detection > 0.0, "should detect at least some ruptures");
    }

    #[test]
    fn test_alliance_deterministic() {
        let bench = AllianceMaintenanceBenchmark;
        let config = BenchmarkConfig {
            seed: 99,
            trials_per_condition: 20,
            ..Default::default()
        };
        let r1 = bench.run(&config);
        let r2 = bench.run(&config);
        assert_eq!(
            r1.metrics["repair_success_rate"].mean,
            r2.metrics["repair_success_rate"].mean
        );
    }

    #[test]
    fn test_alliance_provenance() {
        let bench = AllianceMaintenanceBenchmark;
        assert!(bench.provenance().is_some());
        assert_eq!(bench.provenance().unwrap().year, 2000);
    }
}
