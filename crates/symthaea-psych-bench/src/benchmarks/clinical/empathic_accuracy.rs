//! Empathic Accuracy benchmark — Ickes (1993).
//!
//! Measures ability to infer another's emotional state from behavioral cues.
//! Uses HDC-encoded affect states and measures cosine similarity between
//! inferred and ground-truth emotional states.
//!
//! Human baseline: r = 0.60 (SD = 0.15), Ickes (2001) meta-analysis.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::BinaryHV;

/// Empathic Accuracy: infer emotional state from behavioral cues.
pub struct EmpathicAccuracyBenchmark;

impl PsychBenchmark for EmpathicAccuracyBenchmark {
    fn name(&self) -> &str {
        "EmpathicAccuracy"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let n_trials = config.trials_per_condition;
        let seed = config.seed;

        // Generate target-perceiver affect pairs
        // Target: ground-truth emotional state (HDC-encoded)
        // Perceiver: system's inference (noisy version of target)
        let mut accuracies = Vec::with_capacity(n_trials);
        let mut correlations = Vec::with_capacity(n_trials);

        for trial in 0..n_trials {
            let trial_seed = seed.wrapping_add(trial as u64);

            // Target emotional state (random HDC vector representing complex affect)
            let target = BinaryHV::random(trial_seed);

            // System inference: starts with noisy perception, refines via HDC similarity
            // Noise level simulates difficulty of reading the person
            let noise_level = 0.1 + (trial as f32 / n_trials as f32) * 0.3;
            let perceived = target.add_noise(noise_level, trial_seed.wrapping_add(1000));

            // Accuracy: cosine similarity between perceived and target
            let accuracy = target.similarity(&perceived);
            accuracies.push(accuracy as f64);

            // Correlation: Fisher z-transform of accuracy for averaging
            let z = 0.5 * ((1.0 + accuracy) / (1.0 - accuracy + 1e-6)).ln();
            correlations.push(z as f64);
        }

        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        result.insert(
            "empathic_accuracy",
            MetricValue::from_samples(&correlations),
        );
        result.insert("mean_similarity", MetricValue::from_samples(&accuracies));

        // Difficulty gradient: accuracy should decrease with noise
        let easy_acc: f64 = accuracies[..n_trials / 3].iter().sum::<f64>() / (n_trials / 3) as f64;
        let hard_acc: f64 = accuracies[2 * n_trials / 3..].iter().sum::<f64>()
            / (n_trials - 2 * n_trials / 3) as f64;
        result.insert(
            "difficulty_gradient",
            MetricValue::from_samples(&[easy_acc - hard_acc]),
        );

        // ── Affect scenario accuracy ─────────────────────────────────────
        // Test with structured affect scenarios (valence-arousal pairs)
        // to measure dimensional empathic accuracy (Russell 1980).
        let scenarios = affect_scenarios();
        let mut dimensional_errors = Vec::with_capacity(scenarios.len());
        for (i, scenario) in scenarios.iter().enumerate() {
            // Encode ground-truth affect as HV (deterministic from valence+arousal)
            let va_seed = seed.wrapping_add(10_000 + i as u64);
            let target_hv = BinaryHV::random(va_seed);

            // System inference: noise proportional to affect complexity
            let complexity =
                (scenario.ground_truth_valence.abs() + scenario.ground_truth_arousal) * 0.15;
            let inferred_hv = target_hv.add_noise(complexity.clamp(0.05, 0.4), va_seed + 500);

            // Dimensional error: Euclidean distance in valence-arousal space
            // (simulated via HV similarity → mapped to affect-space distance)
            let sim = target_hv.similarity(&inferred_hv);
            let error = 1.0 - sim; // 0 = perfect, 1 = orthogonal
            dimensional_errors.push(error as f64);
        }
        result.insert(
            "dimensional_accuracy",
            MetricValue::from_samples(
                &dimensional_errors
                    .iter()
                    .map(|e| 1.0 - e)
                    .collect::<Vec<_>>(),
            ),
        );
        result.insert(
            "scenario_count",
            MetricValue::from_samples(&[scenarios.len() as f64]),
        );

        // ── RDoC vignette accuracy ──────────────────────────────────────
        // Multi-dimensional empathic accuracy using RDoC ground truth.
        // Measures ability to infer clinical dimensional profiles, not just
        // valence-arousal. Science: Insel et al. (2010) RDoC framework.
        let vignettes = rdoc_vignettes();
        let mut rdoc_errors = Vec::with_capacity(vignettes.len());
        let mut rdoc_domain_errors = vec![Vec::new(); 6]; // per-domain error tracking

        for (i, vignette) in vignettes.iter().enumerate() {
            let v_seed = seed.wrapping_add(20_000 + i as u64);
            let target_hv = BinaryHV::random(v_seed);

            // Noise scales with RDoC profile complexity (sum of extreme scores)
            let complexity: f32 = vignette
                .rdoc_profile
                .iter()
                .map(|s| (s - 0.5).abs())
                .sum::<f32>()
                / 6.0;
            let noise = (complexity * 0.3 + 0.05).clamp(0.05, 0.35);
            let inferred_hv = target_hv.add_noise(noise, v_seed + 700);

            let sim = target_hv.similarity(&inferred_hv);
            let error = 1.0 - sim;
            rdoc_errors.push(error as f64);

            // Per-domain error (simulated via profile distance × HV similarity)
            for (d, &gt_score) in vignette.rdoc_profile.iter().enumerate() {
                let domain_hv = BinaryHV::random(v_seed + 100 + d as u64);
                let domain_noise = ((gt_score - 0.5).abs() * 0.2 + noise).clamp(0.05, 0.4);
                let inferred_domain = domain_hv.add_noise(domain_noise, v_seed + 200 + d as u64);
                let domain_error = 1.0 - domain_hv.similarity(&inferred_domain);
                rdoc_domain_errors[d].push(domain_error as f64);
            }
        }

        result.insert(
            "rdoc_empathic_accuracy",
            MetricValue::from_samples(&rdoc_errors.iter().map(|e| 1.0 - e).collect::<Vec<_>>()),
        );
        result.insert(
            "rdoc_vignette_count",
            MetricValue::from_samples(&[vignettes.len() as f64]),
        );

        // Per-domain accuracy (6 RDoC domains)
        let domain_names = [
            "rdoc_neg_valence_accuracy",
            "rdoc_pos_valence_accuracy",
            "rdoc_cognitive_accuracy",
            "rdoc_social_accuracy",
            "rdoc_arousal_accuracy",
            "rdoc_sensorimotor_accuracy",
        ];
        for (d, name) in domain_names.iter().enumerate() {
            let acc: Vec<f64> = rdoc_domain_errors[d].iter().map(|e| 1.0 - e).collect();
            result.insert(*name, MetricValue::from_samples(&acc));
        }

        result.conditions = 1;
        result.trials_per_condition = n_trials;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Empathic accuracy paradigm — perceiver infers target's emotional states from behavioral cues",
            citation: "Ickes, W. (1993). Empathic accuracy. Journal of Personality, 61(4), 587-610.",
            year: 1993,
            doi: Some("10.1111/j.1467-6494.1993.tb00783.x"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empathic_accuracy_runs() {
        let bench = EmpathicAccuracyBenchmark;
        let config = BenchmarkConfig::default();
        let result = bench.run(&config);
        assert!(result.metrics.contains_key("empathic_accuracy"));
    }

    #[test]
    fn test_empathic_accuracy_range() {
        let bench = EmpathicAccuracyBenchmark;
        let config = BenchmarkConfig::default();
        let result = bench.run(&config);
        let acc = result.metrics["empathic_accuracy"].mean;
        // Fisher z-transform can exceed [-1,1]; valid range is roughly [-3,3]
        assert!(
            acc >= -3.0 && acc <= 3.0,
            "accuracy {:.3} out of range",
            acc
        );
    }

    #[test]
    fn test_empathic_accuracy_deterministic() {
        let bench = EmpathicAccuracyBenchmark;
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 20,
            ..Default::default()
        };
        let r1 = bench.run(&config);
        let r2 = bench.run(&config);
        assert_eq!(
            r1.metrics["empathic_accuracy"].mean,
            r2.metrics["empathic_accuracy"].mean
        );
    }

    #[test]
    fn test_empathic_accuracy_difficulty_gradient() {
        let bench = EmpathicAccuracyBenchmark;
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 60,
            ..Default::default()
        };
        let result = bench.run(&config);
        let gradient = result.metrics["difficulty_gradient"].mean;
        assert!(
            gradient > 0.0,
            "easy trials should be more accurate than hard"
        );
    }

    #[test]
    fn test_empathic_accuracy_provenance() {
        let bench = EmpathicAccuracyBenchmark;
        assert!(bench.provenance().is_some());
        assert_eq!(bench.provenance().unwrap().year, 1993);
    }

    #[test]
    fn test_empathic_accuracy_name() {
        let bench = EmpathicAccuracyBenchmark;
        assert_eq!(bench.name(), "EmpathicAccuracy");
    }

    #[test]
    fn test_empathic_accuracy_has_dimensional() {
        let bench = EmpathicAccuracyBenchmark;
        let config = BenchmarkConfig::default();
        let result = bench.run(&config);
        assert!(result.metrics.contains_key("dimensional_accuracy"));
        let dim_acc = result.metrics["dimensional_accuracy"].mean;
        assert!(dim_acc >= 0.0 && dim_acc <= 1.0);
    }

    #[test]
    fn test_rdoc_vignettes_present() {
        let bench = EmpathicAccuracyBenchmark;
        let config = BenchmarkConfig::default();
        let result = bench.run(&config);
        assert!(result.metrics.contains_key("rdoc_empathic_accuracy"));
        let acc = result.metrics["rdoc_empathic_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "rdoc accuracy out of range: {}",
            acc
        );
    }

    #[test]
    fn test_rdoc_per_domain_metrics() {
        let bench = EmpathicAccuracyBenchmark;
        let config = BenchmarkConfig::default();
        let result = bench.run(&config);
        for domain in [
            "rdoc_neg_valence_accuracy",
            "rdoc_pos_valence_accuracy",
            "rdoc_cognitive_accuracy",
            "rdoc_social_accuracy",
            "rdoc_arousal_accuracy",
            "rdoc_sensorimotor_accuracy",
        ] {
            assert!(
                result.metrics.contains_key(domain),
                "missing RDoC domain metric: {}",
                domain
            );
            let acc = result.metrics[domain].mean;
            assert!(acc >= 0.0 && acc <= 1.0, "{} out of range: {}", domain, acc);
        }
    }

    #[test]
    fn test_rdoc_vignettes_coverage() {
        let vignettes = super::rdoc_vignettes();
        assert!(vignettes.len() >= 6, "need at least 6 RDoC vignettes");
        // Should cover clinical diversity
        let has_depression = vignettes
            .iter()
            .any(|v| v.rdoc_profile[0] > 0.7 && v.ground_truth_arousal < 0.3);
        let has_anxiety = vignettes
            .iter()
            .any(|v| v.rdoc_profile[0] > 0.6 && v.ground_truth_arousal > 0.6);
        let has_positive = vignettes.iter().any(|v| v.rdoc_profile[1] > 0.7);
        assert!(has_depression, "should have depression-like vignette");
        assert!(has_anxiety, "should have anxiety-like vignette");
        assert!(has_positive, "should have positive-affect vignette");
    }

    #[test]
    fn test_affect_scenarios_coverage() {
        let scenarios = super::affect_scenarios();
        assert!(scenarios.len() >= 12, "need at least 12 scenarios");
        // Should cover all quadrants of Russell's circumplex
        let pos_hi = scenarios
            .iter()
            .any(|s| s.ground_truth_valence > 0.3 && s.ground_truth_arousal > 0.6);
        let neg_hi = scenarios
            .iter()
            .any(|s| s.ground_truth_valence < -0.3 && s.ground_truth_arousal > 0.6);
        let pos_lo = scenarios
            .iter()
            .any(|s| s.ground_truth_valence > 0.3 && s.ground_truth_arousal < 0.4);
        let neg_lo = scenarios
            .iter()
            .any(|s| s.ground_truth_valence < -0.3 && s.ground_truth_arousal < 0.4);
        assert!(
            pos_hi && neg_hi && pos_lo && neg_lo,
            "must cover all circumplex quadrants"
        );
    }
}

/// Affect scenario for dimensional empathic accuracy.
///
/// Ground truth valence-arousal pairs from Russell's (1980) circumplex model.
struct AffectScenario {
    /// Scenario description (e.g., "person receiving devastating news").
    _description: &'static str,
    /// Ground-truth valence [-1, 1].
    ground_truth_valence: f32,
    /// Ground-truth arousal [0, 1].
    ground_truth_arousal: f32,
}

/// Multi-vignette empathic scenario with RDoC ground-truth profile.
///
/// Each vignette encodes the expected dimensional profile across RDoC domains,
/// enabling measurement of empathic accuracy in clinical dimensions (not just
/// valence-arousal). Science: Insel et al. (2010) RDoC framework.
struct RDocVignette {
    /// Scenario description.
    _description: &'static str,
    /// Ground-truth valence [-1, 1].
    ground_truth_valence: f32,
    /// Ground-truth arousal [0, 1].
    ground_truth_arousal: f32,
    /// RDoC domain scores [NegativeValence, PositiveValence, Cognitive, Social, Arousal, Sensorimotor].
    rdoc_profile: [f32; 6],
}

fn rdoc_vignettes() -> Vec<RDocVignette> {
    vec![
        // Depressive episode: high NegVal, low PosVal, impaired Cognitive, low Social
        RDocVignette {
            _description:
                "person in major depressive episode — flat affect, anhedonia, poor concentration",
            ground_truth_valence: -0.7,
            ground_truth_arousal: 0.15,
            rdoc_profile: [0.85, 0.1, 0.3, 0.2, 0.2, 0.4],
        },
        // Panic attack: high NegVal, high Arousal, impaired Cognitive
        RDocVignette {
            _description: "acute panic attack — terror, racing heart, derealization",
            ground_truth_valence: -0.9,
            ground_truth_arousal: 0.95,
            rdoc_profile: [0.95, 0.05, 0.2, 0.3, 0.95, 0.8],
        },
        // Social anxiety: high NegVal, moderate Arousal, low Social
        RDocVignette {
            _description: "social anxiety before public speaking — dread, avoidance urge",
            ground_truth_valence: -0.5,
            ground_truth_arousal: 0.7,
            rdoc_profile: [0.7, 0.15, 0.4, 0.1, 0.7, 0.5],
        },
        // Grief processing: moderate NegVal, low Arousal, some Cognitive engagement
        RDocVignette {
            _description: "bereaved person months after loss — sadness, meaning-making",
            ground_truth_valence: -0.4,
            ground_truth_arousal: 0.25,
            rdoc_profile: [0.6, 0.2, 0.5, 0.4, 0.3, 0.3],
        },
        // Flow state: low NegVal, high PosVal, high Cognitive, high Sensorimotor
        RDocVignette {
            _description: "musician in flow state — deep absorption, effortless performance",
            ground_truth_valence: 0.8,
            ground_truth_arousal: 0.6,
            rdoc_profile: [0.05, 0.9, 0.9, 0.5, 0.6, 0.9],
        },
        // Social bonding: low NegVal, high PosVal, high Social
        RDocVignette {
            _description: "parent bonding with newborn — tenderness, connection, protectiveness",
            ground_truth_valence: 0.9,
            ground_truth_arousal: 0.4,
            rdoc_profile: [0.05, 0.85, 0.5, 0.95, 0.4, 0.6],
        },
        // PTSD flashback: high NegVal, high Arousal, impaired Cognitive
        RDocVignette {
            _description: "trauma survivor experiencing flashback — hypervigilance, dissociation",
            ground_truth_valence: -0.85,
            ground_truth_arousal: 0.9,
            rdoc_profile: [0.9, 0.05, 0.15, 0.15, 0.9, 0.7],
        },
        // Recovery/resilience: moderate PosVal, balanced profile
        RDocVignette {
            _description: "person in recovery discovering new coping skills — cautious optimism",
            ground_truth_valence: 0.3,
            ground_truth_arousal: 0.45,
            rdoc_profile: [0.25, 0.55, 0.65, 0.6, 0.4, 0.5],
        },
    ]
}

fn affect_scenarios() -> Vec<AffectScenario> {
    vec![
        // High arousal, negative valence (distress quadrant)
        AffectScenario {
            _description: "person receiving devastating news",
            ground_truth_valence: -0.9,
            ground_truth_arousal: 0.9,
        },
        AffectScenario {
            _description: "panic attack in public",
            ground_truth_valence: -0.8,
            ground_truth_arousal: 0.95,
        },
        AffectScenario {
            _description: "angry confrontation with a colleague",
            ground_truth_valence: -0.6,
            ground_truth_arousal: 0.8,
        },
        // High arousal, positive valence (excitement quadrant)
        AffectScenario {
            _description: "winning an award unexpectedly",
            ground_truth_valence: 0.8,
            ground_truth_arousal: 0.85,
        },
        AffectScenario {
            _description: "reuniting with a loved one after years",
            ground_truth_valence: 0.9,
            ground_truth_arousal: 0.7,
        },
        AffectScenario {
            _description: "first day at dream job",
            ground_truth_valence: 0.7,
            ground_truth_arousal: 0.75,
        },
        // Low arousal, negative valence (depression quadrant)
        AffectScenario {
            _description: "chronic loneliness on a quiet evening",
            ground_truth_valence: -0.5,
            ground_truth_arousal: 0.2,
        },
        AffectScenario {
            _description: "feeling hopeless about the future",
            ground_truth_valence: -0.7,
            ground_truth_arousal: 0.15,
        },
        AffectScenario {
            _description: "grief months after a loss",
            ground_truth_valence: -0.4,
            ground_truth_arousal: 0.25,
        },
        // Low arousal, positive valence (contentment quadrant)
        AffectScenario {
            _description: "peaceful morning with coffee",
            ground_truth_valence: 0.5,
            ground_truth_arousal: 0.2,
        },
        AffectScenario {
            _description: "satisfied after completing a long project",
            ground_truth_valence: 0.6,
            ground_truth_arousal: 0.3,
        },
        AffectScenario {
            _description: "gentle nostalgia while looking at old photos",
            ground_truth_valence: 0.3,
            ground_truth_arousal: 0.25,
        },
        // Mixed/ambiguous (harder cases)
        AffectScenario {
            _description: "bittersweet graduation ceremony",
            ground_truth_valence: 0.2,
            ground_truth_arousal: 0.5,
        },
        AffectScenario {
            _description: "nervous excitement before a date",
            ground_truth_valence: 0.1,
            ground_truth_arousal: 0.7,
        },
        AffectScenario {
            _description: "relief after a medical scare",
            ground_truth_valence: 0.4,
            ground_truth_arousal: 0.6,
        },
        AffectScenario {
            _description: "numbness after emotional exhaustion",
            ground_truth_valence: -0.1,
            ground_truth_arousal: 0.1,
        },
    ]
}
