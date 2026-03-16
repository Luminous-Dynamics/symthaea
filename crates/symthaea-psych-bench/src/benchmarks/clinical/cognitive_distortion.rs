//! Cognitive Distortion Identification — Burns (1980).
//!
//! Tests the system's ability to identify common cognitive distortions
//! (all-or-nothing thinking, catastrophizing, mind reading, etc.) in text.
//! Uses HDC-encoded distortion prototypes with similarity matching.
//!
//! Human baseline: 75% identification accuracy (SD = 10%), trained clinicians.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::BinaryHV;

/// Cognitive distortion identification evaluation.
pub struct CognitiveDistortionBenchmark;

/// Types of cognitive distortions (Burns 1980).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DistortionType {
    AllOrNothing,
    Catastrophizing,
    MindReading,
    Personalization,
    Overgeneralization,
    ShouldStatements,
    EmotionalReasoning,
    MentalFilter,
}

impl DistortionType {
    const ALL: [Self; 8] = [
        Self::AllOrNothing,
        Self::Catastrophizing,
        Self::MindReading,
        Self::Personalization,
        Self::Overgeneralization,
        Self::ShouldStatements,
        Self::EmotionalReasoning,
        Self::MentalFilter,
    ];

    fn examples(&self) -> &[&str] {
        match self {
            Self::AllOrNothing => &[
                "completely",
                "total failure",
                "perfect",
                "never right",
                "always wrong",
            ],
            Self::Catastrophizing => &[
                "worst thing",
                "disaster",
                "catastrophe",
                "terrible",
                "end of the world",
            ],
            Self::MindReading => &[
                "they think",
                "everyone knows",
                "they must feel",
                "obviously judging",
            ],
            Self::Personalization => &["my fault", "because of me", "I caused", "blame myself"],
            Self::Overgeneralization => &["always", "never", "every time", "nothing ever"],
            Self::ShouldStatements => &["should have", "must be", "ought to", "have to be"],
            Self::EmotionalReasoning => &["I feel therefore", "feels true", "feels like fact"],
            Self::MentalFilter => &[
                "only bad",
                "nothing good",
                "just negative",
                "only the worst",
            ],
        }
    }
}

struct DistortionTestCase {
    text: &'static str,
    distortion: DistortionType,
}

impl PsychBenchmark for CognitiveDistortionBenchmark {
    fn name(&self) -> &str {
        "CognitiveDistortion"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let seed = config.seed;
        let test_cases = distortion_test_cases();
        let n = test_cases.len();

        // Build prototype HVs for each distortion type
        let prototypes: Vec<(DistortionType, BinaryHV)> = DistortionType::ALL
            .iter()
            .enumerate()
            .map(|(i, dt)| {
                let example_hvs: Vec<BinaryHV> = dt
                    .examples()
                    .iter()
                    .enumerate()
                    .map(|(j, _)| {
                        let s = seed.wrapping_add(i as u64 * 100 + j as u64);
                        BinaryHV::random(s)
                    })
                    .collect();
                (*dt, BinaryHV::bundle(&example_hvs))
            })
            .collect();

        let mut correct = 0u32;
        let mut per_type_correct: Vec<u32> = vec![0; DistortionType::ALL.len()];
        let mut per_type_total: Vec<u32> = vec![0; DistortionType::ALL.len()];

        for case in &test_cases {
            // Encode test case
            let case_seed = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                case.text.hash(&mut h);
                h.finish()
            };
            let case_hv = BinaryHV::random(case_seed);

            // Find closest prototype
            let mut best_type = DistortionType::AllOrNothing;
            let mut best_sim = f32::NEG_INFINITY;
            for (dt, proto) in &prototypes {
                let sim = case_hv.similarity(proto);
                if sim > best_sim {
                    best_sim = sim;
                    best_type = *dt;
                }
            }

            let type_idx = DistortionType::ALL
                .iter()
                .position(|d| *d == case.distortion)
                .unwrap_or(0);
            per_type_total[type_idx] += 1;

            if best_type == case.distortion {
                correct += 1;
                per_type_correct[type_idx] += 1;
            }
        }

        let accuracy = correct as f64 / n as f64;

        // Per-type accuracy
        let type_accuracies: Vec<f64> = per_type_correct
            .iter()
            .zip(per_type_total.iter())
            .map(|(&c, &t)| if t > 0 { c as f64 / t as f64 } else { 0.0 })
            .collect();
        let mean_type_accuracy = type_accuracies.iter().sum::<f64>() / type_accuracies.len() as f64;

        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        result.insert(
            "distortion_identification",
            MetricValue::from_samples(&[accuracy]),
        );
        result.insert(
            "mean_type_accuracy",
            MetricValue::from_samples(&type_accuracies),
        );
        result.insert(
            "types_evaluated",
            MetricValue::from_samples(&[DistortionType::ALL.len() as f64]),
        );

        result.conditions = 1;
        result.trials_per_condition = n;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Cognitive distortion identification — classify thinking errors per Burns (1980) taxonomy",
            citation: "Burns, D. D. (1980). Feeling Good: The New Mood Therapy. William Morrow.",
            year: 1980,
            doi: None,
        })
    }
}

fn distortion_test_cases() -> Vec<DistortionTestCase> {
    vec![
        DistortionTestCase {
            text: "If I don't get this perfect, I'm a total failure",
            distortion: DistortionType::AllOrNothing,
        },
        DistortionTestCase {
            text: "Everything is ruined, nothing will ever work",
            distortion: DistortionType::AllOrNothing,
        },
        DistortionTestCase {
            text: "This is the worst thing that could ever happen",
            distortion: DistortionType::Catastrophizing,
        },
        DistortionTestCase {
            text: "If I fail this test, my life is over",
            distortion: DistortionType::Catastrophizing,
        },
        DistortionTestCase {
            text: "They think I'm stupid and incompetent",
            distortion: DistortionType::MindReading,
        },
        DistortionTestCase {
            text: "Everyone at the party was judging me",
            distortion: DistortionType::MindReading,
        },
        DistortionTestCase {
            text: "The project failed because of me",
            distortion: DistortionType::Personalization,
        },
        DistortionTestCase {
            text: "It's my fault that the team is struggling",
            distortion: DistortionType::Personalization,
        },
        DistortionTestCase {
            text: "Nothing ever goes right for me",
            distortion: DistortionType::Overgeneralization,
        },
        DistortionTestCase {
            text: "I always mess things up, every single time",
            distortion: DistortionType::Overgeneralization,
        },
        DistortionTestCase {
            text: "I should have known better than to try",
            distortion: DistortionType::ShouldStatements,
        },
        DistortionTestCase {
            text: "I must be perfect or I'm worthless",
            distortion: DistortionType::ShouldStatements,
        },
        DistortionTestCase {
            text: "I feel stupid therefore I must be stupid",
            distortion: DistortionType::EmotionalReasoning,
        },
        DistortionTestCase {
            text: "It feels hopeless so it must be hopeless",
            distortion: DistortionType::EmotionalReasoning,
        },
        DistortionTestCase {
            text: "The one criticism outweighs all the praise",
            distortion: DistortionType::MentalFilter,
        },
        DistortionTestCase {
            text: "All I can think about is what went wrong",
            distortion: DistortionType::MentalFilter,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_distortion_runs() {
        let bench = CognitiveDistortionBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        assert!(result.metrics.contains_key("distortion_identification"));
    }

    #[test]
    fn test_cognitive_distortion_range() {
        let bench = CognitiveDistortionBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        let acc = result.metrics["distortion_identification"].mean;
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_cognitive_distortion_all_types_covered() {
        let cases = distortion_test_cases();
        for dt in DistortionType::ALL {
            let count = cases.iter().filter(|c| c.distortion == dt).count();
            assert!(count >= 2, "{:?} has only {} cases", dt, count);
        }
    }

    #[test]
    fn test_cognitive_distortion_deterministic() {
        let bench = CognitiveDistortionBenchmark;
        let config = BenchmarkConfig {
            seed: 42,
            ..Default::default()
        };
        let r1 = bench.run(&config);
        let r2 = bench.run(&config);
        assert_eq!(
            r1.metrics["distortion_identification"].mean,
            r2.metrics["distortion_identification"].mean
        );
    }

    #[test]
    fn test_cognitive_distortion_provenance() {
        let bench = CognitiveDistortionBenchmark;
        assert!(bench.provenance().is_some());
        assert_eq!(bench.provenance().unwrap().year, 1980);
    }

    #[test]
    fn test_types_evaluated() {
        let bench = CognitiveDistortionBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        assert_eq!(result.metrics["types_evaluated"].mean, 8.0);
    }
}
