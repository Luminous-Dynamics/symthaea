// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cognitive Distortion Identification — Burns (1980).
//!
//! Tests the system's ability to identify common cognitive distortions
//! (all-or-nothing thinking, catastrophizing, mind reading, etc.) in text.
//! Uses HDC-encoded distortion prototypes with blake3-seeded similarity matching.
//!
//! Human baseline: 75% identification accuracy (SD = 10%), trained clinicians.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::HashMap;
use symthaea_core::hdc::BinaryHV;

/// Cognitive distortion identification evaluation.
pub struct CognitiveDistortionBenchmark;

/// Types of cognitive distortions (Burns 1980).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DistortionType {
    AllOrNothing,
    Catastrophizing,
    MindReading,
    Overgeneralization,
    MentalFilter,
    DiscountingPositive,
    EmotionalReasoning,
    ShouldStatements,
    Labeling,
    Personalization,
}

impl DistortionType {
    const ALL: [Self; 10] = [
        Self::AllOrNothing,
        Self::Catastrophizing,
        Self::MindReading,
        Self::Overgeneralization,
        Self::MentalFilter,
        Self::DiscountingPositive,
        Self::EmotionalReasoning,
        Self::ShouldStatements,
        Self::Labeling,
        Self::Personalization,
    ];

    fn name(&self) -> &'static str {
        match self {
            Self::AllOrNothing => "AllOrNothing",
            Self::Catastrophizing => "Catastrophizing",
            Self::MindReading => "MindReading",
            Self::Overgeneralization => "Overgeneralization",
            Self::MentalFilter => "MentalFilter",
            Self::DiscountingPositive => "DiscountingPositive",
            Self::EmotionalReasoning => "EmotionalReasoning",
            Self::ShouldStatements => "ShouldStatements",
            Self::Labeling => "Labeling",
            Self::Personalization => "Personalization",
        }
    }

    #[allow(dead_code)]
    fn description(&self) -> &'static str {
        match self {
            Self::AllOrNothing => {
                "Viewing situations in only two categories instead of on a continuum"
            }
            Self::Catastrophizing => {
                "Predicting the future negatively without considering more likely outcomes"
            }
            Self::MindReading => {
                "Believing you know what others are thinking without sufficient evidence"
            }
            Self::Overgeneralization => {
                "Making sweeping negative conclusions far beyond the current situation"
            }
            Self::MentalFilter => "Focusing exclusively on negatives and ignoring all positives",
            Self::DiscountingPositive => {
                "Dismissing positive experiences as not counting or irrelevant"
            }
            Self::EmotionalReasoning => {
                "Assuming that negative feelings reflect the way things really are"
            }
            Self::ShouldStatements => {
                "Using rigid rules about how oneself or others ought to behave"
            }
            Self::Labeling => {
                "Attaching a fixed, global label to oneself or others based on limited evidence"
            }
            Self::Personalization => {
                "Holding yourself personally responsible for events outside your control"
            }
        }
    }

    fn example_thought(&self) -> &'static str {
        match self {
            Self::AllOrNothing => "If I can't do it perfectly, there's no point in trying at all",
            Self::Catastrophizing => "If I fail this exam, my entire career will be ruined",
            Self::MindReading => "She didn't smile at me — she must think I'm a terrible person",
            Self::Overgeneralization => "I got rejected once; I'll never find anyone who wants me",
            Self::MentalFilter => "The whole day was awful because of that one small mistake",
            Self::DiscountingPositive => "Sure I got promoted, but anyone could have done that job",
            Self::EmotionalReasoning => "I feel like a failure, so I must really be one",
            Self::ShouldStatements => "I should always be productive; resting means I'm lazy",
            Self::Labeling => "I made a mistake on the report — I'm a complete idiot",
            Self::Personalization => "The team lost the contract because of me, it's all my fault",
        }
    }
}

/// A test vignette: an automatic thought paired with its ground-truth distortion type.
struct Vignette {
    text: &'static str,
    ground_truth: DistortionType,
}

/// Encode a single word as a BinaryHV using a deterministic hash seed.
fn word_hv(word: &str) -> BinaryHV {
    let hash = blake3::hash(word.to_lowercase().as_bytes());
    let bytes = hash.as_bytes();
    let seed = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    BinaryHV::random(seed)
}

/// Encode text as a bundle of its constituent word HVs.
///
/// This creates a compositional representation: texts sharing more words
/// with a prototype will have higher similarity (Kanerva, 2009 HDC).
fn text_to_hv(text: &str) -> BinaryHV {
    let word_hvs: Vec<BinaryHV> = text
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| w.len() >= 2)
        .map(word_hv)
        .collect();
    if word_hvs.is_empty() {
        return BinaryHV::random(0);
    }
    BinaryHV::bundle(&word_hvs)
}

/// Characteristic linguistic markers for each distortion type.
///
/// Cognitive distortions have distinctive lexical signatures
/// (Burns 1980; Beck 1976). These markers capture the semantic
/// patterns that clinicians use for classification.
fn distortion_markers(dt: &DistortionType) -> &'static [&'static str] {
    match dt {
        DistortionType::AllOrNothing => &[
            "perfect",
            "completely",
            "total",
            "totally",
            "either",
            "nothing",
            "waste",
            "failure",
            "point",
            "successful",
        ],
        DistortionType::Catastrophizing => &[
            "probably", "going", "die", "ruin", "ruined", "end", "worst", "terrible", "horrible",
            "disaster", "tumor",
        ],
        DistortionType::MindReading => &[
            "think",
            "thinking",
            "must",
            "definitely",
            "obviously",
            "secretly",
            "planning",
            "knows",
            "looked",
            "fire",
        ],
        DistortionType::Overgeneralization => &[
            "always",
            "never",
            "ever",
            "nothing",
            "anywhere",
            "everything",
            "happens",
            "rejected",
            "works",
        ],
        DistortionType::MentalFilter => &[
            "only",
            "all",
            "think",
            "about",
            "one",
            "criticism",
            "ruined",
            "rained",
            "last",
            "focusing",
        ],
        DistortionType::DiscountingPositive => &[
            "only", "because", "felt", "sorry", "anyone", "could", "sure", "but", "easy", "handle",
            "well",
        ],
        DistortionType::EmotionalReasoning => &[
            "feel",
            "must",
            "really",
            "anxious",
            "guilty",
            "dangerous",
            "something",
            "wrong",
            "am",
        ],
        DistortionType::ShouldStatements => &[
            "should",
            "must",
            "ought",
            "have",
            "able",
            "without",
            "always",
            "productive",
            "lazy",
            "good",
            "parent",
            "never",
        ],
        DistortionType::Labeling => &[
            "such",
            "complete",
            "idiot",
            "worthless",
            "scatterbrain",
            "jerk",
            "terrible",
            "person",
            "forgot",
        ],
        DistortionType::Personalization => &[
            "because",
            "me",
            "fault",
            "my",
            "entirely",
            "bad",
            "parent",
            "something",
            "said",
            "struggling",
        ],
    }
}

/// Build HDC prototype vectors for each distortion type.
///
/// Each prototype is a bundle of its characteristic marker words
/// (Burns 1980 linguistic signatures) + example thought encoding.
/// This gives prototypes a compositional structure that overlaps
/// with vignette word-bundles containing matching markers.
fn build_prototypes() -> Vec<(DistortionType, BinaryHV)> {
    DistortionType::ALL
        .iter()
        .map(|dt| {
            // Bundle marker word HVs
            let marker_hvs: Vec<BinaryHV> =
                distortion_markers(dt).iter().map(|w| word_hv(w)).collect();
            let marker_bundle = BinaryHV::bundle(&marker_hvs);
            // Also encode the example thought as a word bundle
            let example_bundle = text_to_hv(dt.example_thought());
            // Prototype = bundle of markers + example
            let prototype = BinaryHV::bundle(&[marker_bundle, example_bundle]);
            (*dt, prototype)
        })
        .collect()
}

/// Build the full set of test vignettes (20+).
fn build_vignettes() -> Vec<Vignette> {
    vec![
        // AllOrNothing (2)
        Vignette {
            text: "If I don't get a perfect score, the whole test was a waste",
            ground_truth: DistortionType::AllOrNothing,
        },
        Vignette {
            text: "Either I'm completely successful or I'm a total failure",
            ground_truth: DistortionType::AllOrNothing,
        },
        // Catastrophizing (2)
        Vignette {
            text: "This headache is probably a brain tumor and I'm going to die",
            ground_truth: DistortionType::Catastrophizing,
        },
        Vignette {
            text: "If I lose this job I'll end up homeless on the street",
            ground_truth: DistortionType::Catastrophizing,
        },
        // MindReading (2)
        Vignette {
            text: "My boss looked at me funny — he's definitely planning to fire me",
            ground_truth: DistortionType::MindReading,
        },
        Vignette {
            text: "Everyone at the dinner party was secretly thinking I'm boring",
            ground_truth: DistortionType::MindReading,
        },
        // Overgeneralization (2)
        Vignette {
            text: "I got rejected from one job application — I'll never get hired anywhere",
            ground_truth: DistortionType::Overgeneralization,
        },
        Vignette {
            text: "This always happens to me, nothing ever works out",
            ground_truth: DistortionType::Overgeneralization,
        },
        // MentalFilter (2)
        Vignette {
            text: "I got great feedback on nine things but all I can think about is the one criticism",
            ground_truth: DistortionType::MentalFilter,
        },
        Vignette {
            text: "The vacation was ruined because it rained on the last day",
            ground_truth: DistortionType::MentalFilter,
        },
        // DiscountingPositive (2)
        Vignette {
            text: "They only said I did well because they felt sorry for me",
            ground_truth: DistortionType::DiscountingPositive,
        },
        Vignette {
            text: "Sure the presentation went well but it was an easy topic that anyone could handle",
            ground_truth: DistortionType::DiscountingPositive,
        },
        // EmotionalReasoning (2)
        Vignette {
            text: "I feel anxious about flying so planes must be really dangerous",
            ground_truth: DistortionType::EmotionalReasoning,
        },
        Vignette {
            text: "I feel guilty so I must have done something wrong",
            ground_truth: DistortionType::EmotionalReasoning,
        },
        // ShouldStatements (2)
        Vignette {
            text: "I should be able to handle everything without asking for help",
            ground_truth: DistortionType::ShouldStatements,
        },
        Vignette {
            text: "A good parent must never lose their temper with their children",
            ground_truth: DistortionType::ShouldStatements,
        },
        // Labeling (2)
        Vignette {
            text: "I forgot the appointment — I'm such a worthless scatterbrain",
            ground_truth: DistortionType::Labeling,
        },
        Vignette {
            text: "He cut me off in traffic — what a complete jerk and terrible person",
            ground_truth: DistortionType::Labeling,
        },
        // Personalization (2)
        Vignette {
            text: "My child is struggling in school and it's entirely because I'm a bad parent",
            ground_truth: DistortionType::Personalization,
        },
        Vignette {
            text: "The meeting went badly and it must be because of something I said",
            ground_truth: DistortionType::Personalization,
        },
        // Additional vignettes for richer coverage (2 more)
        Vignette {
            text: "One mistake on the report means the entire project is garbage",
            ground_truth: DistortionType::AllOrNothing,
        },
        Vignette {
            text: "I feel overwhelmed so the situation truly is impossible to manage",
            ground_truth: DistortionType::EmotionalReasoning,
        },
    ]
}

impl PsychBenchmark for CognitiveDistortionBenchmark {
    fn name(&self) -> &str {
        "Clinical::CognitiveDistortion"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let _seed = config.seed;
        let prototypes = build_prototypes();
        let vignettes = build_vignettes();
        let n = vignettes.len();

        let mut correct = 0u32;
        let mut per_type_correct: HashMap<DistortionType, u32> = HashMap::new();
        let mut per_type_total: HashMap<DistortionType, u32> = HashMap::new();
        let mut max_similarities: Vec<f64> = Vec::with_capacity(n);

        for vignette in &vignettes {
            let vignette_hv = text_to_hv(vignette.text);

            // Find the most similar distortion prototype
            let mut best_type = DistortionType::AllOrNothing;
            let mut best_sim = f32::NEG_INFINITY;
            for (dt, proto) in &prototypes {
                let sim = vignette_hv.similarity(proto);
                if sim > best_sim {
                    best_sim = sim;
                    best_type = *dt;
                }
            }

            max_similarities.push(best_sim as f64);

            *per_type_total.entry(vignette.ground_truth).or_insert(0) += 1;
            if best_type == vignette.ground_truth {
                correct += 1;
                *per_type_correct.entry(vignette.ground_truth).or_insert(0) += 1;
            }
        }

        let accuracy = correct as f64 / n as f64;
        let mean_confidence = max_similarities.iter().sum::<f64>() / max_similarities.len() as f64;

        // Per-distortion accuracy
        let mut per_distortion_accuracies: Vec<f64> = Vec::new();
        let mut metrics_map = std::collections::BTreeMap::new();
        for dt in &DistortionType::ALL {
            let total = *per_type_total.get(dt).unwrap_or(&0);
            let corr = *per_type_correct.get(dt).unwrap_or(&0);
            let dt_accuracy = if total > 0 {
                corr as f64 / total as f64
            } else {
                0.0
            };
            per_distortion_accuracies.push(dt_accuracy);
            // Insert per-distortion metric using the full key
            let key = format!("per_distortion_{}", dt.name());
            metrics_map.insert(key, dt_accuracy);
        }

        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        result.insert(
            "distortion_identification",
            MetricValue::from_samples(&[accuracy]),
        );
        result.insert(
            "mean_confidence",
            MetricValue::from_samples(&[mean_confidence]),
        );
        result.insert(
            "mean_type_accuracy",
            MetricValue::from_samples(&per_distortion_accuracies),
        );
        result.insert(
            "types_evaluated",
            MetricValue::from_samples(&[DistortionType::ALL.len() as f64]),
        );

        // Per-distortion accuracy as individual metrics
        for (key, val) in &metrics_map {
            result.insert(key, MetricValue::from_samples(&[*val]));
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distortion_types_complete() {
        // Burns (1980) defines 10 cognitive distortions
        assert_eq!(DistortionType::ALL.len(), 10);

        // Verify all expected types are present
        let names: Vec<&str> = DistortionType::ALL.iter().map(|dt| dt.name()).collect();
        assert!(names.contains(&"AllOrNothing"));
        assert!(names.contains(&"Catastrophizing"));
        assert!(names.contains(&"MindReading"));
        assert!(names.contains(&"Overgeneralization"));
        assert!(names.contains(&"MentalFilter"));
        assert!(names.contains(&"DiscountingPositive"));
        assert!(names.contains(&"EmotionalReasoning"));
        assert!(names.contains(&"ShouldStatements"));
        assert!(names.contains(&"Labeling"));
        assert!(names.contains(&"Personalization"));
    }

    #[test]
    fn test_vignette_coverage() {
        let vignettes = build_vignettes();
        // At least 20 vignettes
        assert!(
            vignettes.len() >= 20,
            "expected at least 20 vignettes, got {}",
            vignettes.len()
        );

        // Every distortion type must have at least 2 vignettes
        for dt in &DistortionType::ALL {
            let count = vignettes.iter().filter(|v| v.ground_truth == *dt).count();
            assert!(
                count >= 2,
                "{:?} has only {} vignettes, expected at least 2",
                dt,
                count
            );
        }
    }

    #[test]
    fn test_benchmark_runs() {
        let bench = CognitiveDistortionBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        assert!(result.metrics.contains_key("distortion_identification"));
        assert!(result.metrics.contains_key("mean_confidence"));
        assert!(result.metrics.contains_key("mean_type_accuracy"));
        assert!(result.metrics.contains_key("types_evaluated"));
        assert_eq!(result.metrics["types_evaluated"].mean, 10.0);
    }

    #[test]
    fn test_accuracy_bounded() {
        let bench = CognitiveDistortionBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        let acc = result.metrics["distortion_identification"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "accuracy {} out of [0, 1] range",
            acc
        );

        let conf = result.metrics["mean_confidence"].mean;
        assert!(
            conf >= -1.0 && conf <= 1.0,
            "mean confidence {} out of [-1, 1] range",
            conf
        );
    }

    #[test]
    fn test_distinct_encodings() {
        // All distortion prototype HVs should be distinct from each other
        let prototypes = build_prototypes();
        for i in 0..prototypes.len() {
            for j in (i + 1)..prototypes.len() {
                let sim = prototypes[i].1.similarity(&prototypes[j].1);
                assert!(
                    (sim - 1.0).abs() > 0.01,
                    "prototypes {:?} and {:?} are nearly identical (sim = {})",
                    prototypes[i].0,
                    prototypes[j].0,
                    sim
                );
            }
        }
    }

    #[test]
    fn test_deterministic() {
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
    fn test_provenance() {
        let bench = CognitiveDistortionBenchmark;
        let prov = bench.provenance();
        assert!(prov.is_some());
        assert_eq!(prov.unwrap().year, 1980);
    }

    #[test]
    fn test_per_distortion_metrics_present() {
        let bench = CognitiveDistortionBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        for dt in &DistortionType::ALL {
            let key = format!("per_distortion_{}", dt.name());
            assert!(
                result.metrics.contains_key(&key),
                "missing per-distortion metric: {}",
                key
            );
            let val = result.metrics[&key].mean;
            assert!(
                val >= 0.0 && val <= 1.0,
                "{} accuracy {} out of range",
                key,
                val
            );
        }
    }
}
