// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Moral algebra for reproductive ethics.
//!
//! Encodes ethical tensions (autonomy vs survival vs dignity vs justice)
//! at 16,384D for reasoned weighing. Integrates with the Eight Harmonies
//! framework for value alignment assessment.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_types::Harmony;
#[cfg(test)]
use symthaea_types::N_HARMONIES;

use crate::governance::PopulationDecision;
use crate::types::Population;

/// Deterministic seeds for ethical dimension encoding.
const SEED_AUTONOMY: u64 = 0xA070_0000_0000_0001;
const SEED_SURVIVAL: u64 = 0x5000_0000_0000_0002;
const SEED_DIGNITY: u64 = 0xD100_0000_0000_0003;
const SEED_JUSTICE: u64 = 0x0005_71CE_0000_0004;

/// An ethical tension encoding the competing values in reproductive decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalTension {
    /// Name describing this ethical tension.
    pub name: String,
    /// Weight for individual reproductive autonomy.
    pub autonomy_weight: f64,
    /// Weight for species survival imperative.
    pub survival_weight: f64,
    /// Weight for human dignity (non-optimization of people).
    pub dignity_weight: f64,
    /// Weight for fair resource allocation / justice.
    pub justice_weight: f64,
    /// 16,384D HDC encoding of this tension.
    pub tension_hv: ContinuousHV,
}

impl EthicalTension {
    /// Create a new ethical tension with the given weights.
    pub fn new(name: &str, autonomy: f64, survival: f64, dignity: f64, justice: f64) -> Self {
        let tension_hv = encode_ethical_tension(autonomy, survival, dignity, justice);
        Self {
            name: name.to_string(),
            autonomy_weight: autonomy,
            survival_weight: survival,
            dignity_weight: dignity,
            justice_weight: justice,
            tension_hv,
        }
    }

    /// Total weight (should ideally sum to 1.0 for normalized tensions).
    pub fn total_weight(&self) -> f64 {
        self.autonomy_weight + self.survival_weight + self.dignity_weight + self.justice_weight
    }

    /// Get the dominant ethical value.
    pub fn dominant_value(&self) -> &'static str {
        let weights = [
            (self.autonomy_weight, "autonomy"),
            (self.survival_weight, "survival"),
            (self.dignity_weight, "dignity"),
            (self.justice_weight, "justice"),
        ];
        weights
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|w| w.1)
            .unwrap_or("unknown")
    }
}

/// Encode an ethical tension as a weighted bundle of ethical dimension HVs.
///
/// Each ethical dimension (autonomy, survival, dignity, justice) has a
/// deterministic basis HV. The tension is encoded as the weighted sum
/// of these basis HVs, normalized.
pub fn encode_ethical_tension(
    autonomy: f64,
    survival: f64,
    dignity: f64,
    justice: f64,
) -> ContinuousHV {
    let autonomy_hv = ContinuousHV::random(HDC_DIMENSION, SEED_AUTONOMY);
    let survival_hv = ContinuousHV::random(HDC_DIMENSION, SEED_SURVIVAL);
    let dignity_hv = ContinuousHV::random(HDC_DIMENSION, SEED_DIGNITY);
    let justice_hv = ContinuousHV::random(HDC_DIMENSION, SEED_JUSTICE);

    let weighted = ContinuousHV::weighted_bundle(
        &[&autonomy_hv, &survival_hv, &dignity_hv, &justice_hv],
        &[
            autonomy as f32,
            survival as f32,
            dignity as f32,
            justice as f32,
        ],
    );
    weighted.normalize()
}

/// Score an ethical tension against each of the Eight Harmonies.
///
/// Returns a vector of (Harmony, score) pairs where score is in [0, 1].
/// Higher score indicates greater alignment with that harmony.
pub fn harmony_alignment(tension: &EthicalTension) -> Vec<(Harmony, f64)> {
    Harmony::all()
        .iter()
        .map(|h| {
            let harmony_hv = ContinuousHV::random(HDC_DIMENSION, *h as u64);
            let score = tension.tension_hv.similarity(&harmony_hv) as f64;
            // Normalize from [-1, 1] to [0, 1]
            (*h, (score + 1.0) / 2.0)
        })
        .collect()
}

/// Evaluate the ethical score of a population decision.
///
/// Score range: [0.0, 1.0] where higher = more ethical.
///
/// Considers:
/// - Population urgency (small population = higher survival weight)
/// - Decision type (pairing = low stakes, override = high stakes)
/// - Balance of ethical dimensions
pub fn evaluate_decision_ethics(decision: &PopulationDecision, population: &Population) -> f64 {
    let n = population.size() as f64;

    // Urgency factor: small populations weight survival more heavily
    let urgency = if n <= 0.0 {
        1.0
    } else {
        (50.0 / n).min(1.0) // Maximum urgency at MVP threshold of 50
    };

    // Base ethical weights depending on decision type
    let (autonomy, survival, dignity, justice) = match decision {
        PopulationDecision::ApprovePairing(_) => {
            // Pairing: balance autonomy and genetic diversity
            (0.3 * (1.0 - urgency), 0.3 + 0.2 * urgency, 0.2, 0.2)
        }
        PopulationDecision::SetGrowthTarget { .. } => {
            // Growth: justice and survival focused
            (0.1, 0.4 + 0.2 * urgency, 0.2, 0.3 - 0.1 * urgency)
        }
        PopulationDecision::ApproveGeneticRescue(_) => {
            // Rescue: survival dominant but dignity matters
            (0.1, 0.5 + 0.2 * urgency, 0.2 - 0.1 * urgency, 0.2)
        }
        PopulationDecision::ModifyStrategy(_) => {
            // Strategy change: balanced consideration
            (0.25, 0.25, 0.25, 0.25)
        }
        PopulationDecision::OverrideAiRecommendation(_) => {
            // Override: autonomy and dignity paramount
            (0.35, 0.15, 0.35, 0.15)
        }
    };

    let tension = EthicalTension::new("decision_evaluation", autonomy, survival, dignity, justice);

    // Compute harmony alignment
    let alignments = harmony_alignment(&tension);
    let weighted_score: f64 = alignments
        .iter()
        .map(|(h, score)| score * h.base_weight() as f64)
        .sum();

    // Combine with urgency-adjusted base score
    let base_score = 0.5 + 0.2 * (1.0 - urgency); // Higher base for stable populations
    let final_score = (base_score + weighted_score) / 2.0;

    final_score.clamp(0.0, 1.0)
}

/// Compute an ethical balance score for a set of weights.
///
/// Returns how evenly balanced the ethical dimensions are.
/// 1.0 = perfectly balanced, 0.0 = completely dominated by one dimension.
pub fn ethical_balance(autonomy: f64, survival: f64, dignity: f64, justice: f64) -> f64 {
    let total = autonomy + survival + dignity + justice;
    if total <= 0.0 {
        return 0.0;
    }
    let weights = [
        autonomy / total,
        survival / total,
        dignity / total,
        justice / total,
    ];
    let ideal = 0.25; // Perfect balance = 1/4 each
    let deviation: f64 = weights.iter().map(|w| (w - ideal).powi(2)).sum();
    // Max deviation when one weight is 1.0: 3*(0.25)^2 + (0.75)^2 = 0.75
    let max_deviation = 0.75;
    1.0 - (deviation / max_deviation).min(1.0)
}

/// Check if a decision respects the minimum ethical threshold.
///
/// The threshold is 0.3 for routine decisions, higher for more consequential ones.
pub fn meets_ethical_threshold(decision: &PopulationDecision, population: &Population) -> bool {
    let score = evaluate_decision_ethics(decision, population);
    let threshold = match decision {
        PopulationDecision::ApprovePairing(_) => 0.3,
        PopulationDecision::SetGrowthTarget { .. } => 0.35,
        PopulationDecision::ApproveGeneticRescue(_) => 0.35,
        PopulationDecision::ModifyStrategy(_) => 0.4,
        PopulationDecision::OverrideAiRecommendation(_) => 0.45,
    };
    score >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        BiologicalSex, BreedingStrategy, GeneticRescuePlan, Individual, MatingPair,
    };

    fn make_test_population(n: usize) -> Population {
        let individuals: Vec<Individual> = (0..n)
            .map(|i| Individual {
                id: i as u64,
                sex: if i % 2 == 0 {
                    BiologicalSex::Female
                } else {
                    BiologicalSex::Male
                },
                genotypes: vec![],
                genome_hv: ContinuousHV::random(HDC_DIMENSION, i as u64 * 100),
                generation: 0,
                parent_ids: (None, None),
                fitness: 1.0,
            })
            .collect();

        Population {
            individuals,
            generation: 0,
            founding_size: n,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        }
    }

    #[test]
    fn test_encode_ethical_tension_deterministic() {
        let hv1 = encode_ethical_tension(0.5, 0.3, 0.1, 0.1);
        let hv2 = encode_ethical_tension(0.5, 0.3, 0.1, 0.1);
        let sim = hv1.similarity(&hv2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "same weights should give same HV: sim={sim}"
        );
    }

    #[test]
    fn test_encode_ethical_tension_different_weights() {
        let autonomy_heavy = encode_ethical_tension(0.8, 0.1, 0.05, 0.05);
        let survival_heavy = encode_ethical_tension(0.1, 0.8, 0.05, 0.05);
        let sim = autonomy_heavy.similarity(&survival_heavy);
        assert!(
            sim < 0.5,
            "different weight profiles should produce different HVs: sim={sim}"
        );
    }

    #[test]
    fn test_ethical_tension_creation() {
        let t = EthicalTension::new("test", 0.4, 0.3, 0.2, 0.1);
        assert_eq!(t.name, "test");
        assert!((t.total_weight() - 1.0).abs() < 1e-10);
        assert_eq!(t.dominant_value(), "autonomy");
    }

    #[test]
    fn test_ethical_tension_dominant_value() {
        let autonomy = EthicalTension::new("a", 0.5, 0.2, 0.2, 0.1);
        assert_eq!(autonomy.dominant_value(), "autonomy");

        let survival = EthicalTension::new("s", 0.1, 0.6, 0.2, 0.1);
        assert_eq!(survival.dominant_value(), "survival");

        let dignity = EthicalTension::new("d", 0.1, 0.1, 0.7, 0.1);
        assert_eq!(dignity.dominant_value(), "dignity");

        let justice = EthicalTension::new("j", 0.1, 0.1, 0.1, 0.7);
        assert_eq!(justice.dominant_value(), "justice");
    }

    #[test]
    fn test_harmony_alignment_returns_seven_scores() {
        let tension = EthicalTension::new("test", 0.25, 0.25, 0.25, 0.25);
        let alignments = harmony_alignment(&tension);
        assert_eq!(
            alignments.len(),
            N_HARMONIES,
            "should return N_HARMONIES harmony scores"
        );
    }

    #[test]
    fn test_harmony_alignment_scores_in_range() {
        let tension = EthicalTension::new("test", 0.3, 0.3, 0.2, 0.2);
        let alignments = harmony_alignment(&tension);
        for (h, score) in &alignments {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "harmony {:?} score should be in [0,1], got {score}",
                h
            );
        }
    }

    #[test]
    fn test_different_tensions_different_alignments() {
        let autonomy = EthicalTension::new("autonomy", 0.9, 0.03, 0.03, 0.04);
        let survival = EthicalTension::new("survival", 0.03, 0.9, 0.03, 0.04);

        let align_a = harmony_alignment(&autonomy);
        let align_s = harmony_alignment(&survival);

        // At least some harmony scores should differ
        let diff: f64 = align_a
            .iter()
            .zip(align_s.iter())
            .map(|((_, sa), (_, ss))| (sa - ss).abs())
            .sum();
        assert!(
            diff > 0.01,
            "different tensions should have different harmony alignments: diff={diff}"
        );
    }

    #[test]
    fn test_evaluate_decision_ethics_in_range() {
        let pop = make_test_population(100);
        let decision = PopulationDecision::ApprovePairing(MatingPair {
            parent_a: 0,
            parent_b: 1,
            kinship: 0.05,
            hla_complementarity: 0.8,
        });
        let score = evaluate_decision_ethics(&decision, &pop);
        assert!(
            score >= 0.0 && score <= 1.0,
            "ethical score should be in [0,1], got {score}"
        );
    }

    #[test]
    fn test_urgency_affects_ethics() {
        let small_pop = make_test_population(10);
        let large_pop = make_test_population(200);

        let decision = PopulationDecision::ApproveGeneticRescue(GeneticRescuePlan {
            source_population: "external".to_string(),
            num_migrants: 5,
            target_generation: 3,
            expected_heterozygosity_gain: 0.1,
        });

        let score_small = evaluate_decision_ethics(&decision, &small_pop);
        let score_large = evaluate_decision_ethics(&decision, &large_pop);

        // Both should be valid scores
        assert!(score_small >= 0.0 && score_small <= 1.0);
        assert!(score_large >= 0.0 && score_large <= 1.0);

        // They should differ due to urgency
        assert!(
            (score_small - score_large).abs() > 0.001,
            "urgency should affect score: small={score_small}, large={score_large}"
        );
    }

    #[test]
    fn test_ethical_balance_perfect() {
        let balance = ethical_balance(0.25, 0.25, 0.25, 0.25);
        assert!(
            (balance - 1.0).abs() < 1e-10,
            "equal weights = perfect balance: {balance}"
        );
    }

    #[test]
    fn test_ethical_balance_dominated() {
        let balance = ethical_balance(1.0, 0.0, 0.0, 0.0);
        assert!(
            balance < 0.1,
            "single-dimension should have low balance: {balance}"
        );
    }

    #[test]
    fn test_ethical_balance_partial() {
        let balance = ethical_balance(0.4, 0.3, 0.2, 0.1);
        assert!(
            balance > 0.5 && balance < 1.0,
            "partially balanced: {balance}"
        );
    }

    #[test]
    fn test_ethical_balance_zero_total() {
        assert_eq!(ethical_balance(0.0, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn test_meets_ethical_threshold() {
        let large_pop = make_test_population(200);

        // Pairing decision in large stable population should pass
        let decision = PopulationDecision::ApprovePairing(MatingPair {
            parent_a: 0,
            parent_b: 1,
            kinship: 0.05,
            hla_complementarity: 0.8,
        });
        assert!(
            meets_ethical_threshold(&decision, &large_pop),
            "routine pairing in stable pop should pass"
        );
    }

    #[test]
    fn test_evaluate_all_decision_types() {
        let pop = make_test_population(50);

        let decisions: Vec<PopulationDecision> = vec![
            PopulationDecision::ApprovePairing(MatingPair {
                parent_a: 0,
                parent_b: 1,
                kinship: 0.05,
                hla_complementarity: 0.8,
            }),
            PopulationDecision::SetGrowthTarget {
                size: 200,
                generation: 10,
            },
            PopulationDecision::ApproveGeneticRescue(GeneticRescuePlan {
                source_population: "ext".to_string(),
                num_migrants: 3,
                target_generation: 5,
                expected_heterozygosity_gain: 0.05,
            }),
            PopulationDecision::ModifyStrategy(BreedingStrategy::MinimumKinship),
            PopulationDecision::OverrideAiRecommendation("safety concern".to_string()),
        ];

        for decision in &decisions {
            let score = evaluate_decision_ethics(decision, &pop);
            assert!(
                score >= 0.0 && score <= 1.0,
                "score should be in [0,1] for {:?}: got {score}",
                std::mem::discriminant(decision)
            );
        }
    }

    #[test]
    fn test_encode_ethical_tension_normalized() {
        let hv = encode_ethical_tension(0.5, 0.3, 0.1, 0.1);
        let norm: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "encoded tension should be normalized: norm={norm}"
        );
    }

    #[test]
    fn test_ethical_tension_hv_dimension() {
        let t = EthicalTension::new("test", 0.25, 0.25, 0.25, 0.25);
        assert_eq!(t.tension_hv.dim(), HDC_DIMENSION);
    }
}
