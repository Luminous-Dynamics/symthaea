//! DKG Confidence Score Calculation
//!
//! Computes confidence scores for verifiable triples based on 5 factors:
//!
//! 1. **Attestation count** (25%): More attesters = higher confidence, with diminishing returns
//! 2. **Reputation weighting** (30%): High-reputation attesters contribute more (reputation² voting)
//! 3. **Source quality** (20%): Peer-reviewed > institutional > commercial > social
//! 4. **Time decay** (10%): Older claims decay exponentially
//! 5. **Consistency** (15%): Contradicting claims reduce confidence
//!
//! # Genesis Simulation Results
//!
//! In the "Sky Color" scenario:
//! - Alice's "blue" claim with Bob's endorsement achieved confidence 0.6175
//! - This elevated it above the 0.5 baseline, defeating Mallory's "green" lie
//! - Social consensus (endorsement) elevated truth above unattested claims

use super::phi_integration::{ConsciousnessMetrics, PhiConfidenceFactors};
use super::{EpistemicType, VerifiableTriple, URI};
use serde::{Deserialize, Serialize};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

/// Individual confidence factor scores
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct ConfidenceFactors {
    /// Score from attestation count (0-1)
    pub attestation: f64,
    /// Score from reputation-weighted attestations (0-1)
    pub reputation: f64,
    /// Score from source quality (0-1)
    pub source_quality: f64,
    /// Score from time decay (0-1)
    pub time_decay: f64,
    /// Score from consistency check (0-1)
    pub consistency: f64,
    /// Optional: Phi-derived consciousness factor (0.75-1.2)
    pub phi_factor: Option<f64>,
}

impl ConfidenceFactors {
    /// Default weights for each factor
    /// Attestation: 25%, Reputation: 30%, Source: 20%, Time: 10%, Consistency: 15%
    pub const WEIGHTS: [f64; 5] = [0.25, 0.30, 0.20, 0.10, 0.15];

    /// Compute weighted average (without Phi factor - applied separately as multiplier)
    pub fn weighted_score(&self) -> f64 {
        let scores = [
            self.attestation,
            self.reputation,
            self.source_quality,
            self.time_decay,
            self.consistency,
        ];

        let weighted_sum: f64 = scores
            .iter()
            .zip(Self::WEIGHTS.iter())
            .map(|(s, w)| s * w)
            .sum();

        let weight_sum: f64 = Self::WEIGHTS.iter().sum();
        weighted_sum / weight_sum
    }

    /// Compute weighted score with Phi factor applied
    pub fn weighted_score_with_phi(&self) -> f64 {
        let base_score = self.weighted_score();
        match self.phi_factor {
            Some(phi) => (base_score * phi).clamp(0.0, 0.99),
            None => base_score,
        }
    }

    /// Compute with custom weights
    pub fn weighted_score_custom(&self, weights: &[f64; 5]) -> f64 {
        let scores = [
            self.attestation,
            self.reputation,
            self.source_quality,
            self.time_decay,
            self.consistency,
        ];

        let weighted_sum: f64 = scores.iter().zip(weights.iter()).map(|(s, w)| s * w).sum();

        let weight_sum: f64 = weights.iter().sum();
        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }
}

/// Complete confidence score with factors and metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct ConfidenceScore {
    /// Overall confidence (0-1)
    pub score: f64,
    /// Individual factor scores
    pub factors: ConfidenceFactors,
    /// When this score was computed
    pub computed_at: u64,
    /// Whether any factors were degraded due to data issues
    pub degraded: bool,
}

/// Thresholds for confidence classification
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct ConfidenceThresholds {
    /// Critical applications (medical, defense): 0.90+
    pub critical: f64,
    /// High-stakes applications (financial, legal): 0.75+
    pub high: f64,
    /// General knowledge: 0.60+
    pub medium: f64,
    /// Experimental/social data: 0.40+
    pub low: f64,
}

impl Default for ConfidenceThresholds {
    fn default() -> Self {
        Self {
            critical: 0.90,
            high: 0.75,
            medium: 0.60,
            low: 0.40,
        }
    }
}

/// Check if confidence meets a threshold level
pub fn meets_threshold(confidence: f64, level: &str) -> bool {
    let thresholds = ConfidenceThresholds::default();
    match level {
        "critical" => confidence >= thresholds.critical,
        "high" => confidence >= thresholds.high,
        "medium" => confidence >= thresholds.medium,
        "low" => confidence >= thresholds.low,
        _ => false,
    }
}

/// Input data for confidence calculation
pub struct ConfidenceInput<'a> {
    /// The triple being evaluated
    pub triple: &'a VerifiableTriple,
    /// Number of attestations received
    pub attestation_count: usize,
    /// Reputation scores of endorsing attesters
    pub attester_reputations: &'a [f64],
    /// Current timestamp for time decay calculation
    pub current_time: u64,
    /// Sum of reputation weights from challenging attesters
    pub contradiction_weights: f64,
    /// Optional: Consciousness metrics from Phi-lab for coherence boost
    pub consciousness: Option<&'a ConsciousnessMetrics>,
}

/// Calculate complete confidence score for a triple
pub fn calculate_confidence(input: &ConfidenceInput) -> ConfidenceScore {
    // Calculate Phi factor if consciousness metrics available
    let phi_factor = input.consciousness.map(|m| {
        let factors = PhiConfidenceFactors::from_metrics(m);
        factors.combined_multiplier
    });

    let factors = ConfidenceFactors {
        attestation: attestation_confidence(input.attestation_count),
        reputation: reputation_weighted_confidence(input.attester_reputations),
        source_quality: source_quality_confidence(&input.triple.sources),
        time_decay: time_decay_confidence(
            input.triple.created_at,
            input.current_time,
            get_half_life_days(&input.triple.domain),
        ),
        consistency: consistency_confidence(input.contradiction_weights),
        phi_factor,
    };

    let base_score = factors.weighted_score();

    // Apply domain-specific adjustments
    let adjusted_score =
        domain_adjusted_confidence(base_score, input.triple.domain.as_deref(), input.triple);

    // Apply epistemic type adjustments
    let epistemic_score =
        epistemic_adjusted_confidence(adjusted_score, &input.triple.epistemic_type);

    // Apply Phi boost if available
    let final_score = match phi_factor {
        Some(phi) => epistemic_score * phi,
        None => epistemic_score,
    };

    ConfidenceScore {
        score: final_score.clamp(0.0, 0.99),
        factors,
        computed_at: input.current_time,
        degraded: input.attestation_count == 0 || input.attester_reputations.is_empty(),
    }
}

/// Calculate confidence with explicit Phi metrics
///
/// Convenience function that always applies Phi-based consciousness boost.
/// Use this when you have consciousness metrics from Phi-lab.
pub fn calculate_confidence_with_phi(
    input: &ConfidenceInput,
    consciousness: &ConsciousnessMetrics,
) -> ConfidenceScore {
    let phi_factors = PhiConfidenceFactors::from_metrics(consciousness);

    let factors = ConfidenceFactors {
        attestation: attestation_confidence(input.attestation_count),
        reputation: reputation_weighted_confidence(input.attester_reputations),
        source_quality: source_quality_confidence(&input.triple.sources),
        time_decay: time_decay_confidence(
            input.triple.created_at,
            input.current_time,
            get_half_life_days(&input.triple.domain),
        ),
        consistency: consistency_confidence(input.contradiction_weights),
        phi_factor: Some(phi_factors.combined_multiplier),
    };

    let base_score = factors.weighted_score();

    // Apply all adjustments
    let domain_score =
        domain_adjusted_confidence(base_score, input.triple.domain.as_deref(), input.triple);

    let epistemic_score = epistemic_adjusted_confidence(domain_score, &input.triple.epistemic_type);

    // Apply consciousness boost
    let consciousness_score = epistemic_score * phi_factors.combined_multiplier;

    // Determine degraded status
    let is_degraded = input.attestation_count == 0
        || input.attester_reputations.is_empty()
        || consciousness.coherence_state.requires_approval();

    ConfidenceScore {
        score: consciousness_score.clamp(0.0, 0.99),
        factors,
        computed_at: input.current_time,
        degraded: is_degraded,
    }
}

/// Factor 1: Attestation count confidence
///
/// Logarithmic growth with diminishing returns:
/// - 0 attestations -> 0.10
/// - 1 attestation -> 0.50
/// - 10 attestations -> 0.73
/// - 100 attestations -> 0.96
pub fn attestation_confidence(count: usize) -> f64 {
    if count == 0 {
        return 0.10;
    }

    let base = 0.50;
    let growth = 0.10 * (count as f64).ln();
    (base + growth).clamp(0.10, 0.99)
}

/// Factor 2: Reputation-weighted attestation confidence
///
/// Average reputation of attesters, with sigmoid scaling.
/// This implements the reputation² voting principle from MATL.
pub fn reputation_weighted_confidence(reputations: &[f64]) -> f64 {
    if reputations.is_empty() {
        return 0.10;
    }

    let avg_rep: f64 = reputations.iter().sum::<f64>() / reputations.len() as f64;

    // Sigmoid scaling: maps [0,1] reputation to [0.1, 0.99] confidence
    sigmoid(avg_rep, 4.0, 0.5).clamp(0.10, 0.99)
}

/// Factor 3: Source quality confidence
///
/// Scores sources by domain reputation
pub fn source_quality_confidence(sources: &[URI]) -> f64 {
    if sources.is_empty() {
        return 0.50; // Neutral if no sources
    }

    let qualities: Vec<f64> = sources
        .iter()
        .map(|uri| domain_quality_score(uri.domain()))
        .collect();

    let avg_quality = qualities.iter().sum::<f64>() / qualities.len() as f64;

    // Penalty if all sources are low quality
    let all_low = qualities.iter().all(|&q| q < 0.4);
    if all_low {
        avg_quality * 0.8
    } else {
        avg_quality
    }
}

/// Get quality score for a domain
fn domain_quality_score(domain: &str) -> f64 {
    match domain {
        // Tier 1: Peer-reviewed academic (0.95-0.99)
        "arxiv.org" => 0.95,
        "nature.com" | "science.org" => 0.99,
        "ieee.org" | "acm.org" => 0.98,
        "ncbi.nlm.nih.gov" | "pubmed.gov" => 0.97,
        "springer.com" | "wiley.com" => 0.96,

        // Tier 2: Institutions (0.75-0.90)
        d if d.ends_with(".edu") => 0.85,
        d if d.ends_with(".gov") => 0.88,
        "github.com" => 0.72,
        "gitlab.com" => 0.72,
        "wikipedia.org" => 0.68,
        "stackoverflow.com" => 0.70,

        // Tier 3: Commercial/news (0.50-0.65)
        "medium.com" => 0.52,
        "substack.com" => 0.55,
        d if d.contains("news") => 0.58,
        "youtube.com" => 0.45,

        // Tier 4: Social media (0.25-0.40)
        "twitter.com" | "x.com" => 0.32,
        "reddit.com" => 0.38,
        "facebook.com" => 0.28,
        "tiktok.com" => 0.22,

        // Default
        _ => 0.50,
    }
}

/// Factor 4: Time decay confidence
///
/// Exponential decay based on age
pub fn time_decay_confidence(created_at: u64, current_time: u64, half_life_days: u64) -> f64 {
    if created_at == 0 || current_time <= created_at {
        return 1.0; // No decay for current or invalid timestamps
    }

    let age_secs = current_time.saturating_sub(created_at);
    let age_days = age_secs / 86400;

    // Exponential decay: confidence = 0.5^(age / half_life)
    let decay = 0.5_f64.powf(age_days as f64 / half_life_days as f64);

    decay.clamp(0.10, 1.0)
}

/// Get half-life in days for a domain
fn get_half_life_days(domain: &Option<String>) -> u64 {
    match domain.as_deref() {
        Some("finance") | Some("trading") | Some("market") => 7, // 1 week
        Some("news") | Some("current_events") => 30,             // 1 month
        Some("technology") | Some("software") => 180,            // 6 months
        Some("science") | Some("research") => 365,               // 1 year
        Some("history") | Some("mathematics") => 3650,           // 10 years
        _ => 365,                                                // Default: 1 year
    }
}

/// Factor 5: Consistency confidence
///
/// Reduces confidence based on contradicting claims
pub fn consistency_confidence(contradiction_weight: f64) -> f64 {
    if contradiction_weight <= 0.0 {
        return 0.95; // High confidence with no contradictions
    }

    // Reduce confidence proportionally to contradiction strength
    // weight=1.0 -> confidence=0.45
    // weight=0.5 -> confidence=0.70
    (0.95 - contradiction_weight * 0.50).clamp(0.10, 0.95)
}

/// Domain-specific confidence adjustments
fn domain_adjusted_confidence(base: f64, domain: Option<&str>, triple: &VerifiableTriple) -> f64 {
    match domain {
        Some("medical") | Some("health") | Some("pharma") => {
            // Require high standards for medical claims
            let has_peer_reviewed = triple.sources.iter().any(|uri| {
                matches!(
                    uri.domain(),
                    "nature.com" | "science.org" | "pubmed.gov" | "ncbi.nlm.nih.gov"
                )
            });

            if !has_peer_reviewed && base > 0.7 {
                base * 0.85 // Cap non-peer-reviewed medical claims
            } else {
                base
            }
        }

        Some("finance") | Some("trading") => {
            // Financial data is time-sensitive
            base // Time decay already handles this
        }

        Some("defense") | Some("security") => {
            // Require multiple attestations
            // (handled at validation level, not here)
            base
        }

        _ => base,
    }
}

/// Epistemic type adjustments
fn epistemic_adjusted_confidence(base: f64, epistemic_type: &EpistemicType) -> f64 {
    match epistemic_type {
        EpistemicType::Empirical => base, // No adjustment for empirical claims
        EpistemicType::Normative => base * 0.9, // Slight reduction for value claims
        EpistemicType::Metaphysical => base * 0.7, // Significant reduction for unfalsifiable
    }
}

/// Sigmoid function for smooth scaling
fn sigmoid(x: f64, steepness: f64, midpoint: f64) -> f64 {
    1.0 / (1.0 + (-steepness * (x - midpoint)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attestation_confidence() {
        assert!((attestation_confidence(0) - 0.10).abs() < 0.01);
        assert!(attestation_confidence(1) > 0.45);
        assert!(attestation_confidence(10) > 0.70);
        assert!(attestation_confidence(100) > 0.90);
        assert!(attestation_confidence(1000) <= 0.99);
    }

    #[test]
    fn test_reputation_weighted_confidence() {
        // High reputation attesters
        let high_reps = vec![0.9, 0.85, 0.95];
        assert!(reputation_weighted_confidence(&high_reps) > 0.80);

        // Low reputation attesters
        let low_reps = vec![0.2, 0.3, 0.25];
        assert!(reputation_weighted_confidence(&low_reps) < 0.50);

        // Empty attesters
        assert!((reputation_weighted_confidence(&[]) - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_source_quality_confidence() {
        // High quality sources
        let high = vec![
            URI::new("https://arxiv.org/paper/123"),
            URI::new("https://nature.com/article/456"),
        ];
        assert!(source_quality_confidence(&high) > 0.90);

        // Low quality sources
        let low = vec![
            URI::new("https://twitter.com/post/123"),
            URI::new("https://facebook.com/post/456"),
        ];
        assert!(source_quality_confidence(&low) < 0.35);

        // No sources
        assert!((source_quality_confidence(&[]) - 0.50).abs() < 0.01);
    }

    #[test]
    fn test_time_decay_confidence() {
        let now = 1700000000u64;
        let one_day_ago = now - 86400;
        let one_year_ago = now - 365 * 86400;
        let ten_years_ago = now - 3650 * 86400;

        // Recent data: high confidence
        assert!(time_decay_confidence(one_day_ago, now, 365) > 0.99);

        // 1 year old: ~50% confidence (half-life)
        let one_year_conf = time_decay_confidence(one_year_ago, now, 365);
        assert!(one_year_conf > 0.45 && one_year_conf < 0.55);

        // 10 years old: very low confidence
        assert!(time_decay_confidence(ten_years_ago, now, 365) < 0.15);
    }

    #[test]
    fn test_consistency_confidence() {
        assert!(consistency_confidence(0.0) > 0.90);
        assert!(consistency_confidence(0.5) > 0.65 && consistency_confidence(0.5) < 0.75);
        assert!(consistency_confidence(1.0) > 0.40 && consistency_confidence(1.0) < 0.50);
    }

    #[test]
    fn test_confidence_factors_weighted() {
        let factors = ConfidenceFactors {
            attestation: 0.80,
            reputation: 0.90,
            source_quality: 0.95,
            time_decay: 0.99,
            consistency: 0.95,
            phi_factor: None,
        };

        let score = factors.weighted_score();
        assert!(score > 0.85 && score < 0.95);
    }

    #[test]
    fn test_confidence_factors_with_phi() {
        let factors = ConfidenceFactors {
            attestation: 0.80,
            reputation: 0.90,
            source_quality: 0.95,
            time_decay: 0.99,
            consistency: 0.95,
            phi_factor: Some(1.15), // Coherent agent boost
        };

        let base_score = factors.weighted_score();
        let phi_score = factors.weighted_score_with_phi();

        // Phi should boost the score
        assert!(phi_score > base_score);
        assert!(phi_score <= 0.99);
    }

    #[test]
    fn test_phi_degraded_penalty() {
        let factors = ConfidenceFactors {
            attestation: 0.80,
            reputation: 0.90,
            source_quality: 0.95,
            time_decay: 0.99,
            consistency: 0.95,
            phi_factor: Some(0.6), // Degraded state penalty
        };

        let base_score = factors.weighted_score();
        let phi_score = factors.weighted_score_with_phi();

        // Phi penalty should reduce the score
        assert!(phi_score < base_score);
    }

    #[test]
    fn test_full_confidence_calculation() {
        let triple = VerifiableTriple::new(
            "gradient:abc123",
            "pogq:quality_score",
            super::super::TripleValue::Float(0.95),
        )
        .with_domain("federated_learning")
        .with_source("https://arxiv.org/paper/123")
        .with_timestamp(1700000000 - 86400); // 1 day old

        let input = ConfidenceInput {
            triple: &triple,
            attestation_count: 5,
            attester_reputations: &[0.9, 0.85, 0.88, 0.92, 0.87],
            current_time: 1700000000,
            contradiction_weights: 0.0,
            consciousness: None,
        };

        let result = calculate_confidence(&input);

        assert!(result.score > 0.75);
        assert!(!result.degraded);
        assert!(meets_threshold(result.score, "high"));
    }

    #[test]
    fn test_confidence_with_phi_boost() {
        let triple = VerifiableTriple::new(
            "gradient:abc123",
            "pogq:quality_score",
            super::super::TripleValue::Float(0.95),
        )
        .with_domain("federated_learning")
        .with_source("https://arxiv.org/paper/123")
        .with_timestamp(1700000000 - 86400);

        let metrics = ConsciousnessMetrics::new(0.8, 3, 1700000000); // Coherent

        let input = ConfidenceInput {
            triple: &triple,
            attestation_count: 5,
            attester_reputations: &[0.9, 0.85, 0.88, 0.92, 0.87],
            current_time: 1700000000,
            contradiction_weights: 0.0,
            consciousness: Some(&metrics),
        };

        let with_phi = calculate_confidence(&input);

        // Now without Phi
        let input_no_phi = ConfidenceInput {
            triple: &triple,
            attestation_count: 5,
            attester_reputations: &[0.9, 0.85, 0.88, 0.92, 0.87],
            current_time: 1700000000,
            contradiction_weights: 0.0,
            consciousness: None,
        };

        let without_phi = calculate_confidence(&input_no_phi);

        // Coherent Phi should boost confidence
        assert!(
            with_phi.score > without_phi.score,
            "Coherent Phi ({:.3}) should boost above no-Phi ({:.3})",
            with_phi.score,
            without_phi.score
        );
    }

    #[test]
    fn test_meets_threshold() {
        assert!(meets_threshold(0.95, "critical"));
        assert!(meets_threshold(0.80, "high"));
        assert!(!meets_threshold(0.70, "high"));
        assert!(meets_threshold(0.65, "medium"));
        assert!(meets_threshold(0.45, "low"));
        assert!(!meets_threshold(0.30, "low"));
    }

    #[test]
    fn test_epistemic_adjustments() {
        let base = 0.80;

        let empirical = epistemic_adjusted_confidence(base, &EpistemicType::Empirical);
        let normative = epistemic_adjusted_confidence(base, &EpistemicType::Normative);
        let metaphysical = epistemic_adjusted_confidence(base, &EpistemicType::Metaphysical);

        assert!((empirical - 0.80).abs() < 0.01);
        assert!((normative - 0.72).abs() < 0.01);
        assert!((metaphysical - 0.56).abs() < 0.01);
    }

    #[test]
    fn test_genesis_simulation_scenario() {
        // Alice's claim: "sky is blue" with Bob's endorsement
        let alice_triple = VerifiableTriple::new(
            "sky",
            "color",
            super::super::TripleValue::String("blue".into()),
        )
        .with_epistemic_type(EpistemicType::Empirical)
        .with_timestamp(1700000000);

        let alice_input = ConfidenceInput {
            triple: &alice_triple,
            attestation_count: 1,         // Bob's endorsement
            attester_reputations: &[0.5], // Bob's neutral reputation
            current_time: 1700000005,     // 5 seconds later
            contradiction_weights: 0.0,
            consciousness: None,
        };

        let alice_score = calculate_confidence(&alice_input);

        // Mallory's claim: "sky is green" with no endorsements
        let mallory_triple = VerifiableTriple::new(
            "sky",
            "color",
            super::super::TripleValue::String("green".into()),
        )
        .with_epistemic_type(EpistemicType::Empirical)
        .with_timestamp(1700000000);

        let mallory_input = ConfidenceInput {
            triple: &mallory_triple,
            attestation_count: 0,
            attester_reputations: &[],
            current_time: 1700000005,
            contradiction_weights: 0.0,
            consciousness: None,
        };

        let mallory_score = calculate_confidence(&mallory_input);

        // Truth Engine verdict: Blue should win
        assert!(
            alice_score.score > mallory_score.score,
            "Alice's endorsed claim ({}) should beat Mallory's unattested claim ({})",
            alice_score.score,
            mallory_score.score
        );

        // Alice's score should be above baseline (0.5)
        assert!(
            alice_score.score > 0.5,
            "Endorsed claim confidence ({}) should be above baseline (0.5)",
            alice_score.score
        );
    }

    #[test]
    fn test_genesis_with_phi_consciousness() {
        // Alice is a coherent agent (high Phi)
        let alice_metrics = ConsciousnessMetrics::new(0.85, 4, 1700000000);

        // Mallory is an incoherent agent (low Phi)
        let mallory_metrics = ConsciousnessMetrics::new(0.2, 0, 1700000000);

        let alice_triple = VerifiableTriple::new(
            "knowledge",
            "claim",
            super::super::TripleValue::String("truth".into()),
        )
        .with_timestamp(1700000000);

        let mallory_triple = VerifiableTriple::new(
            "knowledge",
            "claim",
            super::super::TripleValue::String("falsehood".into()),
        )
        .with_timestamp(1700000000);

        // Both have same attestation count for fair comparison
        let alice_input = ConfidenceInput {
            triple: &alice_triple,
            attestation_count: 2,
            attester_reputations: &[0.6, 0.6],
            current_time: 1700000005,
            contradiction_weights: 0.0,
            consciousness: Some(&alice_metrics),
        };

        let mallory_input = ConfidenceInput {
            triple: &mallory_triple,
            attestation_count: 2,
            attester_reputations: &[0.6, 0.6],
            current_time: 1700000005,
            contradiction_weights: 0.0,
            consciousness: Some(&mallory_metrics),
        };

        let alice_score = calculate_confidence(&alice_input);
        let mallory_score = calculate_confidence(&mallory_input);

        // Coherent Alice should beat incoherent Mallory
        assert!(
            alice_score.score > mallory_score.score,
            "Coherent agent ({:.3}) should have higher confidence than incoherent ({:.3})",
            alice_score.score,
            mallory_score.score
        );
    }
}
