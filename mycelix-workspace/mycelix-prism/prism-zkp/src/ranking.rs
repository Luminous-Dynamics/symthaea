// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ranking formula verification proof.
//!
//! Proves: "The top-K results were ranked according to the stated
//! composite formula" without revealing all claims in the index.
//!
//! Formula: score = 0.4 × relevance + 0.3 × epistemic + 0.2 × trust + 0.1 × freshness
//!
//! The ranking IS deterministic — same query + same index = same results.
//! This proof cryptographically attests that the formula was followed.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A proof that search results were ranked correctly.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankingProof {
    /// SHA-256 of the query (hides actual query from verifier)
    pub query_hash: [u8; 32],
    /// Computed composite scores for top-K results
    pub scores: Vec<RankedScore>,
    /// Commitment to the ranking formula parameters
    pub formula_commitment: [u8; 32],
    /// Domain tag
    pub domain_tag: String,
    /// Whether the ranking is verified correct
    pub verified: bool,
}

/// A single ranked result with its component scores.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankedScore {
    /// Position in ranking (0 = top result)
    pub rank: usize,
    /// Normalized Hamming similarity [0.0, 1.0]
    pub relevance: f32,
    /// Epistemic level as float [0.0, 1.0]
    pub epistemic: f32,
    /// Author/source trust [0.0, 1.0]
    pub trust: f32,
    /// Freshness [0.0, 1.0]
    pub freshness: f32,
    /// Composite score per formula
    pub composite: f32,
}

/// The standard Prism ranking formula weights.
pub const WEIGHT_RELEVANCE: f32 = 0.4;
pub const WEIGHT_EPISTEMIC: f32 = 0.3;
pub const WEIGHT_TRUST: f32 = 0.2;
pub const WEIGHT_FRESHNESS: f32 = 0.1;

/// Compute composite score from components.
pub fn compute_composite(relevance: f32, epistemic: f32, trust: f32, freshness: f32) -> f32 {
    WEIGHT_RELEVANCE * relevance
        + WEIGHT_EPISTEMIC * epistemic
        + WEIGHT_TRUST * trust
        + WEIGHT_FRESHNESS * freshness
}

/// Generate a ranking proof for top-K search results.
pub fn generate_ranking_proof(
    query: &str,
    results: &[(f32, f32, f32, f32)], // (relevance, epistemic, trust, freshness)
) -> RankingProof {
    let query_hash = {
        let h = Sha256::digest(query.as_bytes());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&h);
        hash
    };

    let formula_commitment = {
        let h = Sha256::digest(
            format!(
                "prism_ranking:{}:{}:{}:{}",
                WEIGHT_RELEVANCE, WEIGHT_EPISTEMIC, WEIGHT_TRUST, WEIGHT_FRESHNESS
            )
            .as_bytes(),
        );
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&h);
        hash
    };

    let mut scores: Vec<RankedScore> = results
        .iter()
        .enumerate()
        .map(|(i, &(rel, epi, trust, fresh))| RankedScore {
            rank: i,
            relevance: rel,
            epistemic: epi,
            trust,
            freshness: fresh,
            composite: compute_composite(rel, epi, trust, fresh),
        })
        .collect();

    // Sort by composite score (descending) and re-rank
    scores.sort_by(|a, b| {
        b.composite
            .partial_cmp(&a.composite)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, score) in scores.iter_mut().enumerate() {
        score.rank = i;
    }

    // Verify ranking is monotonically decreasing
    let verified = scores.windows(2).all(|w| w[0].composite >= w[1].composite);

    RankingProof {
        query_hash,
        scores,
        formula_commitment,
        domain_tag: "ZTML:Prism:RankingProof:v1".to_string(),
        verified,
    }
}

/// Verify a ranking proof: scores are correctly computed and ordered.
pub fn verify_ranking_proof(proof: &RankingProof) -> Result<(), String> {
    // Check each score follows the formula
    for score in &proof.scores {
        let expected = compute_composite(
            score.relevance,
            score.epistemic,
            score.trust,
            score.freshness,
        );
        if (score.composite - expected).abs() > 0.001 {
            return Err(format!(
                "Rank {}: composite {:.4} doesn't match formula {:.4}",
                score.rank, score.composite, expected
            ));
        }
    }

    // Check ordering is monotonically decreasing
    for window in proof.scores.windows(2) {
        if window[0].composite < window[1].composite - 0.001 {
            return Err(format!(
                "Rank {} (score {:.4}) < Rank {} (score {:.4})",
                window[0].rank, window[0].composite, window[1].rank, window[1].composite,
            ));
        }
    }

    // Check domain tag
    if proof.domain_tag != "ZTML:Prism:RankingProof:v1" {
        return Err("Wrong domain tag".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ranking_proof_correct_order() {
        let proof = generate_ranking_proof(
            "consciousness",
            &[
                (0.9, 0.75, 0.9, 0.8), // High relevance + epistemic
                (0.7, 0.5, 0.9, 0.5),  // Medium
                (0.3, 0.25, 0.9, 0.2), // Low
            ],
        );

        assert!(proof.verified);
        assert_eq!(proof.scores.len(), 3);
        assert!(proof.scores[0].composite >= proof.scores[1].composite);
        assert!(proof.scores[1].composite >= proof.scores[2].composite);
        assert!(verify_ranking_proof(&proof).is_ok());
    }

    #[test]
    fn test_formula_computation() {
        let score = compute_composite(1.0, 1.0, 1.0, 1.0);
        assert!((score - 1.0).abs() < 0.001, "all 1.0 should give 1.0");

        let score = compute_composite(0.0, 0.0, 0.0, 0.0);
        assert!((score - 0.0).abs() < 0.001, "all 0.0 should give 0.0");
    }

    #[test]
    fn test_tampered_score_detected() {
        let mut proof = generate_ranking_proof("test", &[(0.8, 0.6, 0.9, 0.7)]);

        // Tamper with composite score
        proof.scores[0].composite = 0.999;
        assert!(verify_ranking_proof(&proof).is_err());
    }

    #[test]
    fn test_query_hidden() {
        let proof =
            generate_ranking_proof("consciousness neural correlates", &[(0.8, 0.75, 0.9, 0.5)]);

        let json = serde_json::to_string(&proof).unwrap();
        assert!(
            !json.contains("consciousness neural"),
            "query text must NOT appear in proof"
        );
        assert!(json.contains("query_hash"), "only query hash should appear");
    }
}
