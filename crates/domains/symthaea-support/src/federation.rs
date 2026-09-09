// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federation review — evaluates support resolutions for promotion candidacy.
//!
//! Quality graduation is not publication authority. This module no longer turns
//! a high-quality resolution directly into an outbound DHT update. It produces a
//! reviewable candidate; publication remains a distinct transition.

use crate::knowledge::KnowledgeManager;
use crate::knowledge_source::KnowledgeShareabilityV1;
use crate::technology::TechnologyIdentityV1;
use crate::types::*;

#[derive(Debug, Clone)]
pub struct PendingResolution {
    pub resolution_id: String,
    pub phi: f64,
    pub effectiveness_rating: u8,
    pub retrieval_count: u32,
    pub category: SupportCategory,
    pub pattern: String,
    /// Local/privacy sharing disposition established by the owning context.
    pub shareability: KnowledgeShareabilityV1,
    /// Exact technology context when established; absence means applicability is unknown.
    pub technology: Option<TechnologyIdentityV1>,
}

/// Review artifact. Possession of this value is not federation/publication authority.
#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgePromotionCandidateV1 {
    pub resolution_id: String,
    pub category: SupportCategory,
    pub pattern: String,
    pub phi: f64,
    pub effectiveness_rating: u8,
    pub retrieval_count: u32,
    pub technology: Option<TechnologyIdentityV1>,
}

#[derive(Debug, Clone)]
pub struct FederationResult {
    /// Candidates requiring an independent promotion/publication decision.
    pub promotion_candidates: Vec<KnowledgePromotionCandidateV1>,
    /// Legacy field retained for API compatibility. Automatic graduation never
    /// populates it; outbound publication must be explicit downstream.
    pub outbound_updates: Vec<CognitiveUpdateData>,
    pub evaluated: usize,
    /// Number meeting quality thresholds and permitted to enter promotion review.
    pub graduated: usize,
    pub deferred: usize,
    pub rejected: usize,
    /// High-quality items blocked because sharing permission was local-only or unknown.
    pub withheld: usize,
}

/// Evaluate pending resolutions for promotion review.
///
/// `Graduate` here means "quality threshold met". Only items already marked
/// `ReviewRequired` can become promotion candidates. `LocalOnly` and
/// `NotEstablished` are withheld regardless of quality.
pub fn check_graduations(
    manager: &KnowledgeManager,
    pending: &[PendingResolution],
) -> FederationResult {
    let mut candidates = Vec::new();
    let mut graduated = 0;
    let mut deferred = 0;
    let mut rejected = 0;
    let mut withheld = 0;

    for res in pending {
        let decision = manager.evaluate_for_graduation(
            res.phi,
            res.effectiveness_rating,
            res.retrieval_count,
        );
        match decision {
            GraduationDecision::Graduate => {
                if res.shareability == KnowledgeShareabilityV1::ReviewRequired {
                    candidates.push(KnowledgePromotionCandidateV1 {
                        resolution_id: res.resolution_id.clone(),
                        category: res.category.clone(),
                        pattern: res.pattern.clone(),
                        phi: res.phi,
                        effectiveness_rating: res.effectiveness_rating,
                        retrieval_count: res.retrieval_count,
                        technology: res.technology.clone(),
                    });
                    graduated += 1;
                } else {
                    withheld += 1;
                }
            }
            GraduationDecision::Defer(_) => deferred += 1,
            GraduationDecision::Reject(_) => rejected += 1,
        }
    }

    FederationResult {
        promotion_candidates: candidates,
        outbound_updates: Vec::new(),
        evaluated: pending.len(),
        graduated,
        deferred,
        rejected,
        withheld,
    }
}

/// Explicit low-level packaging step for an independently approved promotion.
/// Calling this function is not itself a proof that privacy, applicability,
/// provenance, or organizational publication policy has been satisfied.
pub fn prepare_outbound_update(
    encoding: Vec<u8>,
    phi: f64,
    category: SupportCategory,
    pattern: String,
) -> CognitiveUpdateData {
    CognitiveUpdateData {
        category,
        encoding,
        phi,
        resolution_pattern: pattern,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pending(phi: f64, rating: u8, retrievals: u32) -> PendingResolution {
        PendingResolution {
            resolution_id: format!("res-{phi}"),
            phi,
            effectiveness_rating: rating,
            retrieval_count: retrievals,
            category: SupportCategory::Network,
            pattern: "Restart DNS service".into(),
            shareability: KnowledgeShareabilityV1::ReviewRequired,
            technology: None,
        }
    }

    #[test]
    fn high_quality_resolution_becomes_review_candidate_not_outbound_update() {
        let manager = KnowledgeManager::new();
        let result = check_graduations(&manager, &[make_pending(0.8, 5, 5)]);
        assert_eq!(result.graduated, 1);
        assert_eq!(result.promotion_candidates.len(), 1);
        assert!(result.outbound_updates.is_empty());
    }

    #[test]
    fn local_only_high_quality_resolution_is_withheld() {
        let manager = KnowledgeManager::new();
        let mut item = make_pending(0.9, 5, 5);
        item.shareability = KnowledgeShareabilityV1::LocalOnly;
        let result = check_graduations(&manager, &[item]);
        assert_eq!(result.graduated, 0);
        assert_eq!(result.withheld, 1);
        assert!(result.promotion_candidates.is_empty());
    }

    #[test]
    fn unknown_shareability_is_withheld() {
        let manager = KnowledgeManager::new();
        let mut item = make_pending(0.9, 5, 5);
        item.shareability = KnowledgeShareabilityV1::NotEstablished;
        let result = check_graduations(&manager, &[item]);
        assert_eq!(result.withheld, 1);
    }

    #[test]
    fn low_quality_resolution_rejected() {
        let manager = KnowledgeManager::new();
        let result = check_graduations(&manager, &[make_pending(0.1, 1, 0)]);
        assert_eq!(result.rejected, 1);
        assert_eq!(result.graduated, 0);
    }

    #[test]
    fn medium_quality_deferred() {
        let manager = KnowledgeManager::new();
        let result = check_graduations(&manager, &[make_pending(0.4, 3, 2)]);
        assert_eq!(result.deferred, 1);
    }

    #[test]
    fn explicit_packaging_remains_separate() {
        let update = prepare_outbound_update(
            vec![1, 2, 3],
            0.75,
            SupportCategory::Holochain,
            "Clear lair cache".into(),
        );
        assert_eq!(update.encoding, vec![1, 2, 3]);
        assert_eq!(update.resolution_pattern, "Clear lair cache");
    }
}
