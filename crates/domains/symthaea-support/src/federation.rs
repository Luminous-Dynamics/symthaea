// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federation Loop — Knowledge graduation and outbound cognitive update preparation
//!
//! Evaluates pending resolutions against graduation criteria and packages
//! high-quality knowledge for DHT publication as cognitive updates.

use crate::knowledge::KnowledgeManager;
use crate::types::*;

/// A pending resolution awaiting graduation evaluation.
#[derive(Debug, Clone)]
pub struct PendingResolution {
    pub resolution_id: String,
    pub phi: f64,
    pub effectiveness_rating: u8,
    pub retrieval_count: u32,
    pub category: SupportCategory,
    pub pattern: String,
}

/// Result of a graduation check across pending resolutions.
#[derive(Debug, Clone)]
pub struct FederationResult {
    /// Cognitive updates ready for DHT publication.
    pub outbound_updates: Vec<CognitiveUpdateData>,
    /// Number of resolutions evaluated.
    pub evaluated: usize,
    /// Number that graduated.
    pub graduated: usize,
    /// Number deferred for later evaluation.
    pub deferred: usize,
}

/// Evaluate pending resolutions against graduation criteria.
///
/// Resolutions that meet the graduation threshold (high phi, good effectiveness
/// rating, sufficient retrievals) are packaged as cognitive updates for DHT
/// federation. Deferred resolutions remain in the pending queue for re-evaluation.
pub fn check_graduations(
    manager: &KnowledgeManager,
    pending: &[PendingResolution],
) -> FederationResult {
    let mut outbound = Vec::new();
    let mut graduated = 0;
    let mut deferred = 0;

    for res in pending {
        let decision =
            manager.evaluate_for_graduation(res.phi, res.effectiveness_rating, res.retrieval_count);

        match decision {
            GraduationDecision::Graduate => {
                let update = prepare_outbound_update(
                    Vec::new(), // encoding populated by caller from HDC space
                    res.phi,
                    res.category.clone(),
                    res.pattern.clone(),
                );
                outbound.push(update);
                graduated += 1;
            }
            GraduationDecision::Defer(_) => {
                deferred += 1;
            }
            GraduationDecision::Reject(_) => {
                // Rejected — drop silently
            }
        }
    }

    FederationResult {
        outbound_updates: outbound,
        evaluated: pending.len(),
        graduated,
        deferred,
    }
}

/// Package a graduated resolution as a cognitive update for DHT publication.
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
            resolution_id: format!("res-{}", phi),
            phi,
            effectiveness_rating: rating,
            retrieval_count: retrievals,
            category: SupportCategory::Network,
            pattern: "Restart DNS service".to_string(),
        }
    }

    #[test]
    fn high_quality_resolution_graduates() {
        let manager = KnowledgeManager::new();
        let pending = vec![make_pending(0.8, 5, 5)];
        let result = check_graduations(&manager, &pending);
        assert_eq!(result.graduated, 1);
        assert_eq!(result.outbound_updates.len(), 1);
        assert!((result.outbound_updates[0].phi - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn low_quality_resolution_rejected() {
        let manager = KnowledgeManager::new();
        let pending = vec![make_pending(0.1, 1, 0)];
        let result = check_graduations(&manager, &pending);
        assert_eq!(result.graduated, 0);
        assert_eq!(result.deferred, 0);
        assert!(result.outbound_updates.is_empty());
    }

    #[test]
    fn medium_quality_deferred() {
        let manager = KnowledgeManager::new();
        let pending = vec![make_pending(0.4, 3, 2)];
        let result = check_graduations(&manager, &pending);
        assert_eq!(result.graduated, 0);
        assert_eq!(result.deferred, 1);
    }

    #[test]
    fn mixed_batch_processes_correctly() {
        let manager = KnowledgeManager::new();
        let pending = vec![
            make_pending(0.8, 5, 5), // graduate
            make_pending(0.4, 3, 2), // defer
            make_pending(0.1, 1, 0), // reject
            make_pending(0.9, 4, 3), // graduate
        ];
        let result = check_graduations(&manager, &pending);
        assert_eq!(result.evaluated, 4);
        assert_eq!(result.graduated, 2);
        assert_eq!(result.deferred, 1);
        assert_eq!(result.outbound_updates.len(), 2);
    }

    #[test]
    fn empty_pending_returns_empty() {
        let manager = KnowledgeManager::new();
        let result = check_graduations(&manager, &[]);
        assert_eq!(result.evaluated, 0);
        assert_eq!(result.graduated, 0);
        assert!(result.outbound_updates.is_empty());
    }

    #[test]
    fn prepare_outbound_update_sets_fields() {
        let update = prepare_outbound_update(
            vec![1, 2, 3],
            0.75,
            SupportCategory::Holochain,
            "Clear lair cache".to_string(),
        );
        assert_eq!(update.encoding, vec![1, 2, 3]);
        assert!((update.phi - 0.75).abs() < f64::EPSILON);
        assert_eq!(update.category, SupportCategory::Holochain);
        assert_eq!(update.resolution_pattern, "Clear lair cache");
    }
}
