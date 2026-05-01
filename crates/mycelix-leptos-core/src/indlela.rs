// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Indlela Design System — botanical growth visualization.
//!
//! Named after isiZulu "izindlela" (pathways). Generalizes the organic
//! growth metaphor from EduNet's Knowledge Garden to all clusters:
//! seed → sprout → sapling → tree.
//!
//! Patterns:
//! - `GrowthStage`: maps progress (0.0-1.0) to visual representation
//! - `community_warmth`: trust/engagement metric for a node
//! - `knowledge_freshness`: Ebbinghaus decay for temporal data

/// Growth stage of an entity (project, claim, skill, etc.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrowthStage {
    /// Not started, dormant (progress < 0.1)
    Seed,
    /// Early growth, emerging (0.1-0.4)
    Sprout,
    /// Maturing, substantial (0.4-0.75)
    Sapling,
    /// Fully grown, established (> 0.75)
    Tree,
}

impl GrowthStage {
    /// Determine growth stage from progress (0.0-1.0).
    pub fn from_progress(progress: f64) -> Self {
        if progress < 0.1 {
            Self::Seed
        } else if progress < 0.4 {
            Self::Sprout
        } else if progress < 0.75 {
            Self::Sapling
        } else {
            Self::Tree
        }
    }

    /// CSS class for styling.
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Seed => "growth-seed",
            Self::Sprout => "growth-sprout",
            Self::Sapling => "growth-sapling",
            Self::Tree => "growth-tree",
        }
    }

    /// Display radius for SVG visualization (px).
    pub fn radius(&self) -> f64 {
        match self {
            Self::Seed => 8.0,
            Self::Sprout => 12.0,
            Self::Sapling => 14.0,
            Self::Tree => 16.0,
        }
    }

    /// Display opacity.
    pub fn opacity(&self) -> f64 {
        match self {
            Self::Seed => 0.45,
            Self::Sprout => 0.7,
            Self::Sapling => 0.85,
            Self::Tree => 1.0,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Seed => "Seed",
            Self::Sprout => "Sprout",
            Self::Sapling => "Sapling",
            Self::Tree => "Tree",
        }
    }
}

/// Compute community warmth/trust metric for a node.
///
/// Returns 0.0 (new/isolated) to 1.0 (deeply embedded).
/// Uses a simple hash-based deterministic value for mock mode,
/// ready to be replaced with real conductor data.
pub fn community_warmth(id: &str, peer_count: Option<u32>) -> f64 {
    if let Some(peers) = peer_count {
        // Real data: sigmoid mapping of peer count
        let x = peers as f64;
        1.0 / (1.0 + (-0.1 * (x - 10.0)).exp())
    } else {
        // Mock: deterministic hash for stable visual warmth
        let hash = id
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        (hash % 100) as f64 / 100.0
    }
}

/// Compute knowledge freshness based on Ebbinghaus forgetting curve.
///
/// Returns 0.0 (fresh) to 1.0 (completely stale).
/// 50% forgotten after 7 days.
pub fn knowledge_freshness(last_reviewed_ms: Option<f64>, now_ms: f64) -> f64 {
    match last_reviewed_ms {
        Some(reviewed) => {
            let days_since = (now_ms - reviewed) / 86_400_000.0;
            if days_since <= 0.0 {
                return 0.0;
            }
            // Ebbinghaus: retention = e^(-t/S), where S=10 days for 50% at 7 days
            let decay = 1.0 - (-days_since / 10.0_f64).exp();
            decay.clamp(0.0, 1.0)
        }
        None => 1.0, // Never reviewed = fully stale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_stages() {
        assert_eq!(GrowthStage::from_progress(0.0), GrowthStage::Seed);
        assert_eq!(GrowthStage::from_progress(0.25), GrowthStage::Sprout);
        assert_eq!(GrowthStage::from_progress(0.5), GrowthStage::Sapling);
        assert_eq!(GrowthStage::from_progress(0.9), GrowthStage::Tree);
    }

    #[test]
    fn warmth_deterministic() {
        let w1 = community_warmth("test-id-1", None);
        let w2 = community_warmth("test-id-1", None);
        assert_eq!(w1, w2); // Same ID = same warmth
    }

    #[test]
    fn freshness_decay() {
        let now = 1_000_000.0;
        assert_eq!(knowledge_freshness(Some(now), now), 0.0); // Just reviewed
        assert!(knowledge_freshness(Some(now - 604_800_000.0), now) > 0.4); // 7 days ago
        assert_eq!(knowledge_freshness(None, now), 1.0); // Never reviewed
    }
}
