// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim Evolution & Versioning
//!
//! Implements claim evolution tracking from Constitution Schema v2.0.
//! Claims can evolve through amendments, corrections, and retractions,
//! forming an immutable evolution chain.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type of evolution from one claim version to another
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvolutionType {
    /// Original claim (no parent)
    Genesis,
    /// Minor update that doesn't change core findings
    Amendment,
    /// Correction of errors (methodology, data, analysis)
    Correction,
    /// Full retraction (claim is withdrawn)
    Retraction,
    /// Superseded by newer research
    Supersession,
    /// Extension with new data/findings
    Extension,
    /// Merge of multiple claims into one
    Consolidation,
}

impl EvolutionType {
    pub fn description(&self) -> &'static str {
        match self {
            Self::Genesis => "Original claim",
            Self::Amendment => "Minor update preserving core findings",
            Self::Correction => "Correction of errors",
            Self::Retraction => "Full retraction - claim withdrawn",
            Self::Supersession => "Superseded by newer research",
            Self::Extension => "Extended with new data/findings",
            Self::Consolidation => "Consolidated from multiple claims",
        }
    }

    /// Does this evolution type invalidate the parent?
    pub fn invalidates_parent(&self) -> bool {
        matches!(self, Self::Correction | Self::Retraction | Self::Supersession)
    }
}

/// Tracks the evolution of a claim through versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimEvolution {
    /// Current version number (starts at 1)
    pub version: u32,
    /// ID of the parent claim (None for Genesis)
    pub parent_claim_id: Option<Uuid>,
    /// Type of evolution from parent
    pub evolution_type: EvolutionType,
    /// Human-readable changelog
    pub changelog: String,
    /// Full evolution chain (oldest to newest)
    pub evolution_chain: Vec<Uuid>,
    /// IDs of claims merged in consolidation
    pub merged_from: Vec<Uuid>,
    /// Timestamp of this evolution
    pub evolved_at: DateTime<Utc>,
    /// Who initiated this evolution
    pub evolved_by: String,
}

impl ClaimEvolution {
    /// Create a genesis (original) claim evolution
    pub fn genesis(claim_id: Uuid, creator: String) -> Self {
        Self {
            version: 1,
            parent_claim_id: None,
            evolution_type: EvolutionType::Genesis,
            changelog: "Initial version".to_string(),
            evolution_chain: vec![claim_id],
            merged_from: Vec::new(),
            evolved_at: Utc::now(),
            evolved_by: creator,
        }
    }

    /// Create an evolution from a parent claim
    pub fn evolve(
        new_claim_id: Uuid,
        parent: &ClaimEvolution,
        parent_id: Uuid,
        evolution_type: EvolutionType,
        changelog: String,
        evolved_by: String,
    ) -> Self {
        let mut chain = parent.evolution_chain.clone();
        chain.push(new_claim_id);

        Self {
            version: parent.version + 1,
            parent_claim_id: Some(parent_id),
            evolution_type,
            changelog,
            evolution_chain: chain,
            merged_from: Vec::new(),
            evolved_at: Utc::now(),
            evolved_by,
        }
    }

    /// Create a consolidation from multiple claims
    pub fn consolidate(
        new_claim_id: Uuid,
        source_claims: Vec<Uuid>,
        changelog: String,
        consolidated_by: String,
    ) -> Self {
        Self {
            version: 1,
            parent_claim_id: None,
            evolution_type: EvolutionType::Consolidation,
            changelog,
            evolution_chain: vec![new_claim_id],
            merged_from: source_claims,
            evolved_at: Utc::now(),
            evolved_by: consolidated_by,
        }
    }

    /// Check if this is the original version
    pub fn is_genesis(&self) -> bool {
        self.evolution_type == EvolutionType::Genesis
    }

    /// Check if this claim has been retracted
    pub fn is_retracted(&self) -> bool {
        self.evolution_type == EvolutionType::Retraction
    }

    /// Get the depth of the evolution chain
    pub fn chain_depth(&self) -> usize {
        self.evolution_chain.len()
    }
}

impl Default for ClaimEvolution {
    fn default() -> Self {
        Self::genesis(Uuid::new_v4(), "unknown".to_string())
    }
}

/// Status of a claim in its evolution lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimStatus {
    /// Active and valid
    Active,
    /// Has been superseded by a newer version
    Superseded,
    /// Has been corrected (use newer version)
    Corrected,
    /// Has been retracted (invalid)
    Retracted,
    /// Archived (no longer actively maintained)
    Archived,
}

impl ClaimStatus {
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Active | Self::Archived)
    }

    pub fn should_show_warning(&self) -> bool {
        matches!(self, Self::Superseded | Self::Corrected | Self::Retracted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_evolution() {
        let claim_id = Uuid::new_v4();
        let evo = ClaimEvolution::genesis(claim_id, "creator@test.com".to_string());

        assert_eq!(evo.version, 1);
        assert!(evo.is_genesis());
        assert!(evo.parent_claim_id.is_none());
        assert_eq!(evo.evolution_chain.len(), 1);
    }

    #[test]
    fn test_evolve_claim() {
        let parent_id = Uuid::new_v4();
        let parent = ClaimEvolution::genesis(parent_id, "creator@test.com".to_string());

        let new_id = Uuid::new_v4();
        let evolved = ClaimEvolution::evolve(
            new_id,
            &parent,
            parent_id,
            EvolutionType::Amendment,
            "Fixed typo in methodology".to_string(),
            "editor@test.com".to_string(),
        );

        assert_eq!(evolved.version, 2);
        assert_eq!(evolved.parent_claim_id, Some(parent_id));
        assert_eq!(evolved.evolution_chain.len(), 2);
        assert!(!evolved.evolution_type.invalidates_parent());
    }

    #[test]
    fn test_correction_invalidates() {
        assert!(EvolutionType::Correction.invalidates_parent());
        assert!(EvolutionType::Retraction.invalidates_parent());
        assert!(!EvolutionType::Amendment.invalidates_parent());
    }

    #[test]
    fn test_consolidation() {
        let sources = vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
        let new_id = Uuid::new_v4();

        let consolidated = ClaimEvolution::consolidate(
            new_id,
            sources.clone(),
            "Merged three related studies".to_string(),
            "meta@research.org".to_string(),
        );

        assert_eq!(consolidated.merged_from.len(), 3);
        assert_eq!(consolidated.evolution_type, EvolutionType::Consolidation);
    }

    #[test]
    fn test_claim_status() {
        assert!(ClaimStatus::Active.is_valid());
        assert!(!ClaimStatus::Retracted.is_valid());
        assert!(ClaimStatus::Retracted.should_show_warning());
    }
}
