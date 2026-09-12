// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stakeholder and perspective provenance for Wisdom & Care deliberation.
//!
//! WCARE-02 replaces anonymous stakeholder counts with explicit affected-party
//! records. Unknown perspectives remain unknown; inferred or culturally sourced
//! preference claims are not silently promoted into stated preferences or
//! consent.

use std::collections::HashSet;

/// Stable stakeholder identity within one deliberation scope.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StakeholderId(String);

impl StakeholderId {
    pub fn new(value: impl Into<String>) -> Result<Self, PerspectiveError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(PerspectiveError::EmptyStakeholderId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Where a preference claim came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreferenceProvenance {
    /// Directly stated by the affected person in the relevant scope.
    Explicit,
    /// Derived from observed behavior without a direct statement.
    Observed,
    /// Inferred from other evidence.
    Inferred,
    /// Suggested by a cultural, demographic, organizational, or population default.
    CulturalDefault,
}

/// One bounded preference claim about a stakeholder.
#[derive(Debug, Clone, PartialEq)]
pub struct PreferenceClaim {
    pub summary: String,
    pub provenance: PreferenceProvenance,
    pub confidence: f32,
}

impl PreferenceClaim {
    pub fn new(
        summary: impl Into<String>,
        provenance: PreferenceProvenance,
        confidence: f32,
    ) -> Result<Self, PerspectiveError> {
        let summary = summary.into();
        if summary.trim().is_empty() {
            return Err(PerspectiveError::EmptyPreferenceClaim);
        }
        Ok(Self {
            summary,
            provenance,
            confidence: confidence.clamp(0.0, 1.0),
        })
    }

    /// Only an explicit preference can be considered direct preference evidence.
    /// This still does not establish informed consent by itself.
    pub fn is_direct_preference_evidence(&self) -> bool {
        self.provenance == PreferenceProvenance::Explicit
    }
}

/// How well this deliberation currently represents a stakeholder's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PerspectiveCoverage {
    Unknown,
    Partial,
    Adequate,
}

/// Perspective record for an affected or potentially affected stakeholder.
#[derive(Debug, Clone, PartialEq)]
pub struct StakeholderPerspective {
    pub id: StakeholderId,
    pub affected: bool,
    pub coverage: PerspectiveCoverage,
    pub preferences: Vec<PreferenceClaim>,
}

impl StakeholderPerspective {
    pub fn new(id: StakeholderId, affected: bool, coverage: PerspectiveCoverage) -> Self {
        Self {
            id,
            affected,
            coverage,
            preferences: Vec::new(),
        }
    }

    pub fn with_preference(mut self, preference: PreferenceClaim) -> Self {
        self.preferences.push(preference);
        self
    }

    pub fn has_direct_preference_evidence(&self) -> bool {
        self.preferences
            .iter()
            .any(PreferenceClaim::is_direct_preference_evidence)
    }
}

/// Summary of affected-party perspective coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerspectiveCoverageSummary {
    pub affected: usize,
    pub adequately_represented: usize,
    pub unresolved: usize,
}

/// Explicit stakeholder graph for one deliberation.
#[derive(Debug, Clone, PartialEq)]
pub struct PerspectiveGraph {
    stakeholders: Vec<StakeholderPerspective>,
}

impl PerspectiveGraph {
    pub fn try_new(stakeholders: Vec<StakeholderPerspective>) -> Result<Self, PerspectiveError> {
        let mut seen = HashSet::new();
        for stakeholder in &stakeholders {
            if !seen.insert(stakeholder.id.clone()) {
                return Err(PerspectiveError::DuplicateStakeholderId(
                    stakeholder.id.as_str().to_string(),
                ));
            }
        }
        Ok(Self { stakeholders })
    }

    pub fn stakeholders(&self) -> &[StakeholderPerspective] {
        &self.stakeholders
    }

    pub fn find(&self, id: &StakeholderId) -> Option<&StakeholderPerspective> {
        self.stakeholders.iter().find(|s| &s.id == id)
    }

    pub fn coverage_summary(&self) -> PerspectiveCoverageSummary {
        let affected = self.stakeholders.iter().filter(|s| s.affected).count();
        let adequately_represented = self
            .stakeholders
            .iter()
            .filter(|s| s.affected && s.coverage == PerspectiveCoverage::Adequate)
            .count();
        PerspectiveCoverageSummary {
            affected,
            adequately_represented,
            unresolved: affected.saturating_sub(adequately_represented),
        }
    }

    pub fn unresolved_affected(&self) -> Vec<&StakeholderPerspective> {
        self.stakeholders
            .iter()
            .filter(|s| s.affected && s.coverage != PerspectiveCoverage::Adequate)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerspectiveError {
    EmptyStakeholderId,
    EmptyPreferenceClaim,
    DuplicateStakeholderId(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> StakeholderId {
        StakeholderId::new(value).expect("test id should be valid")
    }

    #[test]
    fn rejects_duplicate_stakeholder_identity() {
        let a = StakeholderPerspective::new(id("person-a"), true, PerspectiveCoverage::Adequate);
        let b = StakeholderPerspective::new(id("person-a"), true, PerspectiveCoverage::Unknown);
        assert_eq!(
            PerspectiveGraph::try_new(vec![a, b]),
            Err(PerspectiveError::DuplicateStakeholderId("person-a".into()))
        );
    }

    #[test]
    fn unknown_affected_perspective_remains_unresolved() {
        let graph = PerspectiveGraph::try_new(vec![StakeholderPerspective::new(
            id("person-a"),
            true,
            PerspectiveCoverage::Unknown,
        )])
        .expect("graph should be valid");

        let summary = graph.coverage_summary();
        assert_eq!(summary.affected, 1);
        assert_eq!(summary.adequately_represented, 0);
        assert_eq!(summary.unresolved, 1);
        assert_eq!(graph.unresolved_affected().len(), 1);
    }

    #[test]
    fn cultural_default_is_not_direct_preference_evidence() {
        let claim = PreferenceClaim::new(
            "family should decide",
            PreferenceProvenance::CulturalDefault,
            0.9,
        )
        .expect("claim should be valid");
        assert!(!claim.is_direct_preference_evidence());
    }

    #[test]
    fn explicit_preference_is_direct_but_not_automatically_consent() {
        let claim = PreferenceClaim::new(
            "I prefer option A",
            PreferenceProvenance::Explicit,
            1.0,
        )
        .expect("claim should be valid");
        assert!(claim.is_direct_preference_evidence());
    }

    #[test]
    fn confidence_is_bounded() {
        let claim = PreferenceClaim::new("option A", PreferenceProvenance::Inferred, 4.0)
            .expect("claim should be valid");
        assert_eq!(claim.confidence, 1.0);
    }

    #[test]
    fn unaffected_unknown_party_does_not_create_affected_gap() {
        let graph = PerspectiveGraph::try_new(vec![StakeholderPerspective::new(
            id("observer"),
            false,
            PerspectiveCoverage::Unknown,
        )])
        .expect("graph should be valid");
        assert_eq!(graph.coverage_summary().unresolved, 0);
    }
}
