// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence-completeness layer for passive/no-moving-parts design claims.
//!
//! The core rule is simple and auditable: absence of a fact is not the same
//! thing as a measured zero. This crate tracks the distinction explicitly.

use symthaea_fabrication_kernel::PassiveEvidenceFact;

/// Safety-critical field required before a passive claim can be considered complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RequiredEvidenceField {
    MovingSolidComponents,
    MechanicalJoints,
    ActivePowerWatts,
    CommandedActuators,
    ExternalControlRequired,
}

impl RequiredEvidenceField {
    /// Every field needed for the minimum passive-safety evidence set.
    pub const ALL: [Self; 5] = [
        Self::MovingSolidComponents,
        Self::MechanicalJoints,
        Self::ActivePowerWatts,
        Self::CommandedActuators,
        Self::ExternalControlRequired,
    ];

    fn from_fact(fact: PassiveEvidenceFact) -> Option<Self> {
        match fact {
            PassiveEvidenceFact::MovingSolidComponents { .. } => Some(Self::MovingSolidComponents),
            PassiveEvidenceFact::MechanicalJoints { .. } => Some(Self::MechanicalJoints),
            PassiveEvidenceFact::ActivePowerWatts { .. } => Some(Self::ActivePowerWatts),
            PassiveEvidenceFact::CommandedActuators { .. } => Some(Self::CommandedActuators),
            PassiveEvidenceFact::ExternalControlRequired { .. } => {
                Some(Self::ExternalControlRequired)
            }
            PassiveEvidenceFact::FluidMotionUsed { .. }
            | PassiveEvidenceFact::DistributedDeformationUsed { .. }
            | PassiveEvidenceFact::PhaseChangeUsed { .. } => None,
        }
    }
}

/// Report showing which safety-critical passive evidence was actually supplied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceCoverage {
    pub present: Vec<RequiredEvidenceField>,
    pub missing: Vec<RequiredEvidenceField>,
}

impl EvidenceCoverage {
    /// Compute coverage without interpreting absent values as zeros.
    pub fn from_facts(facts: &[PassiveEvidenceFact]) -> Self {
        let mut present = Vec::new();

        for fact in facts.iter().copied() {
            if let Some(field) = RequiredEvidenceField::from_fact(fact) {
                if !present.contains(&field) {
                    present.push(field);
                }
            }
        }

        present.sort_unstable();

        let missing = RequiredEvidenceField::ALL
            .iter()
            .copied()
            .filter(|field| !present.contains(field))
            .collect();

        Self { present, missing }
    }

    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::PassiveEvidenceSource;

    #[test]
    fn empty_facts_are_incomplete() {
        let coverage = EvidenceCoverage::from_facts(&[]);
        assert!(!coverage.is_complete());
        assert_eq!(coverage.missing.len(), RequiredEvidenceField::ALL.len());
    }

    #[test]
    fn zero_values_still_count_as_explicit_evidence() {
        let facts = [
            PassiveEvidenceFact::MovingSolidComponents {
                count: 0,
                source: PassiveEvidenceSource::DesignDeclaration,
            },
            PassiveEvidenceFact::MechanicalJoints {
                count: 0,
                source: PassiveEvidenceSource::DesignDeclaration,
            },
            PassiveEvidenceFact::ActivePowerWatts {
                watts: 0.0,
                source: PassiveEvidenceSource::SimulationDeclaration,
            },
            PassiveEvidenceFact::CommandedActuators {
                count: 0,
                source: PassiveEvidenceSource::DesignDeclaration,
            },
            PassiveEvidenceFact::ExternalControlRequired {
                required: false,
                source: PassiveEvidenceSource::DesignDeclaration,
            },
        ];
        let coverage = EvidenceCoverage::from_facts(&facts);
        assert!(coverage.is_complete());
        assert!(coverage.missing.is_empty());
    }

    #[test]
    fn optional_physical_state_does_not_substitute_for_safety_fields() {
        let facts = [PassiveEvidenceFact::FluidMotionUsed {
            used: true,
            source: PassiveEvidenceSource::SimulationDeclaration,
        }];
        let coverage = EvidenceCoverage::from_facts(&facts);
        assert_eq!(coverage.present.len(), 0);
        assert_eq!(coverage.missing.len(), RequiredEvidenceField::ALL.len());
    }
}
