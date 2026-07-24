// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical certifiable-autonomy requirement registry.
//!
//! Requirement identifiers are stable release interfaces. They may be
//! referenced by scenario manifests, traceability records, safety-case claims,
//! and release sign-off bundles without relying on prose matching.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RequirementId {
    SafeCommandBounds,
    HazardPreemption,
    ReturnReserveProtection,
    SensorQuorum,
    ActuatorIsolation,
    ThermalPowerDerating,
    OperatorReplayResistance,
    RecoveryQuorum,
    UpdateRollback,
    PartitionReconciliation,
    EvidenceCompleteness,
    CheckpointContinuity,
}

impl RequirementId {
    pub const ALL: [Self; 12] = [
        Self::SafeCommandBounds,
        Self::HazardPreemption,
        Self::ReturnReserveProtection,
        Self::SensorQuorum,
        Self::ActuatorIsolation,
        Self::ThermalPowerDerating,
        Self::OperatorReplayResistance,
        Self::RecoveryQuorum,
        Self::UpdateRollback,
        Self::PartitionReconciliation,
        Self::EvidenceCompleteness,
        Self::CheckpointContinuity,
    ];

    pub const fn code(self) -> &'static str {
        match self {
            Self::SafeCommandBounds => "SUB-SAF-001",
            Self::HazardPreemption => "SUB-SAF-002",
            Self::ReturnReserveProtection => "SUB-SAF-003",
            Self::SensorQuorum => "SUB-SEN-001",
            Self::ActuatorIsolation => "SUB-ACT-001",
            Self::ThermalPowerDerating => "SUB-PWR-001",
            Self::OperatorReplayResistance => "SUB-AUT-001",
            Self::RecoveryQuorum => "SUB-AUT-002",
            Self::UpdateRollback => "SUB-UPD-001",
            Self::PartitionReconciliation => "SUB-COM-001",
            Self::EvidenceCompleteness => "SUB-EVD-001",
            Self::CheckpointContinuity => "SUB-RST-001",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequirementCriticality {
    Mission,
    Safety,
    Catastrophic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationMethod {
    Analysis,
    DeterministicTest,
    RuntimeInvariant,
    Inspection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementDefinition {
    pub id: RequirementId,
    pub title: String,
    pub criticality: RequirementCriticality,
    pub verification: VerificationMethod,
    pub release_blocking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementRegistry {
    requirements: Vec<RequirementDefinition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequirementRegistryError {
    DuplicateId(RequirementId),
    DuplicateCode(String),
    MissingCanonicalRequirement(RequirementId),
    EmptyTitle(RequirementId),
}

impl RequirementRegistry {
    pub fn canonical() -> Self {
        use RequirementCriticality::{Catastrophic, Safety};
        use VerificationMethod::{DeterministicTest, Inspection, RuntimeInvariant};
        let definitions = [
            (
                RequirementId::SafeCommandBounds,
                "Physical commands remain finite and bounded",
                Catastrophic,
                RuntimeInvariant,
            ),
            (
                RequirementId::HazardPreemption,
                "Physical hazards preempt nominal and operator control",
                Catastrophic,
                RuntimeInvariant,
            ),
            (
                RequirementId::ReturnReserveProtection,
                "Protected return energy cannot fund productive work",
                Catastrophic,
                DeterministicTest,
            ),
            (
                RequirementId::SensorQuorum,
                "Declared redundant critical sensing fails closed without quorum",
                Catastrophic,
                DeterministicTest,
            ),
            (
                RequirementId::ActuatorIsolation,
                "Persistently failed actuators lose authority",
                Catastrophic,
                RuntimeInvariant,
            ),
            (
                RequirementId::ThermalPowerDerating,
                "Thermal and power envelopes preserve survival loads before work",
                Safety,
                DeterministicTest,
            ),
            (
                RequirementId::OperatorReplayResistance,
                "Stale or replayed operator authority is rejected",
                Catastrophic,
                DeterministicTest,
            ),
            (
                RequirementId::RecoveryQuorum,
                "Motion recovery requires independent authorized approvals",
                Catastrophic,
                DeterministicTest,
            ),
            (
                RequirementId::UpdateRollback,
                "Failed provisional activation returns to the prior artifact",
                Safety,
                DeterministicTest,
            ),
            (
                RequirementId::PartitionReconciliation,
                "Restored communications require bounded reconciliation",
                Safety,
                DeterministicTest,
            ),
            (
                RequirementId::EvidenceCompleteness,
                "Every control frame records the authorities that shaped actuation",
                Safety,
                Inspection,
            ),
            (
                RequirementId::CheckpointContinuity,
                "Restart preserves safety-relevant replay and isolation state",
                Catastrophic,
                DeterministicTest,
            ),
        ];
        Self {
            requirements: definitions
                .into_iter()
                .map(
                    |(id, title, criticality, verification)| RequirementDefinition {
                        id,
                        title: title.to_string(),
                        criticality,
                        verification,
                        release_blocking: true,
                    },
                )
                .collect(),
        }
    }

    pub fn from_requirements(requirements: Vec<RequirementDefinition>) -> Self {
        Self { requirements }
    }

    pub fn requirements(&self) -> &[RequirementDefinition] {
        &self.requirements
    }

    pub fn definition(&self, id: RequirementId) -> Option<&RequirementDefinition> {
        self.requirements
            .iter()
            .find(|definition| definition.id == id)
    }

    pub fn validate(&self) -> Result<(), RequirementRegistryError> {
        let mut ids = BTreeSet::new();
        let mut codes = BTreeSet::new();
        for requirement in &self.requirements {
            if requirement.title.trim().is_empty() {
                return Err(RequirementRegistryError::EmptyTitle(requirement.id));
            }
            if !ids.insert(requirement.id) {
                return Err(RequirementRegistryError::DuplicateId(requirement.id));
            }
            let code = requirement.id.code().to_string();
            if !codes.insert(code.clone()) {
                return Err(RequirementRegistryError::DuplicateCode(code));
            }
        }
        for id in RequirementId::ALL {
            if !ids.contains(&id) {
                return Err(RequirementRegistryError::MissingCanonicalRequirement(id));
            }
        }
        Ok(())
    }

    pub fn release_blocking_ids(&self) -> Vec<RequirementId> {
        self.requirements
            .iter()
            .filter(|requirement| requirement.release_blocking)
            .map(|requirement| requirement.id)
            .collect()
    }
}

impl Default for RequirementRegistry {
    fn default() -> Self {
        Self::canonical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_registry_is_complete_and_unique() {
        let registry = RequirementRegistry::canonical();
        assert_eq!(registry.requirements().len(), RequirementId::ALL.len());
        assert_eq!(registry.validate(), Ok(()));
    }

    #[test]
    fn missing_requirement_is_rejected() {
        let mut requirements = RequirementRegistry::canonical().requirements().to_vec();
        requirements.pop();
        assert!(matches!(
            RequirementRegistry::from_requirements(requirements).validate(),
            Err(RequirementRegistryError::MissingCanonicalRequirement(_))
        ));
    }
}
