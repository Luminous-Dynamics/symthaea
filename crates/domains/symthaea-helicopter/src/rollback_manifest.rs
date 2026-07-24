// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deployment rollback lineage and compatibility assessment.
//!
//! A fail-safe deployment mechanism must not treat any older binary as a valid
//! rollback. Rollback targets remain bound to the same airframe and compatibility
//! epoch, must be qualified, and must appear in the authenticated deployment
//! ancestry within a bounded depth.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollbackArtifact {
    pub deployment_id: String,
    pub generation: u64,
    pub parent_deployment_id: Option<String>,
    pub airframe_id: String,
    pub compatibility_epoch: String,
    pub software_digest: String,
    pub qualification_digest: String,
    pub authenticity_reference: Option<String>,
    pub physical_hardware: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RollbackPolicy {
    pub maximum_depth: usize,
    pub require_qualification: bool,
    pub require_authenticity_for_physical: bool,
}

impl Default for RollbackPolicy {
    fn default() -> Self {
        Self {
            maximum_depth: 3,
            require_qualification: true,
            require_authenticity_for_physical: true,
        }
    }
}

impl RollbackPolicy {
    fn validate(&self) -> Result<(), RollbackError> {
        if self.maximum_depth == 0 {
            return Err(RollbackError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RollbackStatus {
    Approved,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RollbackRejection {
    UnknownCurrent,
    UnknownTarget,
    SameDeployment,
    TargetNotOlder,
    AirframeMismatch,
    CompatibilityEpochMismatch,
    NotInAncestry,
    MaximumDepthExceeded,
    MissingQualification,
    MissingAuthenticity,
    InvalidCatalog,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollbackAssessment {
    pub current_deployment_id: String,
    pub target_deployment_id: String,
    pub status: RollbackStatus,
    pub ancestry_path: Vec<String>,
    pub rejection: Option<RollbackRejection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollbackError {
    InvalidPolicy,
    InvalidArtifact,
    DuplicateDeployment,
    MissingParent,
    NonMonotonicGeneration,
    AncestryCycle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollbackCatalog {
    pub artifacts: Vec<RollbackArtifact>,
}

impl RollbackCatalog {
    pub fn new(artifacts: Vec<RollbackArtifact>) -> Result<Self, RollbackError> {
        let catalog = Self { artifacts };
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn validate(&self) -> Result<(), RollbackError> {
        let mut by_id = BTreeMap::new();
        for artifact in &self.artifacts {
            if artifact.deployment_id.trim().is_empty()
                || artifact.airframe_id.trim().is_empty()
                || artifact.compatibility_epoch.trim().is_empty()
                || !valid_digest(&artifact.software_digest)
                || !valid_digest(&artifact.qualification_digest)
                || artifact
                    .authenticity_reference
                    .as_ref()
                    .is_some_and(|reference| reference.trim().is_empty())
            {
                return Err(RollbackError::InvalidArtifact);
            }
            if by_id
                .insert(artifact.deployment_id.as_str(), artifact)
                .is_some()
            {
                return Err(RollbackError::DuplicateDeployment);
            }
        }
        for artifact in &self.artifacts {
            if let Some(parent_id) = artifact.parent_deployment_id.as_deref() {
                let parent = by_id.get(parent_id).ok_or(RollbackError::MissingParent)?;
                if parent.generation >= artifact.generation {
                    return Err(RollbackError::NonMonotonicGeneration);
                }
            }
            let mut seen = BTreeSet::new();
            let mut cursor = Some(artifact.deployment_id.as_str());
            while let Some(id) = cursor {
                if !seen.insert(id) {
                    return Err(RollbackError::AncestryCycle);
                }
                cursor = by_id
                    .get(id)
                    .and_then(|entry| entry.parent_deployment_id.as_deref());
            }
        }
        Ok(())
    }

    pub fn assess(
        &self,
        current_id: &str,
        target_id: &str,
        policy: RollbackPolicy,
    ) -> Result<RollbackAssessment, RollbackError> {
        self.validate()?;
        policy.validate()?;
        let by_id: BTreeMap<_, _> = self
            .artifacts
            .iter()
            .map(|artifact| (artifact.deployment_id.as_str(), artifact))
            .collect();
        let Some(current) = by_id.get(current_id).copied() else {
            return Ok(rejected(
                current_id,
                target_id,
                RollbackRejection::UnknownCurrent,
            ));
        };
        let Some(target) = by_id.get(target_id).copied() else {
            return Ok(rejected(
                current_id,
                target_id,
                RollbackRejection::UnknownTarget,
            ));
        };
        if current_id == target_id {
            return Ok(rejected(
                current_id,
                target_id,
                RollbackRejection::SameDeployment,
            ));
        }
        if target.generation >= current.generation {
            return Ok(rejected(
                current_id,
                target_id,
                RollbackRejection::TargetNotOlder,
            ));
        }
        if target.airframe_id != current.airframe_id {
            return Ok(rejected(
                current_id,
                target_id,
                RollbackRejection::AirframeMismatch,
            ));
        }
        if target.compatibility_epoch != current.compatibility_epoch {
            return Ok(rejected(
                current_id,
                target_id,
                RollbackRejection::CompatibilityEpochMismatch,
            ));
        }
        if policy.require_qualification && !valid_digest(&target.qualification_digest) {
            return Ok(rejected(
                current_id,
                target_id,
                RollbackRejection::MissingQualification,
            ));
        }
        if policy.require_authenticity_for_physical
            && current.physical_hardware
            && target.authenticity_reference.is_none()
        {
            return Ok(rejected(
                current_id,
                target_id,
                RollbackRejection::MissingAuthenticity,
            ));
        }

        let mut ancestry_path = vec![current.deployment_id.clone()];
        let mut cursor = current.parent_deployment_id.as_deref();
        let mut depth = 0usize;
        while let Some(parent_id) = cursor {
            depth += 1;
            ancestry_path.push(parent_id.to_string());
            if parent_id == target_id {
                if depth > policy.maximum_depth {
                    return Ok(RollbackAssessment {
                        current_deployment_id: current_id.to_string(),
                        target_deployment_id: target_id.to_string(),
                        status: RollbackStatus::Rejected,
                        ancestry_path,
                        rejection: Some(RollbackRejection::MaximumDepthExceeded),
                    });
                }
                return Ok(RollbackAssessment {
                    current_deployment_id: current_id.to_string(),
                    target_deployment_id: target_id.to_string(),
                    status: RollbackStatus::Approved,
                    ancestry_path,
                    rejection: None,
                });
            }
            cursor = by_id
                .get(parent_id)
                .and_then(|entry| entry.parent_deployment_id.as_deref());
        }
        Ok(RollbackAssessment {
            current_deployment_id: current_id.to_string(),
            target_deployment_id: target_id.to_string(),
            status: RollbackStatus::Rejected,
            ancestry_path,
            rejection: Some(RollbackRejection::NotInAncestry),
        })
    }
}

fn rejected(current_id: &str, target_id: &str, rejection: RollbackRejection) -> RollbackAssessment {
    RollbackAssessment {
        current_deployment_id: current_id.to_string(),
        target_deployment_id: target_id.to_string(),
        status: RollbackStatus::Rejected,
        ancestry_path: Vec::new(),
        rejection: Some(rejection),
    }
}

fn valid_digest(value: &str) -> bool {
    let Some((algorithm, digest)) = value.split_once(':') else {
        return false;
    };
    !algorithm.trim().is_empty()
        && digest.len() >= 8
        && digest
            .chars()
            .all(|character| character.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(id: &str, generation: u64, parent: Option<&str>) -> RollbackArtifact {
        RollbackArtifact {
            deployment_id: id.into(),
            generation,
            parent_deployment_id: parent.map(str::to_string),
            airframe_id: "airframe-1".into(),
            compatibility_epoch: "epoch-1".into(),
            software_digest: format!("sha256:{:08x}", generation + 100),
            qualification_digest: format!("sha256:{:08x}", generation + 200),
            authenticity_reference: Some(format!("signature-{id}")),
            physical_hardware: true,
        }
    }

    #[test]
    fn direct_qualified_ancestor_is_approved() {
        let catalog =
            RollbackCatalog::new(vec![artifact("v1", 1, None), artifact("v2", 2, Some("v1"))])
                .unwrap();
        let assessment = catalog
            .assess("v2", "v1", RollbackPolicy::default())
            .unwrap();
        assert_eq!(assessment.status, RollbackStatus::Approved);
    }

    #[test]
    fn sibling_is_not_a_valid_rollback_target() {
        let catalog = RollbackCatalog::new(vec![
            artifact("v1", 1, None),
            artifact("v2a", 2, Some("v1")),
            artifact("v2b", 2, Some("v1")),
        ])
        .unwrap();
        let assessment = catalog
            .assess("v2a", "v2b", RollbackPolicy::default())
            .unwrap();
        assert_eq!(
            assessment.rejection,
            Some(RollbackRejection::TargetNotOlder)
        );
    }

    #[test]
    fn incompatible_epoch_is_rejected() {
        let base = artifact("v1", 1, None);
        let mut current = artifact("v2", 2, Some("v1"));
        current.compatibility_epoch = "epoch-2".into();
        let catalog = RollbackCatalog::new(vec![base, current]).unwrap();
        let assessment = catalog
            .assess("v2", "v1", RollbackPolicy::default())
            .unwrap();
        assert_eq!(
            assessment.rejection,
            Some(RollbackRejection::CompatibilityEpochMismatch)
        );
    }

    #[test]
    fn depth_limit_is_enforced() {
        let catalog = RollbackCatalog::new(vec![
            artifact("v1", 1, None),
            artifact("v2", 2, Some("v1")),
            artifact("v3", 3, Some("v2")),
        ])
        .unwrap();
        let assessment = catalog
            .assess(
                "v3",
                "v1",
                RollbackPolicy {
                    maximum_depth: 1,
                    ..RollbackPolicy::default()
                },
            )
            .unwrap();
        assert_eq!(
            assessment.rejection,
            Some(RollbackRejection::MaximumDepthExceeded)
        );
    }

    #[test]
    fn missing_parent_invalidates_catalog() {
        assert_eq!(
            RollbackCatalog::new(vec![artifact("v2", 2, Some("missing"))]).unwrap_err(),
            RollbackError::MissingParent
        );
    }
}
