// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Runtime commissioning binding for locally autonomous safety envelopes.
//!
//! `symthaea-operating-autonomy` defines the locally-owned survival contract.
//! This crate binds that contract to the exact plant/software configuration it was
//! commissioned against. A local envelope may be non-expiring, but it is not
//! configuration-independent authority.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use symthaea_operating_authority::OperatingAuthorityRegistry;
use symthaea_operating_autonomy::{
    AutonomyError, EffectiveOperatingEvaluation, LocalSafetyEnvelope,
    evaluate_effective_operation,
};
use symthaea_operating_envelope::OperatingPoint;
use symthaea_resource_hierarchy::ResourceHierarchy;
use thiserror::Error;

/// Exact digest algorithm and bytes used to identify one safety-relevant
/// configuration image. This crate records/compares digests; digest computation
/// belongs to the provisioning/evidence layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfigurationDigest {
    Blake3_256([u8; 32]),
}

/// Provenance tying a local safety envelope to commissioning evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommissioningBinding {
    configuration_digest: ConfigurationDigest,
    evidence_id: String,
}

impl CommissioningBinding {
    pub fn new(
        configuration_digest: ConfigurationDigest,
        evidence_id: impl Into<String>,
    ) -> Result<Self, CommissioningError> {
        let evidence_id = evidence_id.into();
        if evidence_id.trim().is_empty() {
            return Err(CommissioningError::EmptyEvidenceId);
        }
        Ok(Self {
            configuration_digest,
            evidence_id,
        })
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }

    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }

    fn validate(&self) -> Result<(), CommissioningError> {
        if self.evidence_id.trim().is_empty() {
            return Err(CommissioningError::EmptyEvidenceId);
        }
        Ok(())
    }
}

/// A locally-owned safety envelope qualified against one exact configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommissionedLocalSafetyEnvelope {
    local: LocalSafetyEnvelope,
    binding: CommissioningBinding,
}

impl CommissionedLocalSafetyEnvelope {
    pub fn new(local: LocalSafetyEnvelope, binding: CommissioningBinding) -> Self {
        Self { local, binding }
    }

    pub fn local(&self) -> &LocalSafetyEnvelope {
        &self.local
    }

    pub fn binding(&self) -> &CommissioningBinding {
        &self.binding
    }

    pub fn validate(&self, hierarchy: &ResourceHierarchy) -> Result<(), CommissioningError> {
        self.local.validate(hierarchy)?;
        self.binding.validate()?;
        Ok(())
    }
}

/// Runtime evaluation boundary for locally autonomous operation.
///
/// Configuration equality is checked before either local or parent constraints are
/// evaluated. A parent lease cannot authorize operation under an uncommissioned
/// plant configuration, and parent expiry does not weaken this requirement.
pub fn evaluate_commissioned_operation(
    commissioned: &CommissionedLocalSafetyEnvelope,
    current_configuration: ConfigurationDigest,
    hierarchy: &ResourceHierarchy,
    authority: &OperatingAuthorityRegistry,
    point: &OperatingPoint,
    now: DateTime<Utc>,
) -> Result<EffectiveOperatingEvaluation, CommissioningError> {
    commissioned.validate(hierarchy)?;
    let expected = commissioned.binding.configuration_digest();
    if current_configuration != expected {
        return Err(CommissioningError::ConfigurationDrift {
            expected,
            observed: current_configuration,
            commissioning_evidence_id: commissioned.binding.evidence_id().to_owned(),
        });
    }

    Ok(evaluate_effective_operation(
        commissioned.local(),
        hierarchy,
        authority,
        point,
        now,
    )?)
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum CommissioningError {
    #[error(transparent)]
    Autonomy(#[from] AutonomyError),
    #[error("commissioning evidence id must not be empty")]
    EmptyEvidenceId,
    #[error(
        "configuration drift from commissioned image {expected:?} to {observed:?}; evidence {commissioning_evidence_id} no longer authorizes local survival operation"
    )]
    ConfigurationDrift {
        expected: ConfigurationDigest,
        observed: ConfigurationDigest,
        commissioning_evidence_id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use std::collections::BTreeSet;
    use symthaea_operating_envelope::{
        BoundaryMetric, ObservedResourceMetric, OperatingMode, ResourceConstraint,
    };
    use symthaea_resource_hierarchy::{NodeScale, ResourceNode};
    use symthaea_resource_model::{ResourceAmount, ResourceEnvelope, ResourceKind, ResourceUnit};

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn power_key() -> symthaea_resource_model::ResourceKey {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, 0.0)
            .unwrap()
            .key
    }

    fn hierarchy() -> ResourceHierarchy {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "site",
                "Site",
                NodeScale::Site,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack",
                    "Rack",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
    }

    fn local() -> LocalSafetyEnvelope {
        let mut allowed_modes = BTreeSet::new();
        allowed_modes.insert(OperatingMode::Normal);
        allowed_modes.insert(OperatingMode::Islanded);
        LocalSafetyEnvelope {
            id: "rack-survival-v1".into(),
            subject_node_id: "rack".into(),
            allowed_modes,
            resource_constraints: vec![ResourceConstraint {
                key: power_key(),
                metric: BoundaryMetric::Import,
                minimum: None,
                maximum: Some(100.0),
            }],
            critical_service_floor: 0.70,
            max_shed_fraction: 0.30,
        }
    }

    fn point(import: f64) -> OperatingPoint {
        OperatingPoint {
            mode: OperatingMode::Normal,
            critical_service_fraction: 0.90,
            shed_fraction: 0.10,
            resources: vec![ObservedResourceMetric {
                key: power_key(),
                metric: BoundaryMetric::Import,
                value: import,
            }],
        }
    }

    fn commissioned() -> CommissionedLocalSafetyEnvelope {
        CommissionedLocalSafetyEnvelope::new(
            local(),
            CommissioningBinding::new(digest(0x11), "commissioning/rack/v1").unwrap(),
        )
    }

    #[test]
    fn exact_configuration_can_operate_local_only() {
        let hierarchy = hierarchy();
        let authority = OperatingAuthorityRegistry::new();
        let result = evaluate_commissioned_operation(
            &commissioned(),
            digest(0x11),
            &hierarchy,
            &authority,
            &point(90.0),
            t0() + Duration::minutes(5),
        )
        .unwrap();
        assert!(result.compliant);
    }

    #[test]
    fn configuration_drift_fails_closed_without_parent() {
        let hierarchy = hierarchy();
        let authority = OperatingAuthorityRegistry::new();
        let result = evaluate_commissioned_operation(
            &commissioned(),
            digest(0x22),
            &hierarchy,
            &authority,
            &point(90.0),
            t0() + Duration::minutes(5),
        );
        assert!(matches!(
            result,
            Err(CommissioningError::ConfigurationDrift {
                expected,
                observed,
                ..
            }) if expected == digest(0x11) && observed == digest(0x22)
        ));
    }

    #[test]
    fn configuration_match_does_not_relax_local_limits() {
        let hierarchy = hierarchy();
        let authority = OperatingAuthorityRegistry::new();
        let result = evaluate_commissioned_operation(
            &commissioned(),
            digest(0x11),
            &hierarchy,
            &authority,
            &point(110.0),
            t0() + Duration::minutes(5),
        )
        .unwrap();
        assert!(!result.compliant);
    }

    #[test]
    fn empty_commissioning_evidence_is_rejected() {
        assert!(matches!(
            CommissioningBinding::new(digest(0x11), "  "),
            Err(CommissioningError::EmptyEvidenceId)
        ));
    }

    #[test]
    fn digest_algorithm_is_part_of_the_typed_binding() {
        let binding = CommissioningBinding::new(digest(0x33), "commissioning/v2").unwrap();
        assert_eq!(binding.configuration_digest(), digest(0x33));
        assert_eq!(binding.evidence_id(), "commissioning/v2");
    }
}
