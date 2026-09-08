// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Runtime commissioning binding for locally autonomous safety envelopes.
//!
//! `symthaea-operating-autonomy` defines the locally-owned survival contract.
//! This crate binds that contract to the exact plant/software configuration it was
//! commissioned against. A local envelope may be non-expiring, but it is neither
//! configuration-independent nor rollback-independent authority.

#![deny(unsafe_code)]

pub mod authorization;
pub mod identity;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_operating_authority::OperatingAuthorityRegistry;
use symthaea_operating_autonomy::{
    AutonomyError, EffectiveOperatingEvaluation, LocalSafetyEnvelope,
    evaluate_effective_operation,
};
use symthaea_operating_envelope::OperatingPoint;
use symthaea_resource_hierarchy::ResourceHierarchy;
pub use symthaea_safety_configuration::ConfigurationDigest;
use thiserror::Error;

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

/// One monotone commissioning generation for a node.
///
/// Recommissioning an older plant/software image is allowed only by issuing a
/// strictly newer generation. Replaying an older historical generation never
/// restores local authority merely because its configuration digest still matches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommissioningRecord {
    generation: u64,
    commissioned: CommissionedLocalSafetyEnvelope,
}

impl CommissioningRecord {
    pub fn new(
        generation: u64,
        commissioned: CommissionedLocalSafetyEnvelope,
    ) -> Result<Self, CommissioningError> {
        if generation == 0 {
            return Err(CommissioningError::ZeroGeneration);
        }
        Ok(Self {
            generation,
            commissioned,
        })
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn commissioned(&self) -> &CommissionedLocalSafetyEnvelope {
        &self.commissioned
    }

    pub fn subject_node_id(&self) -> &str {
        &self.commissioned.local.subject_node_id
    }
}

/// Result of attempting to admit a commissioning generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommissioningAdmission {
    Accepted,
    Idempotent,
}

/// Stateful commissioning authority for locally autonomous operation.
///
/// The registry remembers the highest admitted generation for every subject. A
/// lower generation is stale; a different record claiming the same generation is
/// equivocation; an exact replay is idempotent. Deliberate rollback to an older
/// configuration therefore requires a new, higher commissioning generation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CommissioningRegistry {
    current_by_subject: BTreeMap<String, CommissioningRecord>,
}

impl CommissioningRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn admit(
        &mut self,
        hierarchy: &ResourceHierarchy,
        record: CommissioningRecord,
    ) -> Result<CommissioningAdmission, CommissioningError> {
        record.commissioned.validate(hierarchy)?;
        let subject = record.subject_node_id().to_owned();

        if let Some(current) = self.current_by_subject.get(&subject) {
            if record.generation < current.generation {
                return Err(CommissioningError::StaleGeneration {
                    subject,
                    current: current.generation,
                    proposed: record.generation,
                });
            }
            if record.generation == current.generation {
                if record == *current {
                    return Ok(CommissioningAdmission::Idempotent);
                }
                return Err(CommissioningError::GenerationEquivocation {
                    subject,
                    generation: record.generation,
                    current_digest: current.commissioned.binding.configuration_digest(),
                    proposed_digest: record.commissioned.binding.configuration_digest(),
                    current_evidence_id: current.commissioned.binding.evidence_id().to_owned(),
                    proposed_evidence_id: record.commissioned.binding.evidence_id().to_owned(),
                });
            }
        }

        self.current_by_subject.insert(subject, record);
        Ok(CommissioningAdmission::Accepted)
    }

    pub fn current(&self, subject: &str) -> Option<&CommissioningRecord> {
        self.current_by_subject.get(subject)
    }

    pub fn latest_generation(&self, subject: &str) -> Option<u64> {
        self.current(subject).map(CommissioningRecord::generation)
    }

    /// Evaluate operation only through the currently admitted commissioning record.
    pub fn evaluate(
        &self,
        subject: &str,
        current_configuration: ConfigurationDigest,
        hierarchy: &ResourceHierarchy,
        authority: &OperatingAuthorityRegistry,
        point: &OperatingPoint,
        now: DateTime<Utc>,
    ) -> Result<EffectiveOperatingEvaluation, CommissioningError> {
        let current = self
            .current(subject)
            .ok_or_else(|| CommissioningError::NoCurrentCommissioning(subject.to_owned()))?;
        evaluate_commissioned_operation(
            current.commissioned(),
            current_configuration,
            hierarchy,
            authority,
            point,
            now,
        )
    }
}

/// Pure evaluation helper for one already-selected commissioning record.
///
/// Runtime callers that need rollback resistance should use
/// [`CommissioningRegistry::evaluate`], which first selects the highest admitted
/// commissioning generation for the subject.
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
    #[error("commissioning generation must be greater than zero")]
    ZeroGeneration,
    #[error("no current commissioning record for subject {0}")]
    NoCurrentCommissioning(String),
    #[error(
        "stale commissioning generation {proposed} for subject {subject}; current generation is {current}"
    )]
    StaleGeneration {
        subject: String,
        current: u64,
        proposed: u64,
    },
    #[error(
        "commissioning generation {generation} equivocation for subject {subject}: current evidence {current_evidence_id} / {current_digest:?}, proposed evidence {proposed_evidence_id} / {proposed_digest:?}"
    )]
    GenerationEquivocation {
        subject: String,
        generation: u64,
        current_digest: ConfigurationDigest,
        proposed_digest: ConfigurationDigest,
        current_evidence_id: String,
        proposed_evidence_id: String,
    },
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

    fn commissioned_for(byte: u8, evidence: &str) -> CommissionedLocalSafetyEnvelope {
        CommissionedLocalSafetyEnvelope::new(
            local(),
            CommissioningBinding::new(digest(byte), evidence).unwrap(),
        )
    }

    fn commissioned() -> CommissionedLocalSafetyEnvelope {
        commissioned_for(0x11, "commissioning/rack/v1")
    }

    fn record(generation: u64, byte: u8, evidence: &str) -> CommissioningRecord {
        CommissioningRecord::new(generation, commissioned_for(byte, evidence)).unwrap()
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

    #[test]
    fn zero_generation_is_rejected() {
        assert!(matches!(
            CommissioningRecord::new(0, commissioned()),
            Err(CommissioningError::ZeroGeneration)
        ));
    }

    #[test]
    fn higher_generation_replaces_current_commissioning() {
        let hierarchy = hierarchy();
        let mut registry = CommissioningRegistry::new();
        assert_eq!(
            registry.admit(&hierarchy, record(1, 0x11, "commissioning/a/g1")).unwrap(),
            CommissioningAdmission::Accepted
        );
        assert_eq!(
            registry.admit(&hierarchy, record(2, 0x22, "commissioning/b/g2")).unwrap(),
            CommissioningAdmission::Accepted
        );
        assert_eq!(registry.latest_generation("rack"), Some(2));
        assert_eq!(
            registry
                .current("rack")
                .unwrap()
                .commissioned()
                .binding()
                .configuration_digest(),
            digest(0x22)
        );
    }

    #[test]
    fn old_commissioning_cannot_reappear_after_new_generation() {
        let hierarchy = hierarchy();
        let mut registry = CommissioningRegistry::new();
        registry
            .admit(&hierarchy, record(1, 0x11, "commissioning/a/g1"))
            .unwrap();
        registry
            .admit(&hierarchy, record(2, 0x22, "commissioning/b/g2"))
            .unwrap();
        let stale = registry.admit(&hierarchy, record(1, 0x11, "commissioning/a/g1"));
        assert!(matches!(
            stale,
            Err(CommissioningError::StaleGeneration {
                current: 2,
                proposed: 1,
                ..
            })
        ));
    }

    #[test]
    fn explicit_rollback_requires_newer_commissioning_generation() {
        let hierarchy = hierarchy();
        let mut registry = CommissioningRegistry::new();
        registry
            .admit(&hierarchy, record(1, 0x11, "commissioning/a/g1"))
            .unwrap();
        registry
            .admit(&hierarchy, record(2, 0x22, "commissioning/b/g2"))
            .unwrap();
        assert_eq!(
            registry.admit(&hierarchy, record(3, 0x11, "commissioning/a/g3")).unwrap(),
            CommissioningAdmission::Accepted
        );
        assert_eq!(registry.latest_generation("rack"), Some(3));
        assert_eq!(
            registry
                .current("rack")
                .unwrap()
                .commissioned()
                .binding()
                .configuration_digest(),
            digest(0x11)
        );
    }

    #[test]
    fn same_generation_different_record_is_equivocation() {
        let hierarchy = hierarchy();
        let mut registry = CommissioningRegistry::new();
        registry
            .admit(&hierarchy, record(7, 0x11, "commissioning/a/g7"))
            .unwrap();
        let equivocation = registry.admit(&hierarchy, record(7, 0x22, "commissioning/b/g7"));
        assert!(matches!(
            equivocation,
            Err(CommissioningError::GenerationEquivocation { generation: 7, .. })
        ));
    }

    #[test]
    fn exact_same_generation_replay_is_idempotent() {
        let hierarchy = hierarchy();
        let mut registry = CommissioningRegistry::new();
        let record = record(3, 0x11, "commissioning/a/g3");
        registry.admit(&hierarchy, record.clone()).unwrap();
        assert_eq!(
            registry.admit(&hierarchy, record).unwrap(),
            CommissioningAdmission::Idempotent
        );
    }

    #[test]
    fn runtime_evaluation_uses_only_current_generation() {
        let hierarchy = hierarchy();
        let authority = OperatingAuthorityRegistry::new();
        let mut registry = CommissioningRegistry::new();
        registry
            .admit(&hierarchy, record(1, 0x11, "commissioning/a/g1"))
            .unwrap();
        registry
            .admit(&hierarchy, record(2, 0x22, "commissioning/b/g2"))
            .unwrap();

        let old_configuration = registry.evaluate(
            "rack",
            digest(0x11),
            &hierarchy,
            &authority,
            &point(90.0),
            t0() + Duration::minutes(5),
        );
        assert!(matches!(
            old_configuration,
            Err(CommissioningError::ConfigurationDrift {
                expected,
                observed,
                ..
            }) if expected == digest(0x22) && observed == digest(0x11)
        ));

        let current = registry
            .evaluate(
                "rack",
                digest(0x22),
                &hierarchy,
                &authority,
                &point(90.0),
                t0() + Duration::minutes(5),
            )
            .unwrap();
        assert!(current.compliant);
    }

    #[test]
    fn runtime_without_current_commissioning_fails_closed() {
        let hierarchy = hierarchy();
        let authority = OperatingAuthorityRegistry::new();
        let registry = CommissioningRegistry::new();
        assert!(matches!(
            registry.evaluate(
                "rack",
                digest(0x11),
                &hierarchy,
                &authority,
                &point(90.0),
                t0() + Duration::minutes(5),
            ),
            Err(CommissioningError::NoCurrentCommissioning(subject)) if subject == "rack"
        ));
    }
}
