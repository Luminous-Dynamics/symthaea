// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Portable exact-subject envelopes for lifecycle engineering contracts.
//!
//! Cross-repository equality is the raw semantic tuple:
//!
//! ```text
//! { namespace, subject_id, semantic_version, content_blake3 }
//! ```
//!
//! A valid portable reference does not prove the external consumer resolved or admitted the
//! subject, and it creates no manufacturing or lifecycle authority.

use serde::{Deserialize, Serialize};
use symthaea_lifecycle_engineering::LifecycleDesignProfileV1;
use symthaea_manufacturing_lifecycle_bridge::ManufacturingLifecycleBridgeV1;
use thiserror::Error;

pub const LIFECYCLE_DESIGN_PROFILE_NAMESPACE: &str = "symthaea.lifecycle.design-profile";
pub const MANUFACTURING_LIFECYCLE_BRIDGE_NAMESPACE: &str =
    "symthaea.manufacturing.lifecycle-bridge";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PortableSubjectRefError {
    #[error("{field} must be canonical and non-empty")]
    NonCanonical { field: &'static str },
    #[error("subject/content digest must be lowercase 64-character hexadecimal BLAKE3")]
    InvalidDigest,
    #[error("canonical V1 portable subject requires subject_id == content_blake3")]
    SubjectDigestMismatch,
    #[error("source subject invalid: {0}")]
    InvalidSource(String),
}

fn canonical(field: &'static str, value: &str) -> Result<(), PortableSubjectRefError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(PortableSubjectRefError::NonCanonical { field });
    }
    Ok(())
}

fn digest64(value: &str) -> Result<(), PortableSubjectRefError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(PortableSubjectRefError::InvalidDigest);
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PortableExactSubjectRefV1 {
    pub namespace: String,
    pub subject_id: String,
    pub semantic_version: String,
    pub content_blake3: String,
}

impl PortableExactSubjectRefV1 {
    pub fn validate(&self) -> Result<(), PortableSubjectRefError> {
        canonical("portable.namespace", &self.namespace)?;
        canonical("portable.semantic_version", &self.semantic_version)?;
        digest64(&self.subject_id)?;
        digest64(&self.content_blake3)?;
        if self.subject_id != self.content_blake3 {
            return Err(PortableSubjectRefError::SubjectDigestMismatch);
        }
        Ok(())
    }
}

pub trait ToPortableExactSubjectRefV1 {
    fn to_portable_exact_ref(&self) -> Result<PortableExactSubjectRefV1, PortableSubjectRefError>;
}

impl ToPortableExactSubjectRefV1 for LifecycleDesignProfileV1 {
    fn to_portable_exact_ref(&self) -> Result<PortableExactSubjectRefV1, PortableSubjectRefError> {
        let id = self
            .profile_id()
            .map_err(|err| PortableSubjectRefError::InvalidSource(err.to_string()))?;
        let value = PortableExactSubjectRefV1 {
            namespace: LIFECYCLE_DESIGN_PROFILE_NAMESPACE.into(),
            subject_id: id.clone(),
            semantic_version: self.semantic_version.clone(),
            content_blake3: id,
        };
        value.validate()?;
        Ok(value)
    }
}

impl ToPortableExactSubjectRefV1 for ManufacturingLifecycleBridgeV1 {
    fn to_portable_exact_ref(&self) -> Result<PortableExactSubjectRefV1, PortableSubjectRefError> {
        let id = self
            .bridge_id()
            .map_err(|err| PortableSubjectRefError::InvalidSource(err.to_string()))?;
        let value = PortableExactSubjectRefV1 {
            namespace: MANUFACTURING_LIFECYCLE_BRIDGE_NAMESPACE.into(),
            subject_id: id.clone(),
            semantic_version: self.semantic_version.clone(),
            content_blake3: id,
        };
        value.validate()?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_lifecycle_engineering::{
        ExactLifecycleSubjectRefV1, LifecycleAssessmentRefV1, LifecycleAssessmentKindV1,
        LifecycleDesignAuthorityCeilingV1, LifecycleStrategyKindV1, LifecycleStrategyRefV1,
    };
    use symthaea_manufacturing_lifecycle_bridge::{
        ManufacturingLifecycleAuthorityCeilingV1, ProcessLifecycleRelationKindV1,
        ProcessLifecycleRelationV1,
    };

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn exact(namespace: &str, id: &str, fill: char) -> ExactLifecycleSubjectRefV1 {
        ExactLifecycleSubjectRefV1 {
            namespace: namespace.into(),
            subject_id: id.into(),
            semantic_version: "1".into(),
            content_blake3: digest(fill),
        }
    }

    fn plan_ref() -> ExactLifecycleSubjectRefV1 {
        ExactLifecycleSubjectRefV1 {
            namespace: "symthaea.manufacturing.process-plan".into(),
            subject_id: digest('a'),
            semantic_version: "1.0.0".into(),
            content_blake3: digest('a'),
        }
    }

    fn profile(plan: ExactLifecycleSubjectRefV1) -> LifecycleDesignProfileV1 {
        LifecycleDesignProfileV1 {
            semantic_version: "1.0.0".into(),
            subject_configuration: exact("symthaea.configuration", "article-a", 'b'),
            functional_unit: exact("symthaea.functional-unit", "service-unit", 'c'),
            manufacturing_plan_refs: vec![plan],
            strategies: vec![LifecycleStrategyRefV1 {
                kind: LifecycleStrategyKindV1::Repair,
                subject: exact("symthaea.lifecycle-strategy", "repair", 'd'),
            }],
            assessments: vec![LifecycleAssessmentRefV1 {
                kind: LifecycleAssessmentKindV1::MaterialFlowAccount,
                subject: exact("mycelix.material-flow.account", "account", 'e'),
            }],
            authority_ceiling:
                LifecycleDesignAuthorityCeilingV1::DesignIntentAndAssessmentReferencesOnly,
            display_label: Some("Lifecycle profile".into()),
            notes: Some("navigation only".into()),
        }
    }

    fn bridge(profile_ref: PortableExactSubjectRefV1) -> ManufacturingLifecycleBridgeV1 {
        ManufacturingLifecycleBridgeV1 {
            semantic_version: "1.0.0".into(),
            process_definition: exact(
                "symthaea.manufacturing.process-definition",
                "repair-process",
                'f',
            ),
            process_pack_profile: None,
            lifecycle_design_profile: ExactLifecycleSubjectRefV1 {
                namespace: profile_ref.namespace,
                subject_id: profile_ref.subject_id,
                semantic_version: profile_ref.semantic_version,
                content_blake3: profile_ref.content_blake3,
            },
            relations: vec![ProcessLifecycleRelationV1 {
                kind: ProcessLifecycleRelationKindV1::EnablesCandidateRepairRoute,
                subject: exact("symthaea.lifecycle-assessment", "repair-route", '1'),
            }],
            authority_ceiling:
                ManufacturingLifecycleAuthorityCeilingV1::DesignRelationsAndAssessmentRequirementsOnly,
            display_label: Some("Repair-aware route".into()),
            notes: None,
        }
    }

    #[test]
    fn profile_and_bridge_project_portable_exact_refs() {
        let profile = profile(plan_ref());
        let profile_ref = profile.to_portable_exact_ref().unwrap();
        assert_eq!(profile_ref.namespace, LIFECYCLE_DESIGN_PROFILE_NAMESPACE);
        assert_eq!(profile_ref.subject_id, profile.profile_id().unwrap());

        let bridge = bridge(profile_ref.clone());
        assert_eq!(
            bridge.lifecycle_design_profile.namespace,
            profile_ref.namespace
        );
        assert_eq!(
            bridge.lifecycle_design_profile.content_blake3,
            profile_ref.content_blake3
        );

        let bridge_ref = bridge.to_portable_exact_ref().unwrap();
        assert_eq!(
            bridge_ref.namespace,
            MANUFACTURING_LIFECYCLE_BRIDGE_NAMESPACE
        );
        assert_eq!(bridge_ref.subject_id, bridge.bridge_id().unwrap());
    }

    #[test]
    fn display_metadata_does_not_change_portable_refs() {
        let a = profile(plan_ref());
        let mut b = a.clone();
        b.display_label = Some("Renamed".into());
        b.notes = Some("Different prose".into());
        assert_eq!(
            a.to_portable_exact_ref().unwrap(),
            b.to_portable_exact_ref().unwrap()
        );
    }

    #[test]
    fn semantic_plan_reference_change_propagates_to_profile_and_bridge() {
        let a_profile = profile(plan_ref());
        let a_profile_ref = a_profile.to_portable_exact_ref().unwrap();
        let a_bridge_ref = bridge(a_profile_ref.clone()).to_portable_exact_ref().unwrap();

        let mut changed_plan = plan_ref();
        changed_plan.subject_id = digest('9');
        changed_plan.content_blake3 = digest('9');
        let b_profile = profile(changed_plan);
        let b_profile_ref = b_profile.to_portable_exact_ref().unwrap();
        let b_bridge_ref = bridge(b_profile_ref.clone()).to_portable_exact_ref().unwrap();

        assert_ne!(a_profile_ref, b_profile_ref);
        assert_ne!(a_bridge_ref, b_bridge_ref);
    }

    #[test]
    fn invalid_portable_tuple_fails_closed() {
        let mut value = profile(plan_ref()).to_portable_exact_ref().unwrap();
        value.content_blake3 = digest('7');
        assert_eq!(
            value.validate(),
            Err(PortableSubjectRefError::SubjectDigestMismatch)
        );

        value.content_blake3 = value.subject_id.to_uppercase();
        assert_eq!(value.validate(), Err(PortableSubjectRefError::InvalidDigest));
    }

    #[test]
    fn serde_round_trip_preserves_raw_tuple() {
        let value = profile(plan_ref()).to_portable_exact_ref().unwrap();
        let encoded = serde_json::to_string(&value).unwrap();
        let decoded: PortableExactSubjectRefV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(value, decoded);
    }
}
