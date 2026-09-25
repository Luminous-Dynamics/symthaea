// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Portable exact-subject envelope for canonical manufacturing process plans.
//!
//! Cross-repository equality is the raw semantic tuple only:
//!
//! ```text
//! { namespace, subject_id, semantic_version, content_blake3 }
//! ```
//!
//! A valid portable reference does not prove plan feasibility, resource availability, execution,
//! or external admission.

use serde::{Deserialize, Serialize};
use symthaea_manufacturing_contracts::ProcessPlanV1;
use thiserror::Error;

pub const PROCESS_PLAN_NAMESPACE: &str = "symthaea.manufacturing.process-plan";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ManufacturingXrefError {
    #[error("{field} must be canonical and non-empty")]
    NonCanonical { field: &'static str },
    #[error("subject/content digest must be lowercase 64-character hexadecimal BLAKE3")]
    InvalidDigest,
    #[error("canonical V1 portable subject requires subject_id == content_blake3")]
    SubjectDigestMismatch,
    #[error("source plan invalid: {0}")]
    InvalidSource(String),
}

fn canonical(field: &'static str, value: &str) -> Result<(), ManufacturingXrefError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(ManufacturingXrefError::NonCanonical { field });
    }
    Ok(())
}

fn digest64(value: &str) -> Result<(), ManufacturingXrefError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ManufacturingXrefError::InvalidDigest);
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
    pub fn validate(&self) -> Result<(), ManufacturingXrefError> {
        canonical("portable.namespace", &self.namespace)?;
        canonical("portable.semantic_version", &self.semantic_version)?;
        digest64(&self.subject_id)?;
        digest64(&self.content_blake3)?;
        if self.subject_id != self.content_blake3 {
            return Err(ManufacturingXrefError::SubjectDigestMismatch);
        }
        Ok(())
    }
}

pub trait ToPortableProcessPlanRefV1 {
    fn to_portable_process_plan_ref(&self)
        -> Result<PortableExactSubjectRefV1, ManufacturingXrefError>;
}

impl ToPortableProcessPlanRefV1 for ProcessPlanV1 {
    fn to_portable_process_plan_ref(
        &self,
    ) -> Result<PortableExactSubjectRefV1, ManufacturingXrefError> {
        let id = self
            .plan_id()
            .map_err(|err| ManufacturingXrefError::InvalidSource(err.to_string()))?;
        let value = PortableExactSubjectRefV1 {
            namespace: PROCESS_PLAN_NAMESPACE.into(),
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
    use symthaea_manufacturing_contracts::{PlanNodeKindV1, ProcessPlanNodeV1};
    use symthaea_manufacturing_process::ProcessDefinitionId;

    fn plan() -> ProcessPlanV1 {
        ProcessPlanV1 {
            semantic_version: "1.0.0".into(),
            nodes: vec![ProcessPlanNodeV1 {
                node_id: "machine".into(),
                kind: PlanNodeKindV1::ProcessStep {
                    process_id: ProcessDefinitionId("process:precision-machining-v1".into()),
                    recipe_or_commitment_ref: Some("recipe:commitment:1".into()),
                    capability_requirement_ref: "capability:req:1".into(),
                    input_state_refs: vec!["state:input:1".into()],
                    output_state_refs: vec!["state:output:1".into()],
                },
                display_label: Some("Machine coupon".into()),
                ui_x: Some(10),
                ui_y: Some(20),
            }],
            edges: vec![],
            display_label: Some("Precision coupon plan".into()),
        }
    }

    #[test]
    fn portable_ref_delegates_to_exact_plan_id() {
        let value = plan();
        let reference = value.to_portable_process_plan_ref().unwrap();
        assert_eq!(reference.namespace, PROCESS_PLAN_NAMESPACE);
        assert_eq!(reference.subject_id, value.plan_id().unwrap());
        assert_eq!(reference.content_blake3, reference.subject_id);
    }

    #[test]
    fn ui_metadata_does_not_change_portable_ref() {
        let a = plan();
        let mut b = a.clone();
        b.display_label = Some("Renamed plan".into());
        b.nodes[0].display_label = Some("Renamed node".into());
        b.nodes[0].ui_x = Some(900);
        b.nodes[0].ui_y = Some(-400);
        assert_eq!(
            a.to_portable_process_plan_ref().unwrap(),
            b.to_portable_process_plan_ref().unwrap()
        );
    }

    #[test]
    fn semantic_state_change_changes_portable_ref() {
        let a = plan();
        let mut b = a.clone();
        if let PlanNodeKindV1::ProcessStep {
            output_state_refs, ..
        } = &mut b.nodes[0].kind
        {
            output_state_refs[0] = "state:output:2".into();
        }
        assert_ne!(
            a.to_portable_process_plan_ref().unwrap(),
            b.to_portable_process_plan_ref().unwrap()
        );
    }

    #[test]
    fn malformed_portable_tuple_fails_closed() {
        let mut value = plan().to_portable_process_plan_ref().unwrap();
        value.content_blake3 = "a".repeat(64);
        assert_eq!(
            value.validate(),
            Err(ManufacturingXrefError::SubjectDigestMismatch)
        );
        value.content_blake3 = value.subject_id.to_uppercase();
        assert_eq!(value.validate(), Err(ManufacturingXrefError::InvalidDigest));
    }

    #[test]
    fn serde_round_trip_preserves_raw_tuple() {
        let value = plan().to_portable_process_plan_ref().unwrap();
        let encoded = serde_json::to_string(&value).unwrap();
        let decoded: PortableExactSubjectRefV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(value, decoded);
    }
}
