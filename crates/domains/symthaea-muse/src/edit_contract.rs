// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned, pre-render contract schema for symbolic/audio editing.
//!
//! This module intentionally has no dependency on Symthaea's learned cognitive
//! systems. Planners may propose a contract; deterministic callers validate,
//! freeze, and enforce it. A valid hash commits to the contract bytes, but
//! does not prove that an obligation is correct, that a profile is approved,
//! or that a renderer executed the requested edit.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const FROZEN_EDIT_CONTRACT_VERSION: &str = "mel-mr-eval-010-frozen-edit-contract-v0.1";
pub const MAX_CONTRACT_OBLIGATIONS: usize = 256;
const MAX_IDENTIFIER_BYTES: usize = 256;
const MAX_VALUE_BYTES: usize = 1024;

/// Musical or provenance dimension to which an obligation applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditAttribute {
    Pitch,
    Onset,
    Duration,
    RmsEnvelope,
    Timbre,
    Harmony,
    Rhythm,
    Structure,
    Tempo,
    Meter,
    EventIdentity,
    NeighboringContent,
    RendererConfig,
}

/// Scope of an obligation. Frame intervals are half-open: [start_frame, end_frame).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum ObligationScope {
    Event { event_id: String },
    FrameInterval { start_frame: u64, end_frame: u64 },
    ScoreWide,
    ProtectedRemainder,
}

/// Explicitly scaled values avoid ambiguous floating-point thresholds in receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "value_type", rename_all = "snake_case")]
pub enum AttributeValue {
    Numeric {
        value_microunits: i64,
        unit: String,
    },
    Text {
        value: String,
    },
    Digest {
        sha256: String,
    },
    Boolean {
        value: bool,
    },
    Meter {
        numerator: u8,
        denominator: u8,
    },
}

/// A quantized tolerance whose comparator defines the measured quantity and scale.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizedTolerance {
    pub value_microunits: u64,
    pub unit: String,
}

/// What evidence must establish for one attribute and scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "requirement", rename_all = "snake_case")]
pub enum ObligationRequirement {
    ChangeTo { expected: AttributeValue },
    ExactlyMatches { expected: AttributeValue },
    PreserveWithin {
        comparator_id: String,
        tolerance: QuantizedTolerance,
    },
    BindDigest { expected_sha256: String },
}

/// One immutable obligation emitted by the contract compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditObligation {
    pub obligation_id: String,
    pub attribute: EditAttribute,
    pub scope: ObligationScope,
    pub requirement: ObligationRequirement,
    /// Critical obligations are never compensated for by scores on other obligations.
    pub critical: bool,
}

/// A symbolic edit plan's frozen constraints, created before audio rendering.
///
/// target_score_sha256 commits to the exact target Score chosen by the plan.
/// It is not a claim that the renderer actually used that Score; the execution
/// adapter and output-buffer evidence must establish that separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenEditContract {
    pub contract_version: String,
    pub task_id: String,
    pub operation_id: String,
    pub instruction_sha256: String,
    pub source_score_sha256: String,
    pub target_score_sha256: String,
    pub profile_id: String,
    pub profile_sha256: String,
    pub obligations: Vec<EditObligation>,
    /// SHA-256 over the canonical contract with this field blank.
    pub contract_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditContractError {
    UnsupportedVersion,
    InvalidField(&'static str),
    InvalidSha256(&'static str),
    TooManyObligations,
    NoObligations,
    NoCriticalObligations,
    DuplicateObligationId,
    InvalidScope(&'static str),
    InvalidRequirement(&'static str),
    SerializationFailure,
}

impl std::fmt::Display for EditContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion => write!(f, "unsupported frozen edit-contract version"),
            Self::InvalidField(field) => write!(f, "invalid edit-contract field: {field}"),
            Self::InvalidSha256(field) => {
                write!(f, "invalid lowercase SHA-256 digest in field: {field}")
            }
            Self::TooManyObligations => write!(f, "edit contract exceeds the obligation limit"),
            Self::NoObligations => write!(f, "edit contract requires at least one obligation"),
            Self::NoCriticalObligations => {
                write!(f, "edit contract requires at least one critical obligation")
            }
            Self::DuplicateObligationId => write!(f, "duplicate edit-obligation ID"),
            Self::InvalidScope(reason) => write!(f, "invalid edit-obligation scope: {reason}"),
            Self::InvalidRequirement(reason) => {
                write!(f, "invalid edit-obligation requirement: {reason}")
            }
            Self::SerializationFailure => write!(f, "could not serialize edit-contract receipt"),
        }
    }
}

impl std::error::Error for EditContractError {}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_label(value: &str, max_bytes: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_bytes
        && !value.chars().any(char::is_control)
}

impl FrozenEditContract {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task_id: String,
        operation_id: String,
        instruction_sha256: String,
        source_score_sha256: String,
        target_score_sha256: String,
        profile_id: String,
        profile_sha256: String,
        obligations: Vec<EditObligation>,
    ) -> Self {
        Self {
            contract_version: FROZEN_EDIT_CONTRACT_VERSION.to_string(),
            task_id,
            operation_id,
            instruction_sha256,
            source_score_sha256,
            target_score_sha256,
            profile_id,
            profile_sha256,
            obligations,
            contract_sha256: String::new(),
        }
    }

    /// Validate, canonicalize obligation order by ID, and seal this contract.
    ///
    /// Sealing is a content-integrity operation only. Callers must separately
    /// resolve the profile through a trusted registry and validate that the
    /// operation-to-obligation mapping is sound for the supported edit type.
    pub fn seal(mut self) -> Result<Self, EditContractError> {
        self.contract_sha256.clear();
        self.obligations
            .sort_by(|left, right| left.obligation_id.cmp(&right.obligation_id));
        self.validate_fields(false)?;
        self.contract_sha256 = self.compute_sha256()?;
        Ok(self)
    }

    /// Revalidate the fields and recompute the canonical digest.
    ///
    /// Ok(true) means the serialized contract is internally consistent, not
    /// that its obligations are correct or that the referenced profile is trusted.
    pub fn verify(&self) -> Result<bool, EditContractError> {
        self.validate_fields(true)?;
        Ok(self.compute_sha256()? == self.contract_sha256)
    }

    fn compute_sha256(&self) -> Result<String, EditContractError> {
        let mut canonical = self.clone();
        canonical.contract_sha256.clear();
        canonical
            .obligations
            .sort_by(|left, right| left.obligation_id.cmp(&right.obligation_id));
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|_| EditContractError::SerializationFailure)?;
        let mut hasher = Sha256::new();
        hasher.update(b"luminous-dynamics-frozen-edit-contract-v1\0");
        hasher.update(bytes);
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn validate_fields(&self, require_seal: bool) -> Result<(), EditContractError> {
        if self.contract_version != FROZEN_EDIT_CONTRACT_VERSION {
            return Err(EditContractError::UnsupportedVersion);
        }
        for (name, value) in [
            ("task_id", self.task_id.as_str()),
            ("operation_id", self.operation_id.as_str()),
            ("profile_id", self.profile_id.as_str()),
        ] {
            if !valid_label(value, MAX_IDENTIFIER_BYTES) {
                return Err(EditContractError::InvalidField(name));
            }
        }
        for (name, value) in [
            ("instruction_sha256", self.instruction_sha256.as_str()),
            ("source_score_sha256", self.source_score_sha256.as_str()),
            ("target_score_sha256", self.target_score_sha256.as_str()),
            ("profile_sha256", self.profile_sha256.as_str()),
        ] {
            if !is_lower_hex_sha256(value) {
                return Err(EditContractError::InvalidSha256(name));
            }
        }
        if require_seal && !is_lower_hex_sha256(&self.contract_sha256) {
            return Err(EditContractError::InvalidSha256("contract_sha256"));
        }
        if self.obligations.is_empty() {
            return Err(EditContractError::NoObligations);
        }
        if self.obligations.len() > MAX_CONTRACT_OBLIGATIONS {
            return Err(EditContractError::TooManyObligations);
        }
        if !self.obligations.iter().any(|obligation| obligation.critical) {
            return Err(EditContractError::NoCriticalObligations);
        }

        let mut ids = HashSet::with_capacity(self.obligations.len());
        for obligation in &self.obligations {
            if !valid_label(&obligation.obligation_id, MAX_IDENTIFIER_BYTES) {
                return Err(EditContractError::InvalidField("obligation_id"));
            }
            if !ids.insert(obligation.obligation_id.as_str()) {
                return Err(EditContractError::DuplicateObligationId);
            }
            Self::validate_scope(&obligation.scope)?;
            Self::validate_requirement(&obligation.requirement)?;
        }
        Ok(())
    }

    fn validate_scope(scope: &ObligationScope) -> Result<(), EditContractError> {
        match scope {
            ObligationScope::Event { event_id } => {
                if !valid_label(event_id, MAX_IDENTIFIER_BYTES) {
                    return Err(EditContractError::InvalidScope("event ID must be non-empty"));
                }
            }
            ObligationScope::FrameInterval {
                start_frame,
                end_frame,
            } => {
                if start_frame >= end_frame {
                    return Err(EditContractError::InvalidScope(
                        "frame intervals must be non-empty and half-open",
                    ));
                }
            }
            ObligationScope::ScoreWide | ObligationScope::ProtectedRemainder => {}
        }
        Ok(())
    }

    fn validate_requirement(
        requirement: &ObligationRequirement,
    ) -> Result<(), EditContractError> {
        match requirement {
            ObligationRequirement::ChangeTo { expected }
            | ObligationRequirement::ExactlyMatches { expected } => {
                Self::validate_attribute_value(expected)
            }
            ObligationRequirement::PreserveWithin {
                comparator_id,
                tolerance,
            } => {
                if !valid_label(comparator_id, MAX_IDENTIFIER_BYTES) {
                    return Err(EditContractError::InvalidRequirement(
                        "comparator ID must be non-empty",
                    ));
                }
                if !valid_label(&tolerance.unit, MAX_IDENTIFIER_BYTES) {
                    return Err(EditContractError::InvalidRequirement(
                        "tolerance unit must be non-empty",
                    ));
                }
                Ok(())
            }
            ObligationRequirement::BindDigest { expected_sha256 } => {
                if !is_lower_hex_sha256(expected_sha256) {
                    return Err(EditContractError::InvalidSha256("expected_sha256"));
                }
                Ok(())
            }
        }
    }

    fn validate_attribute_value(value: &AttributeValue) -> Result<(), EditContractError> {
        match value {
            AttributeValue::Numeric {
                value_microunits: _,
                unit,
            } => {
                if !valid_label(unit, MAX_IDENTIFIER_BYTES) {
                    return Err(EditContractError::InvalidRequirement(
                        "numeric value unit must be non-empty",
                    ));
                }
            }
            AttributeValue::Text { value } => {
                if !valid_label(value, MAX_VALUE_BYTES) {
                    return Err(EditContractError::InvalidRequirement(
                        "text attribute value must be non-empty",
                    ));
                }
            }
            AttributeValue::Digest { sha256 } => {
                if !is_lower_hex_sha256(sha256) {
                    return Err(EditContractError::InvalidSha256("attribute_value.sha256"));
                }
            }
            AttributeValue::Boolean { value: _ } => {}
            AttributeValue::Meter {
                numerator,
                denominator,
            } => {
                if *numerator == 0 || *denominator == 0 {
                    return Err(EditContractError::InvalidRequirement(
                        "meter numerator and denominator must be non-zero",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat(character).take(64).collect()
    }

    fn pitch_obligation(id: &str) -> EditObligation {
        EditObligation {
            obligation_id: id.to_string(),
            attribute: EditAttribute::Pitch,
            scope: ObligationScope::Event {
                event_id: "note-17".to_string(),
            },
            requirement: ObligationRequirement::ChangeTo {
                expected: AttributeValue::Numeric {
                    value_microunits: 600_000_000,
                    unit: "micro_semitone".to_string(),
                },
            },
            critical: true,
        }
    }

    fn envelope_obligation(id: &str) -> EditObligation {
        EditObligation {
            obligation_id: id.to_string(),
            attribute: EditAttribute::RmsEnvelope,
            scope: ObligationScope::FrameInterval {
                start_frame: 1024,
                end_frame: 2048,
            },
            requirement: ObligationRequirement::PreserveWithin {
                comparator_id: "rms-envelope-delta-v1".to_string(),
                tolerance: QuantizedTolerance {
                    value_microunits: 1_000_000,
                    unit: "micro_db".to_string(),
                },
            },
            critical: true,
        }
    }

    fn fixture() -> FrozenEditContract {
        FrozenEditContract::new(
            "task-1".to_string(),
            "pitch-at".to_string(),
            digest('a'),
            digest('b'),
            digest('c'),
            "profile-v1".to_string(),
            digest('d'),
            vec![envelope_obligation("preserve-envelope"), pitch_obligation("change-pitch")],
        )
    }

    #[test]
    fn sealed_contract_verifies_and_canonicalizes_obligation_order() {
        let sealed = fixture().seal().expect("valid contract seals");
        assert!(sealed.verify().expect("valid contract verifies"));
        assert_eq!(sealed.obligations[0].obligation_id, "change-pitch");
        let mut reordered = sealed.clone();
        reordered.obligations.reverse();
        assert!(reordered.verify().expect("obligation ordering is semantic-free"));
    }

    #[test]
    fn changed_obligation_fails_receipt_replay() {
        let mut sealed = fixture().seal().expect("valid contract seals");
        if let ObligationRequirement::PreserveWithin { tolerance, .. } =
            &mut sealed.obligations[1].requirement
        {
            tolerance.value_microunits += 1;
        } else {
            panic!("fixture contains envelope obligation");
        }
        assert!(!sealed.verify().expect("modified but well-formed contract can be checked"));
    }

    #[test]
    fn duplicate_obligation_ids_are_rejected() {
        let mut contract = fixture();
        contract.obligations[1].obligation_id = contract.obligations[0].obligation_id.clone();
        assert_eq!(
            contract.seal().expect_err("duplicate IDs must fail"),
            EditContractError::DuplicateObligationId
        );
    }

    #[test]
    fn empty_or_reversed_frame_interval_is_rejected() {
        let mut contract = fixture();
        contract.obligations[0].scope = ObligationScope::FrameInterval {
            start_frame: 2048,
            end_frame: 1024,
        };
        assert!(matches!(
            contract.seal(),
            Err(EditContractError::InvalidScope(_))
        ));
    }

    #[test]
    fn malformed_digest_is_rejected_before_sealing() {
        let mut contract = fixture();
        contract.profile_sha256 = "not-a-digest".to_string();
        assert_eq!(
            contract.seal().expect_err("malformed profile digest must fail"),
            EditContractError::InvalidSha256("profile_sha256")
        );
    }

    #[test]
    fn contract_without_critical_obligations_is_rejected() {
        let mut contract = fixture();
        for obligation in &mut contract.obligations {
            obligation.critical = false;
        }
        assert_eq!(
            contract.seal().expect_err("all obligations being noncritical must fail"),
            EditContractError::NoCriticalObligations
        );
    }

    #[test]
    fn unsupported_schema_version_is_rejected() {
        let mut contract = fixture();
        contract.contract_version = "future-schema".to_string();
        assert_eq!(
            contract.seal().expect_err("unknown schema versions fail closed"),
            EditContractError::UnsupportedVersion
        );
    }

    #[test]
    fn digest_binding_is_validated() {
        let mut contract = fixture();
        contract.obligations[0].requirement = ObligationRequirement::BindDigest {
            expected_sha256: "bad".to_string(),
        };
        assert_eq!(
            contract.seal().expect_err("bad obligation digest must fail"),
            EditContractError::InvalidSha256("expected_sha256")
        );
    }
}
