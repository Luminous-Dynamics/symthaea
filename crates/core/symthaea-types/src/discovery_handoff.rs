// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-neutral handoff envelope for independently evaluating scientific hypotheses.
//!
//! This type intentionally carries identities and provenance, not EUREKA outcomes.

use serde::{Deserialize, Serialize};
use crate::physical_type::{ModelMaturity, PhysicalType};

pub const HYPOTHESIS_HANDOFF_SCHEMA: &str = "SCIENTIFIC_HYPOTHESIS_HANDOFF.v1";

fn validate_digest(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{label} must be canonical lowercase 64-hex"));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificHypothesisHandoff {
    pub schema_revision: String,
    /// Digest of the canonical candidate/model representation.
    pub candidate_digest: String,
    /// Canonical physical semantic identity.
    pub physical_type_digest: String,
    pub model_maturity: ModelMaturity,
    pub source_observation_digest: String,
    pub search_configuration_digest: String,
    pub search_seed: u64,
    pub discovery_manifest_digest: String,
    pub candidate_complexity: u64,
    /// Human-readable origin label; not an authority identity.
    pub discovery_source: String,
    /// Explicit scope limits carried with the hypothesis.
    pub non_claims: Vec<String>,
    /// Optional immutable EUREKA challenge-manifest commitment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub challenge_manifest_digest: Option<String>,
}

impl ScientificHypothesisHandoff {
    pub fn new(
        candidate_digest: impl Into<String>,
        physical_type: &PhysicalType,
        model_maturity: ModelMaturity,
        source_observation_digest: impl Into<String>,
        search_configuration_digest: impl Into<String>,
        search_seed: u64,
        discovery_manifest_digest: impl Into<String>,
        candidate_complexity: u64,
        discovery_source: impl Into<String>,
        non_claims: Vec<String>,
    ) -> Self {
        Self {
            schema_revision: HYPOTHESIS_HANDOFF_SCHEMA.into(),
            candidate_digest: candidate_digest.into(),
            physical_type_digest: physical_type.digest_hex(),
            model_maturity,
            source_observation_digest: source_observation_digest.into(),
            search_configuration_digest: search_configuration_digest.into(),
            search_seed,
            discovery_manifest_digest: discovery_manifest_digest.into(),
            candidate_complexity,
            discovery_source: discovery_source.into(),
            non_claims,
            challenge_manifest_digest: None,
        }
    }

    pub fn with_challenge_manifest(mut self, digest: impl Into<String>) -> Self {
        self.challenge_manifest_digest = Some(digest.into());
        self
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("hypothesis handoff serialization must be infallible")
    }

    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_bytes()).as_bytes()
    }

    pub fn digest_hex(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }

    /// Verify that the handoff binds the current physical semantic identity.
    pub fn verify_physical_type(&self, physical_type: &PhysicalType) -> bool {
        self.physical_type_digest == physical_type.digest_hex()
    }

    /// Validates identity/provenance fields without interpreting scientific truth.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_revision != HYPOTHESIS_HANDOFF_SCHEMA {
            return Err("unsupported hypothesis handoff schema revision".into());
        }
        for (label, value) in [
            ("candidate_digest", self.candidate_digest.as_str()),
            ("physical_type_digest", self.physical_type_digest.as_str()),
            ("source_observation_digest", self.source_observation_digest.as_str()),
            ("search_configuration_digest", self.search_configuration_digest.as_str()),
            ("discovery_manifest_digest", self.discovery_manifest_digest.as_str()),
        ] {
            validate_digest(label, value)?;
        }
        if self.discovery_source.trim().is_empty() {
            return Err("discovery_source cannot be empty".into());
        }
        if self.non_claims.is_empty() {
            return Err("hypothesis handoff requires explicit non-claims".into());
        }
        if self.non_claims.iter().any(|claim| claim.trim().is_empty()) {
            return Err("hypothesis handoff contains an empty non-claim".into());
        }
        if let Some(digest) = &self.challenge_manifest_digest {
            validate_digest("challenge_manifest_digest", digest)?;
        }
        Ok(())
    }
}



/// Evidence-neutral lineage record for revising a scientific hypothesis into
/// a new discovery campaign. The prior handoff remains immutable by identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificHypothesisRevisionReceipt {
    pub schema_revision: String,
    pub prior_handoff_digest: String,
    pub new_handoff_digest: String,
    /// Digest of the closed evidence/result that motivated revision.
    pub motivating_evidence_digest: String,
    /// Identity of the new Ramanujan campaign/search.
    pub new_campaign_digest: String,
    pub revision_reason: String,
}

impl ScientificHypothesisRevisionReceipt {
    pub const SCHEMA_REVISION: &'static str = "SCIENTIFIC_HYPOTHESIS_REVISION.v1";

    pub fn new(
        prior_handoff_digest: impl Into<String>,
        new_handoff_digest: impl Into<String>,
        motivating_evidence_digest: impl Into<String>,
        new_campaign_digest: impl Into<String>,
        revision_reason: impl Into<String>,
    ) -> Result<Self, String> {
        let receipt = Self {
            schema_revision: Self::SCHEMA_REVISION.into(),
            prior_handoff_digest: prior_handoff_digest.into(),
            new_handoff_digest: new_handoff_digest.into(),
            motivating_evidence_digest: motivating_evidence_digest.into(),
            new_campaign_digest: new_campaign_digest.into(),
            revision_reason: revision_reason.into(),
        };
        receipt.validate()?;
        if receipt.prior_handoff_digest == receipt.new_handoff_digest {
            return Err("hypothesis revision must create a new handoff identity".into());
        }
        Ok(receipt)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("hypothesis revision serialization must be infallible")
    }

    pub fn digest_hex(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_revision != Self::SCHEMA_REVISION {
            return Err("unsupported hypothesis revision schema revision".into());
        }
        for (label, value) in [
            ("prior_handoff_digest", self.prior_handoff_digest.as_str()),
            ("new_handoff_digest", self.new_handoff_digest.as_str()),
            ("motivating_evidence_digest", self.motivating_evidence_digest.as_str()),
            ("new_campaign_digest", self.new_campaign_digest.as_str()),
        ] {
            validate_digest(label, value)?;
        }
        if self.revision_reason.trim().is_empty() {
            return Err("revision_reason cannot be empty".into());
        }
        Ok(())
    }
}

/// Scope of a scientific inquiry-selection receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InquirySelectionScope {
    /// Records only a decision about which fresh experiment to run.
    ExperimentSelectionOnly,
}

/// Evidence-neutral record of an experiment selected to discriminate hypotheses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificInquirySelectionReceipt {
    pub schema_revision: String,
    pub scope: InquirySelectionScope,
    pub hypothesis_handoff_digest: String,
    pub hypothesis_set_digest: String,
    pub challenge_space_digest: String,
    pub selected_challenge_digest: String,
    pub selector_revision: String,
    pub selection_seed: u64,
    /// Predicted disagreement / information value; this is not realized evidence.
    pub predicted_information_gain_bits: u64,
}

impl ScientificInquirySelectionReceipt {
    pub const SCHEMA_REVISION: &'static str = "SCIENTIFIC_INQUIRY_SELECTION.v1";

    pub fn new(
        hypothesis_handoff_digest: impl Into<String>,
        hypothesis_set_digest: impl Into<String>,
        challenge_space_digest: impl Into<String>,
        selected_challenge_digest: impl Into<String>,
        selector_revision: impl Into<String>,
        selection_seed: u64,
        predicted_information_gain: f64,
    ) -> Result<Self, String> {
        if !predicted_information_gain.is_finite() || predicted_information_gain < 0.0 {
            return Err("predicted information gain must be finite and non-negative".into());
        }
        let receipt = Self {
            schema_revision: Self::SCHEMA_REVISION.into(),
            scope: InquirySelectionScope::ExperimentSelectionOnly,
            hypothesis_handoff_digest: hypothesis_handoff_digest.into(),
            hypothesis_set_digest: hypothesis_set_digest.into(),
            challenge_space_digest: challenge_space_digest.into(),
            selected_challenge_digest: selected_challenge_digest.into(),
            selector_revision: selector_revision.into(),
            selection_seed,
            predicted_information_gain_bits: predicted_information_gain.to_bits(),
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn predicted_information_gain(&self) -> f64 {
        f64::from_bits(self.predicted_information_gain_bits)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("inquiry selection serialization must be infallible")
    }

    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_bytes()).as_bytes()
    }

    pub fn digest_hex(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_revision != Self::SCHEMA_REVISION {
            return Err("unsupported inquiry selection schema revision".into());
        }
        if self.scope != InquirySelectionScope::ExperimentSelectionOnly {
            return Err("unsupported inquiry selection scope".into());
        }
        for (label, value) in [
            ("hypothesis_handoff_digest", self.hypothesis_handoff_digest.as_str()),
            ("hypothesis_set_digest", self.hypothesis_set_digest.as_str()),
            ("challenge_space_digest", self.challenge_space_digest.as_str()),
            ("selected_challenge_digest", self.selected_challenge_digest.as_str()),
            ("selector_revision", self.selector_revision.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{label} cannot be empty"));
            }
        }
        let predicted = self.predicted_information_gain();
        if !predicted.is_finite() || predicted < 0.0 {
            return Err("predicted information gain is invalid".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hypothesis_revision_requires_a_new_campaign_identity() {
        let receipt = ScientificHypothesisRevisionReceipt::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "heldout contradiction",
        )
        .unwrap();
        assert!(receipt.validate().is_ok());
        assert_ne!(receipt.prior_handoff_digest, receipt.new_handoff_digest);

        assert!(ScientificHypothesisRevisionReceipt::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "same hypothesis",
        ).is_err());
    }

    #[test]
    fn inquiry_selection_is_explicitly_not_an_outcome() {
        let receipt = ScientificInquirySelectionReceipt::new(
            "handoff",
            "hypotheses",
            "challenge-space",
            "experiment-7",
            "selector-v1",
            9,
            2.5,
        )
        .unwrap();
        assert_eq!(
            receipt.scope,
            InquirySelectionScope::ExperimentSelectionOnly
        );
        assert!((receipt.predicted_information_gain() - 2.5).abs() < f64::EPSILON);
        assert!(receipt.validate().is_ok());
    }


    use super::*;
    use crate::physical_type::{PhysicalDimension, QuantityKind};

    fn energy() -> PhysicalType {
        PhysicalType::with_kind(QuantityKind::Energy, PhysicalDimension::ENERGY)
    }

    #[test]
    fn handoff_is_deterministic_and_binds_type_identity() {
        let h = ScientificHypothesisHandoff::new(
            "candidate-1",
            &energy(),
            ModelMaturity::ResearchPrototype,
            "obs-1",
            "search-1",
            42,
            "manifest-1",
            7,
            "ramanujan",
            vec!["not an empirical confirmation".into()],
        );
        assert!(h.validate().is_ok());
        assert!(h.verify_physical_type(&energy()));
        assert_eq!(h.digest_hex(), h.clone().digest_hex());
    }

    #[test]
    fn changed_candidate_identity_changes_handoff_digest() {
        let a = ScientificHypothesisHandoff::new(
            "candidate-a", &energy(), ModelMaturity::ResearchPrototype,
            "obs", "search", 1, "manifest", 1, "ramanujan", vec!["none".into()],
        );
        let b = ScientificHypothesisHandoff::new(
            "candidate-b", &energy(), ModelMaturity::ResearchPrototype,
            "obs", "search", 1, "manifest", 1, "ramanujan", vec!["none".into()],
        );
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn malformed_digest_fails_closed() {
        let mut h = ScientificHypothesisHandoff::new(
            "candidate", &energy(), ModelMaturity::ResearchPrototype,
            "obs", "search", 1, "manifest", 1, "ramanujan", vec!["none".into()],
        );
        h.candidate_digest = "not-a-digest".into();
        assert!(h.validate().is_err());
    }

    #[test]
    fn empty_non_claims_fail_closed() {
        let h = ScientificHypothesisHandoff::new(
            "candidate", &energy(), ModelMaturity::ResearchPrototype,
            "obs", "search", 1, "manifest", 1, "ramanujan", vec![],
        );
        assert!(h.validate().is_err());
    }
}
