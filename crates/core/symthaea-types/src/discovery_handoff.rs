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
    /// Canonical physical semantics carried with the hypothesis.
    pub physical_type: PhysicalType,
    /// Digest binding the exact physical-type bytes.
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
            physical_type: physical_type.clone(),
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
        self.physical_type
            .validate()
            .map_err(|error| format!("invalid carried physical type: {}", error.reason))?;
        if self.physical_type_digest != self.physical_type.digest_hex() {
            return Err("physical_type_digest does not match carried physical_type".into());
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



/// Compute an order-sensitive identity for the exact handoff list supplied to
/// an independent challenge. Ordering is bound because deterministic tie-breaking
/// can depend on the presented hypothesis order.
pub fn scientific_hypothesis_set_digest(handoffs: &[ScientificHypothesisHandoff]) -> String {
    let mut canonical = String::from("hypothesis-set-v1|count=");
    canonical.push_str(&handoffs.len().to_string());
    canonical.push('|');
    for (index, handoff) in handoffs.iter().enumerate() {
        canonical.push_str(&index.to_string());
        canonical.push(':');
        canonical.push_str(&handoff.digest_hex());
        canonical.push(';');
    }
    blake3::hash(canonical.as_bytes()).to_hex().to_string()
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

    pub fn validate_against(
        &self,
        prior: &ScientificHypothesisHandoff,
        new: &ScientificHypothesisHandoff,
    ) -> Result<(), String> {
        self.validate()?;
        if self.prior_handoff_digest != prior.digest_hex() {
            return Err("revision receipt references a different prior handoff".into());
        }
        if self.new_handoff_digest != new.digest_hex() {
            return Err("revision receipt references a different new handoff".into());
        }
        Ok(())
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
    /// Prediction-disagreement score used to select the challenge; this is not realized evidence.
    ///
    /// Stored as the exact IEEE-754 bit pattern so the receipt remains deterministic.
    pub predicted_disagreement_score_bits: u64,
    /// Number of hypotheses that produced finite predictions for the selected challenge.
    ///
    /// Active inquiry is only admissible when at least two hypotheses are directly
    /// comparable at the selected challenge.
    pub prediction_count: u32,
}

impl ScientificInquirySelectionReceipt {
    pub const SCHEMA_REVISION: &'static str = "SCIENTIFIC_INQUIRY_SELECTION.v2";

    pub fn new(
        hypothesis_handoff_digest: impl Into<String>,
        hypothesis_set_digest: impl Into<String>,
        challenge_space_digest: impl Into<String>,
        selected_challenge_digest: impl Into<String>,
        selector_revision: impl Into<String>,
        selection_seed: u64,
        predicted_disagreement_score: f64,
        prediction_count: u32,
    ) -> Result<Self, String> {
        if !predicted_disagreement_score.is_finite() || predicted_disagreement_score < 0.0 {
            return Err("predicted disagreement score must be finite and non-negative".into());
        }
        if prediction_count < 2 {
            return Err("inquiry selection requires at least two finite hypothesis predictions".into());
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
            predicted_disagreement_score_bits: predicted_disagreement_score.to_bits(),
            prediction_count,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn predicted_disagreement_score(&self) -> f64 {
        f64::from_bits(self.predicted_disagreement_score_bits)
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

    pub fn validate_against(
        &self,
        hypothesis_handoff: &ScientificHypothesisHandoff,
        hypothesis_set_digest: &str,
    ) -> Result<(), String> {
        self.validate()?;
        if self.hypothesis_handoff_digest != hypothesis_handoff.digest_hex() {
            return Err("inquiry selection is bound to a different hypothesis handoff".into());
        }
        if self.hypothesis_set_digest != hypothesis_set_digest {
            return Err("inquiry selection is bound to a different hypothesis set".into());
        }
        Ok(())
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
        ] {
            validate_digest(label, value)?;
        }
        if self.selector_revision.trim().is_empty() {
            return Err("selector_revision cannot be empty".into());
        }
        let predicted = self.predicted_disagreement_score();
        if !predicted.is_finite() || predicted < 0.0 {
            return Err("predicted disagreement score is invalid".into());
        }
        if self.prediction_count < 2 {
            return Err("inquiry selection requires at least two finite hypothesis predictions".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hypothesis_revision_cross_link_validation_rejects_wrong_handoff() {
        let energy = energy();
        let prior = ScientificHypothesisHandoff::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &energy,
            ModelMaturity::ResearchPrototype,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            1,
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            1,
            "ramanujan",
            vec!["none".into()],
        );
        let new = ScientificHypothesisHandoff::new(
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            &energy,
            ModelMaturity::ValidatedNumerical,
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "1111111111111111111111111111111111111111111111111111111111111111",
            2,
            "2222222222222222222222222222222222222222222222222222222222222222",
            1,
            "ramanujan",
            vec!["none".into()],
        );
        let receipt = ScientificHypothesisRevisionReceipt::new(
            prior.digest_hex(),
            new.digest_hex(),
            "3333333333333333333333333333333333333333333333333333333333333333",
            "4444444444444444444444444444444444444444444444444444444444444444",
            "new evidence",
        )
        .unwrap();
        assert!(receipt.validate_against(&prior, &new).is_ok());
        assert!(receipt.validate_against(&new, &prior).is_err());
    }

    #[test]
    fn hypothesis_set_digest_binds_order_and_membership() {
        let a = ScientificHypothesisHandoff::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &energy(),
            ModelMaturity::ResearchPrototype,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            1,
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            1,
            "ramanujan",
            vec!["none".into()],
        );
        let b = ScientificHypothesisHandoff::new(
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            &energy(),
            ModelMaturity::ResearchPrototype,
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "1111111111111111111111111111111111111111111111111111111111111111",
            2,
            "2222222222222222222222222222222222222222222222222222222222222222",
            1,
            "ramanujan",
            vec!["none".into()],
        );
        assert_ne!(
            scientific_hypothesis_set_digest(&[a.clone(), b.clone()]),
            scientific_hypothesis_set_digest(&[b, a])
        );
    }

    #[test]
    fn inquiry_selection_cross_link_validation_rejects_wrong_handoff() {
        let energy = energy();
        let h1 = ScientificHypothesisHandoff::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &energy,
            ModelMaturity::ResearchPrototype,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            1,
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            1,
            "ramanujan",
            vec!["none".into()],
        );
        let h2 = h1.clone().with_challenge_manifest(
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        );
        let receipt = ScientificInquirySelectionReceipt::new(
            h1.digest_hex(),
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "1111111111111111111111111111111111111111111111111111111111111111",
            "2222222222222222222222222222222222222222222222222222222222222222",
            "selector-v2",
            1,
            1.0,
            2,
        )
        .unwrap();
        assert!(receipt
            .validate_against(&h1, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
            .is_ok());
        assert!(receipt
            .validate_against(&h2, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
            .is_err());
    }

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
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "selector-v2",
            9,
            2.5,
            3,
        )
        .unwrap();
        assert_eq!(
            receipt.scope,
            InquirySelectionScope::ExperimentSelectionOnly
        );
        assert!((receipt.predicted_disagreement_score() - 2.5).abs() < f64::EPSILON);
        assert_eq!(receipt.prediction_count, 3);
        assert!(receipt.validate().is_ok());
    }


    use super::*;
    use crate::physical_type::{PhysicalDimension, QuantityKind};

    fn energy() -> PhysicalType {
        PhysicalType::with_kind(QuantityKind::Energy, PhysicalDimension::ENERGY)
    }

    #[test]
    fn inquiry_selection_rejects_insufficient_prediction_coverage() {
        let result = ScientificInquirySelectionReceipt::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "selector-v2",
            9,
            2.5,
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn handoff_is_deterministic_and_binds_type_identity() {
        let h = ScientificHypothesisHandoff::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &energy(),
            ModelMaturity::ResearchPrototype,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            42,
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            7,
            "ramanujan",
            vec!["not an empirical confirmation".into()],
        );
        assert!(h.validate().is_ok());
        assert!(h.verify_physical_type(&energy()));
        assert_eq!(h.digest_hex(), h.clone().digest_hex());
    }

    #[test]
    fn physical_type_digest_binds_carried_type() {
        let mut h = ScientificHypothesisHandoff::new(
            "candidate-a", &energy(), ModelMaturity::ResearchPrototype,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            1,
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            1,
            "ramanujan",
            vec!["none".into()],
        );
        h.physical_type = PhysicalType::dimensionless();
        assert!(h.validate().is_err());
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
