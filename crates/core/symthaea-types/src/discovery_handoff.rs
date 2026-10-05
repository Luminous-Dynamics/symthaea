// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-neutral handoff envelope for independently evaluating scientific hypotheses.
//!
//! This type intentionally carries identities and provenance, not EUREKA outcomes.

use serde::{Deserialize, Serialize};
use crate::physical_type::{ModelMaturity, PhysicalType};

pub const HYPOTHESIS_HANDOFF_SCHEMA: &str = "SCIENTIFIC_HYPOTHESIS_HANDOFF.v1";

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
            ("discovery_source", self.discovery_source.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{label} cannot be empty"));
            }
        }
        if self.non_claims.is_empty() {
            return Err("hypothesis handoff requires explicit non-claims".into());
        }
        if self.non_claims.iter().any(|claim| claim.trim().is_empty()) {
            return Err("hypothesis handoff contains an empty non-claim".into());
        }
        if let Some(digest) = &self.challenge_manifest_digest
            && digest.trim().is_empty()
        {
            return Err("challenge manifest digest cannot be empty".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
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
    fn empty_non_claims_fail_closed() {
        let h = ScientificHypothesisHandoff::new(
            "candidate", &energy(), ModelMaturity::ResearchPrototype,
            "obs", "search", 1, "manifest", 1, "ramanujan", vec![],
        );
        assert!(h.validate().is_err());
    }
}
