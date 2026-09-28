// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Boundary adapter from temporal scientific-discovery candidates to the
//! domain-neutral evidence-plane candidate source.
//!
//! This module intentionally lives in the root Symthaea crate because it
//! depends on discovery-specific identity while the evidence plane remains
//! reusable and temporal-agnostic.

use serde::{Deserialize, Serialize};
use symthaea_evidence_plane::candidate_commitment::{
    BindingError, CandidatePredictionSource,
};

/// Exact identity exported by a temporal discovery candidate.
///
/// source_candidate_id is authoritative. The adapter never regenerates it,
/// so delimiter-bearing identifiers and versioned upstream identity remain
/// intact across the boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalCandidateIdentity {
    pub candidate_id: String,
    pub source_candidate_id: String,
    pub left_model_id: String,
    pub right_model_id: String,
    pub left_lineage: String,
    pub right_lineage: String,
    pub outcome_id: String,
    pub horizon_seconds: f64,
    pub test_specification_id: String,
    pub measurement_specification_id: String,
}

impl TemporalCandidateIdentity {
    pub fn validate(&self) -> Result<(), TemporalAdapterError> {
        for (name, value) in [
            ("candidate_id", &self.candidate_id),
            ("source_candidate_id", &self.source_candidate_id),
            ("left_model_id", &self.left_model_id),
            ("right_model_id", &self.right_model_id),
            ("left_lineage", &self.left_lineage),
            ("right_lineage", &self.right_lineage),
            ("outcome_id", &self.outcome_id),
            ("test_specification_id", &self.test_specification_id),
            ("measurement_specification_id", &self.measurement_specification_id),
        ] {
            if value.trim().is_empty() {
                return Err(TemporalAdapterError::MissingField(name));
            }
        }
        if !self.horizon_seconds.is_finite() {
            return Err(TemporalAdapterError::NonFiniteHorizon);
        }
        Ok(())
    }

    /// Convert only the identity required by the generic commitment layer.
    ///
    /// The complete temporal identity remains available on this value;
    /// CandidatePredictionSource deliberately contains no temporal outcome
    /// semantics. No prediction payload or evidence event is created here.
    pub fn to_candidate_source(&self) -> Result<CandidatePredictionSource, TemporalAdapterError> {
        self.validate()?;
        let source = CandidatePredictionSource {
            candidate_id: self.candidate_id.clone(),
            source_candidate_id: self.source_candidate_id.clone(),
            test_specification_id: self.test_specification_id.clone(),
            measurement_specification_id: self.measurement_specification_id.clone(),
            left_lineage: self.left_lineage.clone(),
            right_lineage: self.right_lineage.clone(),
        };
        source.validate().map_err(TemporalAdapterError::Binding)
            .map(|_| source)
    }

    /// Exact IEEE-754 identity for the horizon, useful to downstream audit
    /// code that must distinguish values with different bit patterns.
    pub fn horizon_bits(&self) -> u64 {
        self.horizon_seconds.to_bits()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalAdapterError {
    MissingField(&'static str),
    NonFiniteHorizon,
    Binding(BindingError),
}

impl std::fmt::Display for TemporalAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "missing temporal candidate field: {field}"),
            Self::NonFiniteHorizon => write!(f, "temporal candidate horizon must be finite"),
            Self::Binding(error) => write!(f, "generic candidate binding rejected: {error:?}"),
        }
    }
}

impl std::error::Error for TemporalAdapterError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> TemporalCandidateIdentity {
        TemporalCandidateIdentity {
            candidate_id: "candidate:1".into(),
            source_candidate_id: "temporal-discrimination-v1:source:a:b".into(),
            left_model_id: "left:model".into(),
            right_model_id: "right:model".into(),
            left_lineage: "lineage:left:1".into(),
            right_lineage: "lineage:right:2".into(),
            outcome_id: "outcome:velocity".into(),
            horizon_seconds: 3600.5,
            test_specification_id: "test:1".into(),
            measurement_specification_id: "measurement:1".into(),
        }
    }

    #[test]
    fn adapter_preserves_generic_identity_fields_exactly() {
        let temporal = identity();
        let source = temporal.to_candidate_source().unwrap();

        assert_eq!(source.candidate_id, temporal.candidate_id);
        assert_eq!(source.source_candidate_id, temporal.source_candidate_id);
        assert_eq!(source.test_specification_id, temporal.test_specification_id);
        assert_eq!(source.measurement_specification_id, temporal.measurement_specification_id);
        assert_eq!(source.left_lineage, temporal.left_lineage);
        assert_eq!(source.right_lineage, temporal.right_lineage);
    }

    #[test]
    fn adapter_never_rewrites_source_identity() {
        let mut temporal = identity();
        temporal.source_candidate_id = "a:b:c:d:e".into();

        let source = temporal.to_candidate_source().unwrap();

        assert_eq!(source.source_candidate_id, "a:b:c:d:e");
    }

    #[test]
    fn adapter_preserves_exact_horizon_bits() {
        let temporal = identity();
        assert_eq!(temporal.horizon_bits(), temporal.horizon_seconds.to_bits());
        assert_eq!(temporal.horizon_seconds, 3600.5);
    }

    #[test]
    fn malformed_identity_is_rejected_before_conversion() {
        let mut temporal = identity();
        temporal.left_lineage.clear();

        assert_eq!(
            temporal.to_candidate_source(),
            Err(TemporalAdapterError::MissingField("left_lineage"))
        );
    }

    #[test]
    fn nonfinite_horizon_is_rejected() {
        let mut temporal = identity();
        temporal.horizon_seconds = f64::NAN;

        assert_eq!(
            temporal.to_candidate_source(),
            Err(TemporalAdapterError::NonFiniteHorizon)
        );
    }

    #[test]
    fn conversion_is_deterministic_and_has_no_evidence_effect() {
        let temporal = identity();
        assert_eq!(
            temporal.to_candidate_source().unwrap(),
            temporal.to_candidate_source().unwrap()
        );
        // The adapter returns only an identity source; it cannot construct
        // observations, replications, criterion evidence, or commitments.
    }
}
