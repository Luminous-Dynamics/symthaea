// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemically typed narrative ingestion.
//!
//! Narrative integration is useful for both lived observations and generated
//! counterfactuals, but those sources must not be conflated. This adapter records
//! the evidence class and source identity before delegating to the legacy
//! `NarrativeSelfModel::process_experience` API.
//!
//! A positive appraisal means only that the episode was useful/valuable to the
//! narrative model. It is not empirical validation and grants no confidence authority.

use super::epistemic_world::WorldEvidenceKind;
use crate::consciousness::narrative_self::NarrativeSelfModel;
use crate::hdc::binary_hv::BinaryHV;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NarrativeEvidenceReceipt {
    pub evidence_kind: WorldEvidenceKind,
    pub source_digest: String,
    pub description: String,
    pub appraisal_positive: bool,
    pub significance: f64,
    pub episodes_before: usize,
    pub episodes_after: usize,
}

impl NarrativeEvidenceReceipt {
    /// True only for evidence classes that are already empirical by definition.
    /// Generated/counterfactual narrative content never becomes empirical merely
    /// because it was integrated into the self-model or appraised positively.
    pub fn is_direct_empirical(&self) -> bool {
        self.evidence_kind.is_empirical()
    }

    pub fn recorded_episode(&self) -> bool {
        self.episodes_after > self.episodes_before
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrativeEvidenceError {
    EmptySourceDigest,
    NonFiniteEffort,
    NonFiniteSignificance,
}

/// Ingest narrative material while preserving its epistemic source class.
///
/// This does not change the legacy narrative model's internal storage shape; the
/// returned receipt is the explicit provenance boundary used by RSI/dream callers.
/// `appraisal_positive` controls narrative valence only and must not be interpreted
/// as evidence validation.
pub fn ingest_narrative_evidence(
    model: &mut NarrativeSelfModel,
    input: &BinaryHV,
    description: &str,
    evidence_kind: WorldEvidenceKind,
    source_digest: impl Into<String>,
    appraisal_positive: bool,
    effort: f64,
    significance: f64,
) -> Result<NarrativeEvidenceReceipt, NarrativeEvidenceError> {
    let source_digest = source_digest.into();
    if source_digest.trim().is_empty() {
        return Err(NarrativeEvidenceError::EmptySourceDigest);
    }
    if !effort.is_finite() {
        return Err(NarrativeEvidenceError::NonFiniteEffort);
    }
    if !significance.is_finite() {
        return Err(NarrativeEvidenceError::NonFiniteSignificance);
    }

    let effort = effort.clamp(0.0, 1.0);
    let significance = significance.clamp(0.0, 1.0);
    let episodes_before = model.autobio.life_story.len();

    model.process_experience(
        input,
        description,
        appraisal_positive,
        effort,
        significance,
    );

    Ok(NarrativeEvidenceReceipt {
        evidence_kind,
        source_digest,
        description: description.to_string(),
        appraisal_positive,
        significance,
        episodes_before,
        episodes_after: model.autobio.life_story.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::narrative_self::NarrativeSelfConfig;

    fn model() -> NarrativeSelfModel {
        NarrativeSelfModel::new(NarrativeSelfConfig::default())
    }

    #[test]
    fn counterfactual_narrative_is_not_empirical_even_when_positive() {
        let mut model = model();
        let input = BinaryHV::random(77);
        let receipt = ingest_narrative_evidence(
            &mut model,
            &input,
            "dream found a promising alternative",
            WorldEvidenceKind::Counterfactual,
            "blake3:dream-content",
            true,
            0.2,
            0.9,
        )
        .unwrap();

        assert!(receipt.recorded_episode());
        assert!(!receipt.is_direct_empirical());
        assert!(receipt.appraisal_positive);
    }

    #[test]
    fn recorded_observation_remains_explicitly_empirical() {
        let mut model = model();
        let input = BinaryHV::random(78);
        let receipt = ingest_narrative_evidence(
            &mut model,
            &input,
            "observed outcome",
            WorldEvidenceKind::Recorded,
            "blake3:observation",
            true,
            0.4,
            0.8,
        )
        .unwrap();

        assert!(receipt.recorded_episode());
        assert!(receipt.is_direct_empirical());
    }

    #[test]
    fn model_prediction_does_not_gain_authority_from_narrative_storage() {
        let mut model = model();
        let input = BinaryHV::random(79);
        let receipt = ingest_narrative_evidence(
            &mut model,
            &input,
            "predicted future state",
            WorldEvidenceKind::ModelPredicted,
            "blake3:model-output",
            true,
            0.1,
            0.7,
        )
        .unwrap();

        assert!(receipt.recorded_episode());
        assert!(!receipt.is_direct_empirical());
    }

    #[test]
    fn missing_source_identity_fails_closed() {
        let mut model = model();
        let input = BinaryHV::random(80);
        assert_eq!(
            ingest_narrative_evidence(
                &mut model,
                &input,
                "dream",
                WorldEvidenceKind::Counterfactual,
                "   ",
                true,
                0.1,
                0.7,
            ),
            Err(NarrativeEvidenceError::EmptySourceDigest)
        );
    }
}
