// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed trust anchors for restart-safe SYM-RSI fresh qualification recovery.
//!
//! These anchors are integrity objects, not sources of trust by themselves. An
//! anchor becomes authoritative only when its bytes/digest are obtained from an
//! independently immutable source such as a frozen Git evidence commit or signed
//! evidence ledger. Requalification must never source both the measurement receipt
//! and its expected anchor from the same mutable artifact.

use super::sym_rsi_c_fresh_gate::{
    requalify_fresh_c_vs_a_from_receipt, ParentQualifiedReplayFreshReceipt,
    QualifiedReplayFresh, QualifiedReplayFreshError, SYM_RSI_001_QUALIFIED_FRESH_SCHEMA,
};
use super::sym_rsi_dream_parent_gate::{
    requalify_fresh_d_vs_c_from_receipt, ParentCQualification,
    ParentDreamGateError, ParentQualifiedDreamFreshReceipt, QualifiedDreamFresh,
    QualifiedDreamVerification, SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA,
};
use super::sym_rsi_experiment::SymRsiExperimentManifest;
use super::sym_rsi_holdout_gate::HeldOutReplayGateReceipt;
use super::sym_rsi_replay_selection::ReplaySelectionReceipt;
use serde::{Deserialize, Serialize};

pub const SYM_RSI_QUALIFICATION_ANCHOR_SCHEMA: &str =
    "symthaea.sym-rsi.qualified-evidence-anchor.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationAnchorStage {
    ReplayFreshCvsA,
    DreamFreshDvsC,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationAnchor {
    pub schema: String,
    pub stage: QualificationAnchorStage,
    pub wrapper_schema: String,
    pub experiment_id: String,
    pub preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub parent_c_qualification_evidence_digest: String,
    pub parent_holdout_gate_evidence_digest: String,
    /// C-vs-A: canonical training-selection evidence digest.
    /// D-vs-C: qualified 301-304 verification wrapper evidence digest.
    pub upstream_qualification_evidence_digest: String,
    /// The underlying scientific fresh receipt evidence digest (v1).
    pub source_measurement_evidence_digest: String,
    /// The complete semantic qualified-wrapper evidence digest (v2).
    pub qualified_evidence_digest: String,
    /// Self-integrity digest for this anchor object. Authority still comes from the
    /// external immutable source from which this anchor was loaded.
    pub evidence_digest: String,
}

impl QualificationAnchor {
    pub fn from_replay_fresh(qualified: &QualifiedReplayFresh) -> Self {
        let receipt = qualified.receipt();
        let fresh = qualified.fresh();
        let mut anchor = Self {
            schema: SYM_RSI_QUALIFICATION_ANCHOR_SCHEMA.into(),
            stage: QualificationAnchorStage::ReplayFreshCvsA,
            wrapper_schema: receipt.schema.clone(),
            experiment_id: fresh.experiment_id.clone(),
            preregistration_digest: fresh.preregistration_digest.clone(),
            subject_digest: fresh.subject_digest.clone(),
            environment_digest: fresh.environment_digest.clone(),
            parent_c_qualification_evidence_digest:
                receipt.parent_c_qualification_evidence_digest.clone(),
            parent_holdout_gate_evidence_digest:
                receipt.parent_holdout_gate_evidence_digest.clone(),
            upstream_qualification_evidence_digest:
                receipt.training_selection_evidence_digest.clone(),
            source_measurement_evidence_digest: fresh.evidence_digest.clone(),
            qualified_evidence_digest: receipt.evidence_digest.clone(),
            evidence_digest: String::new(),
        };
        anchor.evidence_digest = qualification_anchor_digest(&anchor);
        anchor
    }

    pub fn from_dream_fresh(qualified: &QualifiedDreamFresh) -> Self {
        let receipt = qualified.receipt();
        let fresh = qualified.fresh();
        let mut anchor = Self {
            schema: SYM_RSI_QUALIFICATION_ANCHOR_SCHEMA.into(),
            stage: QualificationAnchorStage::DreamFreshDvsC,
            wrapper_schema: receipt.schema.clone(),
            experiment_id: fresh.experiment_id.clone(),
            preregistration_digest: fresh.preregistration_digest.clone(),
            subject_digest: fresh.subject_digest.clone(),
            environment_digest: fresh.environment_digest.clone(),
            parent_c_qualification_evidence_digest:
                receipt.parent_c_qualification_evidence_digest.clone(),
            parent_holdout_gate_evidence_digest:
                receipt.parent_holdout_gate_evidence_digest.clone(),
            upstream_qualification_evidence_digest:
                receipt.qualified_verification_evidence_digest.clone(),
            source_measurement_evidence_digest: fresh.evidence_digest.clone(),
            qualified_evidence_digest: receipt.evidence_digest.clone(),
            evidence_digest: String::new(),
        };
        anchor.evidence_digest = qualification_anchor_digest(&anchor);
        anchor
    }

    pub fn validate(&self) -> Result<(), QualificationAnchorError> {
        if self.schema != SYM_RSI_QUALIFICATION_ANCHOR_SCHEMA
            || self.wrapper_schema.trim().is_empty()
            || self.experiment_id.trim().is_empty()
            || self.preregistration_digest.trim().is_empty()
            || self.subject_digest.trim().is_empty()
            || self.environment_digest.trim().is_empty()
            || self.parent_c_qualification_evidence_digest.trim().is_empty()
            || self.parent_holdout_gate_evidence_digest.trim().is_empty()
            || self.upstream_qualification_evidence_digest.trim().is_empty()
            || self.source_measurement_evidence_digest.trim().is_empty()
            || self.qualified_evidence_digest.trim().is_empty()
            || self.evidence_digest.trim().is_empty()
        {
            return Err(QualificationAnchorError::MalformedAnchor);
        }

        let expected_wrapper = match self.stage {
            QualificationAnchorStage::ReplayFreshCvsA => {
                SYM_RSI_001_QUALIFIED_FRESH_SCHEMA
            }
            QualificationAnchorStage::DreamFreshDvsC => {
                SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA
            }
        };
        if self.wrapper_schema != expected_wrapper {
            return Err(QualificationAnchorError::WrongWrapperSchema);
        }
        if self.evidence_digest != qualification_anchor_digest(self) {
            return Err(QualificationAnchorError::AnchorDigestMismatch);
        }
        Ok(())
    }
}

/// Preferred restart path for the C-vs-A fresh token.
///
/// The caller is responsible for loading `anchor` from an independently immutable
/// evidence source. The serialized qualified receipt may come from ordinary durable
/// storage because every semantic field is checked against the trusted anchor and
/// frozen parent lineage before the private token is reconstructed.
pub fn requalify_replay_fresh_from_anchor(
    anchor: &QualificationAnchor,
    parent_c: &ParentCQualification,
    manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    parent_holdout_gate: &HeldOutReplayGateReceipt,
    serialized: ParentQualifiedReplayFreshReceipt,
) -> Result<QualifiedReplayFresh, QualificationAnchorError> {
    anchor.validate()?;
    if anchor.stage != QualificationAnchorStage::ReplayFreshCvsA
        || anchor.wrapper_schema != SYM_RSI_001_QUALIFIED_FRESH_SCHEMA
        || anchor.experiment_id != manifest.experiment_id
        || anchor.preregistration_digest != manifest.preregistration_digest
        || anchor.subject_digest != manifest.subject_digest
        || anchor.environment_digest != manifest.environment_digest
        || anchor.parent_c_qualification_evidence_digest
            != parent_c.receipt().evidence_digest
        || anchor.parent_holdout_gate_evidence_digest
            != parent_c.receipt().parent_holdout_gate_evidence_digest
        || anchor.upstream_qualification_evidence_digest != c_selection.evidence_digest
        || anchor.source_measurement_evidence_digest != serialized.fresh.evidence_digest
        || anchor.qualified_evidence_digest != serialized.evidence_digest
    {
        return Err(QualificationAnchorError::AnchorLineageMismatch);
    }

    requalify_fresh_c_vs_a_from_receipt(
        parent_c,
        manifest,
        c_selection,
        parent_holdout_gate,
        &anchor.qualified_evidence_digest,
        serialized,
    )
    .map_err(QualificationAnchorError::ReplayFresh)
}

/// Preferred restart path for the D-vs-C fresh token.
///
/// Verification itself should be recreated from the frozen 301-304 replay corpus;
/// this anchor is only for the one-shot 401-404 fresh measurement wrapper.
pub fn requalify_dream_fresh_from_anchor(
    anchor: &QualificationAnchor,
    parent_c: &ParentCQualification,
    qualified_verification: &QualifiedDreamVerification,
    parent_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    serialized: ParentQualifiedDreamFreshReceipt,
) -> Result<QualifiedDreamFresh, QualificationAnchorError> {
    anchor.validate()?;
    if anchor.stage != QualificationAnchorStage::DreamFreshDvsC
        || anchor.wrapper_schema != SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA
        || anchor.experiment_id != extension_manifest.experiment_id
        || anchor.preregistration_digest != extension_manifest.preregistration_digest
        || anchor.subject_digest != extension_manifest.subject_digest
        || anchor.environment_digest != extension_manifest.environment_digest
        || anchor.subject_digest != parent_manifest.subject_digest
        || anchor.environment_digest != parent_manifest.environment_digest
        || anchor.parent_c_qualification_evidence_digest
            != parent_c.receipt().evidence_digest
        || anchor.parent_holdout_gate_evidence_digest
            != parent_c.receipt().parent_holdout_gate_evidence_digest
        || anchor.upstream_qualification_evidence_digest
            != qualified_verification.receipt().evidence_digest
        || anchor.source_measurement_evidence_digest != serialized.fresh.evidence_digest
        || anchor.qualified_evidence_digest != serialized.evidence_digest
        || serialized.fresh.c_selection_evidence_digest != c_selection.evidence_digest
    {
        return Err(QualificationAnchorError::AnchorLineageMismatch);
    }

    requalify_fresh_d_vs_c_from_receipt(
        parent_c,
        qualified_verification,
        parent_manifest,
        c_selection,
        extension_manifest,
        &anchor.qualified_evidence_digest,
        serialized,
    )
    .map_err(QualificationAnchorError::DreamFresh)
}

fn qualification_anchor_digest(anchor: &QualificationAnchor) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi.qualified-evidence-anchor.v1\0");
    hasher.update(&[stage_tag(anchor.stage)]);
    for value in [
        anchor.schema.as_str(),
        anchor.wrapper_schema.as_str(),
        anchor.experiment_id.as_str(),
        anchor.preregistration_digest.as_str(),
        anchor.subject_digest.as_str(),
        anchor.environment_digest.as_str(),
        anchor.parent_c_qualification_evidence_digest.as_str(),
        anchor.parent_holdout_gate_evidence_digest.as_str(),
        anchor.upstream_qualification_evidence_digest.as_str(),
        anchor.source_measurement_evidence_digest.as_str(),
        anchor.qualified_evidence_digest.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn stage_tag(stage: QualificationAnchorStage) -> u8 {
    match stage {
        QualificationAnchorStage::ReplayFreshCvsA => 0,
        QualificationAnchorStage::DreamFreshDvsC => 1,
    }
}

#[derive(Debug)]
pub enum QualificationAnchorError {
    MalformedAnchor,
    WrongWrapperSchema,
    AnchorDigestMismatch,
    AnchorLineageMismatch,
    ReplayFresh(QualifiedReplayFreshError),
    DreamFresh(ParentDreamGateError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(stage: QualificationAnchorStage, wrapper_schema: &str) -> QualificationAnchor {
        let mut anchor = QualificationAnchor {
            schema: SYM_RSI_QUALIFICATION_ANCHOR_SCHEMA.into(),
            stage,
            wrapper_schema: wrapper_schema.into(),
            experiment_id: "experiment".into(),
            preregistration_digest: "prereg".into(),
            subject_digest: "subject".into(),
            environment_digest: "environment".into(),
            parent_c_qualification_evidence_digest: "parent-c".into(),
            parent_holdout_gate_evidence_digest: "holdout".into(),
            upstream_qualification_evidence_digest: "upstream".into(),
            source_measurement_evidence_digest: "measurement".into(),
            qualified_evidence_digest: "qualified".into(),
            evidence_digest: String::new(),
        };
        anchor.evidence_digest = qualification_anchor_digest(&anchor);
        anchor
    }

    #[test]
    fn replay_anchor_is_self_integrity_bound() {
        let anchor = anchor(
            QualificationAnchorStage::ReplayFreshCvsA,
            SYM_RSI_001_QUALIFIED_FRESH_SCHEMA,
        );
        assert!(anchor.validate().is_ok());
    }

    #[test]
    fn dream_anchor_is_self_integrity_bound() {
        let anchor = anchor(
            QualificationAnchorStage::DreamFreshDvsC,
            SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA,
        );
        assert!(anchor.validate().is_ok());
    }

    #[test]
    fn cross_stage_wrapper_is_rejected() {
        let anchor = anchor(
            QualificationAnchorStage::ReplayFreshCvsA,
            SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA,
        );
        assert!(matches!(
            anchor.validate(),
            Err(QualificationAnchorError::WrongWrapperSchema)
        ));
    }

    #[test]
    fn anchor_mutation_breaks_self_integrity() {
        let mut anchor = anchor(
            QualificationAnchorStage::DreamFreshDvsC,
            SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA,
        );
        anchor.qualified_evidence_digest.push_str("-tampered");
        assert!(matches!(
            anchor.validate(),
            Err(QualificationAnchorError::AnchorDigestMismatch)
        ));
    }
}
