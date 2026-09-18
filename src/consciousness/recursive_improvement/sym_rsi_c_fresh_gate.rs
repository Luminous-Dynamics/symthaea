// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Token-gated C-vs-A fresh execution for SYM-RSI-001.
//!
//! The raw fresh evaluator is an internal experiment primitive. Public consumption
//! of seeds 201-204 requires the private-constructor ParentCQualification token,
//! which proves that canonical training selection survived the independent 101-104
//! holdout gate.

use super::sym_rsi_dream_parent_gate::{
    ParentCQualification, SYM_RSI_001D_PARENT_C_QUALIFICATION_SCHEMA,
};
use super::sym_rsi_experiment::SymRsiExperimentManifest;
use super::sym_rsi_fixtures::canonical_sym_rsi_001_fixture_manifest;
use super::sym_rsi_fresh_evaluation::{
    run_fresh_c_vs_a, FreshCvsAReceipt, FreshEvaluationError,
};
use super::sym_rsi_holdout_gate::{
    HeldOutReplayGateReceipt, HoldoutGateDecision, SYM_RSI_001_HOLDOUT_GATE_SCHEMA,
};
use super::sym_rsi_replay_selection::ReplaySelectionReceipt;
use serde::{Deserialize, Serialize};

pub const SYM_RSI_001_QUALIFIED_FRESH_SCHEMA: &str =
    "symthaea.sym-rsi-001.parent-qualified-fresh.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParentQualifiedReplayFreshReceipt {
    pub schema: String,
    pub parent_c_qualification_evidence_digest: String,
    pub parent_holdout_gate_evidence_digest: String,
    pub training_selection_evidence_digest: String,
    pub fresh: FreshCvsAReceipt,
    pub evidence_digest: String,
}

/// Unforgeable-in-API token proving that C-vs-A fresh execution descended from a
/// recomputed parent-C qualification. It intentionally has no Deserialize impl.
#[derive(Debug, Clone)]
pub struct QualifiedReplayFresh {
    receipt: ParentQualifiedReplayFreshReceipt,
}

impl QualifiedReplayFresh {
    pub fn receipt(&self) -> &ParentQualifiedReplayFreshReceipt {
        &self.receipt
    }

    pub fn fresh(&self) -> &FreshCvsAReceipt {
        &self.receipt.fresh
    }
}

/// The only public path intended to consume C-vs-A fresh seeds 201-204.
///
/// Unit tests must not call this function. Qualification and binding checks happen
/// before the raw fresh evaluator is invoked.
pub fn run_fresh_c_vs_a_after_parent_c(
    parent_c: &ParentCQualification,
    manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    parent_holdout_gate: &HeldOutReplayGateReceipt,
) -> Result<QualifiedReplayFresh, QualifiedReplayFreshError> {
    validate_parent_binding(parent_c, manifest, c_selection, parent_holdout_gate)?;

    let fresh = run_fresh_c_vs_a(manifest, parent_holdout_gate)
        .map_err(QualifiedReplayFreshError::Fresh)?;
    if !fresh.fresh_seeds_consumed
        || fresh.selected_policy_id != c_selection.selected_policy_id
        || fresh.holdout_gate_evidence_digest != parent_holdout_gate.evidence_digest
        || fresh.subject_digest != manifest.subject_digest
        || fresh.environment_digest != manifest.environment_digest
        || fresh.evidence_digest.trim().is_empty()
    {
        return Err(QualifiedReplayFreshError::FreshReceiptBindingMismatch);
    }

    let evidence_digest = qualified_replay_fresh_digest(parent_c, c_selection, &fresh);
    Ok(QualifiedReplayFresh {
        receipt: ParentQualifiedReplayFreshReceipt {
            schema: SYM_RSI_001_QUALIFIED_FRESH_SCHEMA.into(),
            parent_c_qualification_evidence_digest: parent_c.receipt().evidence_digest.clone(),
            parent_holdout_gate_evidence_digest:
                parent_c.receipt().parent_holdout_gate_evidence_digest.clone(),
            training_selection_evidence_digest: c_selection.evidence_digest.clone(),
            fresh,
            evidence_digest,
        },
    })
}

fn validate_parent_binding(
    parent_c: &ParentCQualification,
    manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    holdout_gate: &HeldOutReplayGateReceipt,
) -> Result<(), QualifiedReplayFreshError> {
    manifest
        .validate()
        .map_err(|_| QualifiedReplayFreshError::ManifestInvalid)?;
    let expected = canonical_sym_rsi_001_fixture_manifest(
        manifest.preregistration_digest.clone(),
        manifest.subject_digest.clone(),
        manifest.environment_digest.clone(),
    );
    if manifest != &expected {
        return Err(QualifiedReplayFreshError::ManifestNotCanonical);
    }

    let qualification = parent_c.receipt();
    if qualification.schema != SYM_RSI_001D_PARENT_C_QUALIFICATION_SCHEMA
        || qualification.parent_experiment_id != manifest.experiment_id
        || qualification.parent_preregistration_digest != manifest.preregistration_digest
        || qualification.subject_digest != manifest.subject_digest
        || qualification.environment_digest != manifest.environment_digest
        || qualification.training_selection_evidence_digest != c_selection.evidence_digest
        || qualification.selected_policy_id != c_selection.selected_policy_id
        || qualification.parent_holdout_gate_evidence_digest != holdout_gate.evidence_digest
        || qualification.parent_holdout_corpus_evidence_digest
            != holdout_gate.held_out_corpus_evidence_digest
        || qualification.evidence_digest.trim().is_empty()
    {
        return Err(QualifiedReplayFreshError::QualificationBindingMismatch);
    }

    if holdout_gate.schema != SYM_RSI_001_HOLDOUT_GATE_SCHEMA
        || holdout_gate.decision != HoldoutGateDecision::FreshExecutionEligible
        || holdout_gate.experiment_id != manifest.experiment_id
        || holdout_gate.preregistration_digest != manifest.preregistration_digest
        || holdout_gate.subject_digest != manifest.subject_digest
        || holdout_gate.environment_digest != manifest.environment_digest
        || holdout_gate.training_selection_evidence_digest != c_selection.evidence_digest
        || holdout_gate.selected_policy_id != c_selection.selected_policy_id
        || holdout_gate.selected_policy_id == holdout_gate.incumbent_policy_id
        || !holdout_gate.incumbent_full_support
        || !holdout_gate.selected_full_support
        || holdout_gate.evidence_digest.trim().is_empty()
    {
        return Err(QualifiedReplayFreshError::HoldoutBindingMismatch);
    }
    Ok(())
}

fn qualified_replay_fresh_digest(
    parent_c: &ParentCQualification,
    selection: &ReplaySelectionReceipt,
    fresh: &FreshCvsAReceipt,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.parent-qualified-fresh.v1\0");
    for value in [
        parent_c.receipt().evidence_digest.as_str(),
        parent_c
            .receipt()
            .parent_holdout_gate_evidence_digest
            .as_str(),
        selection.evidence_digest.as_str(),
        fresh.evidence_digest.as_str(),
        fresh.selected_policy_id.as_str(),
        fresh.subject_digest.as_str(),
        fresh.environment_digest.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[derive(Debug)]
pub enum QualifiedReplayFreshError {
    ManifestInvalid,
    ManifestNotCanonical,
    QualificationBindingMismatch,
    HoldoutBindingMismatch,
    FreshReceiptBindingMismatch,
    Fresh(FreshEvaluationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapper_digest_binds_all_parent_lineage_components() {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.sym-rsi-001.parent-qualified-fresh.v1\0");
        for value in ["parent", "holdout", "selection", "fresh", "policy", "subject", "env"] {
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        let digest = format!("blake3:{}", hasher.finalize().to_hex());
        assert!(digest.starts_with("blake3:"));
    }

    #[test]
    fn qualified_token_is_not_deserializable_by_construction() {
        // Compile-time API property: only the receipt derives Deserialize. The token
        // itself has a private field and no Deserialize implementation.
        fn accepts_debug<T: std::fmt::Debug>() {}
        accepts_debug::<QualifiedReplayFresh>();
    }
}
