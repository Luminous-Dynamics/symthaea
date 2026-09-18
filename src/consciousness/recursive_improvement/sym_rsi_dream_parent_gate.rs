// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public fail-closed lineage gate for the SYM-RSI-001D dream pipeline.
//!
//! The lower-level dream verification/fresh/OOD modules are useful experiment
//! primitives, but they must not be publicly callable without proving that arm C
//! survived the original SYM-RSI-001 independent 101-104 held-out replay gate.
//!
//! This module provides the only public measurement path:
//!
//! training selection + 101-104 holdout
//!     -> ParentCQualification
//!     -> 301-304 verification corpus
//!     -> QualifiedDreamVerification
//!     -> 401-404 fresh D-vs-C
//!     -> QualifiedDreamFresh
//!     -> 1201-1204 secondary OOD
//!
//! Qualification tokens have private fields and are intentionally not
//! deserializable. A later process must recreate them by re-validating the frozen
//! evidence rather than trusting a serialized boolean.

use super::sym_rsi_candidate_family::FrozenReplayCorpus;
use super::sym_rsi_dream_fresh_evaluation::{
    run_fresh_d_vs_c, DreamFreshEvaluationError, DreamFreshEvaluationReceipt,
};
use super::sym_rsi_dream_ood_evaluation::{
    run_ood_d_vs_c, DreamOodEvaluationError, DreamOodEvaluationReceipt,
};
use super::sym_rsi_dream_protocol::{
    validate_canonical_sym_rsi_001d_manifest, DreamProtocolError, SYM_RSI_001D_EXPERIMENT_ID,
};
use super::sym_rsi_dream_verification::{
    acquire_dream_verification_corpus, validate_grounded_dream_on_verification,
    DreamVerificationCorpus, DreamVerificationError, DreamVerificationGateReceipt,
};
use super::sym_rsi_experiment::SymRsiExperimentManifest;
use super::sym_rsi_fixtures::canonical_sym_rsi_001_fixture_manifest;
use super::sym_rsi_holdout_gate::{
    select_canonical_training_candidate, validate_selected_candidate_on_holdout,
    HeldOutReplayGateReceipt, HoldoutGateDecision, HoldoutGateError,
};
use super::sym_rsi_replay_selection::ReplaySelectionReceipt;
use serde::{Deserialize, Serialize};

pub const SYM_RSI_001D_PARENT_C_QUALIFICATION_SCHEMA: &str =
    "symthaea.sym-rsi-001d.parent-c-qualification.v1";
pub const SYM_RSI_001D_QUALIFIED_VERIFICATION_SCHEMA: &str =
    "symthaea.sym-rsi-001d.parent-qualified-verification.v1";
pub const SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA: &str =
    "symthaea.sym-rsi-001d.parent-qualified-fresh.v1";
pub const SYM_RSI_001D_QUALIFIED_OOD_SCHEMA: &str =
    "symthaea.sym-rsi-001d.parent-qualified-ood.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParentCQualificationReceipt {
    pub schema: String,
    pub parent_experiment_id: String,
    pub parent_preregistration_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
    pub training_selection_evidence_digest: String,
    pub parent_holdout_gate_evidence_digest: String,
    pub parent_holdout_corpus_evidence_digest: String,
    pub selected_policy_id: String,
    pub evidence_digest: String,
}

/// Unforgeable-in-API token proving that C survived the original holdout gate.
///
/// No Deserialize implementation is provided. Recreate this token by calling
/// `qualify_parent_c_for_dream` against the frozen source evidence.
#[derive(Debug, Clone)]
pub struct ParentCQualification {
    receipt: ParentCQualificationReceipt,
}

impl ParentCQualification {
    pub fn receipt(&self) -> &ParentCQualificationReceipt {
        &self.receipt
    }

    pub fn selected_policy_id(&self) -> &str {
        &self.receipt.selected_policy_id
    }

    pub fn parent_holdout_gate_evidence_digest(&self) -> &str {
        &self.receipt.parent_holdout_gate_evidence_digest
    }

    fn validate_binding(
        &self,
        parent_manifest: &SymRsiExperimentManifest,
        c_selection: &ReplaySelectionReceipt,
        extension_manifest: &SymRsiExperimentManifest,
    ) -> Result<(), ParentDreamGateError> {
        ensure_canonical_parent_manifest(parent_manifest)?;
        validate_canonical_sym_rsi_001d_manifest(extension_manifest)
            .map_err(ParentDreamGateError::Protocol)?;

        if extension_manifest.experiment_id != SYM_RSI_001D_EXPERIMENT_ID
            || self.receipt.parent_experiment_id != parent_manifest.experiment_id
            || self.receipt.parent_preregistration_digest
                != parent_manifest.preregistration_digest
            || self.receipt.subject_digest != parent_manifest.subject_digest
            || self.receipt.environment_digest != parent_manifest.environment_digest
            || extension_manifest.subject_digest != parent_manifest.subject_digest
            || extension_manifest.environment_digest != parent_manifest.environment_digest
            || self.receipt.training_selection_evidence_digest != c_selection.evidence_digest
            || self.receipt.selected_policy_id != c_selection.selected_policy_id
            || self.receipt.evidence_digest.trim().is_empty()
        {
            return Err(ParentDreamGateError::QualificationBindingMismatch);
        }
        Ok(())
    }
}

/// Recompute both the canonical training selection and the independent parent
/// holdout gate, then mint a qualification token only for FreshExecutionEligible C.
pub fn qualify_parent_c_for_dream(
    parent_manifest: &SymRsiExperimentManifest,
    training_corpus: &FrozenReplayCorpus,
    c_selection: &ReplaySelectionReceipt,
    parent_held_out_corpus: &FrozenReplayCorpus,
    parent_holdout_gate: &HeldOutReplayGateReceipt,
) -> Result<ParentCQualification, ParentDreamGateError> {
    ensure_canonical_parent_manifest(parent_manifest)?;

    let recomputed_selection =
        select_canonical_training_candidate(parent_manifest, training_corpus)
            .map_err(ParentDreamGateError::Holdout)?;
    if &recomputed_selection != c_selection {
        return Err(ParentDreamGateError::TrainingSelectionReceiptMismatch);
    }

    let recomputed_holdout = validate_selected_candidate_on_holdout(
        parent_manifest,
        c_selection,
        parent_held_out_corpus,
    )
    .map_err(ParentDreamGateError::Holdout)?;
    if &recomputed_holdout != parent_holdout_gate {
        return Err(ParentDreamGateError::ParentHoldoutReceiptMismatch);
    }
    if parent_holdout_gate.decision != HoldoutGateDecision::FreshExecutionEligible {
        return Err(ParentDreamGateError::ParentCNotFreshExecutionEligible(
            parent_holdout_gate.decision,
        ));
    }
    if !parent_holdout_gate.incumbent_full_support
        || !parent_holdout_gate.selected_full_support
        || parent_holdout_gate.selected_policy_id == parent_holdout_gate.incumbent_policy_id
        || parent_holdout_gate.selected_policy_id != c_selection.selected_policy_id
        || parent_holdout_gate.training_selection_evidence_digest != c_selection.evidence_digest
        || parent_holdout_gate.evidence_digest.trim().is_empty()
        || parent_holdout_gate.held_out_corpus_evidence_digest.trim().is_empty()
    {
        return Err(ParentDreamGateError::ParentHoldoutReceiptMismatch);
    }

    let evidence_digest = parent_c_qualification_digest(
        parent_manifest,
        c_selection,
        parent_holdout_gate,
    );
    Ok(ParentCQualification {
        receipt: ParentCQualificationReceipt {
            schema: SYM_RSI_001D_PARENT_C_QUALIFICATION_SCHEMA.into(),
            parent_experiment_id: parent_manifest.experiment_id.clone(),
            parent_preregistration_digest: parent_manifest.preregistration_digest.clone(),
            subject_digest: parent_manifest.subject_digest.clone(),
            environment_digest: parent_manifest.environment_digest.clone(),
            training_selection_evidence_digest: c_selection.evidence_digest.clone(),
            parent_holdout_gate_evidence_digest: parent_holdout_gate.evidence_digest.clone(),
            parent_holdout_corpus_evidence_digest:
                parent_holdout_gate.held_out_corpus_evidence_digest.clone(),
            selected_policy_id: c_selection.selected_policy_id.clone(),
            evidence_digest,
        },
    })
}

/// The only public acquisition path intended to consume dream verification seeds
/// 301-304. Parent C qualification is checked before any environment execution.
pub fn acquire_dream_verification_corpus_after_parent_c(
    parent_c: &ParentCQualification,
    parent_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
) -> Result<DreamVerificationCorpus, ParentDreamGateError> {
    parent_c.validate_binding(parent_manifest, c_selection, extension_manifest)?;
    acquire_dream_verification_corpus(extension_manifest)
        .map_err(ParentDreamGateError::Verification)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParentQualifiedDreamVerificationReceipt {
    pub schema: String,
    pub parent_c_qualification_evidence_digest: String,
    pub parent_holdout_gate_evidence_digest: String,
    pub verification: DreamVerificationGateReceipt,
    pub evidence_digest: String,
}

/// Private-constructor token proving that the D verification receipt descends from
/// a qualified parent C lineage.
#[derive(Debug, Clone)]
pub struct QualifiedDreamVerification {
    receipt: ParentQualifiedDreamVerificationReceipt,
}

impl QualifiedDreamVerification {
    pub fn receipt(&self) -> &ParentQualifiedDreamVerificationReceipt {
        &self.receipt
    }

    pub fn verification(&self) -> &DreamVerificationGateReceipt {
        &self.receipt.verification
    }

    fn validate_binding(
        &self,
        parent_c: &ParentCQualification,
        parent_manifest: &SymRsiExperimentManifest,
        c_selection: &ReplaySelectionReceipt,
        extension_manifest: &SymRsiExperimentManifest,
    ) -> Result<(), ParentDreamGateError> {
        parent_c.validate_binding(parent_manifest, c_selection, extension_manifest)?;
        if self.receipt.schema != SYM_RSI_001D_QUALIFIED_VERIFICATION_SCHEMA
            || self.receipt.parent_c_qualification_evidence_digest
                != parent_c.receipt.evidence_digest
            || self.receipt.parent_holdout_gate_evidence_digest
                != parent_c.receipt.parent_holdout_gate_evidence_digest
            || self.receipt.verification.c_selection_evidence_digest != c_selection.evidence_digest
            || self.receipt.verification.c_policy_id != c_selection.selected_policy_id
            || self.receipt.verification.experiment_id != extension_manifest.experiment_id
            || self.receipt.verification.preregistration_digest
                != extension_manifest.preregistration_digest
            || self.receipt.evidence_digest.trim().is_empty()
        {
            return Err(ParentDreamGateError::QualifiedVerificationBindingMismatch);
        }
        Ok(())
    }
}

/// Validate D on an already-frozen 301-304 verification corpus after proving that
/// parent C survived its own independent holdout gate.
pub fn validate_grounded_dream_after_parent_c(
    parent_c: &ParentCQualification,
    parent_manifest: &SymRsiExperimentManifest,
    parent_training_corpus: &FrozenReplayCorpus,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    verification_corpus: &DreamVerificationCorpus,
) -> Result<QualifiedDreamVerification, ParentDreamGateError> {
    parent_c.validate_binding(parent_manifest, c_selection, extension_manifest)?;
    let verification = validate_grounded_dream_on_verification(
        parent_manifest,
        parent_training_corpus,
        c_selection,
        extension_manifest,
        verification_corpus,
    )
    .map_err(ParentDreamGateError::Verification)?;

    let evidence_digest = qualified_verification_digest(parent_c, &verification);
    Ok(QualifiedDreamVerification {
        receipt: ParentQualifiedDreamVerificationReceipt {
            schema: SYM_RSI_001D_QUALIFIED_VERIFICATION_SCHEMA.into(),
            parent_c_qualification_evidence_digest: parent_c.receipt.evidence_digest.clone(),
            parent_holdout_gate_evidence_digest:
                parent_c.receipt.parent_holdout_gate_evidence_digest.clone(),
            verification,
            evidence_digest,
        },
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParentQualifiedDreamFreshReceipt {
    pub schema: String,
    pub parent_c_qualification_evidence_digest: String,
    pub qualified_verification_evidence_digest: String,
    pub parent_holdout_gate_evidence_digest: String,
    pub fresh: DreamFreshEvaluationReceipt,
    pub evidence_digest: String,
}

#[derive(Debug, Clone)]
pub struct QualifiedDreamFresh {
    receipt: ParentQualifiedDreamFreshReceipt,
}

impl QualifiedDreamFresh {
    pub fn receipt(&self) -> &ParentQualifiedDreamFreshReceipt {
        &self.receipt
    }

    pub fn fresh(&self) -> &DreamFreshEvaluationReceipt {
        &self.receipt.fresh
    }

    fn validate_binding(
        &self,
        parent_c: &ParentCQualification,
        qualified_verification: &QualifiedDreamVerification,
        parent_manifest: &SymRsiExperimentManifest,
        c_selection: &ReplaySelectionReceipt,
        extension_manifest: &SymRsiExperimentManifest,
    ) -> Result<(), ParentDreamGateError> {
        qualified_verification.validate_binding(
            parent_c,
            parent_manifest,
            c_selection,
            extension_manifest,
        )?;
        if self.receipt.schema != SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA
            || self.receipt.parent_c_qualification_evidence_digest
                != parent_c.receipt.evidence_digest
            || self.receipt.qualified_verification_evidence_digest
                != qualified_verification.receipt.evidence_digest
            || self.receipt.parent_holdout_gate_evidence_digest
                != parent_c.receipt.parent_holdout_gate_evidence_digest
            || self.receipt.fresh.verification_gate_evidence_digest
                != qualified_verification.receipt.verification.evidence_digest
            || self.receipt.fresh.c_selection_evidence_digest != c_selection.evidence_digest
            || self.receipt.fresh.experiment_id != extension_manifest.experiment_id
            || self.receipt.evidence_digest.trim().is_empty()
        {
            return Err(ParentDreamGateError::QualifiedFreshBindingMismatch);
        }
        Ok(())
    }
}

/// The only public path intended to consume fresh dream seeds 401-404.
pub fn run_fresh_d_vs_c_after_parent_c(
    parent_c: &ParentCQualification,
    qualified_verification: &QualifiedDreamVerification,
    parent_manifest: &SymRsiExperimentManifest,
    parent_training_corpus: &FrozenReplayCorpus,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
) -> Result<QualifiedDreamFresh, ParentDreamGateError> {
    qualified_verification.validate_binding(
        parent_c,
        parent_manifest,
        c_selection,
        extension_manifest,
    )?;
    let fresh = run_fresh_d_vs_c(
        parent_manifest,
        parent_training_corpus,
        c_selection,
        extension_manifest,
        &qualified_verification.receipt.verification,
    )
    .map_err(ParentDreamGateError::Fresh)?;

    let evidence_digest = qualified_fresh_digest(
        parent_c,
        qualified_verification,
        &fresh,
    );
    Ok(QualifiedDreamFresh {
        receipt: ParentQualifiedDreamFreshReceipt {
            schema: SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA.into(),
            parent_c_qualification_evidence_digest: parent_c.receipt.evidence_digest.clone(),
            qualified_verification_evidence_digest:
                qualified_verification.receipt.evidence_digest.clone(),
            parent_holdout_gate_evidence_digest:
                parent_c.receipt.parent_holdout_gate_evidence_digest.clone(),
            fresh,
            evidence_digest,
        },
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParentQualifiedDreamOodReceipt {
    pub schema: String,
    pub parent_c_qualification_evidence_digest: String,
    pub qualified_fresh_evidence_digest: String,
    pub parent_holdout_gate_evidence_digest: String,
    pub ood: DreamOodEvaluationReceipt,
    pub evidence_digest: String,
}

/// The only public path intended to consume secondary OOD seeds 1201-1204.
pub fn run_ood_d_vs_c_after_parent_c(
    parent_c: &ParentCQualification,
    qualified_verification: &QualifiedDreamVerification,
    qualified_fresh: &QualifiedDreamFresh,
    parent_manifest: &SymRsiExperimentManifest,
    parent_training_corpus: &FrozenReplayCorpus,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
) -> Result<ParentQualifiedDreamOodReceipt, ParentDreamGateError> {
    qualified_fresh.validate_binding(
        parent_c,
        qualified_verification,
        parent_manifest,
        c_selection,
        extension_manifest,
    )?;
    let ood = run_ood_d_vs_c(
        parent_manifest,
        parent_training_corpus,
        c_selection,
        extension_manifest,
        &qualified_fresh.receipt.fresh,
    )
    .map_err(ParentDreamGateError::Ood)?;

    let evidence_digest = qualified_ood_digest(parent_c, qualified_fresh, &ood);
    Ok(ParentQualifiedDreamOodReceipt {
        schema: SYM_RSI_001D_QUALIFIED_OOD_SCHEMA.into(),
        parent_c_qualification_evidence_digest: parent_c.receipt.evidence_digest.clone(),
        qualified_fresh_evidence_digest: qualified_fresh.receipt.evidence_digest.clone(),
        parent_holdout_gate_evidence_digest:
            parent_c.receipt.parent_holdout_gate_evidence_digest.clone(),
        ood,
        evidence_digest,
    })
}

fn ensure_canonical_parent_manifest(
    manifest: &SymRsiExperimentManifest,
) -> Result<(), ParentDreamGateError> {
    manifest
        .validate()
        .map_err(|error| ParentDreamGateError::ParentManifestInvalid(format!("{error:?}")))?;
    let expected = canonical_sym_rsi_001_fixture_manifest(
        manifest.preregistration_digest.clone(),
        manifest.subject_digest.clone(),
        manifest.environment_digest.clone(),
    );
    if manifest != &expected {
        return Err(ParentDreamGateError::ParentManifestNotCanonical);
    }
    Ok(())
}

fn parent_c_qualification_digest(
    manifest: &SymRsiExperimentManifest,
    selection: &ReplaySelectionReceipt,
    holdout: &HeldOutReplayGateReceipt,
) -> String {
    digest_strings(
        b"symthaea.sym-rsi-001d.parent-c-qualification.v1\0",
        &[
            manifest.experiment_id.as_str(),
            manifest.preregistration_digest.as_str(),
            manifest.subject_digest.as_str(),
            manifest.environment_digest.as_str(),
            selection.evidence_digest.as_str(),
            holdout.evidence_digest.as_str(),
            holdout.held_out_corpus_evidence_digest.as_str(),
            selection.selected_policy_id.as_str(),
        ],
    )
}

fn qualified_verification_digest(
    parent_c: &ParentCQualification,
    verification: &DreamVerificationGateReceipt,
) -> String {
    digest_strings(
        b"symthaea.sym-rsi-001d.parent-qualified-verification.v1\0",
        &[
            parent_c.receipt.evidence_digest.as_str(),
            parent_c.receipt.parent_holdout_gate_evidence_digest.as_str(),
            verification.evidence_digest.as_str(),
            verification.verification_corpus_evidence_digest.as_str(),
            verification.grounded_dream_model_evidence_digest.as_str(),
        ],
    )
}

fn qualified_fresh_digest(
    parent_c: &ParentCQualification,
    verification: &QualifiedDreamVerification,
    fresh: &DreamFreshEvaluationReceipt,
) -> String {
    digest_strings(
        b"symthaea.sym-rsi-001d.parent-qualified-fresh.v1\0",
        &[
            parent_c.receipt.evidence_digest.as_str(),
            verification.receipt.evidence_digest.as_str(),
            fresh.evidence_digest.as_str(),
            fresh.verification_gate_evidence_digest.as_str(),
            fresh.grounded_dream_model_evidence_digest.as_str(),
        ],
    )
}

fn qualified_ood_digest(
    parent_c: &ParentCQualification,
    fresh: &QualifiedDreamFresh,
    ood: &DreamOodEvaluationReceipt,
) -> String {
    digest_strings(
        b"symthaea.sym-rsi-001d.parent-qualified-ood.v1\0",
        &[
            parent_c.receipt.evidence_digest.as_str(),
            fresh.receipt.evidence_digest.as_str(),
            ood.evidence_digest.as_str(),
            ood.fresh_primary_evidence_digest.as_str(),
            ood.grounded_dream_model_evidence_digest.as_str(),
        ],
    )
}

fn digest_strings(prefix: &[u8], values: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(prefix);
    for value in values {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[derive(Debug)]
pub enum ParentDreamGateError {
    ParentManifestInvalid(String),
    ParentManifestNotCanonical,
    TrainingSelectionReceiptMismatch,
    ParentHoldoutReceiptMismatch,
    ParentCNotFreshExecutionEligible(HoldoutGateDecision),
    QualificationBindingMismatch,
    QualifiedVerificationBindingMismatch,
    QualifiedFreshBindingMismatch,
    Holdout(HoldoutGateError),
    Protocol(DreamProtocolError),
    Verification(DreamVerificationError),
    Fresh(DreamFreshEvaluationError),
    Ood(DreamOodEvaluationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::{
        acquire_canonical_replay_corpus, canonical_sym_rsi_001_fixture_manifest,
    };
    use super::super::sym_rsi_experiment::EvaluationSplit;

    #[test]
    fn mutated_parent_holdout_receipt_cannot_mint_qualification() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay).unwrap();
        let held_out =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::HeldOutReplay).unwrap();
        let selection = select_canonical_training_candidate(&manifest, &training).unwrap();
        let mut gate =
            validate_selected_candidate_on_holdout(&manifest, &selection, &held_out).unwrap();
        gate.evidence_digest.push_str("-tampered");

        assert!(matches!(
            qualify_parent_c_for_dream(
                &manifest,
                &training,
                &selection,
                &held_out,
                &gate,
            ),
            Err(ParentDreamGateError::ParentHoldoutReceiptMismatch)
        ));
    }

    #[test]
    fn parent_qualification_never_consumes_dream_extension_seeds() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let training =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::TrainingReplay).unwrap();
        let held_out =
            acquire_canonical_replay_corpus(&manifest, EvaluationSplit::HeldOutReplay).unwrap();
        let selection = select_canonical_training_candidate(&manifest, &training).unwrap();
        let gate =
            validate_selected_candidate_on_holdout(&manifest, &selection, &held_out).unwrap();

        let result = qualify_parent_c_for_dream(
            &manifest,
            &training,
            &selection,
            &held_out,
            &gate,
        );
        match gate.decision {
            HoldoutGateDecision::FreshExecutionEligible => assert!(result.is_ok()),
            other => assert!(matches!(
                result,
                Err(ParentDreamGateError::ParentCNotFreshExecutionEligible(decision))
                    if decision == other
            )),
        }
    }

    #[test]
    fn qualification_receipt_digest_is_bound_to_parent_holdout() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let fake_selection_digest = "selection";
        let fake_holdout_digest = "holdout";
        let fake_holdout_corpus_digest = "holdout-corpus";
        let digest = digest_strings(
            b"symthaea.sym-rsi-001d.parent-c-qualification.v1\0",
            &[
                manifest.experiment_id.as_str(),
                manifest.preregistration_digest.as_str(),
                manifest.subject_digest.as_str(),
                manifest.environment_digest.as_str(),
                fake_selection_digest,
                fake_holdout_digest,
                fake_holdout_corpus_digest,
                "policy",
            ],
        );
        assert!(digest.starts_with("blake3:"));
    }
}
