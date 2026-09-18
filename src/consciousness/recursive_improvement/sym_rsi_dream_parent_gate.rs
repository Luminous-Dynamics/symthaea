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
//! deserializable. A later process must recreate them by re-validating frozen
//! evidence rather than trusting a serialized boolean. Fresh D-vs-C tokens may be
//! recovered without rerunning 401-404 only when a complete semantic receipt
//! reproduces a previously frozen qualified-wrapper digest.

use super::sym_rsi_candidate_family::FrozenReplayCorpus;
use super::sym_rsi_dream_fresh_evaluation::{
    run_fresh_d_vs_c, DreamFreshDisposition, DreamFreshDomainSummary,
    DreamFreshEvaluationError, DreamFreshEvaluationReceipt, DreamFreshPairReceipt,
    SYM_RSI_001D_FRESH_EVALUATION_SCHEMA,
};
use super::sym_rsi_dream_ood_evaluation::{
    run_ood_d_vs_c, DreamOodEvaluationError, DreamOodEvaluationReceipt,
};
use super::sym_rsi_dream_protocol::{
    validate_canonical_sym_rsi_001d_manifest, DreamProtocolError, SYM_RSI_001D_ANALYSIS_RULE,
    SYM_RSI_001D_EXPERIMENT_ID, SYM_RSI_001D_QUALITY_TOLERANCE,
};
use super::sym_rsi_dream_verification::{
    acquire_dream_verification_corpus, validate_grounded_dream_on_verification,
    DreamVerificationCorpus, DreamVerificationDecision, DreamVerificationError,
    DreamVerificationGateReceipt,
};
use super::sym_rsi_experiment::{
    build_primary_contrast, EvaluationSplit, ExperimentArm, PrimaryContrastKind,
    SymRsiExperimentManifest, SymRsiRunReceipt,
};
use super::sym_rsi_fixtures::{
    canonical_sym_rsi_001_fixture_manifest, FixtureDomainKind,
};
use super::sym_rsi_grounded_dream::SYM_RSI_001_GROUNDED_DREAM_POLICY_ID;
use super::sym_rsi_holdout_gate::{
    select_canonical_training_candidate, validate_selected_candidate_on_holdout,
    HeldOutReplayGateReceipt, HoldoutGateDecision, HoldoutGateError,
};
use super::sym_rsi_replay_selection::ReplaySelectionReceipt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_001D_PARENT_C_QUALIFICATION_SCHEMA: &str =
    "symthaea.sym-rsi-001d.parent-c-qualification.v1";
pub const SYM_RSI_001D_QUALIFIED_VERIFICATION_SCHEMA: &str =
    "symthaea.sym-rsi-001d.parent-qualified-verification.v1";
pub const SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA: &str =
    "symthaea.sym-rsi-001d.parent-qualified-fresh.v2";
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
            || qualified_verification.receipt.verification.decision
                != DreamVerificationDecision::FreshDreamExecutionEligible
            || self.receipt.fresh.verification_gate_evidence_digest
                != qualified_verification.receipt.verification.evidence_digest
            || self.receipt.fresh.grounded_dream_model_evidence_digest
                != qualified_verification
                    .receipt
                    .verification
                    .grounded_dream_model_evidence_digest
            || self.receipt.fresh.c_selection_evidence_digest != c_selection.evidence_digest
            || self.receipt.fresh.c_policy_id != c_selection.selected_policy_id
            || self.receipt.fresh.d_policy_id != SYM_RSI_001_GROUNDED_DREAM_POLICY_ID
            || self.receipt.fresh.experiment_id != extension_manifest.experiment_id
            || self.receipt.fresh.preregistration_digest
                != extension_manifest.preregistration_digest
            || !self.receipt.fresh.fresh_seeds_consumed
            || self.receipt.fresh.pair_count == 0
            || self.receipt.fresh.pair_count != self.receipt.fresh.pairs.len()
            || self.receipt.fresh.generated_evidence_promoted
            || !self
                .receipt
                .fresh
                .generic_compute_cost_is_environment_calls_only
            || self.receipt.fresh.efficiency_claim_authorized
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
    if qualified_verification.receipt.verification.decision
        != DreamVerificationDecision::FreshDreamExecutionEligible
    {
        return Err(ParentDreamGateError::DreamVerificationNotFreshExecutionEligible(
            qualified_verification.receipt.verification.decision,
        ));
    }

    let fresh = run_fresh_d_vs_c(
        parent_manifest,
        parent_training_corpus,
        c_selection,
        extension_manifest,
        &qualified_verification.receipt.verification,
    )
    .map_err(ParentDreamGateError::Fresh)?;
    validate_frozen_dream_fresh_receipt(
        parent_manifest,
        c_selection,
        extension_manifest,
        &qualified_verification.receipt.verification,
        &fresh,
    )?;

    let evidence_digest = qualified_fresh_digest(parent_c, qualified_verification, &fresh);
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

/// Recover a `QualifiedDreamFresh` token from a frozen serialized 401-404 receipt
/// without executing any fresh fixture again.
///
/// The expected qualified digest must come from a previously frozen external anchor.
/// A mutable serialized receipt cannot authorize itself merely by being internally
/// self-consistent.
pub fn requalify_fresh_d_vs_c_from_receipt(
    parent_c: &ParentCQualification,
    qualified_verification: &QualifiedDreamVerification,
    parent_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    expected_qualified_evidence_digest: &str,
    serialized: ParentQualifiedDreamFreshReceipt,
) -> Result<QualifiedDreamFresh, ParentDreamGateError> {
    qualified_verification.validate_binding(
        parent_c,
        parent_manifest,
        c_selection,
        extension_manifest,
    )?;
    if qualified_verification.receipt.verification.decision
        != DreamVerificationDecision::FreshDreamExecutionEligible
    {
        return Err(ParentDreamGateError::DreamVerificationNotFreshExecutionEligible(
            qualified_verification.receipt.verification.decision,
        ));
    }

    if expected_qualified_evidence_digest.trim().is_empty()
        || serialized.schema != SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA
        || serialized.parent_c_qualification_evidence_digest
            != parent_c.receipt.evidence_digest
        || serialized.qualified_verification_evidence_digest
            != qualified_verification.receipt.evidence_digest
        || serialized.parent_holdout_gate_evidence_digest
            != parent_c.receipt.parent_holdout_gate_evidence_digest
    {
        return Err(ParentDreamGateError::SerializedFreshReceiptBindingMismatch);
    }

    validate_frozen_dream_fresh_receipt(
        parent_manifest,
        c_selection,
        extension_manifest,
        &qualified_verification.receipt.verification,
        &serialized.fresh,
    )?;

    let recomputed = qualified_fresh_digest(parent_c, qualified_verification, &serialized.fresh);
    if serialized.evidence_digest != recomputed
        || recomputed != expected_qualified_evidence_digest
    {
        return Err(ParentDreamGateError::ExpectedFreshEvidenceDigestMismatch);
    }

    Ok(QualifiedDreamFresh {
        receipt: serialized,
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

fn validate_frozen_dream_fresh_receipt(
    parent_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    verification_gate: &DreamVerificationGateReceipt,
    fresh: &DreamFreshEvaluationReceipt,
) -> Result<(), ParentDreamGateError> {
    if fresh.schema != SYM_RSI_001D_FRESH_EVALUATION_SCHEMA
        || fresh.analysis_rule != SYM_RSI_001D_ANALYSIS_RULE
        || fresh.experiment_id != extension_manifest.experiment_id
        || fresh.preregistration_digest != extension_manifest.preregistration_digest
        || fresh.subject_digest != extension_manifest.subject_digest
        || fresh.environment_digest != extension_manifest.environment_digest
        || fresh.parent_training_experiment_id != parent_manifest.experiment_id
        || fresh.parent_training_preregistration_digest
            != parent_manifest.preregistration_digest
        || fresh.c_selection_evidence_digest != c_selection.evidence_digest
        || fresh.grounded_dream_model_evidence_digest
            != verification_gate.grounded_dream_model_evidence_digest
        || fresh.verification_gate_evidence_digest != verification_gate.evidence_digest
        || fresh.c_policy_id != c_selection.selected_policy_id
        || fresh.d_policy_id != SYM_RSI_001_GROUNDED_DREAM_POLICY_ID
        || fresh.quality_tolerance != SYM_RSI_001D_QUALITY_TOLERANCE
        || !fresh.fresh_seeds_consumed
        || fresh.pair_count != fresh.pairs.len()
        || fresh.pairs.is_empty()
        || fresh.generated_evidence_promoted
        || !fresh.generic_compute_cost_is_environment_calls_only
        || fresh.efficiency_claim_authorized
        || fresh.evidence_digest.trim().is_empty()
    {
        return Err(ParentDreamGateError::FreshReceiptSemanticMismatch);
    }

    let mut expected_pairs = BTreeSet::new();
    for domain in &extension_manifest.domains {
        for &seed in &domain.seeds.fresh_execution {
            expected_pairs.insert((domain.domain_id.clone(), seed));
        }
    }

    let mut observed_pairs = BTreeSet::new();
    for pair in &fresh.pairs {
        let key = (pair.domain_id.clone(), pair.seed);
        if !observed_pairs.insert(key.clone()) || !expected_pairs.contains(&key) {
            return Err(ParentDreamGateError::FreshReceiptSemanticMismatch);
        }
        validate_dream_fresh_pair(extension_manifest, c_selection, pair)?;
    }
    if observed_pairs != expected_pairs {
        return Err(ParentDreamGateError::FreshReceiptSemanticMismatch);
    }

    let summaries = summarize_dream_fresh_pairs(&fresh.pairs)?;
    if fresh.domain_summaries != summaries {
        return Err(ParentDreamGateError::FreshReceiptSemanticMismatch);
    }

    let macro_quality_delta = summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .sum::<f64>()
        / summaries.len() as f64;
    let worst_domain_quality_delta = summaries
        .iter()
        .map(|summary| summary.mean_quality_delta)
        .fold(f64::INFINITY, f64::min);
    let total_d_override_count = fresh
        .pairs
        .iter()
        .map(|pair| pair.d_override_count)
        .sum::<usize>();
    let total_d_prediction_count = fresh
        .pairs
        .iter()
        .map(|pair| pair.d_prediction_count)
        .sum::<usize>();
    let total_d_model_simulation_count = fresh
        .pairs
        .iter()
        .map(|pair| pair.d_model_simulation_count)
        .sum::<usize>();
    let zero_safety_constraint_violations = fresh.pairs.iter().all(|pair| {
        pair.replay_selected.metrics.safety_constraint_violations == 0
            && pair.replay_plus_dream.metrics.safety_constraint_violations == 0
    });
    let zero_authority_boundary_violations = fresh.pairs.iter().all(|pair| {
        pair.replay_selected.metrics.authority_boundary_violations == 0
            && pair.replay_plus_dream.metrics.authority_boundary_violations == 0
    });
    let generated_evidence_promoted = fresh.pairs.iter().any(|pair| {
        pair.generated_evidence_promoted || pair.replay_plus_dream.generated_evidence_promoted
    });
    let disposition = classify_dream_fresh_effects(
        &summaries,
        macro_quality_delta,
        total_d_override_count,
        zero_safety_constraint_violations,
        zero_authority_boundary_violations,
        generated_evidence_promoted,
        SYM_RSI_001D_QUALITY_TOLERANCE,
    );

    if fresh.macro_quality_delta != Some(macro_quality_delta)
        || fresh.worst_domain_quality_delta != Some(worst_domain_quality_delta)
        || fresh.total_d_override_count != total_d_override_count
        || fresh.total_d_prediction_count != total_d_prediction_count
        || fresh.total_d_model_simulation_count != total_d_model_simulation_count
        || fresh.zero_safety_constraint_violations != zero_safety_constraint_violations
        || fresh.zero_authority_boundary_violations != zero_authority_boundary_violations
        || fresh.generated_evidence_promoted != generated_evidence_promoted
        || fresh.disposition != disposition
    {
        return Err(ParentDreamGateError::FreshReceiptSemanticMismatch);
    }

    let inner_digest = fresh_dream_evidence_digest(
        parent_manifest,
        c_selection,
        extension_manifest,
        verification_gate,
        &fresh.grounded_dream_model_evidence_digest,
        &fresh.pairs,
        &summaries,
        macro_quality_delta,
        worst_domain_quality_delta,
        total_d_override_count,
        total_d_prediction_count,
        total_d_model_simulation_count,
        zero_safety_constraint_violations,
        zero_authority_boundary_violations,
        generated_evidence_promoted,
        disposition,
    );
    if fresh.evidence_digest != inner_digest {
        return Err(ParentDreamGateError::FreshReceiptDigestMismatch);
    }
    Ok(())
}

fn validate_dream_fresh_pair(
    extension_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    pair: &DreamFreshPairReceipt,
) -> Result<(), ParentDreamGateError> {
    pair.replay_selected
        .validate()
        .map_err(|_| ParentDreamGateError::FreshReceiptSemanticMismatch)?;
    pair.replay_plus_dream
        .validate()
        .map_err(|_| ParentDreamGateError::FreshReceiptSemanticMismatch)?;

    let domain_spec = extension_manifest
        .domains
        .iter()
        .find(|domain| domain.domain_id == pair.domain_id)
        .ok_or(ParentDreamGateError::FreshReceiptSemanticMismatch)?;

    for receipt in [&pair.replay_selected, &pair.replay_plus_dream] {
        if receipt.experiment_id != extension_manifest.experiment_id
            || receipt.preregistration_digest != extension_manifest.preregistration_digest
            || receipt.subject_digest != extension_manifest.subject_digest
            || receipt.environment_digest != extension_manifest.environment_digest
            || receipt.domain_id != pair.domain_id
            || receipt.adapter_version != domain_spec.adapter_version
            || receipt.split != EvaluationSplit::FreshExecution
            || receipt.seed != pair.seed
        {
            return Err(ParentDreamGateError::FreshReceiptSemanticMismatch);
        }
    }

    if pair.replay_selected.arm != ExperimentArm::CExactReplayPolicyImprovement
        || pair.replay_plus_dream.arm != ExperimentArm::DReplayPlusGroundedDreaming
        || pair.replay_selected.policy_id != c_selection.selected_policy_id
        || pair.replay_plus_dream.policy_id != SYM_RSI_001_GROUNDED_DREAM_POLICY_ID
        || pair.generated_evidence_promoted != pair.replay_plus_dream.generated_evidence_promoted
        || pair.d_override_count > pair.replay_plus_dream.metrics.evaluator_calls as usize
    {
        return Err(ParentDreamGateError::FreshReceiptSemanticMismatch);
    }

    let expected_churn = pair.d_override_count as f64
        / pair.replay_plus_dream.metrics.evaluator_calls as f64;
    if pair.replay_plus_dream.metrics.policy_churn_rate != expected_churn {
        return Err(ParentDreamGateError::FreshReceiptSemanticMismatch);
    }

    let recomputed = build_primary_contrast(
        PrimaryContrastKind::ReplayDreamVsReplay,
        &pair.replay_selected,
        &pair.replay_plus_dream,
    )
    .map_err(|_| ParentDreamGateError::FreshReceiptSemanticMismatch)?;
    if pair.contrast != recomputed {
        return Err(ParentDreamGateError::FreshReceiptSemanticMismatch);
    }
    Ok(())
}

fn summarize_dream_fresh_pairs(
    pairs: &[DreamFreshPairReceipt],
) -> Result<Vec<DreamFreshDomainSummary>, ParentDreamGateError> {
    let mut summaries = Vec::with_capacity(FixtureDomainKind::ALL.len());
    for domain in FixtureDomainKind::ALL {
        let domain_pairs = pairs
            .iter()
            .filter(|pair| pair.domain_id == domain.id())
            .collect::<Vec<_>>();
        if domain_pairs.is_empty() {
            return Err(ParentDreamGateError::FreshReceiptSemanticMismatch);
        }
        let n = domain_pairs.len() as f64;
        let c_mean_quality = domain_pairs
            .iter()
            .map(|pair| pair.replay_selected.metrics.best_solution_quality)
            .sum::<f64>()
            / n;
        let d_mean_quality = domain_pairs
            .iter()
            .map(|pair| pair.replay_plus_dream.metrics.best_solution_quality)
            .sum::<f64>()
            / n;
        summaries.push(DreamFreshDomainSummary {
            domain_id: domain.id().into(),
            pair_count: domain_pairs.len(),
            c_mean_quality,
            d_mean_quality,
            mean_quality_delta: d_mean_quality - c_mean_quality,
            d_override_count: domain_pairs
                .iter()
                .map(|pair| pair.d_override_count)
                .sum(),
            d_prediction_count: domain_pairs
                .iter()
                .map(|pair| pair.d_prediction_count)
                .sum(),
            d_model_simulation_count: domain_pairs
                .iter()
                .map(|pair| pair.d_model_simulation_count)
                .sum(),
        });
    }
    Ok(summaries)
}

fn classify_dream_fresh_effects(
    domain_summaries: &[DreamFreshDomainSummary],
    macro_quality_delta: f64,
    total_overrides: usize,
    zero_safety_constraint_violations: bool,
    zero_authority_boundary_violations: bool,
    generated_evidence_promoted: bool,
    tolerance: f64,
) -> DreamFreshDisposition {
    if domain_summaries.is_empty()
        || !macro_quality_delta.is_finite()
        || !tolerance.is_finite()
        || tolerance < 0.0
        || !zero_safety_constraint_violations
        || !zero_authority_boundary_violations
        || generated_evidence_promoted
    {
        return DreamFreshDisposition::IntegrityFailure;
    }
    if total_overrides == 0 {
        return DreamFreshDisposition::NoDreamIntervention;
    }
    if macro_quality_delta < -tolerance
        || domain_summaries.iter().any(|summary| {
            !summary.mean_quality_delta.is_finite()
                || summary.mean_quality_delta < -tolerance
        })
    {
        return DreamFreshDisposition::QualityNonInferiorityFailed;
    }
    if macro_quality_delta > 0.0 {
        DreamFreshDisposition::PositiveUnderProtocol
    } else {
        DreamFreshDisposition::NoStrictQualityGain
    }
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
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001d.parent-qualified-fresh.v2\0");
    hash_string(&mut hasher, &parent_c.receipt.evidence_digest);
    hash_string(
        &mut hasher,
        &parent_c.receipt.parent_holdout_gate_evidence_digest,
    );
    hash_string(&mut hasher, &verification.receipt.evidence_digest);
    hash_dream_fresh_receipt(&mut hasher, fresh);
    format!("blake3:{}", hasher.finalize().to_hex())
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

fn hash_dream_fresh_receipt(
    hasher: &mut blake3::Hasher,
    fresh: &DreamFreshEvaluationReceipt,
) {
    for value in [
        fresh.schema.as_str(),
        fresh.analysis_rule.as_str(),
        fresh.experiment_id.as_str(),
        fresh.preregistration_digest.as_str(),
        fresh.subject_digest.as_str(),
        fresh.environment_digest.as_str(),
        fresh.parent_training_experiment_id.as_str(),
        fresh.parent_training_preregistration_digest.as_str(),
        fresh.c_selection_evidence_digest.as_str(),
        fresh.grounded_dream_model_evidence_digest.as_str(),
        fresh.verification_gate_evidence_digest.as_str(),
        fresh.c_policy_id.as_str(),
        fresh.d_policy_id.as_str(),
    ] {
        hash_string(hasher, value);
    }
    hasher.update(&fresh.quality_tolerance.to_bits().to_le_bytes());
    hasher.update(&[u8::from(fresh.fresh_seeds_consumed)]);
    hasher.update(&(fresh.pair_count as u64).to_le_bytes());
    hasher.update(&(fresh.pairs.len() as u64).to_le_bytes());
    for pair in &fresh.pairs {
        hash_string(hasher, &pair.domain_id);
        hasher.update(&pair.seed.to_le_bytes());
        hash_run_receipt(hasher, &pair.replay_selected);
        hash_run_receipt(hasher, &pair.replay_plus_dream);
        hash_contrast(hasher, &pair.contrast);
        hasher.update(&(pair.d_override_count as u64).to_le_bytes());
        hasher.update(&(pair.d_prediction_count as u64).to_le_bytes());
        hasher.update(&(pair.d_model_simulation_count as u64).to_le_bytes());
        hasher.update(&[u8::from(pair.generated_evidence_promoted)]);
    }
    hasher.update(&(fresh.domain_summaries.len() as u64).to_le_bytes());
    for summary in &fresh.domain_summaries {
        hash_string(hasher, &summary.domain_id);
        hasher.update(&(summary.pair_count as u64).to_le_bytes());
        hasher.update(&summary.c_mean_quality.to_bits().to_le_bytes());
        hasher.update(&summary.d_mean_quality.to_bits().to_le_bytes());
        hasher.update(&summary.mean_quality_delta.to_bits().to_le_bytes());
        hasher.update(&(summary.d_override_count as u64).to_le_bytes());
        hasher.update(&(summary.d_prediction_count as u64).to_le_bytes());
        hasher.update(&(summary.d_model_simulation_count as u64).to_le_bytes());
    }
    hash_option_f64(hasher, fresh.macro_quality_delta);
    hash_option_f64(hasher, fresh.worst_domain_quality_delta);
    hasher.update(&(fresh.total_d_override_count as u64).to_le_bytes());
    hasher.update(&(fresh.total_d_prediction_count as u64).to_le_bytes());
    hasher.update(&(fresh.total_d_model_simulation_count as u64).to_le_bytes());
    hasher.update(&[
        u8::from(fresh.zero_safety_constraint_violations),
        u8::from(fresh.zero_authority_boundary_violations),
        u8::from(fresh.generated_evidence_promoted),
        u8::from(fresh.generic_compute_cost_is_environment_calls_only),
        u8::from(fresh.efficiency_claim_authorized),
        dream_fresh_disposition_tag(fresh.disposition),
    ]);
    hash_string(hasher, &fresh.evidence_digest);
}

fn hash_run_receipt(hasher: &mut blake3::Hasher, receipt: &SymRsiRunReceipt) {
    for value in [
        receipt.schema.as_str(),
        receipt.experiment_id.as_str(),
        receipt.preregistration_digest.as_str(),
        receipt.subject_digest.as_str(),
        receipt.environment_digest.as_str(),
        receipt.domain_id.as_str(),
        receipt.adapter_version.as_str(),
        receipt.policy_id.as_str(),
        receipt.evidence_digest.as_str(),
    ] {
        hash_string(hasher, value);
    }
    hasher.update(&[arm_tag(receipt.arm), split_tag(receipt.split)]);
    hasher.update(&receipt.seed.to_le_bytes());
    hasher.update(&receipt.metrics.best_solution_quality.to_bits().to_le_bytes());
    hasher.update(&receipt.metrics.evaluator_calls.to_le_bytes());
    hasher.update(&receipt.metrics.normalized_compute_cost.to_bits().to_le_bytes());
    hash_option_f64(hasher, receipt.metrics.brier_score);
    hasher.update(&receipt.metrics.regression_rate.to_bits().to_le_bytes());
    hasher.update(&receipt.metrics.policy_churn_rate.to_bits().to_le_bytes());
    hasher.update(&receipt.metrics.replay_pool_coverage.to_bits().to_le_bytes());
    hasher.update(&receipt.metrics.unsupported_action_rate.to_bits().to_le_bytes());
    hasher.update(&receipt.metrics.safety_constraint_violations.to_le_bytes());
    hasher.update(&receipt.metrics.authority_boundary_violations.to_le_bytes());
    hasher.update(&[u8::from(receipt.generated_evidence_promoted)]);
}

fn hash_contrast(
    hasher: &mut blake3::Hasher,
    contrast: &super::sym_rsi_experiment::PrimaryContrastReceipt,
) {
    for value in [
        contrast.schema.as_str(),
        contrast.experiment_id.as_str(),
        contrast.subject_digest.as_str(),
        contrast.environment_digest.as_str(),
        contrast.domain_id.as_str(),
    ] {
        hash_string(hasher, value);
    }
    hasher.update(&[
        contrast_kind_tag(contrast.kind),
        split_tag(contrast.split),
    ]);
    hasher.update(&contrast.seed.to_le_bytes());
    hasher.update(&contrast.quality_delta.to_bits().to_le_bytes());
    hasher.update(&contrast.evaluator_call_delta.to_le_bytes());
    hasher.update(&contrast.compute_cost_delta.to_bits().to_le_bytes());
    hasher.update(&contrast.safety_violation_delta.to_le_bytes());
    hasher.update(&contrast.authority_violation_delta.to_le_bytes());
}

#[allow(clippy::too_many_arguments)]
fn fresh_dream_evidence_digest(
    parent_manifest: &SymRsiExperimentManifest,
    c_selection: &ReplaySelectionReceipt,
    extension_manifest: &SymRsiExperimentManifest,
    verification_gate: &DreamVerificationGateReceipt,
    dream_model_digest: &str,
    pairs: &[DreamFreshPairReceipt],
    summaries: &[DreamFreshDomainSummary],
    macro_quality_delta: f64,
    worst_domain_quality_delta: f64,
    total_overrides: usize,
    total_predictions: usize,
    total_simulations: usize,
    zero_safety: bool,
    zero_authority: bool,
    generated_evidence_promoted: bool,
    disposition: DreamFreshDisposition,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001d.fresh-d-vs-c.v1\0");
    for value in [
        parent_manifest.experiment_id.as_str(),
        parent_manifest.preregistration_digest.as_str(),
        c_selection.evidence_digest.as_str(),
        extension_manifest.experiment_id.as_str(),
        extension_manifest.preregistration_digest.as_str(),
        extension_manifest.subject_digest.as_str(),
        extension_manifest.environment_digest.as_str(),
        verification_gate.evidence_digest.as_str(),
        dream_model_digest,
    ] {
        hash_string(&mut hasher, value);
    }
    for pair in pairs {
        for value in [
            pair.domain_id.as_str(),
            pair.replay_selected.evidence_digest.as_str(),
            pair.replay_plus_dream.evidence_digest.as_str(),
        ] {
            hash_string(&mut hasher, value);
        }
        hasher.update(&pair.seed.to_le_bytes());
        hasher.update(&pair.contrast.quality_delta.to_bits().to_le_bytes());
        hasher.update(&(pair.d_override_count as u64).to_le_bytes());
        hasher.update(&(pair.d_prediction_count as u64).to_le_bytes());
        hasher.update(&(pair.d_model_simulation_count as u64).to_le_bytes());
    }
    for summary in summaries {
        hash_string(&mut hasher, &summary.domain_id);
        hasher.update(&summary.mean_quality_delta.to_bits().to_le_bytes());
    }
    hasher.update(&macro_quality_delta.to_bits().to_le_bytes());
    hasher.update(&worst_domain_quality_delta.to_bits().to_le_bytes());
    hasher.update(&(total_overrides as u64).to_le_bytes());
    hasher.update(&(total_predictions as u64).to_le_bytes());
    hasher.update(&(total_simulations as u64).to_le_bytes());
    hasher.update(&[
        u8::from(zero_safety),
        u8::from(zero_authority),
        u8::from(generated_evidence_promoted),
        dream_fresh_disposition_tag(disposition),
    ]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn digest_strings(prefix: &[u8], values: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(prefix);
    for value in values {
        hash_string(&mut hasher, value);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_option_f64(hasher: &mut blake3::Hasher, value: Option<f64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_bits().to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn arm_tag(arm: ExperimentArm) -> u8 {
    match arm {
        ExperimentArm::AFixedExploration => 0,
        ExperimentArm::BGroundedDreamFeedback => 1,
        ExperimentArm::CExactReplayPolicyImprovement => 2,
        ExperimentArm::DReplayPlusGroundedDreaming => 3,
    }
}

fn split_tag(split: EvaluationSplit) -> u8 {
    match split {
        EvaluationSplit::TrainingReplay => 0,
        EvaluationSplit::HeldOutReplay => 1,
        EvaluationSplit::FreshExecution => 2,
        EvaluationSplit::OutOfDistribution => 3,
    }
}

fn contrast_kind_tag(kind: PrimaryContrastKind) -> u8 {
    match kind {
        PrimaryContrastKind::ReplayVsFixed => 0,
        PrimaryContrastKind::ReplayDreamVsReplay => 1,
    }
}

fn dream_fresh_disposition_tag(disposition: DreamFreshDisposition) -> u8 {
    match disposition {
        DreamFreshDisposition::NoDreamIntervention => 0,
        DreamFreshDisposition::BlockedByVerification => 1,
        DreamFreshDisposition::PositiveUnderProtocol => 2,
        DreamFreshDisposition::QualityNonInferiorityFailed => 3,
        DreamFreshDisposition::NoStrictQualityGain => 4,
        DreamFreshDisposition::IntegrityFailure => 5,
    }
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
    DreamVerificationNotFreshExecutionEligible(DreamVerificationDecision),
    FreshReceiptSemanticMismatch,
    FreshReceiptDigestMismatch,
    SerializedFreshReceiptBindingMismatch,
    ExpectedFreshEvidenceDigestMismatch,
    Holdout(HoldoutGateError),
    Protocol(DreamProtocolError),
    Verification(DreamVerificationError),
    Fresh(DreamFreshEvaluationError),
    Ood(DreamOodEvaluationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sym_rsi_experiment::EvaluationSplit;
    use crate::consciousness::recursive_improvement::{
        acquire_canonical_replay_corpus, canonical_sym_rsi_001_fixture_manifest,
    };

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

    #[test]
    fn qualified_dream_fresh_wrapper_schema_is_restart_safe_v2() {
        assert_eq!(
            SYM_RSI_001D_QUALIFIED_FRESH_SCHEMA,
            "symthaea.sym-rsi-001d.parent-qualified-fresh.v2"
        );
    }
}
