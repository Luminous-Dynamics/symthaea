// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prospective pre-HeldOut custody for EUREKA-002 V2.
//!
//! This module freezes all target/comparator/schedule/scoring lineage before any
//! HeldOut prediction is allowed. It performs no HeldOut prediction itself.

#![allow(dead_code)]

use symthaea_fep::FepHeldOutSubject;

use super::analysis_plan::{ANALYSIS_PLAN_REVISION, EUREKA_002_ANALYSIS_PLAN_V1};
use super::v2_corpus_schedule::{
    V2ScheduleMaterializationError, materialize_canonical_corpora,
};
use super::v2_fep_development::{V2FepDevelopmentArtifact, V2FepDevelopmentReceipt};
use super::v2_frozen_comparator::V2FrozenComparatorSubject;
use super::v2_heldout_plan::{V2HeldOutPlan, materialize_heldout_plan};
use super::v2_public_schema::public_schema_commitment;
use super::v2_selected_comparator::{
    V2SelectedComparatorError, V2SelectedComparatorSubject,
};
use super::v2_selection_authorization::{
    V2ComparatorSelectionOutcome, V2SelectionAuthorizationError, execute_calibration_selection,
};
use super::v2_target_contract::{V2_FEP_TARGET_ADAPTER_REVISION, V2FepTargetContract};

pub(super) const V2_SCORER_SOURCE_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.CONSEQUENCE_SCORER_SOURCE.v1";
pub(super) const V2_STATIC_PREHELDOUT_REVISION: &str =
    "EUREKA.002.V2.STATIC_PREHELDOUT.v1";
pub(super) const V2_PREHELDOUT_MANIFEST_REVISION: &str =
    "EUREKA.002.V2.PREHELDOUT_MANIFEST.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2PreHeldOutError {
    Materialization(V2ScheduleMaterializationError),
    Selection(V2SelectionAuthorizationError),
    SelectedComparator(V2SelectedComparatorError),
    ComparatorSelectionInconclusive,
    StaticScheduleMismatch,
    TargetSchemaMismatch,
    TargetDevelopmentCommitmentMismatch,
    TargetContractCommitmentMismatch,
    TargetLearnedSubjectMismatch,
    TargetScheduleMismatch,
    ComparatorDevelopmentCommitmentMismatch,
    AnalysisPlanMismatch,
}

impl From<V2ScheduleMaterializationError> for V2PreHeldOutError {
    fn from(value: V2ScheduleMaterializationError) -> Self {
        Self::Materialization(value)
    }
}

impl From<V2SelectionAuthorizationError> for V2PreHeldOutError {
    fn from(value: V2SelectionAuthorizationError) -> Self {
        Self::Selection(value)
    }
}

impl From<V2SelectedComparatorError> for V2PreHeldOutError {
    fn from(value: V2SelectedComparatorError) -> Self {
        Self::SelectedComparator(value)
    }
}

/// Target-independent side of the campaign. It can be frozen before the target
/// is trained because it depends only on canonical schedule/comparator evidence.
#[derive(Debug)]
pub(super) struct V2StaticPreHeldOutBundle {
    selected_comparator: V2SelectedComparatorSubject,
    heldout_plan: V2HeldOutPlan,
    analysis_plan_commitment: [u8; 32],
    scorer_source_commitment: [u8; 32],
    full_schedule_root: [u8; 32],
    commitment: [u8; 32],
}

impl V2StaticPreHeldOutBundle {
    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2PreHeldOutManifest {
    public_schema_commitment: [u8; 32],
    full_schedule_root: [u8; 32],
    development_receipt_commitment: [u8; 32],
    target_contract_commitment: [u8; 32],
    learned_subject_commitment: [u8; 32],
    selected_comparator_commitment: [u8; 32],
    comparator_custody_commitment: [u8; 32],
    selection_authorization_commitment: [u8; 32],
    calibration_corpus_commitment: [u8; 32],
    heldout_plan_commitment: [u8; 32],
    heldout_ordered_root: [u8; 32],
    analysis_plan_commitment: [u8; 32],
    scorer_source_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2PreHeldOutManifest {
    pub(super) const fn full_schedule_root(self) -> [u8; 32] {
        self.full_schedule_root
    }

    pub(super) const fn development_receipt_commitment(self) -> [u8; 32] {
        self.development_receipt_commitment
    }

    pub(super) const fn target_contract_commitment(self) -> [u8; 32] {
        self.target_contract_commitment
    }

    pub(super) const fn learned_subject_commitment(self) -> [u8; 32] {
        self.learned_subject_commitment
    }

    pub(super) const fn selected_comparator_commitment(self) -> [u8; 32] {
        self.selected_comparator_commitment
    }

    pub(super) const fn selection_authorization_commitment(self) -> [u8; 32] {
        self.selection_authorization_commitment
    }

    pub(super) const fn heldout_ordered_root(self) -> [u8; 32] {
        self.heldout_ordered_root
    }

    pub(super) const fn scorer_source_commitment(self) -> [u8; 32] {
        self.scorer_source_commitment
    }

    pub(super) const fn commitment(self) -> [u8; 32] {
        self.commitment
    }
}

/// Non-Clone campaign capability. It owns the sealed target artifact, narrowed
/// selected comparator, and exact HeldOut plan. No training/refit APIs exist.
#[derive(Debug)]
pub(super) struct V2PreHeldOutCampaignCapability {
    development: V2FepDevelopmentArtifact,
    selected_comparator: V2SelectedComparatorSubject,
    heldout_plan: V2HeldOutPlan,
    manifest: V2PreHeldOutManifest,
}

impl V2PreHeldOutCampaignCapability {
    pub(super) fn target_subject(&self) -> &FepHeldOutSubject {
        self.development.subject()
    }

    pub(super) const fn target_contract(&self) -> V2FepTargetContract {
        self.development.target_contract()
    }

    pub(super) const fn development_receipt(&self) -> V2FepDevelopmentReceipt {
        self.development.receipt()
    }

    pub(super) fn selected_comparator(&self) -> &V2SelectedComparatorSubject {
        &self.selected_comparator
    }

    pub(super) fn heldout_plan(&self) -> &V2HeldOutPlan {
        &self.heldout_plan
    }

    pub(super) const fn manifest(&self) -> V2PreHeldOutManifest {
        self.manifest
    }
}

pub(super) fn prepare_static_preheldout_bundle(
) -> Result<V2StaticPreHeldOutBundle, V2PreHeldOutError> {
    let corpora = materialize_canonical_corpora()?;
    let comparator_subject = V2FrozenComparatorSubject::freeze(corpora.development());
    let selection = execute_calibration_selection(
        corpora.development(),
        corpora.calibration(),
        &comparator_subject,
    )?;
    let V2ComparatorSelectionOutcome::Selected(authorization) = selection else {
        return Err(V2PreHeldOutError::ComparatorSelectionInconclusive);
    };
    let selected_comparator = V2SelectedComparatorSubject::freeze(
        corpora.development(),
        comparator_subject,
        &authorization,
    )?;
    let heldout_plan = materialize_heldout_plan()?;
    if heldout_plan.full_schedule_root() != corpora.schedule_root() {
        return Err(V2PreHeldOutError::StaticScheduleMismatch);
    }
    let analysis_plan_commitment = EUREKA_002_ANALYSIS_PLAN_V1.cryptographic_commitment();
    if selected_comparator.analysis_plan_commitment() != analysis_plan_commitment {
        return Err(V2PreHeldOutError::AnalysisPlanMismatch);
    }
    let scorer_source_commitment = consequence_scorer_source_commitment();
    let full_schedule_root = corpora.schedule_root();
    let commitment = static_preheldout_commitment(
        full_schedule_root,
        selected_comparator.commitment(),
        selected_comparator.selection_authorization_commitment(),
        selected_comparator.calibration_corpus_commitment(),
        heldout_plan.commitment(),
        heldout_plan.ordered_root(),
        analysis_plan_commitment,
        scorer_source_commitment,
    );
    Ok(V2StaticPreHeldOutBundle {
        selected_comparator,
        heldout_plan,
        analysis_plan_commitment,
        scorer_source_commitment,
        full_schedule_root,
        commitment,
    })
}

pub(super) fn freeze_preheldout_campaign(
    development: V2FepDevelopmentArtifact,
    static_bundle: V2StaticPreHeldOutBundle,
) -> Result<V2PreHeldOutCampaignCapability, V2PreHeldOutError> {
    let receipt = development.receipt();
    let target_contract = development.target_contract();

    if target_contract.public_schema_commitment() != public_schema_commitment()
        || static_bundle.selected_comparator.schema_commitment() != public_schema_commitment()
    {
        return Err(V2PreHeldOutError::TargetSchemaMismatch);
    }
    if receipt.development_corpus_commitment()
        != static_bundle.selected_comparator.fit_corpus_commitment()
    {
        return Err(V2PreHeldOutError::ComparatorDevelopmentCommitmentMismatch);
    }
    if receipt.full_schedule_root() != static_bundle.full_schedule_root
        || receipt.full_schedule_root() != static_bundle.heldout_plan.full_schedule_root()
    {
        return Err(V2PreHeldOutError::TargetScheduleMismatch);
    }
    if target_contract.commitment() != receipt.target_contract_commitment() {
        return Err(V2PreHeldOutError::TargetContractCommitmentMismatch);
    }
    if target_contract.learned_subject_commitment() != receipt.learned_snapshot_commitment()
        || development.subject().commitment() != receipt.learned_snapshot_commitment()
    {
        return Err(V2PreHeldOutError::TargetLearnedSubjectMismatch);
    }
    if static_bundle.analysis_plan_commitment
        != EUREKA_002_ANALYSIS_PLAN_V1.cryptographic_commitment()
        || static_bundle.selected_comparator.analysis_plan_commitment()
            != static_bundle.analysis_plan_commitment
    {
        return Err(V2PreHeldOutError::AnalysisPlanMismatch);
    }

    let learned_subject_commitment = *receipt.learned_snapshot_commitment().as_bytes();
    let manifest_commitment = preheldout_manifest_commitment(
        receipt.commitment(),
        target_contract.commitment(),
        learned_subject_commitment,
        static_bundle.selected_comparator.commitment(),
        static_bundle.selected_comparator.custody_commitment(),
        static_bundle
            .selected_comparator
            .selection_authorization_commitment(),
        static_bundle
            .selected_comparator
            .calibration_corpus_commitment(),
        static_bundle.heldout_plan.commitment(),
        static_bundle.heldout_plan.ordered_root(),
        static_bundle.full_schedule_root,
        static_bundle.analysis_plan_commitment,
        static_bundle.scorer_source_commitment,
        static_bundle.commitment,
    );
    let manifest = V2PreHeldOutManifest {
        public_schema_commitment: public_schema_commitment(),
        full_schedule_root: static_bundle.full_schedule_root,
        development_receipt_commitment: receipt.commitment(),
        target_contract_commitment: target_contract.commitment(),
        learned_subject_commitment,
        selected_comparator_commitment: static_bundle.selected_comparator.commitment(),
        comparator_custody_commitment: static_bundle.selected_comparator.custody_commitment(),
        selection_authorization_commitment: static_bundle
            .selected_comparator
            .selection_authorization_commitment(),
        calibration_corpus_commitment: static_bundle
            .selected_comparator
            .calibration_corpus_commitment(),
        heldout_plan_commitment: static_bundle.heldout_plan.commitment(),
        heldout_ordered_root: static_bundle.heldout_plan.ordered_root(),
        analysis_plan_commitment: static_bundle.analysis_plan_commitment,
        scorer_source_commitment: static_bundle.scorer_source_commitment,
        commitment: manifest_commitment,
    };

    Ok(V2PreHeldOutCampaignCapability {
        development,
        selected_comparator: static_bundle.selected_comparator,
        heldout_plan: static_bundle.heldout_plan,
        manifest,
    })
}

pub(super) fn consequence_scorer_source_commitment() -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_SCORER_SOURCE_COMMITMENT_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, include_str!("consequence.rs").as_bytes());
    *blake3::hash(&bytes).as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn static_preheldout_commitment(
    full_schedule_root: [u8; 32],
    selected_comparator_commitment: [u8; 32],
    selection_authorization_commitment: [u8; 32],
    calibration_corpus_commitment: [u8; 32],
    heldout_plan_commitment: [u8; 32],
    heldout_ordered_root: [u8; 32],
    analysis_plan_commitment: [u8; 32],
    scorer_source_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_STATIC_PREHELDOUT_REVISION.as_bytes());
    encode_bytes(&mut bytes, ANALYSIS_PLAN_REVISION.as_bytes());
    encode_bytes(&mut bytes, V2_FEP_TARGET_ADAPTER_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&full_schedule_root);
    bytes.extend_from_slice(&selected_comparator_commitment);
    bytes.extend_from_slice(&selection_authorization_commitment);
    bytes.extend_from_slice(&calibration_corpus_commitment);
    bytes.extend_from_slice(&heldout_plan_commitment);
    bytes.extend_from_slice(&heldout_ordered_root);
    bytes.extend_from_slice(&analysis_plan_commitment);
    bytes.extend_from_slice(&scorer_source_commitment);
    *blake3::hash(&bytes).as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn preheldout_manifest_commitment(
    development_receipt_commitment: [u8; 32],
    target_contract_commitment: [u8; 32],
    learned_subject_commitment: [u8; 32],
    selected_comparator_commitment: [u8; 32],
    comparator_custody_commitment: [u8; 32],
    selection_authorization_commitment: [u8; 32],
    calibration_corpus_commitment: [u8; 32],
    heldout_plan_commitment: [u8; 32],
    heldout_ordered_root: [u8; 32],
    full_schedule_root: [u8; 32],
    analysis_plan_commitment: [u8; 32],
    scorer_source_commitment: [u8; 32],
    static_preheldout_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_PREHELDOUT_MANIFEST_REVISION.as_bytes());
    encode_bytes(&mut bytes, ANALYSIS_PLAN_REVISION.as_bytes());
    encode_bytes(&mut bytes, V2_FEP_TARGET_ADAPTER_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&development_receipt_commitment);
    bytes.extend_from_slice(&target_contract_commitment);
    bytes.extend_from_slice(&learned_subject_commitment);
    bytes.extend_from_slice(&selected_comparator_commitment);
    bytes.extend_from_slice(&comparator_custody_commitment);
    bytes.extend_from_slice(&selection_authorization_commitment);
    bytes.extend_from_slice(&calibration_corpus_commitment);
    bytes.extend_from_slice(&heldout_plan_commitment);
    bytes.extend_from_slice(&heldout_ordered_root);
    bytes.extend_from_slice(&full_schedule_root);
    bytes.extend_from_slice(&analysis_plan_commitment);
    bytes.extend_from_slice(&scorer_source_commitment);
    bytes.extend_from_slice(&static_preheldout_commitment);
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
    use super::super::v2_fep_development::run_canonical_v2_fep_development;

    fn service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    #[test]
    fn canonical_preheldout_capability_binds_all_frozen_lineage_without_executing_heldout() {
        let static_bundle = prepare_static_preheldout_bundle().unwrap();
        let static_commitment = static_bundle.commitment();
        let development = run_canonical_v2_fep_development(&service()).unwrap();
        let development_receipt = development.receipt();
        let capability = freeze_preheldout_campaign(development, static_bundle).unwrap();
        let manifest = capability.manifest();

        assert_ne!(static_commitment, [0_u8; 32]);
        assert_ne!(manifest.commitment(), [0_u8; 32]);
        assert_eq!(
            manifest.development_receipt_commitment(),
            development_receipt.commitment()
        );
        assert_eq!(
            manifest.target_contract_commitment(),
            capability.target_contract().commitment()
        );
        assert_eq!(
            manifest.learned_subject_commitment(),
            *capability.target_subject().commitment().as_bytes()
        );
        assert_eq!(
            manifest.selected_comparator_commitment(),
            capability.selected_comparator().commitment()
        );
        assert_eq!(
            manifest.heldout_ordered_root(),
            capability.heldout_plan().ordered_root()
        );
        assert_eq!(capability.heldout_plan().ordered_rows().len(), 128);
        assert_eq!(
            manifest.scorer_source_commitment(),
            consequence_scorer_source_commitment()
        );
    }

    #[test]
    fn mismatched_schedule_lineage_fails_before_campaign_capability() {
        let mut static_bundle = prepare_static_preheldout_bundle().unwrap();
        static_bundle.full_schedule_root[0] ^= 0x80;
        let development = run_canonical_v2_fep_development(&service()).unwrap();
        assert!(matches!(
            freeze_preheldout_campaign(development, static_bundle),
            Err(V2PreHeldOutError::TargetScheduleMismatch)
        ));
    }

    #[test]
    fn scorer_source_commitment_is_nonzero_and_source_sensitive() {
        let canonical = consequence_scorer_source_commitment();
        let mut bytes = Vec::new();
        encode_bytes(
            &mut bytes,
            V2_SCORER_SOURCE_COMMITMENT_REVISION.as_bytes(),
        );
        let mut changed = include_str!("consequence.rs").as_bytes().to_vec();
        changed.push(b'\n');
        encode_bytes(&mut bytes, &changed);
        let changed_commitment = *blake3::hash(&bytes).as_bytes();
        assert_ne!(canonical, [0_u8; 32]);
        assert_ne!(canonical, changed_commitment);
    }
}
