// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Non-promotable one-row canary authorization for EUREKA-002 V2.
//!
//! This layer chooses exactly one HeldOut plumbing-canary row from frozen
//! evaluator structure alone, then binds that row to the exact pre-HeldOut
//! campaign and real-subject adapter pair. It does not execute either subject,
//! reveal an outcome, score a prediction, or mint confirmatory evidence.

#![allow(dead_code)]

use super::hidden_world::PublicAction;
use super::v2_corpus_schedule::V2ScheduleMaterializationError;
use super::v2_heldout_plan::{V2HeldOutPlan, materialize_heldout_plan};
use super::v2_preheldout_custody::V2PreHeldOutCampaignCapability;
use super::v2_public_schema::{V2PublicFamily, V2PublicState, action_index};
use super::v2_real_subject_adapters::{
    V2RealSubjectAdapterError, V2RealSubjectAdapterPair,
};

pub(super) const V2_CANARY_SELECTION_REVISION: &str =
    "EUREKA.002.V2.CANARY_SELECTION.v1";
pub(super) const V2_CANARY_AUTHORIZATION_REVISION: &str =
    "EUREKA.002.V2.CANARY_AUTHORIZATION.v1";
pub(super) const V2_CANARY_SOURCE_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.CANARY_AUTHORIZATION_SOURCE.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2CanaryEvidenceClass {
    PlumbingOnlyNonConfirmatory,
}

impl V2CanaryEvidenceClass {
    const fn stable_id(self) -> &'static str {
        match self {
            Self::PlumbingOnlyNonConfirmatory => "plumbing-only-non-confirmatory",
        }
    }
}

/// Target-blind selection basis. There is deliberately no subject/comparator,
/// manifest, prediction, outcome, score, clock, or RNG field here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2CanarySelectionBasis {
    full_schedule_root: [u8; 32],
    heldout_ordered_root: [u8; 32],
    row_count: u16,
}

impl V2CanarySelectionBasis {
    fn from_plan(plan: &V2HeldOutPlan) -> Result<Self, V2CanaryAuthorizationError> {
        let row_count = u16::try_from(plan.ordered_rows().len())
            .map_err(|_| V2CanaryAuthorizationError::InvalidHeldOutRowCount)?;
        if row_count == 0 {
            return Err(V2CanaryAuthorizationError::InvalidHeldOutRowCount);
        }
        Ok(Self {
            full_schedule_root: plan.full_schedule_root(),
            heldout_ordered_root: plan.ordered_root(),
            row_count,
        })
    }

    pub(super) const fn full_schedule_root(self) -> [u8; 32] {
        self.full_schedule_root
    }

    pub(super) const fn heldout_ordered_root(self) -> [u8; 32] {
        self.heldout_ordered_root
    }

    pub(super) const fn row_count(self) -> u16 {
        self.row_count
    }
}

#[derive(Debug)]
pub(super) enum V2CanaryAuthorizationError {
    Materialization(V2ScheduleMaterializationError),
    Adapter(V2RealSubjectAdapterError),
    InvalidHeldOutRowCount,
    ScheduleRootMismatch,
    HeldOutRootMismatch,
    RowIndexOutOfRange,
}

impl From<V2ScheduleMaterializationError> for V2CanaryAuthorizationError {
    fn from(value: V2ScheduleMaterializationError) -> Self {
        Self::Materialization(value)
    }
}

impl From<V2RealSubjectAdapterError> for V2CanaryAuthorizationError {
    fn from(value: V2RealSubjectAdapterError) -> Self {
        Self::Adapter(value)
    }
}

/// Move-only plumbing authorization. It contains pre-outcome row information
/// only. No post-state and no conversion into final-HeldOut evidence exists.
#[derive(Debug)]
pub(super) struct V2CanaryAuthorization {
    evidence_class: V2CanaryEvidenceClass,
    campaign_manifest_commitment: [u8; 32],
    adapter_pair_commitment: [u8; 32],
    adapter_source_commitment: [u8; 32],
    authorization_source_commitment: [u8; 32],
    selection_commitment: [u8; 32],
    full_schedule_root: [u8; 32],
    heldout_ordered_root: [u8; 32],
    row_index: u16,
    row_identity: [u8; 32],
    family: V2PublicFamily,
    pre: V2PublicState,
    action: PublicAction,
    commitment: [u8; 32],
}

impl V2CanaryAuthorization {
    pub(super) const fn evidence_class(&self) -> V2CanaryEvidenceClass {
        self.evidence_class
    }

    pub(super) const fn campaign_manifest_commitment(&self) -> [u8; 32] {
        self.campaign_manifest_commitment
    }

    pub(super) const fn adapter_pair_commitment(&self) -> [u8; 32] {
        self.adapter_pair_commitment
    }

    pub(super) const fn adapter_source_commitment(&self) -> [u8; 32] {
        self.adapter_source_commitment
    }

    pub(super) const fn selection_commitment(&self) -> [u8; 32] {
        self.selection_commitment
    }

    pub(super) const fn row_index(&self) -> u16 {
        self.row_index
    }

    pub(super) const fn row_identity(&self) -> [u8; 32] {
        self.row_identity
    }

    pub(super) const fn family(&self) -> V2PublicFamily {
        self.family
    }

    pub(super) const fn pre(&self) -> V2PublicState {
        self.pre
    }

    pub(super) const fn action(&self) -> PublicAction {
        self.action
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

pub(super) fn freeze_canary_authorization(
    campaign: &V2PreHeldOutCampaignCapability,
) -> Result<V2CanaryAuthorization, V2CanaryAuthorizationError> {
    let plan = materialize_heldout_plan()?;
    let manifest = campaign.manifest();
    if plan.full_schedule_root() != manifest.full_schedule_root() {
        return Err(V2CanaryAuthorizationError::ScheduleRootMismatch);
    }
    if plan.ordered_root() != manifest.heldout_ordered_root() {
        return Err(V2CanaryAuthorizationError::HeldOutRootMismatch);
    }

    // Selection occurs before and independently of any real-subject binding.
    let selection_basis = V2CanarySelectionBasis::from_plan(&plan)?;
    let selection_commitment = canary_selection_commitment(selection_basis);
    let row_index = select_canary_row(selection_basis)?;
    let row = plan
        .ordered_rows()
        .get(usize::from(row_index))
        .copied()
        .ok_or(V2CanaryAuthorizationError::RowIndexOutOfRange)?;

    // Adapter identity participates in authorization lineage only after the row
    // has already been selected from schedule-only inputs.
    let adapter_pair = V2RealSubjectAdapterPair::bind_campaign(campaign)?;
    let evidence_class = V2CanaryEvidenceClass::PlumbingOnlyNonConfirmatory;
    let authorization_source_commitment = canary_authorization_source_commitment();
    let commitment = canary_authorization_commitment(
        evidence_class,
        manifest.commitment(),
        adapter_pair.commitment(),
        adapter_pair.adapter_source_commitment(),
        authorization_source_commitment,
        selection_commitment,
        selection_basis,
        row_index,
        row.row_identity(),
        row.family(),
        row.pre(),
        row.action(),
    );

    Ok(V2CanaryAuthorization {
        evidence_class,
        campaign_manifest_commitment: manifest.commitment(),
        adapter_pair_commitment: adapter_pair.commitment(),
        adapter_source_commitment: adapter_pair.adapter_source_commitment(),
        authorization_source_commitment,
        selection_commitment,
        full_schedule_root: selection_basis.full_schedule_root(),
        heldout_ordered_root: selection_basis.heldout_ordered_root(),
        row_index,
        row_identity: row.row_identity(),
        family: row.family(),
        pre: row.pre(),
        action: row.action(),
        commitment,
    })
}

fn canary_selection_commitment(basis: V2CanarySelectionBasis) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_CANARY_SELECTION_REVISION.as_bytes());
    bytes.extend_from_slice(&basis.full_schedule_root);
    bytes.extend_from_slice(&basis.heldout_ordered_root);
    bytes.extend_from_slice(&basis.row_count.to_le_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn select_canary_row(
    basis: V2CanarySelectionBasis,
) -> Result<u16, V2CanaryAuthorizationError> {
    if basis.row_count == 0 {
        return Err(V2CanaryAuthorizationError::InvalidHeldOutRowCount);
    }
    let digest = canary_selection_commitment(basis);
    let mut prefix = [0_u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    let selected = u64::from_le_bytes(prefix) % u64::from(basis.row_count);
    u16::try_from(selected).map_err(|_| V2CanaryAuthorizationError::RowIndexOutOfRange)
}

pub(super) fn canary_authorization_source_commitment() -> [u8; 32] {
    let whole_source = include_str!("v2_canary_authorization.rs");
    let production_source = whole_source
        .split("#[cfg(test)]")
        .next()
        .expect("canary authorization source has production section");
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_CANARY_SOURCE_COMMITMENT_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, production_source.as_bytes());
    *blake3::hash(&bytes).as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn canary_authorization_commitment(
    evidence_class: V2CanaryEvidenceClass,
    campaign_manifest_commitment: [u8; 32],
    adapter_pair_commitment: [u8; 32],
    adapter_source_commitment: [u8; 32],
    authorization_source_commitment: [u8; 32],
    selection_commitment: [u8; 32],
    selection_basis: V2CanarySelectionBasis,
    row_index: u16,
    row_identity: [u8; 32],
    family: V2PublicFamily,
    pre: V2PublicState,
    action: PublicAction,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_CANARY_AUTHORIZATION_REVISION.as_bytes());
    encode_bytes(&mut bytes, V2_CANARY_SELECTION_REVISION.as_bytes());
    encode_bytes(&mut bytes, evidence_class.stable_id().as_bytes());
    bytes.extend_from_slice(&campaign_manifest_commitment);
    bytes.extend_from_slice(&adapter_pair_commitment);
    bytes.extend_from_slice(&adapter_source_commitment);
    bytes.extend_from_slice(&authorization_source_commitment);
    bytes.extend_from_slice(&selection_commitment);
    bytes.extend_from_slice(&selection_basis.full_schedule_root);
    bytes.extend_from_slice(&selection_basis.heldout_ordered_root);
    bytes.extend_from_slice(&selection_basis.row_count.to_le_bytes());
    bytes.extend_from_slice(&row_index.to_le_bytes());
    bytes.extend_from_slice(&row_identity);
    encode_family(&mut bytes, family);
    encode_state(&mut bytes, pre);
    encode_action(&mut bytes, action);
    *blake3::hash(&bytes).as_bytes()
}

fn encode_family(bytes: &mut Vec<u8>, family: V2PublicFamily) {
    bytes.push(family.tag());
}

fn encode_state(bytes: &mut Vec<u8>, state: V2PublicState) {
    for field in state.fields() {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
}

fn encode_action(bytes: &mut Vec<u8>, action: PublicAction) {
    let index = action_index(action).expect("canonical HeldOut action must belong to V2 schema");
    bytes.extend_from_slice(&(index as u64).to_le_bytes());
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_selection_is_deterministic_and_in_range() {
        let plan = materialize_heldout_plan().unwrap();
        let basis = V2CanarySelectionBasis::from_plan(&plan).unwrap();
        let first = select_canary_row(basis).unwrap();
        let second = select_canary_row(basis).unwrap();
        assert_eq!(first, second);
        assert!(usize::from(first) < plan.ordered_rows().len());
        assert_ne!(canary_selection_commitment(basis), [0_u8; 32]);
    }

    #[test]
    fn selection_commitment_binds_schedule_identity_without_probabilistic_modulo_assumption() {
        let plan = materialize_heldout_plan().unwrap();
        let canonical = V2CanarySelectionBasis::from_plan(&plan).unwrap();
        let canonical_commitment = canary_selection_commitment(canonical);

        let mut changed_full = canonical;
        changed_full.full_schedule_root[0] ^= 0x40;
        assert_ne!(
            canonical_commitment,
            canary_selection_commitment(changed_full)
        );

        let mut changed_heldout = canonical;
        changed_heldout.heldout_ordered_root[31] ^= 0x01;
        assert_ne!(
            canonical_commitment,
            canary_selection_commitment(changed_heldout)
        );
    }

    #[test]
    fn selection_basis_has_only_schedule_inputs() {
        let source = include_str!("v2_canary_authorization.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let start = source.find("struct V2CanarySelectionBasis").unwrap();
        let tail = &source[start..];
        let end = tail.find("}\n\nimpl V2CanarySelectionBasis").unwrap();
        let body = &tail[..end];
        assert!(body.contains("full_schedule_root"));
        assert!(body.contains("heldout_ordered_root"));
        assert!(body.contains("row_count"));
        for forbidden in [
            "subject",
            "comparator",
            "prediction",
            "outcome",
            "score",
            "manifest",
            "rng",
            "timestamp",
        ] {
            assert!(!body.contains(forbidden), "selection basis leaked: {forbidden}");
        }
    }

    #[test]
    fn selector_body_depends_only_on_selection_basis_and_its_commitment() {
        let source = include_str!("v2_canary_authorization.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let start = source.find("fn select_canary_row(").unwrap();
        let tail = &source[start..];
        let end = tail
            .find("\n}\n\npub(super) fn canary_authorization_source_commitment")
            .unwrap();
        let body = &tail[..end];
        assert!(body.contains("canary_selection_commitment(basis)"));
        for forbidden in [
            "campaign",
            "manifest",
            "subject",
            "comparator",
            "prediction",
            "post",
            "outcome",
            "score",
            "rand",
        ] {
            assert!(!body.contains(forbidden), "selector leaked: {forbidden}");
        }
    }

    #[test]
    fn authorization_source_has_no_real_execution_or_promotion_path() {
        let source = include_str!("v2_canary_authorization.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "predict_ticket(",
            ".reveal(",
            ".score(",
            "score_consequence(",
            "V2HeldOutPairScore",
            "V2ShadowCampaignReport",
            "ScientificDisposition",
            "for row",
            "0..128",
            ".post()",
        ] {
            assert!(
                !source.contains(forbidden),
                "canary authorization must not execute/promote evidence: {forbidden}"
            );
        }
        assert!(source.contains("PlumbingOnlyNonConfirmatory"));
    }

    #[test]
    fn exact_canary_authorization_source_is_cryptographically_bound() {
        let canonical = canary_authorization_source_commitment();
        let whole_source = include_str!("v2_canary_authorization.rs");
        let production_source = whole_source
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let mut changed = production_source.as_bytes().to_vec();
        changed.push(b'\n');
        let mut bytes = Vec::new();
        encode_bytes(
            &mut bytes,
            V2_CANARY_SOURCE_COMMITMENT_REVISION.as_bytes(),
        );
        encode_bytes(&mut bytes, &changed);
        let changed_commitment = *blake3::hash(&bytes).as_bytes();
        assert_ne!(canonical, [0_u8; 32]);
        assert_ne!(canonical, changed_commitment);
    }
}
