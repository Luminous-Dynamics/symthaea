// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Non-promotable one-row plumbing-canary authorization for EUREKA-002 V2.
//!
//! The canary is deliberately selected from the already-consumed Development
//! stream, never from HeldOut. Selection depends only on the frozen ordered
//! Development pre/action surface; Development post-states and real-subject
//! behavior cannot influence which row is chosen. This layer performs no
//! prediction, reveal, scoring, learning, or scientific promotion.

#![allow(dead_code)]

use super::hidden_world::PublicAction;
use super::v2_corpus_schedule::V2ScheduleMaterializationError;
use super::v2_development_order::{V2DevelopmentPlan, materialize_development_plan};
use super::v2_preheldout_custody::V2PreHeldOutCampaignCapability;
use super::v2_public_schema::{
    V2PublicFamily, V2PublicState, action_index, public_schema_commitment,
};
use super::v2_real_subject_adapters::{
    V2RealSubjectAdapterError, V2RealSubjectAdapterPair,
};

pub(super) const V2_CANARY_SELECTION_SURFACE_REVISION: &str =
    "EUREKA.002.V2.DEVELOPMENT_CANARY_PREACTION_SURFACE.v1";
pub(super) const V2_CANARY_SELECTION_REVISION: &str =
    "EUREKA.002.V2.DEVELOPMENT_CANARY_SELECTION.v2";
pub(super) const V2_CANARY_AUTHORIZATION_REVISION: &str =
    "EUREKA.002.V2.DEVELOPMENT_CANARY_AUTHORIZATION.v2";
pub(super) const V2_CANARY_SOURCE_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.DEVELOPMENT_CANARY_AUTHORIZATION_SOURCE.v2";

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2CanarySelectionBasis {
    development_preaction_root: [u8; 32],
    row_count: u16,
}

impl V2CanarySelectionBasis {
    fn from_plan(plan: &V2DevelopmentPlan) -> Result<Self, V2CanaryAuthorizationError> {
        let row_count = u16::try_from(plan.ordered_rows().len())
            .map_err(|_| V2CanaryAuthorizationError::InvalidDevelopmentRowCount)?;
        if row_count == 0 {
            return Err(V2CanaryAuthorizationError::InvalidDevelopmentRowCount);
        }
        Ok(Self {
            development_preaction_root: development_preaction_root(plan),
            row_count,
        })
    }

    pub(super) const fn development_preaction_root(self) -> [u8; 32] {
        self.development_preaction_root
    }

    pub(super) const fn row_count(self) -> u16 {
        self.row_count
    }
}

#[derive(Debug)]
pub(super) enum V2CanaryAuthorizationError {
    Materialization(V2ScheduleMaterializationError),
    Adapter(V2RealSubjectAdapterError),
    InvalidDevelopmentRowCount,
    DevelopmentPlanMismatch,
    ScheduleRootMismatch,
    DevelopmentCorpusMismatch,
    DevelopmentOrderMismatch,
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

#[derive(Debug)]
pub(super) struct V2CanaryAuthorization {
    evidence_class: V2CanaryEvidenceClass,
    campaign_manifest_commitment: [u8; 32],
    adapter_pair_commitment: [u8; 32],
    adapter_source_commitment: [u8; 32],
    authorization_source_commitment: [u8; 32],
    selection_commitment: [u8; 32],
    development_preaction_root: [u8; 32],
    development_plan_commitment: [u8; 32],
    full_schedule_root: [u8; 32],
    development_corpus_commitment: [u8; 32],
    development_order_root: [u8; 32],
    row_index: u16,
    row_identity: [u8; 32],
    family: V2PublicFamily,
    pre: V2PublicState,
    action: PublicAction,
    commitment: [u8; 32],
}

impl V2CanaryAuthorization {
    pub(super) const fn evidence_class(&self) -> V2CanaryEvidenceClass { self.evidence_class }
    pub(super) const fn campaign_manifest_commitment(&self) -> [u8; 32] { self.campaign_manifest_commitment }
    pub(super) const fn adapter_pair_commitment(&self) -> [u8; 32] { self.adapter_pair_commitment }
    pub(super) const fn adapter_source_commitment(&self) -> [u8; 32] { self.adapter_source_commitment }
    pub(super) const fn selection_commitment(&self) -> [u8; 32] { self.selection_commitment }
    pub(super) const fn development_preaction_root(&self) -> [u8; 32] { self.development_preaction_root }
    pub(super) const fn development_plan_commitment(&self) -> [u8; 32] { self.development_plan_commitment }
    pub(super) const fn full_schedule_root(&self) -> [u8; 32] { self.full_schedule_root }
    pub(super) const fn development_corpus_commitment(&self) -> [u8; 32] { self.development_corpus_commitment }
    pub(super) const fn development_order_root(&self) -> [u8; 32] { self.development_order_root }
    pub(super) const fn row_index(&self) -> u16 { self.row_index }
    pub(super) const fn row_identity(&self) -> [u8; 32] { self.row_identity }
    pub(super) const fn family(&self) -> V2PublicFamily { self.family }
    pub(super) const fn pre(&self) -> V2PublicState { self.pre }
    pub(super) const fn action(&self) -> PublicAction { self.action }
    pub(super) const fn commitment(&self) -> [u8; 32] { self.commitment }
}

pub(super) fn freeze_canary_authorization(
    campaign: &V2PreHeldOutCampaignCapability,
) -> Result<V2CanaryAuthorization, V2CanaryAuthorizationError> {
    let plan = materialize_development_plan()?;
    let receipt = campaign.development_receipt();
    let manifest = campaign.manifest();

    if plan.commitment() != receipt.development_plan_commitment() {
        return Err(V2CanaryAuthorizationError::DevelopmentPlanMismatch);
    }
    if plan.full_schedule_root() != receipt.full_schedule_root()
        || plan.full_schedule_root() != manifest.full_schedule_root()
    {
        return Err(V2CanaryAuthorizationError::ScheduleRootMismatch);
    }
    if plan.development_corpus().commitment() != receipt.development_corpus_commitment() {
        return Err(V2CanaryAuthorizationError::DevelopmentCorpusMismatch);
    }
    if plan.development_order_root() != receipt.development_order_root() {
        return Err(V2CanaryAuthorizationError::DevelopmentOrderMismatch);
    }

    let selection_basis = V2CanarySelectionBasis::from_plan(&plan)?;
    let selection_commitment = canary_selection_commitment(selection_basis);
    let row_index = select_canary_row(selection_basis)?;
    let row = plan
        .ordered_rows()
        .get(usize::from(row_index))
        .copied()
        .ok_or(V2CanaryAuthorizationError::RowIndexOutOfRange)?;

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
        plan.commitment(),
        plan.full_schedule_root(),
        plan.development_corpus().commitment(),
        plan.development_order_root(),
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
        development_preaction_root: selection_basis.development_preaction_root(),
        development_plan_commitment: plan.commitment(),
        full_schedule_root: plan.full_schedule_root(),
        development_corpus_commitment: plan.development_corpus().commitment(),
        development_order_root: plan.development_order_root(),
        row_index,
        row_identity: row.row_identity(),
        family: row.family(),
        pre: row.pre(),
        action: row.action(),
        commitment,
    })
}

fn development_preaction_root(plan: &V2DevelopmentPlan) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_CANARY_SELECTION_SURFACE_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&(plan.ordered_rows().len() as u64).to_le_bytes());
    for row in plan.ordered_rows().iter().copied() {
        encode_family(&mut bytes, row.family());
        encode_state(&mut bytes, row.pre());
        encode_action(&mut bytes, row.action());
    }
    *blake3::hash(&bytes).as_bytes()
}

fn canary_selection_commitment(basis: V2CanarySelectionBasis) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_CANARY_SELECTION_REVISION.as_bytes());
    bytes.extend_from_slice(&basis.development_preaction_root);
    bytes.extend_from_slice(&basis.row_count.to_le_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn select_canary_row(basis: V2CanarySelectionBasis) -> Result<u16, V2CanaryAuthorizationError> {
    if basis.row_count == 0 {
        return Err(V2CanaryAuthorizationError::InvalidDevelopmentRowCount);
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
    encode_bytes(&mut bytes, V2_CANARY_SOURCE_COMMITMENT_REVISION.as_bytes());
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
    development_plan_commitment: [u8; 32],
    full_schedule_root: [u8; 32],
    development_corpus_commitment: [u8; 32],
    development_order_root: [u8; 32],
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
    bytes.extend_from_slice(&selection_basis.development_preaction_root);
    bytes.extend_from_slice(&selection_basis.row_count.to_le_bytes());
    bytes.extend_from_slice(&development_plan_commitment);
    bytes.extend_from_slice(&full_schedule_root);
    bytes.extend_from_slice(&development_corpus_commitment);
    bytes.extend_from_slice(&development_order_root);
    bytes.extend_from_slice(&row_index.to_le_bytes());
    bytes.extend_from_slice(&row_identity);
    encode_family(&mut bytes, family);
    encode_state(&mut bytes, pre);
    encode_action(&mut bytes, action);
    *blake3::hash(&bytes).as_bytes()
}

fn encode_family(bytes: &mut Vec<u8>, family: V2PublicFamily) { bytes.push(family.tag()); }
fn encode_state(bytes: &mut Vec<u8>, state: V2PublicState) {
    for field in state.fields() { bytes.extend_from_slice(&field.to_le_bytes()); }
}
fn encode_action(bytes: &mut Vec<u8>, action: PublicAction) {
    let index = action_index(action).expect("canonical Development action must belong to V2 schema");
    bytes.extend_from_slice(&(index as u64).to_le_bytes());
}
fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::v2_corpus_schedule::V2SchedulePartition;

    #[test]
    fn canonical_selection_is_deterministic_development_only_and_in_range() {
        let plan = materialize_development_plan().unwrap();
        let basis = V2CanarySelectionBasis::from_plan(&plan).unwrap();
        let first = select_canary_row(basis).unwrap();
        let second = select_canary_row(basis).unwrap();
        assert_eq!(first, second);
        assert!(usize::from(first) < plan.ordered_rows().len());
        assert_eq!(plan.ordered_rows().len(), 512);
        assert_eq!(plan.ordered_rows()[usize::from(first)].partition(), V2SchedulePartition::Development);
        assert_ne!(canary_selection_commitment(basis), [0_u8; 32]);
    }

    #[test]
    fn selection_commitment_binds_preaction_surface_without_outcome_inputs() {
        let plan = materialize_development_plan().unwrap();
        let canonical = V2CanarySelectionBasis::from_plan(&plan).unwrap();
        let canonical_commitment = canary_selection_commitment(canonical);
        let mut changed_surface = canonical;
        changed_surface.development_preaction_root[17] ^= 0x08;
        assert_ne!(canonical_commitment, canary_selection_commitment(changed_surface));
    }

    #[test]
    fn selection_basis_has_only_preaction_surface_and_count() {
        let source = include_str!("v2_canary_authorization.rs").split("#[cfg(test)]").next().unwrap();
        let start = source.find("struct V2CanarySelectionBasis").unwrap();
        let tail = &source[start..];
        let end = tail.find("}\n\nimpl V2CanarySelectionBasis").unwrap();
        let body = &tail[..end];
        assert!(body.contains("development_preaction_root"));
        assert!(body.contains("row_count"));
        for forbidden in ["full_schedule_root", "corpus", "order_root", "subject", "comparator", "prediction", "outcome", "score", "manifest", "rng", "timestamp"] {
            assert!(!body.contains(forbidden), "selection basis leaked: {forbidden}");
        }
    }

    #[test]
    fn preaction_root_has_no_post_state_or_row_identity_input() {
        let source = include_str!("v2_canary_authorization.rs").split("#[cfg(test)]").next().unwrap();
        let start = source.find("fn development_preaction_root(").unwrap();
        let tail = &source[start..];
        let end = tail.find("\n}\n\nfn canary_selection_commitment").unwrap();
        let body = &tail[..end];
        assert!(body.contains("row.family()"));
        assert!(body.contains("row.pre()"));
        assert!(body.contains("row.action()"));
        assert!(!body.contains("row.post()"));
        assert!(!body.contains("row.row_identity()"));
    }

    #[test]
    fn production_authorization_has_zero_heldout_materialization_or_execution_reachability() {
        let source = include_str!("v2_canary_authorization.rs").split("#[cfg(test)]").next().unwrap();
        for forbidden in ["materialize_heldout_plan", "V2HeldOutPlan", ".heldout_plan(", "predict_ticket(", ".reveal(", ".score(", "score_consequence(", ".post(", "V2HeldOutPairScore", "V2ShadowCampaignReport", "ScientificDisposition", "0..128"] {
            assert!(!source.contains(forbidden), "Development canary authorization must not reach HeldOut/execution/promotion authority: {forbidden}");
        }
        assert!(source.contains("PlumbingOnlyNonConfirmatory"));
        assert!(source.contains("materialize_development_plan"));
    }

    #[test]
    fn exact_canary_authorization_source_is_cryptographically_bound() {
        let canonical = canary_authorization_source_commitment();
        let whole_source = include_str!("v2_canary_authorization.rs");
        let production_source = whole_source.split("#[cfg(test)]").next().unwrap();
        let mut changed = production_source.as_bytes().to_vec();
        changed.push(b'\n');
        let mut bytes = Vec::new();
        encode_bytes(&mut bytes, V2_CANARY_SOURCE_COMMITMENT_REVISION.as_bytes());
        encode_bytes(&mut bytes, &changed);
        let changed_commitment = *blake3::hash(&bytes).as_bytes();
        assert_ne!(canonical, [0_u8; 32]);
        assert_ne!(canonical, changed_commitment);
    }
}
