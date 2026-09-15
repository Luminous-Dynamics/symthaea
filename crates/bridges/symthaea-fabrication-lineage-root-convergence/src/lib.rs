// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Borrowed, non-forgeable projections from lineage-native finalized authority into the shared
//! lineage-bound handoff predecessor/current-head contracts.

#![deny(unsafe_code)]

use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_fabrication_kernel::upgrade_handoff::UpgradeEndpoint;
use symthaea_fabrication_lineage_bound_finalized_head::CurrentLineageBoundFinalizedUpgradeHeadV1;
use symthaea_fabrication_lineage_bound_global_upgrade_head::LineageBoundFinalizedPredecessorRootV1;
use symthaea_fabrication_lineage_bound_upgrade_handoff::{
    LineageBoundCurrentHeadViewV1, LineageBoundPredecessorAuthorityKindV1,
    LineageBoundPredecessorRootViewV1,
};
use symthaea_trust_kernel::{ClockGovernanceEvaluationEnvelopeIdV1, OperationalClockBasisIdV1};

/// Borrowed view over the exact #3328 predecessor root. The wrapper cannot be constructed without
/// already possessing the opaque lineage-native root.
#[derive(Debug, Clone, Copy)]
pub struct LineageNativePredecessorRootViewV1<'a> {
    root: &'a LineageBoundFinalizedPredecessorRootV1,
}

impl<'a> LineageNativePredecessorRootViewV1<'a> {
    pub fn new(root: &'a LineageBoundFinalizedPredecessorRootV1) -> Self { Self { root } }
    pub fn root(&self) -> &'a LineageBoundFinalizedPredecessorRootV1 { self.root }
}

impl LineageBoundPredecessorRootViewV1 for LineageNativePredecessorRootViewV1<'_> {
    fn authority_kind(&self) -> LineageBoundPredecessorAuthorityKindV1 {
        LineageBoundPredecessorAuthorityKindV1::LineageNativeV1
    }
    fn id_digest(&self) -> Sha256Digest { self.root.id().as_digest() }
    fn current_head_digest(&self) -> Sha256Digest { self.root.current_head_id().as_digest() }
    fn endpoint(&self) -> &UpgradeEndpoint { self.root.endpoint() }
    fn endpoint_digest(&self) -> Sha256Digest { self.root.endpoint_digest() }
    fn rollback_target_digest(&self) -> Sha256Digest { self.root.rollback_target_digest() }
    fn finalization_sequence(&self) -> u64 { self.root.finalization_sequence() }
    fn evidence_checkpoint_digest(&self) -> Sha256Digest { self.root.evidence_checkpoint_digest() }
    fn transparency_log_digest(&self) -> Sha256Digest { self.root.transparency_log_digest() }
    fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.root.clock_envelope_id() }
    fn operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.root.operational_basis_id() }
}

/// Borrowed view over the exact #3314 handoff-scoped finalized head paired with #3328.
#[derive(Debug, Clone, Copy)]
pub struct LineageNativeCurrentHeadViewV1<'a> {
    head: &'a CurrentLineageBoundFinalizedUpgradeHeadV1,
}

impl<'a> LineageNativeCurrentHeadViewV1<'a> {
    pub fn new(head: &'a CurrentLineageBoundFinalizedUpgradeHeadV1) -> Self { Self { head } }
    pub fn head(&self) -> &'a CurrentLineageBoundFinalizedUpgradeHeadV1 { self.head }
}

impl LineageBoundCurrentHeadViewV1 for LineageNativeCurrentHeadViewV1<'_> {
    fn authority_kind(&self) -> LineageBoundPredecessorAuthorityKindV1 {
        LineageBoundPredecessorAuthorityKindV1::LineageNativeV1
    }
    fn id_digest(&self) -> Sha256Digest { self.head.id().as_digest() }
    fn governance_view_digest(&self) -> Sha256Digest { self.head.governance_view_id().as_digest() }
    fn checkpoint_digest(&self) -> Sha256Digest { self.head.current_checkpoint_digest() }
    fn transparency_log_digest(&self) -> Sha256Digest { self.head.current_transparency_log_digest() }
    fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.head.current_clock_envelope_id()
    }
    fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.head.current_operational_basis_id()
    }
}

/// Construct the two borrowed views that can be passed directly into the shared
/// `build_lineage_bound_upgrade_handoff_plan_v1` / `prepare_lineage_bound_upgrade_handoff_v1`
/// boundary. No digest, endpoint, sequence, checkpoint or clock fact is caller supplied.
pub fn lineage_native_handoff_views_v1<'a>(
    root: &'a LineageBoundFinalizedPredecessorRootV1,
    head: &'a CurrentLineageBoundFinalizedUpgradeHeadV1,
) -> (
    LineageNativePredecessorRootViewV1<'a>,
    LineageNativeCurrentHeadViewV1<'a>,
) {
    (
        LineageNativePredecessorRootViewV1::new(root),
        LineageNativeCurrentHeadViewV1::new(head),
    )
}
