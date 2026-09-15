// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mint portable predecessor certificates only from one exact opaque finalized governance view.

#![deny(unsafe_code)]

use symthaea_fabrication_lineage_bound_finalized_head::CurrentLineageBoundFinalizedUpgradeHeadV1;
use symthaea_fabrication_lineage_bound_global_upgrade_head::LineageBoundFinalizedPredecessorRootV1;
use symthaea_fabrication_lineage_predecessor_certificate::{
    LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_SCHEMA,
    LineageFinalizedPredecessorCertificateV1,
    LineagePredecessorCertificateError,
    digest_lineage_finalized_predecessor_certificate_v1,
};
use symthaea_fabrication_witness_registry_containment_bound::ContainmentCurrentWitnessRegistryHeadV1;
use symthaea_fabrication_witness_registry_head::QuorumObservedWitnessRegistryHeadV1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineagePredecessorCertificateMintError {
    CurrentHeadMismatch,
    GovernanceViewMismatch,
    RegistryHeadMismatch,
    Certificate(LineagePredecessorCertificateError),
}

pub fn build_lineage_finalized_predecessor_certificate_v1(
    root: &LineageBoundFinalizedPredecessorRootV1,
    current_head: &CurrentLineageBoundFinalizedUpgradeHeadV1,
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    registry_head: &QuorumObservedWitnessRegistryHeadV1,
) -> Result<LineageFinalizedPredecessorCertificateV1, LineagePredecessorCertificateMintError> {
    if root.current_head_id() != current_head.id()
        || root.evidence_checkpoint_digest() != current_head.current_checkpoint_digest()
        || root.transparency_log_digest() != current_head.current_transparency_log_digest()
        || root.clock_envelope_id() != current_head.current_clock_envelope_id()
        || root.operational_basis_id() != current_head.current_operational_basis_id()
    {
        return Err(LineagePredecessorCertificateMintError::CurrentHeadMismatch);
    }
    if current_head.governance_view_id() != governance_view.id()
        || root.evidence_checkpoint_digest() != governance_view.checkpoint_digest()
        || root.transparency_log_digest() != governance_view.transparency_log_digest()
        || root.clock_envelope_id() != governance_view.observation_clock_envelope_id()
        || root.operational_basis_id() != governance_view.observation_operational_basis_id()
    {
        return Err(LineagePredecessorCertificateMintError::GovernanceViewMismatch);
    }
    if governance_view.registry_head_id() != registry_head.id()
        || governance_view.registry_digest() != registry_head.registry_digest()
        || governance_view.registry_sequence() != registry_head.sequence()
    {
        return Err(LineagePredecessorCertificateMintError::RegistryHeadMismatch);
    }

    let certificate = LineageFinalizedPredecessorCertificateV1 {
        schema_version: LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_SCHEMA.into(),
        predecessor_root_id: root.id().to_hex(),
        global_head_id: root.global_head_id().to_hex(),
        current_head_id: root.current_head_id().to_hex(),
        governance_view_id: governance_view.id().to_hex(),
        registry_head_id: registry_head.id().to_hex(),
        finalized_upgrade_id: root.finalized_upgrade_id().to_hex(),
        finalization_record_digest: root.record_digest(),
        prior_predecessor_root_digest: root.prior_predecessor_root_digest(),
        finalization_sequence: root.finalization_sequence(),
        endpoint: root.endpoint().clone(),
        endpoint_digest: root.endpoint_digest(),
        rollback_target_digest: root.rollback_target_digest(),
        evidence_checkpoint_digest: root.evidence_checkpoint_digest(),
        transparency_log_digest: root.transparency_log_digest(),
        registry_digest: registry_head.registry_digest(),
        registry_sequence: registry_head.sequence(),
        trust_snapshot_digest: registry_head.trust_snapshot_digest(),
        containment_state_digest: governance_view.containment_state_digest(),
        compromise_tracker_digest: governance_view.compromise_tracker_digest(),
        containment_generation: governance_view.containment_generation(),
        clock_envelope_id: root.clock_envelope_id().to_hex(),
        operational_basis_id: root.operational_basis_id().to_hex(),
    };
    digest_lineage_finalized_predecessor_certificate_v1(&certificate)
        .map_err(LineagePredecessorCertificateMintError::Certificate)?;
    Ok(certificate)
}
