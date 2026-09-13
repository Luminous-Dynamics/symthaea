// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Partition-neutral pre-outcome prediction tickets for EUREKA-002 V2.
//!
//! A prospective ticket is prediction authority only. It contains no post-state,
//! no evaluator row lookup, and no reveal/scoring method. Scientific role is
//! explicit and commitment-bound so identical row tuples in different domains
//! cannot collide.

#![allow(dead_code)]

use super::hidden_world::PublicAction;
use super::v2_public_schema::{
    V2PublicFamily, V2PublicSchemaError, V2PublicState, action_index,
    public_schema_commitment,
};

pub(super) const V2_PROSPECTIVE_TICKET_REVISION: &str =
    "EUREKA.002.V2.PROSPECTIVE_TICKET.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum V2ProspectiveTicketDomain {
    HeldOutEvaluation,
    DevelopmentPlumbingCanary,
}

impl V2ProspectiveTicketDomain {
    const fn tag(self) -> u8 {
        match self {
            Self::HeldOutEvaluation => 1,
            Self::DevelopmentPlumbingCanary => 2,
        }
    }

    pub(super) const fn fep_modality(self) -> &'static str {
        match self {
            Self::HeldOutEvaluation => "eureka-v2-heldout",
            Self::DevelopmentPlumbingCanary => "eureka-v2-development-canary",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ProspectiveTicketError {
    ZeroCampaignManifestCommitment,
    UnsupportedAction,
}

impl From<V2PublicSchemaError> for V2ProspectiveTicketError {
    fn from(_: V2PublicSchemaError) -> Self {
        Self::UnsupportedAction
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2ProspectiveTicket {
    domain: V2ProspectiveTicketDomain,
    campaign_manifest_commitment: [u8; 32],
    row_index: u16,
    row_identity: [u8; 32],
    family: V2PublicFamily,
    pre: V2PublicState,
    action: PublicAction,
    commitment: [u8; 32],
}

impl V2ProspectiveTicket {
    pub(super) fn new(
        domain: V2ProspectiveTicketDomain,
        campaign_manifest_commitment: [u8; 32],
        row_index: u16,
        row_identity: [u8; 32],
        family: V2PublicFamily,
        pre: V2PublicState,
        action: PublicAction,
    ) -> Result<Self, V2ProspectiveTicketError> {
        if campaign_manifest_commitment == [0_u8; 32] {
            return Err(V2ProspectiveTicketError::ZeroCampaignManifestCommitment);
        }
        let action_index = action_index(action)?;
        let commitment = prospective_ticket_commitment(
            domain,
            campaign_manifest_commitment,
            row_index,
            row_identity,
            family,
            pre,
            action_index,
        );
        Ok(Self {
            domain,
            campaign_manifest_commitment,
            row_index,
            row_identity,
            family,
            pre,
            action,
            commitment,
        })
    }

    pub(super) const fn domain(self) -> V2ProspectiveTicketDomain {
        self.domain
    }

    pub(super) const fn campaign_manifest_commitment(self) -> [u8; 32] {
        self.campaign_manifest_commitment
    }

    pub(super) const fn row_index(self) -> u16 {
        self.row_index
    }

    pub(super) const fn row_identity(self) -> [u8; 32] {
        self.row_identity
    }

    pub(super) const fn family(self) -> V2PublicFamily {
        self.family
    }

    pub(super) const fn pre(self) -> V2PublicState {
        self.pre
    }

    pub(super) const fn action(self) -> PublicAction {
        self.action
    }

    pub(super) const fn commitment(self) -> [u8; 32] {
        self.commitment
    }
}

#[allow(clippy::too_many_arguments)]
fn prospective_ticket_commitment(
    domain: V2ProspectiveTicketDomain,
    campaign_manifest_commitment: [u8; 32],
    row_index: u16,
    row_identity: [u8; 32],
    family: V2PublicFamily,
    pre: V2PublicState,
    action_index: usize,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_PROSPECTIVE_TICKET_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.push(domain.tag());
    bytes.extend_from_slice(&campaign_manifest_commitment);
    bytes.extend_from_slice(&row_index.to_le_bytes());
    bytes.extend_from_slice(&row_identity);
    bytes.push(family.tag());
    for field in pre.fields() {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
    bytes.extend_from_slice(&(action_index as u64).to_le_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::v2_development_order::materialize_development_plan;

    const MANIFEST: [u8; 32] = [0x71; 32];

    fn sample_tuple() -> (u16, [u8; 32], V2PublicFamily, V2PublicState, PublicAction) {
        let plan = materialize_development_plan().unwrap();
        let row = plan.ordered_rows()[0];
        (0, row.row_identity(), row.family(), row.pre(), row.action())
    }

    #[test]
    fn domains_are_commitment_separated_for_identical_row_tuple() {
        let (index, identity, family, pre, action) = sample_tuple();
        let heldout = V2ProspectiveTicket::new(
            V2ProspectiveTicketDomain::HeldOutEvaluation,
            MANIFEST,
            index,
            identity,
            family,
            pre,
            action,
        )
        .unwrap();
        let canary = V2ProspectiveTicket::new(
            V2ProspectiveTicketDomain::DevelopmentPlumbingCanary,
            MANIFEST,
            index,
            identity,
            family,
            pre,
            action,
        )
        .unwrap();
        assert_ne!(heldout.commitment(), canary.commitment());
        assert_ne!(heldout.domain(), canary.domain());
        assert_ne!(heldout.domain().fep_modality(), canary.domain().fep_modality());
    }

    #[test]
    fn ticket_is_pre_outcome_only_and_has_no_reveal_authority() {
        let source = include_str!("v2_prospective_ticket.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let start = source.find("struct V2ProspectiveTicket").unwrap();
        let tail = &source[start..];
        let end = tail.find("}\n\nimpl V2ProspectiveTicket").unwrap();
        let body = &tail[..end];
        assert!(!body.contains("post"));
        for forbidden in [
            "V2HeldOutPlan",
            "materialize_heldout_plan",
            "reveal(",
            "score_consequence(",
        ] {
            assert!(
                !source.contains(forbidden),
                "prospective ticket must not own reveal authority: {forbidden}"
            );
        }
    }

    #[test]
    fn zero_manifest_and_noncanonical_action_fail_closed() {
        let (index, identity, family, pre, action) = sample_tuple();
        assert_eq!(
            V2ProspectiveTicket::new(
                V2ProspectiveTicketDomain::HeldOutEvaluation,
                [0_u8; 32],
                index,
                identity,
                family,
                pre,
                action,
            ),
            Err(V2ProspectiveTicketError::ZeroCampaignManifestCommitment)
        );
        assert_eq!(
            V2ProspectiveTicket::new(
                V2ProspectiveTicketDomain::HeldOutEvaluation,
                MANIFEST,
                index,
                identity,
                family,
                pre,
                PublicAction::Pulse { slot: 3 },
            ),
            Err(V2ProspectiveTicketError::UnsupportedAction)
        );
    }
}
