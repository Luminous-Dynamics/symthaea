// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Guarded composition of independently verified Xenia witness chronology with
//! the local SQLite witness-frontier publication barrier.
//!
//! This crate intentionally adds no new cryptography, time source, persistence
//! semantics, anchoring transport, publication effect, or execution authority.
//! It composes three existing opaque boundaries:
//!
//! - `XeniaExternalWitnessFrontierV1`: authenticated/current Xenia chronology;
//! - `SqliteWitnessFrontierPublicationGuard`: audited local history under a live
//!   SQLite writer reservation;
//! - `GuardedWitnessFrontierDecisionV1`: #452 ancestry classification.
//!
//! The important property is provenance retention. A publication/re-anchor
//! permit exposed by this crate remains paired with the exact Xenia evidence
//! that justified the guarded classification. Source-specific evidence is not
//! discarded merely because the recovery classifier itself is transport-neutral.

#![deny(unsafe_code)]

use symthaea_qualification_witness_frontier::{
    WitnessFrontierPublicationDispositionV1, WitnessFrontierRecoveryRelationV1,
};
use symthaea_qualification_witness_frontier_sqlite::{
    GuardedAnchorPermitV1, GuardedPublicationPermitV1, GuardedWitnessFrontierDecisionV1,
    SqliteWitnessFrontierGuardError, SqliteWitnessFrontierPublicationGuard,
};
use symthaea_xenia_witness_frontier_adapter::XeniaExternalWitnessFrontierV1;

/// One Xenia-backed recovery decision while the local SQLite writer barrier is
/// still held.
///
/// The decision owns neither the guard nor the Xenia evidence; it borrows both.
/// Therefore it cannot outlive either the point-in-time local history or the
/// exact external proof that justified the classification.
#[derive(Debug)]
pub struct GuardedXeniaWitnessFrontierDecisionV1<'g, 'x> {
    decision: GuardedWitnessFrontierDecisionV1<'g>,
    xenia: &'x XeniaExternalWitnessFrontierV1,
}

impl<'g, 'x> GuardedXeniaWitnessFrontierDecisionV1<'g, 'x> {
    /// Exact #452 recovery relation produced while the local writer barrier was
    /// held.
    pub fn relation(&self) -> WitnessFrontierRecoveryRelationV1 {
        self.decision.relation()
    }

    /// Closed-world publication disposition from #452.
    pub fn publication_disposition(&self) -> WitnessFrontierPublicationDispositionV1 {
        self.decision.publication_disposition()
    }

    /// Exact source-specific Xenia evidence that justified this decision.
    pub fn xenia_evidence(&self) -> &'x XeniaExternalWitnessFrontierV1 {
        self.xenia
    }

    /// Produce a provenance-retaining publication permit only when #456 says
    /// publication is allowed under the still-live SQLite guard.
    pub fn publication_permit<'a>(&'a self) -> Option<GuardedXeniaPublicationPermitV1<'a, 'x>> {
        self.decision
            .publication_permit()
            .map(|permit| GuardedXeniaPublicationPermitV1 {
                permit,
                xenia: self.xenia,
            })
    }

    /// Produce a provenance-retaining anchor permit only when #456 says the
    /// guarded local state requires external anchoring.
    pub fn anchor_permit<'a>(&'a self) -> Option<GuardedXeniaAnchorPermitV1<'a, 'x>> {
        self.decision
            .anchor_permit()
            .map(|permit| GuardedXeniaAnchorPermitV1 {
                permit,
                xenia: self.xenia,
            })
    }
}

/// Xenia-specific publication permit. The generic #456 permit remains private
/// so callers cannot accidentally detach the guarded local decision from the
/// external evidence that justified it.
#[derive(Debug)]
pub struct GuardedXeniaPublicationPermitV1<'a, 'x> {
    permit: GuardedPublicationPermitV1<'a>,
    xenia: &'x XeniaExternalWitnessFrontierV1,
}

impl GuardedXeniaPublicationPermitV1<'_, '_> {
    pub fn witness_id(&self) -> [u8; 16] {
        self.permit.witness_id()
    }

    pub fn frontier(&self) -> symthaea_qualification_witness_frontier::WitnessFrontierPointV1 {
        self.permit.frontier()
    }

    pub fn xenia_evidence(&self) -> &XeniaExternalWitnessFrontierV1 {
        self.xenia
    }
}

/// Xenia-specific re-anchor permit. It preserves both the local guarded frontier
/// and the exact external Xenia chronology that was found to be stale-but-ancestral
/// or absent.
#[derive(Debug)]
pub struct GuardedXeniaAnchorPermitV1<'a, 'x> {
    permit: GuardedAnchorPermitV1<'a>,
    xenia: &'x XeniaExternalWitnessFrontierV1,
}

impl GuardedXeniaAnchorPermitV1<'_, '_> {
    pub fn witness_id(&self) -> [u8; 16] {
        self.permit.witness_id()
    }

    pub fn frontier(&self) -> symthaea_qualification_witness_frontier::WitnessFrontierPointV1 {
        self.permit.frontier()
    }

    pub fn relation(&self) -> WitnessFrontierRecoveryRelationV1 {
        self.permit.relation()
    }

    pub fn xenia_evidence(&self) -> &XeniaExternalWitnessFrontierV1 {
        self.xenia
    }
}

/// Classify one already-verified Xenia witness frontier against one audited local
/// SQLite witness history while the writer barrier remains live.
///
/// This function accepts no raw Xenia fields and no copied recovery disposition.
/// Both underlying proofs must already exist as opaque reviewed types.
pub fn classify_guarded_xenia_witness_frontier_v1<'g, 'x>(
    guard: &'g SqliteWitnessFrontierPublicationGuard,
    xenia: &'x XeniaExternalWitnessFrontierV1,
) -> Result<GuardedXeniaWitnessFrontierDecisionV1<'g, 'x>, SqliteWitnessFrontierGuardError> {
    let decision = guard.classify(Some(xenia.external()))?;
    Ok(GuardedXeniaWitnessFrontierDecisionV1 { decision, xenia })
}
