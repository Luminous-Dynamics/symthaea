// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdi::prelude::*;

/// A trust attestation between two agents.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct TrustAttestation {
    /// Agent being trusted.
    pub subject: AgentPubKey,
    /// Trust score [0.0, 1.0].
    pub trust_score: f64,
    /// Whether verified with post-quantum crypto.
    pub pq_verified: bool,
    /// Domain of trust (e.g., "governance", "technical", "general").
    pub domain: String,
    /// Optional note.
    pub note: String,
    /// Timestamp (µs since epoch).
    pub timestamp_us: u64,
}

/// Revocation of a previous trust attestation.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct TrustRevocation {
    /// Action hash of the attestation being revoked.
    pub attestation_hash: ActionHash,
    /// Reason for revocation.
    pub reason: String,
    /// Timestamp (µs since epoch).
    pub timestamp_us: u64,
}

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(name = "TrustAttestation", visibility = "public")]
    TrustAttestation(TrustAttestation),
    #[entry_type(name = "TrustRevocation", visibility = "public")]
    TrustRevocation(TrustRevocation),
    #[entry_type(name = "Anchor", visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    AttestorToAttestations,
    SubjectToAttestations,
    AttestationToRevocations,
    AllAttestations,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::TrustAttestation(att) => {
                        if att.trust_score < 0.0 || att.trust_score > 1.0 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Trust score must be in [0.0, 1.0]".to_string(),
                            ));
                        }
                        if att.domain.len() > 64 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Domain too long (max 64)".to_string(),
                            ));
                        }
                        if att.note.len() > 512 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Note too long (max 512)".to_string(),
                            ));
                        }
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::TrustRevocation(rev) => {
                        if rev.reason.len() > 512 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Reason too long (max 512)".to_string(),
                            ));
                        }
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. }
        | FlatOp::RegisterDeleteLink { .. }
        | FlatOp::StoreRecord(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_)
        | FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
    }
}
