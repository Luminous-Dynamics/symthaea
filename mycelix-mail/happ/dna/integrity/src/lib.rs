// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Holochain 0.6 - Integrity zomes use hdi
use hdi::prelude::*;

/// Core mail message entry type
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MailMessage {
    pub from_did: String,
    pub to_did: String,
    pub subject_encrypted: Vec<u8>,
    pub body_cid: String, // IPFS content ID
    pub timestamp: Timestamp,
    pub thread_id: Option<String>,
    pub epistemic_tier: EpistemicTier,
}

/// Trust score for spam filtering
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrustScore {
    pub did: String,
    pub score: f64, // 0.0 - 1.0
    pub last_updated: Timestamp,
    pub source: String,
}

/// Epistemic tiers from Mycelix Epistemic Charter v2.0
#[derive(Clone, PartialEq, Serialize, Deserialize, SerializedBytes, Debug)]
pub enum EpistemicTier {
    Tier0Null,
    Tier1Testimonial,
    Tier2PrivatelyVerifiable,
    Tier3CryptographicallyProven,
    Tier4PubliclyReproducible,
}

/// Contact entry for address book
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Contact {
    pub name: String,
    pub did: String,
    pub email_alias: Option<String>,
    pub notes: Option<String>,
    pub added_at: Timestamp,
}

/// Mapping between DID and the agent pubkey that owns it
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DidBinding {
    pub did: String,
    pub agent_pub_key: AgentPubKey,
}

/// Persisted spam report
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SpamReport {
    pub reporter: AgentPubKey,
    pub spammer_did: String,
    pub message_hash: ActionHash,
    pub reason: String,
    pub reported_at: Timestamp,
}

/// Entry types for the DNA
#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    MailMessage(MailMessage),
    TrustScore(TrustScore),
    Contact(Contact),
    DidBinding(DidBinding),
    SpamReport(SpamReport),
}

/// Link types for connecting entries
#[hdk_link_types]
pub enum LinkTypes {
    ToInbox,
    FromOutbox,
    ThreadReply,
    TrustByDid,
    TrustIndex,
    ContactLink,
    SpamReports,
    DidBindingLink,
}

/// Basic validation to guard against malformed data
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => match store_entry.action.hashed.content.entry_type() {
            EntryType::App(app_entry_def) => {
                let entry = store_entry.entry;
                match EntryTypes::deserialize_from_type(
                    app_entry_def.zome_index,
                    app_entry_def.entry_index,
                    &entry,
                )? {
                    Some(EntryTypes::TrustScore(score)) => validate_trust_score(score),
                    Some(EntryTypes::MailMessage(message)) => validate_mail_message(message),
                    Some(EntryTypes::DidBinding(binding)) => validate_did_binding(binding),
                    Some(EntryTypes::SpamReport(report)) => validate_spam_report(report),
                    Some(EntryTypes::Contact(contact)) => validate_contact(contact),
                    None => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_trust_score(score: TrustScore) -> ExternResult<ValidateCallbackResult> {
    if !(0.0..=1.0).contains(&score.score) {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score must be between 0.0 and 1.0".into(),
        ));
    }
    if score.did.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score DID cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_mail_message(message: MailMessage) -> ExternResult<ValidateCallbackResult> {
    if message.from_did.trim().is_empty() || message.to_did.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Message DIDs cannot be empty".into(),
        ));
    }
    if message.subject_encrypted.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject cannot be empty".into(),
        ));
    }
    if message.body_cid.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Body CID cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_did_binding(binding: DidBinding) -> ExternResult<ValidateCallbackResult> {
    if binding.did.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "DID cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_spam_report(report: SpamReport) -> ExternResult<ValidateCallbackResult> {
    if report.spammer_did.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Spam report must include spammer DID".into(),
        ));
    }
    if report.reason.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Spam report reason cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_contact(contact: Contact) -> ExternResult<ValidateCallbackResult> {
    if contact.did.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Contact DID cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
