// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measured-boot event-log replay assurance for fresh TPM quote evidence.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_tpm2_attestation_possession::{
    AttestationPossessionRecord, Tpm2QuoteArtifacts, POSSESSION_SCOPE,
};

const LOG_BUNDLE_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-measured-boot-log-bundle-v1\0";
const REPLAY_RECEIPT_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-measured-boot-replay-receipt-v1\0";
const REPLAY_RECORD_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-measured-boot-replay-record-v1\0";

pub const REPLAY_SCOPE: &str =
    "event_log_replays_to_fresh_quoted_pcrs_not_reference_state_approval_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasuredBootLogBundle {
    pub schema_version: String,
    pub boot_session_id: String,
    pub platform_qualification_digest: String,
    pub event_log_ref: String,
    pub event_log_digest: String,
    pub final_events_ref: Option<String>,
    pub final_events_digest: Option<String>,
    pub event_count: u64,
    pub selected_pcr_event_count: u64,
    pub collected_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl MeasuredBootLogBundle {
    pub fn validate(&self) -> bool {
        let final_events_valid = match (&self.final_events_ref, &self.final_events_digest) {
            (Some(reference), Some(digest)) => {
                !reference.trim().is_empty() && valid_digest(digest)
            }
            (None, None) => true,
            _ => false,
        };

        !self.schema_version.trim().is_empty()
            && !self.boot_session_id.trim().is_empty()
            && valid_digest(&self.platform_qualification_digest)
            && !self.event_log_ref.trim().is_empty()
            && valid_digest(&self.event_log_digest)
            && final_events_valid
            && self.event_count > 0
            && self.selected_pcr_event_count > 0
            && self.selected_pcr_event_count <= self.event_count
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn bundle_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(LOG_BUNDLE_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.boot_session_id);
        push_field(&mut hasher, &self.platform_qualification_digest);
        push_field(&mut hasher, &self.event_log_ref);
        push_field(&mut hasher, &self.event_log_digest);
        push_field(
            &mut hasher,
            self.final_events_ref.as_deref().unwrap_or(""),
        );
        push_field(
            &mut hasher,
            self.final_events_digest.as_deref().unwrap_or(""),
        );
        push_field(&mut hasher, &self.event_count.to_string());
        push_field(&mut hasher, &self.selected_pcr_event_count.to_string());
        push_field(&mut hasher, &self.collected_at_ms.to_string());
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasuredBootReplayPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_possession_digest: String,
    pub expected_platform_qualification_digest: String,
    pub expected_pcr_selection_digest: String,
    pub expected_boot_session_id: String,
    pub expected_verifier_ref: String,
    pub expected_replay_tool_digest: String,
    pub require_final_events: bool,
    pub max_possession_to_replay_ms: u64,
    pub max_log_to_verification_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl MeasuredBootReplayPolicy {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.policy_id.trim().is_empty()
            && valid_digest(&self.expected_possession_digest)
            && valid_digest(&self.expected_platform_qualification_digest)
            && valid_digest(&self.expected_pcr_selection_digest)
            && !self.expected_boot_session_id.trim().is_empty()
            && !self.expected_verifier_ref.trim().is_empty()
            && valid_digest(&self.expected_replay_tool_digest)
            && self.max_possession_to_replay_ms > 0
            && self.max_log_to_verification_ms > 0
            && nonempty_refs(&self.evidence_refs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasuredBootReplayVerificationReceipt {
    pub schema_version: String,
    pub receipt_id: String,
    pub verifier_ref: String,
    pub replay_tool_ref: String,
    pub replay_tool_digest: String,
    pub possession_digest: String,
    pub quote_artifact_digest: String,
    pub log_bundle_digest: String,
    pub event_log_digest: String,
    pub final_events_digest: Option<String>,
    pub pcr_selection_digest: String,
    pub quoted_pcr_values_digest: String,
    pub replayed_pcr_values_digest: String,
    pub event_count: u64,
    pub selected_pcr_event_count: u64,
    pub order_preserved: bool,
    pub all_selected_pcr_events_replayed: bool,
    pub final_events_included: bool,
    pub replay_matches_quoted_pcrs: bool,
    pub verified_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl MeasuredBootReplayVerificationReceipt {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.receipt_id.trim().is_empty()
            && !self.verifier_ref.trim().is_empty()
            && !self.replay_tool_ref.trim().is_empty()
            && valid_digest(&self.replay_tool_digest)
            && valid_digest(&self.possession_digest)
            && valid_digest(&self.quote_artifact_digest)
            && valid_digest(&self.log_bundle_digest)
            && valid_digest(&self.event_log_digest)
            && self
                .final_events_digest
                .as_ref()
                .is_none_or(|digest| valid_digest(digest))
            && valid_digest(&self.pcr_selection_digest)
            && valid_digest(&self.quoted_pcr_values_digest)
            && valid_digest(&self.replayed_pcr_values_digest)
            && self.event_count > 0
            && self.selected_pcr_event_count > 0
            && self.selected_pcr_event_count <= self.event_count
            && self.order_preserved
            && self.all_selected_pcr_events_replayed
            && self.replay_matches_quoted_pcrs
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn receipt_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPLAY_RECEIPT_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.receipt_id);
        push_field(&mut hasher, &self.verifier_ref);
        push_field(&mut hasher, &self.replay_tool_ref);
        push_field(&mut hasher, &self.replay_tool_digest);
        push_field(&mut hasher, &self.possession_digest);
        push_field(&mut hasher, &self.quote_artifact_digest);
        push_field(&mut hasher, &self.log_bundle_digest);
        push_field(&mut hasher, &self.event_log_digest);
        push_field(
            &mut hasher,
            self.final_events_digest.as_deref().unwrap_or(""),
        );
        push_field(&mut hasher, &self.pcr_selection_digest);
        push_field(&mut hasher, &self.quoted_pcr_values_digest);
        push_field(&mut hasher, &self.replayed_pcr_values_digest);
        push_field(&mut hasher, &self.event_count.to_string());
        push_field(&mut hasher, &self.selected_pcr_event_count.to_string());
        for value in [
            self.order_preserved,
            self.all_selected_pcr_events_replayed,
            self.final_events_included,
            self.replay_matches_quoted_pcrs,
        ] {
            push_field(&mut hasher, if value { "1" } else { "0" });
        }
        push_field(&mut hasher, &self.verified_at_ms.to_string());
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasuredBootReplayRecord {
    pub schema_version: String,
    pub policy_id: String,
    pub possession_digest: String,
    pub quote_artifact_digest: String,
    pub log_bundle_digest: String,
    pub replay_receipt_digest: String,
    pub platform_qualification_digest: String,
    pub pcr_selection_digest: String,
    pub quoted_pcr_values_digest: String,
    pub qualified_at_ms: u64,
    pub scope: String,
    pub evidence_refs: Vec<String>,
}

impl MeasuredBootReplayRecord {
    pub fn replay_record_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPLAY_RECORD_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);
        push_field(&mut hasher, &self.possession_digest);
        push_field(&mut hasher, &self.quote_artifact_digest);
        push_field(&mut hasher, &self.log_bundle_digest);
        push_field(&mut hasher, &self.replay_receipt_digest);
        push_field(&mut hasher, &self.platform_qualification_digest);
        push_field(&mut hasher, &self.pcr_selection_digest);
        push_field(&mut hasher, &self.quoted_pcr_values_digest);
        push_field(&mut hasher, &self.qualified_at_ms.to_string());
        push_field(&mut hasher, &self.scope);
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasuredBootReplayError {
    InvalidPolicy,
    InvalidPossessionRecord,
    PossessionMismatch,
    PlatformMismatch,
    PcrSelectionMismatch,
    PossessionTooOld,
    InvalidQuoteArtifacts,
    QuoteBindingMismatch,
    InvalidLogBundle,
    BootSessionMismatch,
    FinalEventsRequired,
    InvalidReplayReceipt,
    VerifierMismatch,
    ReplayToolMismatch,
    ReplayBindingMismatch,
    EventCountsMismatch,
    FinalEventsMismatch,
    QuotedPcrValuesMismatch,
    ReplayedPcrValuesMismatch,
    VerificationBeforeLogCollection,
    VerificationLagExceeded,
    VerificationAfterQualification,
}

pub fn qualify_measured_boot_replay(
    policy: &MeasuredBootReplayPolicy,
    possession: &AttestationPossessionRecord,
    quote: &Tpm2QuoteArtifacts,
    log_bundle: &MeasuredBootLogBundle,
    replay: &MeasuredBootReplayVerificationReceipt,
    qualified_at_ms: u64,
) -> Result<MeasuredBootReplayRecord, MeasuredBootReplayError> {
    if !policy.validate() {
        return Err(MeasuredBootReplayError::InvalidPolicy);
    }
    if !valid_possession(possession) {
        return Err(MeasuredBootReplayError::InvalidPossessionRecord);
    }

    let possession_digest = possession.possession_digest();
    if possession_digest != policy.expected_possession_digest {
        return Err(MeasuredBootReplayError::PossessionMismatch);
    }
    if possession.platform_qualification_digest != policy.expected_platform_qualification_digest {
        return Err(MeasuredBootReplayError::PlatformMismatch);
    }
    if possession.pcr_selection_digest != policy.expected_pcr_selection_digest {
        return Err(MeasuredBootReplayError::PcrSelectionMismatch);
    }
    if qualified_at_ms < possession.qualified_at_ms
        || qualified_at_ms - possession.qualified_at_ms > policy.max_possession_to_replay_ms
    {
        return Err(MeasuredBootReplayError::PossessionTooOld);
    }

    if !quote.validate() {
        return Err(MeasuredBootReplayError::InvalidQuoteArtifacts);
    }
    let quote_digest = quote.artifact_digest();
    if quote_digest != possession.quote_artifact_digest
        || quote.platform_qualification_digest != possession.platform_qualification_digest
        || quote.pcr_selection_digest != possession.pcr_selection_digest
    {
        return Err(MeasuredBootReplayError::QuoteBindingMismatch);
    }

    if !log_bundle.validate() {
        return Err(MeasuredBootReplayError::InvalidLogBundle);
    }
    if log_bundle.boot_session_id != policy.expected_boot_session_id {
        return Err(MeasuredBootReplayError::BootSessionMismatch);
    }
    if log_bundle.platform_qualification_digest != possession.platform_qualification_digest {
        return Err(MeasuredBootReplayError::PlatformMismatch);
    }
    if policy.require_final_events
        && (log_bundle.final_events_ref.is_none() || log_bundle.final_events_digest.is_none())
    {
        return Err(MeasuredBootReplayError::FinalEventsRequired);
    }

    if !replay.validate() {
        return Err(MeasuredBootReplayError::InvalidReplayReceipt);
    }
    if replay.verifier_ref != policy.expected_verifier_ref {
        return Err(MeasuredBootReplayError::VerifierMismatch);
    }
    if replay.replay_tool_digest != policy.expected_replay_tool_digest {
        return Err(MeasuredBootReplayError::ReplayToolMismatch);
    }
    if replay.possession_digest != possession_digest
        || replay.quote_artifact_digest != quote_digest
        || replay.log_bundle_digest != log_bundle.bundle_digest()
        || replay.event_log_digest != log_bundle.event_log_digest
        || replay.pcr_selection_digest != possession.pcr_selection_digest
    {
        return Err(MeasuredBootReplayError::ReplayBindingMismatch);
    }
    if replay.event_count != log_bundle.event_count
        || replay.selected_pcr_event_count != log_bundle.selected_pcr_event_count
    {
        return Err(MeasuredBootReplayError::EventCountsMismatch);
    }
    if replay.final_events_digest != log_bundle.final_events_digest
        || (policy.require_final_events && !replay.final_events_included)
    {
        return Err(MeasuredBootReplayError::FinalEventsMismatch);
    }
    if replay.quoted_pcr_values_digest != quote.pcr_values_digest {
        return Err(MeasuredBootReplayError::QuotedPcrValuesMismatch);
    }
    if replay.replayed_pcr_values_digest != quote.pcr_values_digest {
        return Err(MeasuredBootReplayError::ReplayedPcrValuesMismatch);
    }
    if replay.verified_at_ms < log_bundle.collected_at_ms {
        return Err(MeasuredBootReplayError::VerificationBeforeLogCollection);
    }
    if replay.verified_at_ms - log_bundle.collected_at_ms > policy.max_log_to_verification_ms {
        return Err(MeasuredBootReplayError::VerificationLagExceeded);
    }
    if replay.verified_at_ms > qualified_at_ms {
        return Err(MeasuredBootReplayError::VerificationAfterQualification);
    }

    Ok(MeasuredBootReplayRecord {
        schema_version: "1".into(),
        policy_id: policy.policy_id.clone(),
        possession_digest,
        quote_artifact_digest: quote_digest,
        log_bundle_digest: log_bundle.bundle_digest(),
        replay_receipt_digest: replay.receipt_digest(),
        platform_qualification_digest: possession.platform_qualification_digest.clone(),
        pcr_selection_digest: possession.pcr_selection_digest.clone(),
        quoted_pcr_values_digest: quote.pcr_values_digest.clone(),
        qualified_at_ms,
        scope: REPLAY_SCOPE.into(),
        evidence_refs: vec![
            format!("possession:{}", possession.possession_digest()),
            format!("quote:{}", quote.artifact_digest()),
            format!("log-bundle:{}", log_bundle.bundle_digest()),
            format!("replay:{}", replay.receipt_digest()),
        ],
    })
}

fn valid_possession(possession: &AttestationPossessionRecord) -> bool {
    possession.scope == POSSESSION_SCOPE
        && valid_digest(&possession.platform_qualification_digest)
        && valid_digest(&possession.challenge_digest)
        && valid_digest(&possession.ak_binding_digest)
        && valid_digest(&possession.quote_artifact_digest)
        && valid_digest(&possession.verification_receipt_digest)
        && valid_digest(&possession.pcr_selection_digest)
        && nonempty_refs(&possession.evidence_refs)
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

fn nonempty_refs(refs: &[String]) -> bool {
    !refs.is_empty() && refs.iter().all(|value| !value.trim().is_empty())
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn possession() -> AttestationPossessionRecord {
        AttestationPossessionRecord {
            schema_version: "1".into(),
            policy_id: "policy:possession".into(),
            platform_qualification_digest: digest("platform"),
            challenge_id: "challenge:1".into(),
            challenge_digest: digest("challenge"),
            ak_binding_digest: digest("ak"),
            quote_artifact_digest: quote().artifact_digest(),
            verification_receipt_digest: digest("quote-verification"),
            pcr_selection_digest: digest("pcr-selection"),
            qualified_at_ms: 10_400,
            scope: POSSESSION_SCOPE.into(),
            evidence_refs: vec!["evidence:possession".into()],
        }
    }

    fn quote() -> Tpm2QuoteArtifacts {
        Tpm2QuoteArtifacts {
            schema_version: "1".into(),
            quote_id: "quote:1".into(),
            challenge_id: "challenge:1".into(),
            platform_qualification_digest: digest("platform"),
            qualifying_data_hex: "ab".repeat(32),
            quote_message_digest: digest("quote-message"),
            signature_digest: digest("signature"),
            pcr_values_digest: digest("quoted-pcr-values"),
            ak_public_key_digest: digest("ak-public"),
            pcr_selection_digest: digest("pcr-selection"),
            collected_at_ms: 10_200,
            evidence_refs: vec!["artifact:quote".into()],
        }
    }

    fn log_bundle() -> MeasuredBootLogBundle {
        MeasuredBootLogBundle {
            schema_version: "1".into(),
            boot_session_id: "boot:1".into(),
            platform_qualification_digest: digest("platform"),
            event_log_ref: "file:binary_bios_measurements".into(),
            event_log_digest: digest("event-log"),
            final_events_ref: Some("file:final-events".into()),
            final_events_digest: Some(digest("final-events")),
            event_count: 120,
            selected_pcr_event_count: 50,
            collected_at_ms: 10_500,
            evidence_refs: vec!["capture:measured-boot".into()],
        }
    }

    fn replay_receipt() -> MeasuredBootReplayVerificationReceipt {
        let possession = possession();
        let quote = quote();
        let bundle = log_bundle();
        MeasuredBootReplayVerificationReceipt {
            schema_version: "1".into(),
            receipt_id: "replay:1".into(),
            verifier_ref: "verifier:measured-boot".into(),
            replay_tool_ref: "tool:eventlog-replay".into(),
            replay_tool_digest: digest("eventlog-replay-tool"),
            possession_digest: possession.possession_digest(),
            quote_artifact_digest: quote.artifact_digest(),
            log_bundle_digest: bundle.bundle_digest(),
            event_log_digest: bundle.event_log_digest.clone(),
            final_events_digest: bundle.final_events_digest.clone(),
            pcr_selection_digest: possession.pcr_selection_digest.clone(),
            quoted_pcr_values_digest: quote.pcr_values_digest.clone(),
            replayed_pcr_values_digest: quote.pcr_values_digest.clone(),
            event_count: bundle.event_count,
            selected_pcr_event_count: bundle.selected_pcr_event_count,
            order_preserved: true,
            all_selected_pcr_events_replayed: true,
            final_events_included: true,
            replay_matches_quoted_pcrs: true,
            verified_at_ms: 10_600,
            evidence_refs: vec!["audit:eventlog-replay".into()],
        }
    }

    fn policy() -> MeasuredBootReplayPolicy {
        let possession = possession();
        MeasuredBootReplayPolicy {
            schema_version: "1".into(),
            policy_id: "policy:replay:1".into(),
            expected_possession_digest: possession.possession_digest(),
            expected_platform_qualification_digest: possession.platform_qualification_digest,
            expected_pcr_selection_digest: possession.pcr_selection_digest,
            expected_boot_session_id: "boot:1".into(),
            expected_verifier_ref: "verifier:measured-boot".into(),
            expected_replay_tool_digest: digest("eventlog-replay-tool"),
            require_final_events: true,
            max_possession_to_replay_ms: 1_000,
            max_log_to_verification_ms: 500,
            evidence_refs: vec!["review:measured-boot-policy".into()],
        }
    }

    #[test]
    fn exact_log_replay_matches_fresh_quoted_pcr_state() {
        let record = qualify_measured_boot_replay(
            &policy(),
            &possession(),
            &quote(),
            &log_bundle(),
            &replay_receipt(),
            10_700,
        )
        .unwrap();
        assert_eq!(record.scope, REPLAY_SCOPE);
        assert!(!record.grants_physical_authority());
    }

    #[test]
    fn pcr_replay_mismatch_is_rejected() {
        let mut replay = replay_receipt();
        replay.replayed_pcr_values_digest = digest("different-pcr-values");
        assert_eq!(
            qualify_measured_boot_replay(
                &policy(),
                &possession(),
                &quote(),
                &log_bundle(),
                &replay,
                10_700,
            ),
            Err(MeasuredBootReplayError::ReplayedPcrValuesMismatch)
        );
    }

    #[test]
    fn omitted_final_events_are_rejected_when_required() {
        let mut bundle = log_bundle();
        bundle.final_events_ref = None;
        bundle.final_events_digest = None;
        let mut replay = replay_receipt();
        replay.log_bundle_digest = bundle.bundle_digest();
        replay.final_events_digest = None;
        assert_eq!(
            qualify_measured_boot_replay(
                &policy(),
                &possession(),
                &quote(),
                &bundle,
                &replay,
                10_700,
            ),
            Err(MeasuredBootReplayError::FinalEventsRequired)
        );
    }

    #[test]
    fn event_order_loss_is_fail_closed() {
        let mut replay = replay_receipt();
        replay.order_preserved = false;
        assert_eq!(
            qualify_measured_boot_replay(
                &policy(),
                &possession(),
                &quote(),
                &log_bundle(),
                &replay,
                10_700,
            ),
            Err(MeasuredBootReplayError::InvalidReplayReceipt)
        );
    }

    #[test]
    fn stale_possession_cannot_be_promoted_to_current_measured_state() {
        assert_eq!(
            qualify_measured_boot_replay(
                &policy(),
                &possession(),
                &quote(),
                &log_bundle(),
                &replay_receipt(),
                12_000,
            ),
            Err(MeasuredBootReplayError::PossessionTooOld)
        );
    }

    #[test]
    fn replay_tool_substitution_is_rejected() {
        let mut replay = replay_receipt();
        replay.replay_tool_digest = digest("other-tool");
        assert_eq!(
            qualify_measured_boot_replay(
                &policy(),
                &possession(),
                &quote(),
                &log_bundle(),
                &replay,
                10_700,
            ),
            Err(MeasuredBootReplayError::ReplayToolMismatch)
        );
    }
}
