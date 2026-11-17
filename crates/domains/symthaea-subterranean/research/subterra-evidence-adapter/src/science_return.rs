//! Evidence-aware science-return lifecycle.
//!
//! Distinguishes a proposed measurement, execution, acquisition, integrity
//! verification, local persistence, store-and-forward, delivery, and admission
//! to a scientific belief update.
//!
//! The record validator checks lifecycle sequence, canonical UTC timestamp
//! syntax, clock identity and uncertainty, and definite same-clock regressions.
//! Evidence references are opaque: this module does not authenticate them,
//! verify signatures, prove durable storage, or establish custody authority.

use std::collections::BTreeSet;

use crate::{SiteEvidenceRecord, SiteIssue};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MIN_CLOCK_UNCERTAINTY_MS: u64 = 1_000;

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScienceReturnStage {
    Proposed,
    Executed,
    Acquired,
    IntegrityVerified,
    Persisted,
    StoreAndForwarded,
    Delivered,
    Admitted,
    Failed,
    Cancelled,
}

impl ScienceReturnStage {
    const fn is_terminal(self) -> bool {
        matches!(self, Self::Admitted | Self::Failed | Self::Cancelled)
    }

    const fn requires_custody(self) -> bool {
        matches!(
            self,
            Self::Persisted | Self::StoreAndForwarded | Self::Delivered | Self::Admitted
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScienceReturnEvent {
    /// Monotonic, one-based sequence within this observation record.
    pub sequence: u64,
    pub event_id: String,
    pub stage: ScienceReturnStage,
    /// Canonical UTC timestamp: YYYY-MM-DDTHH:MM:SSZ.
    pub recorded_at: String,
    /// Identity of the clock that produced recorded_at; different clocks are
    /// not compared as if they were synchronized.
    pub clock_id: String,
    /// Conservative absolute clock uncertainty in milliseconds.
    pub clock_uncertainty_ms: u64,
    pub actor_id: String,
    /// Opaque source/evidence reference. Not an authenticity proof.
    pub evidence_ref: String,
    #[serde(default)]
    pub calibration_ref: Option<String>,
    #[serde(default)]
    pub custody_ref: Option<String>,
    #[serde(default)]
    pub delivery_receipt_ref: Option<String>,
    #[serde(default)]
    pub admission_ref: Option<String>,
    #[serde(default)]
    pub disposition_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadSignatureAlgorithm {
    Ed25519V1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadAttestation {
    pub algorithm: PayloadSignatureAlgorithm,
    /// Key identifier resolved by the caller against an independently trusted key registry.
    pub key_id: String,
    /// Canonical lowercase, 128-character Ed25519 signature over the bound payload envelope.
    pub signature_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScienceReturnPayload {
    /// Lowercase, canonical 64-character SHA-256 digest of the acquired bytes.
    pub sha256: String,
    /// Exact byte length of the acquired payload.
    pub byte_length: u64,
    /// Detached signature metadata. The record does not itself establish trust in the key.
    #[serde(default)]
    pub attestation: Option<PayloadAttestation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScienceReturnRecord {
    pub site_id: String,
    pub observation_id: String,
    /// Required once an observation has been acquired; absent for pre-acquisition failure.
    #[serde(default)]
    pub payload: Option<ScienceReturnPayload>,
    pub events: Vec<ScienceReturnEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScienceReturnIssue {
    EmptySiteId,
    EmptyObservationId,
    EmptyEventLog,
    InvalidSequence { expected: u64, found: u64 },
    EmptyEventId,
    DuplicateEventId(String),
    EmptyRecordedAt(String),
    InvalidRecordedAt(String),
    EmptyClockId(String),
    ClockUncertaintyBelowResolution {
        event_id: String,
        observed_ms: u64,
        minimum_ms: u64,
    },
    TimestampRegression {
        clock_id: String,
        previous_event_id: String,
        current_event_id: String,
    },
    EmptyActorId(String),
    EmptyEvidenceRef(String),
    InvalidTransition {
        from: ScienceReturnStage,
        to: ScienceReturnStage,
    },
    TerminalEventNotLast(ScienceReturnStage),
    IntegrityVerificationMissingCalibration(String),
    CustodyReferenceMissing(String),
    DeliveryReceiptMissing(String),
    AdmissionReferenceMissing(String),
    TerminalDispositionMissing(ScienceReturnStage),
    SiteIdMismatch { record: String, source: String },
    ObservationNotInSite(String),
    InvalidSiteEvidence(Vec<SiteIssue>),
    PayloadMissingAfterAcquisition,
    PayloadPresentBeforeAcquisition,
    MalformedPayloadSha256,
    PayloadByteLengthMismatch { expected: u64, actual: u64 },
    PayloadSha256Mismatch,
    PayloadNotDeclaredForVerification,
    PayloadAttestationMissing,
    PayloadAttestationKeyIdEmpty,
    EmptyTrustedAttestationActorId,
    EmptyTrustedAttestationKeyId,
    PayloadAttestationActorIdMismatch {
        acquired: String,
        trusted: String,
    },
    PayloadAttestationKeyIdMismatch {
        declared: String,
        trusted: String,
    },
    MalformedPayloadAttestationSignature,
    InvalidPayloadAttestationPublicKey,
    InvalidPayloadAttestationSignature,
    PayloadAttestationSignatureInvalid,
    PayloadAcquisitionEventMissing,
}

fn transition_allowed(
    previous: ScienceReturnStage,
    next: ScienceReturnStage,
) -> bool {
    use ScienceReturnStage::{
        Acquired, Admitted, Cancelled, Delivered, Executed, Failed, IntegrityVerified,
        Persisted, Proposed, StoreAndForwarded,
    };

    match previous {
        Proposed => matches!(next, Executed | Failed | Cancelled),
        Executed => matches!(next, Acquired | Failed | Cancelled),
        Acquired => matches!(next, IntegrityVerified | Failed | Cancelled),
        IntegrityVerified => matches!(next, Persisted | Failed | Cancelled),
        Persisted => matches!(next, Delivered | StoreAndForwarded | Failed | Cancelled),
        StoreAndForwarded => matches!(next, Delivered | Failed | Cancelled),
        Delivered => matches!(next, Admitted | Failed | Cancelled),
        Admitted | Failed | Cancelled => false,
    }
}


fn parse_canonical_utc_millis(value: &str) -> Option<i128> {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return None;
    }

    fn digits(bytes: &[u8], start: usize, end: usize) -> Option<i128> {
        bytes.get(start..end)?.iter().try_fold(0_i128, |value, byte| {
            if !byte.is_ascii_digit() {
                return None;
            }
            Some(value * 10 + i128::from(*byte - b'0'))
        })
    }

    let year = digits(bytes, 0, 4)?;
    let month = digits(bytes, 5, 7)?;
    let day = digits(bytes, 8, 10)?;
    let hour = digits(bytes, 11, 13)?;
    let minute = digits(bytes, 14, 16)?;
    let second = digits(bytes, 17, 19)?;

    if year == 0 || !(1..=12).contains(&month)
        || hour > 23 || minute > 59 || second > 59
    {
        return None;
    }

    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return None,
    };
    if day < 1 || day > days_in_month {
        return None;
    }

    let days_before_year = |y: i128| {
        let prior = y - 1;
        prior * 365 + prior / 4 - prior / 100 + prior / 400
    };
    let days_before_month = match month {
        1 => 0,
        2 => 31,
        3 => 59,
        4 => 90,
        5 => 120,
        6 => 151,
        7 => 181,
        8 => 212,
        9 => 243,
        10 => 273,
        11 => 304,
        12 => 334,
        _ => return None,
    };
    let leap_adjustment = if leap_year && month > 2 { 1 } else { 0 };
    let day_of_year = days_before_month + leap_adjustment + day - 1;
    let days_from_epoch =
        days_before_year(year) - days_before_year(1970) + day_of_year;
    let seconds_from_epoch =
        days_from_epoch * 86_400 + hour * 3_600 + minute * 60 + second;
    Some(seconds_from_epoch * 1_000)
}


fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn sha256_hex(payload: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(payload);
    let mut output = String::with_capacity(64);
    for byte in digest.iter().copied() {
        output.push(char::from(HEX[(byte >> 4) as usize]));
        output.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    output
}


fn is_canonical_lower_hex(value: &str, expected_chars: usize) -> bool {
    value.len() == expected_chars
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn decode_lower_hex(value: &str) -> Option<Vec<u8>> {
    if value.len() % 2 != 0 || !is_canonical_lower_hex(value, value.len()) {
        return None;
    }
    let mut decoded = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let digit = |byte: u8| -> Option<u8> {
            match byte {
                b'0'..=b'9' => Some(byte - b'0'),
                b'a'..=b'f' => Some(byte - b'a' + 10),
                _ => None,
            }
        };
        decoded.push((digit(pair[0])? << 4) | digit(pair[1])?);
    }
    Some(decoded)
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[(byte >> 4) as usize]));
        output.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    output
}

fn append_len_prefixed(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value);
}

fn payload_attestation_message(
    site_id: &str,
    observation_id: &str,
    acquired_event: &ScienceReturnEvent,
    payload: &ScienceReturnPayload,
    key_id: &str,
) -> Vec<u8> {
    let mut message =
        b"LUMINOUS-DYNAMICS\0SUBTERRA-SCIENCE-RETURN-PAYLOAD\0V0.1\0ED25519\0".to_vec();
    append_len_prefixed(&mut message, site_id.as_bytes());
    append_len_prefixed(&mut message, observation_id.as_bytes());
    append_len_prefixed(&mut message, key_id.as_bytes());
    append_len_prefixed(&mut message, acquired_event.event_id.as_bytes());
    message.extend_from_slice(&acquired_event.sequence.to_be_bytes());
    append_len_prefixed(&mut message, acquired_event.actor_id.as_bytes());
    append_len_prefixed(&mut message, acquired_event.evidence_ref.as_bytes());
    append_len_prefixed(&mut message, acquired_event.recorded_at.as_bytes());
    append_len_prefixed(&mut message, acquired_event.clock_id.as_bytes());
    message.extend_from_slice(&acquired_event.clock_uncertainty_ms.to_be_bytes());
    append_len_prefixed(&mut message, payload.sha256.as_bytes());
    message.extend_from_slice(&payload.byte_length.to_be_bytes());
    message
}

impl ScienceReturnRecord {
    /// Validate event ordering and stage-specific references.
    ///
    /// The only successful terminal state is Admitted. Store-and-forward is
    /// explicitly pending delivery, not a delivered observation.
    pub fn validate(&self) -> Result<(), Vec<ScienceReturnIssue>> {
        let mut issues = Vec::new();

        if !nonempty(&self.site_id) {
            issues.push(ScienceReturnIssue::EmptySiteId);
        }
        if !nonempty(&self.observation_id) {
            issues.push(ScienceReturnIssue::EmptyObservationId);
        }
        if self.events.is_empty() {
            issues.push(ScienceReturnIssue::EmptyEventLog);
            return Err(issues);
        }

        let has_acquired_payload = self.events.iter().any(|event| {
            matches!(
                event.stage,
                ScienceReturnStage::Acquired
                    | ScienceReturnStage::IntegrityVerified
                    | ScienceReturnStage::Persisted
                    | ScienceReturnStage::StoreAndForwarded
                    | ScienceReturnStage::Delivered
                    | ScienceReturnStage::Admitted
            )
        });
        match (&self.payload, has_acquired_payload) {
            (None, true) => issues.push(ScienceReturnIssue::PayloadMissingAfterAcquisition),
            (Some(_), false) => issues.push(ScienceReturnIssue::PayloadPresentBeforeAcquisition),
            (Some(payload), true) => {
                if !is_canonical_sha256(&payload.sha256) {
                    issues.push(ScienceReturnIssue::MalformedPayloadSha256);
                }
                if let Some(attestation) = &payload.attestation {
                    if !nonempty(&attestation.key_id) {
                        issues.push(ScienceReturnIssue::PayloadAttestationKeyIdEmpty);
                    }
                    if !is_canonical_lower_hex(&attestation.signature_hex, 128) {
                        issues.push(
                            ScienceReturnIssue::MalformedPayloadAttestationSignature,
                        );
                    }
                }
            }
            (None, false) => {}
        }

        let mut event_ids = BTreeSet::new();
        let mut previous_stage: Option<ScienceReturnStage> = None;
        let mut last_clock_times =
            std::collections::BTreeMap::<String, (i128, u64, String)>::new();

        for (index, event) in self.events.iter().enumerate() {
            let expected_sequence = index as u64 + 1;
            if event.sequence != expected_sequence {
                issues.push(ScienceReturnIssue::InvalidSequence {
                    expected: expected_sequence,
                    found: event.sequence,
                });
            }

            if !nonempty(&event.event_id) {
                issues.push(ScienceReturnIssue::EmptyEventId);
            }
            if !event_ids.insert(event.event_id.as_str()) {
                issues.push(ScienceReturnIssue::DuplicateEventId(
                    event.event_id.clone(),
                ));
            }
            if !nonempty(&event.recorded_at) {
                issues.push(ScienceReturnIssue::EmptyRecordedAt(
                    event.event_id.clone(),
                ));
            } else if let Some(timestamp_ms) = parse_canonical_utc_millis(&event.recorded_at) {
                if !nonempty(&event.clock_id) {
                    issues.push(ScienceReturnIssue::EmptyClockId(event.event_id.clone()));
                } else if event.clock_uncertainty_ms < MIN_CLOCK_UNCERTAINTY_MS {
                    issues.push(ScienceReturnIssue::ClockUncertaintyBelowResolution {
                        event_id: event.event_id.clone(),
                        observed_ms: event.clock_uncertainty_ms,
                        minimum_ms: MIN_CLOCK_UNCERTAINTY_MS,
                    });
                } else if let Some((previous_ms, previous_uncertainty, previous_event_id)) =
                    last_clock_times.get(&event.clock_id)
                {
                    let previous_lower =
                        *previous_ms - i128::from(*previous_uncertainty);
                    let current_upper =
                        timestamp_ms + i128::from(event.clock_uncertainty_ms);
                    if current_upper < previous_lower {
                        issues.push(ScienceReturnIssue::TimestampRegression {
                            clock_id: event.clock_id.clone(),
                            previous_event_id: previous_event_id.clone(),
                            current_event_id: event.event_id.clone(),
                        });
                    }
                    last_clock_times.insert(
                        event.clock_id.clone(),
                        (
                            timestamp_ms,
                            event.clock_uncertainty_ms,
                            event.event_id.clone(),
                        ),
                    );
                } else {
                    last_clock_times.insert(
                        event.clock_id.clone(),
                        (
                            timestamp_ms,
                            event.clock_uncertainty_ms,
                            event.event_id.clone(),
                        ),
                    );
                }
            } else {
                issues.push(ScienceReturnIssue::InvalidRecordedAt(
                    event.event_id.clone(),
                ));
            }
            if !nonempty(&event.actor_id) {
                issues.push(ScienceReturnIssue::EmptyActorId(
                    event.event_id.clone(),
                ));
            }
            if !nonempty(&event.evidence_ref) {
                issues.push(ScienceReturnIssue::EmptyEvidenceRef(
                    event.event_id.clone(),
                ));
            }

            if let Some(previous) = previous_stage {
                if previous.is_terminal() {
                    issues.push(ScienceReturnIssue::TerminalEventNotLast(previous));
                } else if !transition_allowed(previous, event.stage) {
                    issues.push(ScienceReturnIssue::InvalidTransition {
                        from: previous,
                        to: event.stage,
                    });
                }
            } else if event.stage != ScienceReturnStage::Proposed
                && !matches!(event.stage, ScienceReturnStage::Failed | ScienceReturnStage::Cancelled)
            {
                issues.push(ScienceReturnIssue::InvalidTransition {
                    from: ScienceReturnStage::Proposed,
                    to: event.stage,
                });
            }

            if event.stage == ScienceReturnStage::IntegrityVerified
                && !event.calibration_ref.as_deref().is_some_and(nonempty)
            {
                issues.push(
                    ScienceReturnIssue::IntegrityVerificationMissingCalibration(
                        event.event_id.clone(),
                    ),
                );
            }
            if event.stage.requires_custody()
                && !event.custody_ref.as_deref().is_some_and(nonempty)
            {
                issues.push(ScienceReturnIssue::CustodyReferenceMissing(
                    event.event_id.clone(),
                ));
            }
            if matches!(
                event.stage,
                ScienceReturnStage::Delivered | ScienceReturnStage::Admitted
            ) && !event.delivery_receipt_ref.as_deref().is_some_and(nonempty)
            {
                issues.push(ScienceReturnIssue::DeliveryReceiptMissing(
                    event.event_id.clone(),
                ));
            }
            if event.stage == ScienceReturnStage::Admitted
                && !event.admission_ref.as_deref().is_some_and(nonempty)
            {
                issues.push(ScienceReturnIssue::AdmissionReferenceMissing(
                    event.event_id.clone(),
                ));
            }
            if matches!(event.stage, ScienceReturnStage::Failed | ScienceReturnStage::Cancelled)
                && !event.disposition_reason.as_deref().is_some_and(nonempty)
            {
                issues.push(ScienceReturnIssue::TerminalDispositionMissing(
                    event.stage,
                ));
            }

            previous_stage = Some(event.stage);
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    /// Verify the received payload bytes against the record's declared
    /// SHA-256 digest and byte length. This proves byte equality relative to
    /// the declaration; it does not authenticate who supplied the digest.
    pub fn verify_payload_bytes(
        &self,
        payload_bytes: &[u8],
    ) -> Result<(), Vec<ScienceReturnIssue>> {
        let mut issues = self.validate().err().unwrap_or_default();
        let Some(declared) = &self.payload else {
            if !issues.contains(&ScienceReturnIssue::PayloadMissingAfterAcquisition) {
                issues.push(ScienceReturnIssue::PayloadNotDeclaredForVerification);
            }
            return Err(issues);
        };
        let actual_length = payload_bytes.len() as u64;
        if declared.byte_length != actual_length {
            issues.push(ScienceReturnIssue::PayloadByteLengthMismatch {
                expected: declared.byte_length,
                actual: actual_length,
            });
        }
        if !is_canonical_sha256(&declared.sha256) {
            if !issues.contains(&ScienceReturnIssue::MalformedPayloadSha256) {
                issues.push(ScienceReturnIssue::MalformedPayloadSha256);
            }
        } else if declared.sha256 != sha256_hex(payload_bytes) {
            issues.push(ScienceReturnIssue::PayloadSha256Mismatch);
        }
        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }


    /// Verify acquired bytes and their detached Ed25519 attestation using an
    /// expected actor/key identity and public key supplied by an external,
    /// independently trusted registry.
    ///
    /// The signed envelope binds the payload to site/observation identity and
    /// the acquired event's ID, actor, clock and timestamp metadata. This
    /// method does not create or authenticate the registry, authorize the
    /// key's scope, check expiry/revocation, or establish durable custody.
    pub fn verify_signed_payload_bytes(
        &self,
        payload_bytes: &[u8],
        trusted_actor_id: &str,
        trusted_key_id: &str,
        trusted_public_key_bytes: &[u8],
    ) -> Result<(), Vec<ScienceReturnIssue>> {
        self.verify_payload_bytes(payload_bytes)?;

        if !nonempty(trusted_actor_id) {
            return Err(vec![ScienceReturnIssue::EmptyTrustedAttestationActorId]);
        }
        if !nonempty(trusted_key_id) {
            return Err(vec![ScienceReturnIssue::EmptyTrustedAttestationKeyId]);
        }
        let Some(payload) = &self.payload else {
            return Err(vec![ScienceReturnIssue::PayloadNotDeclaredForVerification]);
        };
        let Some(attestation) = &payload.attestation else {
            return Err(vec![ScienceReturnIssue::PayloadAttestationMissing]);
        };
        if attestation.key_id != trusted_key_id {
            return Err(vec![ScienceReturnIssue::PayloadAttestationKeyIdMismatch {
                declared: attestation.key_id.clone(),
                trusted: trusted_key_id.to_string(),
            }]);
        }
        if !is_canonical_lower_hex(&attestation.signature_hex, 128) {
            return Err(vec![ScienceReturnIssue::MalformedPayloadAttestationSignature]);
        }

        let acquired_event = self.events
            .iter()
            .find(|event| event.stage == ScienceReturnStage::Acquired)
            .ok_or_else(|| vec![ScienceReturnIssue::PayloadAcquisitionEventMissing])?;
        if acquired_event.actor_id != trusted_actor_id {
            return Err(vec![ScienceReturnIssue::PayloadAttestationActorIdMismatch {
                acquired: acquired_event.actor_id.clone(),
                trusted: trusted_actor_id.to_string(),
            }]);
        }

        let public_key_array: [u8; 32] = trusted_public_key_bytes
            .try_into()
            .map_err(|_| vec![ScienceReturnIssue::InvalidPayloadAttestationPublicKey])?;
        let verifying_key = VerifyingKey::from_bytes(&public_key_array)
            .map_err(|_| vec![ScienceReturnIssue::InvalidPayloadAttestationPublicKey])?;
        let signature_bytes = decode_lower_hex(&attestation.signature_hex)
            .ok_or_else(|| vec![ScienceReturnIssue::MalformedPayloadAttestationSignature])?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|_| vec![ScienceReturnIssue::InvalidPayloadAttestationSignature])?;
        let message = payload_attestation_message(
            &self.site_id,
            &self.observation_id,
            acquired_event,
            payload,
            &attestation.key_id,
        );
        verifying_key
            .verify_strict(&message, &signature)
            .map_err(|_| vec![ScienceReturnIssue::PayloadAttestationSignatureInvalid])
    }

    /// Check that this record refers to a known observation in the source site
    /// record. This checks identifier linkage only; it does not authenticate
    /// the observation or any event evidence.
    pub fn validate_for_site(
        &self,
        site: &SiteEvidenceRecord,
    ) -> Result<(), Vec<ScienceReturnIssue>> {
        let mut issues = self.validate().err().unwrap_or_default();

        if self.site_id != site.site_id {
            issues.push(ScienceReturnIssue::SiteIdMismatch {
                record: self.site_id.clone(),
                source: site.site_id.clone(),
            });
        }
        if !site
            .observations
            .iter()
            .any(|observation| observation.observation_id == self.observation_id)
        {
            issues.push(ScienceReturnIssue::ObservationNotInSite(
                self.observation_id.clone(),
            ));
        }
        if let Err(site_issues) = site.validate() {
            issues.push(ScienceReturnIssue::InvalidSiteEvidence(site_issues));
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    pub fn last_stage(&self) -> Option<ScienceReturnStage> {
        self.events.last().map(|event| event.stage)
    }

    /// True only after a delivery event exists. Store-and-forward is not
    /// classified as delivery.
    pub fn evidence_delivered(&self) -> bool {
        self.events.iter().any(|event| {
            matches!(
                event.stage,
                ScienceReturnStage::Delivered | ScienceReturnStage::Admitted
            )
        })
    }

    /// True only when an external scientific-admission reference is recorded.
    pub fn scientifically_admitted(&self) -> bool {
        self.last_stage() == Some(ScienceReturnStage::Admitted)
    }

    pub fn awaiting_store_and_forward_retrieval(&self) -> bool {
        self.last_stage() == Some(ScienceReturnStage::StoreAndForwarded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(
        sequence: u64,
        stage: ScienceReturnStage,
        event_id: &str,
    ) -> ScienceReturnEvent {
        ScienceReturnEvent {
            sequence,
            event_id: event_id.into(),
            stage,
            recorded_at: format!("2026-11-16T12:{:02}:00Z", sequence),
            clock_id: "synthetic-robot-clock-01".into(),
            clock_uncertainty_ms: MIN_CLOCK_UNCERTAINTY_MS,
            actor_id: "synthetic-robot-01".into(),
            evidence_ref: format!("synthetic-evidence-{event_id}"),
            calibration_ref: None,
            custody_ref: None,
            delivery_receipt_ref: None,
            admission_ref: None,
            disposition_reason: None,
        }
    }

    fn complete_record() -> ScienceReturnRecord {
        let mut events = vec![
            event(1, ScienceReturnStage::Proposed, "proposed"),
            event(2, ScienceReturnStage::Executed, "executed"),
            event(3, ScienceReturnStage::Acquired, "acquired"),
            event(4, ScienceReturnStage::IntegrityVerified, "verified"),
            event(5, ScienceReturnStage::Persisted, "persisted"),
            event(6, ScienceReturnStage::Delivered, "delivered"),
            event(7, ScienceReturnStage::Admitted, "admitted"),
        ];
        events[3].calibration_ref = Some("synthetic-calibration-record".into());
        events[4].custody_ref = Some("synthetic-storage-receipt".into());
        events[5].custody_ref = Some("synthetic-delivery-custodian".into());
        events[5].delivery_receipt_ref = Some("synthetic-delivery-receipt".into());
        events[6].custody_ref = Some("synthetic-science-custody".into());
        events[6].delivery_receipt_ref = Some("synthetic-delivery-receipt".into());
        events[6].admission_ref = Some("synthetic-science-admission".into());

        ScienceReturnRecord {
            site_id: "synthetic-mars-subterranean-candidate-01".into(),
            observation_id: "synthetic-optical-001".into(),
            payload: Some(ScienceReturnPayload {
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
                byte_length: 3,
                attestation: None,
            }),
            events,
        }
    }

    #[test]
    fn complete_chain_requires_delivery_before_scientific_admission() {
        let record = complete_record();
        assert!(record.validate().is_ok());
        assert!(record.evidence_delivered());
        assert!(record.scientifically_admitted());
        assert!(!record.awaiting_store_and_forward_retrieval());
    }

    #[test]
    fn local_persistence_is_not_delivery() {
        let mut record = complete_record();
        record.events.truncate(5);
        assert!(record.validate().is_ok());
        assert!(!record.evidence_delivered());
        assert!(!record.scientifically_admitted());
    }

    #[test]
    fn store_and_forward_is_pending_not_delivered() {
        let mut record = complete_record();
        record.events.truncate(5);
        let mut event = event(6, ScienceReturnStage::StoreAndForwarded, "store-forward");
        event.custody_ref = Some("local-buffer-custody".into());
        record.events.push(event);

        assert!(record.validate().is_ok());
        assert!(!record.evidence_delivered());
        assert!(record.awaiting_store_and_forward_retrieval());
    }

    #[test]
    fn skipped_integrity_stage_is_rejected() {
        let mut record = complete_record();
        record.events.remove(3);
        for (index, event) in record.events.iter_mut().enumerate() {
            event.sequence = index as u64 + 1;
        }
        assert!(record.validate().unwrap_err().iter().any(|issue| matches!(
            issue,
            ScienceReturnIssue::InvalidTransition {
                from: ScienceReturnStage::Acquired,
                to: ScienceReturnStage::Persisted
            }
        )));
    }

    #[test]
    fn delivered_stage_requires_custody_and_delivery_receipt() {
        let mut record = complete_record();
        record.events[5].custody_ref = None;
        record.events[5].delivery_receipt_ref = None;
        let issues = record.validate().unwrap_err();
        assert!(issues.contains(&ScienceReturnIssue::CustodyReferenceMissing(
            "delivered".into()
        )));
        assert!(issues.contains(&ScienceReturnIssue::DeliveryReceiptMissing(
            "delivered".into()
        )));
    }

    #[test]
    fn integrity_verification_requires_calibration_reference() {
        let mut record = complete_record();
        record.events[3].calibration_ref = None;
        assert!(record.validate().unwrap_err().contains(
            &ScienceReturnIssue::IntegrityVerificationMissingCalibration(
                "verified".into()
            )
        ));
    }

    #[test]
    fn admission_cannot_precede_delivery() {
        let mut record = complete_record();
        record.events.remove(5);
        for (index, event) in record.events.iter_mut().enumerate() {
            event.sequence = index as u64 + 1;
        }
        assert!(record.validate().unwrap_err().iter().any(|issue| matches!(
            issue,
            ScienceReturnIssue::InvalidTransition {
                from: ScienceReturnStage::Persisted,
                to: ScienceReturnStage::Admitted
            }
        )));
    }

    #[test]
    fn failure_requires_reason_and_is_terminal() {
        let mut record = complete_record();
        record.events.truncate(3);
        let failed = event(4, ScienceReturnStage::Failed, "failed");
        record.events.push(failed);
        assert!(record.validate().unwrap_err().contains(
            &ScienceReturnIssue::TerminalDispositionMissing(ScienceReturnStage::Failed)
        ));
        record.events[3].disposition_reason = Some("robot immobilized".into());
        assert!(record.validate().is_ok());
    }

    #[test]
    fn terminal_failure_cannot_be_followed_by_success() {
        let mut record = complete_record();
        record.events.truncate(3);
        let mut failed = event(4, ScienceReturnStage::Failed, "failed");
        failed.disposition_reason = Some("mission aborted".into());
        record.events.push(failed);
        record.events.push(event(5, ScienceReturnStage::Acquired, "late-acquired"));

        assert!(record.validate().unwrap_err().iter().any(|issue| matches!(
            issue,
            ScienceReturnIssue::TerminalEventNotLast(ScienceReturnStage::Failed)
        )));
    }

    #[test]
    fn site_and_observation_linkage_are_checked_without_authenticating_refs() {
        let mut site: SiteEvidenceRecord = serde_json::from_str(include_str!(
            "../fixtures/site-evidence-synthetic.json"
        ))
        .unwrap();
        let record = complete_record();
        assert!(record.validate_for_site(&site).is_ok());
        site.site_id = "different-site".into();
        assert!(record.validate_for_site(&site).unwrap_err().iter().any(
            |issue| matches!(issue, ScienceReturnIssue::SiteIdMismatch { .. })
        ));
    }
    #[test]
    fn malformed_or_impossible_utc_timestamps_are_rejected() {
        let mut record = complete_record();
        record.events[2].recorded_at = "2026-11-16 12:03:00".into();
        assert!(record.validate().unwrap_err().contains(
            &ScienceReturnIssue::InvalidRecordedAt("acquired".into())
        ));

        let mut record = complete_record();
        record.events[2].recorded_at = "2026-02-30T12:03:00Z".into();
        assert!(record.validate().unwrap_err().contains(
            &ScienceReturnIssue::InvalidRecordedAt("acquired".into())
        ));
    }

    #[test]
    fn certain_same_clock_regression_is_rejected() {
        let mut record = complete_record();
        record.events[4].recorded_at = "2026-11-16T12:02:00Z".into();
        let issues = record.validate().unwrap_err();
        assert!(issues.contains(&ScienceReturnIssue::TimestampRegression {
            clock_id: "synthetic-robot-clock-01".into(),
            previous_event_id: "verified".into(),
            current_event_id: "persisted".into()
        }));
    }

    #[test]
    fn overlapping_clock_uncertainty_does_not_claim_a_certain_regression() {
        let mut record = complete_record();
        record.events[4].recorded_at = "2026-11-16T12:03:59Z".into();
        record.events[3].clock_uncertainty_ms = 1_000;
        record.events[4].clock_uncertainty_ms = 1_000;
        assert!(record.validate().is_ok());
    }

    #[test]
    fn timestamps_from_distinct_clocks_are_not_naively_compared() {
        let mut record = complete_record();
        record.events[5].clock_id = "ground-station-clock-01".into();
        record.events[5].recorded_at = "2026-11-16T08:00:00Z".into();
        assert!(record.validate().is_ok());
    }

    #[test]
    fn event_clock_identity_is_required() {
        let mut record = complete_record();
        record.events[1].clock_id = " ".into();
        assert!(record.validate().unwrap_err().contains(
            &ScienceReturnIssue::EmptyClockId("executed".into())
        ));
    }

    #[test]
    fn utc_timestamp_parser_handles_leap_day_and_rejects_non_leap_day() {
        assert!(parse_canonical_utc_millis("2024-02-29T12:00:00Z").is_some());
        assert!(parse_canonical_utc_millis("2025-02-29T12:00:00Z").is_none());
        assert!(parse_canonical_utc_millis("2026-11-16T24:00:00Z").is_none());
        assert!(parse_canonical_utc_millis("2026-11-16T12:60:00Z").is_none());
    }

    #[test]
    fn timestamp_equality_is_allowed_when_event_sequence_disambiguates() {
        let mut record = complete_record();
        record.events[1].recorded_at = record.events[0].recorded_at.clone();
        assert!(record.validate().is_ok());
    }

    #[test]
    fn empty_clock_and_bad_timestamp_are_reported_independently() {
        let mut record = complete_record();
        record.events[0].clock_id.clear();
        record.events[0].recorded_at.clear();
        let issues = record.validate().unwrap_err();
        assert!(issues.contains(&ScienceReturnIssue::EmptyRecordedAt("proposed".into())));
        assert!(!issues.contains(&ScienceReturnIssue::EmptyClockId("proposed".into())));
    }

    #[test]
    fn declared_clock_uncertainty_must_cover_second_precision_quantization() {
        let mut record = complete_record();
        record.events[0].clock_uncertainty_ms = MIN_CLOCK_UNCERTAINTY_MS - 1;
        assert!(record.validate().unwrap_err().contains(
            &ScienceReturnIssue::ClockUncertaintyBelowResolution {
                event_id: "proposed".into(),
                observed_ms: MIN_CLOCK_UNCERTAINTY_MS - 1,
                minimum_ms: MIN_CLOCK_UNCERTAINTY_MS,
            }
        ));
    }

    #[test]
    fn sha256_payload_digest_matches_actual_acquired_bytes() {
        let record = complete_record();
        assert!(record.verify_payload_bytes(b"abc").is_ok());
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn altered_payload_bytes_are_rejected() {
        let record = complete_record();
        assert!(record
            .verify_payload_bytes(b"abd")
            .unwrap_err()
            .contains(&ScienceReturnIssue::PayloadSha256Mismatch));
    }

    #[test]
    fn payload_byte_length_is_checked_separately_from_digest() {
        let record = complete_record();
        assert!(record
            .verify_payload_bytes(b"abcd")
            .unwrap_err()
            .contains(&ScienceReturnIssue::PayloadByteLengthMismatch {
                expected: 3,
                actual: 4
            }));
    }

    #[test]
    fn acquired_observation_requires_content_addressed_payload_metadata() {
        let mut record = complete_record();
        record.payload = None;
        assert!(record
            .validate()
            .unwrap_err()
            .contains(&ScienceReturnIssue::PayloadMissingAfterAcquisition));
    }

    #[test]
    fn content_addressed_payload_cannot_precede_acquisition() {
        let mut record = complete_record();
        record.events.truncate(2);
        assert!(record
            .validate()
            .unwrap_err()
            .contains(&ScienceReturnIssue::PayloadPresentBeforeAcquisition));
    }

    #[test]
    fn malformed_or_noncanonical_payload_digest_is_rejected() {
        let mut record = complete_record();
        record.payload.as_mut().unwrap().sha256 = "ABC".into();
        assert!(record
            .validate()
            .unwrap_err()
            .contains(&ScienceReturnIssue::MalformedPayloadSha256));
    }

    #[test]
    fn payload_verification_cannot_succeed_without_a_declared_payload() {
        let mut record = complete_record();
        record.events.truncate(2);
        record.payload = None;
        assert!(record
            .verify_payload_bytes(b"abc")
            .unwrap_err()
            .contains(&ScienceReturnIssue::PayloadNotDeclaredForVerification));
    }

    fn signed_record_and_public_key() -> (ScienceReturnRecord, [u8; 32]) {
        use ed25519_dalek::{Signer, SigningKey};

        let mut record = complete_record();
        let key_id = "trusted-sensor-acquisition-key-01";
        let acquired_event = record.events
            .iter()
            .find(|event| event.stage == ScienceReturnStage::Acquired)
            .unwrap()
            .clone();
        let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
        let payload = record.payload.as_ref().unwrap();
        let message = payload_attestation_message(
            &record.site_id,
            &record.observation_id,
            &acquired_event,
            payload,
            key_id,
        );
        let signature = signing_key.sign(&message);
        record.payload.as_mut().unwrap().attestation = Some(PayloadAttestation {
            algorithm: PayloadSignatureAlgorithm::Ed25519V1,
            key_id: key_id.into(),
            signature_hex: encode_lower_hex(&signature.to_bytes()),
        });
        (record, signing_key.verifying_key().to_bytes())
    }

    const TRUSTED_ACTOR_ID: &str = "synthetic-robot-01";
    const TRUSTED_KEY_ID: &str = "trusted-sensor-acquisition-key-01";

    #[test]
    fn signed_payload_verifies_against_the_declared_trusted_key() {
        let (record, public_key) = signed_record_and_public_key();
        assert!(record
            .verify_signed_payload_bytes(
                b"abc",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &public_key
            )
            .is_ok());
    }

    #[test]
    fn signature_rejects_a_different_public_key() {
        use ed25519_dalek::SigningKey;
        let (record, _) = signed_record_and_public_key();
        let other_key = SigningKey::from_bytes(&[8_u8; 32]).verifying_key().to_bytes();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abc",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &other_key
            ),
            Err(vec![ScienceReturnIssue::PayloadAttestationSignatureInvalid])
        );
    }

    #[test]
    fn signed_payload_requires_the_exact_trusted_key_id() {
        let (record, public_key) = signed_record_and_public_key();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abc",
                TRUSTED_ACTOR_ID,
                "another-key",
                &public_key
            ),
            Err(vec![ScienceReturnIssue::PayloadAttestationKeyIdMismatch {
                declared: TRUSTED_KEY_ID.into(),
                trusted: "another-key".into()
            }])
        );
    }

    #[test]
    fn signed_payload_requires_the_exact_trusted_actor_id() {
        let (record, public_key) = signed_record_and_public_key();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abc",
                "another-robot",
                TRUSTED_KEY_ID,
                &public_key
            ),
            Err(vec![ScienceReturnIssue::PayloadAttestationActorIdMismatch {
                acquired: TRUSTED_ACTOR_ID.into(),
                trusted: "another-robot".into()
            }])
        );
    }

    #[test]
    fn empty_trusted_actor_or_key_ids_fail_closed() {
        let (record, public_key) = signed_record_and_public_key();
        assert_eq!(
            record.verify_signed_payload_bytes(b"abc", " ", TRUSTED_KEY_ID, &public_key),
            Err(vec![ScienceReturnIssue::EmptyTrustedAttestationActorId])
        );
        assert_eq!(
            record.verify_signed_payload_bytes(b"abc", TRUSTED_ACTOR_ID, " ", &public_key),
            Err(vec![ScienceReturnIssue::EmptyTrustedAttestationKeyId])
        );
    }

    #[test]
    fn altered_signed_payload_fails_before_signature_acceptance() {
        let (record, public_key) = signed_record_and_public_key();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abd",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &public_key
            ),
            Err(vec![ScienceReturnIssue::PayloadSha256Mismatch])
        );
    }

    #[test]
    fn attestation_cannot_be_replayed_to_another_site_or_observation() {
        let (mut record, public_key) = signed_record_and_public_key();
        record.site_id = "another-site".into();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abc",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &public_key
            ),
            Err(vec![ScienceReturnIssue::PayloadAttestationSignatureInvalid])
        );
        record.site_id = "synthetic-mars-subterranean-candidate-01".into();
        record.observation_id = "another-observation".into();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abc",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &public_key
            ),
            Err(vec![ScienceReturnIssue::PayloadAttestationSignatureInvalid])
        );
    }

    #[test]
    fn acquired_event_identity_is_signed() {
        let (mut record, public_key) = signed_record_and_public_key();
        record.events[2].event_id = "acquired-substituted".into();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abc",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &public_key
            ),
            Err(vec![ScienceReturnIssue::PayloadAttestationSignatureInvalid])
        );
        record.events[2].event_id = "acquired".into();
        record.events[2].evidence_ref = "substituted-evidence-ref".into();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abc",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &public_key
            ),
            Err(vec![ScienceReturnIssue::PayloadAttestationSignatureInvalid])
        );
    }

    #[test]
    fn malformed_or_noncanonical_attestation_signature_is_rejected() {
        let (mut record, public_key) = signed_record_and_public_key();
        record.payload.as_mut().unwrap().attestation.as_mut().unwrap().signature_hex =
            "ABC".into();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abc",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &public_key
            ),
            Err(vec![ScienceReturnIssue::MalformedPayloadAttestationSignature])
        );
    }

    #[test]
    fn missing_attestation_cannot_pass_the_signed_verification_path() {
        let record = complete_record();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abc",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &[1_u8; 32]
            ),
            Err(vec![ScienceReturnIssue::PayloadAttestationMissing])
        );
    }

    #[test]
    fn malformed_trusted_public_key_is_rejected() {
        let (record, _) = signed_record_and_public_key();
        assert_eq!(
            record.verify_signed_payload_bytes(
                b"abc",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &[1_u8; 31]
            ),
            Err(vec![ScienceReturnIssue::InvalidPayloadAttestationPublicKey])
        );
    }

    #[test]
    fn signature_is_bound_to_payload_length_as_well_as_the_digest() {
        let (mut record, public_key) = signed_record_and_public_key();
        let payload = record.payload.as_mut().unwrap();
        payload.byte_length = 4;
        payload.sha256 = sha256_hex(b"abcd");
        assert!(record
            .verify_signed_payload_bytes(
                b"abcd",
                TRUSTED_ACTOR_ID,
                TRUSTED_KEY_ID,
                &public_key
            )
            .unwrap_err()
            .contains(&ScienceReturnIssue::PayloadAttestationSignatureInvalid));
    }

    #[test]
    fn unknown_signature_suites_fail_deserialization() {
        let invalid = br#"{"algorithm":"rsa_sha256","key_id":"key-1","signature_hex":"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"}"#;
        assert!(serde_json::from_slice::<PayloadAttestation>(invalid).is_err());
    }

}
