// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Ed25519 attestations for policy-pinned memory events.
//!
//! Signatures authenticate which registered key signed exact canonical bytes;
//! they do not establish that a signed statement is factually true. Production
//! ingestion must anchor recorded_at to a trusted append log and provision the
//! trust registry securely. This module is not wired into the JSONL worker.

use crate::memory_governance::{AuthorityPolicy, LedgerEvent, PolicyPinnedLedger};
use chrono::{DateTime, SecondsFormat, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const ATTESTATION_SCHEMA_V1: &str = "BROCA-ATTESTATION-V1";
const POLICY_DOMAIN: &str = "LUMINOUS-DYNAMICS/BROCA/AUTHORITY-POLICY/V1";
const POLICY_INGESTION_DOMAIN: &str = "LUMINOUS-DYNAMICS/BROCA/POLICY-INGESTION-RECEIPT/V1";
const EVENT_DOMAIN: &str = "LUMINOUS-DYNAMICS/BROCA/LEDGER-EVENT/V1";
const INGESTION_DOMAIN: &str = "LUMINOUS-DYNAMICS/BROCA/INGESTION-RECEIPT/V1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedAuthorityPolicy {
    pub schema_version: String,
    pub signer_key_id: String,
    /// Must equal policy.recorded_at after canonical timestamp normalization.
    pub signed_at: String,
    pub policy: AuthorityPolicy,
    pub signature_hex: String,
    /// Trusted-ingress receipt; policy timestamps are not trusted without it.
    pub ingestion_receipt: Option<PolicyIngestionReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyIngestionReceipt {
    pub schema_version: String,
    pub ingester_key_id: String,
    /// Trusted receiver's transaction time, not supplied by the policy signer.
    pub accepted_at: String,
    pub signature_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedLedgerEvent {
    pub schema_version: String,
    /// Event-source key; must be bound to event.source in the trusted registry.
    pub signer_key_id: String,
    pub event: LedgerEvent,
    /// Signature over the source-authored event payload, including its claimed timestamp.
    pub signature_hex: String,
    /// Trusted-ingress receipt. Without this receipt the event cannot enter a verified batch.
    pub ingestion_receipt: Option<IngestionReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngestionReceipt {
    pub schema_version: String,
    pub ingester_key_id: String,
    /// Trusted receiver's transaction time, not supplied by the event source.
    pub accepted_at: String,
    pub signature_hex: String,
}

/// Explicitly provisioned trusted key. Do not populate this registry from
/// untrusted event payloads or from the unsigned policy it is intended to verify.
#[derive(Debug, Clone)]
pub struct TrustedSignerConfig {
    pub key_id: String,
    pub verifying_key_hex: String,
    /// Required for event signatures and must equal the event's source.
    pub source: Option<String>,
    pub can_sign_policies: bool,
    pub can_sign_events: bool,
    pub can_sign_ingestion_receipts: bool,
    pub valid_from: String,
    /// Exclusive end; None means no scheduled end.
    pub valid_until: Option<String>,
}

#[derive(Debug, Clone)]
struct TrustedSigner {
    source: Option<String>,
    can_sign_policies: bool,
    can_sign_events: bool,
    can_sign_ingestion_receipts: bool,
    valid_from: DateTime<Utc>,
    valid_until: Option<DateTime<Utc>>,
    verifying_key: VerifyingKey,
}

/// A batch that has passed signature, signer-role, key-window and identity
/// checks. Its fields are private so external callers cannot forge this type.
#[derive(Debug, Clone)]
pub struct VerifiedLedgerBatch {
    pub(crate) policies: Vec<AuthorityPolicy>,
    pub(crate) events: Vec<LedgerEvent>,
    pub(crate) policies_sha256: String,
    pub(crate) events_sha256: String,
}

impl VerifiedLedgerBatch {
    pub fn policy_chain_sha256(&self) -> &str {
        &self.policies_sha256
    }

    pub fn event_chain_sha256(&self) -> &str {
        &self.events_sha256
    }

    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn into_ledger(self) -> Result<PolicyPinnedLedger, String> {
        PolicyPinnedLedger::from_verified(self)
    }
}

#[derive(Debug, Clone)]
pub struct EvidenceTrustStore {
    signers: BTreeMap<String, TrustedSigner>,
}

#[derive(Serialize)]
struct PolicySigningPayload {
    domain: &'static str,
    schema_version: &'static str,
    signer_key_id: String,
    signed_at: String,
    policy: AuthorityPolicy,
}

#[derive(Serialize)]
struct EventSigningPayload {
    domain: &'static str,
    schema_version: &'static str,
    signer_key_id: String,
    event: LedgerEvent,
}

#[derive(Serialize)]
struct PolicyIngestionSigningPayload {
    domain: &'static str,
    schema_version: &'static str,
    ingester_key_id: String,
    accepted_at: String,
    policy_version: u64,
    policy_signer_key_id: String,
    policy_signature_sha256: String,
}

#[derive(Serialize)]
struct IngestionSigningPayload {
    domain: &'static str,
    schema_version: &'static str,
    ingester_key_id: String,
    accepted_at: String,
    event_id: String,
    source_signer_key_id: String,
    source_signature_sha256: String,
}

fn normalize_time(value: &str, field: &str) -> Result<String, String> {
    let parsed = DateTime::parse_from_rfc3339(value)
        .map_err(|error| format!("{field} must be RFC3339 with UTC offset: {error}"))?
        .with_timezone(&Utc);
    Ok(parsed.to_rfc3339_opts(SecondsFormat::AutoSi, true))
}

fn policy_for_signing(policy: &AuthorityPolicy) -> Result<AuthorityPolicy, String> {
    let mut normalized = policy.clone();
    normalized.recorded_at = normalize_time(&normalized.recorded_at, "policy.recorded_at")?;
    normalized.sources.sort_by(|left, right| left.source.cmp(&right.source));
    for source in &normalized.sources {
        if source.source.trim().is_empty() {
            return Err("policy source must not be empty".into());
        }
        if source.attributes.is_empty()
            || source.attributes.iter().any(|attribute| attribute.trim().is_empty())
        {
            return Err(format!("source {} has invalid attribute scope", source.source));
        }
    }
    Ok(normalized)
}

fn event_for_signing(event: &LedgerEvent) -> Result<LedgerEvent, String> {
    let mut normalized = event.clone();
    match &mut normalized {
        LedgerEvent::Assert {
            event_id,
            subject,
            attribute,
            value,
            valid_from,
            valid_to,
            recorded_at,
            source,
        } => {
            for (field, value) in [
                ("event_id", event_id),
                ("subject", subject),
                ("attribute", attribute),
                ("value", value),
                ("source", source),
            ] {
                if value.trim().is_empty() {
                    return Err(format!("event {field} must not be empty"));
                }
            }
            chrono::NaiveDate::parse_from_str(valid_from, "%Y-%m-%d")
                .map_err(|error| format!("event valid_from must be YYYY-MM-DD: {error}"))?;
            if let Some(end) = valid_to {
                let end_date = chrono::NaiveDate::parse_from_str(end, "%Y-%m-%d")
                    .map_err(|error| format!("event valid_to must be YYYY-MM-DD: {error}"))?;
                let start_date = chrono::NaiveDate::parse_from_str(valid_from, "%Y-%m-%d")
                    .map_err(|error| format!("event valid_from must be YYYY-MM-DD: {error}"))?;
                if end_date <= start_date {
                    return Err("event valid_to must be after valid_from".into());
                }
            }
            *recorded_at = normalize_time(recorded_at, "event.recorded_at")?;
        }
        LedgerEvent::Retract {
            event_id,
            target_event_id,
            recorded_at,
            source,
        } => {
            for (field, value) in [
                ("event_id", event_id),
                ("target_event_id", target_event_id),
                ("source", source),
            ] {
                if value.trim().is_empty() {
                    return Err(format!("retraction {field} must not be empty"));
                }
            }
            *recorded_at = normalize_time(recorded_at, "event.recorded_at")?;
        }
    }
    Ok(normalized)
}

fn policy_payload_bytes(
    schema_version: &str,
    signer_key_id: &str,
    signed_at: &str,
    policy: &AuthorityPolicy,
) -> Result<Vec<u8>, String> {
    if schema_version != ATTESTATION_SCHEMA_V1 {
        return Err("unsupported policy attestation schema".into());
    }
    let signed_at = normalize_time(signed_at, "signed_at")?;
    let policy = policy_for_signing(policy)?;
    if signed_at != policy.recorded_at {
        return Err("policy signed_at must equal policy.recorded_at".into());
    }
    serde_json::to_vec(&PolicySigningPayload {
        domain: POLICY_DOMAIN,
        schema_version: ATTESTATION_SCHEMA_V1,
        signer_key_id: signer_key_id.to_owned(),
        signed_at,
        policy,
    })
    .map_err(|error| format!("cannot canonicalize policy payload: {error}"))
}

fn event_payload_bytes(
    schema_version: &str,
    signer_key_id: &str,
    event: &LedgerEvent,
) -> Result<Vec<u8>, String> {
    if schema_version != ATTESTATION_SCHEMA_V1 {
        return Err("unsupported event attestation schema".into());
    }
    let event = event_for_signing(event)?;
    serde_json::to_vec(&EventSigningPayload {
        domain: EVENT_DOMAIN,
        schema_version: ATTESTATION_SCHEMA_V1,
        signer_key_id: signer_key_id.to_owned(),
        event,
    })
    .map_err(|error| format!("cannot canonicalize event payload: {error}"))
}

impl SignedAuthorityPolicy {
    pub fn sign(
        signer_key_id: impl Into<String>,
        policy: AuthorityPolicy,
        signing_key: &SigningKey,
    ) -> Result<Self, String> {
        let signer_key_id = signer_key_id.into();
        let signed_at = normalize_time(&policy.recorded_at, "policy.recorded_at")?;
        let schema_version = ATTESTATION_SCHEMA_V1.to_owned();
        let bytes = policy_payload_bytes(
            &schema_version,
            &signer_key_id,
            &signed_at,
            &policy,
        )?;
        let signature = signing_key.sign(&bytes);
        Ok(Self {
            schema_version,
            signer_key_id,
            signed_at,
            policy,
            signature_hex: hex::encode(signature.to_bytes()),
            ingestion_receipt: None,
        })
    }

    /// Attach a receipt from a separately provisioned trusted ingress signer.
    /// The receipt time, not the policy signer's claimed recorded_at, controls
    /// when this policy becomes part of the trusted policy timeline.
    pub fn with_ingestion_receipt(
        mut self,
        ingester_key_id: impl Into<String>,
        accepted_at: impl Into<String>,
        ingestion_signing_key: &SigningKey,
    ) -> Result<Self, String> {
        let ingester_key_id = ingester_key_id.into();
        let accepted_at = normalize_time(&accepted_at.into(), "policy receipt.accepted_at")?;
        let policy_signature = hex::decode(&self.signature_hex)
            .map_err(|error| format!("invalid policy signature hex: {error}"))?;
        let payload = PolicyIngestionSigningPayload {
            domain: POLICY_INGESTION_DOMAIN,
            schema_version: ATTESTATION_SCHEMA_V1,
            ingester_key_id: ingester_key_id.clone(),
            accepted_at: accepted_at.clone(),
            policy_version: self.policy.policy_version,
            policy_signer_key_id: self.signer_key_id.clone(),
            policy_signature_sha256: format!("{:x}", Sha256::digest(policy_signature)),
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|error| format!("cannot canonicalize policy ingestion receipt: {error}"))?;
        let signature = ingestion_signing_key.sign(&bytes);
        self.ingestion_receipt = Some(PolicyIngestionReceipt {
            schema_version: ATTESTATION_SCHEMA_V1.to_owned(),
            ingester_key_id,
            accepted_at,
            signature_hex: hex::encode(signature.to_bytes()),
        });
        Ok(self)
    }

    fn signing_bytes(&self) -> Result<Vec<u8>, String> {
        policy_payload_bytes(
            &self.schema_version,
            &self.signer_key_id,
            &self.signed_at,
            &self.policy,
        )
    }
}

impl SignedLedgerEvent {
    pub fn sign(
        signer_key_id: impl Into<String>,
        event: LedgerEvent,
        signing_key: &SigningKey,
    ) -> Result<Self, String> {
        let signer_key_id = signer_key_id.into();
        let schema_version = ATTESTATION_SCHEMA_V1.to_owned();
        let bytes = event_payload_bytes(&schema_version, &signer_key_id, &event)?;
        let signature = signing_key.sign(&bytes);
        Ok(Self {
            schema_version,
            signer_key_id,
            event,
            signature_hex: hex::encode(signature.to_bytes()),
            ingestion_receipt: None,
        })
    }

    /// Attach an ingress-signed receipt after the receiver accepts the signed source event.
    /// accepted_at is assigned by the trusted ingest service, not copied from event.recorded_at.
    pub fn with_ingestion_receipt(
        mut self,
        ingester_key_id: impl Into<String>,
        accepted_at: impl Into<String>,
        ingestion_signing_key: &SigningKey,
    ) -> Result<Self, String> {
        let ingester_key_id = ingester_key_id.into();
        let accepted_at = normalize_time(&accepted_at.into(), "receipt.accepted_at")?;
        let source_signature = hex::decode(&self.signature_hex)
            .map_err(|error| format!("invalid source signature hex: {error}"))?;
        let receipt_payload = IngestionSigningPayload {
            domain: INGESTION_DOMAIN,
            schema_version: ATTESTATION_SCHEMA_V1,
            ingester_key_id: ingester_key_id.clone(),
            accepted_at: accepted_at.clone(),
            event_id: self.event.event_id().to_owned(),
            source_signer_key_id: self.signer_key_id.clone(),
            source_signature_sha256: format!("{:x}", Sha256::digest(source_signature)),
        };
        let payload_bytes = serde_json::to_vec(&receipt_payload)
            .map_err(|error| format!("cannot canonicalize ingestion receipt: {error}"))?;
        let signature = ingestion_signing_key.sign(&payload_bytes);
        self.ingestion_receipt = Some(IngestionReceipt {
            schema_version: ATTESTATION_SCHEMA_V1.to_owned(),
            ingester_key_id,
            accepted_at,
            signature_hex: hex::encode(signature.to_bytes()),
        });
        Ok(self)
    }

    fn signing_bytes(&self) -> Result<Vec<u8>, String> {
        event_payload_bytes(&self.schema_version, &self.signer_key_id, &self.event)
    }
}

impl EvidenceTrustStore {
    pub fn new(configs: Vec<TrustedSignerConfig>) -> Result<Self, String> {
        if configs.is_empty() {
            return Err("trusted signer registry must not be empty".into());
        }
        let mut signers = BTreeMap::new();
        let mut verifying_key_material = BTreeSet::new();
        for config in configs {
            if config.key_id.trim().is_empty() {
                return Err("trusted key_id must not be empty".into());
            }
            let key_bytes = hex::decode(&config.verifying_key_hex)
                .map_err(|error| format!("invalid public key hex for {}: {error}", config.key_id))?;
            let key_array: [u8; 32] = key_bytes.try_into().map_err(|bytes: Vec<u8>| {
                format!(
                    "public key for {} must be 32 bytes, got {}",
                    config.key_id,
                    bytes.len()
                )
            })?;
            let verifying_key = VerifyingKey::from_bytes(&key_array)
                .map_err(|error| format!("invalid public key for {}: {error}", config.key_id))?;
            if !verifying_key_material.insert(key_array) {
                return Err(
                    "duplicate trusted public key material across key IDs; signing roles require distinct keypairs"
                        .into(),
                );
            }
            let valid_from = DateTime::parse_from_rfc3339(&config.valid_from)
                .map_err(|error| format!("valid_from must be RFC3339 with UTC offset: {error}"))?
                .with_timezone(&Utc);
            let valid_until = config
                .valid_until
                .as_deref()
                .map(|value| {
                    DateTime::parse_from_rfc3339(value)
                        .map(|parsed| parsed.with_timezone(&Utc))
                        .map_err(|error| format!("valid_until must be RFC3339: {error}"))
                })
                .transpose()?;
            if valid_until.is_some_and(|until| until <= valid_from) {
                return Err(format!("key {} has an empty validity window", config.key_id));
            }
            let role_count = [
                config.can_sign_policies,
                config.can_sign_events,
                config.can_sign_ingestion_receipts,
            ]
            .into_iter()
            .filter(|enabled| *enabled)
            .count();
            if role_count == 0 {
                return Err(format!("key {} has no signing role", config.key_id));
            }
            if role_count != 1 {
                return Err(format!(
                    "key {} must have exactly one signing role; ingress keys must be independent",
                    config.key_id
                ));
            }
            if config.can_sign_events
                && config.source.as_deref().is_none_or(str::is_empty)
            {
                return Err(format!("event-signing key {} must bind to a source", config.key_id));
            }
            let signer = TrustedSigner {
                source: config.source,
                can_sign_policies: config.can_sign_policies,
                can_sign_events: config.can_sign_events,
                can_sign_ingestion_receipts: config.can_sign_ingestion_receipts,
                valid_from,
                valid_until,
                verifying_key,
            };
            if signers.insert(config.key_id.clone(), signer).is_some() {
                return Err(format!("duplicate trusted key_id {:?}", config.key_id));
            }
        }
        Ok(Self { signers })
    }

    fn signer_at(&self, key_id: &str, timestamp: DateTime<Utc>) -> Result<&TrustedSigner, String> {
        let signer = self
            .signers
            .get(key_id)
            .ok_or_else(|| format!("untrusted signer key_id {key_id:?}"))?;
        if timestamp < signer.valid_from
            || signer.valid_until.as_ref().is_some_and(|until| &timestamp >= until)
        {
            return Err(format!("signer key {key_id:?} was not valid at the supplied verification time"));
        }
        Ok(signer)
    }

    fn verify_signature(
        signer: &TrustedSigner,
        signature_hex: &str,
        message: &[u8],
    ) -> Result<(), String> {
        let signature_bytes = hex::decode(signature_hex)
            .map_err(|error| format!("invalid signature hex: {error}"))?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|error| format!("invalid Ed25519 signature encoding: {error}"))?;
        signer
            .verifying_key
            .verify(message, &signature)
            .map_err(|_| "Ed25519 signature verification failed".into())
    }

    pub fn verify_policy(&self, envelope: &SignedAuthorityPolicy) -> Result<AuthorityPolicy, String> {
        let bytes = envelope.signing_bytes()?;
        let receipt = envelope
            .ingestion_receipt
            .as_ref()
            .ok_or_else(|| "policy lacks trusted-ingress receipt".to_string())?;
        if receipt.schema_version != ATTESTATION_SCHEMA_V1 {
            return Err("unsupported policy ingestion receipt schema".into());
        }
        let accepted_at = normalize_time(&receipt.accepted_at, "policy receipt.accepted_at")?;
        let accepted_time = parse_utc(&accepted_at, "policy receipt.accepted_at")?;

        // A policy author's signed_at is not a trusted clock. Evaluate key
        // validity at trusted ingress time so a retired policy key cannot
        // create a policy now and backdate it into its old validity window.
        let signer = self.signer_at(&envelope.signer_key_id, accepted_time.clone())?;
        if !signer.can_sign_policies {
            return Err(format!("key {:?} lacks policy-signing role", envelope.signer_key_id));
        }
        Self::verify_signature(signer, &envelope.signature_hex, &bytes)?;

        let ingester = self.signer_at(&receipt.ingester_key_id, accepted_time.clone())?;
        if !ingester.can_sign_ingestion_receipts {
            return Err(format!(
                "key {:?} lacks ingestion-receipt signing role",
                receipt.ingester_key_id
            ));
        }
        let policy_signature = hex::decode(&envelope.signature_hex)
            .map_err(|error| format!("invalid policy signature hex: {error}"))?;
        let receipt_payload = PolicyIngestionSigningPayload {
            domain: POLICY_INGESTION_DOMAIN,
            schema_version: ATTESTATION_SCHEMA_V1,
            ingester_key_id: receipt.ingester_key_id.clone(),
            accepted_at: accepted_at.clone(),
            policy_version: envelope.policy.policy_version,
            policy_signer_key_id: envelope.signer_key_id.clone(),
            policy_signature_sha256: format!("{:x}", Sha256::digest(policy_signature)),
        };
        let receipt_bytes = serde_json::to_vec(&receipt_payload)
            .map_err(|error| format!("cannot canonicalize policy receipt: {error}"))?;
        Self::verify_signature(ingester, &receipt.signature_hex, &receipt_bytes)?;

        let mut policy = policy_for_signing(&envelope.policy)?;
        // The timeline uses the trusted accepted time, never the self-asserted
        // timestamp from the policy's signed payload.
        policy.recorded_at = accepted_at;
        Ok(policy)
    }

    pub fn verify_event(&self, envelope: &SignedLedgerEvent) -> Result<LedgerEvent, String> {
        let bytes = envelope.signing_bytes()?;
        let mut event = event_for_signing(&envelope.event)?;
        let receipt = envelope
            .ingestion_receipt
            .as_ref()
            .ok_or_else(|| "event lacks trusted-ingress receipt".to_string())?;
        if receipt.schema_version != ATTESTATION_SCHEMA_V1 {
            return Err("unsupported ingestion receipt schema".into());
        }
        let accepted_at = normalize_time(&receipt.accepted_at, "receipt.accepted_at")?;
        let accepted_time = parse_utc(&accepted_at, "receipt.accepted_at")?;

        // The source key must be authorized at trusted acceptance time, not at
        // a timestamp the source chose to put inside its own payload.
        let source_signer = self.signer_at(&envelope.signer_key_id, accepted_time.clone())?;
        if !source_signer.can_sign_events {
            return Err(format!("key {:?} lacks event-signing role", envelope.signer_key_id));
        }
        if source_signer.source.as_deref() != Some(event.source()) {
            return Err(format!(
                "signer {:?} is not authorized for source {:?}",
                envelope.signer_key_id,
                event.source()
            ));
        }
        Self::verify_signature(source_signer, &envelope.signature_hex, &bytes)?;

        let ingester = self.signer_at(&receipt.ingester_key_id, accepted_time.clone())?;
        if !ingester.can_sign_ingestion_receipts {
            return Err(format!(
                "key {:?} lacks ingestion-receipt signing role",
                receipt.ingester_key_id
            ));
        }
        let source_signature = hex::decode(&envelope.signature_hex)
            .map_err(|error| format!("invalid source signature hex: {error}"))?;
        let receipt_payload = IngestionSigningPayload {
            domain: INGESTION_DOMAIN,
            schema_version: ATTESTATION_SCHEMA_V1,
            ingester_key_id: receipt.ingester_key_id.clone(),
            accepted_at: accepted_at.clone(),
            event_id: event.event_id().to_owned(),
            source_signer_key_id: envelope.signer_key_id.clone(),
            source_signature_sha256: format!("{:x}", Sha256::digest(source_signature)),
        };
        let receipt_bytes = serde_json::to_vec(&receipt_payload)
            .map_err(|error| format!("cannot canonicalize ingestion receipt: {error}"))?;
        Self::verify_signature(ingester, &receipt.signature_hex, &receipt_bytes)?;

        // Replace the source-claimed timestamp with the timestamp authenticated
        // by the trusted ingestion receipt before entering the policy ledger.
        match &mut event {
            LedgerEvent::Assert { recorded_at, .. }
            | LedgerEvent::Retract { recorded_at, .. } => *recorded_at = accepted_at,
        }
        Ok(event)
    }

    pub fn verify_batch(
        &self,
        signed_policies: &[SignedAuthorityPolicy],
        signed_events: &[SignedLedgerEvent],
    ) -> Result<VerifiedLedgerBatch, String> {
        if signed_policies.is_empty() {
            return Err("signed policy chain must not be empty".into());
        }
        if signed_events.is_empty() {
            return Err("signed event batch must not be empty".into());
        }
        let policies: Vec<AuthorityPolicy> = signed_policies
            .iter()
            .map(|envelope| self.verify_policy(envelope))
            .collect::<Result<_, _>>()?;
        let events: Vec<LedgerEvent> = signed_events
            .iter()
            .map(|envelope| self.verify_event(envelope))
            .collect::<Result<_, _>>()?;
        // Hash the verified signed envelopes (not just unsigned decoded
        // fields), in stable order, so the receipt digest binds signature and
        // trusted-ingress metadata as well as the accepted event contents.
        let mut canonical_signed_policies = signed_policies.to_vec();
        for envelope in &mut canonical_signed_policies {
            envelope.policy = policy_for_signing(&envelope.policy)?;
            envelope.signed_at = normalize_time(&envelope.signed_at, "policy signed_at")?;
            if let Some(receipt) = envelope.ingestion_receipt.as_mut() {
                receipt.accepted_at =
                    normalize_time(&receipt.accepted_at, "policy receipt.accepted_at")?;
            }
        }
        canonical_signed_policies.sort_by_key(|envelope| envelope.policy.policy_version);
        let policy_bytes = serde_json::to_vec(&canonical_signed_policies)
            .map_err(|error| format!("cannot serialize verified policy envelopes: {error}"))?;

        let mut canonical_signed_events = signed_events.to_vec();
        for envelope in &mut canonical_signed_events {
            envelope.event = event_for_signing(&envelope.event)?;
            if let Some(receipt) = envelope.ingestion_receipt.as_mut() {
                receipt.accepted_at = normalize_time(&receipt.accepted_at, "receipt.accepted_at")?;
            }
        }
        canonical_signed_events.sort_by(|left, right| left.event.event_id().cmp(right.event.event_id()));
        let event_bytes = serde_json::to_vec(&canonical_signed_events)
            .map_err(|error| format!("cannot serialize verified signed events: {error}"))?;
        Ok(VerifiedLedgerBatch {
            policies,
            events,
            policies_sha256: format!("{:x}", Sha256::digest(policy_bytes)),
            events_sha256: format!("{:x}", Sha256::digest(event_bytes)),
        })
    }
}

fn parse_utc(value: &str, field: &str) -> Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(value)
        .map(|parsed| parsed.with_timezone(&Utc))
        .map_err(|error| format!("{field} must be RFC3339 with UTC offset: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_governance::{LedgerQuery, SourcePolicy};

    fn policy(version: u64, timestamp: &str) -> AuthorityPolicy {
        AuthorityPolicy {
            policy_version: version,
            recorded_at: timestamp.into(),
            sources: vec![
                SourcePolicy {
                    source: "operations".into(),
                    authority_rank: 10,
                    attributes: ["endpoint".to_owned(), "notes".to_owned()].into_iter().collect(),
                    can_retract: true,
                },
                SourcePolicy {
                    source: "incident-review".into(),
                    authority_rank: 12,
                    attributes: ["endpoint".to_owned()].into_iter().collect(),
                    can_retract: false,
                },
            ],
        }
    }

    fn assertion(id: &str, source: &str, value: &str, claimed_recorded_at: &str) -> LedgerEvent {
        LedgerEvent::Assert {
            event_id: id.into(),
            subject: "db-prod".into(),
            attribute: "endpoint".into(),
            value: value.into(),
            valid_from: "2026-01-01".into(),
            valid_to: None,
            recorded_at: claimed_recorded_at.into(),
            source: source.into(),
        }
    }

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    fn trust_config(
        key_id: &str,
        source: Option<&str>,
        key: &SigningKey,
        can_sign_policies: bool,
        can_sign_events: bool,
        can_sign_ingestion_receipts: bool,
        from: &str,
        until: Option<&str>,
    ) -> TrustedSignerConfig {
        TrustedSignerConfig {
            key_id: key_id.into(),
            verifying_key_hex: hex::encode(key.verifying_key().to_bytes()),
            source: source.map(str::to_owned),
            can_sign_policies,
            can_sign_events,
            can_sign_ingestion_receipts,
            valid_from: from.into(),
            valid_until: until.map(str::to_owned),
        }
    }

    fn keys() -> (SigningKey, SigningKey, SigningKey) {
        (key(7), key(9), key(11))
    }

    fn store() -> (EvidenceTrustStore, SigningKey, SigningKey, SigningKey) {
        let (policy_key, ops_key, ingest_key) = keys();
        let store = EvidenceTrustStore::new(vec![
            trust_config(
                "policy-root", None, &policy_key, true, false, false,
                "2026-01-01T00:00:00Z", None,
            ),
            trust_config(
                "ops-key-1", Some("operations"), &ops_key, false, true, false,
                "2026-01-01T00:00:00Z", Some("2027-01-01T00:00:00Z"),
            ),
            trust_config(
                "ingest-root", None, &ingest_key, false, false, true,
                "2026-01-01T00:00:00Z", Some("2027-01-01T00:00:00Z"),
            ),
        ]).unwrap();
        (store, policy_key, ops_key, ingest_key)
    }

    fn sign_policy(
        signer_key_id: &str,
        policy: AuthorityPolicy,
        policy_key: &SigningKey,
        accepted_at: &str,
        ingestion_key: &SigningKey,
    ) -> SignedAuthorityPolicy {
        SignedAuthorityPolicy::sign(signer_key_id, policy, policy_key)
            .unwrap()
            .with_ingestion_receipt("ingest-root", accepted_at, ingestion_key)
            .unwrap()
    }

    fn sign_event(
        signer_key_id: &str,
        event: LedgerEvent,
        source_key: &SigningKey,
        accepted_at: &str,
        ingestion_key: &SigningKey,
    ) -> SignedLedgerEvent {
        SignedLedgerEvent::sign(signer_key_id, event, source_key)
            .unwrap()
            .with_ingestion_receipt("ingest-root", accepted_at, ingestion_key)
            .unwrap()
    }

    fn query(query_id: &str, known_at: &str) -> LedgerQuery {
        LedgerQuery {
            query_id: query_id.into(),
            subject: "db-prod".into(),
            attribute: "endpoint".into(),
            valid_at: "2026-01-03".into(),
            known_at: known_at.into(),
        }
    }

    #[test]
    fn valid_policy_and_event_signatures_verify() {
        let (store, policy_key, ops_key, ingest_key) = store();
        let signed_policy = sign_policy(
            "policy-root", policy(1, "2026-01-01T00:00:00Z"), &policy_key,
            "2026-01-01T00:00:01Z", &ingest_key,
        );
        assert_eq!(store.verify_policy(&signed_policy).unwrap().policy_version, 1);
        let signed_event = sign_event(
            "ops-key-1",
            assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:01Z", &ingest_key,
        );
        let verified = store.verify_event(&signed_event).unwrap();
        assert!(matches!(verified, LedgerEvent::Assert { .. }));
        assert_eq!(verified.recorded_at(), "2026-01-02T00:00:01Z");
    }

    #[test]
    fn signed_batches_produce_verified_ledger_constructor_input() {
        let (store, policy_key, ops_key, ingest_key) = store();
        let signed_policy = sign_policy(
            "policy-root", policy(1, "2026-01-01T00:00:00Z"), &policy_key,
            "2026-01-01T00:00:01Z", &ingest_key,
        );
        let signed_event = sign_event(
            "ops-key-1",
            assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:01Z", &ingest_key,
        );
        let batch = store.verify_batch(&[signed_policy], &[signed_event]).unwrap();
        assert_eq!(batch.policy_count(), 1);
        assert_eq!(batch.event_count(), 1);
        assert_eq!(batch.policy_chain_sha256().len(), 64);
        assert_eq!(batch.event_chain_sha256().len(), 64);
        let ledger = batch.into_ledger().unwrap();
        let result = ledger.resolve(&query("Q", "2026-01-03T00:00:00Z")).unwrap();
        assert_eq!(result.value.as_deref(), Some("A"));
    }

    #[test]
    fn trusted_ingress_time_not_source_claim_controls_temporal_visibility() {
        let (store, policy_key, ops_key, ingest_key) = store();
        let signed_policy = sign_policy(
            "policy-root", policy(1, "2026-01-01T00:00:00Z"), &policy_key,
            "2026-01-01T00:00:01Z", &ingest_key,
        );
        let signed_event = sign_event(
            "ops-key-1",
            assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-03-01T10:00:00Z", &ingest_key,
        );
        let batch = store.verify_batch(&[signed_policy], &[signed_event]).unwrap();
        let ledger = batch.into_ledger().unwrap();
        let before_receipt = ledger.resolve(&query("before", "2026-02-20T00:00:00Z")).unwrap();
        let after_receipt = ledger.resolve(&query("after", "2026-03-02T00:00:00Z")).unwrap();
        assert_eq!(before_receipt.state, crate::memory_governance::ResolutionState::Unknown);
        assert_eq!(after_receipt.value.as_deref(), Some("A"));
    }

    #[test]
    fn verified_digest_is_order_independent_and_binds_receipts() {
        let (store, policy_key, ops_key, ingest_key) = store();
        let signed_policy = sign_policy(
            "policy-root", policy(1, "2026-01-01T00:00:00Z"), &policy_key,
            "2026-01-01T00:00:01Z", &ingest_key,
        );
        let e1 = sign_event(
            "ops-key-1",
            assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:01Z", &ingest_key,
        );
        let e2 = sign_event(
            "ops-key-1",
            assertion("E2", "operations", "B", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:02Z", &ingest_key,
        );
        let left = store.verify_batch(&[signed_policy.clone()], &[e1.clone(), e2.clone()]).unwrap();
        let right = store.verify_batch(&[signed_policy.clone()], &[e2.clone(), e1.clone()]).unwrap();
        assert_eq!(left.event_chain_sha256(), right.event_chain_sha256());
        assert_eq!(left.policy_chain_sha256(), right.policy_chain_sha256());

        let changed_receipt = sign_event(
            "ops-key-1",
            assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:03Z", &ingest_key,
        );
        let changed = store.verify_batch(&[signed_policy], &[changed_receipt, e2]).unwrap();
        assert_ne!(left.event_chain_sha256(), changed.event_chain_sha256());
    }

    #[test]
    fn missing_ingestion_receipt_is_rejected() {
        let (store, _, ops_key, _) = store();
        let signed = SignedLedgerEvent::sign(
            "ops-key-1", assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"), &ops_key,
        ).unwrap();
        assert!(store.verify_event(&signed).unwrap_err().contains("lacks trusted-ingress receipt"));
    }

    #[test]
    fn source_payload_tampering_breaks_signature() {
        let (store, _, ops_key, ingest_key) = store();
        let mut signed = sign_event(
            "ops-key-1",
            assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:01Z", &ingest_key,
        );
        if let LedgerEvent::Assert { value, .. } = &mut signed.event {
            *value = "tampered".into();
        }
        assert!(store.verify_event(&signed).unwrap_err().contains("signature verification failed"));
    }

    #[test]
    fn receipt_time_tampering_breaks_ingestion_signature() {
        let (store, _, ops_key, ingest_key) = store();
        let mut signed = sign_event(
            "ops-key-1",
            assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:01Z", &ingest_key,
        );
        signed.ingestion_receipt.as_mut().unwrap().accepted_at = "2026-01-03T00:00:00Z".into();
        assert!(store.verify_event(&signed).unwrap_err().contains("signature verification failed"));
    }

    #[test]
    fn signer_cannot_claim_another_source() {
        let (store, _, ops_key, ingest_key) = store();
        let signed = sign_event(
            "ops-key-1",
            assertion("E1", "incident-review", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:01Z", &ingest_key,
        );
        assert!(store.verify_event(&signed).unwrap_err().contains("not authorized for source"));
    }

    #[test]
    fn unknown_source_key_is_rejected() {
        let (store, _, ops_key, ingest_key) = store();
        let signed = sign_event(
            "not-registered",
            assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:01Z", &ingest_key,
        );
        assert!(store.verify_event(&signed).unwrap_err().contains("untrusted signer"));
    }

    #[test]
    fn key_role_must_match_operation() {
        let (store, policy_key, _, ingest_key) = store();
        let signed = sign_event(
            "policy-root",
            assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &policy_key, "2026-01-02T00:00:01Z", &ingest_key,
        );
        assert!(store.verify_event(&signed).unwrap_err().contains("lacks event-signing role"));
    }

    #[test]
    fn policy_payload_tampering_breaks_signature() {
        let (store, policy_key, _, ingest_key) = store();
        let mut signed = sign_policy(
            "policy-root", policy(1, "2026-01-01T00:00:00Z"), &policy_key,
            "2026-01-01T00:00:01Z", &ingest_key,
        );
        signed.policy.sources[0].authority_rank += 100;
        assert!(store.verify_policy(&signed).unwrap_err().contains("signature verification failed"));
    }

    #[test]
    fn policy_signature_requires_recorded_time_match() {
        let (store, policy_key, _, ingest_key) = store();
        let mut signed = sign_policy(
            "policy-root", policy(1, "2026-01-01T00:00:00Z"), &policy_key,
            "2026-01-01T00:00:01Z", &ingest_key,
        );
        signed.signed_at = "2026-01-02T00:00:00Z".into();
        assert!(store.verify_policy(&signed).unwrap_err().contains("must equal policy.recorded_at"));
    }

    #[test]
    fn policy_source_order_is_canonicalized_for_signatures() {
        let (store, policy_key, _, ingest_key) = store();
        let mut policy = policy(1, "2026-01-01T00:00:00Z");
        let signed = sign_policy(
            "policy-root", policy.clone(), &policy_key, "2026-01-01T00:00:01Z", &ingest_key,
        );
        policy.sources.reverse();
        let mut reordered = signed.clone();
        reordered.policy = policy;
        assert_eq!(store.verify_policy(&signed).unwrap(), store.verify_policy(&reordered).unwrap());
    }

    #[test]
    fn policy_requires_a_trusted_ingress_receipt() {
        let (store, policy_key, _, _) = store();
        let signed = SignedAuthorityPolicy::sign(
            "policy-root", policy(1, "2026-01-01T00:00:00Z"), &policy_key,
        ).unwrap();
        assert!(store.verify_policy(&signed).unwrap_err().contains("lacks trusted-ingress receipt"));
    }

    #[test]
    fn policy_transaction_time_comes_from_trusted_ingress() {
        let (store, policy_key, _, ingest_key) = store();
        let signed = sign_policy(
            "policy-root", policy(1, "2026-01-01T00:00:00Z"), &policy_key,
            "2026-01-03T12:00:00Z", &ingest_key,
        );
        let verified = store.verify_policy(&signed).unwrap();
        assert_eq!(verified.recorded_at, "2026-01-03T12:00:00Z");
    }

    #[test]
    fn policy_receipt_time_tampering_breaks_ingestion_signature() {
        let (store, policy_key, _, ingest_key) = store();
        let mut signed = sign_policy(
            "policy-root", policy(1, "2026-01-01T00:00:00Z"), &policy_key,
            "2026-01-03T12:00:00Z", &ingest_key,
        );
        signed.ingestion_receipt.as_mut().unwrap().accepted_at = "2026-01-02T00:00:00Z".into();
        assert!(store.verify_policy(&signed).unwrap_err().contains("signature verification failed"));
    }

    #[test]
    fn policy_key_validity_is_checked_at_trusted_ingress_time() {
        let policy_key = key(7);
        let ingest_key = key(11);
        let store = EvidenceTrustStore::new(vec![
            trust_config("policy-root", None, &policy_key, true, false, false,
                "2026-01-01T00:00:00Z", Some("2026-02-01T00:00:00Z")),
            trust_config("ingest-root", None, &ingest_key, false, false, true,
                "2026-01-01T00:00:00Z", None),
        ]).unwrap();
        // The source-authored timestamp is backdated into the key's validity
        // window, but the independently signed receipt says it arrived later.
        let signed = sign_policy(
            "policy-root", policy(1, "2026-01-10T00:00:00Z"), &policy_key,
            "2026-02-02T00:00:00Z", &ingest_key,
        );
        assert!(store.verify_policy(&signed).unwrap_err().contains("not valid at the supplied verification time"));
    }

    #[test]
    fn ingress_keys_cannot_share_signing_roles() {
        let policy_key = key(7);
        let store = EvidenceTrustStore::new(vec![trust_config(
            "dual-role", None, &policy_key, true, false, true,
            "2026-01-01T00:00:00Z", None,
        )]);
        assert!(store.unwrap_err().contains("exactly one signing role"));
    }

    #[test]
    fn same_public_key_cannot_be_aliased_across_signing_roles() {
        let reused_key = key(7);
        let store = EvidenceTrustStore::new(vec![
            trust_config("policy-root", None, &reused_key, true, false, false,
                "2026-01-01T00:00:00Z", None),
            trust_config("ingest-root", None, &reused_key, false, false, true,
                "2026-01-01T00:00:00Z", None),
        ]);
        assert!(store.unwrap_err().contains("duplicate trusted public key material"));
    }

    #[test]
    fn event_key_validity_window_is_enforced_at_ingress_time() {
        let ops_key = key(9);
        let ingest_key = key(11);
        let store = EvidenceTrustStore::new(vec![
            trust_config("ops", Some("operations"), &ops_key, false, true, false,
                "2026-02-01T00:00:00Z", Some("2026-03-01T00:00:00Z")),
            trust_config("ingest-root", None, &ingest_key, false, false, true,
                "2026-01-01T00:00:00Z", None),
        ]).unwrap();
        let signed = sign_event(
            "ops", assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:01Z", &ingest_key,
        );
        assert!(store.verify_event(&signed).unwrap_err().contains("not valid at the supplied verification time"));
    }

    #[test]
    fn event_signature_cannot_be_replayed_with_another_signer_id() {
        let (store, _, ops_key, ingest_key) = store();
        let mut signed = sign_event(
            "ops-key-1",
            assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"),
            &ops_key, "2026-01-02T00:00:01Z", &ingest_key,
        );
        signed.signer_key_id = "different-key-id".into();
        assert!(store.verify_event(&signed).is_err());
    }

    #[test]
    fn policy_root_cannot_sign_ingestion_receipts() {
        let (store, _, ops_key, _) = store();
        let mut signed = SignedLedgerEvent::sign(
            "ops-key-1", assertion("E1", "operations", "A", "2026-01-02T00:00:00Z"), &ops_key,
        ).unwrap();
        let mut receipt = signed.clone().with_ingestion_receipt(
            "ops-key-1", "2026-01-02T00:00:01Z", &ops_key,
        ).unwrap().ingestion_receipt.unwrap();
        receipt.ingester_key_id = "policy-root".into();
        signed.ingestion_receipt = Some(receipt);
        assert!(store.verify_event(&signed).unwrap_err().contains("lacks ingestion-receipt signing role"));
    }

    #[test]
    fn strict_envelopes_reject_unknown_fields() {
        let event = r#"{"schema_version":"BROCA-ATTESTATION-V1","signer_key_id":"ops-key-1","event":{"kind":"assert","event_id":"E","subject":"x","attribute":"a","value":"v","valid_from":"2026-01-01","valid_to":null,"recorded_at":"2026-01-01T00:00:00Z","source":"operations"},"signature_hex":"00","ingestion_receipt":null,"authority_rank":9999}"#;
        assert!(serde_json::from_str::<SignedLedgerEvent>(event).is_err());
        let receipt = r#"{"schema_version":"BROCA-ATTESTATION-V1","ingester_key_id":"root","accepted_at":"2026-01-01T00:00:00Z","signature_hex":"00","extra":true}"#;
        assert!(serde_json::from_str::<IngestionReceipt>(receipt).is_err());
    }

    #[test]
    fn duplicate_trusted_key_id_is_rejected() {
        let k = key(2);
        let a = trust_config("same", Some("a"), &k, false, true, false, "2026-01-01T00:00:00Z", None);
        let b = trust_config("same", Some("b"), &k, false, true, false, "2026-01-01T00:00:00Z", None);
        assert!(EvidenceTrustStore::new(vec![a, b]).unwrap_err().contains("duplicate trusted key_id"));
    }

    #[test]
    fn malformed_public_key_length_is_rejected() {
        let result = EvidenceTrustStore::new(vec![TrustedSignerConfig {
            key_id: "bad".into(),
            verifying_key_hex: "00".into(),
            source: Some("operations".into()),
            can_sign_policies: false,
            can_sign_events: true,
            can_sign_ingestion_receipts: false,
            valid_from: "2026-01-01T00:00:00Z".into(),
            valid_until: None,
        }]);
        assert!(result.unwrap_err().contains("must be 32 bytes"));
    }

    #[test]
    fn missing_roles_and_source_binding_are_rejected() {
        let k = key(3);
        let config = trust_config("none", None, &k, false, false, false, "2026-01-01T00:00:00Z", None);
        assert!(EvidenceTrustStore::new(vec![config]).unwrap_err().contains("no signing role"));
        let no_source = trust_config("event", None, &k, false, true, false, "2026-01-01T00:00:00Z", None);
        assert!(EvidenceTrustStore::new(vec![no_source]).unwrap_err().contains("must bind to a source"));
    }
}
