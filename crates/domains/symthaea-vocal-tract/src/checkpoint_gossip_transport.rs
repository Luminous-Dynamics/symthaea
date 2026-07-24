// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent transport receipts for authenticated transparency gossip.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{
    CheckpointPublicSignature, CheckpointPublicSigningKey, CheckpointPublicVerificationError,
    CheckpointPublicVerifyingKey, CheckpointTransparencyGossipBundle,
    CheckpointTransparencyGossipPolicy, CheckpointTransparencyOriginId,
    MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES,
};

pub const CHECKPOINT_GOSSIP_TRANSPORT_MEMBER_SCHEMA: &str =
    "symthaea.checkpoint-gossip-transport-member.v1";
pub const CHECKPOINT_GOSSIP_TRANSPORT_POLICY_SCHEMA: &str =
    "symthaea.checkpoint-gossip-transport-policy.v1";
pub const CHECKPOINT_GOSSIP_DELIVERY_RECEIPT_SCHEMA: &str =
    "symthaea.checkpoint-gossip-delivery-receipt.v1";
pub const CHECKPOINT_GOSSIP_TRANSPORT_BUNDLE_SCHEMA: &str =
    "symthaea.checkpoint-gossip-transport-bundle.v1";
pub const CHECKPOINT_GOSSIP_TRANSPORT_SUMMARY_SCHEMA: &str =
    "symthaea.checkpoint-gossip-transport-summary.v1";

pub const MAX_CHECKPOINT_GOSSIP_TRANSPORTS: usize = 64;
pub const MAX_CHECKPOINT_GOSSIP_DELIVERY_RECEIPTS: usize = 1_024;
pub const MAX_CHECKPOINT_GOSSIP_DELIVERY_SECONDS: u64 = 3_600;

const GOSSIP_TRANSPORT_POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-gossip-transport-policy-digest-v1\0";
const GOSSIP_TRANSPORT_STATEMENT_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-gossip-transport-statement-digest-v1\0";
const GOSSIP_DELIVERY_RECEIPT_BODY_DOMAIN: &[u8] =
    b"symthaea-checkpoint-gossip-delivery-receipt-body-v1\0";
const GOSSIP_DELIVERY_RECEIPT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea-checkpoint-gossip-delivery-receipt-signature-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointGossipTransportId(pub [u8; 16]);

impl CheckpointGossipTransportId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointGossipTransportError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointGossipTransportError::InvalidTransport);
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointGossipTransportMember {
    pub schema: String,
    pub transport_id: CheckpointGossipTransportId,
    pub verifying_key: CheckpointPublicVerifyingKey,
    pub organization_binding: [u8; 32],
    pub network_binding: [u8; 32],
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointGossipTransportMember {
    pub fn validate(&self) -> Result<(), CheckpointGossipTransportError> {
        self.verifying_key.validate()?;
        if self.schema != CHECKPOINT_GOSSIP_TRANSPORT_MEMBER_SCHEMA
            || self.transport_id.0 == [0u8; 16]
            || self.organization_binding == [0u8; 32]
            || self.network_binding == [0u8; 32]
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointGossipTransportError::InvalidTransport);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointGossipTransportPolicy {
    pub schema: String,
    pub policy_id: [u8; 16],
    pub transports: Vec<CheckpointGossipTransportMember>,
    pub receipts_per_statement: u16,
    pub minimum_organizations: u16,
    pub maximum_delivery_seconds: u64,
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointGossipTransportPolicy {
    pub fn validate(&self) -> Result<(), CheckpointGossipTransportError> {
        if self.schema != CHECKPOINT_GOSSIP_TRANSPORT_POLICY_SCHEMA
            || self.policy_id == [0u8; 16]
            || self.transports.len() < 2
            || self.transports.len() > MAX_CHECKPOINT_GOSSIP_TRANSPORTS
            || self.receipts_per_statement < 2
            || usize::from(self.receipts_per_statement) > self.transports.len()
            || self.minimum_organizations < 2
            || self.minimum_organizations > self.receipts_per_statement
            || self.maximum_delivery_seconds == 0
            || self.maximum_delivery_seconds > MAX_CHECKPOINT_GOSSIP_DELIVERY_SECONDS
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointGossipTransportError::InvalidPolicy);
        }
        let mut ids = HashSet::with_capacity(self.transports.len());
        let mut keys = HashSet::with_capacity(self.transports.len());
        let mut networks = HashSet::with_capacity(self.transports.len());
        let mut organizations = HashSet::with_capacity(self.transports.len());
        for transport in &self.transports {
            transport.validate()?;
            if transport.valid_from_unix_seconds < self.valid_from_unix_seconds
                || transport.valid_until_unix_seconds > self.valid_until_unix_seconds
                || !ids.insert(transport.transport_id)
                || !keys.insert(transport.verifying_key.key_id)
                || !networks.insert(transport.network_binding)
            {
                return Err(CheckpointGossipTransportError::InvalidPolicy);
            }
            organizations.insert(transport.organization_binding);
        }
        if organizations.len() < usize::from(self.minimum_organizations) {
            return Err(CheckpointGossipTransportError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointGossipTransportError> {
        self.validate()?;
        gossip_transport_digest(GOSSIP_TRANSPORT_POLICY_DIGEST_DOMAIN, self)
    }

    pub fn transport(
        &self,
        transport_id: CheckpointGossipTransportId,
    ) -> Option<&CheckpointGossipTransportMember> {
        self.transports
            .iter()
            .find(|transport| transport.transport_id == transport_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CheckpointGossipDeliveryReceiptBody {
    policy_digest: [u8; 32],
    transport_id: CheckpointGossipTransportId,
    delivery_id: [u8; 16],
    statement_digest: [u8; 32],
    origin_id: CheckpointTransparencyOriginId,
    source_endpoint_binding: [u8; 32],
    destination_endpoint_binding: [u8; 32],
    received_at_unix_seconds: u64,
    delivered_at_unix_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointGossipDeliveryReceipt {
    pub schema: String,
    pub policy_digest: [u8; 32],
    pub transport_id: CheckpointGossipTransportId,
    pub delivery_id: [u8; 16],
    pub statement_digest: [u8; 32],
    pub origin_id: CheckpointTransparencyOriginId,
    pub source_endpoint_binding: [u8; 32],
    pub destination_endpoint_binding: [u8; 32],
    pub received_at_unix_seconds: u64,
    pub delivered_at_unix_seconds: u64,
    pub signature: CheckpointPublicSignature,
}

impl CheckpointGossipDeliveryReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        signing_key: &CheckpointPublicSigningKey,
        policy: &CheckpointGossipTransportPolicy,
        transport_id: CheckpointGossipTransportId,
        delivery_id: [u8; 16],
        statement_digest: [u8; 32],
        origin_id: CheckpointTransparencyOriginId,
        source_endpoint_binding: [u8; 32],
        destination_endpoint_binding: [u8; 32],
        received_at_unix_seconds: u64,
        delivered_at_unix_seconds: u64,
    ) -> Result<Self, CheckpointGossipTransportError> {
        let transport = policy
            .transport(transport_id)
            .ok_or(CheckpointGossipTransportError::UnknownTransport)?;
        if signing_key.key_id() != transport.verifying_key.key_id
            || delivery_id == [0u8; 16]
            || statement_digest == [0u8; 32]
            || origin_id.0 == [0u8; 16]
            || source_endpoint_binding == [0u8; 32]
            || destination_endpoint_binding == [0u8; 32]
            || source_endpoint_binding == destination_endpoint_binding
            || received_at_unix_seconds < transport.valid_from_unix_seconds
            || delivered_at_unix_seconds < received_at_unix_seconds
            || delivered_at_unix_seconds > transport.valid_until_unix_seconds
            || delivered_at_unix_seconds.saturating_sub(received_at_unix_seconds)
                > policy.maximum_delivery_seconds
        {
            return Err(CheckpointGossipTransportError::InvalidReceipt);
        }
        let policy_digest = policy.digest()?;
        let body = CheckpointGossipDeliveryReceiptBody {
            policy_digest,
            transport_id,
            delivery_id,
            statement_digest,
            origin_id,
            source_endpoint_binding,
            destination_endpoint_binding,
            received_at_unix_seconds,
            delivered_at_unix_seconds,
        };
        let body_digest = gossip_transport_digest(GOSSIP_DELIVERY_RECEIPT_BODY_DOMAIN, &body)?;
        Ok(Self {
            schema: CHECKPOINT_GOSSIP_DELIVERY_RECEIPT_SCHEMA.to_owned(),
            policy_digest,
            transport_id,
            delivery_id,
            statement_digest,
            origin_id,
            source_endpoint_binding,
            destination_endpoint_binding,
            received_at_unix_seconds,
            delivered_at_unix_seconds,
            signature: signing_key.sign(GOSSIP_DELIVERY_RECEIPT_SIGNATURE_DOMAIN, &body_digest)?,
        })
    }

    fn body_digest(&self) -> Result<[u8; 32], CheckpointGossipTransportError> {
        if self.schema != CHECKPOINT_GOSSIP_DELIVERY_RECEIPT_SCHEMA
            || self.policy_digest == [0u8; 32]
            || self.transport_id.0 == [0u8; 16]
            || self.delivery_id == [0u8; 16]
            || self.statement_digest == [0u8; 32]
            || self.origin_id.0 == [0u8; 16]
            || self.source_endpoint_binding == [0u8; 32]
            || self.destination_endpoint_binding == [0u8; 32]
            || self.source_endpoint_binding == self.destination_endpoint_binding
            || self.received_at_unix_seconds == 0
            || self.delivered_at_unix_seconds < self.received_at_unix_seconds
        {
            return Err(CheckpointGossipTransportError::InvalidReceipt);
        }
        gossip_transport_digest(
            GOSSIP_DELIVERY_RECEIPT_BODY_DOMAIN,
            &CheckpointGossipDeliveryReceiptBody {
                policy_digest: self.policy_digest,
                transport_id: self.transport_id,
                delivery_id: self.delivery_id,
                statement_digest: self.statement_digest,
                origin_id: self.origin_id,
                source_endpoint_binding: self.source_endpoint_binding,
                destination_endpoint_binding: self.destination_endpoint_binding,
                received_at_unix_seconds: self.received_at_unix_seconds,
                delivered_at_unix_seconds: self.delivered_at_unix_seconds,
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointGossipTransportBundle {
    pub schema: String,
    pub policy: CheckpointGossipTransportPolicy,
    pub gossip_anchor_digest: [u8; 32],
    pub receipts: Vec<CheckpointGossipDeliveryReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointGossipTransportSummary {
    pub schema: String,
    pub gossip_anchor_digest: [u8; 32],
    pub policy_digest: [u8; 32],
    pub delivered_statements: usize,
    pub valid_receipts: usize,
    pub unique_transports: usize,
    pub unique_organizations: usize,
    pub unique_networks: usize,
    pub maximum_observed_delivery_seconds: u64,
}

impl CheckpointGossipTransportSummary {
    pub fn validate(&self) -> Result<(), CheckpointGossipTransportError> {
        if self.schema != CHECKPOINT_GOSSIP_TRANSPORT_SUMMARY_SCHEMA
            || self.gossip_anchor_digest == [0u8; 32]
            || self.policy_digest == [0u8; 32]
            || self.delivered_statements < 2
            || self.valid_receipts < 4
            || self.unique_transports < 2
            || self.unique_organizations < 2
            || self.unique_networks < 2
        {
            return Err(CheckpointGossipTransportError::InvalidBundle);
        }
        Ok(())
    }
}

impl CheckpointGossipTransportBundle {
    pub fn verify(
        &self,
        gossip_bundle: &CheckpointTransparencyGossipBundle,
        gossip_policy: &CheckpointTransparencyGossipPolicy,
        transparency_authority_key: &CheckpointPublicVerifyingKey,
        verification_time_unix_seconds: u64,
    ) -> Result<CheckpointGossipTransportSummary, CheckpointGossipTransportError> {
        self.policy.validate()?;
        let gossip_summary = gossip_bundle
            .verify(gossip_policy, transparency_authority_key, verification_time_unix_seconds)
            .map_err(|_| CheckpointGossipTransportError::InvalidGossipEvidence)?;
        if self.schema != CHECKPOINT_GOSSIP_TRANSPORT_BUNDLE_SCHEMA
            || self.gossip_anchor_digest != gossip_summary.anchor_head_digest
            || self.receipts.is_empty()
            || self.receipts.len() > MAX_CHECKPOINT_GOSSIP_DELIVERY_RECEIPTS
            || verification_time_unix_seconds < self.policy.valid_from_unix_seconds
            || verification_time_unix_seconds > self.policy.valid_until_unix_seconds
        {
            return Err(CheckpointGossipTransportError::InvalidBundle);
        }
        let mut statements = HashMap::with_capacity(gossip_bundle.observations.len());
        for observation in &gossip_bundle.observations {
            let digest = gossip_transport_digest(
                GOSSIP_TRANSPORT_STATEMENT_DIGEST_DOMAIN,
                &observation.statement,
            )?;
            if statements.insert(digest, observation.statement.origin_id).is_some() {
                return Err(CheckpointGossipTransportError::InvalidGossipEvidence);
            }
        }
        if statements.len() != gossip_summary.valid_observations {
            return Err(CheckpointGossipTransportError::InvalidGossipEvidence);
        }
        let policy_digest = self.policy.digest()?;
        let mut receipts_by_statement: HashMap<[u8; 32], HashSet<CheckpointGossipTransportId>> =
            HashMap::new();
        let mut delivery_ids = HashSet::new();
        let mut transports = HashSet::new();
        let mut organizations = HashSet::new();
        let mut networks = HashSet::new();
        let mut maximum_delivery_seconds = 0u64;
        for receipt in &self.receipts {
            let transport = self
                .policy
                .transport(receipt.transport_id)
                .ok_or(CheckpointGossipTransportError::UnknownTransport)?;
            let expected_origin = statements
                .get(&receipt.statement_digest)
                .ok_or(CheckpointGossipTransportError::InvalidReceipt)?;
            if receipt.policy_digest != policy_digest
                || receipt.origin_id != *expected_origin
                || receipt.delivered_at_unix_seconds > verification_time_unix_seconds
                || receipt.delivered_at_unix_seconds
                    .saturating_sub(receipt.received_at_unix_seconds)
                    > self.policy.maximum_delivery_seconds
                || !delivery_ids.insert(receipt.delivery_id)
            {
                return Err(CheckpointGossipTransportError::InvalidReceipt);
            }
            let body_digest = receipt.body_digest()?;
            transport.verifying_key.verify(
                GOSSIP_DELIVERY_RECEIPT_SIGNATURE_DOMAIN,
                &body_digest,
                &receipt.signature,
            )?;
            let members = receipts_by_statement
                .entry(receipt.statement_digest)
                .or_default();
            if !members.insert(receipt.transport_id) {
                return Err(CheckpointGossipTransportError::DuplicateReceipt);
            }
            transports.insert(receipt.transport_id);
            organizations.insert(transport.organization_binding);
            networks.insert(transport.network_binding);
            maximum_delivery_seconds = maximum_delivery_seconds.max(
                receipt
                    .delivered_at_unix_seconds
                    .saturating_sub(receipt.received_at_unix_seconds),
            );
        }
        for statement_digest in statements.keys() {
            let members = receipts_by_statement
                .get(statement_digest)
                .ok_or(CheckpointGossipTransportError::MissingReceipt)?;
            if members.len() < usize::from(self.policy.receipts_per_statement) {
                return Err(CheckpointGossipTransportError::MissingReceipt);
            }
        }
        if organizations.len() < usize::from(self.policy.minimum_organizations)
            || networks.len() < usize::from(self.policy.receipts_per_statement)
        {
            return Err(CheckpointGossipTransportError::InsufficientTransports);
        }
        let summary = CheckpointGossipTransportSummary {
            schema: CHECKPOINT_GOSSIP_TRANSPORT_SUMMARY_SCHEMA.to_owned(),
            gossip_anchor_digest: self.gossip_anchor_digest,
            policy_digest,
            delivered_statements: statements.len(),
            valid_receipts: self.receipts.len(),
            unique_transports: transports.len(),
            unique_organizations: organizations.len(),
            unique_networks: networks.len(),
            maximum_observed_delivery_seconds: maximum_delivery_seconds,
        };
        summary.validate()?;
        Ok(summary)
    }
}

fn gossip_transport_digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], CheckpointGossipTransportError> {
    let encoded = postcard::to_stdvec(value).map_err(|_| CheckpointGossipTransportError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointGossipTransportError::TooLarge);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(*hasher.finalize().as_bytes())
}

#[derive(Debug)]
pub enum CheckpointGossipTransportError {
    InvalidTransport,
    InvalidPolicy,
    UnknownTransport,
    InvalidReceipt,
    DuplicateReceipt,
    MissingReceipt,
    InsufficientTransports,
    InvalidGossipEvidence,
    InvalidBundle,
    Encoding,
    TooLarge,
    PublicVerification,
}

impl std::fmt::Display for CheckpointGossipTransportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidTransport => "invalid gossip transport",
            Self::InvalidPolicy => "invalid gossip transport policy",
            Self::UnknownTransport => "unknown gossip transport",
            Self::InvalidReceipt => "invalid gossip delivery receipt",
            Self::DuplicateReceipt => "duplicate gossip delivery receipt",
            Self::MissingReceipt => "missing required gossip delivery receipt",
            Self::InsufficientTransports => "insufficient independent gossip transports",
            Self::InvalidGossipEvidence => "invalid source gossip evidence",
            Self::InvalidBundle => "invalid gossip transport bundle",
            Self::Encoding => "gossip transport encoding failed",
            Self::TooLarge => "gossip transport artifact exceeds its bound",
            Self::PublicVerification => "gossip transport signature verification failed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CheckpointGossipTransportError {}

impl From<CheckpointPublicVerificationError> for CheckpointGossipTransportError {
    fn from(_: CheckpointPublicVerificationError) -> Self {
        Self::PublicVerification
    }
}
