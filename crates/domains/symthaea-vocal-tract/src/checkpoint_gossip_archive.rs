// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independently retained archives for authenticated transparency gossip.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{
    CheckpointPublicSignature, CheckpointPublicSigningKey,
    CheckpointPublicVerificationError, CheckpointPublicVerifyingKey,
    CheckpointTransparencyGossipBundle, CheckpointTransparencyGossipPolicy,
    MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES,
};

pub const CHECKPOINT_GOSSIP_ARCHIVE_MEMBER_SCHEMA: &str =
    "symthaea.checkpoint-gossip-archive-member.v1";
pub const CHECKPOINT_GOSSIP_ARCHIVE_POLICY_SCHEMA: &str =
    "symthaea.checkpoint-gossip-archive-policy.v1";
pub const CHECKPOINT_GOSSIP_ARCHIVE_RECEIPT_SCHEMA: &str =
    "symthaea.checkpoint-gossip-archive-receipt.v1";
pub const CHECKPOINT_GOSSIP_ARCHIVE_BUNDLE_SCHEMA: &str =
    "symthaea.checkpoint-gossip-archive-bundle.v1";
pub const CHECKPOINT_GOSSIP_ARCHIVE_SUMMARY_SCHEMA: &str =
    "symthaea.checkpoint-gossip-archive-summary.v1";

pub const MAX_CHECKPOINT_GOSSIP_ARCHIVES: usize = 64;
pub const MAX_CHECKPOINT_GOSSIP_ARCHIVE_RECEIPTS: usize = 1_024;

const GOSSIP_ARCHIVE_POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-gossip-archive-policy-digest-v1\0";
const GOSSIP_STATEMENT_ARCHIVE_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-gossip-statement-archive-digest-v1\0";
const GOSSIP_ARCHIVE_RECEIPT_BODY_DOMAIN: &[u8] =
    b"symthaea-checkpoint-gossip-archive-receipt-body-v1\0";
const GOSSIP_ARCHIVE_RECEIPT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea-checkpoint-gossip-archive-receipt-signature-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointGossipArchiveId(pub [u8; 16]);

impl CheckpointGossipArchiveId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointGossipArchiveError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointGossipArchiveError::InvalidArchive);
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointGossipArchiveMember {
    pub schema: String,
    pub archive_id: CheckpointGossipArchiveId,
    pub verifying_key: CheckpointPublicVerifyingKey,
    pub organization_binding: [u8; 32],
    pub repository_binding: [u8; 32],
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointGossipArchiveMember {
    pub fn validate(&self) -> Result<(), CheckpointGossipArchiveError> {
        self.verifying_key.validate()?;
        if self.schema != CHECKPOINT_GOSSIP_ARCHIVE_MEMBER_SCHEMA
            || self.archive_id.0 == [0u8; 16]
            || self.organization_binding == [0u8; 32]
            || self.repository_binding == [0u8; 32]
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointGossipArchiveError::InvalidArchive);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointGossipArchivePolicy {
    pub schema: String,
    pub policy_id: [u8; 16],
    pub archives: Vec<CheckpointGossipArchiveMember>,
    pub receipts_per_statement: u16,
    pub minimum_organizations: u16,
    pub minimum_retention_seconds: u64,
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointGossipArchivePolicy {
    pub fn validate(&self) -> Result<(), CheckpointGossipArchiveError> {
        if self.schema != CHECKPOINT_GOSSIP_ARCHIVE_POLICY_SCHEMA
            || self.policy_id == [0u8; 16]
            || self.archives.len() < 2
            || self.archives.len() > MAX_CHECKPOINT_GOSSIP_ARCHIVES
            || self.receipts_per_statement < 2
            || usize::from(self.receipts_per_statement) > self.archives.len()
            || self.minimum_organizations < 2
            || self.minimum_organizations > self.receipts_per_statement
            || self.minimum_retention_seconds == 0
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointGossipArchiveError::InvalidPolicy);
        }
        let mut ids = HashSet::with_capacity(self.archives.len());
        let mut keys = HashSet::with_capacity(self.archives.len());
        let mut repos = HashSet::with_capacity(self.archives.len());
        let mut organizations = HashSet::with_capacity(self.archives.len());
        for archive in &self.archives {
            archive.validate()?;
            if archive.valid_from_unix_seconds < self.valid_from_unix_seconds
                || archive.valid_until_unix_seconds > self.valid_until_unix_seconds
                || !ids.insert(archive.archive_id)
                || !keys.insert(archive.verifying_key.key_id)
                || !repos.insert(archive.repository_binding)
            {
                return Err(CheckpointGossipArchiveError::InvalidPolicy);
            }
            organizations.insert(archive.organization_binding);
        }
        if organizations.len() < usize::from(self.minimum_organizations) {
            return Err(CheckpointGossipArchiveError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointGossipArchiveError> {
        self.validate()?;
        gossip_archive_digest(GOSSIP_ARCHIVE_POLICY_DIGEST_DOMAIN, self)
    }

    pub fn archive(
        &self,
        archive_id: CheckpointGossipArchiveId,
    ) -> Option<&CheckpointGossipArchiveMember> {
        self.archives
            .iter()
            .find(|archive| archive.archive_id == archive_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CheckpointGossipArchiveReceiptBody {
    policy_digest: [u8; 32],
    archive_id: CheckpointGossipArchiveId,
    statement_digest: [u8; 32],
    sequence: u64,
    previous_receipt_digest: [u8; 32],
    stored_at_unix_seconds: u64,
    retained_until_unix_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointGossipArchiveReceipt {
    pub schema: String,
    pub policy_digest: [u8; 32],
    pub archive_id: CheckpointGossipArchiveId,
    pub statement_digest: [u8; 32],
    pub sequence: u64,
    pub previous_receipt_digest: [u8; 32],
    pub stored_at_unix_seconds: u64,
    pub retained_until_unix_seconds: u64,
    pub signature: CheckpointPublicSignature,
}

impl CheckpointGossipArchiveReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        signing_key: &CheckpointPublicSigningKey,
        policy: &CheckpointGossipArchivePolicy,
        archive_id: CheckpointGossipArchiveId,
        statement_digest: [u8; 32],
        sequence: u64,
        previous_receipt_digest: [u8; 32],
        stored_at_unix_seconds: u64,
        retained_until_unix_seconds: u64,
    ) -> Result<Self, CheckpointGossipArchiveError> {
        let archive = policy
            .archive(archive_id)
            .ok_or(CheckpointGossipArchiveError::UnknownArchive)?;
        if signing_key.key_id() != archive.verifying_key.key_id
            || statement_digest == [0u8; 32]
            || sequence == 0
            || stored_at_unix_seconds < archive.valid_from_unix_seconds
            || stored_at_unix_seconds > archive.valid_until_unix_seconds
            || retained_until_unix_seconds
                < stored_at_unix_seconds.saturating_add(policy.minimum_retention_seconds)
        {
            return Err(CheckpointGossipArchiveError::InvalidReceipt);
        }
        let policy_digest = policy.digest()?;
        let body = CheckpointGossipArchiveReceiptBody {
            policy_digest,
            archive_id,
            statement_digest,
            sequence,
            previous_receipt_digest,
            stored_at_unix_seconds,
            retained_until_unix_seconds,
        };
        let body_digest = gossip_archive_digest(GOSSIP_ARCHIVE_RECEIPT_BODY_DOMAIN, &body)?;
        Ok(Self {
            schema: CHECKPOINT_GOSSIP_ARCHIVE_RECEIPT_SCHEMA.to_owned(),
            policy_digest,
            archive_id,
            statement_digest,
            sequence,
            previous_receipt_digest,
            stored_at_unix_seconds,
            retained_until_unix_seconds,
            signature: signing_key.sign(GOSSIP_ARCHIVE_RECEIPT_SIGNATURE_DOMAIN, &body_digest)?,
        })
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointGossipArchiveError> {
        gossip_archive_digest(GOSSIP_ARCHIVE_RECEIPT_BODY_DOMAIN, &self.body()?)
    }

    fn body(&self) -> Result<CheckpointGossipArchiveReceiptBody, CheckpointGossipArchiveError> {
        if self.schema != CHECKPOINT_GOSSIP_ARCHIVE_RECEIPT_SCHEMA
            || self.policy_digest == [0u8; 32]
            || self.archive_id.0 == [0u8; 16]
            || self.statement_digest == [0u8; 32]
            || self.sequence == 0
            || self.stored_at_unix_seconds == 0
            || self.retained_until_unix_seconds <= self.stored_at_unix_seconds
        {
            return Err(CheckpointGossipArchiveError::InvalidReceipt);
        }
        Ok(CheckpointGossipArchiveReceiptBody {
            policy_digest: self.policy_digest,
            archive_id: self.archive_id,
            statement_digest: self.statement_digest,
            sequence: self.sequence,
            previous_receipt_digest: self.previous_receipt_digest,
            stored_at_unix_seconds: self.stored_at_unix_seconds,
            retained_until_unix_seconds: self.retained_until_unix_seconds,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointGossipArchiveBundle {
    pub schema: String,
    pub policy: CheckpointGossipArchivePolicy,
    pub gossip_anchor_digest: [u8; 32],
    pub receipts: Vec<CheckpointGossipArchiveReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointGossipArchiveSummary {
    pub schema: String,
    pub gossip_anchor_digest: [u8; 32],
    pub policy_digest: [u8; 32],
    pub archived_statements: usize,
    pub valid_receipts: usize,
    pub unique_archives: usize,
    pub unique_organizations: usize,
    pub minimum_retained_until_unix_seconds: u64,
}

impl CheckpointGossipArchiveSummary {
    pub fn validate(&self) -> Result<(), CheckpointGossipArchiveError> {
        if self.schema != CHECKPOINT_GOSSIP_ARCHIVE_SUMMARY_SCHEMA
            || self.gossip_anchor_digest == [0u8; 32]
            || self.policy_digest == [0u8; 32]
            || self.archived_statements < 2
            || self.valid_receipts < 4
            || self.unique_archives < 2
            || self.unique_organizations < 2
            || self.minimum_retained_until_unix_seconds == 0
        {
            return Err(CheckpointGossipArchiveError::InvalidBundle);
        }
        Ok(())
    }
}

impl CheckpointGossipArchiveBundle {
    pub fn verify(
        &self,
        gossip_bundle: &CheckpointTransparencyGossipBundle,
        gossip_policy: &CheckpointTransparencyGossipPolicy,
        transparency_authority_key: &CheckpointPublicVerifyingKey,
        verification_time_unix_seconds: u64,
    ) -> Result<CheckpointGossipArchiveSummary, CheckpointGossipArchiveError> {
        self.policy.validate()?;
        let gossip_summary = gossip_bundle
            .verify(gossip_policy, transparency_authority_key, verification_time_unix_seconds)
            .map_err(|_| CheckpointGossipArchiveError::InvalidGossipEvidence)?;
        if self.schema != CHECKPOINT_GOSSIP_ARCHIVE_BUNDLE_SCHEMA
            || self.gossip_anchor_digest != gossip_summary.anchor_head_digest
            || self.receipts.is_empty()
            || self.receipts.len() > MAX_CHECKPOINT_GOSSIP_ARCHIVE_RECEIPTS
            || verification_time_unix_seconds < self.policy.valid_from_unix_seconds
            || verification_time_unix_seconds > self.policy.valid_until_unix_seconds
        {
            return Err(CheckpointGossipArchiveError::InvalidBundle);
        }
        let mut statement_digests = HashSet::with_capacity(gossip_bundle.observations.len());
        for observation in &gossip_bundle.observations {
            statement_digests.insert(gossip_archive_digest(
                GOSSIP_STATEMENT_ARCHIVE_DIGEST_DOMAIN,
                &observation.statement,
            )?);
        }
        if statement_digests.len() != gossip_summary.valid_observations {
            return Err(CheckpointGossipArchiveError::InvalidGossipEvidence);
        }
        let policy_digest = self.policy.digest()?;
        let mut receipts_by_statement: HashMap<[u8; 32], HashSet<CheckpointGossipArchiveId>> =
            HashMap::new();
        let mut archive_last: HashMap<CheckpointGossipArchiveId, (u64, [u8; 32])> = HashMap::new();
        let mut organizations = HashSet::new();
        let mut minimum_retained_until = u64::MAX;
        for receipt in &self.receipts {
            let archive = self
                .policy
                .archive(receipt.archive_id)
                .ok_or(CheckpointGossipArchiveError::UnknownArchive)?;
            if receipt.policy_digest != policy_digest
                || !statement_digests.contains(&receipt.statement_digest)
                || receipt.stored_at_unix_seconds > verification_time_unix_seconds
                || receipt.retained_until_unix_seconds
                    < verification_time_unix_seconds
                        .saturating_add(self.policy.minimum_retention_seconds)
            {
                return Err(CheckpointGossipArchiveError::InvalidReceipt);
            }
            let body_digest = receipt.digest()?;
            archive.verifying_key.verify(
                GOSSIP_ARCHIVE_RECEIPT_SIGNATURE_DOMAIN,
                &body_digest,
                &receipt.signature,
            )?;
            if let Some((previous_sequence, previous_digest)) = archive_last.get(&receipt.archive_id) {
                if receipt.sequence != previous_sequence.saturating_add(1)
                    || receipt.previous_receipt_digest != *previous_digest
                {
                    return Err(CheckpointGossipArchiveError::ArchiveChainFork);
                }
            } else if receipt.sequence != 1
                || receipt.previous_receipt_digest != [0u8; 32]
            {
                return Err(CheckpointGossipArchiveError::ArchiveChainFork);
            }
            let receipt_digest = gossip_archive_digest(
                GOSSIP_ARCHIVE_RECEIPT_BODY_DOMAIN,
                &receipt.body()?,
            )?;
            archive_last.insert(receipt.archive_id, (receipt.sequence, receipt_digest));
            let archives = receipts_by_statement
                .entry(receipt.statement_digest)
                .or_default();
            if !archives.insert(receipt.archive_id) {
                return Err(CheckpointGossipArchiveError::DuplicateReceipt);
            }
            organizations.insert(archive.organization_binding);
            minimum_retained_until = minimum_retained_until.min(receipt.retained_until_unix_seconds);
        }
        for statement_digest in &statement_digests {
            let archives = receipts_by_statement
                .get(statement_digest)
                .ok_or(CheckpointGossipArchiveError::MissingReceipt)?;
            if archives.len() < usize::from(self.policy.receipts_per_statement) {
                return Err(CheckpointGossipArchiveError::MissingReceipt);
            }
        }
        if organizations.len() < usize::from(self.policy.minimum_organizations) {
            return Err(CheckpointGossipArchiveError::InsufficientArchives);
        }
        let summary = CheckpointGossipArchiveSummary {
            schema: CHECKPOINT_GOSSIP_ARCHIVE_SUMMARY_SCHEMA.to_owned(),
            gossip_anchor_digest: self.gossip_anchor_digest,
            policy_digest,
            archived_statements: statement_digests.len(),
            valid_receipts: self.receipts.len(),
            unique_archives: archive_last.len(),
            unique_organizations: organizations.len(),
            minimum_retained_until_unix_seconds: minimum_retained_until,
        };
        summary.validate()?;
        Ok(summary)
    }
}

fn gossip_archive_digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], CheckpointGossipArchiveError> {
    let encoded = postcard::to_stdvec(value).map_err(|_| CheckpointGossipArchiveError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointGossipArchiveError::TooLarge);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(*hasher.finalize().as_bytes())
}

#[derive(Debug)]
pub enum CheckpointGossipArchiveError {
    InvalidArchive,
    InvalidPolicy,
    UnknownArchive,
    InvalidReceipt,
    DuplicateReceipt,
    MissingReceipt,
    ArchiveChainFork,
    InsufficientArchives,
    InvalidGossipEvidence,
    InvalidBundle,
    Encoding,
    TooLarge,
    PublicVerification,
}

impl std::fmt::Display for CheckpointGossipArchiveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidArchive => "invalid gossip archive",
            Self::InvalidPolicy => "invalid gossip archive policy",
            Self::UnknownArchive => "unknown gossip archive",
            Self::InvalidReceipt => "invalid gossip archive receipt",
            Self::DuplicateReceipt => "duplicate gossip archive receipt",
            Self::MissingReceipt => "missing required gossip archive receipt",
            Self::ArchiveChainFork => "gossip archive receipt chain forked",
            Self::InsufficientArchives => "insufficient independent gossip archives",
            Self::InvalidGossipEvidence => "invalid source gossip evidence",
            Self::InvalidBundle => "invalid gossip archive bundle",
            Self::Encoding => "gossip archive encoding failed",
            Self::TooLarge => "gossip archive artifact exceeds its bound",
            Self::PublicVerification => "gossip archive signature verification failed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CheckpointGossipArchiveError {}

impl From<CheckpointPublicVerificationError> for CheckpointGossipArchiveError {
    fn from(_: CheckpointPublicVerificationError) -> Self {
        Self::PublicVerification
    }
}
