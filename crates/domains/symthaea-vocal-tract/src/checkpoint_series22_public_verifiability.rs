// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Portable Series 22 hardware-custody, trusted-time, and gossip-durability bundle.

use serde::{Deserialize, Serialize};

use crate::{
    CheckpointGossipArchiveBundle, CheckpointGossipArchiveError,
    CheckpointGossipArchiveSummary, CheckpointGossipTransportBundle,
    CheckpointGossipTransportError, CheckpointGossipTransportSummary,
    CheckpointHardwareCustodyDowngradeNegativeSummary, CheckpointHardwareSecurityLevel,
    CheckpointHardwareSigningBundle, CheckpointHardwareSigningError,
    CheckpointHardwareSigningSummary, CheckpointSeries21PublicVerificationBundle,
    CheckpointSeries21PublicVerificationError, CheckpointSeries21PublicVerificationSummary,
    CheckpointTrustedTimeBundle, CheckpointTrustedTimeError,
    CheckpointTrustedTimeStaleNegativeSummary, CheckpointTrustedTimeSummary,
    MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES, verify_hardware_custody_downgrade_negative,
    verify_trusted_time_stale_negative,
};

pub const CHECKPOINT_SERIES22_PUBLIC_VERIFICATION_BUNDLE_SCHEMA: &str =
    "symthaea.checkpoint-series22-public-verification-bundle.v1";
pub const CHECKPOINT_SERIES22_PUBLIC_VERIFICATION_SUMMARY_SCHEMA: &str =
    "symthaea.checkpoint-series22-public-verification-summary.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointSeries22PublicVerificationBundle {
    pub schema: String,
    pub series21_bundle: CheckpointSeries21PublicVerificationBundle,
    pub hardware_signing_bundle: CheckpointHardwareSigningBundle,
    pub hardware_custody_downgrade_candidate: CheckpointHardwareSigningBundle,
    pub trusted_time_bundle: CheckpointTrustedTimeBundle,
    pub stale_time_candidate: CheckpointTrustedTimeBundle,
    pub gossip_archive_bundle: CheckpointGossipArchiveBundle,
    pub gossip_transport_bundle: CheckpointGossipTransportBundle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSeries22PublicVerificationSummary {
    pub(crate) schema: String,
    pub(crate) series21: CheckpointSeries21PublicVerificationSummary,
    pub(crate) hardware_signing: CheckpointHardwareSigningSummary,
    pub(crate) hardware_downgrade_negative: CheckpointHardwareCustodyDowngradeNegativeSummary,
    pub(crate) trusted_time: CheckpointTrustedTimeSummary,
    pub(crate) stale_time_negative: CheckpointTrustedTimeStaleNegativeSummary,
    pub(crate) gossip_archive: CheckpointGossipArchiveSummary,
    pub(crate) gossip_transport: CheckpointGossipTransportSummary,
}

impl CheckpointSeries22PublicVerificationSummary {
    pub fn series21(&self) -> &CheckpointSeries21PublicVerificationSummary {
        &self.series21
    }

    pub fn hardware_signing(&self) -> &CheckpointHardwareSigningSummary {
        &self.hardware_signing
    }

    pub fn hardware_downgrade_negative(
        &self,
    ) -> &CheckpointHardwareCustodyDowngradeNegativeSummary {
        &self.hardware_downgrade_negative
    }

    pub fn trusted_time(&self) -> &CheckpointTrustedTimeSummary {
        &self.trusted_time
    }

    pub fn stale_time_negative(&self) -> &CheckpointTrustedTimeStaleNegativeSummary {
        &self.stale_time_negative
    }

    pub fn gossip_archive(&self) -> &CheckpointGossipArchiveSummary {
        &self.gossip_archive
    }

    pub fn gossip_transport(&self) -> &CheckpointGossipTransportSummary {
        &self.gossip_transport
    }

    pub fn validate(&self) -> Result<(), CheckpointSeries22PublicVerificationError> {
        if self.schema != CHECKPOINT_SERIES22_PUBLIC_VERIFICATION_SUMMARY_SCHEMA
            || self.hardware_signing.minimum_security_level
                < CheckpointHardwareSecurityLevel::HardwareSecurityModule
            || self.hardware_signing.publication_digest
                != self.series21.hybrid().publication_digest
            || self.trusted_time.subject_digest != self.series21.hybrid().publication_digest
            || self.gossip_archive.gossip_anchor_digest
                != self.series21.gossip().anchor_head_digest
            || self.gossip_transport.gossip_anchor_digest
                != self.series21.gossip().anchor_head_digest
            || self.gossip_archive.minimum_retained_until_unix_seconds
                < self.trusted_time.consensus_not_after_unix_seconds
        {
            return Err(CheckpointSeries22PublicVerificationError::InvalidBundle);
        }
        self.series21.validate()?;
        self.hardware_signing.validate()?;
        self.hardware_downgrade_negative.validate()?;
        self.trusted_time.validate()?;
        self.stale_time_negative.validate()?;
        self.gossip_archive.validate()?;
        self.gossip_transport.validate()?;
        Ok(())
    }
}

impl CheckpointSeries22PublicVerificationBundle {
    pub fn verify(
        &self,
        verification_time_unix_seconds: u64,
    ) -> Result<CheckpointSeries22PublicVerificationSummary, CheckpointSeries22PublicVerificationError>
    {
        if self.schema != CHECKPOINT_SERIES22_PUBLIC_VERIFICATION_BUNDLE_SCHEMA
            || self.trusted_time_bundle.subject_digest
                != self.stale_time_candidate.subject_digest
            || self.hardware_signing_bundle.policy.required_security_level
                < CheckpointHardwareSecurityLevel::HardwareSecurityModule
            || self.hardware_custody_downgrade_candidate
                .policy
                .required_security_level
                != CheckpointHardwareSecurityLevel::SoftwareReference
        {
            return Err(CheckpointSeries22PublicVerificationError::InvalidBundle);
        }
        let series21 = self
            .series21_bundle
            .verify(verification_time_unix_seconds)?;
        if self.trusted_time_bundle.subject_digest != series21.hybrid().publication_digest {
            return Err(CheckpointSeries22PublicVerificationError::InvalidBundle);
        }
        let hardware_signing = self.hardware_signing_bundle.verify(
            &self.series21_bundle.hybrid_bundle,
            series21.hybrid(),
            verification_time_unix_seconds,
        )?;
        let hardware_downgrade_negative = verify_hardware_custody_downgrade_negative(
            &self.hardware_custody_downgrade_candidate,
            &self.series21_bundle.hybrid_bundle,
            series21.hybrid(),
            verification_time_unix_seconds,
        )?;
        let trusted_time = self
            .trusted_time_bundle
            .verify(verification_time_unix_seconds)?;
        let stale_time_negative = verify_trusted_time_stale_negative(
            &self.stale_time_candidate,
            verification_time_unix_seconds,
        )?;
        let gossip_archive = self.gossip_archive_bundle.verify(
            &self.series21_bundle.gossip_bundle,
            &self.series21_bundle.gossip_policy,
            &self.series21_bundle.transparency_authority_key,
            verification_time_unix_seconds,
        )?;
        let gossip_transport = self.gossip_transport_bundle.verify(
            &self.series21_bundle.gossip_bundle,
            &self.series21_bundle.gossip_policy,
            &self.series21_bundle.transparency_authority_key,
            verification_time_unix_seconds,
        )?;
        let summary = CheckpointSeries22PublicVerificationSummary {
            schema: CHECKPOINT_SERIES22_PUBLIC_VERIFICATION_SUMMARY_SCHEMA.to_owned(),
            series21,
            hardware_signing,
            hardware_downgrade_negative,
            trusted_time,
            stale_time_negative,
            gossip_archive,
            gossip_transport,
        };
        summary.validate()?;
        Ok(summary)
    }
}

pub fn encode_checkpoint_series22_public_verification_bundle(
    bundle: &CheckpointSeries22PublicVerificationBundle,
) -> Result<Vec<u8>, CheckpointSeries22PublicVerificationError> {
    let encoded = postcard::to_stdvec(bundle)
        .map_err(|_| CheckpointSeries22PublicVerificationError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointSeries22PublicVerificationError::TooLarge);
    }
    Ok(encoded)
}

pub fn decode_checkpoint_series22_public_verification_bundle(
    encoded: &[u8],
) -> Result<CheckpointSeries22PublicVerificationBundle, CheckpointSeries22PublicVerificationError> {
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointSeries22PublicVerificationError::TooLarge);
    }
    let bundle: CheckpointSeries22PublicVerificationBundle = postcard::from_bytes(encoded)
        .map_err(|_| CheckpointSeries22PublicVerificationError::Encoding)?;
    if bundle.schema != CHECKPOINT_SERIES22_PUBLIC_VERIFICATION_BUNDLE_SCHEMA {
        return Err(CheckpointSeries22PublicVerificationError::InvalidBundle);
    }
    Ok(bundle)
}

#[derive(Debug)]
pub enum CheckpointSeries22PublicVerificationError {
    InvalidBundle,
    Encoding,
    TooLarge,
    Series21(CheckpointSeries21PublicVerificationError),
    Hardware(CheckpointHardwareSigningError),
    TrustedTime(CheckpointTrustedTimeError),
    GossipArchive(CheckpointGossipArchiveError),
    GossipTransport(CheckpointGossipTransportError),
}

impl std::fmt::Display for CheckpointSeries22PublicVerificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBundle => formatter.write_str("invalid Series 22 public bundle"),
            Self::Encoding => formatter.write_str("Series 22 public bundle encoding failed"),
            Self::TooLarge => formatter.write_str("Series 22 public bundle exceeds its bound"),
            Self::Series21(error) => write!(formatter, "Series 21 verification failed: {error}"),
            Self::Hardware(error) => write!(formatter, "hardware custody verification failed: {error}"),
            Self::TrustedTime(error) => write!(formatter, "trusted-time verification failed: {error}"),
            Self::GossipArchive(error) => write!(formatter, "gossip archive verification failed: {error}"),
            Self::GossipTransport(error) => write!(formatter, "gossip transport verification failed: {error}"),
        }
    }
}

impl std::error::Error for CheckpointSeries22PublicVerificationError {}

impl From<CheckpointSeries21PublicVerificationError> for CheckpointSeries22PublicVerificationError {
    fn from(error: CheckpointSeries21PublicVerificationError) -> Self {
        Self::Series21(error)
    }
}

impl From<CheckpointHardwareSigningError> for CheckpointSeries22PublicVerificationError {
    fn from(error: CheckpointHardwareSigningError) -> Self {
        Self::Hardware(error)
    }
}

impl From<CheckpointTrustedTimeError> for CheckpointSeries22PublicVerificationError {
    fn from(error: CheckpointTrustedTimeError) -> Self {
        Self::TrustedTime(error)
    }
}

impl From<CheckpointGossipArchiveError> for CheckpointSeries22PublicVerificationError {
    fn from(error: CheckpointGossipArchiveError) -> Self {
        Self::GossipArchive(error)
    }
}

impl From<CheckpointGossipTransportError> for CheckpointSeries22PublicVerificationError {
    fn from(error: CheckpointGossipTransportError) -> Self {
        Self::GossipTransport(error)
    }
}
