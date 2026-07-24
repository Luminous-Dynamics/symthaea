// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Portable Series 21 hybrid and gossip verification artifact.

use serde::{Deserialize, Serialize};

use crate::{
    CheckpointHybridDowngradeNegativeSummary, CheckpointHybridVerificationBundle,
    CheckpointHybridVerificationError, CheckpointHybridVerificationSummary,
    CheckpointPublicVerifyingKey, CheckpointTransparencyGossipBundle,
    CheckpointTransparencyGossipError, CheckpointTransparencyGossipPolicy,
    CheckpointTransparencyGossipSummary, CheckpointTransparencySplitViewNegativeSummary,
    MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES, verify_hybrid_downgrade_negative,
    verify_transparency_split_view_negative,
};

pub const CHECKPOINT_SERIES21_PUBLIC_VERIFICATION_BUNDLE_SCHEMA: &str =
    "symthaea.checkpoint-series21-public-verification-bundle.v1";
pub const CHECKPOINT_SERIES21_PUBLIC_VERIFICATION_SUMMARY_SCHEMA: &str =
    "symthaea.checkpoint-series21-public-verification-summary.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointSeries21PublicVerificationBundle {
    pub schema: String,
    pub hybrid_bundle: CheckpointHybridVerificationBundle,
    pub downgrade_candidate: CheckpointHybridVerificationBundle,
    pub transparency_authority_key: CheckpointPublicVerifyingKey,
    pub gossip_policy: CheckpointTransparencyGossipPolicy,
    pub gossip_bundle: CheckpointTransparencyGossipBundle,
    pub split_view_candidate: CheckpointTransparencyGossipBundle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSeries21PublicVerificationSummary {
    pub(crate) schema: String,
    pub(crate) hybrid: CheckpointHybridVerificationSummary,
    pub(crate) hybrid_downgrade_negative: CheckpointHybridDowngradeNegativeSummary,
    pub(crate) gossip: CheckpointTransparencyGossipSummary,
    pub(crate) split_view_negative: CheckpointTransparencySplitViewNegativeSummary,
}

impl CheckpointSeries21PublicVerificationSummary {
    pub fn hybrid(&self) -> &CheckpointHybridVerificationSummary {
        &self.hybrid
    }

    pub fn gossip(&self) -> &CheckpointTransparencyGossipSummary {
        &self.gossip
    }

    pub fn hybrid_downgrade_negative(&self) -> &CheckpointHybridDowngradeNegativeSummary {
        &self.hybrid_downgrade_negative
    }

    pub fn split_view_negative(&self) -> &CheckpointTransparencySplitViewNegativeSummary {
        &self.split_view_negative
    }

    pub fn validate(&self) -> Result<(), CheckpointSeries21PublicVerificationError> {
        if self.schema != CHECKPOINT_SERIES21_PUBLIC_VERIFICATION_SUMMARY_SCHEMA {
            return Err(CheckpointSeries21PublicVerificationError::InvalidBundle);
        }
        self.hybrid.validate()?;
        self.hybrid_downgrade_negative.validate()?;
        self.gossip.validate()?;
        self.split_view_negative.validate()?;
        Ok(())
    }
}

impl CheckpointSeries21PublicVerificationBundle {
    pub fn verify(
        &self,
        verification_time_unix_seconds: u64,
    ) -> Result<CheckpointSeries21PublicVerificationSummary, CheckpointSeries21PublicVerificationError>
    {
        if self.schema != CHECKPOINT_SERIES21_PUBLIC_VERIFICATION_BUNDLE_SCHEMA
            || self.hybrid_bundle.classical_bundle
                != self.downgrade_candidate.classical_bundle
            || self.hybrid_bundle.policy != self.downgrade_candidate.policy
            || self.gossip_bundle.anchor_head != self.split_view_candidate.anchor_head
            || self.gossip_policy.transparency_authority_key_id
                != self.transparency_authority_key.key_id
        {
            return Err(CheckpointSeries21PublicVerificationError::InvalidBundle);
        }
        let hybrid = self.hybrid_bundle.verify(verification_time_unix_seconds)?;
        let hybrid_downgrade_negative = verify_hybrid_downgrade_negative(
            &self.downgrade_candidate,
            verification_time_unix_seconds,
        )?;
        let gossip = self.gossip_bundle.verify(
            &self.gossip_policy,
            &self.transparency_authority_key,
            verification_time_unix_seconds,
        )?;
        let split_view_negative = verify_transparency_split_view_negative(
            &self.split_view_candidate,
            &self.gossip_policy,
            &self.transparency_authority_key,
            verification_time_unix_seconds,
        )?;
        let summary = CheckpointSeries21PublicVerificationSummary {
            schema: CHECKPOINT_SERIES21_PUBLIC_VERIFICATION_SUMMARY_SCHEMA.to_owned(),
            hybrid,
            hybrid_downgrade_negative,
            gossip,
            split_view_negative,
        };
        summary.validate()?;
        Ok(summary)
    }
}

pub fn encode_checkpoint_series21_public_verification_bundle(
    bundle: &CheckpointSeries21PublicVerificationBundle,
) -> Result<Vec<u8>, CheckpointSeries21PublicVerificationError> {
    let encoded = postcard::to_stdvec(bundle)
        .map_err(|_| CheckpointSeries21PublicVerificationError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointSeries21PublicVerificationError::TooLarge);
    }
    Ok(encoded)
}

pub fn decode_checkpoint_series21_public_verification_bundle(
    encoded: &[u8],
) -> Result<CheckpointSeries21PublicVerificationBundle, CheckpointSeries21PublicVerificationError>
{
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointSeries21PublicVerificationError::TooLarge);
    }
    let bundle: CheckpointSeries21PublicVerificationBundle = postcard::from_bytes(encoded)
        .map_err(|_| CheckpointSeries21PublicVerificationError::Encoding)?;
    if bundle.schema != CHECKPOINT_SERIES21_PUBLIC_VERIFICATION_BUNDLE_SCHEMA {
        return Err(CheckpointSeries21PublicVerificationError::InvalidBundle);
    }
    Ok(bundle)
}

#[derive(Debug)]
pub enum CheckpointSeries21PublicVerificationError {
    InvalidBundle,
    Encoding,
    TooLarge,
    Hybrid(CheckpointHybridVerificationError),
    Gossip(CheckpointTransparencyGossipError),
}

impl std::fmt::Display for CheckpointSeries21PublicVerificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBundle => formatter.write_str("invalid Series 21 public bundle"),
            Self::Encoding => formatter.write_str("Series 21 public bundle encoding failed"),
            Self::TooLarge => formatter.write_str("Series 21 public bundle exceeds its bound"),
            Self::Hybrid(error) => write!(formatter, "hybrid verification failed: {error}"),
            Self::Gossip(error) => write!(formatter, "gossip verification failed: {error}"),
        }
    }
}

impl std::error::Error for CheckpointSeries21PublicVerificationError {}

impl From<CheckpointHybridVerificationError> for CheckpointSeries21PublicVerificationError {
    fn from(error: CheckpointHybridVerificationError) -> Self {
        Self::Hybrid(error)
    }
}

impl From<CheckpointTransparencyGossipError> for CheckpointSeries21PublicVerificationError {
    fn from(error: CheckpointTransparencyGossipError) -> Self {
        Self::Gossip(error)
    }
}
