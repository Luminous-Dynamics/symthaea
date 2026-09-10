// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Proposal-aware semantic verification for persisted Forge bundles.
//!
//! This composes the existing bundle semantic verifier with raw proposal rehydration and exact
//! trace/proposal coverage. It does not assign semantic `DiscoveryRun` meaning; that remains an
//! external qualification step.

use crate::bundle::RAW_PROPOSALS_FILE;
use crate::bundle_verify::{read_completed_bundle, BundleSemanticError, VerifiedForgeBundle};
use crate::proposal_coverage::{
    validate_forge_raw_proposal_coverage, ForgeProposalCoverageError,
};
use crate::proposal_trace::{ForgeRawProposalArchive, ForgeRawProposalError};
use serde_json;
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BundleProposalSemanticError {
    #[error(transparent)]
    Bundle(#[from] BundleSemanticError),
    #[error(transparent)]
    Raw(#[from] ForgeRawProposalError),
    #[error(transparent)]
    Coverage(#[from] ForgeProposalCoverageError),
    #[error("Forge bundle raw proposal archive I/O failed: {0}")]
    Io(String),
    #[error("Forge bundle raw proposal archive JSON decode failed: {0}")]
    Json(String),
}

#[derive(Debug)]
pub struct VerifiedForgeProposalBundle {
    pub base: VerifiedForgeBundle,
    pub raw_proposals: ForgeRawProposalArchive,
}

/// Rehydrate and verify a completed Forge bundle including generator-local proposal evidence.
pub fn read_completed_proposal_bundle(
    root: &Path,
) -> Result<VerifiedForgeProposalBundle, BundleProposalSemanticError> {
    let base = read_completed_bundle(root)?;
    let bytes = fs::read(root.join(RAW_PROPOSALS_FILE))
        .map_err(|error| BundleProposalSemanticError::Io(error.to_string()))?;
    let raw_proposals: ForgeRawProposalArchive = serde_json::from_slice(&bytes)
        .map_err(|error| BundleProposalSemanticError::Json(error.to_string()))?;
    raw_proposals.validate()?;
    validate_forge_raw_proposal_coverage(
        &base.trace,
        &base.observations,
        &raw_proposals,
    )?;
    Ok(VerifiedForgeProposalBundle {
        base,
        raw_proposals,
    })
}
