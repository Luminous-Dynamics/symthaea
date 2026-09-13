// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-authenticated official execution facade for Matbench Benchmark Zero.
//!
//! Unlike the lower-level frozen-measurement kernel, the public entry point in
//! this crate accepts the compressed artifact bytes and invokes the pinned
//! #1722 parser internally. Callers cannot supply a hand-constructed
//! `MatbenchGapDataset` to the authoritative official path.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use symthaea_matbench_gap::{
    AdapterError, MATBENCH_EXPT_GAP_SHA256, parse_official_matbench_expt_gap,
};
use symthaea_matbench_gap_comparison_freeze::ComparisonFreezeReceipt;
use symthaea_matbench_gap_exposure_plan::ExposedTrainingExclusionPlan;
use symthaea_matbench_gap_frozen_measurement::{
    FrozenComparisonMeasurementReceipt, FrozenMeasurementError, RestrictedTruthReceipt,
    measure_frozen_comparison, restrict_official_truth_after_freeze,
};
use thiserror::Error;

pub const RECEIPT_SCHEMA: &str =
    "symthaea.matbench-gap.official-frozen-measurement.v0";
pub const CAPABILITY_CLASSIFICATION: &str =
    "SOURCE-AUTHENTICATED OFFICIAL MATBENCH MEASUREMENT RECEIPT -- measurement only; not a clean-holdout certificate, superiority claim, material certification, or promotion authority.";

const RECEIPT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.official-frozen-measurement-receipt.v0\0";

/// Replayable official-source measurement evidence.
///
/// Construction through [`measure_official_matbench_bytes`] starts from the
/// compressed source bytes and therefore necessarily crosses #1722's exact
/// SHA-256 gate before any truth restriction or measurement occurs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OfficialFrozenMeasurementReceipt {
    pub schema: String,
    pub capability_classification: String,
    /// Exact compressed source-artifact SHA-256 required by the official parser.
    pub source_artifact_sha256: String,
    /// Lower-level post-freeze truth restriction evidence.
    pub restricted_truth: RestrictedTruthReceipt,
    /// Lower-level exact-universe measurement evidence.
    pub measurement: FrozenComparisonMeasurementReceipt,
}

impl OfficialFrozenMeasurementReceipt {
    /// Validate internal receipt composition only.
    ///
    /// This cannot authenticate source bytes by itself. Use
    /// [`verify_official_matbench_measurement`] with the compressed artifact for
    /// the source-authentication theorem.
    pub fn validate(&self) -> Result<(), OfficialMeasurementError> {
        if self.schema != RECEIPT_SCHEMA
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(OfficialMeasurementError::InvalidReceipt(
                "schema or capability classification was altered".into(),
            ));
        }
        if self.source_artifact_sha256 != MATBENCH_EXPT_GAP_SHA256 {
            return Err(OfficialMeasurementError::InvalidReceipt(
                "source artifact digest is not the pinned official Matbench digest".into(),
            ));
        }
        self.restricted_truth.validate()?;
        self.measurement.validate()?;

        if self.restricted_truth.parent_compressed_sha256 != self.source_artifact_sha256
            || self.measurement.source_plan_sha256 != self.restricted_truth.source_plan_sha256
            || self.measurement.comparison_subject_sha256
                != self.restricted_truth.comparison_subject_sha256
            || self.measurement.restricted_truth_sha256
                != self.restricted_truth.restricted_truth_sha256
            || self.measurement.candidate_universe_sha256
                != self.restricted_truth.candidate_universe_sha256
            || self.measurement.candidate_count != self.restricted_truth.candidate_count
            || self.measurement.restricted_truth_receipt_sha256
                != self.restricted_truth.sha256()?
        {
            return Err(OfficialMeasurementError::InvalidReceipt(
                "nested restricted-truth and measurement receipts are not bound to the same source/comparison universe"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, OfficialMeasurementError> {
        self.validate()?;
        domain_separated_sha256(RECEIPT_DIGEST_DOMAIN, self)
    }
}

/// Execute the authoritative official Benchmark Zero measurement path.
///
/// The compressed bytes are parsed internally with
/// `parse_official_matbench_expt_gap`, which verifies the exact upstream
/// SHA-256 before decompression and schema parsing. Only that parsed value is
/// then passed to the lower-level restriction/measurement kernel.
pub fn measure_official_matbench_bytes(
    plan: &ExposedTrainingExclusionPlan,
    freeze: &ComparisonFreezeReceipt,
    compressed_bytes: &[u8],
) -> Result<OfficialFrozenMeasurementReceipt, OfficialMeasurementError> {
    let dataset = parse_official_matbench_expt_gap(compressed_bytes)?;
    let restricted = restrict_official_truth_after_freeze(plan, freeze, &dataset)?;
    let measurement = measure_frozen_comparison(freeze, &restricted)?;

    let receipt = OfficialFrozenMeasurementReceipt {
        schema: RECEIPT_SCHEMA.into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        source_artifact_sha256: dataset.compressed_sha256,
        restricted_truth: restricted,
        measurement,
    };
    receipt.validate()?;
    Ok(receipt)
}

/// Replay source authentication, truth restriction, and all frozen policy
/// measurements from the compressed bytes and require exact receipt equality.
pub fn verify_official_matbench_measurement(
    plan: &ExposedTrainingExclusionPlan,
    freeze: &ComparisonFreezeReceipt,
    compressed_bytes: &[u8],
    expected: &OfficialFrozenMeasurementReceipt,
) -> Result<(), OfficialMeasurementError> {
    expected.validate()?;
    let observed = measure_official_matbench_bytes(plan, freeze, compressed_bytes)?;
    if &observed != expected {
        return Err(OfficialMeasurementError::ReplayMismatch);
    }
    Ok(())
}

fn domain_separated_sha256<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, OfficialMeasurementError> {
    let encoded = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(encoded);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

#[derive(Debug, Error)]
pub enum OfficialMeasurementError {
    #[error("official Matbench parser rejected source bytes: {0}")]
    Adapter(#[from] AdapterError),
    #[error("frozen measurement kernel rejected evidence: {0}")]
    Measurement(#[from] FrozenMeasurementError),
    #[error("invalid official measurement receipt: {0}")]
    InvalidReceipt(String),
    #[error("official measurement replay differs from supplied receipt")]
    ReplayMismatch,
    #[error("JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_source_pin_is_frozen() {
        assert_eq!(
            MATBENCH_EXPT_GAP_SHA256,
            "783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2"
        );
    }
}
