// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic raw 2x2 estimators for qualified ablation contrast sets.
//!
//! This module consumes one already-validated [`AblationContrastSet`] and one
//! scalar outcome observed in each A/B/C/D cell. It computes signed arithmetic
//! contrasts only. Sign is not interpreted as improvement/degradation, and this
//! contract grants no significance, cross-seed, comparability, or causal authority.
//!
//! Factor-state naming is explicit: `mechanism_intact` means the subsystem is
//! present, while `mechanism_ablated` means the mechanism-only intervention is
//! active. This avoids the ambiguous phrase “mechanism present”.

use crate::ablation_contract::{AblationMechanism, LegacyAblationPresetId};
use crate::ablation_contrast::{AblationContrastError, AblationContrastSet};
use serde::{Deserialize, Serialize};
use std::fmt;

pub const ABLATION_FACTORIAL_ESTIMATE_SCHEMA_VERSION: &str =
    "psych-ablation-factorial-estimate-v1";
const ESTIMATE_DOMAIN: &[u8] = b"symthaea.psych.ablation-factorial-estimate.v1\0";

/// Raw scalar observations for the qualified A/B/C/D cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AblationFactorialCellValues {
    /// A: baseline (mechanism intact, nuisance absent).
    pub baseline: f64,
    /// B: mechanism-only intervention (mechanism ablated, nuisance absent).
    pub mechanism_only: f64,
    /// C: nuisance-only intervention (mechanism intact, nuisance present).
    pub nuisance_only: f64,
    /// D: combined intervention (mechanism ablated, nuisance present).
    pub combined: f64,
}

/// Exact IEEE-754 identities of the four observed cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AblationFactorialCellBits {
    pub baseline: u64,
    pub mechanism_only: u64,
    pub nuisance_only: u64,
    pub combined: u64,
}

impl AblationFactorialCellBits {
    pub fn values(self) -> AblationFactorialCellValues {
        AblationFactorialCellValues {
            baseline: f64::from_bits(self.baseline),
            mechanism_only: f64::from_bits(self.mechanism_only),
            nuisance_only: f64::from_bits(self.nuisance_only),
            combined: f64::from_bits(self.combined),
        }
    }
}

impl From<AblationFactorialCellValues> for AblationFactorialCellBits {
    fn from(values: AblationFactorialCellValues) -> Self {
        Self {
            baseline: values.baseline.to_bits(),
            mechanism_only: values.mechanism_only.to_bits(),
            nuisance_only: values.nuisance_only.to_bits(),
            combined: values.combined.to_bits(),
        }
    }
}

/// Exact IEEE-754 identities of the deterministic raw contrasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AblationFactorialContrastBits {
    /// B - A: mechanism ablation while nuisance is absent.
    pub mechanism_ablation_when_nuisance_absent: u64,
    /// D - C: mechanism ablation while nuisance is present.
    pub mechanism_ablation_when_nuisance_present: u64,
    /// C - A: nuisance intervention while the mechanism is intact.
    pub nuisance_when_mechanism_intact: u64,
    /// D - B: nuisance intervention while the mechanism is ablated.
    pub nuisance_when_mechanism_ablated: u64,
    /// 0.5 * ((B - A) + (D - C)).
    pub mechanism_ablation_main: u64,
    /// 0.5 * ((C - A) + (D - B)).
    pub nuisance_main: u64,
    /// D - B - C + A.
    pub interaction: u64,
}

impl AblationFactorialContrastBits {
    pub fn mechanism_ablation_when_nuisance_absent(self) -> f64 {
        f64::from_bits(self.mechanism_ablation_when_nuisance_absent)
    }

    pub fn mechanism_ablation_when_nuisance_present(self) -> f64 {
        f64::from_bits(self.mechanism_ablation_when_nuisance_present)
    }

    pub fn nuisance_when_mechanism_intact(self) -> f64 {
        f64::from_bits(self.nuisance_when_mechanism_intact)
    }

    pub fn nuisance_when_mechanism_ablated(self) -> f64 {
        f64::from_bits(self.nuisance_when_mechanism_ablated)
    }

    pub fn mechanism_ablation_main(self) -> f64 {
        f64::from_bits(self.mechanism_ablation_main)
    }

    pub fn nuisance_main(self) -> f64 {
        f64::from_bits(self.nuisance_main)
    }

    pub fn interaction(self) -> f64 {
        f64::from_bits(self.interaction)
    }
}

/// Content-addressed arithmetic receipt for one scalar outcome in one contrast set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AblationFactorialEstimateReceipt {
    pub schema_version: String,
    pub contrast_set_id: String,
    pub contrast_digest: String,
    pub source_preset: LegacyAblationPresetId,
    pub mechanism: AblationMechanism,
    pub seed: u64,
    pub outcome_id: String,
    pub cells: AblationFactorialCellBits,
    pub contrasts: AblationFactorialContrastBits,
    pub estimate_digest: String,
}

impl AblationFactorialEstimateReceipt {
    /// Build a raw arithmetic estimate from one qualified contrast set.
    pub fn estimate(
        set: &AblationContrastSet,
        outcome_id: impl Into<String>,
        cells: AblationFactorialCellValues,
    ) -> Result<Self, AblationFactorialError> {
        set.validate()?;
        let outcome_id = outcome_id.into();
        validate_outcome_id(&outcome_id)?;
        validate_cells(cells)?;
        let contrasts = compute_contrasts(cells)?;
        let parent = set.receipt();
        let cell_bits = AblationFactorialCellBits::from(cells);

        let material = AblationFactorialEstimateDigestMaterial {
            schema_version: ABLATION_FACTORIAL_ESTIMATE_SCHEMA_VERSION,
            contrast_set_id: &parent.contrast_set_id,
            contrast_digest: &parent.contrast_digest,
            source_preset: parent.source_preset,
            mechanism: parent.mechanism,
            seed: parent.seed,
            outcome_id: &outcome_id,
            cells: cell_bits,
            contrasts,
        };
        let estimate_digest = estimate_digest_hex(&material)?;

        Ok(Self {
            schema_version: ABLATION_FACTORIAL_ESTIMATE_SCHEMA_VERSION.to_string(),
            contrast_set_id: parent.contrast_set_id.clone(),
            contrast_digest: parent.contrast_digest.clone(),
            source_preset: parent.source_preset,
            mechanism: parent.mechanism,
            seed: parent.seed,
            outcome_id,
            cells: cell_bits,
            contrasts,
            estimate_digest,
        })
    }

    /// Recompute the full receipt from the supplied parent set and stored raw cells.
    pub fn validate_against(
        &self,
        set: &AblationContrastSet,
    ) -> Result<(), AblationFactorialError> {
        if self.schema_version != ABLATION_FACTORIAL_ESTIMATE_SCHEMA_VERSION {
            return Err(AblationFactorialError::UnsupportedSchema);
        }
        validate_outcome_id(&self.outcome_id)?;
        let expected = Self::estimate(set, self.outcome_id.clone(), self.cells.values())?;
        if &expected == self {
            Ok(())
        } else {
            Err(AblationFactorialError::ReceiptMismatch)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AblationFactorialError {
    Parent(AblationContrastError),
    UnsupportedSchema,
    EmptyOutcomeId,
    NonCanonicalOutcomeId,
    NonFiniteCell(&'static str),
    NonFiniteContrast(&'static str),
    ReceiptMismatch,
    Serialization(String),
}

impl From<AblationContrastError> for AblationFactorialError {
    fn from(value: AblationContrastError) -> Self {
        Self::Parent(value)
    }
}

impl fmt::Display for AblationFactorialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parent(error) => write!(f, "invalid parent contrast set: {error:?}"),
            Self::UnsupportedSchema => write!(f, "unsupported ablation factorial estimate schema"),
            Self::EmptyOutcomeId => write!(f, "outcome_id must not be empty"),
            Self::NonCanonicalOutcomeId => {
                write!(f, "outcome_id must not contain leading/trailing whitespace or controls")
            }
            Self::NonFiniteCell(role) => write!(f, "non-finite factorial cell: {role}"),
            Self::NonFiniteContrast(kind) => write!(f, "non-finite factorial contrast: {kind}"),
            Self::ReceiptMismatch => write!(f, "factorial estimate receipt does not revalidate"),
            Self::Serialization(error) => write!(f, "factorial estimate serialization failed: {error}"),
        }
    }
}

impl std::error::Error for AblationFactorialError {}

fn validate_outcome_id(outcome_id: &str) -> Result<(), AblationFactorialError> {
    if outcome_id.is_empty() {
        return Err(AblationFactorialError::EmptyOutcomeId);
    }
    if outcome_id.trim() != outcome_id || outcome_id.chars().any(char::is_control) {
        return Err(AblationFactorialError::NonCanonicalOutcomeId);
    }
    Ok(())
}

fn validate_cells(cells: AblationFactorialCellValues) -> Result<(), AblationFactorialError> {
    for (role, value) in [
        ("baseline", cells.baseline),
        ("mechanism_only", cells.mechanism_only),
        ("nuisance_only", cells.nuisance_only),
        ("combined", cells.combined),
    ] {
        if !value.is_finite() {
            return Err(AblationFactorialError::NonFiniteCell(role));
        }
    }
    Ok(())
}

fn compute_contrasts(
    cells: AblationFactorialCellValues,
) -> Result<AblationFactorialContrastBits, AblationFactorialError> {
    let a = cells.baseline;
    let b = cells.mechanism_only;
    let c = cells.nuisance_only;
    let d = cells.combined;

    let mechanism_ablation_when_nuisance_absent = b - a;
    let mechanism_ablation_when_nuisance_present = d - c;
    let nuisance_when_mechanism_intact = c - a;
    let nuisance_when_mechanism_ablated = d - b;
    let mechanism_ablation_main =
        0.5 * (mechanism_ablation_when_nuisance_absent + mechanism_ablation_when_nuisance_present);
    let nuisance_main =
        0.5 * (nuisance_when_mechanism_intact + nuisance_when_mechanism_ablated);
    let interaction = d - b - c + a;

    let named = [
        (
            "mechanism_ablation_when_nuisance_absent",
            mechanism_ablation_when_nuisance_absent,
        ),
        (
            "mechanism_ablation_when_nuisance_present",
            mechanism_ablation_when_nuisance_present,
        ),
        (
            "nuisance_when_mechanism_intact",
            nuisance_when_mechanism_intact,
        ),
        (
            "nuisance_when_mechanism_ablated",
            nuisance_when_mechanism_ablated,
        ),
        ("mechanism_ablation_main", mechanism_ablation_main),
        ("nuisance_main", nuisance_main),
        ("interaction", interaction),
    ];
    for (kind, value) in named {
        if !value.is_finite() {
            return Err(AblationFactorialError::NonFiniteContrast(kind));
        }
    }

    Ok(AblationFactorialContrastBits {
        mechanism_ablation_when_nuisance_absent: mechanism_ablation_when_nuisance_absent.to_bits(),
        mechanism_ablation_when_nuisance_present: mechanism_ablation_when_nuisance_present.to_bits(),
        nuisance_when_mechanism_intact: nuisance_when_mechanism_intact.to_bits(),
        nuisance_when_mechanism_ablated: nuisance_when_mechanism_ablated.to_bits(),
        mechanism_ablation_main: mechanism_ablation_main.to_bits(),
        nuisance_main: nuisance_main.to_bits(),
        interaction: interaction.to_bits(),
    })
}

#[derive(Serialize)]
struct AblationFactorialEstimateDigestMaterial<'a> {
    schema_version: &'a str,
    contrast_set_id: &'a str,
    contrast_digest: &'a str,
    source_preset: LegacyAblationPresetId,
    mechanism: AblationMechanism,
    seed: u64,
    outcome_id: &'a str,
    cells: AblationFactorialCellBits,
    contrasts: AblationFactorialContrastBits,
}

fn estimate_digest_hex(
    material: &AblationFactorialEstimateDigestMaterial<'_>,
) -> Result<String, AblationFactorialError> {
    let bytes = serde_json::to_vec(material)
        .map_err(|error| AblationFactorialError::Serialization(error.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(ESTIMATE_DOMAIN);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::AblationPreset;

    fn set(seed: u64) -> AblationContrastSet {
        AblationContrastSet::single_mechanism(AblationPreset::NoFep, seed).unwrap()
    }

    fn cells() -> AblationFactorialCellValues {
        AblationFactorialCellValues {
            baseline: 1.0,
            mechanism_only: 3.0,
            nuisance_only: 2.0,
            combined: 8.0,
        }
    }

    #[test]
    fn raw_factorial_estimator_matches_factor_states_and_preregistered_arithmetic() {
        let parent = set(42);
        let values = cells();
        let receipt =
            AblationFactorialEstimateReceipt::estimate(&parent, "score", values).unwrap();

        assert_eq!(
            receipt.contrasts.mechanism_ablation_when_nuisance_absent(),
            values.mechanism_only - values.baseline,
        );
        assert_eq!(
            receipt.contrasts.mechanism_ablation_when_nuisance_present(),
            values.combined - values.nuisance_only,
        );
        assert_eq!(
            receipt.contrasts.nuisance_when_mechanism_intact(),
            values.nuisance_only - values.baseline,
        );
        assert_eq!(
            receipt.contrasts.nuisance_when_mechanism_ablated(),
            values.combined - values.mechanism_only,
        );
        assert_eq!(receipt.contrasts.mechanism_ablation_main(), 4.0);
        assert_eq!(receipt.contrasts.nuisance_main(), 3.0);
        assert_eq!(receipt.contrasts.interaction(), 4.0);
        assert!(receipt.validate_against(&parent).is_ok());
    }

    #[test]
    fn raw_factorial_estimator_preserves_sign_without_direction_semantics() {
        let receipt = AblationFactorialEstimateReceipt::estimate(
            &set(42),
            "signed-score",
            AblationFactorialCellValues {
                baseline: 8.0,
                mechanism_only: 3.0,
                nuisance_only: 6.0,
                combined: 1.0,
            },
        )
        .unwrap();
        assert!(receipt.contrasts.mechanism_ablation_main() < 0.0);
        assert!(receipt.contrasts.nuisance_main() < 0.0);
    }

    #[test]
    fn exact_float_bits_and_signed_zero_are_identity_bearing() {
        let parent = set(42);
        let positive = AblationFactorialEstimateReceipt::estimate(
            &parent,
            "zero-sensitive",
            AblationFactorialCellValues {
                baseline: 0.0,
                mechanism_only: 1.0,
                nuisance_only: 2.0,
                combined: 3.0,
            },
        )
        .unwrap();
        let negative = AblationFactorialEstimateReceipt::estimate(
            &parent,
            "zero-sensitive",
            AblationFactorialCellValues {
                baseline: -0.0,
                mechanism_only: 1.0,
                nuisance_only: 2.0,
                combined: 3.0,
            },
        )
        .unwrap();
        assert_ne!(positive.cells.baseline, negative.cells.baseline);
        assert_ne!(positive.estimate_digest, negative.estimate_digest);
    }

    #[test]
    fn nonfinite_cells_and_overflowed_contrasts_fail_closed() {
        let parent = set(42);
        assert!(matches!(
            AblationFactorialEstimateReceipt::estimate(
                &parent,
                "score",
                AblationFactorialCellValues {
                    baseline: f64::NAN,
                    ..cells()
                },
            ),
            Err(AblationFactorialError::NonFiniteCell("baseline"))
        ));
        assert!(matches!(
            AblationFactorialEstimateReceipt::estimate(
                &parent,
                "score",
                AblationFactorialCellValues {
                    baseline: -f64::MAX,
                    mechanism_only: f64::MAX,
                    nuisance_only: 0.0,
                    combined: 0.0,
                },
            ),
            Err(AblationFactorialError::NonFiniteContrast(_))
        ));
    }

    #[test]
    fn outcome_identity_must_be_canonical() {
        let parent = set(42);
        assert_eq!(
            AblationFactorialEstimateReceipt::estimate(&parent, "", cells()),
            Err(AblationFactorialError::EmptyOutcomeId)
        );
        assert_eq!(
            AblationFactorialEstimateReceipt::estimate(&parent, " score", cells()),
            Err(AblationFactorialError::NonCanonicalOutcomeId)
        );
        assert_eq!(
            AblationFactorialEstimateReceipt::estimate(&parent, "score\nother", cells()),
            Err(AblationFactorialError::NonCanonicalOutcomeId)
        );
    }

    #[test]
    fn receipt_revalidation_rejects_parent_cell_contrast_and_outcome_tampering() {
        let parent = set(42);
        let receipt = AblationFactorialEstimateReceipt::estimate(&parent, "score", cells()).unwrap();

        assert!(matches!(
            receipt.validate_against(&set(43)),
            Err(AblationFactorialError::ReceiptMismatch)
        ));

        let mut cell_tamper = receipt.clone();
        cell_tamper.cells.combined = 9.0f64.to_bits();
        assert_eq!(
            cell_tamper.validate_against(&parent),
            Err(AblationFactorialError::ReceiptMismatch)
        );

        let mut contrast_tamper = receipt.clone();
        contrast_tamper.contrasts.interaction = 99.0f64.to_bits();
        assert_eq!(
            contrast_tamper.validate_against(&parent),
            Err(AblationFactorialError::ReceiptMismatch)
        );

        let mut outcome_tamper = receipt;
        outcome_tamper.outcome_id = "other-score".into();
        assert_eq!(
            outcome_tamper.validate_against(&parent),
            Err(AblationFactorialError::ReceiptMismatch)
        );
    }

    #[test]
    fn serde_roundtrip_preserves_exact_receipt() {
        let parent = set(42);
        let receipt = AblationFactorialEstimateReceipt::estimate(&parent, "score", cells()).unwrap();
        let json = serde_json::to_vec(&receipt).unwrap();
        let decoded: AblationFactorialEstimateReceipt = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded, receipt);
        decoded.validate_against(&parent).unwrap();
    }
}
