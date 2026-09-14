// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-anchored calibration inventory for incremental benchmark migration.
//!
//! This layer is deliberately weaker than an authority-bearing
//! `CalibrationManifest`: source observation does not by itself prove that the
//! complete scientific parameter surface has been enumerated, nor that an
//! observed expression is the value that reached a scored runtime path.
//!
//! The migration invariant is therefore:
//!
//! ```text
//! source-anchored inventory
//!     != complete calibration manifest
//!     != frozen-holdout authority
//! ```
//!
//! Inventories can be inspected, hashed and diffed. This module intentionally
//! exposes no conversion into `CalibrationManifest`; completeness and runtime
//! binding require a later, independently reviewed tranche.

use crate::calibration_contract::{
    CalibrationClass, CalibrationManifest, CalibrationParameter, CalibrationParameterSource,
    ComparisonTargetClass,
};
use serde::{Deserialize, Serialize};

pub const CALIBRATION_INVENTORY_SCHEMA_VERSION: &str = "psych-calibration-inventory-v1";

/// Descriptive source-coverage state only. Neither variant grants scientific
/// authority or permits conversion into `CalibrationManifest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationInventoryCoverage {
    /// Known calibration-relevant expressions have been captured, but omitted
    /// source/runtime/config surfaces may remain.
    Partial,
    /// A source audit declares the relevant source expressions enumerated.
    /// Runtime reachability and scientific completeness are still unproven.
    SourceEnumerated,
}

/// Source-observed calibration inventory for one benchmark revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationInventory {
    pub schema_version: String,
    pub benchmark: String,
    pub revision: u32,
    pub comparison_target: ComparisonTargetClass,
    pub coverage: CalibrationInventoryCoverage,
    pub source_path: String,
    pub parameters: Vec<CalibrationParameter>,
}

impl CalibrationInventory {
    pub fn new(
        benchmark: impl Into<String>,
        revision: u32,
        comparison_target: ComparisonTargetClass,
        coverage: CalibrationInventoryCoverage,
        source_path: impl Into<String>,
        parameters: Vec<CalibrationParameter>,
    ) -> Self {
        Self {
            schema_version: CALIBRATION_INVENTORY_SCHEMA_VERSION.to_string(),
            benchmark: benchmark.into(),
            revision,
            comparison_target,
            coverage,
            source_path: source_path.into(),
            parameters,
        }
    }

    pub fn validate(&self) -> Result<(), CalibrationInventoryError> {
        if self.schema_version != CALIBRATION_INVENTORY_SCHEMA_VERSION {
            return Err(CalibrationInventoryError::UnsupportedSchema);
        }
        if self.source_path.trim().is_empty() {
            return Err(CalibrationInventoryError::EmptySourcePath);
        }

        // Reuse the base manifest's structural validation without exposing the
        // resulting manifest as authority-bearing output.
        self.unpromoted_manifest()
            .validate()
            .map_err(|_| CalibrationInventoryError::InvalidBaseManifest)
    }

    /// Stable digest for source-audit comparison. Coverage and source identity
    /// are committed in addition to the underlying calibration-data digest.
    pub fn digest_hex(&self) -> Result<String, CalibrationInventoryError> {
        self.validate()?;
        let base_digest = self
            .unpromoted_manifest()
            .digest_hex()
            .map_err(|_| CalibrationInventoryError::InvalidBaseManifest)?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.psych.calibration-inventory.v1\0");
        hash_field(&mut hasher, self.schema_version.as_bytes());
        hash_field(&mut hasher, self.source_path.as_bytes());
        hasher.update(&[match self.coverage {
            CalibrationInventoryCoverage::Partial => 1,
            CalibrationInventoryCoverage::SourceEnumerated => 2,
        }]);
        hash_field(&mut hasher, base_digest.as_bytes());
        Ok(hasher.finalize().to_hex().to_string())
    }

    /// Internal structural reuse only. There is intentionally no public
    /// promotion/conversion API in this tranche.
    fn unpromoted_manifest(&self) -> CalibrationManifest {
        CalibrationManifest::new(
            self.benchmark.clone(),
            self.revision,
            self.comparison_target,
            self.parameters.clone(),
        )
    }
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationInventoryError {
    UnsupportedSchema,
    EmptySourcePath,
    InvalidBaseManifest,
    SourceAnchorMissing,
    EmptyObservedExpression,
}

fn collapse_ascii_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn extract_expression(
    source: &str,
    marker: &str,
    terminator: char,
) -> Result<String, CalibrationInventoryError> {
    let start = source
        .find(marker)
        .ok_or(CalibrationInventoryError::SourceAnchorMissing)?
        + marker.len();
    let rest = &source[start..];
    let end = rest
        .find(terminator)
        .ok_or(CalibrationInventoryError::SourceAnchorMissing)?;
    let observed = collapse_ascii_whitespace(&rest[..end]);
    if observed.is_empty() {
        return Err(CalibrationInventoryError::EmptyObservedExpression);
    }
    Ok(observed)
}

/// Extract an anonymous-loop count only after a unique semantic anchor.
///
/// This avoids silently binding to an unrelated `for _ in 0..N` that might be
/// introduced earlier in the same source file.
fn extract_loop_count_after(
    source: &str,
    anchor: &str,
    marker: &str,
) -> Result<String, CalibrationInventoryError> {
    let anchor_start = source
        .find(anchor)
        .ok_or(CalibrationInventoryError::SourceAnchorMissing)?
        + anchor.len();
    let anchored = &source[anchor_start..];
    let count_start = anchored
        .find(marker)
        .ok_or(CalibrationInventoryError::SourceAnchorMissing)?
        + marker.len();
    let digits = anchored[count_start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return Err(CalibrationInventoryError::EmptyObservedExpression);
    }
    Ok(digits)
}

fn source_parameter(
    name: &str,
    canonical_value: String,
    class: CalibrationClass,
) -> CalibrationParameter {
    CalibrationParameter::new(
        name,
        canonical_value,
        class,
        CalibrationParameterSource::BenchmarkLocal,
    )
}

/// Partial source inventory for the current Stroop implementation.
///
/// This intentionally records only the historically identified calibrated
/// expressions in tranche 1. It is not a complete runtime parameter surface.
pub fn stroop_source_inventory() -> Result<CalibrationInventory, CalibrationInventoryError> {
    const SOURCE_PATH: &str = "src/benchmarks/executive/stroop.rs";
    let source = include_str!("benchmarks/executive/stroop.rs");

    let base_automaticity = extract_expression(source, "let base_automaticity: f32 = ", ';')?;
    let base_temperature = extract_expression(source, "let base_temperature: f64 = ", ';')?;

    Ok(CalibrationInventory::new(
        "Executive::Stroop",
        1,
        ComparisonTargetClass::HumanEmpirical,
        CalibrationInventoryCoverage::Partial,
        SOURCE_PATH,
        vec![
            source_parameter(
                "base_automaticity",
                base_automaticity,
                CalibrationClass::PostHoc,
            ),
            source_parameter(
                "base_temperature_expression",
                base_temperature,
                CalibrationClass::PostHoc,
            ),
        ],
    ))
}

/// Partial source inventory for the current N-back implementation.
///
/// The historical audit classified the `0.50` base threshold as a-priori, but
/// the executable expression also contains time-pressure and noise coefficients
/// whose selection provenance is not established here. The composite expression
/// therefore remains `Ambiguous` rather than inheriting the base term's class.
pub fn nback_source_inventory() -> Result<CalibrationInventory, CalibrationInventoryError> {
    const SOURCE_PATH: &str = "src/benchmarks/worm/nback.rs";
    let source = include_str!("benchmarks/worm/nback.rs");

    let base_threshold = extract_expression(source, "let base_threshold = ", ';')?;

    Ok(CalibrationInventory::new(
        "WorM::N-back",
        1,
        ComparisonTargetClass::HumanEmpirical,
        CalibrationInventoryCoverage::Partial,
        SOURCE_PATH,
        vec![source_parameter(
            "base_threshold_expression",
            base_threshold,
            CalibrationClass::Ambiguous,
        )],
    ))
}

/// Partial source inventory for the current synthetic ARC-Fluid implementation.
///
/// The benchmark explicitly states that it exercises HDC algebra rather than
/// genuine ARC generalization. Its quoted human baselines therefore do not map
/// cleanly onto this synthetic task, so comparison-target provenance remains
/// ambiguous in this tranche.
pub fn arc_fluid_source_inventory() -> Result<CalibrationInventory, CalibrationInventoryError> {
    const SOURCE_PATH: &str = "src/benchmarks/reasoning/arc_fluid.rs";
    let source = include_str!("benchmarks/reasoning/arc_fluid.rs");

    let noise_weight = extract_expression(source, "let noise_weight = ", ';')?;
    let training_pairs = extract_loop_count_after(
        source,
        "let mut train_rules = Vec::new();",
        "for _ in 0..",
    )?;

    Ok(CalibrationInventory::new(
        "Reasoning::ArcFluid",
        1,
        ComparisonTargetClass::Ambiguous,
        CalibrationInventoryCoverage::Partial,
        SOURCE_PATH,
        vec![
            source_parameter(
                "noise_weight_expression",
                noise_weight,
                CalibrationClass::PostHoc,
            ),
            source_parameter(
                "training_pairs",
                training_pairs,
                CalibrationClass::PostHoc,
            ),
        ],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_state_is_committed_in_inventory_digest_without_granting_authority() {
        let partial = stroop_source_inventory().unwrap();
        let mut enumerated = partial.clone();
        enumerated.coverage = CalibrationInventoryCoverage::SourceEnumerated;
        assert_ne!(partial.digest_hex().unwrap(), enumerated.digest_hex().unwrap());
    }

    #[test]
    fn stroop_inventory_is_anchored_to_current_source_expressions() {
        let inventory = stroop_source_inventory().unwrap();
        assert_eq!(inventory.coverage, CalibrationInventoryCoverage::Partial);
        assert_eq!(inventory.parameters[0].canonical_value, "0.35");
        assert_eq!(
            inventory.parameters[1].canonical_value,
            "0.25 + config.time_pressure * 0.15"
        );
        assert_eq!(
            inventory.comparison_target,
            ComparisonTargetClass::HumanEmpirical
        );
    }

    #[test]
    fn nback_inventory_is_anchored_without_overclaiming_composite_provenance() {
        let inventory = nback_source_inventory().unwrap();
        assert_eq!(
            inventory.parameters[0].canonical_value,
            "0.50 - config.time_pressure * 0.25 - noise * 0.15"
        );
        assert_eq!(inventory.parameters[0].class, CalibrationClass::Ambiguous);
    }

    #[test]
    fn arc_fluid_inventory_tracks_current_source_without_upgrading_target_provenance() {
        let inventory = arc_fluid_source_inventory().unwrap();
        assert_eq!(inventory.benchmark, "Reasoning::ArcFluid");
        assert_eq!(inventory.parameters[0].canonical_value, "(0.008 + pressure * 0.12 + config.encoding_noise * 0.15) * diff_model.temperature_multiplier(config.difficulty)");
        assert_eq!(inventory.parameters[1].canonical_value, "6");
        assert_eq!(inventory.comparison_target, ComparisonTargetClass::Ambiguous);
    }

    #[test]
    fn contextual_loop_anchor_ignores_unrelated_earlier_anonymous_loop() {
        let source = "for _ in 0..99 { unrelated(); }\nlet mut train_rules = Vec::new();\nfor _ in 0..6 { train(); }";
        assert_eq!(
            extract_loop_count_after(source, "let mut train_rules = Vec::new();", "for _ in 0..")
                .unwrap(),
            "6"
        );
    }

    #[test]
    fn source_anchor_failure_is_fail_closed() {
        assert_eq!(
            extract_expression("let x = 1;", "let missing = ", ';'),
            Err(CalibrationInventoryError::SourceAnchorMissing)
        );
    }

    #[test]
    fn source_expression_drift_changes_inventory_digest() {
        let first = CalibrationInventory::new(
            "Synthetic",
            1,
            ComparisonTargetClass::HumanEmpirical,
            CalibrationInventoryCoverage::Partial,
            "synthetic.rs",
            vec![source_parameter(
                "threshold_expression",
                extract_expression("let threshold = 0.50;", "let threshold = ", ';').unwrap(),
                CalibrationClass::APriori,
            )],
        );
        let second = CalibrationInventory::new(
            "Synthetic",
            1,
            ComparisonTargetClass::HumanEmpirical,
            CalibrationInventoryCoverage::Partial,
            "synthetic.rs",
            vec![source_parameter(
                "threshold_expression",
                extract_expression("let threshold = 0.55;", "let threshold = ", ';').unwrap(),
                CalibrationClass::APriori,
            )],
        );
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn comparison_target_drift_changes_inventory_digest() {
        let first = arc_fluid_source_inventory().unwrap();
        let mut second = first.clone();
        second.comparison_target = ComparisonTargetClass::HumanEmpirical;
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn serialization_preserves_non_authoritative_coverage() {
        let inventory = arc_fluid_source_inventory().unwrap();
        let json = serde_json::to_string(&inventory).unwrap();
        let decoded: CalibrationInventory = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, inventory);
        assert_eq!(decoded.coverage, CalibrationInventoryCoverage::Partial);
    }
}
