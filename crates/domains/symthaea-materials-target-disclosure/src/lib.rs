// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-granularity authority for historical materials benchmark targets.
//!
//! A published abstract can establish a descriptive fact without supplying the exact
//! structure, method, or conditions needed for a quantitative benchmark label. This
//! crate makes that distinction machine-visible so retrospective benchmarks cannot
//! silently promote coarse publication metadata into a stronger answer key.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use thiserror::Error;

/// Public/source granularity of one captured disclosure artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisclosureAccess {
    /// Public abstract/metadata only.
    Abstract,
    /// Full article text.
    FullText,
    /// Supplementary information.
    Supplement,
    /// Author/repository structure/property dataset.
    Dataset,
}

impl DisclosureAccess {
    fn supports_structure_targets(self) -> bool {
        matches!(self, Self::FullText | Self::Supplement | Self::Dataset)
    }

    fn supports_quantitative_property_targets(self) -> bool {
        matches!(self, Self::Supplement | Self::Dataset)
    }
}

/// Exact source artifact from which target facts were transcribed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisclosureSource {
    /// Stable source identifier.
    pub source_id: String,
    /// DOI or other stable publication identifier.
    pub publication_id: String,
    /// Source locator. This is navigation metadata, not content identity.
    pub locator: String,
    /// SHA-256 of the exact captured source bytes used for transcription.
    pub source_artifact_sha256: String,
    /// Granularity of those captured bytes.
    pub access: DisclosureAccess,
    /// Gregorian disclosure/publication date `YYYY-MM-DD`.
    pub disclosure_date: String,
}

impl DisclosureSource {
    /// Validate source identity before target facts may refer to it.
    pub fn validate(&self) -> Result<(), TargetDisclosureError> {
        for (name, value) in [
            ("source_id", &self.source_id),
            ("publication_id", &self.publication_id),
            ("locator", &self.locator),
            ("disclosure_date", &self.disclosure_date),
        ] {
            nonempty(name, value)?;
        }
        sha256(&self.source_artifact_sha256)?;
        validate_date(&self.disclosure_date)
    }
}

/// How strongly one source fact can be used in a benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkTargetAuthority {
    /// Publication fact is descriptive only and must not become a scored target.
    DescriptiveOnly,
    /// Exact composition identity may be scored, but no structure/property claim follows.
    CompositionIdentity,
    /// Exact structure identity may be scored.
    StructureIdentity,
    /// Exact quantitative property label may be scored under its bound structure/method/conditions.
    QuantitativeProperty,
}

/// Role attached to a composition mention in the disclosure source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionRole {
    /// Source calls this material a discovered/search result.
    DiscoveredCandidate,
    /// Source identifies this as a parent/precursor used for later optimization.
    OptimizationPrecursor,
    /// Source reports a chemically substituted/optimized candidate.
    SubstitutedCandidate,
    /// Source reports the material as thermodynamically stable.
    ReportedStable,
    /// Other source-language role preserved without semantic promotion.
    Other(String),
}

/// One publication fact available to target-manifest construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisclosureFact {
    /// Aggregate discovery count/threshold reported by the source.
    AggregateCount {
        /// Stable local fact ID.
        fact_id: String,
        /// Human/machine category, e.g. `thermodynamically_stable_ternaries`.
        category: String,
        /// Reported count.
        count: u32,
        /// Optional exact canonical decimal threshold.
        threshold_value: Option<String>,
        /// Optional threshold unit.
        threshold_unit: Option<String>,
    },
    /// Exact composition identity mentioned by the source, without assuming structure identity.
    CompositionMention {
        /// Stable local fact ID.
        fact_id: String,
        /// Canonical composition SHA-256.
        composition_sha256: String,
        /// Human-readable formula for audit/navigation only.
        formula: String,
        /// Source role.
        role: CompositionRole,
    },
    /// Exact structure identity sourced from sufficiently detailed bytes.
    StructureTarget {
        /// Stable local fact ID.
        fact_id: String,
        /// Canonical composition SHA-256.
        composition_sha256: String,
        /// Canonical structure SHA-256.
        structure_sha256: String,
        /// Artifact containing the exact structure realization (e.g. CIF/POSCAR bundle).
        structure_artifact_sha256: String,
    },
    /// Numeric property statement transcribed from the source.
    QuantitativePropertyMention {
        /// Stable local fact ID.
        fact_id: String,
        /// Canonical composition SHA-256.
        composition_sha256: String,
        /// Optional exact structure identity.
        structure_sha256: Option<String>,
        /// Stable scientific property identifier.
        property_id: String,
        /// Exact canonical decimal text.
        value: String,
        /// Exact unit string.
        unit: String,
        /// Exact method/condition signature when the source establishes it.
        condition_signature: Option<String>,
        /// Exact source/method artifact establishing the numeric label when available.
        method_artifact_sha256: Option<String>,
    },
}

impl DisclosureFact {
    fn fact_id(&self) -> &str {
        match self {
            Self::AggregateCount { fact_id, .. }
            | Self::CompositionMention { fact_id, .. }
            | Self::StructureTarget { fact_id, .. }
            | Self::QuantitativePropertyMention { fact_id, .. } => fact_id,
        }
    }

    fn validate(&self) -> Result<(), TargetDisclosureError> {
        nonempty("fact_id", self.fact_id())?;
        match self {
            Self::AggregateCount {
                category,
                count,
                threshold_value,
                threshold_unit,
                ..
            } => {
                nonempty("aggregate category", category)?;
                if *count == 0 {
                    return Err(TargetDisclosureError::ZeroAggregateCount);
                }
                match (threshold_value, threshold_unit) {
                    (Some(value), Some(unit)) => {
                        validate_decimal(value)?;
                        nonempty("threshold_unit", unit)
                    }
                    (None, None) => Ok(()),
                    _ => Err(TargetDisclosureError::IncompleteThreshold),
                }
            }
            Self::CompositionMention {
                composition_sha256,
                formula,
                ..
            } => {
                sha256(composition_sha256)?;
                nonempty("formula", formula)
            }
            Self::StructureTarget {
                composition_sha256,
                structure_sha256,
                structure_artifact_sha256,
                ..
            } => {
                sha256(composition_sha256)?;
                sha256(structure_sha256)?;
                sha256(structure_artifact_sha256)
            }
            Self::QuantitativePropertyMention {
                composition_sha256,
                structure_sha256,
                property_id,
                value,
                unit,
                condition_signature,
                method_artifact_sha256,
                ..
            } => {
                sha256(composition_sha256)?;
                if let Some(structure) = structure_sha256 {
                    sha256(structure)?;
                }
                nonempty("property_id", property_id)?;
                validate_decimal(value)?;
                nonempty("unit", unit)?;
                if let Some(signature) = condition_signature {
                    nonempty("condition_signature", signature)?;
                }
                if let Some(method) = method_artifact_sha256 {
                    sha256(method)?;
                }
                Ok(())
            }
        }
    }

    /// Compute the maximum benchmark authority justified by this fact and source granularity.
    pub fn authority(
        &self,
        source: &DisclosureSource,
    ) -> Result<BenchmarkTargetAuthority, TargetDisclosureError> {
        source.validate()?;
        self.validate()?;
        Ok(match self {
            Self::AggregateCount { .. } => BenchmarkTargetAuthority::DescriptiveOnly,
            Self::CompositionMention { .. } => BenchmarkTargetAuthority::CompositionIdentity,
            Self::StructureTarget { .. } if source.access.supports_structure_targets() => {
                BenchmarkTargetAuthority::StructureIdentity
            }
            Self::StructureTarget { .. } => BenchmarkTargetAuthority::DescriptiveOnly,
            Self::QuantitativePropertyMention {
                structure_sha256: Some(_),
                condition_signature: Some(signature),
                method_artifact_sha256: Some(_),
                ..
            } if !signature.trim().is_empty()
                && source.access.supports_quantitative_property_targets() =>
            {
                BenchmarkTargetAuthority::QuantitativeProperty
            }
            Self::QuantitativePropertyMention { .. } => BenchmarkTargetAuthority::DescriptiveOnly,
        })
    }
}

/// Frozen target-disclosure manifest transcribed from one exact source artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetDisclosureManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Exact source artifact.
    pub source: DisclosureSource,
    /// Source facts in canonical fact-ID order.
    pub facts: Vec<DisclosureFact>,
}

impl TargetDisclosureManifest {
    /// Validate canonical ordering, uniqueness, and every source fact.
    pub fn validate(&self) -> Result<(), TargetDisclosureError> {
        if self.schema_version != 1 {
            return Err(TargetDisclosureError::UnsupportedSchema(self.schema_version));
        }
        self.source.validate()?;
        if self.facts.is_empty() {
            return Err(TargetDisclosureError::EmptyFactSet);
        }
        let mut ids = HashSet::new();
        let mut previous: Option<&str> = None;
        for fact in &self.facts {
            fact.validate()?;
            let id = fact.fact_id();
            if !ids.insert(id) {
                return Err(TargetDisclosureError::DuplicateFactId(id.to_string()));
            }
            if previous.is_some_and(|prior| id <= prior) {
                return Err(TargetDisclosureError::NonCanonicalFactOrder);
            }
            previous = Some(id);
        }
        Ok(())
    }

    /// Deterministic SHA-256 of this exact source/target disclosure manifest.
    pub fn manifest_sha256(&self) -> Result<String, TargetDisclosureError> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

fn validate_date(value: &str) -> Result<(), TargetDisclosureError> {
    let bytes = value.as_bytes();
    let digit_ranges = [&bytes.get(0..4), &bytes.get(5..7), &bytes.get(8..10)];
    if bytes.len() != 10
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || digit_ranges
            .iter()
            .any(|range| range.is_none_or(|digits| !digits.iter().all(u8::is_ascii_digit)))
    {
        return Err(TargetDisclosureError::InvalidDate(value.to_string()));
    }

    let parse_digits = |digits: &[u8]| -> u16 {
        digits
            .iter()
            .fold(0_u16, |acc, byte| acc * 10 + u16::from(byte - b'0'))
    };
    let year = parse_digits(&bytes[0..4]);
    let month = parse_digits(&bytes[5..7]) as u8;
    let day = parse_digits(&bytes[8..10]) as u8;

    if year < 1900 || !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
        return Err(TargetDisclosureError::InvalidDate(value.to_string()));
    }
    Ok(())
}

fn days_in_month(year: u16, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: u16) -> bool {
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
}

fn validate_decimal(value: &str) -> Result<(), TargetDisclosureError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| TargetDisclosureError::InvalidDecimal(value.to_string()))?;
    if !parsed.is_finite() || parsed.to_string() != value {
        return Err(TargetDisclosureError::InvalidDecimal(value.to_string()));
    }
    Ok(())
}

fn nonempty(name: &'static str, value: &str) -> Result<(), TargetDisclosureError> {
    if value.trim().is_empty() {
        return Err(TargetDisclosureError::EmptyField(name));
    }
    Ok(())
}

fn sha256(value: &str) -> Result<(), TargetDisclosureError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(TargetDisclosureError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Target-disclosure validation failures.
#[derive(Debug, Error)]
pub enum TargetDisclosureError {
    /// Manifest schema unsupported.
    #[error("unsupported target disclosure schema {0}")]
    UnsupportedSchema(u32),
    /// Required text field empty.
    #[error("required field {0} is empty")]
    EmptyField(&'static str),
    /// SHA-256 malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Date not canonical/valid Gregorian date.
    #[error("invalid disclosure date: {0}")]
    InvalidDate(String),
    /// Canonical numeric text malformed/nonfinite/noncanonical.
    #[error("invalid canonical decimal: {0}")]
    InvalidDecimal(String),
    /// Aggregate count must be positive.
    #[error("aggregate discovery count must be positive")]
    ZeroAggregateCount,
    /// Threshold value/unit are only valid as a complete pair.
    #[error("threshold value and threshold unit must either both be present or both absent")]
    IncompleteThreshold,
    /// No source facts supplied.
    #[error("target disclosure manifest contains no facts")]
    EmptyFactSet,
    /// Repeated fact ID.
    #[error("duplicate target fact id {0}")]
    DuplicateFactId(String),
    /// Facts not strictly ordered by fact ID.
    #[error("target facts must be strictly ordered by fact id")]
    NonCanonicalFactOrder,
    /// JSON serialization failure.
    #[error("serialization failure: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn source(access: DisclosureAccess) -> DisclosureSource {
        DisclosureSource {
            source_id: "fe-co-zr-2026-abstract".to_string(),
            publication_id: "10.1103/jx5f-kzl8".to_string(),
            locator: "aps-abstract".to_string(),
            source_artifact_sha256: sha('a'),
            access,
            disclosure_date: "2026-02-09".to_string(),
        }
    }

    #[test]
    fn abstract_composition_can_be_composition_target_only() {
        let fact = DisclosureFact::CompositionMention {
            fact_id: "composition-fe5co18zr6".to_string(),
            composition_sha256: sha('b'),
            formula: "Fe5Co18Zr6".to_string(),
            role: CompositionRole::SubstitutedCandidate,
        };
        assert_eq!(
            fact.authority(&source(DisclosureAccess::Abstract)).unwrap(),
            BenchmarkTargetAuthority::CompositionIdentity
        );
    }

    #[test]
    fn abstract_numeric_property_is_descriptive_without_exact_method_conditions() {
        let fact = DisclosureFact::QuantitativePropertyMention {
            fact_id: "k1-fe5co18zr6".to_string(),
            composition_sha256: sha('b'),
            structure_sha256: None,
            property_id: "magnetocrystalline_anisotropy_k1".to_string(),
            value: "1.1".to_string(),
            unit: "MJ/m^3".to_string(),
            condition_signature: None,
            method_artifact_sha256: None,
        };
        assert_eq!(
            fact.authority(&source(DisclosureAccess::Abstract)).unwrap(),
            BenchmarkTargetAuthority::DescriptiveOnly
        );
    }

    #[test]
    fn guessed_structure_cannot_gain_structure_authority_from_abstract() {
        let fact = DisclosureFact::StructureTarget {
            fact_id: "structure-guessed".to_string(),
            composition_sha256: sha('b'),
            structure_sha256: sha('c'),
            structure_artifact_sha256: sha('d'),
        };
        assert_eq!(
            fact.authority(&source(DisclosureAccess::Abstract)).unwrap(),
            BenchmarkTargetAuthority::DescriptiveOnly
        );
    }

    #[test]
    fn supplement_bound_property_can_be_quantitative_target() {
        let fact = DisclosureFact::QuantitativePropertyMention {
            fact_id: "k1-exact".to_string(),
            composition_sha256: sha('b'),
            structure_sha256: Some(sha('c')),
            property_id: "magnetocrystalline_anisotropy_k1".to_string(),
            value: "1.1".to_string(),
            unit: "MJ/m^3".to_string(),
            condition_signature: Some("soc-dft|0K|easy-axis-bound".to_string()),
            method_artifact_sha256: Some(sha('d')),
        };
        assert_eq!(
            fact.authority(&source(DisclosureAccess::Supplement)).unwrap(),
            BenchmarkTargetAuthority::QuantitativeProperty
        );
    }

    #[test]
    fn aggregate_counts_never_become_scored_material_targets() {
        let fact = DisclosureFact::AggregateCount {
            fact_id: "count-stable".to_string(),
            category: "thermodynamically_stable_ternaries".to_string(),
            count: 9,
            threshold_value: None,
            threshold_unit: None,
        };
        assert_eq!(
            fact.authority(&source(DisclosureAccess::Dataset)).unwrap(),
            BenchmarkTargetAuthority::DescriptiveOnly
        );
    }

    #[test]
    fn date_validation_rejects_non_ascii_and_impossible_dates() {
        assert!(validate_date("2026-02-09").is_ok());
        assert!(validate_date("2026-02-30").is_err());
        assert!(validate_date("2025-02-29").is_err());
        assert!(validate_date("2024-02-29").is_ok());
        assert!(validate_date("２０２６-02-09").is_err());
    }

    #[test]
    fn manifest_order_is_canonical_and_content_addressed() {
        let manifest = TargetDisclosureManifest {
            schema_version: 1,
            source: source(DisclosureAccess::Abstract),
            facts: vec![
                DisclosureFact::AggregateCount {
                    fact_id: "a-count".to_string(),
                    category: "stable".to_string(),
                    count: 9,
                    threshold_value: None,
                    threshold_unit: None,
                },
                DisclosureFact::CompositionMention {
                    fact_id: "b-composition".to_string(),
                    composition_sha256: sha('b'),
                    formula: "Fe5Co18Zr6".to_string(),
                    role: CompositionRole::SubstitutedCandidate,
                },
            ],
        };
        manifest.validate().unwrap();
        assert_eq!(manifest.manifest_sha256().unwrap().len(), 64);
    }
}
