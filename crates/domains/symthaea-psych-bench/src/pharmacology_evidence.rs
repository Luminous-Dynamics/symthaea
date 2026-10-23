// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Provenance-first evidence records for safety-oriented pharmacology research.
//!
//! This module models evidence about already-characterized compounds. It is
//! intentionally not a molecule generator, potency ranker, dosing tool, or
//! clinical decision system. Raw source values are preserved alongside any
//! normalized measurement so that normalization never erases the measurement
//! that a paper or assay database actually reported.

use serde::{Deserialize, Serialize};

/// Current serialized record schema version.
pub const EVIDENCE_SCHEMA_VERSION: &str = "1.0";

/// Kind of source evidence. The category describes the source, not its truth
/// or causal strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    ExperimentalAssay,
    PreclinicalStudy,
    ClinicalStudy,
    RegulatoryDocument,
    TrialRegistry,
    PharmacovigilanceSignal,
    LiteratureReview,
    ParticipantReport,
    EthnobotanicalAccount,
    Other,
}

/// Reuse status known to the collector. `Unknown` is explicit and must not be
/// silently interpreted as permission to redistribute the source content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReuseStatus {
    Unknown,
    ReferenceOnly,
    RedistributionPermitted,
    PublicDomain,
    Restricted,
}

/// Relational qualifier for a reported measurement. Censored measurements
/// such as `< 10` must not be converted to exact values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementRelation {
    Exact,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Interval,
    Qualitative,
}

/// Review state for the extraction and interpretation process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdjudicationStatus {
    Unreviewed,
    SingleReviewer,
    IndependentlyReviewed,
    Disputed,
    Rejected,
}

/// Citation and acquisition provenance for one source record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceProvenance {
    /// Stable source URI, DOI URL, registry URL, or documented local reference.
    pub uri: String,
    /// Identifier assigned by the source, when available.
    pub source_record_id: Option<String>,
    pub title: Option<String>,
    /// ISO-8601 calendar date (YYYY-MM-DD), if known.
    pub published_on: Option<String>,
    /// ISO-8601 calendar date (YYYY-MM-DD) when this source was retrieved.
    pub retrieved_on: String,
    /// SHA-256 of the acquired source artifact, when the artifact can be kept.
    pub content_sha256: Option<String>,
    pub reuse_status: ReuseStatus,
}

/// Experimental context needed to interpret an assay or study measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssayContext {
    /// Measured endpoint, written as observed by the source before any mapping.
    pub endpoint: String,
    pub assay_id: Option<String>,
    pub assay_type: Option<String>,
    pub organism: Option<String>,
    pub biological_system: Option<String>,
    pub protocol_reference: Option<String>,
}

/// Raw and optional normalized representation of one measured quantity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    /// Exact source representation, including inequality symbols when present.
    pub raw_value: String,
    pub raw_unit: Option<String>,
    pub relation: MeasurementRelation,
    /// Normalized point value, only when the transformation is justified.
    pub normalized_value: Option<f64>,
    pub normalized_unit: Option<String>,
    /// Optional bounds in the normalized unit.
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
}

/// How a record was created and reviewed. Keep transformation descriptions
/// stable and explicit rather than storing only a final number.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessingTrace {
    pub extractor: String,
    pub extractor_version: String,
    pub transformations: Vec<String>,
    pub adjudication: AdjudicationStatus,
}

/// One source-linked item of evidence about a known, identified compound.
///
/// This record deliberately does not store a universal "compound quality"
/// score. Conflicting observations and evidence types remain distinct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub schema_version: String,
    pub record_id: String,
    /// Stable source-of-truth identifier (for example, a public database ID).
    /// This is not a user-provided compound name.
    pub compound_id: String,
    pub evidence_kind: EvidenceKind,
    pub source: SourceProvenance,
    pub assay_context: Option<AssayContext>,
    pub measurement: Option<Measurement>,
    pub study_design: Option<String>,
    pub limitations: Vec<String>,
    /// Record IDs for evidence that conflicts with this record. Conflict links
    /// are pointers only; they do not establish which record is correct.
    pub conflicts_with: Vec<String>,
    pub processing: ProcessingTrace,
}

impl EvidenceRecord {
    /// Validate invariants required before a record enters a frozen benchmark.
    ///
    /// This is a structural/provenance check, not a pharmacological safety
    /// judgment. Unknown reuse rights, disputed evidence, or weak study design
    /// remain visible and require separate policy/review decisions.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.schema_version != EVIDENCE_SCHEMA_VERSION {
            errors.push(format!(
                "unsupported schema_version {:?}; expected {:?}",
                self.schema_version, EVIDENCE_SCHEMA_VERSION
            ));
        }
        require_nonempty(&self.record_id, "record_id", &mut errors);
        require_nonempty(&self.compound_id, "compound_id", &mut errors);
        require_nonempty(&self.source.uri, "source.uri", &mut errors);
        require_date(&self.source.retrieved_on, "source.retrieved_on", &mut errors);

        if let Some(date) = self.source.published_on.as_deref() {
            require_date(date, "source.published_on", &mut errors);
        }
        if let Some(digest) = self.source.content_sha256.as_deref() {
            if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                errors.push("source.content_sha256 must be 64 hexadecimal characters".into());
            }
        }
        if let Some(context) = &self.assay_context {
            require_nonempty(&context.endpoint, "assay_context.endpoint", &mut errors);
        }
        if let Some(measurement) = &self.measurement {
            if measurement.raw_value.trim().is_empty() {
                errors.push("measurement.raw_value must not be empty".into());
            }
            if measurement.normalized_value.is_some() != measurement.normalized_unit.is_some() {
                errors.push(
                    "measurement.normalized_value and normalized_unit must be present together"
                        .into(),
                );
            }
            for (label, value) in [
                ("normalized_value", measurement.normalized_value),
                ("lower_bound", measurement.lower_bound),
                ("upper_bound", measurement.upper_bound),
            ] {
                if value.is_some_and(|v| !v.is_finite()) {
                    errors.push(format!("measurement.{label} must be finite when present"));
                }
            }
            if (measurement.lower_bound.is_some() || measurement.upper_bound.is_some())
                && measurement.normalized_unit.is_none()
            {
                errors.push("measurement bounds require normalized_unit".into());
            }
            if let (Some(lower), Some(upper)) =
                (measurement.lower_bound, measurement.upper_bound)
            {
                if lower > upper {
                    errors.push("measurement.lower_bound must be <= upper_bound".into());
                }
            }
            if measurement.relation == MeasurementRelation::Interval
                && (measurement.lower_bound.is_none() || measurement.upper_bound.is_none())
            {
                errors.push(
                    "interval measurements require both lower_bound and upper_bound".into(),
                );
            }
        }

        if self.conflicts_with.iter().any(|id| id == &self.record_id) {
            errors.push("conflicts_with must not point to the record itself".into());
        }
        let mut seen_conflicts = std::collections::BTreeSet::new();
        if self.conflicts_with.iter().any(|id| !seen_conflicts.insert(id)) {
            errors.push("conflicts_with contains duplicate record IDs".into());
        }

        require_nonempty(&self.processing.extractor, "processing.extractor", &mut errors);
        require_nonempty(
            &self.processing.extractor_version,
            "processing.extractor_version",
            &mut errors,
        );

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Deterministic BLAKE3 digest of this record's serialized representation.
    ///
    /// The digest binds the exact record content; it does not prove the source
    /// is truthful, that the extraction is correct, or that the compound is
    /// safe. Struct fields and ordered vectors are serialized deterministically;
    /// callers should not add unordered maps to this wire type without defining
    /// canonicalization first.
    pub fn content_digest_blake3(&self) -> Result<String, String> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("serialize evidence record for digest: {error}"))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

fn require_nonempty(value: &str, field: &str, errors: &mut Vec<String>) {
    if value.trim().is_empty() {
        errors.push(format!("{field} must not be empty"));
    }
}

fn require_date(value: &str, field: &str, errors: &mut Vec<String>) {
    if chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d").is_err() {
        errors.push(format!("{field} must be a valid YYYY-MM-DD date"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_record() -> EvidenceRecord {
        EvidenceRecord {
            schema_version: EVIDENCE_SCHEMA_VERSION.to_string(),
            record_id: "evidence-001".to_string(),
            compound_id: "public-compound-id-001".to_string(),
            evidence_kind: EvidenceKind::ExperimentalAssay,
            source: SourceProvenance {
                uri: "https://example.org/source/record-1".to_string(),
                source_record_id: Some("record-1".to_string()),
                title: Some("Illustrative assay record".to_string()),
                published_on: Some("2024-01-15".to_string()),
                retrieved_on: "2026-10-20".to_string(),
                content_sha256: Some("a".repeat(64)),
                reuse_status: ReuseStatus::ReferenceOnly,
            },
            assay_context: Some(AssayContext {
                endpoint: "example_endpoint".to_string(),
                assay_id: Some("assay-1".to_string()),
                assay_type: Some("functional".to_string()),
                organism: Some("example organism".to_string()),
                biological_system: Some("example system".to_string()),
                protocol_reference: None,
            }),
            measurement: Some(Measurement {
                raw_value: "< 10".to_string(),
                raw_unit: Some("uM".to_string()),
                relation: MeasurementRelation::LessThan,
                normalized_value: None,
                normalized_unit: None,
                lower_bound: None,
                upper_bound: None,
            }),
            study_design: Some("in vitro assay".to_string()),
            limitations: vec!["Illustrative test fixture only".to_string()],
            conflicts_with: vec![],
            processing: ProcessingTrace {
                extractor: "manual".to_string(),
                extractor_version: "1".to_string(),
                transformations: vec!["source metadata captured".to_string()],
                adjudication: AdjudicationStatus::Unreviewed,
            },
        }
    }

    #[test]
    fn valid_raw_record_passes_and_keeps_censored_measurement() {
        let record = valid_record();
        assert!(record.validate().is_ok());
        let measurement = record.measurement.expect("measurement present");
        assert_eq!(measurement.raw_value, "< 10");
        assert_eq!(measurement.relation, MeasurementRelation::LessThan);
        assert!(measurement.normalized_value.is_none());
    }

    #[test]
    fn normalized_value_requires_a_unit() {
        let mut record = valid_record();
        let measurement = record.measurement.as_mut().expect("measurement present");
        measurement.normalized_value = Some(9.0);
        assert!(
            record
                .validate()
                .expect_err("missing normalized unit must fail")
                .iter()
                .any(|error| error.contains("must be present together"))
        );
    }

    #[test]
    fn rejects_non_finite_measurement_values() {
        let mut record = valid_record();
        let measurement = record.measurement.as_mut().expect("measurement present");
        measurement.normalized_value = Some(f64::NAN);
        measurement.normalized_unit = Some("uM".to_string());
        assert!(
            record
                .validate()
                .expect_err("NaN must fail validation")
                .iter()
                .any(|error| error.contains("must be finite"))
        );
    }

    #[test]
    fn interval_requires_ordered_bounds() {
        let mut record = valid_record();
        let measurement = record.measurement.as_mut().expect("measurement present");
        measurement.relation = MeasurementRelation::Interval;
        measurement.lower_bound = Some(2.0);
        measurement.upper_bound = Some(1.0);
        assert!(
            record
                .validate()
                .expect_err("reversed interval must fail validation")
                .iter()
                .any(|error| error.contains("lower_bound must be <="))
        );
    }

    #[test]
    fn rejects_invalid_dates_and_checksums() {
        let mut record = valid_record();
        record.source.retrieved_on = "2026-99-55".to_string();
        record.source.content_sha256 = Some("not-a-digest".to_string());
        let errors = record.validate().expect_err("invalid provenance must fail");
        assert!(errors.iter().any(|error| error.contains("retrieved_on")));
        assert!(errors.iter().any(|error| error.contains("content_sha256")));
    }

    #[test]
    fn rejects_self_conflict_and_duplicate_conflict_links() {
        let mut record = valid_record();
        record.conflicts_with = vec!["evidence-001".into(), "other".into(), "other".into()];
        let errors = record.validate().expect_err("invalid conflict links must fail");
        assert!(errors.iter().any(|error| error.contains("must not point to the record itself")));
        assert!(errors.iter().any(|error| error.contains("duplicate record IDs")));
    }

    #[test]
    fn digest_is_deterministic_and_content_sensitive() {
        let first = valid_record();
        let same = valid_record();
        assert_eq!(
            first.content_digest_blake3().expect("digest"),
            same.content_digest_blake3().expect("digest")
        );

        let mut changed = valid_record();
        changed.source.reuse_status = ReuseStatus::Unknown;
        assert_ne!(
            first.content_digest_blake3().expect("digest"),
            changed.content_digest_blake3().expect("digest")
        );
    }

    #[test]
    fn json_round_trip_preserves_the_record() {
        let original = valid_record();
        let json = serde_json::to_string(&original).expect("serialize record");
        let restored: EvidenceRecord = serde_json::from_str(&json).expect("deserialize record");
        assert_eq!(restored, original);
        assert!(restored.validate().is_ok());
    }
}
