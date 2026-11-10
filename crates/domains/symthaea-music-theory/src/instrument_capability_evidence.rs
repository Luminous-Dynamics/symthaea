// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound qualification for concrete instrument capability claims.
//!
//! This first slice qualifies sounding range and polyphony for one exact
//! performer/model + technique + optional renderer/patch/asset scope. It
//! deliberately does not infer capability from an instrument family or name.
//! Measured claims can pass the range/polyphony hard gate; source-cited models
//! are conditional discovery evidence; derived, estimated and unverified
//! claims remain research-only. Tessitura is recorded separately for later
//! demand-specific checks. This module does not certify articulation,
//! ergonomics or actual renderer fidelity.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::orchestration::{
    InstrumentCapabilityV1, InstrumentKindV1, SoundingRangeV1,
    MAX_INSTRUMENT_POLYPHONY,
};

pub const INSTRUMENT_CAPABILITY_EVIDENCE_SCHEMA_V1: &str =
    "instrument-capability-evidence-v1";
pub const MAX_CAPABILITY_EVIDENCE_RECORDS_V1: usize = 8_192;
pub const MAX_EVIDENCE_PARENTS_V1: usize = 64;

/// Scope is exact. A claim for one technique, patch or asset cannot silently
/// qualify a different rendering/performance context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEvidenceScopeV1 {
    pub performer_or_model_id: String,
    pub technique_id: String,
    pub renderer_id: Option<String>,
    pub patch_id: Option<String>,
    /// Exact sample/asset content digest when a sample-backed claim is scoped.
    pub asset_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEvidenceBasisV1 {
    Measured,
    SourceCitedModelled,
    Derived,
    Estimated,
    Unverified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityLicenseStatusV1 {
    Permitted,
    Restricted,
    Unknown,
}

/// A versioned provenance record for one or more claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEvidenceRecordV1 {
    pub evidence_id: String,
    pub basis: CapabilityEvidenceBasisV1,
    pub scope: CapabilityEvidenceScopeV1,
    pub source_ref: Option<String>,
    pub source_revision: Option<String>,
    pub content_sha256: Option<String>,
    pub method_id: Option<String>,
    pub parent_evidence_ids: Vec<String>,
    pub transform_version: Option<String>,
    /// Uncertainty in parts per million; used for estimated claims only.
    pub uncertainty_ppm: Option<u32>,
    pub license_status: CapabilityLicenseStatusV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityClaimV1<T> {
    pub value: T,
    pub evidence_id: String,
}

/// Evidence profile for the hard range/polyphony capability fields plus an
/// optional practical tessitura claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentCapabilityEvidenceProfileV1 {
    pub schema: String,
    pub instrument_id: String,
    pub instrument_kind: InstrumentKindV1,
    pub scope: CapabilityEvidenceScopeV1,
    pub sounding_range: Option<CapabilityClaimV1<SoundingRangeV1>>,
    pub tessitura: Option<CapabilityClaimV1<SoundingRangeV1>>,
    pub max_polyphony: Option<CapabilityClaimV1<u8>>,
    pub evidence: Vec<CapabilityEvidenceRecordV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityQualificationDispositionV1 {
    MeasuredHardGate,
    ConditionalModelled,
    ResearchOnly,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEvidenceIssueCodeV1 {
    InvalidSchema,
    InvalidInstrumentId,
    InstrumentIdMismatch,
    InstrumentKindMismatch,
    InvalidScope,
    ScopeMismatch,
    MissingRangeClaim,
    MissingPolyphonyClaim,
    MissingEvidenceRecord,
    InvalidRange,
    DeclaredRangeMismatch,
    InvalidPolyphony,
    DeclaredPolyphonyMismatch,
    InvalidTessitura,
    TessituraOutsideSoundingRange,
    TooManyEvidenceRecords,
    InvalidEvidenceId,
    DuplicateEvidenceId,
    InvalidSourceReference,
    InvalidSourceRevision,
    InvalidContentDigest,
    InvalidMethodId,
    InvalidParentReferences,
    UnknownParentEvidence,
    EvidenceCycle,
    InvalidUncertainty,
    LicenseNotPermitted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEvidenceIssueV1 {
    pub code: CapabilityEvidenceIssueCodeV1,
    pub evidence_id: Option<String>,
    pub claim: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualifiedInstrumentCapabilityV1 {
    pub capability: InstrumentCapabilityV1,
    pub scope: CapabilityEvidenceScopeV1,
    pub disposition: CapabilityQualificationDispositionV1,
    pub range_evidence_basis: Option<CapabilityEvidenceBasisV1>,
    pub polyphony_evidence_basis: Option<CapabilityEvidenceBasisV1>,
    pub tessitura_evidence_basis: Option<CapabilityEvidenceBasisV1>,
    /// True only when both hard claims are measured at the exact declared scope.
    pub may_pass_hard_range_polyphony_gate: bool,
    /// Measured or source-cited/modelled evidence may be used for conditional
    /// candidate discovery, but only measured evidence passes the hard gate.
    pub may_enter_conditional_discovery: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEvidenceReportV1 {
    pub valid: bool,
    pub disposition: CapabilityQualificationDispositionV1,
    pub evidence_record_count: usize,
    pub issues: Vec<CapabilityEvidenceIssueV1>,
    pub qualified: Option<QualifiedInstrumentCapabilityV1>,
}

impl CapabilityEvidenceReportV1 {
    fn new(count: usize) -> Self {
        Self {
            valid: true,
            disposition: CapabilityQualificationDispositionV1::Abstain,
            evidence_record_count: count,
            issues: Vec::new(),
            qualified: None,
        }
    }

    fn push(
        &mut self,
        code: CapabilityEvidenceIssueCodeV1,
        evidence_id: Option<&str>,
        claim: Option<&str>,
        message: impl Into<String>,
    ) {
        self.valid = false;
        self.issues.push(CapabilityEvidenceIssueV1 {
            code,
            evidence_id: evidence_id.map(str::to_owned),
            claim: claim.map(str::to_owned),
            message: message.into(),
        });
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn non_empty(value: &Option<String>) -> bool {
    value.as_ref().is_some_and(|text| !text.trim().is_empty())
}

fn valid_scope(scope: &CapabilityEvidenceScopeV1) -> bool {
    !scope.performer_or_model_id.trim().is_empty()
        && !scope.technique_id.trim().is_empty()
        && scope.renderer_id.as_ref().map_or(true, |value| !value.trim().is_empty())
        && scope.patch_id.as_ref().map_or(true, |value| !value.trim().is_empty())
        && scope.asset_sha256.as_ref().map_or(true, |value| valid_sha256(value))
}

fn source_fields_valid(record: &CapabilityEvidenceRecordV1) -> bool {
    non_empty(&record.source_ref)
        && non_empty(&record.source_revision)
        && record.content_sha256.as_ref().map_or(false, |value| valid_sha256(value))
}

fn evidence_graph_is_acyclic(
    records: &BTreeMap<String, &CapabilityEvidenceRecordV1>,
    report: &mut CapabilityEvidenceReportV1,
) {
    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    let mut indegree = BTreeMap::<String, usize>::new();
    for id in records.keys() {
        adjacency.insert(id.clone(), BTreeSet::new());
        indegree.insert(id.clone(), 0);
    }

    for record in records.values() {
        if record.basis != CapabilityEvidenceBasisV1::Derived {
            continue;
        }
        for parent in &record.parent_evidence_ids {
            if !records.contains_key(parent) {
                report.push(
                    CapabilityEvidenceIssueCodeV1::UnknownParentEvidence,
                    Some(&record.evidence_id),
                    None,
                    format!("derived evidence references unknown parent {parent}"),
                );
                continue;
            }
            let inserted = adjacency
                .entry(parent.clone())
                .or_default()
                .insert(record.evidence_id.clone());
            if inserted {
                if let Some(degree) = indegree.get_mut(&record.evidence_id) {
                    *degree = degree.saturating_add(1);
                }
            }
        }
    }

    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut visited = 0usize;
    while let Some(id) = ready.pop_first() {
        visited += 1;
        if let Some(children) = adjacency.get(&id) {
            for child in children {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        ready.insert(child.clone());
                    }
                }
            }
        }
    }
    if visited != records.len() {
        report.push(
            CapabilityEvidenceIssueCodeV1::EvidenceCycle,
            None,
            None,
            "derived evidence parent graph must be acyclic",
        );
    }
}

fn index_evidence(
    profile: &InstrumentCapabilityEvidenceProfileV1,
    report: &mut CapabilityEvidenceReportV1,
) -> BTreeMap<String, CapabilityEvidenceBasisV1> {
    let mut by_id = BTreeMap::<String, &CapabilityEvidenceRecordV1>::new();
    if profile.evidence.len() > MAX_CAPABILITY_EVIDENCE_RECORDS_V1 {
        report.push(
            CapabilityEvidenceIssueCodeV1::TooManyEvidenceRecords,
            None,
            None,
            "evidence record count exceeds the v1 resource limit",
        );
        return BTreeMap::new();
    }

    for record in &profile.evidence {
        if record.evidence_id.trim().is_empty() {
            report.push(
                CapabilityEvidenceIssueCodeV1::InvalidEvidenceId,
                Some(&record.evidence_id),
                None,
                "evidence_id must be non-empty",
            );
            continue;
        }
        if by_id.insert(record.evidence_id.clone(), record).is_some() {
            report.push(
                CapabilityEvidenceIssueCodeV1::DuplicateEvidenceId,
                Some(&record.evidence_id),
                None,
                "evidence_id must be unique within the profile",
            );
        }
        if !valid_scope(&record.scope) {
            report.push(
                CapabilityEvidenceIssueCodeV1::InvalidScope,
                Some(&record.evidence_id),
                None,
                "evidence scope must have a performer/model, technique and valid optional renderer/patch/asset fields",
            );
        }
        if record.scope != profile.scope {
            report.push(
                CapabilityEvidenceIssueCodeV1::ScopeMismatch,
                Some(&record.evidence_id),
                None,
                "evidence scope does not exactly match the profile scope",
            );
        }

        match record.basis {
            CapabilityEvidenceBasisV1::Measured => {
                if !source_fields_valid(record) {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::InvalidContentDigest,
                        Some(&record.evidence_id),
                        None,
                        "measured evidence requires source reference, revision and artifact SHA-256",
                    );
                }
                if !non_empty(&record.method_id) {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::InvalidMethodId,
                        Some(&record.evidence_id),
                        None,
                        "measured evidence requires a measurement method identifier",
                    );
                }
                if record.license_status != CapabilityLicenseStatusV1::Permitted {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::LicenseNotPermitted,
                        Some(&record.evidence_id),
                        None,
                        "measured evidence requires permitted-use status",
                    );
                }
            }
            CapabilityEvidenceBasisV1::SourceCitedModelled => {
                if !source_fields_valid(record) {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::InvalidContentDigest,
                        Some(&record.evidence_id),
                        None,
                        "source-cited/modelled evidence requires source reference, revision and content SHA-256",
                    );
                }
                if !non_empty(&record.method_id) {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::InvalidMethodId,
                        Some(&record.evidence_id),
                        None,
                        "source-cited/modelled evidence requires declared assumptions or method",
                    );
                }
                if record.license_status != CapabilityLicenseStatusV1::Permitted {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::LicenseNotPermitted,
                        Some(&record.evidence_id),
                        None,
                        "source-cited/modelled evidence requires permitted-use status",
                    );
                }
            }
            CapabilityEvidenceBasisV1::Derived => {
                if record.parent_evidence_ids.is_empty()
                    || record.parent_evidence_ids.len() > MAX_EVIDENCE_PARENTS_V1
                    || !non_empty(&record.transform_version)
                    || !non_empty(&record.method_id)
                {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::InvalidParentReferences,
                        Some(&record.evidence_id),
                        None,
                        "derived evidence requires bounded parents, a method and transform version",
                    );
                }
            }
            CapabilityEvidenceBasisV1::Estimated => {
                if !non_empty(&record.method_id)
                    || record.uncertainty_ppm.map_or(true, |value| value == 0)
                {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::InvalidUncertainty,
                        Some(&record.evidence_id),
                        None,
                        "estimated evidence requires an estimator identifier and positive uncertainty_ppm",
                    );
                }
            }
            CapabilityEvidenceBasisV1::Unverified => {}
        }
    }

    evidence_graph_is_acyclic(&by_id, report);
    by_id
        .into_iter()
        .map(|(id, record)| (id, record.basis))
        .collect()
}

/// Qualify the hard range/polyphony claims for one concrete instance.
///
/// A measured range and measured polyphony claim at the exact same scope can
/// pass the current hard gate. Source-cited/modelled claims remain conditional.
/// Estimated, derived and unverified claims are research-only, not passes.
pub fn qualify_instrument_capability(
    capability: &InstrumentCapabilityV1,
    profile: &InstrumentCapabilityEvidenceProfileV1,
) -> CapabilityEvidenceReportV1 {
    let mut report = CapabilityEvidenceReportV1::new(profile.evidence.len());

    if profile.schema != INSTRUMENT_CAPABILITY_EVIDENCE_SCHEMA_V1 {
        report.push(
            CapabilityEvidenceIssueCodeV1::InvalidSchema,
            None,
            None,
            "unsupported instrument capability evidence schema",
        );
    }
    if profile.instrument_id.trim().is_empty() {
        report.push(
            CapabilityEvidenceIssueCodeV1::InvalidInstrumentId,
            None,
            None,
            "instrument_id must be non-empty",
        );
    }
    if profile.instrument_id != capability.id {
        report.push(
            CapabilityEvidenceIssueCodeV1::InstrumentIdMismatch,
            None,
            None,
            "evidence profile ID does not match the concrete orchestration instance",
        );
    }
    if profile.instrument_kind != capability.kind {
        report.push(
            CapabilityEvidenceIssueCodeV1::InstrumentKindMismatch,
            None,
            None,
            "evidence instrument kind does not match the concrete capability",
        );
    }
    if !valid_scope(&profile.scope) {
        report.push(
            CapabilityEvidenceIssueCodeV1::InvalidScope,
            None,
            None,
            "profile scope must name performer/model and technique and have valid optional scope fields",
        );
    }

    let evidence = index_evidence(profile, &mut report);

    let range_basis = match &profile.sounding_range {
        Some(claim) => {
            if claim.value.lowest_midi > claim.value.highest_midi
                || claim.value.highest_midi > 127
            {
                report.push(
                    CapabilityEvidenceIssueCodeV1::InvalidRange,
                    Some(&claim.evidence_id),
                    Some("sounding_range"),
                    "sounding range must be an ordered inclusive MIDI interval within 0..=127",
                );
            }
            if claim.value != capability.sounding_range {
                report.push(
                    CapabilityEvidenceIssueCodeV1::DeclaredRangeMismatch,
                    Some(&claim.evidence_id),
                    Some("sounding_range"),
                    "evidence-backed range must exactly match the capability sent to orchestration",
                );
            }
            if let Some(basis) = evidence.get(&claim.evidence_id) {
                Some(*basis)
            } else {
                report.push(
                    CapabilityEvidenceIssueCodeV1::MissingEvidenceRecord,
                    Some(&claim.evidence_id),
                    Some("sounding_range"),
                    "sounding range claim references no evidence record",
                );
                None
            }
        }
        None => {
            report.push(
                CapabilityEvidenceIssueCodeV1::MissingRangeClaim,
                None,
                Some("sounding_range"),
                "a hard sounding-range claim is required",
            );
            None
        }
    };

    let polyphony_basis = match &profile.max_polyphony {
        Some(claim) => {
            if claim.value == 0 || claim.value > MAX_INSTRUMENT_POLYPHONY {
                report.push(
                    CapabilityEvidenceIssueCodeV1::InvalidPolyphony,
                    Some(&claim.evidence_id),
                    Some("max_polyphony"),
                    "polyphony must be within the orchestration validator's supported bounds",
                );
            }
            if claim.value != capability.max_polyphony {
                report.push(
                    CapabilityEvidenceIssueCodeV1::DeclaredPolyphonyMismatch,
                    Some(&claim.evidence_id),
                    Some("max_polyphony"),
                    "evidence-backed polyphony must exactly match the capability sent to orchestration",
                );
            }
            if let Some(basis) = evidence.get(&claim.evidence_id) {
                Some(*basis)
            } else {
                report.push(
                    CapabilityEvidenceIssueCodeV1::MissingEvidenceRecord,
                    Some(&claim.evidence_id),
                    Some("max_polyphony"),
                    "polyphony claim references no evidence record",
                );
                None
            }
        }
        None => {
            report.push(
                CapabilityEvidenceIssueCodeV1::MissingPolyphonyClaim,
                None,
                Some("max_polyphony"),
                "a hard polyphony claim is required",
            );
            None
        }
    };

    let tessitura_basis = if let Some(claim) = &profile.tessitura {
        if claim.value.lowest_midi > claim.value.highest_midi
            || claim.value.highest_midi > 127
        {
            report.push(
                CapabilityEvidenceIssueCodeV1::InvalidTessitura,
                Some(&claim.evidence_id),
                Some("tessitura"),
                "tessitura must be an ordered inclusive MIDI interval within 0..=127",
            );
        }
        if let Some(range) = &profile.sounding_range {
            if claim.value.lowest_midi < range.value.lowest_midi
                || claim.value.highest_midi > range.value.highest_midi
            {
                report.push(
                    CapabilityEvidenceIssueCodeV1::TessituraOutsideSoundingRange,
                    Some(&claim.evidence_id),
                    Some("tessitura"),
                    "tessitura must be contained inside the claimed sounding range",
                );
            }
        }
        if let Some(basis) = evidence.get(&claim.evidence_id) {
            Some(*basis)
        } else {
            report.push(
                CapabilityEvidenceIssueCodeV1::MissingEvidenceRecord,
                Some(&claim.evidence_id),
                Some("tessitura"),
                "tessitura claim references no evidence record",
            );
            None
        }
    } else {
        None
    };

    if !report.valid {
        report.disposition = CapabilityQualificationDispositionV1::Abstain;
        return report;
    }

    let hard_measured = range_basis == Some(CapabilityEvidenceBasisV1::Measured)
        && polyphony_basis == Some(CapabilityEvidenceBasisV1::Measured);
    let hard_conditional = matches!(
        range_basis,
        Some(CapabilityEvidenceBasisV1::Measured | CapabilityEvidenceBasisV1::SourceCitedModelled)
    ) && matches!(
        polyphony_basis,
        Some(CapabilityEvidenceBasisV1::Measured | CapabilityEvidenceBasisV1::SourceCitedModelled)
    );
    let disposition = if hard_measured {
        CapabilityQualificationDispositionV1::MeasuredHardGate
    } else if hard_conditional {
        CapabilityQualificationDispositionV1::ConditionalModelled
    } else {
        CapabilityQualificationDispositionV1::ResearchOnly
    };

    report.disposition = disposition;
    report.qualified = Some(QualifiedInstrumentCapabilityV1 {
        capability: capability.clone(),
        scope: profile.scope.clone(),
        disposition,
        range_evidence_basis: range_basis,
        polyphony_evidence_basis: polyphony_basis,
        tessitura_evidence_basis: tessitura_basis,
        may_pass_hard_range_polyphony_gate: hard_measured,
        may_enter_conditional_discovery: hard_measured || hard_conditional,
    });
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> CapabilityEvidenceScopeV1 {
        CapabilityEvidenceScopeV1 {
            performer_or_model_id: "cello-model-a".into(),
            technique_id: "normal-bowed-playing".into(),
            renderer_id: Some("renderer-1".into()),
            patch_id: Some("cello-sustain".into()),
            asset_sha256: Some("d".repeat(64)),
        }
    }

    fn capability() -> InstrumentCapabilityV1 {
        InstrumentCapabilityV1 {
            id: "cello-solo-1".into(),
            kind: InstrumentKindV1::Cello,
            sounding_range: SoundingRangeV1 {
                lowest_midi: 36,
                highest_midi: 84,
            },
            max_polyphony: 2,
        }
    }

    fn record(id: &str, basis: CapabilityEvidenceBasisV1) -> CapabilityEvidenceRecordV1 {
        let measured_or_cited = matches!(
            basis,
            CapabilityEvidenceBasisV1::Measured | CapabilityEvidenceBasisV1::SourceCitedModelled
        );
        CapabilityEvidenceRecordV1 {
            evidence_id: id.into(),
            basis,
            scope: scope(),
            source_ref: measured_or_cited.then(|| "urn:evidence:cello".into()),
            source_revision: measured_or_cited.then(|| "revision-1".into()),
            content_sha256: measured_or_cited.then(|| "a".repeat(64)),
            method_id: match basis {
                CapabilityEvidenceBasisV1::Measured => Some("chromatic-playthrough-v1".into()),
                CapabilityEvidenceBasisV1::SourceCitedModelled => Some("published-model-v1".into()),
                CapabilityEvidenceBasisV1::Derived => Some("range-transform-v1".into()),
                CapabilityEvidenceBasisV1::Estimated => Some("estimator-v1".into()),
                CapabilityEvidenceBasisV1::Unverified => None,
            },
            parent_evidence_ids: if basis == CapabilityEvidenceBasisV1::Derived {
                vec!["parent".into()]
            } else {
                Vec::new()
            },
            transform_version: (basis == CapabilityEvidenceBasisV1::Derived).then(|| "v1".into()),
            uncertainty_ppm: (basis == CapabilityEvidenceBasisV1::Estimated).then_some(100_000),
            license_status: CapabilityLicenseStatusV1::Permitted,
        }
    }

    fn profile(
        range_basis: CapabilityEvidenceBasisV1,
        polyphony_basis: CapabilityEvidenceBasisV1,
    ) -> InstrumentCapabilityEvidenceProfileV1 {
        InstrumentCapabilityEvidenceProfileV1 {
            schema: INSTRUMENT_CAPABILITY_EVIDENCE_SCHEMA_V1.into(),
            instrument_id: "cello-solo-1".into(),
            instrument_kind: InstrumentKindV1::Cello,
            scope: scope(),
            sounding_range: Some(CapabilityClaimV1 {
                value: SoundingRangeV1 {
                    lowest_midi: 36,
                    highest_midi: 84,
                },
                evidence_id: "range".into(),
            }),
            tessitura: Some(CapabilityClaimV1 {
                value: SoundingRangeV1 {
                    lowest_midi: 48,
                    highest_midi: 72,
                },
                evidence_id: "tessitura".into(),
            }),
            max_polyphony: Some(CapabilityClaimV1 {
                value: 2,
                evidence_id: "polyphony".into(),
            }),
            evidence: vec![
                record("range", range_basis),
                record("tessitura", range_basis),
                record("polyphony", polyphony_basis),
            ],
        }
    }

    #[test]
    fn measured_range_and_polyphony_qualify_the_hard_gate() {
        let report = qualify_instrument_capability(
            &capability(),
            &profile(
                CapabilityEvidenceBasisV1::Measured,
                CapabilityEvidenceBasisV1::Measured,
            ),
        );
        assert!(report.valid, "{:?}", report.issues);
        let qualified = report.qualified.unwrap();
        assert_eq!(
            qualified.disposition,
            CapabilityQualificationDispositionV1::MeasuredHardGate
        );
        assert!(qualified.may_pass_hard_range_polyphony_gate);
    }

    #[test]
    fn source_cited_models_are_conditional_not_measured() {
        let report = qualify_instrument_capability(
            &capability(),
            &profile(
                CapabilityEvidenceBasisV1::SourceCitedModelled,
                CapabilityEvidenceBasisV1::SourceCitedModelled,
            ),
        );
        assert!(report.valid, "{:?}", report.issues);
        let qualified = report.qualified.unwrap();
        assert_eq!(
            qualified.disposition,
            CapabilityQualificationDispositionV1::ConditionalModelled
        );
        assert!(!qualified.may_pass_hard_range_polyphony_gate);
        assert!(qualified.may_enter_conditional_discovery);
    }

    #[test]
    fn estimated_and_unverified_claims_never_pass_the_hard_gate() {
        for basis in [
            CapabilityEvidenceBasisV1::Estimated,
            CapabilityEvidenceBasisV1::Unverified,
            CapabilityEvidenceBasisV1::Derived,
        ] {
            let report = qualify_instrument_capability(
                &capability(),
                &profile(basis, CapabilityEvidenceBasisV1::Measured),
            );
            assert!(report.valid, "{:?}", report.issues);
            let qualified = report.qualified.unwrap();
            assert_eq!(qualified.disposition, CapabilityQualificationDispositionV1::ResearchOnly);
            assert!(!qualified.may_pass_hard_range_polyphony_gate);
        }
    }

    #[test]
    fn rejects_scope_and_declared_capability_mismatches() {
        let mut profile = profile(
            CapabilityEvidenceBasisV1::Measured,
            CapabilityEvidenceBasisV1::Measured,
        );
        profile.evidence[0].scope.performer_or_model_id = "different-cello".into();
        profile.sounding_range.as_mut().unwrap().value.highest_midi = 90;
        let report = qualify_instrument_capability(&capability(), &profile);
        assert!(report.issues.iter().any(|issue| issue.code == CapabilityEvidenceIssueCodeV1::ScopeMismatch));
        assert!(report.issues.iter().any(|issue| issue.code == CapabilityEvidenceIssueCodeV1::DeclaredRangeMismatch));
        assert_eq!(report.disposition, CapabilityQualificationDispositionV1::Abstain);
        assert!(report.qualified.is_none());
    }

    #[test]
    fn rejects_missing_or_restricted_source_provenance() {
        let mut missing = profile(
            CapabilityEvidenceBasisV1::Measured,
            CapabilityEvidenceBasisV1::Measured,
        );
        missing.evidence[0].content_sha256 = None;
        let report = qualify_instrument_capability(&capability(), &missing);
        assert!(report.issues.iter().any(|issue| issue.code == CapabilityEvidenceIssueCodeV1::InvalidContentDigest));

        let mut restricted = profile(
            CapabilityEvidenceBasisV1::Measured,
            CapabilityEvidenceBasisV1::Measured,
        );
        restricted.evidence[0].license_status = CapabilityLicenseStatusV1::Restricted;
        let report = qualify_instrument_capability(&capability(), &restricted);
        assert!(report.issues.iter().any(|issue| issue.code == CapabilityEvidenceIssueCodeV1::LicenseNotPermitted));
    }

    #[test]
    fn rejects_tessitura_outside_the_sounding_range() {
        let mut profile = profile(
            CapabilityEvidenceBasisV1::Measured,
            CapabilityEvidenceBasisV1::Measured,
        );
        profile.tessitura.as_mut().unwrap().value.highest_midi = 100;
        let report = qualify_instrument_capability(&capability(), &profile);
        assert!(report.issues.iter().any(|issue| issue.code == CapabilityEvidenceIssueCodeV1::TessituraOutsideSoundingRange));
    }

    #[test]
    fn derived_evidence_requires_existing_acyclic_parents() {
        let mut missing_parent = profile(
            CapabilityEvidenceBasisV1::Derived,
            CapabilityEvidenceBasisV1::Measured,
        );
        missing_parent.evidence[0].parent_evidence_ids = vec!["missing-parent".into()];
        let report = qualify_instrument_capability(&capability(), &missing_parent);
        assert!(report.issues.iter().any(|issue| issue.code == CapabilityEvidenceIssueCodeV1::UnknownParentEvidence));

        let mut cycle = profile(
            CapabilityEvidenceBasisV1::Measured,
            CapabilityEvidenceBasisV1::Measured,
        );
        cycle.evidence.push(CapabilityEvidenceRecordV1 {
            evidence_id: "derived-cycle".into(),
            basis: CapabilityEvidenceBasisV1::Derived,
            scope: scope(),
            source_ref: None,
            source_revision: None,
            content_sha256: None,
            method_id: Some("derive-v1".into()),
            parent_evidence_ids: vec!["derived-cycle".into()],
            transform_version: Some("v1".into()),
            uncertainty_ppm: None,
            license_status: CapabilityLicenseStatusV1::Permitted,
        });
        let report = qualify_instrument_capability(&capability(), &cycle);
        assert!(report.issues.iter().any(|issue| issue.code == CapabilityEvidenceIssueCodeV1::EvidenceCycle));
    }

    #[test]
    fn rejects_instance_identity_mismatch() {
        let mut profile = profile(
            CapabilityEvidenceBasisV1::Measured,
            CapabilityEvidenceBasisV1::Measured,
        );
        profile.instrument_id = "different-cello".into();
        let report = qualify_instrument_capability(&capability(), &profile);
        assert!(report.issues.iter().any(|issue| issue.code == CapabilityEvidenceIssueCodeV1::InstrumentIdMismatch));
    }
}
