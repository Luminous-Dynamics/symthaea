// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen pilot authority for the Symthaea–Muse cognition study.
//!
//! The pilot is allowed to estimate feasibility, variance, exclusion rates,
//! and operational failure modes. It is not allowed to support confirmatory
//! musical-quality claims or silently alter the frozen confirmatory corpus.

use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::{FrozenStudyManifest, StudySplit};
use crate::methodology_plan::FrozenMethodologyPlan;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PILOT_PROTOCOL_VERSION: &str = "symthaea-muse-pilot-protocol-v1";
pub const PILOT_AMENDMENT_LEDGER_VERSION: &str = "symthaea-muse-pilot-amendment-ledger-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PilotObjective {
    RecruitmentFeasibility,
    SessionCompletion,
    InstrumentComprehension,
    TechnicalReliability,
    ExclusionRateEstimation,
    VarianceEstimation,
    SessionDurationEstimation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PermittedPilotAdaptation {
    ClarifyInstructions,
    RepairRunnerDefect,
    AdjustRecruitmentChannel,
    AdjustNonOutcomeStoppingThreshold,
    IncreasePilotSampleWithinMaximum,
    ReviseAttentionCheckWording,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotOperationalThresholds {
    pub minimum_completed_participants: usize,
    pub maximum_enrolled_participants: usize,
    pub cohort_wave_size: usize,
    pub minimum_completion_rate: f64,
    pub minimum_attention_pass_rate: f64,
    pub maximum_technical_failure_rate: f64,
    pub maximum_exclusion_rate: f64,
    pub maximum_median_session_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenPilotProtocol {
    pub protocol_version: String,
    pub manifest_sha256: String,
    pub methodology_sha256: String,
    pub external_registration_uri: String,
    pub external_registration_sha256: String,
    /// Independent commitment for pilot assignment randomization. This must not
    /// reuse the confirmatory randomization key.
    pub pilot_randomization_commitment_sha256: String,
    pub frozen_at_utc: String,
    pub pilot_fixture_ids: Vec<String>,
    pub objectives: Vec<PilotObjective>,
    pub permitted_adaptations: Vec<PermittedPilotAdaptation>,
    pub forbidden_confirmatory_claims: Vec<String>,
    pub thresholds: PilotOperationalThresholds,
    /// Pilot outcomes may inform a future confirmatory sample-size amendment,
    /// but they may never enter the confirmatory outcome dataset.
    pub pilot_data_may_enter_confirmatory_analysis: bool,
    /// Arm-labelled outcome inspection is prohibited while pilot collection is
    /// open. Operational monitoring remains blinded and aggregate-only.
    pub arm_labelled_monitoring_allowed_during_collection: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotAmendmentCategory {
    Operational,
    Instrument,
    Recruitment,
    SampleSize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PilotAmendment {
    pub sequence: u32,
    pub category: PilotAmendmentCategory,
    pub recorded_at_utc: String,
    pub rationale: String,
    pub prior_protocol_sha256: String,
    pub amended_protocol_sha256: String,
    pub changed_fields: Vec<String>,
    pub confirmatory_manifest_unchanged: bool,
    pub confirmatory_outcomes_uninspected: bool,
    pub external_receipt_uri: String,
    pub external_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PilotAmendmentLedger {
    pub ledger_version: String,
    pub initial_protocol_sha256: String,
    pub amendments: Vec<PilotAmendment>,
    pub ledger_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotProtocolIssue {
    WrongProtocolVersion {
        found: String,
    },
    InvalidManifest,
    InvalidMethodology,
    ManifestSerializationFailed,
    ManifestDigestMismatch,
    MethodologySerializationFailed,
    MethodologyDigestMismatch,
    InvalidDigest {
        field: String,
    },
    MissingExternalRegistrationField {
        field: String,
    },
    EmptyPilotFixtureRegistry,
    DuplicatePilotFixture {
        fixture_id: String,
    },
    UnknownPilotFixture {
        fixture_id: String,
    },
    NonPilotFixture {
        fixture_id: String,
    },
    MissingPilotFixture {
        fixture_id: String,
    },
    EmptyObjectiveRegistry,
    DuplicateObjective {
        objective: PilotObjective,
    },
    EmptyAdaptationRegistry,
    DuplicateAdaptation {
        adaptation: PermittedPilotAdaptation,
    },
    EmptyForbiddenClaimRegistry,
    EmptyForbiddenClaim {
        index: usize,
    },
    PilotDataMayEnterConfirmatoryAnalysis,
    ArmLabelledMonitoringAllowed,
    PilotRandomizationCommitmentReused,
    ZeroThreshold {
        field: String,
    },
    InvalidRate {
        field: String,
    },
    CompletionExceedsEnrollment,
    WaveExceedsEnrollment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PilotAmendmentIssue {
    WrongLedgerVersion { found: String },
    InvalidDigest { field: String },
    LedgerSerializationFailed,
    LedgerDigestMismatch,
    NonSequentialAmendment { expected: u32, found: u32 },
    EmptyRationale { sequence: u32 },
    EmptyChangedFields { sequence: u32 },
    EmptyChangedField { sequence: u32 },
    ConfirmatoryManifestChanged { sequence: u32 },
    ConfirmatoryOutcomesInspected { sequence: u32 },
    MissingExternalReceipt { sequence: u32 },
    BrokenProtocolChain { sequence: u32 },
}

impl FrozenPilotProtocol {
    pub fn validate(
        &self,
        manifest: &FrozenStudyManifest,
        methodology: &FrozenMethodologyPlan,
    ) -> Vec<PilotProtocolIssue> {
        let mut issues = Vec::new();
        if !manifest.validate().is_empty() {
            issues.push(PilotProtocolIssue::InvalidManifest);
        }
        if !methodology.validate(manifest).is_empty() {
            issues.push(PilotProtocolIssue::InvalidMethodology);
        }
        if self.protocol_version != PILOT_PROTOCOL_VERSION {
            issues.push(PilotProtocolIssue::WrongProtocolVersion {
                found: self.protocol_version.clone(),
            });
        }
        match canonical_json_sha256(manifest) {
            Ok(value) if value == self.manifest_sha256 => {}
            Ok(_) => issues.push(PilotProtocolIssue::ManifestDigestMismatch),
            Err(_) => issues.push(PilotProtocolIssue::ManifestSerializationFailed),
        }
        match canonical_json_sha256(methodology) {
            Ok(value) if value == self.methodology_sha256 => {}
            Ok(_) => issues.push(PilotProtocolIssue::MethodologyDigestMismatch),
            Err(_) => issues.push(PilotProtocolIssue::MethodologySerializationFailed),
        }
        for (field, digest) in [
            ("manifest_sha256", self.manifest_sha256.as_str()),
            ("methodology_sha256", self.methodology_sha256.as_str()),
            (
                "external_registration_sha256",
                self.external_registration_sha256.as_str(),
            ),
            (
                "pilot_randomization_commitment_sha256",
                self.pilot_randomization_commitment_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(PilotProtocolIssue::InvalidDigest {
                    field: field.into(),
                });
            }
        }
        for (field, value) in [
            (
                "external_registration_uri",
                self.external_registration_uri.as_str(),
            ),
            ("frozen_at_utc", self.frozen_at_utc.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(PilotProtocolIssue::MissingExternalRegistrationField {
                    field: field.into(),
                });
            }
        }

        let manifest_pilot_ids: BTreeSet<_> = manifest
            .fixtures
            .iter()
            .filter(|fixture| fixture.split == StudySplit::Pilot)
            .map(|fixture| fixture.key.fixture_id.as_str())
            .collect();
        if self.pilot_fixture_ids.is_empty() {
            issues.push(PilotProtocolIssue::EmptyPilotFixtureRegistry);
        }
        let mut declared = BTreeSet::new();
        for fixture_id in &self.pilot_fixture_ids {
            if !declared.insert(fixture_id.as_str()) {
                issues.push(PilotProtocolIssue::DuplicatePilotFixture {
                    fixture_id: fixture_id.clone(),
                });
            }
            match manifest
                .fixtures
                .iter()
                .find(|fixture| fixture.key.fixture_id == *fixture_id)
            {
                None => issues.push(PilotProtocolIssue::UnknownPilotFixture {
                    fixture_id: fixture_id.clone(),
                }),
                Some(fixture) if fixture.split != StudySplit::Pilot => {
                    issues.push(PilotProtocolIssue::NonPilotFixture {
                        fixture_id: fixture_id.clone(),
                    });
                }
                Some(_) => {}
            }
        }
        for fixture_id in manifest_pilot_ids.difference(&declared) {
            issues.push(PilotProtocolIssue::MissingPilotFixture {
                fixture_id: (*fixture_id).to_string(),
            });
        }

        validate_unique_registry(
            &self.objectives,
            PilotProtocolIssue::EmptyObjectiveRegistry,
            |value| PilotProtocolIssue::DuplicateObjective { objective: value },
            &mut issues,
        );
        validate_unique_registry(
            &self.permitted_adaptations,
            PilotProtocolIssue::EmptyAdaptationRegistry,
            |value| PilotProtocolIssue::DuplicateAdaptation { adaptation: value },
            &mut issues,
        );
        if self.forbidden_confirmatory_claims.is_empty() {
            issues.push(PilotProtocolIssue::EmptyForbiddenClaimRegistry);
        }
        for (index, claim) in self.forbidden_confirmatory_claims.iter().enumerate() {
            if claim.trim().is_empty() {
                issues.push(PilotProtocolIssue::EmptyForbiddenClaim { index });
            }
        }
        if self.pilot_randomization_commitment_sha256 == manifest.randomization_commitment_sha256 {
            issues.push(PilotProtocolIssue::PilotRandomizationCommitmentReused);
        }
        if self.pilot_data_may_enter_confirmatory_analysis {
            issues.push(PilotProtocolIssue::PilotDataMayEnterConfirmatoryAnalysis);
        }
        if self.arm_labelled_monitoring_allowed_during_collection {
            issues.push(PilotProtocolIssue::ArmLabelledMonitoringAllowed);
        }
        let thresholds = &self.thresholds;
        for (field, value) in [
            (
                "minimum_completed_participants",
                thresholds.minimum_completed_participants,
            ),
            (
                "maximum_enrolled_participants",
                thresholds.maximum_enrolled_participants,
            ),
            ("cohort_wave_size", thresholds.cohort_wave_size),
            (
                "maximum_median_session_seconds",
                thresholds.maximum_median_session_seconds as usize,
            ),
        ] {
            if value == 0 {
                issues.push(PilotProtocolIssue::ZeroThreshold {
                    field: field.into(),
                });
            }
        }
        for (field, value) in [
            (
                "minimum_completion_rate",
                thresholds.minimum_completion_rate,
            ),
            (
                "minimum_attention_pass_rate",
                thresholds.minimum_attention_pass_rate,
            ),
            (
                "maximum_technical_failure_rate",
                thresholds.maximum_technical_failure_rate,
            ),
            ("maximum_exclusion_rate", thresholds.maximum_exclusion_rate),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                issues.push(PilotProtocolIssue::InvalidRate {
                    field: field.into(),
                });
            }
        }
        if thresholds.minimum_completed_participants > thresholds.maximum_enrolled_participants {
            issues.push(PilotProtocolIssue::CompletionExceedsEnrollment);
        }
        if thresholds.cohort_wave_size > thresholds.maximum_enrolled_participants {
            issues.push(PilotProtocolIssue::WaveExceedsEnrollment);
        }
        issues
    }
}

pub fn pilot_amendment_ledger_commitment(
    ledger: &PilotAmendmentLedger,
) -> Result<String, serde_json::Error> {
    #[derive(Serialize)]
    struct Commitment<'a> {
        ledger_version: &'a str,
        initial_protocol_sha256: &'a str,
        amendments: &'a [PilotAmendment],
    }
    canonical_json_sha256(&Commitment {
        ledger_version: &ledger.ledger_version,
        initial_protocol_sha256: &ledger.initial_protocol_sha256,
        amendments: &ledger.amendments,
    })
}

pub fn seal_pilot_amendment_ledger(
    ledger: &mut PilotAmendmentLedger,
) -> Result<(), serde_json::Error> {
    ledger.ledger_sha256 = pilot_amendment_ledger_commitment(ledger)?;
    Ok(())
}

pub fn validate_pilot_amendment_ledger(ledger: &PilotAmendmentLedger) -> Vec<PilotAmendmentIssue> {
    let mut issues = Vec::new();
    if ledger.ledger_version != PILOT_AMENDMENT_LEDGER_VERSION {
        issues.push(PilotAmendmentIssue::WrongLedgerVersion {
            found: ledger.ledger_version.clone(),
        });
    }
    for (field, digest) in [
        (
            "initial_protocol_sha256",
            ledger.initial_protocol_sha256.as_str(),
        ),
        ("ledger_sha256", ledger.ledger_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PilotAmendmentIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    let mut previous = ledger.initial_protocol_sha256.as_str();
    for (index, amendment) in ledger.amendments.iter().enumerate() {
        let expected = index as u32 + 1;
        if amendment.sequence != expected {
            issues.push(PilotAmendmentIssue::NonSequentialAmendment {
                expected,
                found: amendment.sequence,
            });
        }
        for (field, digest) in [
            (
                "prior_protocol_sha256",
                amendment.prior_protocol_sha256.as_str(),
            ),
            (
                "amended_protocol_sha256",
                amendment.amended_protocol_sha256.as_str(),
            ),
            (
                "external_receipt_sha256",
                amendment.external_receipt_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(PilotAmendmentIssue::InvalidDigest {
                    field: format!("amendment.{}.{}", amendment.sequence, field),
                });
            }
        }
        if amendment.prior_protocol_sha256 != previous {
            issues.push(PilotAmendmentIssue::BrokenProtocolChain {
                sequence: amendment.sequence,
            });
        }
        previous = amendment.amended_protocol_sha256.as_str();
        if amendment.rationale.trim().is_empty() {
            issues.push(PilotAmendmentIssue::EmptyRationale {
                sequence: amendment.sequence,
            });
        }
        if amendment.changed_fields.is_empty() {
            issues.push(PilotAmendmentIssue::EmptyChangedFields {
                sequence: amendment.sequence,
            });
        }
        if amendment
            .changed_fields
            .iter()
            .any(|field| field.trim().is_empty())
        {
            issues.push(PilotAmendmentIssue::EmptyChangedField {
                sequence: amendment.sequence,
            });
        }
        if !amendment.confirmatory_manifest_unchanged {
            issues.push(PilotAmendmentIssue::ConfirmatoryManifestChanged {
                sequence: amendment.sequence,
            });
        }
        if !amendment.confirmatory_outcomes_uninspected {
            issues.push(PilotAmendmentIssue::ConfirmatoryOutcomesInspected {
                sequence: amendment.sequence,
            });
        }
        if amendment.external_receipt_uri.trim().is_empty() {
            issues.push(PilotAmendmentIssue::MissingExternalReceipt {
                sequence: amendment.sequence,
            });
        }
    }
    match pilot_amendment_ledger_commitment(ledger) {
        Ok(value) if value == ledger.ledger_sha256 => {}
        Ok(_) => issues.push(PilotAmendmentIssue::LedgerDigestMismatch),
        Err(_) => issues.push(PilotAmendmentIssue::LedgerSerializationFailed),
    }
    issues
}

fn validate_unique_registry<T, F>(
    values: &[T],
    empty_issue: PilotProtocolIssue,
    duplicate_issue: F,
    issues: &mut Vec<PilotProtocolIssue>,
) where
    T: Copy + Ord,
    F: Fn(T) -> PilotProtocolIssue,
{
    if values.is_empty() {
        issues.push(empty_issue);
        return;
    }
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(*value) {
            issues.push(duplicate_issue(*value));
        }
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amendment_ledger_detects_chain_breaks() {
        let digest = "a".repeat(64);
        let mut ledger = PilotAmendmentLedger {
            ledger_version: PILOT_AMENDMENT_LEDGER_VERSION.into(),
            initial_protocol_sha256: digest.clone(),
            amendments: vec![PilotAmendment {
                sequence: 1,
                category: PilotAmendmentCategory::Operational,
                recorded_at_utc: "2026-07-14T00:00:00Z".into(),
                rationale: "repair a runner defect".into(),
                prior_protocol_sha256: "b".repeat(64),
                amended_protocol_sha256: "c".repeat(64),
                changed_fields: vec!["runner_version".into()],
                confirmatory_manifest_unchanged: true,
                confirmatory_outcomes_uninspected: true,
                external_receipt_uri: "registry:pilot-amendment-1".into(),
                external_receipt_sha256: digest,
            }],
            ledger_sha256: String::new(),
        };
        seal_pilot_amendment_ledger(&mut ledger).unwrap();
        assert!(
            validate_pilot_amendment_ledger(&ledger)
                .iter()
                .any(|issue| matches!(
                    issue,
                    PilotAmendmentIssue::BrokenProtocolChain { sequence: 1 }
                ))
        );
    }

    #[test]
    fn arm_labelled_monitoring_is_forbidden() {
        let thresholds = PilotOperationalThresholds {
            minimum_completed_participants: 8,
            maximum_enrolled_participants: 16,
            cohort_wave_size: 4,
            minimum_completion_rate: 0.8,
            minimum_attention_pass_rate: 0.8,
            maximum_technical_failure_rate: 0.2,
            maximum_exclusion_rate: 0.25,
            maximum_median_session_seconds: 1800,
        };
        assert!(thresholds.minimum_completion_rate > thresholds.maximum_technical_failure_rate);
    }
}
