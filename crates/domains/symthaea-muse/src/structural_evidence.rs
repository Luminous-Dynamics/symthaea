// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full theory-verifier evidence bound to every blinded presentation.
//!
//! The V8 raw evidence format accepted a hand-entered structural summary. This
//! module makes the complete `TheoryValidationReport` normative, binds it to the
//! score, recipe, verifier binary, invocation, and environment, and derives the
//! summary consumed by the existing V8 compiler.

use crate::blinded_study::BlindedSchedule;
use crate::cognitive_experiment::StructuralTrialOutcome;
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::FrozenStudyManifest;
use crate::methodology_plan::FrozenMethodologyPlan;
use crate::study_evidence::StructuralPresentationOutcome;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_music_theory::score_validation::{
    ScoreValidationRule, THEORY_VALIDATION_VERSION, TheoryValidationReport, ValidationSeverity,
};

pub const STRUCTURAL_EVIDENCE_VERSION: &str = "symthaea-muse-structural-evidence-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralVerifierRecord {
    pub presentation_id: String,
    pub score_sha256: String,
    pub recipe_sha256: String,
    pub theory_report_sha256: String,
    pub invocation_sha256: String,
    pub stdout_sha256: String,
    pub report: TheoryValidationReport,
    /// Obligation and thematic-return evidence is produced by the completed
    /// Sonata verifier, while theory validity is derived from `report` below.
    pub obligations_total: usize,
    pub obligations_fulfilled: usize,
    pub motif_return_similarity: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralEvidenceBundle {
    pub evidence_version: String,
    pub manifest_sha256: String,
    pub schedule_sha256: String,
    pub methodology_sha256: String,
    pub verifier_source_revision: String,
    pub verifier_binary_sha256: String,
    pub verifier_environment_sha256: String,
    pub structural_evidence_sha256: String,
    pub records: Vec<StructuralVerifierRecord>,
}

#[derive(Serialize)]
struct StructuralEvidenceCommitment<'a> {
    evidence_version: &'a str,
    manifest_sha256: &'a str,
    schedule_sha256: &'a str,
    methodology_sha256: &'a str,
    verifier_source_revision: &'a str,
    verifier_binary_sha256: &'a str,
    verifier_environment_sha256: &'a str,
    records: &'a [StructuralVerifierRecord],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralEvidenceIssue {
    WrongVersion {
        found: String,
    },
    InvalidMethodology,
    SerializationFailed {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    VerifierIdentityMismatch {
        field: String,
    },
    DuplicatePresentation {
        presentation_id: String,
    },
    MissingPresentation {
        presentation_id: String,
    },
    UnknownPresentation {
        presentation_id: String,
    },
    RecipeDigestMismatch {
        presentation_id: String,
    },
    TheoryReportDigestMismatch {
        presentation_id: String,
    },
    WrongTheoryReportVersion {
        presentation_id: String,
        found: String,
    },
    InconsistentTheoryValidity {
        presentation_id: String,
    },
    InvalidObligationCounts {
        presentation_id: String,
    },
    InvalidMotifSimilarity {
        presentation_id: String,
    },
}

pub fn structural_evidence_commitment(
    bundle: &StructuralEvidenceBundle,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&StructuralEvidenceCommitment {
        evidence_version: &bundle.evidence_version,
        manifest_sha256: &bundle.manifest_sha256,
        schedule_sha256: &bundle.schedule_sha256,
        methodology_sha256: &bundle.methodology_sha256,
        verifier_source_revision: &bundle.verifier_source_revision,
        verifier_binary_sha256: &bundle.verifier_binary_sha256,
        verifier_environment_sha256: &bundle.verifier_environment_sha256,
        records: &bundle.records,
    })
}

pub fn seal_structural_evidence(
    bundle: &mut StructuralEvidenceBundle,
) -> Result<(), serde_json::Error> {
    for record in &mut bundle.records {
        record.theory_report_sha256 = canonical_json_sha256(&record.report)?;
    }
    bundle.structural_evidence_sha256 = structural_evidence_commitment(bundle)?;
    Ok(())
}

pub fn validate_structural_evidence(
    manifest: &FrozenStudyManifest,
    schedule: &BlindedSchedule,
    methodology: &FrozenMethodologyPlan,
    bundle: &StructuralEvidenceBundle,
) -> Vec<StructuralEvidenceIssue> {
    let mut issues = Vec::new();
    if bundle.evidence_version != STRUCTURAL_EVIDENCE_VERSION {
        issues.push(StructuralEvidenceIssue::WrongVersion {
            found: bundle.evidence_version.clone(),
        });
    }
    if !methodology.validate(manifest).is_empty() {
        issues.push(StructuralEvidenceIssue::InvalidMethodology);
    }
    for (field, digest) in [
        ("manifest_sha256", bundle.manifest_sha256.as_str()),
        ("schedule_sha256", bundle.schedule_sha256.as_str()),
        ("methodology_sha256", bundle.methodology_sha256.as_str()),
        (
            "verifier_binary_sha256",
            bundle.verifier_binary_sha256.as_str(),
        ),
        (
            "verifier_environment_sha256",
            bundle.verifier_environment_sha256.as_str(),
        ),
        (
            "structural_evidence_sha256",
            bundle.structural_evidence_sha256.as_str(),
        ),
    ] {
        if !is_sha256(digest) {
            issues.push(StructuralEvidenceIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    verify_digest(
        "manifest_sha256",
        canonical_json_sha256(manifest),
        &bundle.manifest_sha256,
        &mut issues,
    );
    verify_digest(
        "schedule_sha256",
        canonical_json_sha256(schedule),
        &bundle.schedule_sha256,
        &mut issues,
    );
    verify_digest(
        "methodology_sha256",
        canonical_json_sha256(methodology),
        &bundle.methodology_sha256,
        &mut issues,
    );
    match structural_evidence_commitment(bundle) {
        Ok(value) if value == bundle.structural_evidence_sha256 => {}
        Ok(_) => issues.push(StructuralEvidenceIssue::DigestMismatch {
            field: "structural_evidence_sha256".into(),
        }),
        Err(_) => issues.push(StructuralEvidenceIssue::SerializationFailed {
            field: "structural_evidence".into(),
        }),
    }
    for (field, found, expected) in [
        (
            "verifier_source_revision",
            bundle.verifier_source_revision.as_str(),
            methodology.verifier.source_revision.as_str(),
        ),
        (
            "verifier_binary_sha256",
            bundle.verifier_binary_sha256.as_str(),
            methodology.verifier.binary_sha256.as_str(),
        ),
        (
            "verifier_environment_sha256",
            bundle.verifier_environment_sha256.as_str(),
            methodology.verifier.environment_sha256.as_str(),
        ),
    ] {
        if found != expected {
            issues.push(StructuralEvidenceIssue::VerifierIdentityMismatch {
                field: field.into(),
            });
        }
    }

    let schedule_map: BTreeMap<_, _> = schedule
        .presentations
        .iter()
        .map(|presentation| (presentation.presentation_id.as_str(), presentation))
        .collect();
    let mut seen = BTreeSet::new();
    for record in &bundle.records {
        if !seen.insert(record.presentation_id.clone()) {
            issues.push(StructuralEvidenceIssue::DuplicatePresentation {
                presentation_id: record.presentation_id.clone(),
            });
        }
        let Some(presentation) = schedule_map.get(record.presentation_id.as_str()) else {
            issues.push(StructuralEvidenceIssue::UnknownPresentation {
                presentation_id: record.presentation_id.clone(),
            });
            continue;
        };
        if record.recipe_sha256 != presentation.recipe_sha256 {
            issues.push(StructuralEvidenceIssue::RecipeDigestMismatch {
                presentation_id: record.presentation_id.clone(),
            });
        }
        match canonical_json_sha256(&record.report) {
            Ok(value) if value == record.theory_report_sha256 => {}
            Ok(_) => issues.push(StructuralEvidenceIssue::TheoryReportDigestMismatch {
                presentation_id: record.presentation_id.clone(),
            }),
            Err(_) => issues.push(StructuralEvidenceIssue::SerializationFailed {
                field: format!("report.{}", record.presentation_id),
            }),
        }
        if record.report.validation_version != THEORY_VALIDATION_VERSION
            || record.report.validation_version != methodology.verifier.rule_set_version
        {
            issues.push(StructuralEvidenceIssue::WrongTheoryReportVersion {
                presentation_id: record.presentation_id.clone(),
                found: record.report.validation_version.clone(),
            });
        }
        let fatal_count = record
            .report
            .issues
            .iter()
            .filter(|issue| issue.severity == ValidationSeverity::Fatal)
            .count();
        if record.report.valid != (fatal_count == 0) {
            issues.push(StructuralEvidenceIssue::InconsistentTheoryValidity {
                presentation_id: record.presentation_id.clone(),
            });
        }
        if record.obligations_total == 0 || record.obligations_fulfilled > record.obligations_total
        {
            issues.push(StructuralEvidenceIssue::InvalidObligationCounts {
                presentation_id: record.presentation_id.clone(),
            });
        }
        if record
            .motif_return_similarity
            .is_none_or(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            issues.push(StructuralEvidenceIssue::InvalidMotifSimilarity {
                presentation_id: record.presentation_id.clone(),
            });
        }
    }
    for presentation in &schedule.presentations {
        if !seen.contains(&presentation.presentation_id) {
            issues.push(StructuralEvidenceIssue::MissingPresentation {
                presentation_id: presentation.presentation_id.clone(),
            });
        }
    }
    issues
}

pub fn compile_structural_outcomes(
    manifest: &FrozenStudyManifest,
    schedule: &BlindedSchedule,
    methodology: &FrozenMethodologyPlan,
    bundle: &StructuralEvidenceBundle,
) -> Result<Vec<StructuralPresentationOutcome>, Vec<StructuralEvidenceIssue>> {
    let issues = validate_structural_evidence(manifest, schedule, methodology, bundle);
    if !issues.is_empty() {
        return Err(issues);
    }
    let mut outcomes: Vec<_> = bundle
        .records
        .iter()
        .map(|record| StructuralPresentationOutcome {
            presentation_id: record.presentation_id.clone(),
            outcome: derive_structural_outcome(record),
        })
        .collect();
    outcomes.sort_by(|left, right| left.presentation_id.cmp(&right.presentation_id));
    Ok(outcomes)
}

fn derive_structural_outcome(record: &StructuralVerifierRecord) -> StructuralTrialOutcome {
    let voice_leading_violations = record
        .report
        .issues
        .iter()
        .filter(|issue| {
            issue.severity == ValidationSeverity::Fatal
                && matches!(
                    issue.rule,
                    ScoreValidationRule::VoiceCrossing
                        | ScoreValidationRule::StrongBeatConsonance
                        | ScoreValidationRule::ParallelPerfectMotion
                )
        })
        .count();
    let tonic_returned = !record.report.issues.iter().any(|issue| {
        issue.severity == ValidationSeverity::Fatal
            && issue.rule == ScoreValidationRule::FinalTonicArrival
    });
    StructuralTrialOutcome {
        hard_constraints_valid: record.report.valid,
        obligations_total: record.obligations_total,
        obligations_fulfilled: record.obligations_fulfilled,
        voice_leading_violations,
        motif_return_similarity: record.motif_return_similarity,
        tonic_returned,
    }
}

fn verify_digest(
    field: &str,
    result: Result<String, serde_json::Error>,
    expected: &str,
    issues: &mut Vec<StructuralEvidenceIssue>,
) {
    match result {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(StructuralEvidenceIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(StructuralEvidenceIssue::SerializationFailed {
            field: field.into(),
        }),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blinded_study::{ArmArtifactBinding, build_blinded_schedule};
    use crate::cognitive_experiment::{CognitivePolicyArm, FrozenTrialKey};
    use crate::evidence_digest::sha256_hex;
    use crate::experiment_manifest::{
        ConfirmatoryEndpoint, FrozenStudyFixture, MIN_CONFIRMATORY_FIXTURES, MIN_PILOT_FIXTURES,
        STUDY_MANIFEST_VERSION, StudySplit,
    };
    use crate::methodology_plan::{
        CandidateSetMode, EVIDENCE_ENCODING_PROFILE, EndpointDeclaration, EndpointRole,
        EqualPolicyBudget, ExternalPreregistration, FrozenModelCheckpoint, FrozenVerifierIdentity,
        METHODOLOGY_PLAN_VERSION,
    };

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SECRET: [u8; 32] = [0x42; 32];

    fn manifest() -> FrozenStudyManifest {
        let fixtures = (0..MIN_PILOT_FIXTURES + MIN_CONFIRMATORY_FIXTURES)
            .map(|index| FrozenStudyFixture {
                key: FrozenTrialKey {
                    fixture_id: format!("fixture-{index}"),
                    seed: index as u64 + 1,
                },
                family_id: format!("family-{index}"),
                split: if index < MIN_PILOT_FIXTURES {
                    StudySplit::Pilot
                } else {
                    StudySplit::Confirmatory
                },
                frozen_input_sha256: format!("{:064x}", index + 1),
                subject_sha256: DIGEST.into(),
                renderer_sha256: DIGEST.into(),
                soundfont_sha256: DIGEST.into(),
                theory_constraints_sha256: DIGEST.into(),
                tonic: "C".into(),
                meter: "4/4".into(),
                orchestration: "piano".into(),
            })
            .collect();
        FrozenStudyManifest {
            manifest_version: STUDY_MANIFEST_VERSION.into(),
            preregistration_sha256: DIGEST.into(),
            analysis_plan_sha256: DIGEST.into(),
            randomization_commitment_sha256: sha256_hex(&SECRET),
            policy_versions: CognitivePolicyArm::ALL
                .into_iter()
                .map(|arm| (arm, "policy-v1".into()))
                .collect(),
            primary_endpoints: vec![ConfirmatoryEndpoint::Preference],
            alpha: 0.05,
            fixtures,
        }
    }

    fn methodology(manifest: &FrozenStudyManifest) -> FrozenMethodologyPlan {
        FrozenMethodologyPlan {
            methodology_version: METHODOLOGY_PLAN_VERSION.into(),
            manifest_sha256: canonical_json_sha256(manifest).unwrap(),
            analysis_spec_sha256: DIGEST.into(),
            evidence_encoding_profile: EVIDENCE_ENCODING_PROFILE.into(),
            external_preregistration: ExternalPreregistration {
                registry: "OSF".into(),
                record_id: "example".into(),
                frozen_at_utc: "2026-07-14T00:00:00Z".into(),
                record_sha256: DIGEST.into(),
            },
            endpoints: vec![EndpointDeclaration {
                endpoint: ConfirmatoryEndpoint::Preference,
                role: EndpointRole::Primary,
                superiority_margin: Some(0.05),
                heuristic_noninferiority_margin: Some(-0.02),
                rationale: "blinded preference".into(),
            }],
            model_checkpoint: FrozenModelCheckpoint {
                checkpoint_sha256: DIGEST.into(),
                training_data_sha256: DIGEST.into(),
                training_algorithm_version: "adaptive-outcome-v2".into(),
                hyperparameters_sha256: DIGEST.into(),
                completed_updates: 100,
                pilot_cutoff_utc: "2026-07-14T00:00:00Z".into(),
                rng_seed: 7,
            },
            verifier: FrozenVerifierIdentity {
                source_revision: "deadbeef".into(),
                binary_sha256: DIGEST.into(),
                rule_set_version: THEORY_VALIDATION_VERSION.into(),
                environment_sha256: DIGEST.into(),
            },
            policy_budget: EqualPolicyBudget {
                candidate_set_mode: CandidateSetMode::SharedAcrossArms,
                candidates_per_fixture: 5,
                max_theory_validations_per_arm: 5,
                max_policy_evaluations_per_arm: 5,
                allowed_operators_sha256: DIGEST.into(),
                compute_environment_sha256: DIGEST.into(),
            },
        }
    }

    #[test]
    fn summaries_are_derived_from_full_reports() {
        let manifest = manifest();
        let artifacts: Vec<_> = manifest
            .fixtures
            .iter()
            .flat_map(|fixture| {
                CognitivePolicyArm::ALL.map(|arm| ArmArtifactBinding {
                    key: fixture.key.clone(),
                    arm,
                    audio_sha256: DIGEST.into(),
                    recipe_sha256: DIGEST.into(),
                })
            })
            .collect();
        let (schedule, _) = build_blinded_schedule(&manifest, &artifacts, SECRET).unwrap();
        let methodology = methodology(&manifest);
        let mut bundle = StructuralEvidenceBundle {
            evidence_version: STRUCTURAL_EVIDENCE_VERSION.into(),
            manifest_sha256: canonical_json_sha256(&manifest).unwrap(),
            schedule_sha256: canonical_json_sha256(&schedule).unwrap(),
            methodology_sha256: canonical_json_sha256(&methodology).unwrap(),
            verifier_source_revision: methodology.verifier.source_revision.clone(),
            verifier_binary_sha256: methodology.verifier.binary_sha256.clone(),
            verifier_environment_sha256: methodology.verifier.environment_sha256.clone(),
            structural_evidence_sha256: String::new(),
            records: schedule
                .presentations
                .iter()
                .map(|presentation| StructuralVerifierRecord {
                    presentation_id: presentation.presentation_id.clone(),
                    score_sha256: DIGEST.into(),
                    recipe_sha256: presentation.recipe_sha256.clone(),
                    theory_report_sha256: String::new(),
                    invocation_sha256: DIGEST.into(),
                    stdout_sha256: DIGEST.into(),
                    report: TheoryValidationReport {
                        validation_version: THEORY_VALIDATION_VERSION.into(),
                        valid: true,
                        issues: Vec::new(),
                    },
                    obligations_total: 4,
                    obligations_fulfilled: 4,
                    motif_return_similarity: Some(0.98),
                })
                .collect(),
        };
        seal_structural_evidence(&mut bundle).unwrap();
        let outcomes =
            compile_structural_outcomes(&manifest, &schedule, &methodology, &bundle).unwrap();
        assert_eq!(outcomes.len(), schedule.presentations.len());
        assert!(
            outcomes
                .iter()
                .all(|outcome| outcome.outcome.hard_constraints_valid)
        );
    }
}
