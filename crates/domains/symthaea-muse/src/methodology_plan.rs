// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen methodological authority layered on top of the V8 study manifest.
//!
//! V8 intentionally kept every confirmatory endpoint in one list. V8.2 adds a
//! separately committed plan that names exactly one primary endpoint, labels
//! all remaining outcomes as secondary or exploratory, freezes the model and
//! verifier identities, and gives every policy arm the same candidate and
//! validation budget.

use crate::cognitive_experiment::CognitivePolicyArm;
use crate::evidence_digest::canonical_json_sha256;
use crate::experiment_manifest::{ConfirmatoryEndpoint, FrozenStudyManifest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const METHODOLOGY_PLAN_VERSION: &str = "symthaea-muse-methodology-plan-v1";
pub const EVIDENCE_ENCODING_PROFILE: &str = "symthaea-canonical-json-sha256-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointRole {
    Primary,
    Secondary,
    Exploratory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EndpointDeclaration {
    pub endpoint: ConfirmatoryEndpoint,
    pub role: EndpointRole,
    /// Positive practical advantage required over fixed and random-valid.
    pub superiority_margin: Option<f64>,
    /// Minimum acceptable Symthaea-minus-heuristic difference. This is usually
    /// zero or negative because it is a non-inferiority margin.
    pub heuristic_noninferiority_margin: Option<f64>,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalPreregistration {
    pub registry: String,
    pub record_id: String,
    pub frozen_at_utc: String,
    pub record_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenModelCheckpoint {
    pub checkpoint_sha256: String,
    pub training_data_sha256: String,
    pub training_algorithm_version: String,
    pub hyperparameters_sha256: String,
    pub completed_updates: u64,
    pub pilot_cutoff_utc: String,
    pub rng_seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenVerifierIdentity {
    pub source_revision: String,
    pub binary_sha256: String,
    pub rule_set_version: String,
    pub environment_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateSetMode {
    /// Every policy ranks the same pre-generated, theory-valid candidate set.
    SharedAcrossArms,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EqualPolicyBudget {
    pub candidate_set_mode: CandidateSetMode,
    pub candidates_per_fixture: usize,
    pub max_theory_validations_per_arm: usize,
    pub max_policy_evaluations_per_arm: usize,
    pub allowed_operators_sha256: String,
    pub compute_environment_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenMethodologyPlan {
    pub methodology_version: String,
    pub manifest_sha256: String,
    pub analysis_spec_sha256: String,
    pub evidence_encoding_profile: String,
    pub external_preregistration: ExternalPreregistration,
    pub endpoints: Vec<EndpointDeclaration>,
    pub model_checkpoint: FrozenModelCheckpoint,
    pub verifier: FrozenVerifierIdentity,
    pub policy_budget: EqualPolicyBudget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MethodologyPlanIssue {
    WrongVersion { found: String },
    ManifestSerializationFailed,
    ManifestDigestMismatch,
    InvalidDigest { field: String },
    WrongEvidenceEncodingProfile { found: String },
    MissingExternalPreregistrationField { field: String },
    EmptyEndpointRegistry,
    DuplicateEndpoint { endpoint: ConfirmatoryEndpoint },
    EndpointNotFrozenInManifest { endpoint: ConfirmatoryEndpoint },
    ManifestEndpointNotDeclared { endpoint: ConfirmatoryEndpoint },
    PrimaryEndpointCount { found: usize },
    MissingPrimaryMargin { field: String },
    InvalidPrimaryMargin { field: String },
    EmptyEndpointRationale { endpoint: ConfirmatoryEndpoint },
    EmptyModelField { field: String },
    EmptyVerifierField { field: String },
    ZeroCandidateBudget { field: String },
}

impl FrozenMethodologyPlan {
    pub fn validate(&self, manifest: &FrozenStudyManifest) -> Vec<MethodologyPlanIssue> {
        let mut issues = Vec::new();
        if self.methodology_version != METHODOLOGY_PLAN_VERSION {
            issues.push(MethodologyPlanIssue::WrongVersion {
                found: self.methodology_version.clone(),
            });
        }
        match canonical_json_sha256(manifest) {
            Ok(value) if value == self.manifest_sha256 => {}
            Ok(_) => issues.push(MethodologyPlanIssue::ManifestDigestMismatch),
            Err(_) => issues.push(MethodologyPlanIssue::ManifestSerializationFailed),
        }
        for (field, digest) in [
            ("manifest_sha256", self.manifest_sha256.as_str()),
            ("analysis_spec_sha256", self.analysis_spec_sha256.as_str()),
            (
                "external_preregistration.record_sha256",
                self.external_preregistration.record_sha256.as_str(),
            ),
            (
                "model_checkpoint.checkpoint_sha256",
                self.model_checkpoint.checkpoint_sha256.as_str(),
            ),
            (
                "model_checkpoint.training_data_sha256",
                self.model_checkpoint.training_data_sha256.as_str(),
            ),
            (
                "model_checkpoint.hyperparameters_sha256",
                self.model_checkpoint.hyperparameters_sha256.as_str(),
            ),
            (
                "verifier.binary_sha256",
                self.verifier.binary_sha256.as_str(),
            ),
            (
                "verifier.environment_sha256",
                self.verifier.environment_sha256.as_str(),
            ),
            (
                "policy_budget.allowed_operators_sha256",
                self.policy_budget.allowed_operators_sha256.as_str(),
            ),
            (
                "policy_budget.compute_environment_sha256",
                self.policy_budget.compute_environment_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(MethodologyPlanIssue::InvalidDigest {
                    field: field.into(),
                });
            }
        }
        if self.evidence_encoding_profile != EVIDENCE_ENCODING_PROFILE {
            issues.push(MethodologyPlanIssue::WrongEvidenceEncodingProfile {
                found: self.evidence_encoding_profile.clone(),
            });
        }
        for (field, value) in [
            ("registry", self.external_preregistration.registry.as_str()),
            (
                "record_id",
                self.external_preregistration.record_id.as_str(),
            ),
            (
                "frozen_at_utc",
                self.external_preregistration.frozen_at_utc.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                issues.push(MethodologyPlanIssue::MissingExternalPreregistrationField {
                    field: field.into(),
                });
            }
        }
        if self.endpoints.is_empty() {
            issues.push(MethodologyPlanIssue::EmptyEndpointRegistry);
        }
        let frozen: BTreeSet<_> = manifest.primary_endpoints.iter().copied().collect();
        let mut declared = BTreeSet::new();
        let mut primary_count = 0usize;
        for endpoint in &self.endpoints {
            if !declared.insert(endpoint.endpoint) {
                issues.push(MethodologyPlanIssue::DuplicateEndpoint {
                    endpoint: endpoint.endpoint,
                });
            }
            if !frozen.contains(&endpoint.endpoint) {
                issues.push(MethodologyPlanIssue::EndpointNotFrozenInManifest {
                    endpoint: endpoint.endpoint,
                });
            }
            if endpoint.rationale.trim().is_empty() {
                issues.push(MethodologyPlanIssue::EmptyEndpointRationale {
                    endpoint: endpoint.endpoint,
                });
            }
            if endpoint.role == EndpointRole::Primary {
                primary_count += 1;
                for (field, margin) in [
                    ("superiority_margin", endpoint.superiority_margin),
                    (
                        "heuristic_noninferiority_margin",
                        endpoint.heuristic_noninferiority_margin,
                    ),
                ] {
                    match margin {
                        None => issues.push(MethodologyPlanIssue::MissingPrimaryMargin {
                            field: field.into(),
                        }),
                        Some(value) if !value.is_finite() => {
                            issues.push(MethodologyPlanIssue::InvalidPrimaryMargin {
                                field: field.into(),
                            });
                        }
                        Some(_) => {}
                    }
                }
                if endpoint
                    .superiority_margin
                    .is_some_and(|value| value <= 0.0)
                {
                    issues.push(MethodologyPlanIssue::InvalidPrimaryMargin {
                        field: "superiority_margin".into(),
                    });
                }
                if endpoint
                    .heuristic_noninferiority_margin
                    .is_some_and(|value| value > 0.0)
                {
                    issues.push(MethodologyPlanIssue::InvalidPrimaryMargin {
                        field: "heuristic_noninferiority_margin".into(),
                    });
                }
            }
        }
        for endpoint in frozen.difference(&declared) {
            issues.push(MethodologyPlanIssue::ManifestEndpointNotDeclared {
                endpoint: *endpoint,
            });
        }
        if primary_count != 1 {
            issues.push(MethodologyPlanIssue::PrimaryEndpointCount {
                found: primary_count,
            });
        }
        for (field, value) in [
            (
                "training_algorithm_version",
                self.model_checkpoint.training_algorithm_version.as_str(),
            ),
            (
                "pilot_cutoff_utc",
                self.model_checkpoint.pilot_cutoff_utc.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                issues.push(MethodologyPlanIssue::EmptyModelField {
                    field: field.into(),
                });
            }
        }
        for (field, value) in [
            ("source_revision", self.verifier.source_revision.as_str()),
            ("rule_set_version", self.verifier.rule_set_version.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(MethodologyPlanIssue::EmptyVerifierField {
                    field: field.into(),
                });
            }
        }
        for (field, value) in [
            (
                "candidates_per_fixture",
                self.policy_budget.candidates_per_fixture,
            ),
            (
                "max_theory_validations_per_arm",
                self.policy_budget.max_theory_validations_per_arm,
            ),
            (
                "max_policy_evaluations_per_arm",
                self.policy_budget.max_policy_evaluations_per_arm,
            ),
        ] {
            if value == 0 {
                issues.push(MethodologyPlanIssue::ZeroCandidateBudget {
                    field: field.into(),
                });
            }
        }
        issues
    }

    pub fn primary_endpoint(&self) -> Option<&EndpointDeclaration> {
        self.endpoints
            .iter()
            .find(|endpoint| endpoint.role == EndpointRole::Primary)
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_experiment::FrozenTrialKey;
    use crate::experiment_manifest::{
        FrozenStudyFixture, MIN_CONFIRMATORY_FIXTURES, MIN_PILOT_FIXTURES, STUDY_MANIFEST_VERSION,
        StudySplit,
    };
    use std::collections::BTreeMap;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

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
            randomization_commitment_sha256: DIGEST.into(),
            policy_versions: CognitivePolicyArm::ALL
                .into_iter()
                .map(|arm| (arm, "policy-v1".into()))
                .collect::<BTreeMap<_, _>>(),
            primary_endpoints: vec![
                ConfirmatoryEndpoint::Preference,
                ConfirmatoryEndpoint::ReturnRecognition,
            ],
            alpha: 0.05,
            fixtures,
        }
    }

    fn plan(manifest: &FrozenStudyManifest) -> FrozenMethodologyPlan {
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
            endpoints: vec![
                EndpointDeclaration {
                    endpoint: ConfirmatoryEndpoint::Preference,
                    role: EndpointRole::Primary,
                    superiority_margin: Some(0.05),
                    heuristic_noninferiority_margin: Some(-0.02),
                    rationale: "direct blinded preference".into(),
                },
                EndpointDeclaration {
                    endpoint: ConfirmatoryEndpoint::ReturnRecognition,
                    role: EndpointRole::Secondary,
                    superiority_margin: None,
                    heuristic_noninferiority_margin: None,
                    rationale: "thematic identity check".into(),
                },
            ],
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
                rule_set_version: "theory-validation-v1".into(),
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
    fn one_primary_endpoint_and_equal_budget_validate() {
        let manifest = manifest();
        assert!(plan(&manifest).validate(&manifest).is_empty());
    }

    #[test]
    fn multiple_primary_endpoints_are_rejected() {
        let manifest = manifest();
        let mut plan = plan(&manifest);
        plan.endpoints[1].role = EndpointRole::Primary;
        assert!(
            plan.validate(&manifest)
                .iter()
                .any(|issue| matches!(issue, MethodologyPlanIssue::PrimaryEndpointCount { .. }))
        );
    }
}
