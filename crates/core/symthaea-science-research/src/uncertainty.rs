// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed scientific uncertainty budgets.
//!
//! Different uncertainty sources are retained separately rather than collapsed
//! into a universal scalar confidence. Missing required uncertainty is explicit
//! `Incomplete`, never an implicit zero. Domain-specific numerical aggregation
//! belongs in adapters; this module governs identity, completeness, provenance,
//! and conservative closure.

use crate::{FramedDigest, ResearchId, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const UNCERTAINTY_BUDGET_SCHEMA: &str = "symthaea.uncertainty-budget.v1";
const BUDGET_DOMAIN: &str = "symthaea.uncertainty-budget.identity.v1";
const REPORT_DOMAIN: &str = "symthaea.uncertainty-report.identity.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum UncertaintyKind {
    Measurement,
    Sampling,
    Aleatoric,
    Epistemic,
    Parametric,
    Numerical,
    Structural,
    ModelForm,
    DistributionShift,
    Formalization,
    Provenance,
    Calibration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum UncertaintyEstimateStatus {
    /// A distribution, variance, covariance, posterior, or other quantitative
    /// representation is committed by `estimate_artifact_sha256`.
    Quantified,
    /// Only a defensible bound/range is available.
    Bounded,
    /// The uncertainty is relevant but currently unknown.
    Unknown,
    /// The source does not apply to this scientific subject under the committed
    /// scope. This is complete only when the requirement explicitly allows it.
    NotApplicable,
    /// Evidence establishes that the available uncertainty estimate is invalid.
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UncertaintyRequirement {
    pub kind: UncertaintyKind,
    pub required: bool,
    pub allow_not_applicable: bool,
    /// Exact domain-specific definition of what must be assessed for this kind.
    pub requirement_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UncertaintyEstimate {
    pub kind: UncertaintyKind,
    pub status: UncertaintyEstimateStatus,
    /// Quantitative distribution/bound artifact. Required for Quantified/Bounded.
    pub estimate_artifact_sha256: Option<Sha256Digest>,
    /// Exact estimation/bounding method. Required for Quantified/Bounded.
    pub method_sha256: Option<Sha256Digest>,
    /// Required when status is Unknown, NotApplicable, or Invalid.
    pub justification_sha256: Option<Sha256Digest>,
    /// Evidence/data/calibration/model roots supporting this estimate.
    pub provenance_roots: BTreeSet<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UncertaintyBudget {
    pub schema_version: String,
    pub budget_id: ResearchId,
    pub subject_sha256: Sha256Digest,
    /// Optional lineage binding for study-specific uncertainty.
    pub protocol_sha256: Option<Sha256Digest>,
    /// Execution binding requires a protocol binding as well.
    pub execution_sha256: Option<Sha256Digest>,
    pub requirements: Vec<UncertaintyRequirement>,
    pub estimates: Vec<UncertaintyEstimate>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UncertaintyIssue {
    WrongSchemaVersion { found: String },
    ExecutionWithoutProtocol,
    EmptyRequirements,
    DuplicateRequirement { kind: UncertaintyKind },
    DuplicateEstimate { kind: UncertaintyKind },
    EstimateWithoutRequirement { kind: UncertaintyKind },
    QuantifiedEstimateMissingArtifact { kind: UncertaintyKind },
    QuantifiedEstimateMissingMethod { kind: UncertaintyKind },
    QuantifiedEstimateMissingProvenance { kind: UncertaintyKind },
    NonQuantifiedEstimateCarriesArtifact { kind: UncertaintyKind },
    MissingJustification { kind: UncertaintyKind },
}

impl UncertaintyBudget {
    pub fn validate(&self) -> Vec<UncertaintyIssue> {
        let mut issues = Vec::new();
        if self.schema_version != UNCERTAINTY_BUDGET_SCHEMA {
            issues.push(UncertaintyIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if self.execution_sha256.is_some() && self.protocol_sha256.is_none() {
            issues.push(UncertaintyIssue::ExecutionWithoutProtocol);
        }
        if self.requirements.is_empty() {
            issues.push(UncertaintyIssue::EmptyRequirements);
        }

        let mut requirements = BTreeSet::new();
        for requirement in &self.requirements {
            if !requirements.insert(requirement.kind) {
                issues.push(UncertaintyIssue::DuplicateRequirement {
                    kind: requirement.kind,
                });
            }
        }

        let mut estimates = BTreeSet::new();
        for estimate in &self.estimates {
            if !estimates.insert(estimate.kind) {
                issues.push(UncertaintyIssue::DuplicateEstimate {
                    kind: estimate.kind,
                });
            }
            if !requirements.contains(&estimate.kind) {
                issues.push(UncertaintyIssue::EstimateWithoutRequirement {
                    kind: estimate.kind,
                });
            }
            match estimate.status {
                UncertaintyEstimateStatus::Quantified | UncertaintyEstimateStatus::Bounded => {
                    if estimate.estimate_artifact_sha256.is_none() {
                        issues.push(UncertaintyIssue::QuantifiedEstimateMissingArtifact {
                            kind: estimate.kind,
                        });
                    }
                    if estimate.method_sha256.is_none() {
                        issues.push(UncertaintyIssue::QuantifiedEstimateMissingMethod {
                            kind: estimate.kind,
                        });
                    }
                    if estimate.provenance_roots.is_empty() {
                        issues.push(UncertaintyIssue::QuantifiedEstimateMissingProvenance {
                            kind: estimate.kind,
                        });
                    }
                }
                UncertaintyEstimateStatus::Unknown
                | UncertaintyEstimateStatus::NotApplicable
                | UncertaintyEstimateStatus::Invalid => {
                    if estimate.estimate_artifact_sha256.is_some() {
                        issues.push(UncertaintyIssue::NonQuantifiedEstimateCarriesArtifact {
                            kind: estimate.kind,
                        });
                    }
                    if estimate.justification_sha256.is_none() {
                        issues.push(UncertaintyIssue::MissingJustification {
                            kind: estimate.kind,
                        });
                    }
                }
            }
        }
        issues
    }

    pub fn freeze(self) -> Result<FrozenUncertaintyBudget, Vec<UncertaintyIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        let budget_sha256 = self.compute_digest();
        Ok(FrozenUncertaintyBudget {
            budget: self,
            budget_sha256,
        })
    }

    fn compute_digest(&self) -> Sha256Digest {
        let mut digest = FramedDigest::new(BUDGET_DOMAIN);
        digest.text(UNCERTAINTY_BUDGET_SCHEMA);
        digest.text(self.budget_id.as_str());
        digest.text(self.subject_sha256.as_str());
        digest_optional_sha(&mut digest, self.protocol_sha256.as_ref());
        digest_optional_sha(&mut digest, self.execution_sha256.as_ref());

        let mut requirements = self.requirements.clone();
        requirements.sort_by_key(|item| item.kind);
        for item in requirements {
            digest.text("requirement");
            digest.text(uncertainty_kind_tag(item.kind));
            digest.text(if item.required { "required" } else { "optional" });
            digest.text(if item.allow_not_applicable {
                "not-applicable-allowed"
            } else {
                "not-applicable-forbidden"
            });
            digest.text(item.requirement_sha256.as_str());
        }

        let mut estimates = self.estimates.clone();
        estimates.sort_by_key(|item| item.kind);
        for item in estimates {
            digest.text("estimate");
            digest.text(uncertainty_kind_tag(item.kind));
            digest.text(estimate_status_tag(item.status));
            digest_optional_sha(&mut digest, item.estimate_artifact_sha256.as_ref());
            digest_optional_sha(&mut digest, item.method_sha256.as_ref());
            digest_optional_sha(&mut digest, item.justification_sha256.as_ref());
            for root in item.provenance_roots {
                digest.text("root");
                digest.text(root.as_str());
            }
        }
        digest.digest()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenUncertaintyBudget {
    budget: UncertaintyBudget,
    budget_sha256: Sha256Digest,
}

impl FrozenUncertaintyBudget {
    pub fn budget(&self) -> &UncertaintyBudget {
        &self.budget
    }
    pub fn budget_sha256(&self) -> &Sha256Digest {
        &self.budget_sha256
    }
    pub fn report(&self) -> UncertaintyReport {
        UncertaintyReport::derive(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum UncertaintyClosure {
    Complete,
    Incomplete,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UncertaintyFinding {
    MissingRequiredEstimate { kind: UncertaintyKind },
    RequiredEstimateUnknown { kind: UncertaintyKind },
    RequiredEstimateInvalid { kind: UncertaintyKind },
    NotApplicableForbidden { kind: UncertaintyKind },
    OptionalEstimateUnknown { kind: UncertaintyKind },
    OptionalEstimateInvalid { kind: UncertaintyKind },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UncertaintyReport {
    budget_sha256: Sha256Digest,
    closure: UncertaintyClosure,
    findings: Vec<UncertaintyFinding>,
    report_sha256: Sha256Digest,
}

impl UncertaintyReport {
    fn derive(frozen: &FrozenUncertaintyBudget) -> Self {
        let budget = frozen.budget();
        let estimates = budget
            .estimates
            .iter()
            .map(|item| (item.kind, item))
            .collect::<BTreeMap<_, _>>();
        let mut findings = Vec::new();

        for requirement in &budget.requirements {
            let Some(estimate) = estimates.get(&requirement.kind).copied() else {
                if requirement.required {
                    findings.push(UncertaintyFinding::MissingRequiredEstimate {
                        kind: requirement.kind,
                    });
                }
                continue;
            };
            match estimate.status {
                UncertaintyEstimateStatus::Quantified | UncertaintyEstimateStatus::Bounded => {}
                UncertaintyEstimateStatus::Unknown if requirement.required => {
                    findings.push(UncertaintyFinding::RequiredEstimateUnknown {
                        kind: requirement.kind,
                    });
                }
                UncertaintyEstimateStatus::Unknown => {
                    findings.push(UncertaintyFinding::OptionalEstimateUnknown {
                        kind: requirement.kind,
                    });
                }
                UncertaintyEstimateStatus::Invalid if requirement.required => {
                    findings.push(UncertaintyFinding::RequiredEstimateInvalid {
                        kind: requirement.kind,
                    });
                }
                UncertaintyEstimateStatus::Invalid => {
                    findings.push(UncertaintyFinding::OptionalEstimateInvalid {
                        kind: requirement.kind,
                    });
                }
                UncertaintyEstimateStatus::NotApplicable if !requirement.allow_not_applicable => {
                    findings.push(UncertaintyFinding::NotApplicableForbidden {
                        kind: requirement.kind,
                    });
                }
                UncertaintyEstimateStatus::NotApplicable => {}
            }
        }
        findings.sort();

        let invalid = findings.iter().any(|finding| {
            matches!(
                finding,
                UncertaintyFinding::RequiredEstimateInvalid { .. }
                    | UncertaintyFinding::NotApplicableForbidden { .. }
            )
        });
        let incomplete = findings.iter().any(|finding| {
            matches!(
                finding,
                UncertaintyFinding::MissingRequiredEstimate { .. }
                    | UncertaintyFinding::RequiredEstimateUnknown { .. }
            )
        });
        let closure = if invalid {
            UncertaintyClosure::Invalid
        } else if incomplete {
            UncertaintyClosure::Incomplete
        } else {
            UncertaintyClosure::Complete
        };
        let report_sha256 = report_digest(frozen.budget_sha256(), closure, &findings);
        Self {
            budget_sha256: frozen.budget_sha256().clone(),
            closure,
            findings,
            report_sha256,
        }
    }

    pub fn closure(&self) -> UncertaintyClosure {
        self.closure
    }
    pub fn findings(&self) -> &[UncertaintyFinding] {
        &self.findings
    }
    pub fn report_sha256(&self) -> &Sha256Digest {
        &self.report_sha256
    }
}

fn report_digest(
    budget_sha256: &Sha256Digest,
    closure: UncertaintyClosure,
    findings: &[UncertaintyFinding],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(REPORT_DOMAIN);
    digest.text(budget_sha256.as_str());
    digest.text(closure_tag(closure));
    for finding in findings {
        digest.text(finding_tag(finding));
        digest.text(uncertainty_kind_tag(finding_kind(finding)));
    }
    digest.digest()
}

fn finding_kind(finding: &UncertaintyFinding) -> UncertaintyKind {
    match finding {
        UncertaintyFinding::MissingRequiredEstimate { kind }
        | UncertaintyFinding::RequiredEstimateUnknown { kind }
        | UncertaintyFinding::RequiredEstimateInvalid { kind }
        | UncertaintyFinding::NotApplicableForbidden { kind }
        | UncertaintyFinding::OptionalEstimateUnknown { kind }
        | UncertaintyFinding::OptionalEstimateInvalid { kind } => *kind,
    }
}

const fn finding_tag(finding: &UncertaintyFinding) -> &'static str {
    match finding {
        UncertaintyFinding::MissingRequiredEstimate { .. } => "missing-required-estimate",
        UncertaintyFinding::RequiredEstimateUnknown { .. } => "required-estimate-unknown",
        UncertaintyFinding::RequiredEstimateInvalid { .. } => "required-estimate-invalid",
        UncertaintyFinding::NotApplicableForbidden { .. } => "not-applicable-forbidden",
        UncertaintyFinding::OptionalEstimateUnknown { .. } => "optional-estimate-unknown",
        UncertaintyFinding::OptionalEstimateInvalid { .. } => "optional-estimate-invalid",
    }
}

fn digest_optional_sha(digest: &mut FramedDigest, value: Option<&Sha256Digest>) {
    match value {
        Some(value) => {
            digest.text("some");
            digest.text(value.as_str());
        }
        None => digest.text("none"),
    }
}

const fn uncertainty_kind_tag(value: UncertaintyKind) -> &'static str {
    match value {
        UncertaintyKind::Measurement => "measurement",
        UncertaintyKind::Sampling => "sampling",
        UncertaintyKind::Aleatoric => "aleatoric",
        UncertaintyKind::Epistemic => "epistemic",
        UncertaintyKind::Parametric => "parametric",
        UncertaintyKind::Numerical => "numerical",
        UncertaintyKind::Structural => "structural",
        UncertaintyKind::ModelForm => "model-form",
        UncertaintyKind::DistributionShift => "distribution-shift",
        UncertaintyKind::Formalization => "formalization",
        UncertaintyKind::Provenance => "provenance",
        UncertaintyKind::Calibration => "calibration",
    }
}
const fn estimate_status_tag(value: UncertaintyEstimateStatus) -> &'static str {
    match value {
        UncertaintyEstimateStatus::Quantified => "quantified",
        UncertaintyEstimateStatus::Bounded => "bounded",
        UncertaintyEstimateStatus::Unknown => "unknown",
        UncertaintyEstimateStatus::NotApplicable => "not-applicable",
        UncertaintyEstimateStatus::Invalid => "invalid",
    }
}
const fn closure_tag(value: UncertaintyClosure) -> &'static str {
    match value {
        UncertaintyClosure::Complete => "complete",
        UncertaintyClosure::Incomplete => "incomplete",
        UncertaintyClosure::Invalid => "invalid",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }
    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }
    fn requirement(kind: UncertaintyKind, required: bool) -> UncertaintyRequirement {
        UncertaintyRequirement {
            kind,
            required,
            allow_not_applicable: false,
            requirement_sha256: sha(&format!("requirement-{kind:?}")),
        }
    }
    fn quantified(kind: UncertaintyKind) -> UncertaintyEstimate {
        UncertaintyEstimate {
            kind,
            status: UncertaintyEstimateStatus::Quantified,
            estimate_artifact_sha256: Some(sha(&format!("estimate-{kind:?}"))),
            method_sha256: Some(sha(&format!("method-{kind:?}"))),
            justification_sha256: None,
            provenance_roots: BTreeSet::from([sha(&format!("root-{kind:?}"))]),
        }
    }
    fn budget() -> UncertaintyBudget {
        UncertaintyBudget {
            schema_version: UNCERTAINTY_BUDGET_SCHEMA.into(),
            budget_id: id("SCI-UNCERTAINTY-001"),
            subject_sha256: sha("subject"),
            protocol_sha256: Some(sha("protocol")),
            execution_sha256: Some(sha("execution")),
            requirements: vec![
                requirement(UncertaintyKind::Measurement, true),
                requirement(UncertaintyKind::Epistemic, true),
                requirement(UncertaintyKind::DistributionShift, false),
            ],
            estimates: vec![
                quantified(UncertaintyKind::Measurement),
                quantified(UncertaintyKind::Epistemic),
            ],
        }
    }

    #[test]
    fn missing_required_uncertainty_is_incomplete_not_zero() {
        let mut draft = budget();
        draft
            .estimates
            .retain(|item| item.kind != UncertaintyKind::Epistemic);
        let report = draft.freeze().unwrap().report();
        assert_eq!(report.closure(), UncertaintyClosure::Incomplete);
        assert!(report.findings().iter().any(|finding| matches!(
            finding,
            UncertaintyFinding::MissingRequiredEstimate {
                kind: UncertaintyKind::Epistemic
            }
        )));
    }

    #[test]
    fn unknown_required_uncertainty_is_incomplete() {
        let mut draft = budget();
        let estimate = draft
            .estimates
            .iter_mut()
            .find(|item| item.kind == UncertaintyKind::Epistemic)
            .unwrap();
        estimate.status = UncertaintyEstimateStatus::Unknown;
        estimate.estimate_artifact_sha256 = None;
        estimate.method_sha256 = None;
        estimate.justification_sha256 = Some(sha("cannot-estimate-yet"));
        estimate.provenance_roots.clear();
        assert_eq!(draft.freeze().unwrap().report().closure(), UncertaintyClosure::Incomplete);
    }

    #[test]
    fn quantified_estimate_requires_method_artifact_and_provenance() {
        let mut draft = budget();
        let estimate = draft
            .estimates
            .iter_mut()
            .find(|item| item.kind == UncertaintyKind::Measurement)
            .unwrap();
        estimate.method_sha256 = None;
        estimate.provenance_roots.clear();
        let issues = draft.freeze().unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            UncertaintyIssue::QuantifiedEstimateMissingMethod { .. }
        )));
        assert!(issues.iter().any(|issue| matches!(
            issue,
            UncertaintyIssue::QuantifiedEstimateMissingProvenance { .. }
        )));
    }

    #[test]
    fn forbidden_not_applicable_is_invalid() {
        let mut draft = budget();
        let estimate = draft
            .estimates
            .iter_mut()
            .find(|item| item.kind == UncertaintyKind::Epistemic)
            .unwrap();
        estimate.status = UncertaintyEstimateStatus::NotApplicable;
        estimate.estimate_artifact_sha256 = None;
        estimate.method_sha256 = None;
        estimate.justification_sha256 = Some(sha("claimed-not-applicable"));
        estimate.provenance_roots.clear();
        assert_eq!(draft.freeze().unwrap().report().closure(), UncertaintyClosure::Invalid);
    }

    #[test]
    fn explicit_allowed_not_applicable_can_close() {
        let mut draft = budget();
        let requirement = draft
            .requirements
            .iter_mut()
            .find(|item| item.kind == UncertaintyKind::DistributionShift)
            .unwrap();
        requirement.required = true;
        requirement.allow_not_applicable = true;
        draft.estimates.push(UncertaintyEstimate {
            kind: UncertaintyKind::DistributionShift,
            status: UncertaintyEstimateStatus::NotApplicable,
            estimate_artifact_sha256: None,
            method_sha256: None,
            justification_sha256: Some(sha("closed-domain-proof")),
            provenance_roots: BTreeSet::new(),
        });
        assert_eq!(draft.freeze().unwrap().report().closure(), UncertaintyClosure::Complete);
    }

    #[test]
    fn invalid_required_estimate_invalidates_budget() {
        let mut draft = budget();
        let estimate = draft
            .estimates
            .iter_mut()
            .find(|item| item.kind == UncertaintyKind::Epistemic)
            .unwrap();
        estimate.status = UncertaintyEstimateStatus::Invalid;
        estimate.estimate_artifact_sha256 = None;
        estimate.method_sha256 = None;
        estimate.justification_sha256 = Some(sha("calibration-failure"));
        estimate.provenance_roots.clear();
        assert_eq!(draft.freeze().unwrap().report().closure(), UncertaintyClosure::Invalid);
    }

    #[test]
    fn order_does_not_change_budget_identity() {
        let left = budget().freeze().unwrap();
        let mut reordered = budget();
        reordered.requirements.reverse();
        reordered.estimates.reverse();
        let right = reordered.freeze().unwrap();
        assert_eq!(left.budget_sha256(), right.budget_sha256());
    }

    #[test]
    fn execution_requires_protocol_binding() {
        let mut draft = budget();
        draft.protocol_sha256 = None;
        assert!(draft
            .freeze()
            .unwrap_err()
            .contains(&UncertaintyIssue::ExecutionWithoutProtocol));
    }
}
