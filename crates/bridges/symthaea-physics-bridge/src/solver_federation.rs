// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing cross-solver comparison receipts.
//!
//! Numerical agreement and methodological independence are different claims.
//! This module can establish that exact result artifacts agree or disagree under
//! an exact comparison policy. It also records known shared lineage, but it never
//! upgrades method diversity or absence of known shared roots into independence.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_science_research::{ResearchId, Sha256Digest, SharedRoot};

pub const SOLVER_FEDERATION_SCHEMA: &str = "symthaea.solver-federation.v1";
const FEDERATION_DIGEST_DOMAIN: &str = "symthaea.solver-federation.identity.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum SolverMethodFamily {
    Analytic,
    Symbolic,
    OdeIntegrator,
    FiniteDifference,
    FiniteVolume,
    FiniteElement,
    Spectral,
    Lattice,
    MonteCarlo,
    ReducedOrder,
    PhysicsInformedNeuralNetwork,
    NeuralOperator,
    HdcDiscoveredModel,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverReceipt {
    pub solver_id: ResearchId,
    pub subject_sha256: Sha256Digest,
    pub model_contract_sha256: Sha256Digest,
    pub method_family: SolverMethodFamily,
    pub implementation_sha256: Sha256Digest,
    pub environment_sha256: Sha256Digest,
    pub input_sha256: Sha256Digest,
    pub result_sha256: Sha256Digest,
    pub diagnostics_sha256: Sha256Digest,
    /// Known dependency ancestry beyond the explicit implementation,
    /// environment, and input identities above.
    pub dependency_roots: BTreeSet<SharedRoot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverFederationSpec {
    pub schema_version: String,
    pub federation_id: ResearchId,
    pub subject_sha256: Sha256Digest,
    pub model_contract_sha256: Sha256Digest,
    pub comparison_metric_sha256: Sha256Digest,
    pub agreement_policy_sha256: Sha256Digest,
    pub minimum_solvers: usize,
    pub minimum_distinct_method_families: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverFederationSpecIssue {
    WrongSchemaVersion { found: String },
    MinimumSolversTooSmall,
    MethodFamilyRequirementZero,
    MethodFamilyRequirementExceedsSolverMinimum,
}

impl SolverFederationSpec {
    pub fn validate(&self) -> Vec<SolverFederationSpecIssue> {
        let mut issues = Vec::new();
        if self.schema_version != SOLVER_FEDERATION_SCHEMA {
            issues.push(SolverFederationSpecIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if self.minimum_solvers < 2 {
            issues.push(SolverFederationSpecIssue::MinimumSolversTooSmall);
        }
        if self.minimum_distinct_method_families == 0 {
            issues.push(SolverFederationSpecIssue::MethodFamilyRequirementZero);
        }
        if self.minimum_distinct_method_families > self.minimum_solvers {
            issues.push(SolverFederationSpecIssue::MethodFamilyRequirementExceedsSolverMinimum);
        }
        issues
    }

    pub fn freeze(self) -> Result<FrozenSolverFederationSpec, Vec<SolverFederationSpecIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        let spec_sha256 = spec_digest(&self);
        Ok(FrozenSolverFederationSpec {
            spec: self,
            spec_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenSolverFederationSpec {
    spec: SolverFederationSpec,
    spec_sha256: Sha256Digest,
}

impl FrozenSolverFederationSpec {
    pub fn spec(&self) -> &SolverFederationSpec {
        &self.spec
    }

    pub fn spec_sha256(&self) -> &Sha256Digest {
        &self.spec_sha256
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum PairwiseAgreement {
    Agree,
    Disagree,
    Incomparable,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverPairComparison {
    pub left_solver_id: ResearchId,
    pub right_solver_id: ResearchId,
    pub left_result_sha256: Sha256Digest,
    pub right_result_sha256: Sha256Digest,
    pub comparison_metric_sha256: Sha256Digest,
    pub agreement_policy_sha256: Sha256Digest,
    pub outcome: PairwiseAgreement,
    /// Required for Agree/Disagree.
    pub comparison_artifact_sha256: Option<Sha256Digest>,
    /// Required for Incomparable/Invalid.
    pub justification_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PairLineage {
    pub left_solver_id: ResearchId,
    pub right_solver_id: ResearchId,
    pub same_method_family: bool,
    pub shared_implementation: bool,
    pub shared_environment: bool,
    pub shared_input: bool,
    pub shared_declared_roots: BTreeSet<SharedRoot>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum FederationFinding {
    DuplicateSolver { solver_id: ResearchId },
    SolverSubjectMismatch { solver_id: ResearchId },
    SolverContractMismatch { solver_id: ResearchId },
    DuplicatePair { left: ResearchId, right: ResearchId },
    SelfComparison { solver_id: ResearchId },
    UnknownSolver { solver_id: ResearchId },
    ResultSubstitution { solver_id: ResearchId },
    MetricSubstitution { left: ResearchId, right: ResearchId },
    AgreementPolicySubstitution { left: ResearchId, right: ResearchId },
    ComparisonArtifactMissing { left: ResearchId, right: ResearchId },
    ComparisonJustificationMissing { left: ResearchId, right: ResearchId },
    MissingComparison { left: ResearchId, right: ResearchId },
    InsufficientSolvers { observed: usize, required: usize },
    InsufficientMethodDiversity { observed: usize, required: usize },
    SharedResultArtifact { left: ResearchId, right: ResearchId },
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum FederationCoverage {
    Complete,
    Incomplete,
    Invalid,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum SolverAgreementState {
    AgreementObserved,
    DisagreementObserved,
    Incomplete,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SolverFederationReport {
    pub federation_sha256: Sha256Digest,
    pub subject_sha256: Sha256Digest,
    pub model_contract_sha256: Sha256Digest,
    pub solver_ids: BTreeSet<ResearchId>,
    pub distinct_method_families: BTreeSet<SolverMethodFamily>,
    pub coverage: FederationCoverage,
    pub agreement: SolverAgreementState,
    pub lineage: Vec<PairLineage>,
    pub findings: Vec<FederationFinding>,
    /// Deliberately fixed. Agreement or method diversity never establishes
    /// independence authority in this layer.
    pub independence_established: bool,
}

pub fn evaluate_solver_federation(
    frozen: &FrozenSolverFederationSpec,
    solver_receipts: &[SolverReceipt],
    comparisons: &[SolverPairComparison],
) -> SolverFederationReport {
    let spec = frozen.spec();
    let mut findings = Vec::new();
    let mut receipts = BTreeMap::new();
    let mut methods = BTreeSet::new();
    let mut invalid = false;

    for receipt in solver_receipts {
        if receipts
            .insert(receipt.solver_id.clone(), receipt)
            .is_some()
        {
            findings.push(FederationFinding::DuplicateSolver {
                solver_id: receipt.solver_id.clone(),
            });
            invalid = true;
            continue;
        }
        if receipt.subject_sha256 != spec.subject_sha256 {
            findings.push(FederationFinding::SolverSubjectMismatch {
                solver_id: receipt.solver_id.clone(),
            });
            invalid = true;
        }
        if receipt.model_contract_sha256 != spec.model_contract_sha256 {
            findings.push(FederationFinding::SolverContractMismatch {
                solver_id: receipt.solver_id.clone(),
            });
            invalid = true;
        }
        methods.insert(receipt.method_family);
    }

    if receipts.len() < spec.minimum_solvers {
        findings.push(FederationFinding::InsufficientSolvers {
            observed: receipts.len(),
            required: spec.minimum_solvers,
        });
    }
    if methods.len() < spec.minimum_distinct_method_families {
        findings.push(FederationFinding::InsufficientMethodDiversity {
            observed: methods.len(),
            required: spec.minimum_distinct_method_families,
        });
    }

    let solver_ids: Vec<_> = receipts.keys().cloned().collect();
    let mut expected_pairs = BTreeSet::new();
    let mut lineage = Vec::new();
    for left_index in 0..solver_ids.len() {
        for right_index in (left_index + 1)..solver_ids.len() {
            let left_id = solver_ids[left_index].clone();
            let right_id = solver_ids[right_index].clone();
            expected_pairs.insert((left_id.clone(), right_id.clone()));
            let left = receipts[&left_id];
            let right = receipts[&right_id];
            let shared_declared_roots = left
                .dependency_roots
                .intersection(&right.dependency_roots)
                .cloned()
                .collect();
            lineage.push(PairLineage {
                left_solver_id: left_id.clone(),
                right_solver_id: right_id.clone(),
                same_method_family: left.method_family == right.method_family,
                shared_implementation: left.implementation_sha256 == right.implementation_sha256,
                shared_environment: left.environment_sha256 == right.environment_sha256,
                shared_input: left.input_sha256 == right.input_sha256,
                shared_declared_roots,
            });
            if left.result_sha256 == right.result_sha256 {
                findings.push(FederationFinding::SharedResultArtifact {
                    left: left_id,
                    right: right_id,
                });
            }
        }
    }

    let mut seen_pairs = BTreeSet::new();
    let mut valid_pairs = BTreeMap::new();
    for comparison in comparisons {
        let pair = normalized_pair(&comparison.left_solver_id, &comparison.right_solver_id);
        if comparison.left_solver_id == comparison.right_solver_id {
            findings.push(FederationFinding::SelfComparison {
                solver_id: comparison.left_solver_id.clone(),
            });
            invalid = true;
            continue;
        }
        if !seen_pairs.insert(pair.clone()) {
            findings.push(FederationFinding::DuplicatePair {
                left: pair.0,
                right: pair.1,
            });
            invalid = true;
            continue;
        }
        let Some(left) = receipts.get(&comparison.left_solver_id) else {
            findings.push(FederationFinding::UnknownSolver {
                solver_id: comparison.left_solver_id.clone(),
            });
            invalid = true;
            continue;
        };
        let Some(right) = receipts.get(&comparison.right_solver_id) else {
            findings.push(FederationFinding::UnknownSolver {
                solver_id: comparison.right_solver_id.clone(),
            });
            invalid = true;
            continue;
        };
        if comparison.left_result_sha256 != left.result_sha256 {
            findings.push(FederationFinding::ResultSubstitution {
                solver_id: comparison.left_solver_id.clone(),
            });
            invalid = true;
            continue;
        }
        if comparison.right_result_sha256 != right.result_sha256 {
            findings.push(FederationFinding::ResultSubstitution {
                solver_id: comparison.right_solver_id.clone(),
            });
            invalid = true;
            continue;
        }
        if comparison.comparison_metric_sha256 != spec.comparison_metric_sha256 {
            findings.push(FederationFinding::MetricSubstitution {
                left: pair.0,
                right: pair.1,
            });
            invalid = true;
            continue;
        }
        if comparison.agreement_policy_sha256 != spec.agreement_policy_sha256 {
            findings.push(FederationFinding::AgreementPolicySubstitution {
                left: pair.0,
                right: pair.1,
            });
            invalid = true;
            continue;
        }
        if matches!(comparison.outcome, PairwiseAgreement::Agree | PairwiseAgreement::Disagree)
            && comparison.comparison_artifact_sha256.is_none()
        {
            findings.push(FederationFinding::ComparisonArtifactMissing {
                left: pair.0,
                right: pair.1,
            });
            invalid = true;
            continue;
        }
        if matches!(
            comparison.outcome,
            PairwiseAgreement::Incomparable | PairwiseAgreement::Invalid
        ) && comparison.justification_sha256.is_none()
        {
            findings.push(FederationFinding::ComparisonJustificationMissing {
                left: pair.0,
                right: pair.1,
            });
            invalid = true;
            continue;
        }
        valid_pairs.insert(pair, comparison.outcome);
    }

    for pair in &expected_pairs {
        if !valid_pairs.contains_key(pair) {
            findings.push(FederationFinding::MissingComparison {
                left: pair.0.clone(),
                right: pair.1.clone(),
            });
        }
    }

    let insufficient = receipts.len() < spec.minimum_solvers
        || methods.len() < spec.minimum_distinct_method_families;
    let missing_pair = expected_pairs
        .iter()
        .any(|pair| !valid_pairs.contains_key(pair));
    let comparison_invalid = valid_pairs
        .values()
        .any(|outcome| *outcome == PairwiseAgreement::Invalid);
    let incomparable = valid_pairs
        .values()
        .any(|outcome| *outcome == PairwiseAgreement::Incomparable);
    let disagreement = valid_pairs
        .values()
        .any(|outcome| *outcome == PairwiseAgreement::Disagree);

    let coverage = if invalid || comparison_invalid {
        FederationCoverage::Invalid
    } else if insufficient || missing_pair || incomparable {
        FederationCoverage::Incomplete
    } else {
        FederationCoverage::Complete
    };

    let agreement = if invalid || comparison_invalid {
        SolverAgreementState::Invalid
    } else if disagreement {
        // Disagreement remains first-class even if some other pair is missing.
        SolverAgreementState::DisagreementObserved
    } else if coverage != FederationCoverage::Complete {
        SolverAgreementState::Incomplete
    } else {
        SolverAgreementState::AgreementObserved
    };

    findings.sort();
    lineage.sort_by(|left, right| {
        (&left.left_solver_id, &left.right_solver_id)
            .cmp(&(&right.left_solver_id, &right.right_solver_id))
    });

    SolverFederationReport {
        federation_sha256: frozen.spec_sha256().clone(),
        subject_sha256: spec.subject_sha256.clone(),
        model_contract_sha256: spec.model_contract_sha256.clone(),
        solver_ids: receipts.keys().cloned().collect(),
        distinct_method_families: methods,
        coverage,
        agreement,
        lineage,
        findings,
        independence_established: false,
    }
}

fn normalized_pair(left: &ResearchId, right: &ResearchId) -> (ResearchId, ResearchId) {
    if left <= right {
        (left.clone(), right.clone())
    } else {
        (right.clone(), left.clone())
    }
}

fn spec_digest(spec: &SolverFederationSpec) -> Sha256Digest {
    let mut digest = FramedDigest::new(FEDERATION_DIGEST_DOMAIN);
    digest.text(SOLVER_FEDERATION_SCHEMA);
    digest.text(spec.federation_id.as_str());
    digest.text(spec.subject_sha256.as_str());
    digest.text(spec.model_contract_sha256.as_str());
    digest.text(spec.comparison_metric_sha256.as_str());
    digest.text(spec.agreement_policy_sha256.as_str());
    digest.text(&spec.minimum_solvers.to_string());
    digest.text(&spec.minimum_distinct_method_families.to_string());
    digest.finish()
}

struct FramedDigest {
    bytes: Vec<u8>,
}

impl FramedDigest {
    fn new(domain: &str) -> Self {
        let mut digest = Self { bytes: Vec::new() };
        digest.text(domain);
        digest
    }

    fn text(&mut self, value: &str) {
        self.bytes
            .extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn finish(self) -> Sha256Digest {
        Sha256Digest::of_bytes(&self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_science_research::{SharedRoot, SharedRootKind};

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn spec() -> FrozenSolverFederationSpec {
        SolverFederationSpec {
            schema_version: SOLVER_FEDERATION_SCHEMA.into(),
            federation_id: id("FED-1"),
            subject_sha256: sha("subject"),
            model_contract_sha256: sha("contract"),
            comparison_metric_sha256: sha("metric"),
            agreement_policy_sha256: sha("policy"),
            minimum_solvers: 2,
            minimum_distinct_method_families: 2,
        }
        .freeze()
        .unwrap()
    }

    fn receipt(name: &str, family: SolverMethodFamily) -> SolverReceipt {
        SolverReceipt {
            solver_id: id(name),
            subject_sha256: sha("subject"),
            model_contract_sha256: sha("contract"),
            method_family: family,
            implementation_sha256: sha(&format!("implementation-{name}")),
            environment_sha256: sha(&format!("environment-{name}")),
            input_sha256: sha("shared-input"),
            result_sha256: sha(&format!("result-{name}")),
            diagnostics_sha256: sha(&format!("diagnostics-{name}")),
            dependency_roots: BTreeSet::new(),
        }
    }

    fn comparison(left: &SolverReceipt, right: &SolverReceipt, outcome: PairwiseAgreement) -> SolverPairComparison {
        SolverPairComparison {
            left_solver_id: left.solver_id.clone(),
            right_solver_id: right.solver_id.clone(),
            left_result_sha256: left.result_sha256.clone(),
            right_result_sha256: right.result_sha256.clone(),
            comparison_metric_sha256: sha("metric"),
            agreement_policy_sha256: sha("policy"),
            outcome,
            comparison_artifact_sha256: matches!(outcome, PairwiseAgreement::Agree | PairwiseAgreement::Disagree)
                .then(|| sha("comparison-artifact")),
            justification_sha256: matches!(outcome, PairwiseAgreement::Incomparable | PairwiseAgreement::Invalid)
                .then(|| sha("comparison-justification")),
        }
    }

    #[test]
    fn complete_cross_method_agreement_does_not_establish_independence() {
        let a = receipt("A", SolverMethodFamily::FiniteElement);
        let b = receipt("B", SolverMethodFamily::Spectral);
        let report = evaluate_solver_federation(&spec(), &[a.clone(), b.clone()], &[comparison(&a, &b, PairwiseAgreement::Agree)]);
        assert_eq!(report.coverage, FederationCoverage::Complete);
        assert_eq!(report.agreement, SolverAgreementState::AgreementObserved);
        assert!(!report.independence_established);
        assert!(report.lineage[0].shared_input);
    }

    #[test]
    fn declared_shared_roots_survive_method_diversity() {
        let root = SharedRoot {
            kind: SharedRootKind::RawData,
            digest: sha("dataset"),
        };
        let mut a = receipt("A", SolverMethodFamily::FiniteDifference);
        let mut b = receipt("B", SolverMethodFamily::NeuralOperator);
        a.dependency_roots.insert(root.clone());
        b.dependency_roots.insert(root.clone());
        let report = evaluate_solver_federation(&spec(), &[a.clone(), b.clone()], &[comparison(&a, &b, PairwiseAgreement::Agree)]);
        assert!(report.lineage[0].shared_declared_roots.contains(&root));
        assert!(!report.independence_established);
    }

    #[test]
    fn disagreement_is_first_class_even_with_other_missing_pairs() {
        let mut frozen = spec();
        frozen.spec.minimum_solvers = 3;
        let a = receipt("A", SolverMethodFamily::FiniteElement);
        let b = receipt("B", SolverMethodFamily::Spectral);
        let c = receipt("C", SolverMethodFamily::MonteCarlo);
        let report = evaluate_solver_federation(
            &frozen,
            &[a.clone(), b.clone(), c],
            &[comparison(&a, &b, PairwiseAgreement::Disagree)],
        );
        assert_eq!(report.coverage, FederationCoverage::Incomplete);
        assert_eq!(report.agreement, SolverAgreementState::DisagreementObserved);
    }

    #[test]
    fn missing_pairwise_coverage_cannot_be_called_agreement() {
        let a = receipt("A", SolverMethodFamily::FiniteElement);
        let b = receipt("B", SolverMethodFamily::Spectral);
        let report = evaluate_solver_federation(&spec(), &[a, b], &[]);
        assert_eq!(report.coverage, FederationCoverage::Incomplete);
        assert_eq!(report.agreement, SolverAgreementState::Incomplete);
    }

    #[test]
    fn result_substitution_invalidates_report() {
        let a = receipt("A", SolverMethodFamily::FiniteElement);
        let b = receipt("B", SolverMethodFamily::Spectral);
        let mut pair = comparison(&a, &b, PairwiseAgreement::Agree);
        pair.left_result_sha256 = sha("substituted-result");
        let report = evaluate_solver_federation(&spec(), &[a, b], &[pair]);
        assert_eq!(report.coverage, FederationCoverage::Invalid);
        assert_eq!(report.agreement, SolverAgreementState::Invalid);
    }

    #[test]
    fn duplicate_pair_is_invalid() {
        let a = receipt("A", SolverMethodFamily::FiniteElement);
        let b = receipt("B", SolverMethodFamily::Spectral);
        let first = comparison(&a, &b, PairwiseAgreement::Agree);
        let second = comparison(&b, &a, PairwiseAgreement::Agree);
        let report = evaluate_solver_federation(&spec(), &[a, b], &[first, second]);
        assert_eq!(report.coverage, FederationCoverage::Invalid);
    }
}
