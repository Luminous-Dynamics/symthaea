// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing localization of cross-solver disagreement.
//!
//! This layer answers a narrow question: *where, under a frozen localization
//! plan, does an already-observed solver disagreement appear?* It does not infer
//! the cause of disagreement, select a correct solver, or promote scientific
//! authority. Every result is bound to the exact `SolverFederationEvidence`
//! envelope and to all disagreement pairs observed in that envelope.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_science_research::{ResearchId, Sha256Digest};

use crate::{
    FederationCoverage, PairwiseAgreement, SolverAgreementState, SolverFederationEvidence,
    SolverPairComparison,
};

pub const DISCREPANCY_LOCALIZATION_SCHEMA: &str =
    "symthaea.solver-discrepancy-localization.v1";
const LOCALIZATION_PLAN_DIGEST_DOMAIN: &str =
    "symthaea.solver-discrepancy-localization-plan.identity.v1";
const LOCALIZATION_REPORT_DIGEST_DOMAIN: &str =
    "symthaea.solver-discrepancy-localization-report.identity.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum DiscrepancyAxisKind {
    Time,
    Space,
    StateVariable,
    Parameter,
    InitialCondition,
    BoundaryCondition,
    Resolution,
    Precision,
    Regime,
    Observable,
    StochasticSeed,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalizationPartitionSpec {
    pub partition_id: ResearchId,
    /// Exact selector/range/mask semantics for this partition.
    pub selector_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalizationAxisSpec {
    pub axis_id: ResearchId,
    pub kind: DiscrepancyAxisKind,
    /// Exact implementation/schema that interprets every selector on this axis.
    pub selector_policy_sha256: Sha256Digest,
    pub partitions: Vec<LocalizationPartitionSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SolverPairTarget {
    pub left_solver_id: ResearchId,
    pub right_solver_id: ResearchId,
    /// Exact parent comparison artifact that established the global disagreement.
    pub parent_comparison_artifact_sha256: Sha256Digest,
}

impl SolverPairTarget {
    fn canonicalized(&self) -> Self {
        if self.left_solver_id <= self.right_solver_id {
            self.clone()
        } else {
            Self {
                left_solver_id: self.right_solver_id.clone(),
                right_solver_id: self.left_solver_id.clone(),
                parent_comparison_artifact_sha256: self.parent_comparison_artifact_sha256.clone(),
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscrepancyLocalizationPlan {
    pub schema_version: String,
    pub localization_id: ResearchId,
    pub federation_evidence_sha256: Sha256Digest,
    /// V1 is intentionally non-cherry-pickable: evaluation requires this set to
    /// equal all disagreement pairs observed in the bound federation evidence.
    pub targets: Vec<SolverPairTarget>,
    pub axes: Vec<LocalizationAxisSpec>,
    pub local_comparison_metric_sha256: Sha256Digest,
    pub local_agreement_policy_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscrepancyLocalizationPlanIssue {
    WrongSchemaVersion { found: String },
    MissingTargets,
    MissingAxes,
    SelfTarget { solver_id: ResearchId },
    DuplicateTarget { left: ResearchId, right: ResearchId },
    DuplicateAxis { axis_id: ResearchId },
    AxisWithoutPartitions { axis_id: ResearchId },
    DuplicatePartition {
        axis_id: ResearchId,
        partition_id: ResearchId,
    },
}

impl DiscrepancyLocalizationPlan {
    pub fn validate(&self) -> Vec<DiscrepancyLocalizationPlanIssue> {
        let mut issues = Vec::new();
        if self.schema_version != DISCREPANCY_LOCALIZATION_SCHEMA {
            issues.push(DiscrepancyLocalizationPlanIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if self.targets.is_empty() {
            issues.push(DiscrepancyLocalizationPlanIssue::MissingTargets);
        }
        if self.axes.is_empty() {
            issues.push(DiscrepancyLocalizationPlanIssue::MissingAxes);
        }

        let mut targets = BTreeSet::new();
        for target in &self.targets {
            let target = target.canonicalized();
            if target.left_solver_id == target.right_solver_id {
                issues.push(DiscrepancyLocalizationPlanIssue::SelfTarget {
                    solver_id: target.left_solver_id,
                });
                continue;
            }
            let pair = (target.left_solver_id.clone(), target.right_solver_id.clone());
            if !targets.insert(pair.clone()) {
                issues.push(DiscrepancyLocalizationPlanIssue::DuplicateTarget {
                    left: pair.0,
                    right: pair.1,
                });
            }
        }

        let mut axes = BTreeSet::new();
        for axis in &self.axes {
            if !axes.insert(axis.axis_id.clone()) {
                issues.push(DiscrepancyLocalizationPlanIssue::DuplicateAxis {
                    axis_id: axis.axis_id.clone(),
                });
            }
            if axis.partitions.is_empty() {
                issues.push(DiscrepancyLocalizationPlanIssue::AxisWithoutPartitions {
                    axis_id: axis.axis_id.clone(),
                });
            }
            let mut partitions = BTreeSet::new();
            for partition in &axis.partitions {
                if !partitions.insert(partition.partition_id.clone()) {
                    issues.push(DiscrepancyLocalizationPlanIssue::DuplicatePartition {
                        axis_id: axis.axis_id.clone(),
                        partition_id: partition.partition_id.clone(),
                    });
                }
            }
        }
        issues
    }

    pub fn freeze(
        self,
    ) -> Result<FrozenDiscrepancyLocalizationPlan, Vec<DiscrepancyLocalizationPlanIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        let plan_sha256 = plan_digest(&self);
        Ok(FrozenDiscrepancyLocalizationPlan {
            plan: self,
            plan_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenDiscrepancyLocalizationPlan {
    plan: DiscrepancyLocalizationPlan,
    plan_sha256: Sha256Digest,
}

impl FrozenDiscrepancyLocalizationPlan {
    pub fn plan(&self) -> &DiscrepancyLocalizationPlan {
        &self.plan
    }

    pub fn plan_sha256(&self) -> &Sha256Digest {
        &self.plan_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscrepancyCell {
    pub left_solver_id: ResearchId,
    pub right_solver_id: ResearchId,
    pub axis_id: ResearchId,
    pub partition_id: ResearchId,
    pub selector_sha256: Sha256Digest,
    pub left_local_result_sha256: Sha256Digest,
    pub right_local_result_sha256: Sha256Digest,
    pub comparison_metric_sha256: Sha256Digest,
    pub agreement_policy_sha256: Sha256Digest,
    pub outcome: PairwiseAgreement,
    /// Required for Agree/Disagree.
    pub comparison_artifact_sha256: Option<Sha256Digest>,
    /// Required for Incomparable/Invalid.
    pub justification_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum DiscrepancyLocalizationFinding {
    FederationEvidenceMismatch,
    ParentFederationInvalid,
    NoDisagreementObserved,
    TargetSetDoesNotCoverAllObservedDisagreements,
    TargetComparisonArtifactMismatch { left: ResearchId, right: ResearchId },
    DuplicateCell {
        left: ResearchId,
        right: ResearchId,
        axis_id: ResearchId,
        partition_id: ResearchId,
    },
    UnknownTarget { left: ResearchId, right: ResearchId },
    UnknownAxis { axis_id: ResearchId },
    UnknownPartition {
        axis_id: ResearchId,
        partition_id: ResearchId,
    },
    SelectorSubstitution {
        axis_id: ResearchId,
        partition_id: ResearchId,
    },
    MetricSubstitution {
        left: ResearchId,
        right: ResearchId,
    },
    AgreementPolicySubstitution {
        left: ResearchId,
        right: ResearchId,
    },
    ComparisonArtifactMissing {
        left: ResearchId,
        right: ResearchId,
        axis_id: ResearchId,
        partition_id: ResearchId,
    },
    ComparisonJustificationMissing {
        left: ResearchId,
        right: ResearchId,
        axis_id: ResearchId,
        partition_id: ResearchId,
    },
    MissingCell {
        left: ResearchId,
        right: ResearchId,
        axis_id: ResearchId,
        partition_id: ResearchId,
    },
    InvalidCellOutcome {
        left: ResearchId,
        right: ResearchId,
        axis_id: ResearchId,
        partition_id: ResearchId,
    },
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum LocalizationClosure {
    CompleteForObservedDisagreements,
    Incomplete,
    Invalid,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum DiscrepancyExtent {
    /// At least one planned cell agrees and at least one disagrees.
    LocalizedWithinPlan,
    /// Every planned cell still disagrees at the chosen localization resolution.
    DistributedAcrossPlan,
    /// The global disagreement was not reproduced in any planned local cell.
    NotLocalizedAtPlannedResolution,
    Incomplete,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct DisagreeingCellRef {
    pub left_solver_id: ResearchId,
    pub right_solver_id: ResearchId,
    pub axis_id: ResearchId,
    pub partition_id: ResearchId,
    pub comparison_artifact_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscrepancyLocalizationReport {
    pub report_sha256: Sha256Digest,
    pub plan_sha256: Sha256Digest,
    pub federation_evidence_sha256: Sha256Digest,
    pub parent_federation_coverage: FederationCoverage,
    pub closure: LocalizationClosure,
    pub extent: DiscrepancyExtent,
    /// Canonicalized exact local comparison evidence supplied to this report,
    /// including rejected/invalid cells. Invalid evidence must remain auditable.
    pub cells: Vec<DiscrepancyCell>,
    pub disagreeing_cells: BTreeSet<DisagreeingCellRef>,
    pub findings: Vec<DiscrepancyLocalizationFinding>,
    /// Localization is descriptive evidence, never a causal explanation.
    pub causal_explanation_established: bool,
    /// Pairwise disagreement does not select the physically correct solver.
    pub solver_correctness_established: bool,
}

pub fn evaluate_discrepancy_localization(
    evidence: &SolverFederationEvidence,
    frozen: &FrozenDiscrepancyLocalizationPlan,
    cells: &[DiscrepancyCell],
) -> DiscrepancyLocalizationReport {
    let plan = frozen.plan();
    let parent_report = evidence.validated_report().report();
    let mut findings = Vec::new();
    let mut invalid = false;

    if &plan.federation_evidence_sha256 != evidence.evidence_sha256() {
        findings.push(DiscrepancyLocalizationFinding::FederationEvidenceMismatch);
        invalid = true;
    }
    if parent_report.coverage == FederationCoverage::Invalid
        || parent_report.agreement == SolverAgreementState::Invalid
    {
        findings.push(DiscrepancyLocalizationFinding::ParentFederationInvalid);
        invalid = true;
    }

    let observed_disagreements = parent_disagreements(evidence.comparisons());
    if observed_disagreements.is_empty() {
        findings.push(DiscrepancyLocalizationFinding::NoDisagreementObserved);
        invalid = true;
    }

    let planned_targets: BTreeMap<_, _> = plan
        .targets
        .iter()
        .map(|target| {
            let target = target.canonicalized();
            (
                (target.left_solver_id.clone(), target.right_solver_id.clone()),
                target,
            )
        })
        .collect();

    let observed_pairs: BTreeSet<_> = observed_disagreements.keys().cloned().collect();
    let planned_pairs: BTreeSet<_> = planned_targets.keys().cloned().collect();
    if observed_pairs != planned_pairs {
        findings.push(
            DiscrepancyLocalizationFinding::TargetSetDoesNotCoverAllObservedDisagreements,
        );
        invalid = true;
    }

    for (pair, target) in &planned_targets {
        if let Some(parent) = observed_disagreements.get(pair) {
            if parent.comparison_artifact_sha256.as_ref()
                != Some(&target.parent_comparison_artifact_sha256)
            {
                findings.push(
                    DiscrepancyLocalizationFinding::TargetComparisonArtifactMismatch {
                        left: pair.0.clone(),
                        right: pair.1.clone(),
                    },
                );
                invalid = true;
            }
        }
    }

    let axes: BTreeMap<_, _> = plan
        .axes
        .iter()
        .map(|axis| (axis.axis_id.clone(), axis))
        .collect();
    let mut partitions = BTreeMap::new();
    for axis in &plan.axes {
        for partition in &axis.partitions {
            partitions.insert(
                (axis.axis_id.clone(), partition.partition_id.clone()),
                partition,
            );
        }
    }

    let mut expected_cells = BTreeSet::new();
    for pair in planned_targets.keys() {
        for axis in &plan.axes {
            for partition in &axis.partitions {
                expected_cells.insert((
                    pair.0.clone(),
                    pair.1.clone(),
                    axis.axis_id.clone(),
                    partition.partition_id.clone(),
                ));
            }
        }
    }

    let mut seen = BTreeSet::new();
    let mut valid_cells = BTreeMap::new();
    for cell in cells {
        let pair = canonical_pair(&cell.left_solver_id, &cell.right_solver_id);
        let key = (
            pair.0.clone(),
            pair.1.clone(),
            cell.axis_id.clone(),
            cell.partition_id.clone(),
        );
        if !seen.insert(key.clone()) {
            findings.push(DiscrepancyLocalizationFinding::DuplicateCell {
                left: key.0,
                right: key.1,
                axis_id: key.2,
                partition_id: key.3,
            });
            invalid = true;
            continue;
        }
        if !planned_targets.contains_key(&pair) {
            findings.push(DiscrepancyLocalizationFinding::UnknownTarget {
                left: pair.0,
                right: pair.1,
            });
            invalid = true;
            continue;
        }
        if !axes.contains_key(&cell.axis_id) {
            findings.push(DiscrepancyLocalizationFinding::UnknownAxis {
                axis_id: cell.axis_id.clone(),
            });
            invalid = true;
            continue;
        }
        let Some(partition) = partitions.get(&(cell.axis_id.clone(), cell.partition_id.clone()))
        else {
            findings.push(DiscrepancyLocalizationFinding::UnknownPartition {
                axis_id: cell.axis_id.clone(),
                partition_id: cell.partition_id.clone(),
            });
            invalid = true;
            continue;
        };
        if cell.selector_sha256 != partition.selector_sha256 {
            findings.push(DiscrepancyLocalizationFinding::SelectorSubstitution {
                axis_id: cell.axis_id.clone(),
                partition_id: cell.partition_id.clone(),
            });
            invalid = true;
            continue;
        }
        if cell.comparison_metric_sha256 != plan.local_comparison_metric_sha256 {
            findings.push(DiscrepancyLocalizationFinding::MetricSubstitution {
                left: pair.0,
                right: pair.1,
            });
            invalid = true;
            continue;
        }
        if cell.agreement_policy_sha256 != plan.local_agreement_policy_sha256 {
            findings.push(DiscrepancyLocalizationFinding::AgreementPolicySubstitution {
                left: pair.0,
                right: pair.1,
            });
            invalid = true;
            continue;
        }

        let executed = matches!(cell.outcome, PairwiseAgreement::Agree | PairwiseAgreement::Disagree);
        if executed && cell.comparison_artifact_sha256.is_none() {
            findings.push(DiscrepancyLocalizationFinding::ComparisonArtifactMissing {
                left: pair.0,
                right: pair.1,
                axis_id: cell.axis_id.clone(),
                partition_id: cell.partition_id.clone(),
            });
            invalid = true;
            continue;
        }
        if !executed && cell.justification_sha256.is_none() {
            findings.push(DiscrepancyLocalizationFinding::ComparisonJustificationMissing {
                left: pair.0,
                right: pair.1,
                axis_id: cell.axis_id.clone(),
                partition_id: cell.partition_id.clone(),
            });
            invalid = true;
            continue;
        }
        if cell.outcome == PairwiseAgreement::Invalid {
            findings.push(DiscrepancyLocalizationFinding::InvalidCellOutcome {
                left: pair.0,
                right: pair.1,
                axis_id: cell.axis_id.clone(),
                partition_id: cell.partition_id.clone(),
            });
            invalid = true;
            continue;
        }
        valid_cells.insert(key, cell);
    }

    for key in &expected_cells {
        if !valid_cells.contains_key(key) {
            findings.push(DiscrepancyLocalizationFinding::MissingCell {
                left: key.0.clone(),
                right: key.1.clone(),
                axis_id: key.2.clone(),
                partition_id: key.3.clone(),
            });
        }
    }

    let has_missing = expected_cells
        .iter()
        .any(|key| !valid_cells.contains_key(key));
    let has_incomparable = valid_cells
        .values()
        .any(|cell| cell.outcome == PairwiseAgreement::Incomparable);
    let closure = if invalid {
        LocalizationClosure::Invalid
    } else if has_missing || has_incomparable {
        LocalizationClosure::Incomplete
    } else {
        LocalizationClosure::CompleteForObservedDisagreements
    };

    let mut disagreeing_cells = BTreeSet::new();
    for (key, cell) in &valid_cells {
        if cell.outcome == PairwiseAgreement::Disagree {
            if let Some(artifact) = &cell.comparison_artifact_sha256 {
                disagreeing_cells.insert(DisagreeingCellRef {
                    left_solver_id: key.0.clone(),
                    right_solver_id: key.1.clone(),
                    axis_id: key.2.clone(),
                    partition_id: key.3.clone(),
                    comparison_artifact_sha256: artifact.clone(),
                });
            }
        }
    }

    let extent = if closure == LocalizationClosure::Invalid {
        DiscrepancyExtent::Invalid
    } else if closure == LocalizationClosure::Incomplete {
        DiscrepancyExtent::Incomplete
    } else if disagreeing_cells.is_empty() {
        DiscrepancyExtent::NotLocalizedAtPlannedResolution
    } else if disagreeing_cells.len() == expected_cells.len() {
        DiscrepancyExtent::DistributedAcrossPlan
    } else {
        DiscrepancyExtent::LocalizedWithinPlan
    };

    let mut retained_cells: Vec<_> = cells.iter().map(canonicalized_cell).collect();
    retained_cells.sort_by(compare_cells);
    findings.sort();
    let report_sha256 = report_digest(
        frozen.plan_sha256(),
        evidence.evidence_sha256(),
        parent_report.coverage,
        closure,
        extent,
        &retained_cells,
        &findings,
    );

    DiscrepancyLocalizationReport {
        report_sha256,
        plan_sha256: frozen.plan_sha256().clone(),
        federation_evidence_sha256: evidence.evidence_sha256().clone(),
        parent_federation_coverage: parent_report.coverage,
        closure,
        extent,
        cells: retained_cells,
        disagreeing_cells,
        findings,
        causal_explanation_established: false,
        solver_correctness_established: false,
    }
}

fn parent_disagreements(
    comparisons: &[SolverPairComparison],
) -> BTreeMap<(ResearchId, ResearchId), &SolverPairComparison> {
    comparisons
        .iter()
        .filter(|comparison| comparison.outcome == PairwiseAgreement::Disagree)
        .map(|comparison| {
            (
                canonical_pair(&comparison.left_solver_id, &comparison.right_solver_id),
                comparison,
            )
        })
        .collect()
}

fn canonical_pair(left: &ResearchId, right: &ResearchId) -> (ResearchId, ResearchId) {
    if left <= right {
        (left.clone(), right.clone())
    } else {
        (right.clone(), left.clone())
    }
}

fn canonicalized_cell(cell: &DiscrepancyCell) -> DiscrepancyCell {
    if cell.left_solver_id <= cell.right_solver_id {
        cell.clone()
    } else {
        DiscrepancyCell {
            left_solver_id: cell.right_solver_id.clone(),
            right_solver_id: cell.left_solver_id.clone(),
            axis_id: cell.axis_id.clone(),
            partition_id: cell.partition_id.clone(),
            selector_sha256: cell.selector_sha256.clone(),
            left_local_result_sha256: cell.right_local_result_sha256.clone(),
            right_local_result_sha256: cell.left_local_result_sha256.clone(),
            comparison_metric_sha256: cell.comparison_metric_sha256.clone(),
            agreement_policy_sha256: cell.agreement_policy_sha256.clone(),
            outcome: cell.outcome,
            comparison_artifact_sha256: cell.comparison_artifact_sha256.clone(),
            justification_sha256: cell.justification_sha256.clone(),
        }
    }
}

fn compare_cells(left: &DiscrepancyCell, right: &DiscrepancyCell) -> Ordering {
    left.left_solver_id
        .cmp(&right.left_solver_id)
        .then_with(|| left.right_solver_id.cmp(&right.right_solver_id))
        .then_with(|| left.axis_id.cmp(&right.axis_id))
        .then_with(|| left.partition_id.cmp(&right.partition_id))
        .then_with(|| left.selector_sha256.cmp(&right.selector_sha256))
        .then_with(|| left.left_local_result_sha256.cmp(&right.left_local_result_sha256))
        .then_with(|| left.right_local_result_sha256.cmp(&right.right_local_result_sha256))
        .then_with(|| left.comparison_metric_sha256.cmp(&right.comparison_metric_sha256))
        .then_with(|| left.agreement_policy_sha256.cmp(&right.agreement_policy_sha256))
        .then_with(|| left.outcome.cmp(&right.outcome))
        .then_with(|| left.comparison_artifact_sha256.cmp(&right.comparison_artifact_sha256))
        .then_with(|| left.justification_sha256.cmp(&right.justification_sha256))
}

fn plan_digest(plan: &DiscrepancyLocalizationPlan) -> Sha256Digest {
    let mut digest = FramedDigest::new(LOCALIZATION_PLAN_DIGEST_DOMAIN);
    digest.text(DISCREPANCY_LOCALIZATION_SCHEMA);
    digest.text(plan.localization_id.as_str());
    digest.text(plan.federation_evidence_sha256.as_str());
    digest.text(plan.local_comparison_metric_sha256.as_str());
    digest.text(plan.local_agreement_policy_sha256.as_str());

    let mut targets: Vec<_> = plan.targets.iter().map(SolverPairTarget::canonicalized).collect();
    targets.sort();
    for target in &targets {
        digest.text("target");
        digest.text(target.left_solver_id.as_str());
        digest.text(target.right_solver_id.as_str());
        digest.text(target.parent_comparison_artifact_sha256.as_str());
    }

    let mut axes = plan.axes.clone();
    axes.sort_by(|left, right| left.axis_id.cmp(&right.axis_id));
    for axis in &axes {
        digest.text("axis");
        digest.text(axis.axis_id.as_str());
        digest.text(axis_kind_tag(axis.kind));
        digest.text(axis.selector_policy_sha256.as_str());
        let mut partitions = axis.partitions.clone();
        partitions.sort_by(|left, right| left.partition_id.cmp(&right.partition_id));
        for partition in &partitions {
            digest.text("partition");
            digest.text(partition.partition_id.as_str());
            digest.text(partition.selector_sha256.as_str());
        }
    }
    digest.finish()
}

fn report_digest(
    plan_sha256: &Sha256Digest,
    evidence_sha256: &Sha256Digest,
    parent_coverage: FederationCoverage,
    closure: LocalizationClosure,
    extent: DiscrepancyExtent,
    cells: &[DiscrepancyCell],
    findings: &[DiscrepancyLocalizationFinding],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(LOCALIZATION_REPORT_DIGEST_DOMAIN);
    digest.text(plan_sha256.as_str());
    digest.text(evidence_sha256.as_str());
    digest.text(federation_coverage_tag(parent_coverage));
    digest.text(localization_closure_tag(closure));
    digest.text(discrepancy_extent_tag(extent));
    for cell in cells {
        digest_cell(&mut digest, cell);
    }
    for finding in findings {
        digest_finding(&mut digest, finding);
    }
    digest.text("causal-explanation-not-established");
    digest.text("solver-correctness-not-established");
    digest.finish()
}

fn digest_cell(digest: &mut FramedDigest, cell: &DiscrepancyCell) {
    digest.text("cell");
    digest.text(cell.left_solver_id.as_str());
    digest.text(cell.right_solver_id.as_str());
    digest.text(cell.axis_id.as_str());
    digest.text(cell.partition_id.as_str());
    digest.text(cell.selector_sha256.as_str());
    digest.text(cell.left_local_result_sha256.as_str());
    digest.text(cell.right_local_result_sha256.as_str());
    digest.text(cell.comparison_metric_sha256.as_str());
    digest.text(cell.agreement_policy_sha256.as_str());
    digest.text(pairwise_outcome_tag(cell.outcome));
    digest.optional_sha(cell.comparison_artifact_sha256.as_ref());
    digest.optional_sha(cell.justification_sha256.as_ref());
}

fn digest_finding(digest: &mut FramedDigest, finding: &DiscrepancyLocalizationFinding) {
    match finding {
        DiscrepancyLocalizationFinding::FederationEvidenceMismatch => {
            digest.text("finding:federation-evidence-mismatch");
        }
        DiscrepancyLocalizationFinding::ParentFederationInvalid => {
            digest.text("finding:parent-federation-invalid");
        }
        DiscrepancyLocalizationFinding::NoDisagreementObserved => {
            digest.text("finding:no-disagreement-observed");
        }
        DiscrepancyLocalizationFinding::TargetSetDoesNotCoverAllObservedDisagreements => {
            digest.text("finding:target-set-not-exhaustive");
        }
        DiscrepancyLocalizationFinding::TargetComparisonArtifactMismatch { left, right } => {
            digest_pair_finding(digest, "target-comparison-artifact-mismatch", left, right);
        }
        DiscrepancyLocalizationFinding::DuplicateCell {
            left,
            right,
            axis_id,
            partition_id,
        } => digest_cell_finding(digest, "duplicate-cell", left, right, axis_id, partition_id),
        DiscrepancyLocalizationFinding::UnknownTarget { left, right } => {
            digest_pair_finding(digest, "unknown-target", left, right);
        }
        DiscrepancyLocalizationFinding::UnknownAxis { axis_id } => {
            digest.text("finding:unknown-axis");
            digest.text(axis_id.as_str());
        }
        DiscrepancyLocalizationFinding::UnknownPartition {
            axis_id,
            partition_id,
        } => {
            digest.text("finding:unknown-partition");
            digest.text(axis_id.as_str());
            digest.text(partition_id.as_str());
        }
        DiscrepancyLocalizationFinding::SelectorSubstitution {
            axis_id,
            partition_id,
        } => {
            digest.text("finding:selector-substitution");
            digest.text(axis_id.as_str());
            digest.text(partition_id.as_str());
        }
        DiscrepancyLocalizationFinding::MetricSubstitution { left, right } => {
            digest_pair_finding(digest, "metric-substitution", left, right);
        }
        DiscrepancyLocalizationFinding::AgreementPolicySubstitution { left, right } => {
            digest_pair_finding(digest, "agreement-policy-substitution", left, right);
        }
        DiscrepancyLocalizationFinding::ComparisonArtifactMissing {
            left,
            right,
            axis_id,
            partition_id,
        } => digest_cell_finding(
            digest,
            "comparison-artifact-missing",
            left,
            right,
            axis_id,
            partition_id,
        ),
        DiscrepancyLocalizationFinding::ComparisonJustificationMissing {
            left,
            right,
            axis_id,
            partition_id,
        } => digest_cell_finding(
            digest,
            "comparison-justification-missing",
            left,
            right,
            axis_id,
            partition_id,
        ),
        DiscrepancyLocalizationFinding::MissingCell {
            left,
            right,
            axis_id,
            partition_id,
        } => digest_cell_finding(digest, "missing-cell", left, right, axis_id, partition_id),
        DiscrepancyLocalizationFinding::InvalidCellOutcome {
            left,
            right,
            axis_id,
            partition_id,
        } => digest_cell_finding(
            digest,
            "invalid-cell-outcome",
            left,
            right,
            axis_id,
            partition_id,
        ),
    }
}

fn digest_pair_finding(
    digest: &mut FramedDigest,
    tag: &str,
    left: &ResearchId,
    right: &ResearchId,
) {
    digest.text("finding");
    digest.text(tag);
    digest.text(left.as_str());
    digest.text(right.as_str());
}

fn digest_cell_finding(
    digest: &mut FramedDigest,
    tag: &str,
    left: &ResearchId,
    right: &ResearchId,
    axis_id: &ResearchId,
    partition_id: &ResearchId,
) {
    digest_pair_finding(digest, tag, left, right);
    digest.text(axis_id.as_str());
    digest.text(partition_id.as_str());
}

const fn axis_kind_tag(kind: DiscrepancyAxisKind) -> &'static str {
    match kind {
        DiscrepancyAxisKind::Time => "time",
        DiscrepancyAxisKind::Space => "space",
        DiscrepancyAxisKind::StateVariable => "state-variable",
        DiscrepancyAxisKind::Parameter => "parameter",
        DiscrepancyAxisKind::InitialCondition => "initial-condition",
        DiscrepancyAxisKind::BoundaryCondition => "boundary-condition",
        DiscrepancyAxisKind::Resolution => "resolution",
        DiscrepancyAxisKind::Precision => "precision",
        DiscrepancyAxisKind::Regime => "regime",
        DiscrepancyAxisKind::Observable => "observable",
        DiscrepancyAxisKind::StochasticSeed => "stochastic-seed",
        DiscrepancyAxisKind::Other => "other",
    }
}

const fn pairwise_outcome_tag(outcome: PairwiseAgreement) -> &'static str {
    match outcome {
        PairwiseAgreement::Agree => "agree",
        PairwiseAgreement::Disagree => "disagree",
        PairwiseAgreement::Incomparable => "incomparable",
        PairwiseAgreement::Invalid => "invalid",
    }
}

const fn federation_coverage_tag(coverage: FederationCoverage) -> &'static str {
    match coverage {
        FederationCoverage::Complete => "complete",
        FederationCoverage::Incomplete => "incomplete",
        FederationCoverage::Invalid => "invalid",
    }
}

const fn localization_closure_tag(closure: LocalizationClosure) -> &'static str {
    match closure {
        LocalizationClosure::CompleteForObservedDisagreements => "complete-observed-disagreements",
        LocalizationClosure::Incomplete => "incomplete",
        LocalizationClosure::Invalid => "invalid",
    }
}

const fn discrepancy_extent_tag(extent: DiscrepancyExtent) -> &'static str {
    match extent {
        DiscrepancyExtent::LocalizedWithinPlan => "localized-within-plan",
        DiscrepancyExtent::DistributedAcrossPlan => "distributed-across-plan",
        DiscrepancyExtent::NotLocalizedAtPlannedResolution => "not-localized-at-planned-resolution",
        DiscrepancyExtent::Incomplete => "incomplete",
        DiscrepancyExtent::Invalid => "invalid",
    }
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

    fn optional_sha(&mut self, value: Option<&Sha256Digest>) {
        match value {
            Some(value) => {
                self.text("some");
                self.text(value.as_str());
            }
            None => self.text("none"),
        }
    }

    fn finish(self) -> Sha256Digest {
        Sha256Digest::of_bytes(&self.bytes)
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

    fn target(left: &str, right: &str) -> SolverPairTarget {
        SolverPairTarget {
            left_solver_id: id(left),
            right_solver_id: id(right),
            parent_comparison_artifact_sha256: sha("parent-comparison"),
        }
    }

    fn axis() -> LocalizationAxisSpec {
        LocalizationAxisSpec {
            axis_id: id("TIME"),
            kind: DiscrepancyAxisKind::Time,
            selector_policy_sha256: sha("selector-policy"),
            partitions: vec![
                LocalizationPartitionSpec {
                    partition_id: id("EARLY"),
                    selector_sha256: sha("early"),
                },
                LocalizationPartitionSpec {
                    partition_id: id("LATE"),
                    selector_sha256: sha("late"),
                },
            ],
        }
    }

    fn plan(targets: Vec<SolverPairTarget>) -> DiscrepancyLocalizationPlan {
        DiscrepancyLocalizationPlan {
            schema_version: DISCREPANCY_LOCALIZATION_SCHEMA.into(),
            localization_id: id("LOCALIZE-1"),
            federation_evidence_sha256: sha("federation-evidence"),
            targets,
            axes: vec![axis()],
            local_comparison_metric_sha256: sha("local-metric"),
            local_agreement_policy_sha256: sha("local-policy"),
        }
    }

    #[test]
    fn plan_identity_is_order_independent() {
        let first = plan(vec![target("A", "B"), target("A", "C")]);
        let mut second = first.clone();
        second.targets.reverse();
        second.axes[0].partitions.reverse();
        assert_eq!(
            first.freeze().unwrap().plan_sha256(),
            second.freeze().unwrap().plan_sha256()
        );
    }

    #[test]
    fn duplicate_target_fails_freeze_even_when_reversed() {
        let draft = plan(vec![target("A", "B"), target("B", "A")]);
        assert!(draft.freeze().is_err());
    }

    #[test]
    fn empty_axis_partitions_fail_freeze() {
        let mut draft = plan(vec![target("A", "B")]);
        draft.axes[0].partitions.clear();
        assert!(draft.freeze().is_err());
    }
}
