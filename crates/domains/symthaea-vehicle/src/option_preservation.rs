// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Observation-constrained future-option measurement for SAM.
//!
//! This module consumes a frozen finite action set and exact successor-cell
//! feasibility outcomes. It does not simulate, predict, plan, calculate
//! reachability, qualify reference models, or authorize vehicle commands.
//!
//! The key distinction is between a clairvoyant continuation and a policy
//! continuation. A clairvoyant evaluator may choose a different next action
//! for every hidden state. A real policy may choose differently only after it
//! receives an observation that distinguishes those states. Within one
//! observation-equivalence class, the evaluator intersects feasible
//! continuation actions across all declared hidden-state/reference cells.
//!
//! Results are finite-envelope measurements. They do not establish global
//! viability, physical safety, or completeness of upstream action,
//! observation, scenario, or reference domains.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Admission outcome supplied by an upstream owner.
///
/// The evaluator checks that a receipt identity is present and preserves it.
/// It does not authenticate the receipt or independently qualify its subject.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EvidenceAdmission {
    /// The referenced upstream qualification process admitted this evidence.
    Qualified { receipt_digest: String },
    /// The upstream process rejected the evidence for this use.
    Unqualified { reason_id: String },
    /// The upstream process could not determine applicability or validity.
    Unknown { reason_id: String },
}

/// Qualification receipts required before an option-preservation result can be
/// interpreted over the declared finite envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationReceipts {
    /// #10338 action-domain coverage / decision-sufficiency receipt.
    pub action_domain: EvidenceAdmission,
    /// #10266 / #10318 reference-portfolio qualification receipt.
    pub reference_portfolio: EvidenceAdmission,
    /// #3387 / #3532 exact branch-execution and replay-manifest receipt.
    pub execution_manifest: EvidenceAdmission,
    /// Receipt for the observation-equivalence partition and delivery semantics.
    pub observation_partition: EvidenceAdmission,
}

impl QualificationReceipts {
    fn gate_status(&self) -> EvidenceGateStatus {
        let statuses = [
            &self.action_domain,
            &self.reference_portfolio,
            &self.execution_manifest,
            &self.observation_partition,
        ];

        if statuses
            .iter()
            .any(|status| matches!(status, EvidenceAdmission::Unqualified { .. }))
        {
            EvidenceGateStatus::NotQualified
        } else if statuses
            .iter()
            .any(|status| matches!(status, EvidenceAdmission::Unknown { .. }))
        {
            EvidenceGateStatus::Incomplete
        } else {
            EvidenceGateStatus::QualifiedForDeclaredEnvelope
        }
    }
}

/// Overall upstream-evidence gate. This is not a vehicle safety status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceGateStatus {
    /// All four required upstream receipt classes are present as qualified.
    QualifiedForDeclaredEnvelope,
    /// At least one required evidence domain is unqualified.
    NotQualified,
    /// At least one required evidence domain remains unknown.
    Incomplete,
}

/// Stable identity for one successor state evaluated after a frozen parent action.
///
/// The observation class identifies information actually available to the
/// controller at its next decision. Hidden-state and reference IDs must not be
/// used to split a class unless the delivered-observation partition receipt
/// establishes that the controller can distinguish those cases.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SuccessorCellKey {
    /// First action whose consequences produced this successor cell.
    pub parent_action_id: String,
    /// Exact controller-visible observation-equivalence class.
    pub observation_class_id: String,
    /// Hidden scene/state hypothesis within that observation class.
    pub hidden_state_id: String,
    /// Qualified reference-model identity applicable to this cell.
    pub reference_id: String,
}

/// Frozen manifest entry for one required successor cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedSuccessorCell {
    /// Exact cell identity.
    pub key: SuccessorCellKey,
    /// Receipt proving the observation delivered to this cell's controller view.
    ///
    /// All cells in the same parent-action / observation-class pair must bind
    /// the same receipt digest. Different hidden-state or reference IDs alone
    /// do not justify different observation classes.
    pub observation_equivalence_receipt_digest: String,
}

/// Compute the canonical BLAKE3 digest of the frozen expected-cell manifest.
///
/// Entries are sorted by the complete cell key before serialization, so the
/// digest is invariant to caller ordering but sensitive to any declared cell
/// or observation-equivalence receipt mutation.
pub fn compute_expected_cell_manifest_digest(
    expected_cells: &[ExpectedSuccessorCell],
) -> Result<String, String> {
    let mut canonical_cells = expected_cells.to_vec();
    canonical_cells.sort_by(|left, right| left.key.cmp(&right.key));
    let bytes = serde_json::to_vec(&canonical_cells)
        .map_err(|error| format!("serialize expected-cell manifest: {error}"))?;
    Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
}

/// Feasibility disposition for one continuation action in one successor cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "disposition", rename_all = "snake_case")]
pub enum FeasibilityDisposition {
    /// Feasible under the cell's declared constraints, with a retained witness.
    Feasible { witness_digest: String },
    /// Refuted by a retained, replayable constraint witness.
    Infeasible { witness_digest: String },
    /// Not determined; the reason must remain explicit.
    Unknown { reason_id: String },
}

/// Exact successor-cell evaluation.
///
/// Every action in the frozen continuation set must have one disposition.
/// Missing outcomes make the observation class incomplete. Explicit
/// Unknown outcomes widen the possible-option upper set but never enter the
/// robust lower set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuationCell {
    /// Exact manifest identity for this cell.
    pub key: SuccessorCellKey,
    /// Receipt for the actual controller-visible observation.
    pub observation_equivalence_receipt_digest: String,
    /// Exact Symtropy execution / branch receipt.
    pub execution_receipt_digest: String,
    /// Exact cell-evaluation receipt binding execution, reference and outcomes.
    pub cell_receipt_digest: String,
    /// Must equal the input's frozen continuation-action-set digest.
    pub continuation_action_set_digest: String,
    /// Must equal the input's named feasibility-evaluator profile digest.
    pub evaluator_profile_digest: String,
    /// One disposition for each action in the frozen continuation action set.
    pub dispositions: BTreeMap<String, FeasibilityDisposition>,
}

/// Complete offline measurement input.
///
/// Receipt digests are identifiers from upstream qualification/execution
/// systems, not proofs verified by this module. The expected successor-cell
/// manifest defines the finite envelope being measured. Passing this evaluator
/// does not prove that the manifest exhausts the real world or a continuous
/// action domain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionPreservationInput {
    /// Contract version. Currently sam.option-preservation.v0.1.
    pub contract_version: String,
    /// Frozen current-decision identity.
    pub decision_id: String,
    /// Exact scenario identity/digest.
    pub scenario_digest: String,
    /// Exact initial-state and observation-history digest.
    pub initial_context_digest: String,
    /// Frozen first-action-set digest.
    pub current_action_set_digest: String,
    /// Candidate first actions evaluated by this report.
    pub current_action_ids: Vec<String>,
    /// Frozen next-decision action-set digest.
    pub continuation_action_set_digest: String,
    /// Candidate continuation actions evaluated in each successor cell.
    pub continuation_action_ids: Vec<String>,
    /// Digest of the frozen expected successor-cell manifest.
    pub expected_cell_manifest_digest: String,
    /// Exact expected cell keys, frozen before outcomes are inspected.
    pub expected_cells: Vec<ExpectedSuccessorCell>,
    /// Actual exact cell outcomes returned by the execution/evaluator path.
    pub observed_cells: Vec<ContinuationCell>,
    /// Exact feasibility / hard-constraint evaluator profile.
    pub evaluator_profile_digest: String,
    /// Upstream action, reference, execution and observation receipts.
    pub qualifications: QualificationReceipts,
}

/// Status of one observation-equivalence class for one parent action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationClassStatus {
    /// At least one next action is feasible in every declared cell of the class.
    RobustOptionsExist,
    /// Every continuation action is refuted by at least one cell in the class.
    NoPossibleOptions,
    /// A robust action is not established, but one or more actions remain possible.
    Unresolved,
    /// Expected cells or action outcomes are missing.
    Incomplete,
    /// An upstream evidence gate rejected this measurement.
    NotQualified,
}

/// Status of one frozen first action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParentActionStatus {
    /// Every declared observation class has at least one common feasible action.
    RobustlyRecoverable,
    /// At least one declared observation class has no possible continuation.
    NoContinuationForAtLeastOneObservationClass,
    /// No class is proven dead, but a robust continuation is not established.
    Unresolved,
    /// Required cells, outcomes or upstream qualification are unknown/missing.
    Incomplete,
    /// An upstream evidence domain is explicitly unqualified.
    NotQualified,
}

/// Bounded option result for one controller-visible observation class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationClassReport {
    /// Controller-visible observation-equivalence class.
    pub observation_class_id: String,
    /// Receipt establishing that class's controller-visible observation.
    pub observation_equivalence_receipt_digest: String,
    /// Number of expected hidden-state/reference cells in this class.
    pub expected_cell_count: usize,
    /// Number of actual cells supplied.
    pub observed_cell_count: usize,
    /// Number of expected cells not supplied.
    pub missing_cell_count: usize,
    /// Number of missing action outcomes across supplied cells.
    pub missing_action_outcome_count: usize,
    /// Class disposition.
    pub status: ObservationClassStatus,
    /// Actions feasible in every cell. None means incomplete or unqualified;
    /// Some([]) means none are robust.
    pub robust_option_ids: Option<Vec<String>>,
    /// Actions not refuted by any cell (feasible or unknown in every cell).
    pub possible_option_ids: Option<Vec<String>>,
    /// Actions feasible in at least one cell. This is the clairvoyant union,
    /// not a policy-feasible set.
    pub clairvoyant_union_option_ids: Option<Vec<String>>,
    /// Lower-bound number of common feasible continuation actions.
    pub robust_option_width: Option<usize>,
    /// Upper-bound number of continuation actions not refuted.
    pub possible_option_width: Option<usize>,
    /// Clairvoyance gap = statewise feasible union size - robust intersection size.
    /// Only emitted when every action outcome in every cell is known.
    pub clairvoyance_gap: Option<usize>,
}

/// Future-option result for one first action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParentActionReport {
    /// First action whose successor cells were evaluated.
    pub parent_action_id: String,
    /// Overall finite-envelope result.
    pub status: ParentActionStatus,
    /// Result for each controller-visible observation class.
    pub observation_classes: Vec<ObservationClassReport>,
    /// Minimum robust lower-set width across complete observation classes.
    pub minimum_robust_option_width: Option<usize>,
    /// Minimum possible upper-set width across complete observation classes.
    pub minimum_possible_option_width: Option<usize>,
    /// Sum of per-class clairvoyance gaps, only when all class outcomes are known.
    pub total_clairvoyance_gap: Option<usize>,
}

/// Complete report over the frozen parent-action set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionPreservationReport {
    /// Contract version.
    pub contract_version: String,
    /// Frozen decision and scenario identities.
    pub decision_id: String,
    pub scenario_digest: String,
    /// The exact finite action-set and manifest identities being measured.
    pub current_action_set_digest: String,
    pub continuation_action_set_digest: String,
    pub expected_cell_manifest_digest: String,
    /// Upstream evidence gate.
    pub evidence_gate_status: EvidenceGateStatus,
    /// Reports sorted by parent action ID for order-independent comparison.
    pub parent_actions: Vec<ParentActionReport>,
}

/// Evaluate observation-constrained future options over a frozen finite envelope.
///
/// The result uses set intersections, not probabilities. It never assumes that
/// a controller can observe hidden-state or reference-model labels.
pub fn evaluate_option_preservation(
    input: &OptionPreservationInput,
) -> Result<OptionPreservationReport, Vec<String>> {
    let mut errors = Vec::new();
    validate_input(input, &mut errors);
    if !errors.is_empty() {
        return Err(errors);
    }

    let evidence_gate_status = input.qualifications.gate_status();
    let actual_by_key: BTreeMap<SuccessorCellKey, &ContinuationCell> = input
        .observed_cells
        .iter()
        .map(|cell| (cell.key.clone(), cell))
        .collect();

    let mut parent_action_ids = input.current_action_ids.clone();
    parent_action_ids.sort();
    let mut continuation_action_ids = input.continuation_action_ids.clone();
    continuation_action_ids.sort();

    let mut parent_actions = Vec::with_capacity(parent_action_ids.len());
    for parent_action_id in parent_action_ids {
        let mut expected_by_class: BTreeMap<String, Vec<&ExpectedSuccessorCell>> = BTreeMap::new();
        for expected in input
            .expected_cells
            .iter()
            .filter(|cell| cell.key.parent_action_id == parent_action_id)
        {
            expected_by_class
                .entry(expected.key.observation_class_id.clone())
                .or_default()
                .push(expected);
        }

        let mut observation_classes = Vec::with_capacity(expected_by_class.len());
        for (observation_class_id, expected_cells) in expected_by_class {
            let observation_receipt = expected_cells[0]
                .observation_equivalence_receipt_digest
                .clone();
            let mut observed_cells = Vec::new();
            let mut missing_cell_count = 0usize;
            let mut missing_action_outcome_count = 0usize;

            for expected in &expected_cells {
                match actual_by_key.get(&expected.key) {
                    Some(cell) => {
                        let missing = continuation_action_ids
                            .iter()
                            .filter(|action_id| !cell.dispositions.contains_key(*action_id))
                            .count();
                        missing_action_outcome_count += missing;
                        observed_cells.push(*cell);
                    }
                    None => missing_cell_count += 1,
                }
            }

            let class_complete = missing_cell_count == 0 && missing_action_outcome_count == 0;
            let class_status = if evidence_gate_status == EvidenceGateStatus::NotQualified {
                ObservationClassStatus::NotQualified
            } else if evidence_gate_status == EvidenceGateStatus::Incomplete || !class_complete {
                ObservationClassStatus::Incomplete
            } else {
                ObservationClassStatus::Unresolved
            };

            let mut report = ObservationClassReport {
                observation_class_id,
                observation_equivalence_receipt_digest: observation_receipt,
                expected_cell_count: expected_cells.len(),
                observed_cell_count: observed_cells.len(),
                missing_cell_count,
                missing_action_outcome_count,
                status: class_status,
                robust_option_ids: None,
                possible_option_ids: None,
                clairvoyant_union_option_ids: None,
                robust_option_width: None,
                possible_option_width: None,
                clairvoyance_gap: None,
            };

            if evidence_gate_status == EvidenceGateStatus::QualifiedForDeclaredEnvelope
                && class_complete
            {
                let mut robust_options = Vec::new();
                let mut possible_options = Vec::new();
                let mut clairvoyant_union = Vec::new();
                let mut all_outcomes_known = true;

                for action_id in &continuation_action_ids {
                    let dispositions: Vec<&FeasibilityDisposition> = observed_cells
                        .iter()
                        .filter_map(|cell| cell.dispositions.get(action_id))
                        .collect();

                    let all_feasible = dispositions.iter().all(|disposition| {
                        matches!(disposition, FeasibilityDisposition::Feasible { .. })
                    });
                    let any_infeasible = dispositions.iter().any(|disposition| {
                        matches!(disposition, FeasibilityDisposition::Infeasible { .. })
                    });
                    let any_feasible = dispositions.iter().any(|disposition| {
                        matches!(disposition, FeasibilityDisposition::Feasible { .. })
                    });
                    let any_unknown = dispositions.iter().any(|disposition| {
                        matches!(disposition, FeasibilityDisposition::Unknown { .. })
                    });

                    if all_feasible {
                        robust_options.push(action_id.clone());
                    }
                    if !any_infeasible {
                        possible_options.push(action_id.clone());
                    }
                    if any_feasible {
                        clairvoyant_union.push(action_id.clone());
                    }
                    if any_unknown {
                        all_outcomes_known = false;
                    }
                }

                report.robust_option_width = Some(robust_options.len());
                report.possible_option_width = Some(possible_options.len());
                report.robust_option_ids = Some(robust_options.clone());
                report.possible_option_ids = Some(possible_options);
                report.clairvoyant_union_option_ids = Some(clairvoyant_union.clone());

                if all_outcomes_known {
                    report.clairvoyance_gap =
                        Some(clairvoyant_union.len().saturating_sub(robust_options.len()));
                }

                report.status = if !robust_options.is_empty() {
                    ObservationClassStatus::RobustOptionsExist
                } else if report.possible_option_width == Some(0) {
                    ObservationClassStatus::NoPossibleOptions
                } else {
                    ObservationClassStatus::Unresolved
                };
            }

            observation_classes.push(report);
        }

        let parent_status = if evidence_gate_status == EvidenceGateStatus::NotQualified {
            ParentActionStatus::NotQualified
        } else if evidence_gate_status == EvidenceGateStatus::Incomplete
            || observation_classes.is_empty()
            || observation_classes
                .iter()
                .any(|class| class.status == ObservationClassStatus::Incomplete)
        {
            ParentActionStatus::Incomplete
        } else if observation_classes
            .iter()
            .any(|class| class.status == ObservationClassStatus::NoPossibleOptions)
        {
            ParentActionStatus::NoContinuationForAtLeastOneObservationClass
        } else if observation_classes
            .iter()
            .all(|class| class.status == ObservationClassStatus::RobustOptionsExist)
        {
            ParentActionStatus::RobustlyRecoverable
        } else {
            ParentActionStatus::Unresolved
        };

        let all_classes_have_bounds = !observation_classes.is_empty()
            && observation_classes.iter().all(|class| {
                class.robust_option_width.is_some() && class.possible_option_width.is_some()
            });
        let minimum_robust_option_width = all_classes_have_bounds.then(|| {
            observation_classes
                .iter()
                .filter_map(|class| class.robust_option_width)
                .min()
                .unwrap_or(0)
        });
        let minimum_possible_option_width = all_classes_have_bounds.then(|| {
            observation_classes
                .iter()
                .filter_map(|class| class.possible_option_width)
                .min()
                .unwrap_or(0)
        });
        let total_clairvoyance_gap = (!observation_classes.is_empty()
            && observation_classes
                .iter()
                .all(|class| class.clairvoyance_gap.is_some()))
        .then(|| {
            observation_classes
                .iter()
                .filter_map(|class| class.clairvoyance_gap)
                .sum()
        });

        parent_actions.push(ParentActionReport {
            parent_action_id,
            status: parent_status,
            observation_classes,
            minimum_robust_option_width,
            minimum_possible_option_width,
            total_clairvoyance_gap,
        });
    }

    Ok(OptionPreservationReport {
        contract_version: input.contract_version.clone(),
        decision_id: input.decision_id.clone(),
        scenario_digest: input.scenario_digest.clone(),
        current_action_set_digest: input.current_action_set_digest.clone(),
        continuation_action_set_digest: input.continuation_action_set_digest.clone(),
        expected_cell_manifest_digest: input.expected_cell_manifest_digest.clone(),
        evidence_gate_status,
        parent_actions,
    })
}

fn validate_input(input: &OptionPreservationInput, errors: &mut Vec<String>) {
    require_non_empty(errors, "contract_version", &input.contract_version);
    require_non_empty(errors, "decision_id", &input.decision_id);
    require_non_empty(errors, "scenario_digest", &input.scenario_digest);
    require_non_empty(
        errors,
        "initial_context_digest",
        &input.initial_context_digest,
    );
    require_non_empty(
        errors,
        "current_action_set_digest",
        &input.current_action_set_digest,
    );
    require_non_empty(
        errors,
        "continuation_action_set_digest",
        &input.continuation_action_set_digest,
    );
    require_non_empty(
        errors,
        "expected_cell_manifest_digest",
        &input.expected_cell_manifest_digest,
    );
    require_non_empty(
        errors,
        "evaluator_profile_digest",
        &input.evaluator_profile_digest,
    );

    if input.current_action_ids.is_empty() {
        errors.push("current_action_ids must not be empty".into());
    }
    if input.continuation_action_ids.is_empty() {
        errors.push("continuation_action_ids must not be empty".into());
    }
    if input.expected_cells.is_empty() {
        errors.push("expected_cells must not be empty".into());
    }

    let current_ids: BTreeSet<&str> = input
        .current_action_ids
        .iter()
        .map(String::as_str)
        .collect();
    if current_ids.len() != input.current_action_ids.len() {
        errors.push("current_action_ids contains duplicates".into());
    }
    for (index, action_id) in input.current_action_ids.iter().enumerate() {
        require_non_empty(errors, &format!("current_action_ids[{index}]"), action_id);
    }

    let continuation_ids: BTreeSet<&str> = input
        .continuation_action_ids
        .iter()
        .map(String::as_str)
        .collect();
    if continuation_ids.len() != input.continuation_action_ids.len() {
        errors.push("continuation_action_ids contains duplicates".into());
    }
    for (index, action_id) in input.continuation_action_ids.iter().enumerate() {
        require_non_empty(
            errors,
            &format!("continuation_action_ids[{index}]"),
            action_id,
        );
    }

    validate_admission(
        errors,
        "qualifications.action_domain",
        &input.qualifications.action_domain,
    );
    validate_admission(
        errors,
        "qualifications.reference_portfolio",
        &input.qualifications.reference_portfolio,
    );
    validate_admission(
        errors,
        "qualifications.execution_manifest",
        &input.qualifications.execution_manifest,
    );
    validate_admission(
        errors,
        "qualifications.observation_partition",
        &input.qualifications.observation_partition,
    );

    let computed_manifest_digest =
        match compute_expected_cell_manifest_digest(&input.expected_cells) {
            Ok(digest) => Some(digest),
            Err(error) => {
                errors.push(error);
                None
            }
        };
    if computed_manifest_digest.as_deref() != Some(input.expected_cell_manifest_digest.as_str()) {
        errors.push("expected_cell_manifest_digest does not match the canonical expected_cells manifest".into());
    }

    let mut expected_by_key = BTreeSet::new();
    let mut observation_receipts: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut observation_classes_by_receipt: BTreeMap<(String, String), String> = BTreeMap::new();
    for (index, expected) in input.expected_cells.iter().enumerate() {
        validate_cell_key(
            errors,
            &format!("expected_cells[{index}].key"),
            &expected.key,
        );
        require_non_empty(
            errors,
            &format!("expected_cells[{index}].observation_equivalence_receipt_digest"),
            &expected.observation_equivalence_receipt_digest,
        );

        if !current_ids.contains(expected.key.parent_action_id.as_str()) {
            errors.push(format!(
                "expected_cells[{index}] references undeclared parent action {}",
                expected.key.parent_action_id
            ));
        }
        if !expected_by_key.insert(expected.key.clone()) {
            errors.push(format!("expected_cells[{index}] duplicates a cell key"));
        }

        let class_key = (
            expected.key.parent_action_id.clone(),
            expected.key.observation_class_id.clone(),
        );
        if let Some(existing) = observation_receipts.get(&class_key) {
            if existing != &expected.observation_equivalence_receipt_digest {
                errors.push(format!(
                    "observation class {} for parent action {} has multiple observation receipts",
                    expected.key.observation_class_id, expected.key.parent_action_id
                ));
            }
        } else {
            observation_receipts.insert(
                class_key,
                expected.observation_equivalence_receipt_digest.clone(),
            );
        }

        let receipt_key = (
            expected.key.parent_action_id.clone(),
            expected.observation_equivalence_receipt_digest.clone(),
        );
        if let Some(existing_class) = observation_classes_by_receipt.get(&receipt_key) {
            if existing_class != &expected.key.observation_class_id {
                errors.push(format!(
                    "one observation receipt is assigned to multiple classes for parent action {}",
                    expected.key.parent_action_id
                ));
            }
        } else {
            observation_classes_by_receipt
                .insert(receipt_key, expected.key.observation_class_id.clone());
        }
    }

    let mut observed_by_key = BTreeSet::new();
    let mut cell_receipts = BTreeSet::new();
    let mut execution_receipts = BTreeSet::new();
    for (index, cell) in input.observed_cells.iter().enumerate() {
        validate_cell_key(errors, &format!("observed_cells[{index}].key"), &cell.key);
        require_non_empty(
            errors,
            &format!("observed_cells[{index}].observation_equivalence_receipt_digest"),
            &cell.observation_equivalence_receipt_digest,
        );
        require_non_empty(
            errors,
            &format!("observed_cells[{index}].execution_receipt_digest"),
            &cell.execution_receipt_digest,
        );
        require_non_empty(
            errors,
            &format!("observed_cells[{index}].cell_receipt_digest"),
            &cell.cell_receipt_digest,
        );
        require_non_empty(
            errors,
            &format!("observed_cells[{index}].continuation_action_set_digest"),
            &cell.continuation_action_set_digest,
        );
        require_non_empty(
            errors,
            &format!("observed_cells[{index}].evaluator_profile_digest"),
            &cell.evaluator_profile_digest,
        );

        if !expected_by_key.contains(&cell.key) {
            errors.push(format!(
                "observed_cells[{index}] is not present in the frozen expected-cell manifest"
            ));
        }
        if !observed_by_key.insert(cell.key.clone()) {
            errors.push(format!("observed_cells[{index}] duplicates a cell key"));
        }
        if !cell_receipts.insert(cell.cell_receipt_digest.clone()) {
            errors.push(format!(
                "observed_cells[{index}] duplicates cell receipt {}",
                cell.cell_receipt_digest
            ));
        }
        if !execution_receipts.insert(cell.execution_receipt_digest.clone()) {
            errors.push(format!(
                "observed_cells[{index}] duplicates execution receipt {}",
                cell.execution_receipt_digest
            ));
        }

        if let Some(expected_receipt) = observation_receipts.get(&(
            cell.key.parent_action_id.clone(),
            cell.key.observation_class_id.clone(),
        )) {
            if expected_receipt != &cell.observation_equivalence_receipt_digest {
                errors.push(format!(
                    "observed_cells[{index}] observation receipt does not match its expected class"
                ));
            }
        }
        if cell.continuation_action_set_digest != input.continuation_action_set_digest {
            errors.push(format!(
                "observed_cells[{index}] continuation action-set digest mismatch"
            ));
        }
        if cell.evaluator_profile_digest != input.evaluator_profile_digest {
            errors.push(format!(
                "observed_cells[{index}] evaluator profile digest mismatch"
            ));
        }

        for action_id in cell.dispositions.keys() {
            if !continuation_ids.contains(action_id.as_str()) {
                errors.push(format!(
                    "observed_cells[{index}] contains undeclared continuation action {action_id}"
                ));
            }
        }
        for (action_id, disposition) in &cell.dispositions {
            match disposition {
                FeasibilityDisposition::Feasible { witness_digest }
                | FeasibilityDisposition::Infeasible { witness_digest } => require_non_empty(
                    errors,
                    &format!("observed_cells[{index}].dispositions[{action_id}].witness_digest"),
                    witness_digest,
                ),
                FeasibilityDisposition::Unknown { reason_id } => require_non_empty(
                    errors,
                    &format!("observed_cells[{index}].dispositions[{action_id}].reason_id"),
                    reason_id,
                ),
            }
        }
    }
}

fn validate_cell_key(errors: &mut Vec<String>, label: &str, key: &SuccessorCellKey) {
    require_non_empty(
        errors,
        &format!("{label}.parent_action_id"),
        &key.parent_action_id,
    );
    require_non_empty(
        errors,
        &format!("{label}.observation_class_id"),
        &key.observation_class_id,
    );
    require_non_empty(
        errors,
        &format!("{label}.hidden_state_id"),
        &key.hidden_state_id,
    );
    require_non_empty(errors, &format!("{label}.reference_id"), &key.reference_id);
}

fn validate_admission(errors: &mut Vec<String>, label: &str, admission: &EvidenceAdmission) {
    match admission {
        EvidenceAdmission::Qualified { receipt_digest } => {
            require_non_empty(errors, &format!("{label}.receipt_digest"), receipt_digest);
        }
        EvidenceAdmission::Unqualified { reason_id } | EvidenceAdmission::Unknown { reason_id } => {
            require_non_empty(errors, &format!("{label}.reason_id"), reason_id);
        }
    }
}

fn require_non_empty(errors: &mut Vec<String>, label: &str, value: &str) {
    if value.trim().is_empty() {
        errors.push(format!("{label} must not be empty"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qualified() -> EvidenceAdmission {
        EvidenceAdmission::Qualified {
            receipt_digest: "receipt-qualified-v1".into(),
        }
    }

    fn qualifications() -> QualificationReceipts {
        QualificationReceipts {
            action_domain: qualified(),
            reference_portfolio: qualified(),
            execution_manifest: qualified(),
            observation_partition: qualified(),
        }
    }

    fn disposition(action_id: &str, kind: &str) -> (String, FeasibilityDisposition) {
        let result = match kind {
            "feasible" => FeasibilityDisposition::Feasible {
                witness_digest: format!("witness-{action_id}"),
            },
            "infeasible" => FeasibilityDisposition::Infeasible {
                witness_digest: format!("witness-{action_id}"),
            },
            "unknown" => FeasibilityDisposition::Unknown {
                reason_id: format!("unknown-{action_id}"),
            },
            _ => panic!("unsupported test disposition"),
        };
        (action_id.into(), result)
    }

    fn cell(
        parent_action_id: &str,
        observation_class_id: &str,
        hidden_state_id: &str,
        reference_id: &str,
        observation_receipt: &str,
        outcomes: &[(&str, &str)],
        cell_receipt: &str,
    ) -> ContinuationCell {
        ContinuationCell {
            key: SuccessorCellKey {
                parent_action_id: parent_action_id.into(),
                observation_class_id: observation_class_id.into(),
                hidden_state_id: hidden_state_id.into(),
                reference_id: reference_id.into(),
            },
            observation_equivalence_receipt_digest: observation_receipt.into(),
            execution_receipt_digest: format!("execution-{hidden_state_id}-{reference_id}"),
            cell_receipt_digest: cell_receipt.into(),
            continuation_action_set_digest: "continuation-set-v1".into(),
            evaluator_profile_digest: "evaluator-v1".into(),
            dispositions: outcomes
                .iter()
                .map(|(action_id, kind)| disposition(action_id, kind))
                .collect(),
        }
    }

    fn expected_from(cell: &ContinuationCell) -> ExpectedSuccessorCell {
        ExpectedSuccessorCell {
            key: cell.key.clone(),
            observation_equivalence_receipt_digest: cell
                .observation_equivalence_receipt_digest
                .clone(),
        }
    }

    fn input(cells: Vec<ContinuationCell>) -> OptionPreservationInput {
        let expected_cells = cells.iter().map(expected_from).collect::<Vec<_>>();
        let expected_cell_manifest_digest =
            compute_expected_cell_manifest_digest(&expected_cells).expect("manifest digest");
        OptionPreservationInput {
            contract_version: "sam.option-preservation.v0.1".into(),
            decision_id: "decision-01".into(),
            scenario_digest: "scenario-digest-01".into(),
            initial_context_digest: "initial-context-01".into(),
            current_action_set_digest: "current-set-v1".into(),
            current_action_ids: vec!["continue".into()],
            continuation_action_set_digest: "continuation-set-v1".into(),
            continuation_action_ids: vec!["brake".into(), "coast".into()],
            expected_cell_manifest_digest,
            expected_cells,
            observed_cells: cells,
            evaluator_profile_digest: "evaluator-v1".into(),
            qualifications: qualifications(),
        }
    }

    #[test]
    fn persistent_clairvoyance_trap_fixture_reproduces_observation_constrained_dead_end() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../docs/autonomy/fixtures/SAM_OPTION_PRESERVATION_CLAIRVOYANCE_TRAP_V0_1.json"
        ))
        .expect("canonical clairvoyance fixture must be valid JSON");
        let mut input: OptionPreservationInput =
            serde_json::from_value(fixture["input"].clone())
                .expect("fixture input must deserialize");

        // The manifest digest is deliberately derived from the frozen expected
        // cells by the loader, not accepted from a fixture-supplied string.
        input.expected_cell_manifest_digest =
            compute_expected_cell_manifest_digest(&input.expected_cells)
                .expect("fixture expected-cell manifest must hash");

        let report =
            evaluate_option_preservation(&input).expect("canonical fixture must evaluate");
        let expected_parent_id = fixture["expected"]["parent_action_id"]
            .as_str()
            .expect("fixture parent action ID");
        let expected_class_id = fixture["expected"]["observation_class_id"]
            .as_str()
            .expect("fixture observation class ID");
        let parent = report
            .parent_actions
            .iter()
            .find(|action| action.parent_action_id == expected_parent_id)
            .expect("expected parent action report");
        let class = parent
            .observation_classes
            .iter()
            .find(|class| class.observation_class_id == expected_class_id)
            .expect("expected observation class report");

        assert_eq!(
            parent.status,
            ParentActionStatus::NoContinuationForAtLeastOneObservationClass
        );
        assert_eq!(
            class.status,
            ObservationClassStatus::NoPossibleOptions
        );
        assert_eq!(class.robust_option_ids, Some(vec![]));
        assert_eq!(class.possible_option_ids, Some(vec![]));
        assert_eq!(
            class.clairvoyant_union_option_ids,
            Some(vec!["brake".into(), "coast".into()])
        );
        assert_eq!(class.robust_option_width, Some(0));
        assert_eq!(class.possible_option_width, Some(0));
        assert_eq!(class.clairvoyance_gap, Some(2));
    }

    #[test]
    fn common_continuation_is_robustly_recoverable() {
        let cells = vec![
            cell(
                "continue",
                "obs-a",
                "hidden-a",
                "reference-a",
                "observation-receipt-a",
                &[("brake", "feasible"), ("coast", "infeasible")],
                "cell-receipt-a",
            ),
            cell(
                "continue",
                "obs-a",
                "hidden-b",
                "reference-a",
                "observation-receipt-a",
                &[("brake", "feasible"), ("coast", "feasible")],
                "cell-receipt-b",
            ),
        ];
        let report = evaluate_option_preservation(&input(cells)).expect("valid input");
        let action = &report.parent_actions[0];
        assert_eq!(action.status, ParentActionStatus::RobustlyRecoverable);
        assert_eq!(action.minimum_robust_option_width, Some(1));
        assert_eq!(action.minimum_possible_option_width, Some(1));
        assert_eq!(
            action.observation_classes[0].robust_option_ids,
            Some(vec!["brake".into()])
        );
    }

    #[test]
    fn statewise_options_that_disagree_are_not_a_policy_continuation() {
        let cells = vec![
            cell(
                "continue",
                "same-observation",
                "hidden-a",
                "reference-a",
                "same-observation-receipt",
                &[("brake", "feasible"), ("coast", "infeasible")],
                "cell-receipt-a",
            ),
            cell(
                "continue",
                "same-observation",
                "hidden-b",
                "reference-a",
                "same-observation-receipt",
                &[("brake", "infeasible"), ("coast", "feasible")],
                "cell-receipt-b",
            ),
        ];
        let report = evaluate_option_preservation(&input(cells)).expect("valid input");
        let action = &report.parent_actions[0];
        assert_eq!(
            action.status,
            ParentActionStatus::NoContinuationForAtLeastOneObservationClass
        );
        let class = &action.observation_classes[0];
        assert_eq!(class.robust_option_width, Some(0));
        assert_eq!(class.possible_option_width, Some(0));
        assert_eq!(class.clairvoyance_gap, Some(2));
    }

    #[test]
    fn distinct_receipt_backed_observations_may_use_distinct_continuations() {
        let cells = vec![
            cell(
                "continue",
                "observation-a",
                "hidden-a",
                "reference-a",
                "observation-receipt-a",
                &[("brake", "feasible"), ("coast", "infeasible")],
                "cell-receipt-a",
            ),
            cell(
                "continue",
                "observation-b",
                "hidden-b",
                "reference-a",
                "observation-receipt-b",
                &[("brake", "infeasible"), ("coast", "feasible")],
                "cell-receipt-b",
            ),
        ];
        let report = evaluate_option_preservation(&input(cells)).expect("valid input");
        let action = &report.parent_actions[0];
        assert_eq!(action.status, ParentActionStatus::RobustlyRecoverable);
        assert_eq!(action.observation_classes.len(), 2);
        assert_eq!(action.minimum_robust_option_width, Some(1));
    }

    #[test]
    fn unknown_outcomes_widen_upper_bound_without_entering_lower_bound() {
        let cells = vec![
            cell(
                "continue",
                "obs-a",
                "hidden-a",
                "reference-a",
                "observation-receipt-a",
                &[("brake", "feasible"), ("coast", "unknown")],
                "cell-receipt-a",
            ),
            cell(
                "continue",
                "obs-a",
                "hidden-b",
                "reference-a",
                "observation-receipt-a",
                &[("brake", "unknown"), ("coast", "feasible")],
                "cell-receipt-b",
            ),
        ];
        let report = evaluate_option_preservation(&input(cells)).expect("valid input");
        let class = &report.parent_actions[0].observation_classes[0];
        assert_eq!(class.status, ObservationClassStatus::Unresolved);
        assert_eq!(class.robust_option_ids, Some(vec![]));
        assert_eq!(
            class.possible_option_ids,
            Some(vec!["brake".into(), "coast".into()])
        );
        assert_eq!(class.clairvoyance_gap, None);
    }

    #[test]
    fn missing_expected_successor_cell_is_incomplete_not_dead_end() {
        let first = cell(
            "continue",
            "obs-a",
            "hidden-a",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "infeasible")],
            "cell-receipt-a",
        );
        let second = cell(
            "continue",
            "obs-a",
            "hidden-b",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "feasible")],
            "cell-receipt-b",
        );
        let mut measurement = input(vec![first.clone()]);
        measurement.expected_cells.push(expected_from(&second));
        measurement.expected_cell_manifest_digest =
            compute_expected_cell_manifest_digest(&measurement.expected_cells)
                .expect("updated manifest digest");

        let report = evaluate_option_preservation(&measurement).expect("valid input");
        let action = &report.parent_actions[0];
        assert_eq!(action.status, ParentActionStatus::Incomplete);
        assert_eq!(action.observation_classes[0].missing_cell_count, 1);
        assert_eq!(action.observation_classes[0].robust_option_ids, None);
    }

    #[test]
    fn missing_action_outcome_is_incomplete_not_infeasible() {
        let mut first = cell(
            "continue",
            "obs-a",
            "hidden-a",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "infeasible")],
            "cell-receipt-a",
        );
        first.dispositions.remove("coast");
        let report =
            evaluate_option_preservation(&input(vec![first])).expect("missing outcome is retained");
        let class = &report.parent_actions[0].observation_classes[0];
        assert_eq!(class.status, ObservationClassStatus::Incomplete);
        assert_eq!(class.missing_action_outcome_count, 1);
        assert_eq!(class.possible_option_ids, None);
    }

    #[test]
    fn continuation_action_set_mismatch_is_rejected() {
        let mut first = cell(
            "continue",
            "obs-a",
            "hidden-a",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "infeasible")],
            "cell-receipt-a",
        );
        first.continuation_action_set_digest = "different-action-set".into();
        assert!(evaluate_option_preservation(&input(vec![first])).is_err());
    }

    #[test]
    fn expected_cell_manifest_digest_is_order_invariant_and_content_sensitive() {
        let first = cell(
            "continue",
            "obs-a",
            "hidden-a",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "infeasible")],
            "cell-receipt-a",
        );
        let second = cell(
            "continue",
            "obs-a",
            "hidden-b",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "feasible")],
            "cell-receipt-b",
        );
        let mut expected = vec![expected_from(&first), expected_from(&second)];
        let digest = compute_expected_cell_manifest_digest(&expected).expect("manifest digest");
        expected.reverse();
        assert_eq!(
            compute_expected_cell_manifest_digest(&expected).expect("permuted digest"),
            digest
        );
        expected[0].observation_equivalence_receipt_digest = "mutated-receipt".into();
        assert_ne!(
            compute_expected_cell_manifest_digest(&expected).expect("mutated digest"),
            digest
        );
    }

    #[test]
    fn mismatched_expected_cell_manifest_digest_is_rejected() {
        let first = cell(
            "continue",
            "obs-a",
            "hidden-a",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "infeasible")],
            "cell-receipt-a",
        );
        let mut measurement = input(vec![first]);
        measurement.expected_cell_manifest_digest = "blake3:forged-manifest-digest".into();
        assert!(evaluate_option_preservation(&measurement).is_err());
    }

    #[test]
    fn duplicate_execution_receipts_are_rejected() {
        let first = cell(
            "continue",
            "obs-a",
            "hidden-a",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "infeasible")],
            "cell-receipt-a",
        );
        let mut second = cell(
            "continue",
            "obs-a",
            "hidden-b",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "feasible")],
            "cell-receipt-b",
        );
        second.execution_receipt_digest = first.execution_receipt_digest.clone();
        assert!(evaluate_option_preservation(&input(vec![first, second])).is_err());
    }

    #[test]
    fn duplicate_cell_receipts_are_rejected() {
        let first = cell(
            "continue",
            "obs-a",
            "hidden-a",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "infeasible")],
            "duplicate-cell-receipt",
        );
        let second = cell(
            "continue",
            "obs-a",
            "hidden-b",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "feasible")],
            "duplicate-cell-receipt",
        );
        assert!(evaluate_option_preservation(&input(vec![first, second])).is_err());
    }

    #[test]
    fn unqualified_reference_portfolio_blocks_option_claim() {
        let first = cell(
            "continue",
            "obs-a",
            "hidden-a",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "infeasible")],
            "cell-receipt-a",
        );
        let mut measurement = input(vec![first]);
        measurement.qualifications.reference_portfolio = EvidenceAdmission::Unqualified {
            reason_id: "reference-context-not-qualified".into(),
        };

        let report = evaluate_option_preservation(&measurement).expect("valid but unqualified");
        assert_eq!(
            report.evidence_gate_status,
            EvidenceGateStatus::NotQualified
        );
        assert_eq!(
            report.parent_actions[0].status,
            ParentActionStatus::NotQualified
        );
        assert_eq!(
            report.parent_actions[0].observation_classes[0].robust_option_ids,
            None
        );
    }

    #[test]
    fn unknown_upstream_receipt_makes_report_incomplete() {
        let first = cell(
            "continue",
            "obs-a",
            "hidden-a",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "infeasible")],
            "cell-receipt-a",
        );
        let mut measurement = input(vec![first]);
        measurement.qualifications.observation_partition = EvidenceAdmission::Unknown {
            reason_id: "partition-receipt-pending".into(),
        };

        let report = evaluate_option_preservation(&measurement).expect("valid incomplete evidence");
        assert_eq!(report.evidence_gate_status, EvidenceGateStatus::Incomplete);
        assert_eq!(
            report.parent_actions[0].status,
            ParentActionStatus::Incomplete
        );
    }

    #[test]
    fn hidden_state_or_reference_id_cannot_split_one_observation_class() {
        let cells = vec![
            cell(
                "continue",
                "same-observation",
                "hidden-a",
                "reference-a",
                "same-observation-receipt",
                &[("brake", "feasible"), ("coast", "infeasible")],
                "cell-receipt-a",
            ),
            cell(
                "continue",
                "same-observation",
                "hidden-b",
                "reference-b",
                "same-observation-receipt",
                &[("brake", "infeasible"), ("coast", "feasible")],
                "cell-receipt-b",
            ),
        ];
        let report = evaluate_option_preservation(&input(cells)).expect("valid input");
        assert_eq!(
            report.parent_actions[0].status,
            ParentActionStatus::NoContinuationForAtLeastOneObservationClass
        );
    }

    #[test]
    fn observation_class_cannot_carry_multiple_receipts() {
        let cells = vec![
            cell(
                "continue",
                "same-observation",
                "hidden-a",
                "reference-a",
                "observation-receipt-a",
                &[("brake", "feasible"), ("coast", "infeasible")],
                "cell-receipt-a",
            ),
            cell(
                "continue",
                "same-observation",
                "hidden-b",
                "reference-a",
                "different-observation-receipt",
                &[("brake", "feasible"), ("coast", "feasible")],
                "cell-receipt-b",
            ),
        ];
        assert!(evaluate_option_preservation(&input(cells)).is_err());
    }

    #[test]
    fn one_observation_receipt_cannot_be_split_into_multiple_classes() {
        let cells = vec![
            cell(
                "continue",
                "observation-a",
                "hidden-a",
                "reference-a",
                "same-observation-receipt",
                &[("brake", "feasible"), ("coast", "infeasible")],
                "cell-receipt-a",
            ),
            cell(
                "continue",
                "observation-b",
                "hidden-b",
                "reference-a",
                "same-observation-receipt",
                &[("brake", "infeasible"), ("coast", "feasible")],
                "cell-receipt-b",
            ),
        ];
        assert!(evaluate_option_preservation(&input(cells)).is_err());
    }

    #[test]
    fn report_is_invariant_to_cell_and_action_input_order() {
        let cells = vec![
            cell(
                "continue",
                "obs-a",
                "hidden-a",
                "reference-a",
                "observation-receipt-a",
                &[("brake", "feasible"), ("coast", "infeasible")],
                "cell-receipt-a",
            ),
            cell(
                "continue",
                "obs-a",
                "hidden-b",
                "reference-a",
                "observation-receipt-a",
                &[("brake", "feasible"), ("coast", "feasible")],
                "cell-receipt-b",
            ),
        ];
        let original = input(cells);
        let expected = evaluate_option_preservation(&original).expect("valid input");

        let mut permuted = original.clone();
        permuted.current_action_ids.reverse();
        permuted.continuation_action_ids.reverse();
        permuted.expected_cells.reverse();
        permuted.observed_cells.reverse();

        let actual = evaluate_option_preservation(&permuted).expect("permuted input");
        assert_eq!(actual, expected);
    }

    #[test]
    fn disposition_for_undeclared_action_is_rejected() {
        let mut first = cell(
            "continue",
            "obs-a",
            "hidden-a",
            "reference-a",
            "observation-receipt-a",
            &[("brake", "feasible"), ("coast", "infeasible")],
            "cell-receipt-a",
        );
        first.dispositions.insert(
            "unlisted-action".into(),
            FeasibilityDisposition::Feasible {
                witness_digest: "witness-extra".into(),
            },
        );
        assert!(evaluate_option_preservation(&input(vec![first])).is_err());
    }
}
