//! Auditable multi-stage candidate screening funnel for fusion design search.
//!
//! This module does not screen designs or determine physical feasibility. It
//! verifies that the frozen candidate universe remains visible across
//! analytical, low-fidelity and high-fidelity stages, including every skipped,
//! failed and non-converged evaluation. Reference labels are retrospective
//! audit inputs and must never be used by the screening policy itself.

use std::collections::{BTreeMap, BTreeSet};

const STAGES: [ScreeningStage; 3] = [
    ScreeningStage::Analytical,
    ScreeningStage::LowFidelity,
    ScreeningStage::HighFidelity,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScreeningStage {
    Analytical,
    LowFidelity,
    HighFidelity,
}

impl ScreeningStage {
    fn prior_stages(self) -> &'static [ScreeningStage] {
        match self {
            Self::Analytical => &[],
            Self::LowFidelity => &[Self::Analytical],
            Self::HighFidelity => &[Self::Analytical, Self::LowFidelity],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreeningDisposition {
    /// This stage was deliberately not run. It still occupies a denominator row.
    NotEvaluated,
    Passed,
    Rejected,
    Inconclusive,
    Failed,
    NonConverged,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionBasis {
    /// Initial candidate admitted to the frozen universe.
    InitialGeneration,
    /// Candidate advanced because every preceding stage passed.
    PipelineAdvance,
    /// Candidate was deliberately sampled to audit a previous screen.
    StratifiedAudit,
    /// A documented human exception allowed an otherwise-pruned candidate onward.
    HumanOverride,
    /// Candidate was deliberately not scheduled at this stage.
    NotScheduled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreeningRecord {
    pub candidate_id: String,
    pub stage: ScreeningStage,
    pub disposition: ScreeningDisposition,
    pub reason_code: Option<String>,
    pub selection_basis: SelectionBasis,
    /// Required for StratifiedAudit and HumanOverride.
    pub selection_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceDisposition {
    Feasible,
    Infeasible,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceLabel {
    pub candidate_id: String,
    pub disposition: ReferenceDisposition,
}

/// Retrospective reference labels for benchmark audit only.
///
/// These labels are not part of the screening input or selection policy. On an
/// exhaustive reduced design space they permit exact recall calculation. On a
/// partial audit sample, the module reports counts but deliberately withholds
/// an exact screening-recall value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceAuditInput {
    pub labels: Vec<ReferenceLabel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunnelClaimCeiling {
    SyntheticCandidateFunnelAuditOnly,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScreeningStageSummary {
    pub stage: ScreeningStage,
    pub total_candidates: usize,
    pub evaluated: usize,
    pub not_evaluated: usize,
    pub passed: usize,
    pub rejected: usize,
    pub inconclusive: usize,
    pub failed: usize,
    pub non_converged: usize,
    pub unsupported: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceAuditSummary {
    pub labels_supplied: usize,
    pub universe_size: usize,
    pub exhaustive_labels: bool,
    pub reference_feasible: usize,
    pub reference_infeasible: usize,
    pub reference_indeterminate: usize,
    pub feasible_not_evaluated_at_high_fidelity: usize,
    /// Exact only when every candidate in the frozen universe has a reference label.
    pub screening_recall: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScreeningFunnelReport {
    pub candidate_count: usize,
    pub stages: Vec<ScreeningStageSummary>,
    pub high_fidelity_calls: usize,
    pub reference_audit: ReferenceAuditSummary,
    pub claim_ceiling: FunnelClaimCeiling,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunnelValidationError {
    EmptyCandidateUniverse,
    EmptyCandidateId,
    DuplicateCandidateId(String),
    UnknownCandidateId(String),
    DuplicateStageRecord {
        candidate_id: String,
        stage: ScreeningStage,
    },
    MissingStageRecord {
        candidate_id: String,
        stage: ScreeningStage,
    },
    MissingReasonCode {
        candidate_id: String,
        stage: ScreeningStage,
    },
    InvalidSelectionBasis {
        candidate_id: String,
        stage: ScreeningStage,
    },
    MissingSelectionReason {
        candidate_id: String,
        stage: ScreeningStage,
    },
    DuplicateReferenceLabel(String),
    UnknownReferenceLabel(String),
}

/// Validate and summarize the full candidate universe and its stage records.
///
/// Every candidate must have exactly one record at every stage. A candidate
/// skipped after an earlier stage must have an explicit NotEvaluated record
/// and a reason. Passing through the ordinary pipeline requires all earlier
/// stages to have passed. Rejected or unresolved candidates may proceed only
/// through a documented stratified audit or human override.
pub fn audit_screening_funnel(
    candidate_ids: &[String],
    records: &[ScreeningRecord],
    reference: &ReferenceAuditInput,
) -> Result<ScreeningFunnelReport, Vec<FunnelValidationError>> {
    let mut errors = Vec::new();
    if candidate_ids.is_empty() {
        return Err(vec![FunnelValidationError::EmptyCandidateUniverse]);
    }

    let mut universe = BTreeSet::new();
    for id in candidate_ids {
        if id.trim().is_empty() {
            errors.push(FunnelValidationError::EmptyCandidateId);
        } else if !universe.insert(id.clone()) {
            errors.push(FunnelValidationError::DuplicateCandidateId(id.clone()));
        }
    }

    let mut record_map: BTreeMap<(String, ScreeningStage), ScreeningRecord> = BTreeMap::new();
    for record in records {
        if record.candidate_id.trim().is_empty() {
            errors.push(FunnelValidationError::EmptyCandidateId);
            continue;
        }
        if !universe.contains(&record.candidate_id) {
            errors.push(FunnelValidationError::UnknownCandidateId(
                record.candidate_id.clone(),
            ));
            continue;
        }
        let key = (record.candidate_id.clone(), record.stage);
        if record_map.insert(key, record.clone()).is_some() {
            errors.push(FunnelValidationError::DuplicateStageRecord {
                candidate_id: record.candidate_id.clone(),
                stage: record.stage,
            });
        }
    }

    for id in &universe {
        for stage in STAGES {
            if !record_map.contains_key(&(id.clone(), stage)) {
                errors.push(FunnelValidationError::MissingStageRecord {
                    candidate_id: id.clone(),
                    stage,
                });
            }
        }
    }

    for id in &universe {
        for stage in STAGES {
            let Some(record) = record_map.get(&(id.clone(), stage)) else {
                continue;
            };

            if record.disposition != ScreeningDisposition::Passed
                && record
                    .reason_code
                    .as_deref()
                    .is_none_or(|reason| reason.trim().is_empty())
            {
                errors.push(FunnelValidationError::MissingReasonCode {
                    candidate_id: id.clone(),
                    stage,
                });
            }

            if !valid_selection_basis(record, &record_map) {
                errors.push(FunnelValidationError::InvalidSelectionBasis {
                    candidate_id: id.clone(),
                    stage,
                });
            }

            if matches!(
                record.selection_basis,
                SelectionBasis::StratifiedAudit | SelectionBasis::HumanOverride
            ) && record
                .selection_reason
                .as_deref()
                .is_none_or(|reason| reason.trim().is_empty())
            {
                errors.push(FunnelValidationError::MissingSelectionReason {
                    candidate_id: id.clone(),
                    stage,
                });
            }
        }
    }

    let mut reference_labels = BTreeMap::new();
    for label in &reference.labels {
        if !universe.contains(&label.candidate_id) {
            errors.push(FunnelValidationError::UnknownReferenceLabel(
                label.candidate_id.clone(),
            ));
            continue;
        }
        if reference_labels
            .insert(label.candidate_id.clone(), label.disposition)
            .is_some()
        {
            errors.push(FunnelValidationError::DuplicateReferenceLabel(
                label.candidate_id.clone(),
            ));
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let mut stages = Vec::with_capacity(STAGES.len());
    for stage in STAGES {
        let mut summary = ScreeningStageSummary {
            stage,
            total_candidates: universe.len(),
            evaluated: 0,
            not_evaluated: 0,
            passed: 0,
            rejected: 0,
            inconclusive: 0,
            failed: 0,
            non_converged: 0,
            unsupported: 0,
        };

        for id in &universe {
            let record = record_map
                .get(&(id.clone(), stage))
                .expect("complete stage matrix was validated");
            match record.disposition {
                ScreeningDisposition::NotEvaluated => summary.not_evaluated += 1,
                ScreeningDisposition::Passed => {
                    summary.evaluated += 1;
                    summary.passed += 1;
                }
                ScreeningDisposition::Rejected => {
                    summary.evaluated += 1;
                    summary.rejected += 1;
                }
                ScreeningDisposition::Inconclusive => {
                    summary.evaluated += 1;
                    summary.inconclusive += 1;
                }
                ScreeningDisposition::Failed => {
                    summary.evaluated += 1;
                    summary.failed += 1;
                }
                ScreeningDisposition::NonConverged => {
                    summary.evaluated += 1;
                    summary.non_converged += 1;
                }
                ScreeningDisposition::Unsupported => {
                    summary.evaluated += 1;
                    summary.unsupported += 1;
                }
            }
        }
        stages.push(summary);
    }

    let high_fidelity_calls = stages
        .iter()
        .find(|summary| summary.stage == ScreeningStage::HighFidelity)
        .map_or(0, |summary| summary.evaluated);

    let mut reference_feasible = 0usize;
    let mut reference_infeasible = 0usize;
    let mut reference_indeterminate = 0usize;
    let mut feasible_not_evaluated_at_high_fidelity = 0usize;

    for (id, disposition) in &reference_labels {
        match disposition {
            ReferenceDisposition::Feasible => {
                reference_feasible += 1;
                let high_fidelity = record_map
                    .get(&(id.clone(), ScreeningStage::HighFidelity))
                    .expect("complete stage matrix was validated");
                if high_fidelity.disposition == ScreeningDisposition::NotEvaluated {
                    feasible_not_evaluated_at_high_fidelity += 1;
                }
            }
            ReferenceDisposition::Infeasible => reference_infeasible += 1,
            ReferenceDisposition::Indeterminate => reference_indeterminate += 1,
        }
    }

    let exhaustive_labels = reference_labels.len() == universe.len();
    let screening_recall = if exhaustive_labels && reference_feasible > 0 {
        Some(
            (reference_feasible - feasible_not_evaluated_at_high_fidelity) as f64
                / reference_feasible as f64,
        )
    } else {
        None
    };

    Ok(ScreeningFunnelReport {
        candidate_count: universe.len(),
        stages,
        high_fidelity_calls,
        reference_audit: ReferenceAuditSummary {
            labels_supplied: reference_labels.len(),
            universe_size: universe.len(),
            exhaustive_labels,
            reference_feasible,
            reference_infeasible,
            reference_indeterminate,
            feasible_not_evaluated_at_high_fidelity,
            screening_recall,
        },
        claim_ceiling: FunnelClaimCeiling::SyntheticCandidateFunnelAuditOnly,
    })
}

fn valid_selection_basis(
    record: &ScreeningRecord,
    records: &BTreeMap<(String, ScreeningStage), ScreeningRecord>,
) -> bool {
    if record.stage == ScreeningStage::Analytical {
        return record.selection_basis == SelectionBasis::InitialGeneration
            && record.disposition != ScreeningDisposition::NotEvaluated
            && record.selection_reason.is_none();
    }

    if record.disposition == ScreeningDisposition::NotEvaluated {
        return record.selection_basis == SelectionBasis::NotScheduled;
    }

    match record.selection_basis {
        SelectionBasis::InitialGeneration | SelectionBasis::NotScheduled => false,
        SelectionBasis::PipelineAdvance => record.stage.prior_stages().iter().all(|stage| {
            records
                .get(&(record.candidate_id.clone(), *stage))
                .is_some_and(|prior| prior.disposition == ScreeningDisposition::Passed)
        }),
        SelectionBasis::StratifiedAudit | SelectionBasis::HumanOverride => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_records(candidate_ids: &[String]) -> Vec<ScreeningRecord> {
        let mut records = Vec::new();
        for id in candidate_ids {
            records.push(ScreeningRecord {
                candidate_id: id.clone(),
                stage: ScreeningStage::Analytical,
                disposition: ScreeningDisposition::Passed,
                reason_code: None,
                selection_basis: SelectionBasis::InitialGeneration,
                selection_reason: None,
            });
            records.push(ScreeningRecord {
                candidate_id: id.clone(),
                stage: ScreeningStage::LowFidelity,
                disposition: ScreeningDisposition::Passed,
                reason_code: None,
                selection_basis: SelectionBasis::PipelineAdvance,
                selection_reason: None,
            });
            records.push(ScreeningRecord {
                candidate_id: id.clone(),
                stage: ScreeningStage::HighFidelity,
                disposition: ScreeningDisposition::Passed,
                reason_code: None,
                selection_basis: SelectionBasis::PipelineAdvance,
                selection_reason: None,
            });
        }
        records
    }

    fn feasible_labels(candidate_ids: &[String]) -> ReferenceAuditInput {
        ReferenceAuditInput {
            labels: candidate_ids
                .iter()
                .map(|id| ReferenceLabel {
                    candidate_id: id.clone(),
                    disposition: ReferenceDisposition::Feasible,
                })
                .collect(),
        }
    }

    fn two_candidates() -> Vec<String> {
        vec!["candidate-a".to_string(), "candidate-b".to_string()]
    }

    #[test]
    fn complete_funnel_reports_full_denominator_and_exact_recall() {
        let ids = two_candidates();
        let report =
            audit_screening_funnel(&ids, &complete_records(&ids), &feasible_labels(&ids)).unwrap();

        assert_eq!(report.candidate_count, 2);
        assert_eq!(report.high_fidelity_calls, 2);
        assert!(report.reference_audit.exhaustive_labels);
        assert_eq!(report.reference_audit.screening_recall, Some(1.0));
    }

    #[test]
    fn skipped_candidate_remains_in_denominator_and_reduces_recall() {
        let ids = two_candidates();
        let mut records = complete_records(&ids);
        let high = records
            .iter_mut()
            .find(|record| {
                record.candidate_id == "candidate-b" && record.stage == ScreeningStage::HighFidelity
            })
            .unwrap();
        high.disposition = ScreeningDisposition::NotEvaluated;
        high.reason_code = Some("budget-deferred".to_string());
        high.selection_basis = SelectionBasis::NotScheduled;

        let report = audit_screening_funnel(&ids, &records, &feasible_labels(&ids)).unwrap();
        assert_eq!(report.candidate_count, 2);
        assert_eq!(report.high_fidelity_calls, 1);
        assert_eq!(
            report
                .reference_audit
                .feasible_not_evaluated_at_high_fidelity,
            1
        );
        assert_eq!(report.reference_audit.screening_recall, Some(0.5));
    }

    #[test]
    fn missing_stage_record_is_rejected_not_silently_dropped() {
        let ids = two_candidates();
        let mut records = complete_records(&ids);
        records.retain(|record| {
            !(record.candidate_id == "candidate-b" && record.stage == ScreeningStage::LowFidelity)
        });
        let errors = audit_screening_funnel(&ids, &records, &feasible_labels(&ids)).unwrap_err();
        assert!(errors.contains(&FunnelValidationError::MissingStageRecord {
            candidate_id: "candidate-b".to_string(),
            stage: ScreeningStage::LowFidelity,
        }));
    }

    #[test]
    fn duplicate_stage_record_is_rejected() {
        let ids = two_candidates();
        let mut records = complete_records(&ids);
        records.push(records[0].clone());
        let errors = audit_screening_funnel(&ids, &records, &feasible_labels(&ids)).unwrap_err();
        assert!(
            errors.contains(&FunnelValidationError::DuplicateStageRecord {
                candidate_id: "candidate-a".to_string(),
                stage: ScreeningStage::Analytical,
            })
        );
    }

    #[test]
    fn unknown_candidate_record_is_rejected() {
        let ids = two_candidates();
        let mut records = complete_records(&ids);
        records.push(ScreeningRecord {
            candidate_id: "not-in-frozen-universe".to_string(),
            stage: ScreeningStage::Analytical,
            disposition: ScreeningDisposition::Passed,
            reason_code: None,
            selection_basis: SelectionBasis::InitialGeneration,
            selection_reason: None,
        });
        let errors = audit_screening_funnel(&ids, &records, &feasible_labels(&ids)).unwrap_err();
        assert!(errors.contains(&FunnelValidationError::UnknownCandidateId(
            "not-in-frozen-universe".to_string()
        )));
    }

    #[test]
    fn ordinary_pipeline_cannot_advance_after_a_rejected_stage() {
        let ids = two_candidates();
        let mut records = complete_records(&ids);
        let analytical = records
            .iter_mut()
            .find(|record| {
                record.candidate_id == "candidate-b" && record.stage == ScreeningStage::Analytical
            })
            .unwrap();
        analytical.disposition = ScreeningDisposition::Rejected;
        analytical.reason_code = Some("analytical-limit".to_string());

        let errors = audit_screening_funnel(&ids, &records, &feasible_labels(&ids)).unwrap_err();
        assert!(
            errors.contains(&FunnelValidationError::InvalidSelectionBasis {
                candidate_id: "candidate-b".to_string(),
                stage: ScreeningStage::LowFidelity,
            })
        );
    }

    #[test]
    fn stratified_audit_can_recheck_a_rejected_candidate_with_reason() {
        let ids = two_candidates();
        let mut records = complete_records(&ids);
        let analytical = records
            .iter_mut()
            .find(|record| {
                record.candidate_id == "candidate-b" && record.stage == ScreeningStage::Analytical
            })
            .unwrap();
        analytical.disposition = ScreeningDisposition::Rejected;
        analytical.reason_code = Some("analytical-limit".to_string());

        let low = records
            .iter_mut()
            .find(|record| {
                record.candidate_id == "candidate-b" && record.stage == ScreeningStage::LowFidelity
            })
            .unwrap();
        low.selection_basis = SelectionBasis::StratifiedAudit;
        low.selection_reason = Some("sample rejected stratum for false-negative audit".to_string());

        let high = records
            .iter_mut()
            .find(|record| {
                record.candidate_id == "candidate-b" && record.stage == ScreeningStage::HighFidelity
            })
            .unwrap();
        high.selection_basis = SelectionBasis::StratifiedAudit;
        high.selection_reason =
            Some("complete reference audit of sampled rejected candidate".to_string());

        let report = audit_screening_funnel(&ids, &records, &feasible_labels(&ids)).unwrap();
        assert_eq!(report.candidate_count, 2);
        assert_eq!(report.high_fidelity_calls, 2);
    }

    #[test]
    fn stratified_audit_without_selection_reason_is_rejected() {
        let ids = two_candidates();
        let mut records = complete_records(&ids);
        let low = records
            .iter_mut()
            .find(|record| {
                record.candidate_id == "candidate-b" && record.stage == ScreeningStage::LowFidelity
            })
            .unwrap();
        low.selection_basis = SelectionBasis::StratifiedAudit;
        let errors = audit_screening_funnel(&ids, &records, &feasible_labels(&ids)).unwrap_err();
        assert!(
            errors.contains(&FunnelValidationError::MissingSelectionReason {
                candidate_id: "candidate-b".to_string(),
                stage: ScreeningStage::LowFidelity,
            })
        );
    }

    #[test]
    fn failure_and_nonconvergence_remain_separate_counts() {
        let ids = two_candidates();
        let mut records = complete_records(&ids);
        let first = records
            .iter_mut()
            .find(|record| {
                record.candidate_id == "candidate-a" && record.stage == ScreeningStage::HighFidelity
            })
            .unwrap();
        first.disposition = ScreeningDisposition::NonConverged;
        first.reason_code = Some("solver-did-not-converge".to_string());

        let second = records
            .iter_mut()
            .find(|record| {
                record.candidate_id == "candidate-b" && record.stage == ScreeningStage::HighFidelity
            })
            .unwrap();
        second.disposition = ScreeningDisposition::Failed;
        second.reason_code = Some("solver-process-failed".to_string());

        let report =
            audit_screening_funnel(&ids, &records, &ReferenceAuditInput { labels: vec![] })
                .unwrap();
        let high = report
            .stages
            .iter()
            .find(|summary| summary.stage == ScreeningStage::HighFidelity)
            .unwrap();
        assert_eq!(high.non_converged, 1);
        assert_eq!(high.failed, 1);
        assert_eq!(high.evaluated, 2);
        assert_eq!(report.reference_audit.screening_recall, None);
    }

    #[test]
    fn partial_reference_labels_do_not_claim_exact_recall() {
        let ids = two_candidates();
        let partial = ReferenceAuditInput {
            labels: vec![ReferenceLabel {
                candidate_id: "candidate-a".to_string(),
                disposition: ReferenceDisposition::Feasible,
            }],
        };
        let report = audit_screening_funnel(&ids, &complete_records(&ids), &partial).unwrap();
        assert!(!report.reference_audit.exhaustive_labels);
        assert_eq!(report.reference_audit.labels_supplied, 1);
        assert_eq!(report.reference_audit.screening_recall, None);
    }

    #[test]
    fn non_pass_disposition_requires_a_reason_code() {
        let ids = two_candidates();
        let mut records = complete_records(&ids);
        let analytical = records
            .iter_mut()
            .find(|record| {
                record.candidate_id == "candidate-b" && record.stage == ScreeningStage::Analytical
            })
            .unwrap();
        analytical.disposition = ScreeningDisposition::Rejected;
        analytical.reason_code = None;

        let errors = audit_screening_funnel(&ids, &records, &feasible_labels(&ids)).unwrap_err();
        assert!(errors.contains(&FunnelValidationError::MissingReasonCode {
            candidate_id: "candidate-b".to_string(),
            stage: ScreeningStage::Analytical,
        }));
    }

    #[test]
    fn analytical_stage_cannot_be_marked_not_evaluated() {
        let ids = two_candidates();
        let mut records = complete_records(&ids);
        let analytical = records
            .iter_mut()
            .find(|record| {
                record.candidate_id == "candidate-b" && record.stage == ScreeningStage::Analytical
            })
            .unwrap();
        analytical.disposition = ScreeningDisposition::NotEvaluated;
        analytical.reason_code = Some("analytical-stage-skipped".to_string());

        let errors = audit_screening_funnel(&ids, &records, &feasible_labels(&ids)).unwrap_err();
        assert!(
            errors.contains(&FunnelValidationError::InvalidSelectionBasis {
                candidate_id: "candidate-b".to_string(),
                stage: ScreeningStage::Analytical,
            })
        );
    }

    #[test]
    fn reference_label_for_unknown_candidate_is_rejected() {
        let ids = two_candidates();
        let labels = ReferenceAuditInput {
            labels: vec![ReferenceLabel {
                candidate_id: "not-in-universe".to_string(),
                disposition: ReferenceDisposition::Feasible,
            }],
        };
        let errors = audit_screening_funnel(&ids, &complete_records(&ids), &labels).unwrap_err();
        assert!(
            errors.contains(&FunnelValidationError::UnknownReferenceLabel(
                "not-in-universe".to_string()
            ))
        );
    }

    #[test]
    fn duplicate_reference_label_is_rejected() {
        let ids = two_candidates();
        let labels = ReferenceAuditInput {
            labels: vec![
                ReferenceLabel {
                    candidate_id: "candidate-a".to_string(),
                    disposition: ReferenceDisposition::Feasible,
                },
                ReferenceLabel {
                    candidate_id: "candidate-a".to_string(),
                    disposition: ReferenceDisposition::Infeasible,
                },
            ],
        };
        let errors = audit_screening_funnel(&ids, &complete_records(&ids), &labels).unwrap_err();
        assert!(
            errors.contains(&FunnelValidationError::DuplicateReferenceLabel(
                "candidate-a".to_string()
            ))
        );
    }

    #[test]
    fn candidate_universe_duplicate_is_rejected() {
        let ids = vec!["candidate-a".to_string(), "candidate-a".to_string()];
        let errors = audit_screening_funnel(&ids, &complete_records(&ids), &feasible_labels(&ids))
            .unwrap_err();
        assert!(
            errors.contains(&FunnelValidationError::DuplicateCandidateId(
                "candidate-a".to_string()
            ))
        );
    }

    #[test]
    fn empty_reference_truth_never_claims_zero_feasible_recall() {
        let ids = two_candidates();
        let report = audit_screening_funnel(
            &ids,
            &complete_records(&ids),
            &ReferenceAuditInput { labels: vec![] },
        )
        .unwrap();
        assert_eq!(report.reference_audit.screening_recall, None);
    }
}
