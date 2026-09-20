// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Candidate ensemble and independent critique for adult fantasy dialogue.
//!
//! Hard admissibility failures are lexicographically disqualifying. Creative
//! quality scores are considered only after scope/boundary/safety gates pass.

use crate::fantasy_style_lattice::FantasyStyleProposalV1;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const FANTASY_CANDIDATE_CRITIC_SCHEMA_V1: &str =
    "symthaea.communication.fantasy-candidate-critic.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FantasyCandidateHardFailureV1 {
    ScopeBinding,
    HardBoundary,
    RealityNamespace,
    IdentityPolicy,
    Privacy,
    AntiCoercion,
    StopOrPacingViolation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FantasyCandidateSoftMetricV1 {
    StyleFit,
    PacingFit,
    Continuity,
    PersonaConsistency,
    Naturalness,
    RepetitionAvoidance,
    CallbackQuality,
    Novelty,
}

pub const ALL_FANTASY_CANDIDATE_SOFT_METRICS_V1: [FantasyCandidateSoftMetricV1; 8] = [
    FantasyCandidateSoftMetricV1::StyleFit,
    FantasyCandidateSoftMetricV1::PacingFit,
    FantasyCandidateSoftMetricV1::Continuity,
    FantasyCandidateSoftMetricV1::PersonaConsistency,
    FantasyCandidateSoftMetricV1::Naturalness,
    FantasyCandidateSoftMetricV1::RepetitionAvoidance,
    FantasyCandidateSoftMetricV1::CallbackQuality,
    FantasyCandidateSoftMetricV1::Novelty,
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FantasyDialogueCandidateV1 {
    pub candidate_id: String,
    pub session_id: String,
    pub session_epoch: u64,
    pub style_proposal_epoch: u64,
    pub topic_id: String,
    pub generator_id: String,
    /// Opaque reference to separately governed candidate content.
    pub content_ref: String,
    /// Opaque commitment to the exact candidate bytes or canonical content form.
    pub content_commitment: String,
}

impl FantasyDialogueCandidateV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        candidate_id: impl Into<String>,
        session_id: impl Into<String>,
        session_epoch: u64,
        style_proposal_epoch: u64,
        topic_id: impl Into<String>,
        generator_id: impl Into<String>,
        content_ref: impl Into<String>,
        content_commitment: impl Into<String>,
    ) -> Result<Self, FantasyCandidateCriticErrorV1> {
        if session_epoch == 0 || style_proposal_epoch == 0 {
            return Err(FantasyCandidateCriticErrorV1::InvalidEpoch);
        }
        Ok(Self {
            candidate_id: canonical_id(candidate_id.into())?,
            session_id: canonical_id(session_id.into())?,
            session_epoch,
            style_proposal_epoch,
            topic_id: canonical_id(topic_id.into())?,
            generator_id: canonical_id(generator_id.into())?,
            content_ref: canonical_ref(content_ref.into())?,
            content_commitment: canonical_ref(content_commitment.into())?,
        })
    }

    fn scope_matches(&self, style: &FantasyStyleProposalV1) -> bool {
        self.session_id == style.session_id
            && self.session_epoch == style.session_epoch
            && self.style_proposal_epoch == style.proposal_epoch
            && self.topic_id == style.topic_id
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FantasyCandidateCriticReportV1 {
    pub report_id: String,
    pub critic_id: String,
    pub candidate_id: String,
    pub hard_failures: BTreeSet<FantasyCandidateHardFailureV1>,
    pub soft_scores: BTreeMap<FantasyCandidateSoftMetricV1, f32>,
    /// Material uncertainty blocks automatic selection but is not converted into
    /// a fabricated safety failure.
    pub material_uncertainty: bool,
}

impl FantasyCandidateCriticReportV1 {
    pub fn new(
        report_id: impl Into<String>,
        critic_id: impl Into<String>,
        candidate_id: impl Into<String>,
        hard_failures: BTreeSet<FantasyCandidateHardFailureV1>,
        soft_scores: BTreeMap<FantasyCandidateSoftMetricV1, f32>,
        material_uncertainty: bool,
    ) -> Result<Self, FantasyCandidateCriticErrorV1> {
        validate_complete_scores(&soft_scores)?;
        Ok(Self {
            report_id: canonical_id(report_id.into())?,
            critic_id: canonical_id(critic_id.into())?,
            candidate_id: canonical_id(candidate_id.into())?,
            hard_failures,
            soft_scores,
            material_uncertainty,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FantasyCandidateSelectionPolicyV1 {
    pub minimum_independent_critics: u8,
    /// Maximum permitted max-min disagreement per metric before a candidate is
    /// retained as uncertain rather than automatically selected.
    pub disagreement_threshold: f32,
    /// Lexicographic soft-quality priority after all hard gates pass.
    pub metric_priority: Vec<FantasyCandidateSoftMetricV1>,
}

impl Default for FantasyCandidateSelectionPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_independent_critics: 2,
            disagreement_threshold: 0.25,
            metric_priority: ALL_FANTASY_CANDIDATE_SOFT_METRICS_V1.to_vec(),
        }
    }
}

impl FantasyCandidateSelectionPolicyV1 {
    pub fn validate(&self) -> Result<(), FantasyCandidateCriticErrorV1> {
        if self.minimum_independent_critics == 0 || self.minimum_independent_critics > 16 {
            return Err(FantasyCandidateCriticErrorV1::InvalidSelectionPolicy);
        }
        if !self.disagreement_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.disagreement_threshold)
        {
            return Err(FantasyCandidateCriticErrorV1::InvalidSelectionPolicy);
        }
        if self.metric_priority.len() != ALL_FANTASY_CANDIDATE_SOFT_METRICS_V1.len() {
            return Err(FantasyCandidateCriticErrorV1::InvalidSelectionPolicy);
        }
        let unique: BTreeSet<_> = self.metric_priority.iter().copied().collect();
        if unique.len() != ALL_FANTASY_CANDIDATE_SOFT_METRICS_V1.len() {
            return Err(FantasyCandidateCriticErrorV1::InvalidSelectionPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyCandidateAdmissibilityV1 {
    Admissible,
    Uncertain,
    Disqualified,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FantasyCandidateAssessmentV1 {
    pub candidate_id: String,
    pub admissibility: FantasyCandidateAdmissibilityV1,
    pub critic_count: u64,
    pub hard_failures: BTreeSet<FantasyCandidateHardFailureV1>,
    pub disagreement_metrics: BTreeSet<FantasyCandidateSoftMetricV1>,
    pub mean_soft_scores: BTreeMap<FantasyCandidateSoftMetricV1, f32>,
    pub material_uncertainty: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FantasyCandidateSelectionReceiptV1 {
    pub schema: String,
    pub session_id: String,
    pub session_epoch: u64,
    pub style_proposal_epoch: u64,
    pub topic_id: String,
    pub selected_candidate_id: Option<String>,
    pub revision_required: bool,
    pub assessments: Vec<FantasyCandidateAssessmentV1>,
}

pub fn select_fantasy_dialogue_candidate_v1(
    style: &FantasyStyleProposalV1,
    candidates: &[FantasyDialogueCandidateV1],
    reports: &[FantasyCandidateCriticReportV1],
    policy: &FantasyCandidateSelectionPolicyV1,
) -> Result<FantasyCandidateSelectionReceiptV1, FantasyCandidateCriticErrorV1> {
    policy.validate()?;
    if candidates.is_empty() || candidates.len() > 64 || reports.len() > 1024 {
        return Err(FantasyCandidateCriticErrorV1::InvalidBatchSize);
    }

    let mut candidate_ids = BTreeSet::new();
    for candidate in candidates {
        if !candidate_ids.insert(candidate.candidate_id.clone()) {
            return Err(FantasyCandidateCriticErrorV1::DuplicateCandidateId);
        }
    }

    let mut report_ids = BTreeSet::new();
    let mut critic_candidate_pairs = BTreeSet::new();
    for report in reports {
        if !candidate_ids.contains(&report.candidate_id) {
            return Err(FantasyCandidateCriticErrorV1::UnknownCandidateInReport);
        }
        if !report_ids.insert(report.report_id.clone()) {
            return Err(FantasyCandidateCriticErrorV1::DuplicateReportId);
        }
        if !critic_candidate_pairs.insert((report.candidate_id.clone(), report.critic_id.clone())) {
            return Err(FantasyCandidateCriticErrorV1::DuplicateCriticForCandidate);
        }
        validate_complete_scores(&report.soft_scores)?;
    }

    let mut assessments = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let candidate_reports: Vec<&FantasyCandidateCriticReportV1> = reports
            .iter()
            .filter(|report| report.candidate_id == candidate.candidate_id)
            .collect();

        let mut hard_failures = BTreeSet::new();
        if !candidate.scope_matches(style) {
            hard_failures.insert(FantasyCandidateHardFailureV1::ScopeBinding);
        }
        for report in &candidate_reports {
            hard_failures.extend(report.hard_failures.iter().copied());
        }

        let material_uncertainty = candidate_reports
            .iter()
            .any(|report| report.material_uncertainty);
        let mean_soft_scores = mean_scores(&candidate_reports);
        let disagreement_metrics = disagreement_metrics(&candidate_reports, policy.disagreement_threshold);

        let admissibility = if !hard_failures.is_empty() {
            FantasyCandidateAdmissibilityV1::Disqualified
        } else if candidate_reports.len() < usize::from(policy.minimum_independent_critics)
            || material_uncertainty
            || !disagreement_metrics.is_empty()
        {
            FantasyCandidateAdmissibilityV1::Uncertain
        } else {
            FantasyCandidateAdmissibilityV1::Admissible
        };

        assessments.push(FantasyCandidateAssessmentV1 {
            candidate_id: candidate.candidate_id.clone(),
            admissibility,
            critic_count: candidate_reports.len() as u64,
            hard_failures,
            disagreement_metrics,
            mean_soft_scores,
            material_uncertainty,
        });
    }

    assessments.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let selected_candidate_id = assessments
        .iter()
        .filter(|assessment| assessment.admissibility == FantasyCandidateAdmissibilityV1::Admissible)
        .max_by(|left, right| compare_admissible(left, right, policy))
        .map(|assessment| assessment.candidate_id.clone());

    Ok(FantasyCandidateSelectionReceiptV1 {
        schema: FANTASY_CANDIDATE_CRITIC_SCHEMA_V1.into(),
        session_id: style.session_id.clone(),
        session_epoch: style.session_epoch,
        style_proposal_epoch: style.proposal_epoch,
        topic_id: style.topic_id.clone(),
        revision_required: selected_candidate_id.is_none(),
        selected_candidate_id,
        assessments,
    })
}

fn compare_admissible(
    left: &FantasyCandidateAssessmentV1,
    right: &FantasyCandidateAssessmentV1,
    policy: &FantasyCandidateSelectionPolicyV1,
) -> Ordering {
    for metric in &policy.metric_priority {
        let left_score = left.mean_soft_scores.get(metric).copied().unwrap_or(0.0);
        let right_score = right.mean_soft_scores.get(metric).copied().unwrap_or(0.0);
        let order = left_score.total_cmp(&right_score);
        if order != Ordering::Equal {
            return order;
        }
    }
    // Deterministic final tie-break. Lower lexical ID wins, therefore reverse
    // the comparison because the caller uses `max_by`.
    right.candidate_id.cmp(&left.candidate_id)
}

fn mean_scores(
    reports: &[&FantasyCandidateCriticReportV1],
) -> BTreeMap<FantasyCandidateSoftMetricV1, f32> {
    let mut means = BTreeMap::new();
    if reports.is_empty() {
        return means;
    }
    for metric in ALL_FANTASY_CANDIDATE_SOFT_METRICS_V1 {
        let sum: f32 = reports
            .iter()
            .map(|report| report.soft_scores[&metric])
            .sum();
        means.insert(metric, sum / reports.len() as f32);
    }
    means
}

fn disagreement_metrics(
    reports: &[&FantasyCandidateCriticReportV1],
    threshold: f32,
) -> BTreeSet<FantasyCandidateSoftMetricV1> {
    let mut disagreements = BTreeSet::new();
    if reports.len() < 2 {
        return disagreements;
    }
    for metric in ALL_FANTASY_CANDIDATE_SOFT_METRICS_V1 {
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        for report in reports {
            let value = report.soft_scores[&metric];
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        if maximum - minimum > threshold {
            disagreements.insert(metric);
        }
    }
    disagreements
}

fn validate_complete_scores(
    scores: &BTreeMap<FantasyCandidateSoftMetricV1, f32>,
) -> Result<(), FantasyCandidateCriticErrorV1> {
    if scores.len() != ALL_FANTASY_CANDIDATE_SOFT_METRICS_V1.len() {
        return Err(FantasyCandidateCriticErrorV1::IncompleteSoftScores);
    }
    for metric in ALL_FANTASY_CANDIDATE_SOFT_METRICS_V1 {
        let Some(value) = scores.get(&metric) else {
            return Err(FantasyCandidateCriticErrorV1::IncompleteSoftScores);
        };
        if !value.is_finite() || !(0.0..=1.0).contains(value) {
            return Err(FantasyCandidateCriticErrorV1::InvalidSoftScore);
        }
    }
    Ok(())
}

fn canonical_id(value: String) -> Result<String, FantasyCandidateCriticErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 256 {
        return Err(FantasyCandidateCriticErrorV1::InvalidIdentity);
    }
    Ok(value)
}

fn canonical_ref(value: String) -> Result<String, FantasyCandidateCriticErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 1024 {
        return Err(FantasyCandidateCriticErrorV1::InvalidReference);
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FantasyCandidateCriticErrorV1 {
    InvalidEpoch,
    InvalidIdentity,
    InvalidReference,
    InvalidSelectionPolicy,
    InvalidBatchSize,
    DuplicateCandidateId,
    DuplicateReportId,
    DuplicateCriticForCandidate,
    UnknownCandidateInReport,
    IncompleteSoftScores,
    InvalidSoftScore,
}
