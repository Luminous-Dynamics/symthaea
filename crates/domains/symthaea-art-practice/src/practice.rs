// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistent artistic-practice records shared across studio hosts.
//!
//! These types model questions, intentions, studies, works, discoveries, and
//! portfolio memory. They deliberately avoid a universal aesthetic score. The
//! point is to preserve developmental causation: what problem was being worked
//! on, what was tried, what was rejected, and what was learned.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

use crate::{IntentId, ProposalId, RevisionId};

pub const ART_PRACTICE_SCHEMA_V1: &str = "symthaea.art-practice.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuestionStatus {
    Open,
    Dormant,
    Resolved,
    Abandoned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtisticQuestion {
    pub schema: String,
    pub question_id: String,
    pub text: String,
    pub status: QuestionStatus,
    pub opened_at_sequence: u64,
    pub closed_at_sequence: Option<u64>,
    pub origin_evidence_refs: Vec<String>,
    pub related_question_ids: Vec<String>,
}

impl ArtisticQuestion {
    pub fn open(
        question_id: impl Into<String>,
        text: impl Into<String>,
        opened_at_sequence: u64,
    ) -> Self {
        Self {
            schema: ART_PRACTICE_SCHEMA_V1.into(),
            question_id: question_id.into(),
            text: text.into(),
            status: QuestionStatus::Open,
            opened_at_sequence,
            closed_at_sequence: None,
            origin_evidence_refs: Vec::new(),
            related_question_ids: Vec::new(),
        }
    }

    pub fn close(
        &mut self,
        status: QuestionStatus,
        sequence: u64,
    ) -> Result<(), PracticeError> {
        if !matches!(status, QuestionStatus::Resolved | QuestionStatus::Abandoned) {
            return Err(PracticeError::InvalidClosingStatus(status));
        }
        if matches!(self.status, QuestionStatus::Resolved | QuestionStatus::Abandoned) {
            return Err(PracticeError::QuestionAlreadyClosed(self.question_id.clone()));
        }
        if sequence < self.opened_at_sequence {
            return Err(PracticeError::NonMonotonicSequence);
        }
        self.status = status;
        self.closed_at_sequence = Some(sequence);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkingIntention {
    pub intent_id: IntentId,
    pub question_id: String,
    pub description: String,
    pub constraints: Vec<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortfolioEntryKind {
    Work,
    Study,
    Sketch,
    AbandonedWork,
    Failure,
    Experiment,
}

/// Developmental record for one work or study. `rejected_proposals` is retained
/// because restraint and failed alternatives are part of artistic history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortfolioEntry {
    pub schema: String,
    pub work_id: String,
    pub kind: PortfolioEntryKind,
    pub title: Option<String>,
    pub parent_work_id: Option<String>,
    pub question_ids: Vec<String>,
    pub intent_ids: Vec<IntentId>,
    pub committed_revisions: Vec<RevisionId>,
    pub rejected_proposals: Vec<ProposalId>,
    pub discovery_ids: Vec<String>,
    pub artifact_refs: Vec<String>,
    pub started_at_sequence: u64,
    pub ended_at_sequence: Option<u64>,
}

impl PortfolioEntry {
    pub fn new(
        work_id: impl Into<String>,
        kind: PortfolioEntryKind,
        started_at_sequence: u64,
    ) -> Self {
        Self {
            schema: ART_PRACTICE_SCHEMA_V1.into(),
            work_id: work_id.into(),
            kind,
            title: None,
            parent_work_id: None,
            question_ids: Vec::new(),
            intent_ids: Vec::new(),
            committed_revisions: Vec::new(),
            rejected_proposals: Vec::new(),
            discovery_ids: Vec::new(),
            artifact_refs: Vec::new(),
            started_at_sequence,
            ended_at_sequence: None,
        }
    }

    pub fn record_revision(&mut self, revision: RevisionId) -> Result<(), PracticeError> {
        if self
            .committed_revisions
            .last()
            .is_some_and(|existing| existing == &revision)
        {
            return Err(PracticeError::DuplicateRevision(revision));
        }
        self.committed_revisions.push(revision);
        Ok(())
    }

    pub fn finish(&mut self, sequence: u64) -> Result<(), PracticeError> {
        if sequence < self.started_at_sequence {
            return Err(PracticeError::NonMonotonicSequence);
        }
        if self.ended_at_sequence.is_some() {
            return Err(PracticeError::WorkAlreadyClosed(self.work_id.clone()));
        }
        self.ended_at_sequence = Some(sequence);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TechniqueAttempt {
    pub attempt_id: String,
    pub medium: String,
    pub intended_effect: String,
    pub action_refs: Vec<String>,
    pub observed_consequence_refs: Vec<String>,
    pub prediction_error: Option<f64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TechniqueStudy {
    pub study_id: String,
    pub question_id: Option<String>,
    pub technique_label: String,
    pub attempts: Vec<TechniqueAttempt>,
    pub transfer_targets: Vec<String>,
    pub unresolved_notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtisticDiscovery {
    pub discovery_id: String,
    pub sequence: u64,
    pub statement: String,
    pub evidence_refs: Vec<String>,
    pub related_work_ids: Vec<String>,
    pub related_question_ids: Vec<String>,
}

/// Append-only developmental memory. This is intentionally an evidence ledger,
/// not a style vector or reward history.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ArtisticDevelopmentLedger {
    pub questions: Vec<ArtisticQuestion>,
    pub works: Vec<PortfolioEntry>,
    pub technique_studies: Vec<TechniqueStudy>,
    pub discoveries: Vec<ArtisticDiscovery>,
}

impl ArtisticDevelopmentLedger {
    pub fn validate_unique_ids(&self) -> Result<(), PracticeError> {
        unique(
            self.questions.iter().map(|item| item.question_id.as_str()),
            "question",
        )?;
        unique(self.works.iter().map(|item| item.work_id.as_str()), "work")?;
        unique(
            self.technique_studies.iter().map(|item| item.study_id.as_str()),
            "technique-study",
        )?;
        unique(
            self.discoveries.iter().map(|item| item.discovery_id.as_str()),
            "discovery",
        )?;
        Ok(())
    }

    pub fn unresolved_questions(&self) -> impl Iterator<Item = &ArtisticQuestion> {
        self.questions.iter().filter(|question| {
            matches!(question.status, QuestionStatus::Open | QuestionStatus::Dormant)
        })
    }
}

fn unique<'a>(values: impl Iterator<Item = &'a str>, kind: &'static str) -> Result<(), PracticeError> {
    let mut seen = BTreeSet::new();
    for value in values {
        if value.trim().is_empty() {
            return Err(PracticeError::EmptyId(kind));
        }
        if !seen.insert(value) {
            return Err(PracticeError::DuplicateId {
                kind,
                id: value.to_string(),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PracticeError {
    #[error("invalid closing status: {0:?}")]
    InvalidClosingStatus(QuestionStatus),
    #[error("question already closed: {0}")]
    QuestionAlreadyClosed(String),
    #[error("work already closed: {0}")]
    WorkAlreadyClosed(String),
    #[error("sequence would move backward")]
    NonMonotonicSequence,
    #[error("duplicate committed revision: {0:?}")]
    DuplicateRevision(RevisionId),
    #[error("{0} id may not be empty")]
    EmptyId(&'static str),
    #[error("duplicate {kind} id: {id}")]
    DuplicateId { kind: &'static str, id: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_a_question_is_monotonic() {
        let mut question = ArtisticQuestion::open("q1", "What can a shadow conceal?", 10);
        question.close(QuestionStatus::Resolved, 20).unwrap();
        assert_eq!(question.status, QuestionStatus::Resolved);
        assert!(question.close(QuestionStatus::Abandoned, 21).is_err());
    }

    #[test]
    fn portfolio_keeps_rejected_candidates_as_history() {
        let mut work = PortfolioEntry::new("w1", PortfolioEntryKind::Work, 1);
        work.rejected_proposals.push(ProposalId::from("p-rejected"));
        work.record_revision(RevisionId::from("r1")).unwrap();
        assert_eq!(work.rejected_proposals.len(), 1);
    }

    #[test]
    fn duplicate_revision_is_rejected() {
        let mut work = PortfolioEntry::new("w1", PortfolioEntryKind::Study, 1);
        work.record_revision(RevisionId::from("r1")).unwrap();
        assert!(matches!(
            work.record_revision(RevisionId::from("r1")),
            Err(PracticeError::DuplicateRevision(_))
        ));
    }

    #[test]
    fn ledger_rejects_duplicate_developmental_identity() {
        let mut ledger = ArtisticDevelopmentLedger::default();
        ledger.questions.push(ArtisticQuestion::open("q1", "first", 1));
        ledger.questions.push(ArtisticQuestion::open("q1", "second", 2));
        assert!(matches!(
            ledger.validate_unique_ids(),
            Err(PracticeError::DuplicateId { kind: "question", .. })
        ));
    }

    #[test]
    fn unresolved_questions_are_preserved_without_ranking() {
        let mut ledger = ArtisticDevelopmentLedger::default();
        ledger.questions.push(ArtisticQuestion::open("q1", "open", 1));
        let mut resolved = ArtisticQuestion::open("q2", "resolved", 1);
        resolved.close(QuestionStatus::Resolved, 2).unwrap();
        ledger.questions.push(resolved);
        assert_eq!(ledger.unresolved_questions().count(), 1);
    }
}
