// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generator-local proposal records emitted before any semantic `DiscoveryRun` binding.
//!
//! These records answer only "what opportunity set did Forge see and what pair did it draw?".
//! They intentionally do not contain `ProblemId`, `DiscoveryRun`, evaluator, promotion, or runtime
//! authority. A later bridge may qualify a raw record against a semantic proposal policy.

use crate::mutations::RecordedMutation;
use crate::trace::{ForgeAttemptId, ForgeTraceError};
use serde::{Deserialize, Serialize};
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ForgeRawProposalError {
    #[error("Forge raw proposal attempt identity is invalid: {0}")]
    InvalidAttempt(String),
    #[error("Forge raw proposal generation disagrees with its attempt identity")]
    GenerationMismatch,
    #[error("Forge raw proposal operator text must be non-empty canonical single-line text")]
    InvalidOperator,
    #[error("Forge raw proposal mutation detail must be canonical single-line text")]
    InvalidMutationDetail,
    #[error("Forge raw proposal operator indexes are not canonical and contiguous")]
    OperatorIndexMismatch,
    #[error("Forge raw proposal eligible-site count cannot be represented")]
    CountOverflow,
    #[error("Forge raw proposal total does not equal its checked opportunity sum")]
    TotalMismatch,
    #[error("Forge raw proposal decision is inconsistent with its opportunity set")]
    DecisionMismatch,
    #[error("Forge raw proposal mutation effect disagrees with the selected operator")]
    MutationMismatch,
    #[error("Forge raw proposal identity does not match canonical fields")]
    IdentityMismatch,
    #[error("Forge raw proposal archive contains mixed baseline artifacts or search seeds")]
    MixedSearchIdentity,
    #[error("Forge raw proposal archive contains a duplicate/non-canonical attempt")]
    NonCanonicalAttemptOrder,
    #[error("Forge raw proposal archive identity does not match canonical fields")]
    ArchiveIdentityMismatch,
}

impl From<ForgeTraceError> for ForgeRawProposalError {
    fn from(error: ForgeTraceError) -> Self {
        Self::InvalidAttempt(error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeRawOpportunity {
    operator_index: u64,
    operator: String,
    eligible_sites: u64,
}

impl ForgeRawOpportunity {
    fn new(
        operator_index: usize,
        operator: &str,
        eligible_sites: usize,
    ) -> Result<Self, ForgeRawProposalError> {
        validate_operator(operator)?;
        Ok(Self {
            operator_index: u64::try_from(operator_index)
                .map_err(|_| ForgeRawProposalError::CountOverflow)?,
            operator: operator.to_string(),
            eligible_sites: u64::try_from(eligible_sites)
                .map_err(|_| ForgeRawProposalError::CountOverflow)?,
        })
    }

    pub fn operator_index(&self) -> u64 { self.operator_index }
    pub fn operator(&self) -> &str { &self.operator }
    pub fn eligible_sites(&self) -> u64 { self.eligible_sites }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeRawSelection {
    operator_index: u64,
    operator: String,
    site_index: u64,
    global_pair_index: u64,
}

impl ForgeRawSelection {
    pub fn operator_index(&self) -> u64 { self.operator_index }
    pub fn operator(&self) -> &str { &self.operator }
    pub fn site_index(&self) -> u64 { self.site_index }
    pub fn global_pair_index(&self) -> u64 { self.global_pair_index }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ForgeRawProposalDecision {
    NoEligibleSites,
    Selected { selection: ForgeRawSelection },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeRawMutationEffect {
    operator: String,
    detail: String,
}

impl ForgeRawMutationEffect {
    pub fn operator(&self) -> &str { &self.operator }
    pub fn detail(&self) -> &str { &self.detail }
}

/// Exact generator-local record of one proposal attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeRawProposalRecord {
    id: ContentId,
    attempt_id: ForgeAttemptId,
    generation: u64,
    parent_artifact_id: ContentId,
    opportunities: Vec<ForgeRawOpportunity>,
    total_eligible_sites: u64,
    decision: ForgeRawProposalDecision,
    mutation_effect: Option<ForgeRawMutationEffect>,
}

impl ForgeRawProposalRecord {
    pub fn from_recorded(
        attempt_id: ForgeAttemptId,
        parent_artifact_id: ContentId,
        recorded: &RecordedMutation,
    ) -> Result<Self, ForgeRawProposalError> {
        attempt_id.validate()?;
        let generation = attempt_id.generation();
        let opportunities = recorded
            .opportunities
            .iter()
            .enumerate()
            .map(|(index, opportunity)| {
                ForgeRawOpportunity::new(index, opportunity.operator, opportunity.eligible_sites)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let total_eligible_sites = u64::try_from(recorded.total_eligible_sites)
            .map_err(|_| ForgeRawProposalError::CountOverflow)?;

        let decision = match &recorded.selection {
            None => ForgeRawProposalDecision::NoEligibleSites,
            Some(selection) => {
                validate_operator(selection.operator)?;
                ForgeRawProposalDecision::Selected {
                    selection: ForgeRawSelection {
                        operator_index: u64::try_from(selection.operator_index)
                            .map_err(|_| ForgeRawProposalError::CountOverflow)?,
                        operator: selection.operator.to_string(),
                        site_index: u64::try_from(selection.site_index)
                            .map_err(|_| ForgeRawProposalError::CountOverflow)?,
                        global_pair_index: u64::try_from(selection.global_pair_index)
                            .map_err(|_| ForgeRawProposalError::CountOverflow)?,
                    },
                }
            }
        };
        let mutation_effect = recorded
            .mutation
            .as_ref()
            .map(|mutation| {
                validate_operator(mutation.operator)?;
                validate_detail(&mutation.detail)?;
                Ok(ForgeRawMutationEffect {
                    operator: mutation.operator.to_string(),
                    detail: mutation.detail.clone(),
                })
            })
            .transpose()?;

        let mut record = Self {
            id: ContentId::derive("symthaea.forge-raw-proposal.uninitialized", [b"v1".as_slice()]),
            attempt_id,
            generation,
            parent_artifact_id,
            opportunities,
            total_eligible_sites,
            decision,
            mutation_effect,
        };
        record.id = derive_record_id(&record);
        record.validate()?;
        Ok(record)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn attempt_id(&self) -> &ForgeAttemptId { &self.attempt_id }
    pub fn generation(&self) -> u64 { self.generation }
    pub fn parent_artifact_id(&self) -> &ContentId { &self.parent_artifact_id }
    pub fn opportunities(&self) -> &[ForgeRawOpportunity] { &self.opportunities }
    pub fn total_eligible_sites(&self) -> u64 { self.total_eligible_sites }
    pub fn decision(&self) -> &ForgeRawProposalDecision { &self.decision }
    pub fn mutation_effect(&self) -> Option<&ForgeRawMutationEffect> { self.mutation_effect.as_ref() }

    pub fn validate(&self) -> Result<(), ForgeRawProposalError> {
        self.attempt_id.validate()?;
        if self.generation != self.attempt_id.generation() {
            return Err(ForgeRawProposalError::GenerationMismatch);
        }
        let mut total = 0u64;
        for (expected_index, opportunity) in self.opportunities.iter().enumerate() {
            validate_operator(&opportunity.operator)?;
            let expected_index = u64::try_from(expected_index)
                .map_err(|_| ForgeRawProposalError::CountOverflow)?;
            if opportunity.operator_index != expected_index {
                return Err(ForgeRawProposalError::OperatorIndexMismatch);
            }
            total = total
                .checked_add(opportunity.eligible_sites)
                .ok_or(ForgeRawProposalError::CountOverflow)?;
        }
        if total != self.total_eligible_sites {
            return Err(ForgeRawProposalError::TotalMismatch);
        }
        validate_decision(
            &self.opportunities,
            self.total_eligible_sites,
            &self.decision,
            self.mutation_effect.as_ref(),
        )?;
        if derive_record_id(self) == self.id {
            Ok(())
        } else {
            Err(ForgeRawProposalError::IdentityMismatch)
        }
    }
}

fn validate_decision(
    opportunities: &[ForgeRawOpportunity],
    total: u64,
    decision: &ForgeRawProposalDecision,
    mutation_effect: Option<&ForgeRawMutationEffect>,
) -> Result<(), ForgeRawProposalError> {
    match decision {
        ForgeRawProposalDecision::NoEligibleSites => {
            if total != 0 || mutation_effect.is_some() {
                return Err(ForgeRawProposalError::DecisionMismatch);
            }
        }
        ForgeRawProposalDecision::Selected { selection } => {
            validate_operator(&selection.operator)?;
            if total == 0 {
                return Err(ForgeRawProposalError::DecisionMismatch);
            }
            let index = usize::try_from(selection.operator_index)
                .map_err(|_| ForgeRawProposalError::CountOverflow)?;
            let selected = opportunities
                .get(index)
                .ok_or(ForgeRawProposalError::DecisionMismatch)?;
            if selected.operator != selection.operator
                || selection.site_index >= selected.eligible_sites
                || selection.global_pair_index >= total
            {
                return Err(ForgeRawProposalError::DecisionMismatch);
            }
            let offset = opportunities[..index].iter().try_fold(0u64, |sum, opportunity| {
                sum.checked_add(opportunity.eligible_sites)
                    .ok_or(ForgeRawProposalError::CountOverflow)
            })?;
            if offset
                .checked_add(selection.site_index)
                .ok_or(ForgeRawProposalError::CountOverflow)?
                != selection.global_pair_index
            {
                return Err(ForgeRawProposalError::DecisionMismatch);
            }
            if let Some(effect) = mutation_effect {
                validate_operator(&effect.operator)?;
                validate_detail(&effect.detail)?;
                if effect.operator != selection.operator {
                    return Err(ForgeRawProposalError::MutationMismatch);
                }
            }
        }
    }
    Ok(())
}

fn validate_operator(operator: &str) -> Result<(), ForgeRawProposalError> {
    if operator.is_empty()
        || operator.trim() != operator
        || operator.chars().any(char::is_control)
    {
        Err(ForgeRawProposalError::InvalidOperator)
    } else {
        Ok(())
    }
}

fn validate_detail(detail: &str) -> Result<(), ForgeRawProposalError> {
    if detail.trim() != detail || detail.chars().any(char::is_control) {
        Err(ForgeRawProposalError::InvalidMutationDetail)
    } else {
        Ok(())
    }
}

fn derive_record_id(record: &ForgeRawProposalRecord) -> ContentId {
    let generation = record.generation.to_be_bytes();
    let count = (record.opportunities.len() as u64).to_be_bytes();
    let total = record.total_eligible_sites.to_be_bytes();
    let mut parts = vec![
        record.attempt_id.as_content_id().as_str().as_bytes().to_vec(),
        generation.to_vec(),
        record.parent_artifact_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    for opportunity in &record.opportunities {
        parts.push(opportunity.operator_index.to_be_bytes().to_vec());
        parts.push(opportunity.operator.as_bytes().to_vec());
        parts.push(opportunity.eligible_sites.to_be_bytes().to_vec());
    }
    parts.push(total.to_vec());
    match &record.decision {
        ForgeRawProposalDecision::NoEligibleSites => parts.push(b"no-eligible-sites".to_vec()),
        ForgeRawProposalDecision::Selected { selection } => {
            parts.push(b"selected".to_vec());
            parts.push(selection.operator_index.to_be_bytes().to_vec());
            parts.push(selection.operator.as_bytes().to_vec());
            parts.push(selection.site_index.to_be_bytes().to_vec());
            parts.push(selection.global_pair_index.to_be_bytes().to_vec());
        }
    }
    match &record.mutation_effect {
        None => parts.push(b"no-source-effect".to_vec()),
        Some(effect) => {
            parts.push(b"source-effect".to_vec());
            parts.push(effect.operator.as_bytes().to_vec());
            parts.push(effect.detail.as_bytes().to_vec());
        }
    }
    ContentId::derive(
        "symthaea.forge-raw-proposal.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Canonical generator-local proposal archive. It can be empty when a search aborts before the
/// proposal stage. Semantic run/policy binding is intentionally absent here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeRawProposalArchive {
    id: ContentId,
    records: Vec<ForgeRawProposalRecord>,
}

impl ForgeRawProposalArchive {
    pub fn from_records(
        mut records: Vec<ForgeRawProposalRecord>,
    ) -> Result<Self, ForgeRawProposalError> {
        records.sort_by_key(|record| record.attempt_id().ordinal());
        validate_records(&records)?;
        let id = derive_archive_id(&records);
        Ok(Self { id, records })
    }

    pub fn empty() -> Self {
        Self::from_records(Vec::new()).expect("empty raw proposal archive is canonical")
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn records(&self) -> &[ForgeRawProposalRecord] { &self.records }

    pub fn get(&self, attempt: &ForgeAttemptId) -> Option<&ForgeRawProposalRecord> {
        self.records
            .binary_search_by_key(&attempt.ordinal(), |record| record.attempt_id().ordinal())
            .ok()
            .map(|index| &self.records[index])
            .filter(|record| record.attempt_id() == attempt)
    }

    pub fn validate(&self) -> Result<(), ForgeRawProposalError> {
        validate_records(&self.records)?;
        if derive_archive_id(&self.records) == self.id {
            Ok(())
        } else {
            Err(ForgeRawProposalError::ArchiveIdentityMismatch)
        }
    }
}

fn validate_records(records: &[ForgeRawProposalRecord]) -> Result<(), ForgeRawProposalError> {
    let mut baseline: Option<&ContentId> = None;
    let mut seed = None;
    let mut previous = None;
    for record in records {
        record.validate()?;
        let attempt = record.attempt_id();
        if baseline.is_some_and(|value| value != attempt.baseline_artifact_id())
            || seed.is_some_and(|value| value != attempt.seed())
        {
            return Err(ForgeRawProposalError::MixedSearchIdentity);
        }
        if previous.is_some_and(|ordinal| attempt.ordinal() <= ordinal) {
            return Err(ForgeRawProposalError::NonCanonicalAttemptOrder);
        }
        baseline = Some(attempt.baseline_artifact_id());
        seed = Some(attempt.seed());
        previous = Some(attempt.ordinal());
    }
    Ok(())
}

fn derive_archive_id(records: &[ForgeRawProposalRecord]) -> ContentId {
    let count = (records.len() as u64).to_be_bytes();
    let mut parts = vec![count.to_vec()];
    parts.extend(
        records
            .iter()
            .map(|record| record.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-raw-proposal-archive.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::full_source_artifact_id;
    use crate::mutations::{ComparisonOperatorSwap, Mutator, NumericLiteralPerturb};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn parse_body(src: &str) -> syn::Block {
        let file: syn::File = syn::parse_str(src).unwrap();
        match &file.items[0] {
            syn::Item::Fn(function) => (*function.block).clone(),
            _ => panic!("expected free function fixture"),
        }
    }

    #[test]
    fn recorded_draw_round_trips_through_raw_identity() {
        let baseline = full_source_artifact_id("fn f(x: i32) -> bool { x < 5 }\n");
        let attempt = ForgeAttemptId::derive(&baseline, 7, 0, 0);
        let mut body = parse_body("fn f(x: i32, y: i32) -> bool { x < 5 && y + 2 > 9 }");
        let mut rng = StdRng::seed_from_u64(11);
        let recorded = Mutator::default().mutate_one_recorded(&mut body, &mut rng);
        let raw = ForgeRawProposalRecord::from_recorded(attempt, baseline, &recorded).unwrap();
        raw.validate().unwrap();
        assert_eq!(raw.total_eligible_sites(), recorded.total_eligible_sites as u64);
        assert_eq!(raw.opportunities().len(), recorded.opportunities.len());
    }

    #[test]
    fn duplicate_operator_names_are_faithfully_recorded_by_slot() {
        let baseline = full_source_artifact_id("fn f(x: i32) -> bool { x < 5 }\n");
        let attempt = ForgeAttemptId::derive(&baseline, 7, 0, 0);
        let mut body = parse_body("fn f(x: i32) -> bool { x < 5 }");
        let mut rng = StdRng::seed_from_u64(13);
        let mutator = Mutator::new(vec![
            Box::new(ComparisonOperatorSwap),
            Box::new(ComparisonOperatorSwap),
        ]);
        let recorded = mutator.mutate_one_recorded(&mut body, &mut rng);
        let raw = ForgeRawProposalRecord::from_recorded(attempt, baseline, &recorded).unwrap();
        assert_eq!(raw.opportunities().len(), 2);
        let ForgeRawProposalDecision::Selected { selection } = raw.decision() else {
            panic!("fixture must select one eligible slot");
        };
        assert!(selection.operator_index() < 2);
    }

    #[test]
    fn selected_pair_with_no_source_effect_remains_selected() {
        let baseline = full_source_artifact_id("fn f() -> i64 { 0 }\n");
        let attempt = ForgeAttemptId::derive(&baseline, 7, 0, 0);
        let mut body = parse_body("fn f() -> i64 { 0 }");
        let mut rng = StdRng::seed_from_u64(7);
        let mutator = Mutator::new(vec![Box::new(NumericLiteralPerturb { max_fraction: 0.1 })]);
        let recorded = mutator.mutate_one_recorded(&mut body, &mut rng);
        let raw = ForgeRawProposalRecord::from_recorded(attempt, baseline, &recorded).unwrap();
        assert!(matches!(raw.decision(), ForgeRawProposalDecision::Selected { .. }));
        assert!(raw.mutation_effect().is_none());
    }

    #[test]
    fn archive_is_canonical_by_attempt_ordinal() {
        let baseline = full_source_artifact_id("fn f(x: i32) -> bool { x < 5 }\n");
        let mut body_a = parse_body("fn f(x: i32) -> bool { x < 5 }");
        let mut body_b = body_a.clone();
        let mut rng_a = StdRng::seed_from_u64(1);
        let mut rng_b = StdRng::seed_from_u64(2);
        let mutator = Mutator::default();
        let a = ForgeRawProposalRecord::from_recorded(
            ForgeAttemptId::derive(&baseline, 7, 0, 0),
            baseline.clone(),
            &mutator.mutate_one_recorded(&mut body_a, &mut rng_a),
        ).unwrap();
        let b = ForgeRawProposalRecord::from_recorded(
            ForgeAttemptId::derive(&baseline, 7, 1, 0),
            baseline,
            &mutator.mutate_one_recorded(&mut body_b, &mut rng_b),
        ).unwrap();
        let archive = ForgeRawProposalArchive::from_records(vec![b, a]).unwrap();
        assert_eq!(archive.records()[0].attempt_id().ordinal(), 0);
        assert_eq!(archive.records()[1].attempt_id().ordinal(), 1);
        archive.validate().unwrap();
    }
}
