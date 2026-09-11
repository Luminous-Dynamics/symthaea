// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit cumulative outcome endpoints for proposal-policy study.
//!
//! Later estimators must not turn an unselected family into a negative example or an apparatus-
//! interrupted candidate into an observed failure. This module freezes that distinction before any
//! weighting or model fitting: each endpoint value is either observed, censored, or counterfactual-
//! unobserved.

use crate::proposal_dataset::{
    ForgeProposalDatasetError, ForgeProposalObservationRow, ForgeProposalObservationTable,
    ForgeProposalRowDecision, ForgeProposalRowOutcome,
};
use crate::trials::ForgeTrialOutcome;
use serde::Serialize;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalEndpointError {
    #[error(transparent)]
    Dataset(#[from] ForgeProposalDatasetError),
    #[error("proposal endpoint record is inconsistent with its source row")]
    SourceMismatch,
    #[error("proposal endpoint record identity does not match canonical fields")]
    RecordIdentityMismatch,
    #[error("proposal endpoint table is not canonical for its source observation table")]
    NonCanonicalTable,
    #[error("proposal endpoint table identity does not match canonical fields")]
    TableIdentityMismatch,
}

/// Cumulative selected-proposal endpoint. Downstream endpoints imply every earlier stage passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalEndpoint {
    DistinctCandidate,
    CompilePassed,
    CorrectnessPassed,
    EvaluationValid,
    SelectedForContinuation,
}

impl ForgeProposalEndpoint {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::DistinctCandidate => b"distinct-candidate",
            Self::CompilePassed => b"compile-passed",
            Self::CorrectnessPassed => b"correctness-passed",
            Self::EvaluationValid => b"evaluation-valid",
            Self::SelectedForContinuation => b"selected-for-continuation",
        }
    }
}

/// Whether an endpoint value is actually observable for one family/attempt row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "status", content = "value")]
pub enum ForgeProposalEndpointValue {
    /// The selected proposal has an observed true/false value for this cumulative endpoint.
    Observed(bool),
    /// A concrete candidate existed, but apparatus failure ended the attempt before this endpoint
    /// could be defensibly classified.
    Censored,
    /// This family was not selected, so its potential outcome was never observed.
    CounterfactualUnobserved,
}

/// One source proposal row projected onto one frozen cumulative endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalEndpointRecord {
    id: ContentId,
    source_row_id: ContentId,
    endpoint: ForgeProposalEndpoint,
    value: ForgeProposalEndpointValue,
}

impl ForgeProposalEndpointRecord {
    pub fn from_row(
        row: &ForgeProposalObservationRow,
        endpoint: ForgeProposalEndpoint,
    ) -> Result<Self, ForgeProposalEndpointError> {
        row.validate()?;
        let value = classify_endpoint(row.decision(), row.outcome(), endpoint);
        let id = derive_record_id(row.id(), endpoint, value);
        Ok(Self {
            id,
            source_row_id: row.id().clone(),
            endpoint,
            value,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn source_row_id(&self) -> &ContentId { &self.source_row_id }
    pub fn endpoint(&self) -> ForgeProposalEndpoint { self.endpoint }
    pub fn value(&self) -> ForgeProposalEndpointValue { self.value }

    pub fn validate_for(
        &self,
        row: &ForgeProposalObservationRow,
    ) -> Result<(), ForgeProposalEndpointError> {
        row.validate()?;
        if self.source_row_id != *row.id()
            || self.value != classify_endpoint(row.decision(), row.outcome(), self.endpoint)
        {
            return Err(ForgeProposalEndpointError::SourceMismatch);
        }
        let expected = derive_record_id(row.id(), self.endpoint, self.value);
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalEndpointError::RecordIdentityMismatch)
        }
    }
}

fn endpoint_value_tag(value: ForgeProposalEndpointValue) -> &'static [u8] {
    match value {
        ForgeProposalEndpointValue::Observed(false) => b"observed-false",
        ForgeProposalEndpointValue::Observed(true) => b"observed-true",
        ForgeProposalEndpointValue::Censored => b"censored",
        ForgeProposalEndpointValue::CounterfactualUnobserved => b"counterfactual-unobserved",
    }
}

fn derive_record_id(
    source_row_id: &ContentId,
    endpoint: ForgeProposalEndpoint,
    value: ForgeProposalEndpointValue,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-endpoint-record.v1",
        [
            source_row_id.as_str().as_bytes(),
            endpoint.tag(),
            endpoint_value_tag(value),
        ],
    )
}

fn classify_endpoint(
    decision: &ForgeProposalRowDecision,
    outcome: ForgeProposalRowOutcome,
    endpoint: ForgeProposalEndpoint,
) -> ForgeProposalEndpointValue {
    if !matches!(decision, ForgeProposalRowDecision::Selected { .. }) {
        return ForgeProposalEndpointValue::CounterfactualUnobserved;
    }

    match outcome {
        ForgeProposalRowOutcome::Unselected => {
            // A valid source row cannot reach this branch while selected; retaining the conservative
            // missing-outcome classification is safer than manufacturing a negative label.
            ForgeProposalEndpointValue::CounterfactualUnobserved
        }
        ForgeProposalRowOutcome::SelectedNoCandidate => ForgeProposalEndpointValue::Observed(false),
        ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::Interrupted) => match endpoint {
            ForgeProposalEndpoint::DistinctCandidate => ForgeProposalEndpointValue::Observed(true),
            _ => ForgeProposalEndpointValue::Censored,
        },
        ForgeProposalRowOutcome::Concrete(outcome) => {
            ForgeProposalEndpointValue::Observed(concrete_endpoint_passed(outcome, endpoint))
        }
    }
}

fn concrete_endpoint_passed(
    outcome: ForgeTrialOutcome,
    endpoint: ForgeProposalEndpoint,
) -> bool {
    match endpoint {
        ForgeProposalEndpoint::DistinctCandidate => true,
        ForgeProposalEndpoint::CompilePassed => !matches!(outcome, ForgeTrialOutcome::RejectedCompilation),
        ForgeProposalEndpoint::CorrectnessPassed => !matches!(
            outcome,
            ForgeTrialOutcome::RejectedCompilation | ForgeTrialOutcome::RejectedCorrectness
        ),
        ForgeProposalEndpoint::EvaluationValid => matches!(
            outcome,
            ForgeTrialOutcome::ValidNotSelected | ForgeTrialOutcome::SelectedForContinuation
        ),
        ForgeProposalEndpoint::SelectedForContinuation => {
            outcome == ForgeTrialOutcome::SelectedForContinuation
        }
    }
}

/// One source proposal table projected onto one endpoint without dropping missing/censored rows.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalEndpointTable {
    id: ContentId,
    source_table_id: ContentId,
    endpoint: ForgeProposalEndpoint,
    records: Vec<ForgeProposalEndpointRecord>,
}

impl ForgeProposalEndpointTable {
    pub fn from_observations(
        source: &ForgeProposalObservationTable,
        endpoint: ForgeProposalEndpoint,
    ) -> Result<Self, ForgeProposalEndpointError> {
        source.validate()?;
        let records = source
            .rows()
            .iter()
            .map(|row| ForgeProposalEndpointRecord::from_row(row, endpoint))
            .collect::<Result<Vec<_>, _>>()?;
        let id = derive_table_id(source.id(), endpoint, &records);
        let table = Self {
            id,
            source_table_id: source.id().clone(),
            endpoint,
            records,
        };
        table.validate_for(source)?;
        Ok(table)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn source_table_id(&self) -> &ContentId { &self.source_table_id }
    pub fn endpoint(&self) -> ForgeProposalEndpoint { self.endpoint }
    pub fn records(&self) -> &[ForgeProposalEndpointRecord] { &self.records }

    pub fn observed(&self) -> impl Iterator<Item = &ForgeProposalEndpointRecord> {
        self.records.iter().filter(|record| {
            matches!(record.value(), ForgeProposalEndpointValue::Observed(_))
        })
    }

    pub fn censored(&self) -> impl Iterator<Item = &ForgeProposalEndpointRecord> {
        self.records
            .iter()
            .filter(|record| record.value() == ForgeProposalEndpointValue::Censored)
    }

    pub fn counterfactual_unobserved(
        &self,
    ) -> impl Iterator<Item = &ForgeProposalEndpointRecord> {
        self.records.iter().filter(|record| {
            record.value() == ForgeProposalEndpointValue::CounterfactualUnobserved
        })
    }

    pub fn validate_for(
        &self,
        source: &ForgeProposalObservationTable,
    ) -> Result<(), ForgeProposalEndpointError> {
        source.validate()?;
        if self.source_table_id != *source.id() || self.records.len() != source.rows().len() {
            return Err(ForgeProposalEndpointError::NonCanonicalTable);
        }
        for (record, row) in self.records.iter().zip(source.rows()) {
            if record.endpoint() != self.endpoint {
                return Err(ForgeProposalEndpointError::NonCanonicalTable);
            }
            record.validate_for(row)?;
        }
        let expected = derive_table_id(source.id(), self.endpoint, &self.records);
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalEndpointError::TableIdentityMismatch)
        }
    }
}

fn derive_table_id(
    source_table_id: &ContentId,
    endpoint: ForgeProposalEndpoint,
    records: &[ForgeProposalEndpointRecord],
) -> ContentId {
    let count = (records.len() as u64).to_be_bytes();
    let mut parts = vec![
        source_table_id.as_str().as_bytes().to_vec(),
        endpoint.tag().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        records
            .iter()
            .map(|record| record.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-proposal-endpoint-table.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selected() -> ForgeProposalRowDecision {
        ForgeProposalRowDecision::Selected {
            site_index: 0,
            global_pair_index: 0,
        }
    }

    #[test]
    fn unselected_family_is_not_encoded_as_failure() {
        let value = classify_endpoint(
            &ForgeProposalRowDecision::NotSelected,
            ForgeProposalRowOutcome::Unselected,
            ForgeProposalEndpoint::SelectedForContinuation,
        );
        assert_eq!(value, ForgeProposalEndpointValue::CounterfactualUnobserved);
    }

    #[test]
    fn selected_no_candidate_is_observed_false_for_cumulative_endpoints() {
        for endpoint in [
            ForgeProposalEndpoint::DistinctCandidate,
            ForgeProposalEndpoint::CompilePassed,
            ForgeProposalEndpoint::CorrectnessPassed,
            ForgeProposalEndpoint::EvaluationValid,
            ForgeProposalEndpoint::SelectedForContinuation,
        ] {
            assert_eq!(
                classify_endpoint(
                    &selected(),
                    ForgeProposalRowOutcome::SelectedNoCandidate,
                    endpoint,
                ),
                ForgeProposalEndpointValue::Observed(false)
            );
        }
    }

    #[test]
    fn interrupted_candidate_is_observed_distinct_but_downstream_censored() {
        assert_eq!(
            classify_endpoint(
                &selected(),
                ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::Interrupted),
                ForgeProposalEndpoint::DistinctCandidate,
            ),
            ForgeProposalEndpointValue::Observed(true)
        );
        for endpoint in [
            ForgeProposalEndpoint::CompilePassed,
            ForgeProposalEndpoint::CorrectnessPassed,
            ForgeProposalEndpoint::EvaluationValid,
            ForgeProposalEndpoint::SelectedForContinuation,
        ] {
            assert_eq!(
                classify_endpoint(
                    &selected(),
                    ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::Interrupted),
                    endpoint,
                ),
                ForgeProposalEndpointValue::Censored
            );
        }
    }

    #[test]
    fn terminal_outcomes_map_to_cumulative_stage_success() {
        assert_eq!(
            classify_endpoint(
                &selected(),
                ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::RejectedCorrectness),
                ForgeProposalEndpoint::CompilePassed,
            ),
            ForgeProposalEndpointValue::Observed(true)
        );
        assert_eq!(
            classify_endpoint(
                &selected(),
                ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::RejectedCorrectness),
                ForgeProposalEndpoint::CorrectnessPassed,
            ),
            ForgeProposalEndpointValue::Observed(false)
        );
        assert_eq!(
            classify_endpoint(
                &selected(),
                ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::ValidNotSelected),
                ForgeProposalEndpoint::EvaluationValid,
            ),
            ForgeProposalEndpointValue::Observed(true)
        );
        assert_eq!(
            classify_endpoint(
                &selected(),
                ForgeProposalRowOutcome::Concrete(ForgeTrialOutcome::ValidNotSelected),
                ForgeProposalEndpoint::SelectedForContinuation,
            ),
            ForgeProposalEndpointValue::Observed(false)
        );
    }
}
