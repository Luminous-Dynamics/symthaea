// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Machine-readable deviation records for frozen chemosensation protocols.
//!
//! Once an outcome-bearing protocol is frozen, changes that can affect results
//! should be represented explicitly rather than disappearing into commit history
//! or prose. A deviation record does not make a changed run confirmatory again;
//! its disposition states how the affected evidence must be treated.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpectedBiasDirection {
    /// Change is expected to favor a positive/confirmatory conclusion.
    TowardConfirmation,
    /// Change is expected to favor a negative/practical-failure conclusion.
    TowardFailure,
    /// No material directional effect is expected.
    Neutral,
    /// Direction cannot be established before re-analysis.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviationDisposition {
    /// Discard the affected outcome-bearing run and repeat under the frozen rules.
    RestartAffectedRun,
    /// Create/freeze a new protocol version before collecting new confirmatory data.
    NewProtocolVersion,
    /// Keep the affected result only as exploratory evidence.
    ExploratoryOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolDeviation {
    pub protocol_id: String,
    pub protocol_version: String,
    /// Event timestamp in microseconds in the experiment's recorded time basis.
    pub timestamp_us: u64,
    /// Run IDs affected by the change. May be empty when the deviation applies
    /// before any run begins or affects an entire protocol version.
    pub affected_run_ids: Vec<String>,
    /// Frozen protocol/configuration fields whose interpretation changed.
    pub affected_fields: Vec<String>,
    pub reason: String,
    /// Whether outcome-bearing values had already been inspected when the change
    /// was proposed or applied.
    pub outcomes_inspected: bool,
    pub expected_bias: ExpectedBiasDirection,
    pub disposition: DeviationDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolDeviationError {
    BlankProtocolId,
    BlankProtocolVersion,
    BlankReason,
    EmptyAffectedFields,
    BlankAffectedField,
    DuplicateAffectedField(String),
    BlankRunId,
    DuplicateRunId(String),
}

impl ProtocolDeviation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        protocol_id: impl Into<String>,
        protocol_version: impl Into<String>,
        timestamp_us: u64,
        affected_run_ids: Vec<String>,
        affected_fields: Vec<String>,
        reason: impl Into<String>,
        outcomes_inspected: bool,
        expected_bias: ExpectedBiasDirection,
        disposition: DeviationDisposition,
    ) -> Result<Self, ProtocolDeviationError> {
        let protocol_id = protocol_id.into();
        if protocol_id.trim().is_empty() {
            return Err(ProtocolDeviationError::BlankProtocolId);
        }
        let protocol_version = protocol_version.into();
        if protocol_version.trim().is_empty() {
            return Err(ProtocolDeviationError::BlankProtocolVersion);
        }
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(ProtocolDeviationError::BlankReason);
        }
        if affected_fields.is_empty() {
            return Err(ProtocolDeviationError::EmptyAffectedFields);
        }

        let mut seen_fields = BTreeSet::new();
        for field in &affected_fields {
            if field.trim().is_empty() {
                return Err(ProtocolDeviationError::BlankAffectedField);
            }
            if !seen_fields.insert(field.as_str()) {
                return Err(ProtocolDeviationError::DuplicateAffectedField(field.clone()));
            }
        }

        let mut seen_runs = BTreeSet::new();
        for run_id in &affected_run_ids {
            if run_id.trim().is_empty() {
                return Err(ProtocolDeviationError::BlankRunId);
            }
            if !seen_runs.insert(run_id.as_str()) {
                return Err(ProtocolDeviationError::DuplicateRunId(run_id.clone()));
            }
        }

        Ok(Self {
            protocol_id,
            protocol_version,
            timestamp_us,
            affected_run_ids,
            affected_fields,
            reason,
            outcomes_inspected,
            expected_bias,
            disposition,
        })
    }

    /// Conservative signal for downstream reporting: if outcomes were inspected,
    /// the affected result should never be silently presented as untouched
    /// preregistered confirmation evidence.
    pub fn requires_confirmatory_separation(&self) -> bool {
        self.outcomes_inspected
            || matches!(
                self.disposition,
                DeviationDisposition::NewProtocolVersion | DeviationDisposition::ExploratoryOnly
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deviation_preserves_outcome_inspection_and_disposition() {
        let deviation = ProtocolDeviation::new(
            "od001-v1",
            "1.0.0",
            42,
            vec!["run-17".into()],
            vec!["humidity_compensation".into()],
            "sensor firmware exposed a compensation defect",
            true,
            ExpectedBiasDirection::Unknown,
            DeviationDisposition::NewProtocolVersion,
        )
        .unwrap();

        assert!(deviation.outcomes_inspected);
        assert!(deviation.requires_confirmatory_separation());
        assert_eq!(deviation.affected_run_ids, vec!["run-17"]);
    }

    #[test]
    fn duplicate_or_blank_fields_are_rejected() {
        assert!(matches!(
            ProtocolDeviation::new(
                "od001-v1",
                "1.0.0",
                42,
                vec![],
                vec!["threshold".into(), "threshold".into()],
                "reason",
                false,
                ExpectedBiasDirection::Neutral,
                DeviationDisposition::RestartAffectedRun,
            ),
            Err(ProtocolDeviationError::DuplicateAffectedField(field)) if field == "threshold"
        ));

        assert!(matches!(
            ProtocolDeviation::new(
                "od001-v1",
                "1.0.0",
                42,
                vec![],
                vec![],
                "reason",
                false,
                ExpectedBiasDirection::Neutral,
                DeviationDisposition::RestartAffectedRun,
            ),
            Err(ProtocolDeviationError::EmptyAffectedFields)
        ));
    }

    #[test]
    fn uninspected_restart_can_remain_separate_without_forcing_new_version() {
        let deviation = ProtocolDeviation::new(
            "gt004-v1",
            "1.0.0",
            7,
            vec!["run-3".into()],
            vec!["pump_timeout".into()],
            "predeclared hardware timeout triggered before result inspection",
            false,
            ExpectedBiasDirection::Neutral,
            DeviationDisposition::RestartAffectedRun,
        )
        .unwrap();
        assert!(!deviation.requires_confirmatory_separation());
    }
}
