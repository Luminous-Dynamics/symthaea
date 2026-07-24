// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded command-response attribution without post-hoc causal overclaiming.
//!
//! A compatible delay and sign are necessary but not sufficient for causal
//! attribution. Overlapping candidate causes remain ambiguous, and the module
//! never upgrades correlation into proof.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const CAUSAL_ATTRIBUTION_SCHEMA_VERSION: u16 = 1;
pub const MAX_PENDING_CAUSES: usize = 32;
pub const MAX_ATTRIBUTION_RECORDS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpectedResponseSign {
    Increase,
    Decrease,
    AnyChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CommandCause {
    pub cause_id: u64,
    pub command_step: u64,
    pub earliest_effect_step: u64,
    pub latest_effect_step: u64,
    pub channel: u8,
    pub expected_sign: ExpectedResponseSign,
    pub minimum_magnitude: f64,
    pub command_digest: u64,
}

impl CommandCause {
    pub fn validate(self) -> bool {
        self.cause_id > 0
            && self.command_step <= self.earliest_effect_step
            && self.earliest_effect_step <= self.latest_effect_step
            && self.minimum_magnitude.is_finite()
            && self.minimum_magnitude >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResponseObservation {
    pub observation_id: u64,
    pub observed_step: u64,
    pub channel: u8,
    pub delta: f64,
    pub confidence: f64,
    pub confounder_count: u8,
}

impl ResponseObservation {
    pub fn validate(self) -> bool {
        self.observation_id > 0
            && self.delta.is_finite()
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributionDisposition {
    Supported,
    Ambiguous,
    Contradicted,
    Unattributed,
}

impl AttributionDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Ambiguous => "ambiguous",
            Self::Contradicted => "contradicted",
            Self::Unattributed => "unattributed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionRecord {
    pub observation_id: u64,
    pub observed_step: u64,
    pub channel: u8,
    pub disposition: AttributionDisposition,
    pub candidate_causes: Vec<u64>,
    pub selected_cause: Option<u64>,
    pub delta: f64,
    pub confounder_count: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalAttributionLedger {
    schema_version: u16,
    pending: VecDeque<CommandCause>,
    records: VecDeque<AttributionRecord>,
    dropped_causes: u64,
    dropped_records: u64,
    contradictions: u64,
    ambiguities: u64,
}

impl Default for CausalAttributionLedger {
    fn default() -> Self {
        Self {
            schema_version: CAUSAL_ATTRIBUTION_SCHEMA_VERSION,
            pending: VecDeque::with_capacity(MAX_PENDING_CAUSES),
            records: VecDeque::with_capacity(MAX_ATTRIBUTION_RECORDS),
            dropped_causes: 0,
            dropped_records: 0,
            contradictions: 0,
            ambiguities: 0,
        }
    }
}

impl CausalAttributionLedger {
    pub fn validate(&self) -> bool {
        self.schema_version == CAUSAL_ATTRIBUTION_SCHEMA_VERSION
            && self.pending.len() <= MAX_PENDING_CAUSES
            && self.records.len() <= MAX_ATTRIBUTION_RECORDS
            && self.pending.iter().all(|cause| cause.validate())
            && self.records.iter().all(|record| {
                record.delta.is_finite()
                    && record.candidate_causes.len() <= MAX_PENDING_CAUSES
                    && record
                        .selected_cause
                        .is_none_or(|id| record.candidate_causes.contains(&id))
            })
    }

    pub fn register_cause(&mut self, cause: CommandCause) -> bool {
        if !cause.validate()
            || self
                .pending
                .iter()
                .any(|candidate| candidate.cause_id == cause.cause_id)
        {
            return false;
        }
        if self.pending.len() == MAX_PENDING_CAUSES {
            self.pending.pop_front();
            self.dropped_causes = self.dropped_causes.saturating_add(1);
        }
        self.pending.push_back(cause);
        true
    }

    pub fn attribute(&mut self, observation: ResponseObservation) -> AttributionRecord {
        let mut candidates = Vec::new();
        let mut compatible = Vec::new();
        if observation.validate() {
            for cause in &self.pending {
                if cause.channel == observation.channel
                    && observation.observed_step >= cause.earliest_effect_step
                    && observation.observed_step <= cause.latest_effect_step
                {
                    candidates.push(cause.cause_id);
                    if sign_matches(*cause, observation.delta) {
                        compatible.push(cause.cause_id);
                    }
                }
            }
        }
        let (disposition, selected_cause) = if !observation.validate() || candidates.is_empty() {
            (AttributionDisposition::Unattributed, None)
        } else if compatible.is_empty() {
            self.contradictions = self.contradictions.saturating_add(1);
            (AttributionDisposition::Contradicted, None)
        } else if compatible.len() > 1 || observation.confounder_count > 0 {
            self.ambiguities = self.ambiguities.saturating_add(1);
            (AttributionDisposition::Ambiguous, None)
        } else {
            (AttributionDisposition::Supported, compatible.first().copied())
        };
        let record = AttributionRecord {
            observation_id: observation.observation_id,
            observed_step: observation.observed_step,
            channel: observation.channel,
            disposition,
            candidate_causes: candidates,
            selected_cause,
            delta: observation.delta,
            confounder_count: observation.confounder_count,
        };
        if self.records.len() == MAX_ATTRIBUTION_RECORDS {
            self.records.pop_front();
            self.dropped_records = self.dropped_records.saturating_add(1);
        }
        self.records.push_back(record.clone());
        self.pending
            .retain(|cause| cause.latest_effect_step >= observation.observed_step);
        record
    }

    pub fn records(&self) -> Vec<AttributionRecord> {
        self.records.iter().cloned().collect()
    }

    pub const fn contradictions(&self) -> u64 {
        self.contradictions
    }

    pub const fn ambiguities(&self) -> u64 {
        self.ambiguities
    }
}

fn sign_matches(cause: CommandCause, delta: f64) -> bool {
    if delta.abs() < cause.minimum_magnitude {
        return false;
    }
    match cause.expected_sign {
        ExpectedResponseSign::Increase => delta > 0.0,
        ExpectedResponseSign::Decrease => delta < 0.0,
        ExpectedResponseSign::AnyChange => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cause(id: u64) -> CommandCause {
        CommandCause {
            cause_id: id,
            command_step: 1,
            earliest_effect_step: 2,
            latest_effect_step: 4,
            channel: 4,
            expected_sign: ExpectedResponseSign::Increase,
            minimum_magnitude: 0.1,
            command_digest: id,
        }
    }

    #[test]
    fn overlapping_causes_are_not_claimed_as_unique() {
        let mut ledger = CausalAttributionLedger::default();
        assert!(ledger.register_cause(cause(1)));
        assert!(ledger.register_cause(cause(2)));
        let result = ledger.attribute(ResponseObservation {
            observation_id: 1,
            observed_step: 3,
            channel: 4,
            delta: 0.5,
            confidence: 0.9,
            confounder_count: 0,
        });
        assert_eq!(result.disposition, AttributionDisposition::Ambiguous);
        assert_eq!(result.selected_cause, None);
    }

    #[test]
    fn opposite_response_is_recorded_as_contradiction() {
        let mut ledger = CausalAttributionLedger::default();
        ledger.register_cause(cause(1));
        let result = ledger.attribute(ResponseObservation {
            observation_id: 1,
            observed_step: 3,
            channel: 4,
            delta: -0.5,
            confidence: 0.9,
            confounder_count: 0,
        });
        assert_eq!(result.disposition, AttributionDisposition::Contradicted);
        assert_eq!(ledger.contradictions(), 1);
    }
}
