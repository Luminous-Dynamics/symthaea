//! Candidate-facing projection for evaluator-owned scenario records.
//!
//! This is a serialization/API boundary, not an OS process sandbox. Production
//! runners must serialize only CandidatePacketV1 to an isolated candidate
//! process and keep EvaluatorTicketBook in the trusted evaluator process.
//! Tickets must be fresh, uniformly random 128-bit values from an OS CSPRNG
//! for each evaluator run; this module validates their shape and collisions,
//! but deliberately does not implement randomness.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::manifest::{Scenario, ScenarioGroup, SplitRole};

pub const CANDIDATE_PACKET_SCHEMA_V1: &str =
    "luminous.decision-transfer.candidate-packet.v1";
const MAX_FRAMES: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateViewError {
    InvalidTicket,
    InvalidDigest,
    InvalidField(&'static str),
    DuplicateValue(&'static str),
    NonFiniteValue(&'static str),
    NonIncreasingObservationTime,
    ObservationAfterAsOfTime,
    DuplicateTicket,
}

impl fmt::Display for CandidateViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTicket => write!(formatter, "candidate ticket must be 32 lowercase hexadecimal characters"),
            Self::InvalidDigest => write!(formatter, "task contract digest must be sha256 followed by 64 lowercase hexadecimal characters"),
            Self::InvalidField(field) => write!(formatter, "invalid candidate packet field: {field}"),
            Self::DuplicateValue(field) => write!(formatter, "duplicate candidate packet value: {field}"),
            Self::NonFiniteValue(field) => write!(formatter, "{field} must be finite"),
            Self::NonIncreasingObservationTime => write!(formatter, "observation timestamps must be strictly increasing"),
            Self::ObservationAfterAsOfTime => write!(formatter, "candidate packet contains an observation after its as-of time"),
            Self::DuplicateTicket => write!(formatter, "candidate ticket is already bound in this evaluator run"),
        }
    }
}

impl Error for CandidateViewError {}

/// One candidate-visible observation row. Missing channels are explicit None
/// values; evaluator truth, latent state and split metadata are not represented.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CandidateObservationFrame {
    pub timestamp: f64,
    pub values: Vec<Option<f64>>,
}

/// The only scenario object intended for serialization into a candidate
/// process. It contains causal observations and the declared task interface,
/// but no split role, scenario/group IDs, seed commitment, reference digest,
/// solver qualification digest, or evaluator-only truth.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CandidatePacketV1 {
    pub schema: String,
    pub ticket: String,
    pub task_contract_digest: String,
    pub time_unit: String,
    pub as_of_time: f64,
    pub channel_names: Vec<String>,
    pub allowed_action_ids: Vec<String>,
    pub observations: Vec<CandidateObservationFrame>,
}

impl CandidatePacketV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ticket: String,
        task_contract_digest: String,
        time_unit: String,
        as_of_time: f64,
        channel_names: Vec<String>,
        allowed_action_ids: Vec<String>,
        observations: Vec<CandidateObservationFrame>,
    ) -> Result<Self, CandidateViewError> {
        let packet = Self {
            schema: CANDIDATE_PACKET_SCHEMA_V1.to_owned(),
            ticket,
            task_contract_digest,
            time_unit,
            as_of_time,
            channel_names,
            allowed_action_ids,
            observations,
        };
        packet.validate()?;
        Ok(packet)
    }

    pub fn validate(&self) -> Result<(), CandidateViewError> {
        if self.schema != CANDIDATE_PACKET_SCHEMA_V1 {
            return Err(CandidateViewError::InvalidField("schema"));
        }
        if !is_lower_hex(&self.ticket, 32) {
            return Err(CandidateViewError::InvalidTicket);
        }
        if !is_sha256(&self.task_contract_digest) {
            return Err(CandidateViewError::InvalidDigest);
        }
        if self.time_unit.trim().is_empty() {
            return Err(CandidateViewError::InvalidField("time_unit"));
        }
        if !self.as_of_time.is_finite() {
            return Err(CandidateViewError::NonFiniteValue("as_of_time"));
        }
        if self.channel_names.is_empty() || self.channel_names.len() > 4096 {
            return Err(CandidateViewError::InvalidField("channel_names"));
        }
        let mut channels = BTreeSet::new();
        for channel in &self.channel_names {
            if channel.trim().is_empty() || !channels.insert(channel) {
                return Err(CandidateViewError::DuplicateValue("channel_names"));
            }
        }
        if self.allowed_action_ids.is_empty() || self.allowed_action_ids.len() > 4096 {
            return Err(CandidateViewError::InvalidField("allowed_action_ids"));
        }
        let mut actions = BTreeSet::new();
        for action in &self.allowed_action_ids {
            if !is_safe_label(action) || !actions.insert(action) {
                return Err(CandidateViewError::DuplicateValue("allowed_action_ids"));
            }
        }
        if self.observations.is_empty() || self.observations.len() > MAX_FRAMES {
            return Err(CandidateViewError::InvalidField("observations"));
        }

        let mut previous_time = f64::NEG_INFINITY;
        for frame in &self.observations {
            if !frame.timestamp.is_finite() {
                return Err(CandidateViewError::NonFiniteValue("observation timestamp"));
            }
            if frame.timestamp <= previous_time {
                return Err(CandidateViewError::NonIncreasingObservationTime);
            }
            if frame.timestamp > self.as_of_time {
                return Err(CandidateViewError::ObservationAfterAsOfTime);
            }
            if frame.values.len() != self.channel_names.len() {
                return Err(CandidateViewError::InvalidField("observation channel width"));
            }
            if frame.values.iter().flatten().any(|value| !value.is_finite()) {
                return Err(CandidateViewError::NonFiniteValue("observation value"));
            }
            previous_time = frame.timestamp;
        }
        Ok(())
    }
}

/// Trusted evaluator-only scenario association. It intentionally does not
/// implement Serialize and is never nested in CandidatePacketV1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluatorOnlyBinding {
    scenario_id: String,
    split_role: SplitRole,
    groups: Vec<ScenarioGroup>,
    seed_commitment: String,
    reference_data_digest: String,
    solver_qualification_digest: String,
}

impl EvaluatorOnlyBinding {
    pub fn scenario_id(&self) -> &str {
        &self.scenario_id
    }

    pub fn split_role(&self) -> SplitRole {
        self.split_role
    }

    pub fn groups(&self) -> &[ScenarioGroup] {
        &self.groups
    }

    pub fn seed_commitment(&self) -> &str {
        &self.seed_commitment
    }

    pub fn reference_data_digest(&self) -> &str {
        &self.reference_data_digest
    }

    pub fn solver_qualification_digest(&self) -> &str {
        &self.solver_qualification_digest
    }
}

/// Evaluator-owned ticket association. Do not serialize or pass this book to
/// candidate code. The ticket is one-use within this book; callers must create
/// a new CSPRNG ticket for each scenario in each evaluator run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvaluatorTicketBook {
    bindings: BTreeMap<String, EvaluatorOnlyBinding>,
}

impl EvaluatorTicketBook {
    pub fn bind(
        &mut self,
        packet: &CandidatePacketV1,
        scenario: &Scenario,
    ) -> Result<(), CandidateViewError> {
        packet.validate()?;
        if self.bindings.contains_key(&packet.ticket) {
            return Err(CandidateViewError::DuplicateTicket);
        }
        let binding = EvaluatorOnlyBinding {
            scenario_id: scenario.scenario_id.clone(),
            split_role: scenario.split_role,
            groups: scenario.groups.clone(),
            seed_commitment: scenario.seed_commitment.clone(),
            reference_data_digest: scenario.reference_data_digest.clone(),
            solver_qualification_digest: scenario.solver_qualification_digest.clone(),
        };
        self.bindings.insert(packet.ticket.clone(), binding);
        Ok(())
    }

    pub fn binding_for(&self, ticket: &str) -> Option<&EvaluatorOnlyBinding> {
        self.bindings.get(ticket)
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_sha256(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|digest| is_lower_hex(digest, 64))
}

fn is_safe_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ScenarioGroup;

    const TICKET: &str = "0123456789abcdef0123456789abcdef";
    const DIGEST: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn scenario() -> Scenario {
        Scenario {
            scenario_id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1".to_owned(),
            split_role: SplitRole::ConfirmatoryTest,
            groups: vec![
                ScenarioGroup {
                    axis: crate::manifest::SplitAxisKind::EquationFamily,
                    group_id: "11111111111111111111111111111111".to_owned(),
                },
                ScenarioGroup {
                    axis: crate::manifest::SplitAxisKind::ParameterCell,
                    group_id: "22222222222222222222222222222222".to_owned(),
                },
            ],
            seed_commitment: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
            reference_data_digest: "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
            solver_qualification_digest: "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_owned(),
        }
    }

    fn packet(ticket: &str) -> CandidatePacketV1 {
        CandidatePacketV1::new(
            ticket.to_owned(),
            DIGEST.to_owned(),
            "seconds".to_owned(),
            2.0,
            vec!["sensor_x".to_owned(), "sensor_y".to_owned()],
            vec!["hold".to_owned(), "intervene".to_owned()],
            vec![
                CandidateObservationFrame {
                    timestamp: 1.0,
                    values: vec![Some(0.2), Some(0.4)],
                },
                CandidateObservationFrame {
                    timestamp: 2.0,
                    values: vec![Some(0.3), None],
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn candidate_serialization_excludes_evaluator_only_fields() {
        let mut book = EvaluatorTicketBook::default();
        let scenario = scenario();
        let packet = packet(TICKET);
        book.bind(&packet, &scenario).unwrap();

        let json = serde_json::to_value(&packet).unwrap();
        let object = json.as_object().unwrap();
        for field in [
            "scenario_id",
            "split_role",
            "groups",
            "seed_commitment",
            "reference_data_digest",
            "solver_qualification_digest",
            "latent_state",
            "ground_truth",
            "intervention_label",
        ] {
            assert!(!object.contains_key(field), "candidate JSON leaked {field}");
        }
        assert_eq!(book.binding_for(TICKET).unwrap().split_role(), SplitRole::ConfirmatoryTest);
        assert_eq!(book.binding_for(TICKET).unwrap().scenario_id(), scenario.scenario_id);
    }

    #[test]
    fn rejects_observations_after_as_of_time() {
        let mut packet = packet(TICKET);
        packet.observations[1].timestamp = 2.1;
        assert_eq!(packet.validate(), Err(CandidateViewError::ObservationAfterAsOfTime));
    }

    #[test]
    fn rejects_duplicate_channels_and_action_ids() {
        let mut packet = packet(TICKET);
        packet.channel_names[1] = packet.channel_names[0].clone();
        assert_eq!(packet.validate(), Err(CandidateViewError::DuplicateValue("channel_names")));

        let mut packet = packet(TICKET);
        packet.allowed_action_ids[1] = packet.allowed_action_ids[0].clone();
        assert_eq!(packet.validate(), Err(CandidateViewError::DuplicateValue("allowed_action_ids")));
    }

    #[test]
    fn rejects_unknown_candidate_packet_fields() {
        let packet = packet(TICKET);
        let mut value = serde_json::to_value(packet).unwrap();
        value.as_object_mut().unwrap().insert("split_role".to_owned(), serde_json::json!("confirmatory_test"));
        assert!(serde_json::from_value::<CandidatePacketV1>(value).is_err());
    }

    #[test]
    fn evaluator_ticket_book_rejects_ticket_reuse() {
        let scenario = scenario();
        let packet = packet(TICKET);
        let mut book = EvaluatorTicketBook::default();
        book.bind(&packet, &scenario).unwrap();
        assert_eq!(book.bind(&packet, &scenario), Err(CandidateViewError::DuplicateTicket));
        assert_eq!(book.len(), 1);
    }

    #[test]
    fn candidate_packet_rejects_invalid_digest_and_nonfinite_values() {
        let mut packet = packet(TICKET);
        packet.task_contract_digest = "sha256:abc".to_owned();
        assert_eq!(packet.validate(), Err(CandidateViewError::InvalidDigest));

        let mut packet = packet(TICKET);
        packet.observations[0].values[0] = Some(f64::NAN);
        assert_eq!(packet.validate(), Err(CandidateViewError::NonFiniteValue("observation value")));
    }
}
