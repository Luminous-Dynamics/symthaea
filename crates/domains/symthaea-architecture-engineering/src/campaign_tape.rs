//! Shared, replayable action tape for matched-budget direction-search campaigns.
//!
//! The tape makes action costs, policy visibility, claims, verifier direction mapping and
//! evidence outcomes explicit. It is an integrity structure, not an authentication boundary:
//! callers still need a separately trusted verifier and authenticated receipt delivery.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const TAPE_SCHEMA: &str = "symthaea.architecture-engineering.campaign-tape";
const TAPE_VERSION: &str = "0.1.0";
const EVENT_DOMAIN: &[u8] = b"symthaea.campaign-event.v0.1\0";
const TAPE_DOMAIN: &[u8] = b"symthaea.campaign-tape.v0.1\0";

/// Hard limits are independent dimensions. None means no configured cap, not zero usage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceBudget {
    pub max_action_units: u64,
    pub max_wall_millis: Option<u64>,
    pub max_cpu_millis: Option<u64>,
    pub max_gpu_millis: Option<u64>,
    pub max_host_memory_peak_mib: Option<u64>,
    pub max_gpu_memory_peak_mib: Option<u64>,
    pub max_energy_mwh: Option<u64>,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub max_tool_calls: Option<u64>,
    pub max_evaluator_calls: Option<u64>,
    pub max_verifier_calls: Option<u64>,
}

impl ResourceBudget {
    pub fn validate(&self) -> Result<(), CampaignTapeError> {
        if self.max_action_units == 0 {
            return Err(CampaignTapeError::new(
                "config.resource_budget.max_action_units",
                "must be non-zero",
            ));
        }
        Ok(())
    }
}

/// Per-action usage receipt. Unknown measurements remain None throughout aggregation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceUsage {
    pub action_units: u64,
    pub wall_millis: Option<u64>,
    pub cpu_millis: Option<u64>,
    pub gpu_millis: Option<u64>,
    /// Per-worker peak during this action. Parallel workers in a round are summed
    /// conservatively; the tape reports the maximum such round total.
    pub host_memory_peak_mib: Option<u64>,
    /// Per-worker peak during this action. Parallel workers in a round are summed
    /// conservatively; the tape reports the maximum such round total.
    pub gpu_memory_peak_mib: Option<u64>,
    pub energy_mwh: Option<u64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub tool_calls: Option<u64>,
    pub evaluator_calls: Option<u64>,
    pub verifier_calls: Option<u64>,
}

impl ResourceUsage {
    fn validate(&self) -> Result<(), CampaignTapeError> {
        if self.action_units == 0 {
            return Err(CampaignTapeError::new(
                "event.resource_usage.action_units",
                "must be positive for every charged action",
            ));
        }
        Ok(())
    }
}

/// Aggregated cost. An additive dimension becomes unknown if any action omitted it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceTotals {
    pub action_units: u64,
    pub wall_millis: Option<u64>,
    pub cpu_millis: Option<u64>,
    pub gpu_millis: Option<u64>,
    pub host_memory_peak_mib: Option<u64>,
    pub gpu_memory_peak_mib: Option<u64>,
    pub energy_mwh: Option<u64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub tool_calls: Option<u64>,
    pub evaluator_calls: Option<u64>,
    pub verifier_calls: Option<u64>,
}

/// Immutable comparison envelope shared by every policy in a matched campaign.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignTapeConfig {
    pub task_recipe_sha256: String,
    pub evaluator_sha256: String,
    pub comparison_profile_sha256: String,
    /// Digest of the evaluator-owned candidate/proposal stream supplied to all policies.
    pub candidate_stream_sha256: String,
    pub worker_count: u16,
    pub first_round_direction_claim_cap: u16,
    pub steady_state_direction_claim_cap: u16,
    pub new_direction_claim_cap: u16,
    pub initial_direction_ids: Vec<String>,
    pub resource_budget: ResourceBudget,
}

impl CampaignTapeConfig {
    pub fn validate(&self) -> Result<(), CampaignTapeError> {
        for (field, value) in [
            ("task_recipe_sha256", self.task_recipe_sha256.as_str()),
            ("evaluator_sha256", self.evaluator_sha256.as_str()),
            ("comparison_profile_sha256", self.comparison_profile_sha256.as_str()),
            ("candidate_stream_sha256", self.candidate_stream_sha256.as_str()),
        ] {
            validate_sha256(field, value)?;
        }
        if self.worker_count == 0 {
            return Err(CampaignTapeError::new("config.worker_count", "must be non-zero"));
        }
        if self.first_round_direction_claim_cap == 0
            || self.steady_state_direction_claim_cap == 0
            || self.new_direction_claim_cap == 0
        {
            return Err(CampaignTapeError::new(
                "config.claim_caps",
                "all claim caps must be non-zero",
            ));
        }
        let mut directions = BTreeSet::new();
        for (index, direction_id) in self.initial_direction_ids.iter().enumerate() {
            validate_id(&format!("config.initial_direction_ids[{index}]"), direction_id)?;
            if !directions.insert(direction_id) {
                return Err(CampaignTapeError::new(
                    format!("config.initial_direction_ids[{index}]"),
                    "duplicate initial direction ID",
                ));
            }
        }
        if self.initial_direction_ids.is_empty() {
            return Err(CampaignTapeError::new(
                "config.initial_direction_ids",
                "at least one initial direction is required",
            ));
        }
        self.resource_budget.validate()
    }
}

/// Policy assignment for a charged worker action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "arm", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArmClaim {
    Direction { direction_id: String },
    NewDirection,
}

/// Outcome of a candidate execution. Only Verified may carry a score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateDisposition {
    Verified,
    Rejected,
    Inconclusive,
    Crashed,
    TimedOut,
}

/// Verifier's canonical mapping for a direction proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case", deny_unknown_fields)]
pub enum DirectionProposalDisposition {
    Admitted { canonical_direction_id: String },
    Duplicate { existing_direction_id: String },
    Rejected { reason_code: String },
    Inconclusive { reason_code: String },
}

/// Evaluator-authored event payload. Direction identity in a proposal is never self-certified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event_kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CampaignEventPayload {
    CandidateEvaluation {
        candidate_sha256: String,
        parent_candidate_sha256: String,
        semantic_delta_sha256: String,
        direction_id: String,
        disposition: CandidateDisposition,
        score: Option<f64>,
    },
    DirectionProposal {
        proposal_sha256: String,
        descriptor_sha256: String,
        parent_candidate_sha256: Option<String>,
        disposition: DirectionProposalDisposition,
    },
    /// One paid candidate evaluation that also asks the verifier to admit or map its
    /// proposed direction. This is the correct representation for Coral-style NEW agents:
    /// proposal and scored attempt are one work unit, not two separately charged actions.
    NovelCandidateEvaluation {
        candidate_sha256: String,
        parent_candidate_sha256: String,
        semantic_delta_sha256: String,
        proposal_sha256: String,
        descriptor_sha256: String,
        disposition: DirectionProposalDisposition,
        score: Option<f64>,
    },
}

/// One action and its visibility/accounting context in the append-only campaign tape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignEvent {
    pub event_id: String,
    pub sequence_number: u64,
    pub round_id: u64,
    pub worker_id: String,
    /// Maximum prior event sequence visible to this policy action. All actions in a
    /// batch share a cutoff that precedes the first event in that batch.
    pub policy_visible_through_sequence: u64,
    pub claimed_arm: ArmClaim,
    pub task_recipe_sha256: String,
    pub evaluator_sha256: String,
    pub comparison_profile_sha256: String,
    pub payload: CampaignEventPayload,
    pub resource_usage: ResourceUsage,
    pub previous_event_sha256: String,
    pub event_sha256: String,
}

impl CampaignEvent {
    fn calculate_digest(&self) -> Result<String, CampaignTapeError> {
        let mut unsigned = self.clone();
        unsigned.event_sha256.clear();
        let bytes = serde_json::to_vec(&unsigned)
            .map_err(|error| CampaignTapeError::new("event", error.to_string()))?;
        Ok(domain_digest(EVENT_DOMAIN, &bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TapeDigestPayload<'a> {
    schema: &'a str,
    version: &'a str,
    tape_id: &'a str,
    policy_id: &'a str,
    policy_config_sha256: &'a str,
    config: &'a CampaignTapeConfig,
    events: &'a [CampaignEvent],
}

/// A content-addressed action tape for one policy, one fixed task and one resource envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignTape {
    schema: String,
    version: String,
    pub tape_id: String,
    pub policy_id: String,
    /// Hash of the exact scheduler parameters/config, distinct across compared policies.
    pub policy_config_sha256: String,
    pub config: CampaignTapeConfig,
    pub events: Vec<CampaignEvent>,
    pub tape_sha256: String,
}

impl CampaignTape {
    pub fn new(
        tape_id: impl Into<String>,
        policy_id: impl Into<String>,
        policy_config_sha256: impl Into<String>,
        config: CampaignTapeConfig,
    ) -> Result<Self, CampaignTapeError> {
        config.validate()?;
        let tape_id = tape_id.into();
        let policy_id = policy_id.into();
        let policy_config_sha256 = policy_config_sha256.into();
        validate_id("tape_id", &tape_id)?;
        validate_id("policy_id", &policy_id)?;
        validate_sha256("policy_config_sha256", &policy_config_sha256)?;
        Ok(Self {
            schema: TAPE_SCHEMA.to_owned(),
            version: TAPE_VERSION.to_owned(),
            tape_id,
            policy_id,
            policy_config_sha256,
            config,
            events: Vec::new(),
            tape_sha256: String::new(),
        })
    }

    /// Append an action. Sequence and hash-chain links are assigned by the tape, not the caller.
    pub fn append(&mut self, mut event: CampaignEvent) -> Result<(), CampaignTapeError> {
        if !self.tape_sha256.is_empty() {
            return Err(CampaignTapeError::new("tape", "sealed tape is immutable"));
        }
        if event.sequence_number != 0 || !event.previous_event_sha256.is_empty()
            || !event.event_sha256.is_empty()
        {
            return Err(CampaignTapeError::new(
                "event",
                "new events must have sequence zero and empty chain/digest fields",
            ));
        }
        event.sequence_number = self.events.len() as u64 + 1;
        event.previous_event_sha256 = self.events.last()
            .map(|previous| previous.event_sha256.clone())
            .unwrap_or_default();
        event.event_sha256 = event.calculate_digest()?;
        self.events.push(event);
        Ok(())
    }

    /// Seal the full event chain and tape header. Validation is performed before sealing.
    pub fn seal(&mut self) -> Result<String, CampaignTapeError> {
        if !self.tape_sha256.is_empty() {
            return self.validate_integrity().map(|()| self.tape_sha256.clone());
        }
        self.validate_events()?;
        self.tape_sha256 = self.calculate_tape_digest()?;
        if let Err(error) = self.validate_integrity() {
            self.tape_sha256.clear();
            return Err(error);
        }
        Ok(self.tape_sha256.clone())
    }

    pub fn totals(&self) -> Result<ResourceTotals, CampaignTapeError> {
        self.validate_events()?;
        aggregate_resource_usage(&self.events)
    }

    pub fn validate_integrity(&self) -> Result<(), CampaignTapeError> {
        if self.schema != TAPE_SCHEMA || self.version != TAPE_VERSION {
            return Err(CampaignTapeError::new("tape", "unsupported schema/version"));
        }
        validate_id("tape_id", &self.tape_id)?;
        validate_id("policy_id", &self.policy_id)?;
        validate_sha256("policy_config_sha256", &self.policy_config_sha256)?;
        self.config.validate()?;
        if self.events.is_empty() {
            return Err(CampaignTapeError::new("events", "tape must contain at least one action"));
        }
        self.validate_events()?;
        validate_sha256("tape_sha256", &self.tape_sha256)?;
        let expected = self.calculate_tape_digest()?;
        if expected != self.tape_sha256 {
            return Err(CampaignTapeError::new(
                "tape_sha256",
                "does not match the tape header and event chain",
            ));
        }
        Ok(())
    }

    fn calculate_tape_digest(&self) -> Result<String, CampaignTapeError> {
        let payload = TapeDigestPayload {
            schema: &self.schema,
            version: &self.version,
            tape_id: &self.tape_id,
            policy_id: &self.policy_id,
            policy_config_sha256: &self.policy_config_sha256,
            config: &self.config,
            events: &self.events,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|error| CampaignTapeError::new("tape", error.to_string()))?;
        Ok(domain_digest(TAPE_DOMAIN, &bytes))
    }

    fn validate_events(&self) -> Result<(), CampaignTapeError> {
        if self.events.is_empty() {
            return Err(CampaignTapeError::new("events", "tape must contain at least one action"));
        }
        let mut event_ids = BTreeSet::new();
        let mut previous_digest = String::new();
        let mut expected_sequence = 1_u64;
        let mut previous_round = 0_u64;
        let mut known_directions: BTreeSet<String> =
            self.config.initial_direction_ids.iter().cloned().collect();
        let mut index = 0usize;

        while index < self.events.len() {
            let round_id = self.events[index].round_id;
            if round_id == 0 || round_id != previous_round + 1 {
                return Err(CampaignTapeError::new(
                    format!("events[{index}].round_id"),
                    "round IDs must start at one and advance contiguously",
                ));
            }
            let round_start = index;
            while index < self.events.len() && self.events[index].round_id == round_id {
                index += 1;
            }
            let round = &self.events[round_start..index];
            if round.len() > self.config.worker_count as usize {
                return Err(CampaignTapeError::new(
                    format!("events[{round_start}..{index}]"),
                    "round contains more actions than configured workers",
                ));
            }
            let visibility_cutoff = round[0].policy_visible_through_sequence;
            if visibility_cutoff != round[0].sequence_number - 1 {
                return Err(CampaignTapeError::new(
                    format!("events[{round_start}].policy_visible_through_sequence"),
                    "a round must observe all completed prior actions and none from its own batch",
                ));
            }
            let mut worker_ids = BTreeSet::new();
            let mut direction_claims: BTreeMap<String, usize> = BTreeMap::new();
            let mut new_claims = 0usize;
            let mut pending_new_directions = BTreeSet::new();

            for (offset, event) in round.iter().enumerate() {
                let event_index = round_start + offset;
                if event.sequence_number != expected_sequence {
                    return Err(CampaignTapeError::new(
                        format!("events[{event_index}].sequence_number"),
                        "sequence numbers must be contiguous and start at one",
                    ));
                }
                expected_sequence = expected_sequence.checked_add(1).ok_or_else(|| {
                    CampaignTapeError::new("events.sequence_number", "overflow")
                })?;
                if !event_ids.insert(&event.event_id) {
                    return Err(CampaignTapeError::new(
                        format!("events[{event_index}].event_id"),
                        "duplicate event ID",
                    ));
                }
                validate_id(&format!("events[{event_index}].event_id"), &event.event_id)?;
                validate_id(&format!("events[{event_index}].worker_id"), &event.worker_id)?;
                if !worker_ids.insert(&event.worker_id) {
                    return Err(CampaignTapeError::new(
                        format!("events[{event_index}].worker_id"),
                        "a worker may claim at most one action per round",
                    ));
                }
                if event.policy_visible_through_sequence != visibility_cutoff {
                    return Err(CampaignTapeError::new(
                        format!("events[{event_index}].policy_visible_through_sequence"),
                        "all actions in a batch must share one observation cutoff",
                    ));
                }
                if visibility_cutoff > event.sequence_number - 1 {
                    return Err(CampaignTapeError::new(
                        format!("events[{event_index}].policy_visible_through_sequence"),
                        "observation cutoff points beyond completed prior events",
                    ));
                }
                if event.previous_event_sha256 != previous_digest {
                    return Err(CampaignTapeError::new(
                        format!("events[{event_index}].previous_event_sha256"),
                        "hash-chain predecessor mismatch",
                    ));
                }
                validate_sha256(
                    &format!("events[{event_index}].event_sha256"),
                    &event.event_sha256,
                )?;
                let expected_digest = event.calculate_digest()?;
                if expected_digest != event.event_sha256 {
                    return Err(CampaignTapeError::new(
                        format!("events[{event_index}].event_sha256"),
                        "event digest does not match event content",
                    ));
                }
                previous_digest.clone_from(&event.event_sha256);
                validate_sha256(
                    &format!("events[{event_index}].task_recipe_sha256"),
                    &event.task_recipe_sha256,
                )?;
                validate_sha256(
                    &format!("events[{event_index}].evaluator_sha256"),
                    &event.evaluator_sha256,
                )?;
                validate_sha256(
                    &format!("events[{event_index}].comparison_profile_sha256"),
                    &event.comparison_profile_sha256,
                )?;
                if event.task_recipe_sha256 != self.config.task_recipe_sha256
                    || event.evaluator_sha256 != self.config.evaluator_sha256
                    || event.comparison_profile_sha256 != self.config.comparison_profile_sha256
                {
                    return Err(CampaignTapeError::new(
                        format!("events[{event_index}]"),
                        "event task/evaluator/profile differs from the tape envelope",
                    ));
                }
                event.resource_usage.validate()?;

                match (&event.claimed_arm, &event.payload) {
                    (
                        ArmClaim::Direction { direction_id },
                        CampaignEventPayload::CandidateEvaluation {
                            candidate_sha256,
                            parent_candidate_sha256,
                            semantic_delta_sha256,
                            direction_id: payload_direction,
                            disposition,
                            score,
                        },
                    ) => {
                        validate_id(
                            &format!("events[{event_index}].claimed_arm.direction_id"),
                            direction_id,
                        )?;
                        if direction_id != payload_direction || !known_directions.contains(direction_id) {
                            return Err(CampaignTapeError::new(
                                format!("events[{event_index}].direction_id"),
                                "evaluation direction must match a previously verifier-known direction",
                            ));
                        }
                        validate_sha256("candidate_sha256", candidate_sha256)?;
                        validate_sha256("parent_candidate_sha256", parent_candidate_sha256)?;
                        validate_sha256("semantic_delta_sha256", semantic_delta_sha256)?;
                        match (disposition, score) {
                            (CandidateDisposition::Verified, Some(value)) if value.is_finite() => {}
                            (CandidateDisposition::Verified, _) => {
                                return Err(CampaignTapeError::new(
                                    format!("events[{event_index}].score"),
                                    "verified evaluation requires a finite score",
                                ));
                            }
                            (_, None) => {}
                            (_, Some(_)) => {
                                return Err(CampaignTapeError::new(
                                    format!("events[{event_index}].score"),
                                    "non-verified evaluation cannot publish a score",
                                ));
                            }
                        }
                        *direction_claims.entry(direction_id.clone()).or_default() += 1;
                    }
                    (
                        ArmClaim::NewDirection,
                        CampaignEventPayload::NovelCandidateEvaluation {
                            candidate_sha256,
                            parent_candidate_sha256,
                            semantic_delta_sha256,
                            proposal_sha256,
                            descriptor_sha256,
                            disposition,
                            score,
                        },
                    ) => {
                        validate_sha256("candidate_sha256", candidate_sha256)?;
                        validate_sha256("parent_candidate_sha256", parent_candidate_sha256)?;
                        validate_sha256("semantic_delta_sha256", semantic_delta_sha256)?;
                        validate_sha256("proposal_sha256", proposal_sha256)?;
                        validate_sha256("descriptor_sha256", descriptor_sha256)?;
                        match (disposition, score) {
                            (
                                DirectionProposalDisposition::Admitted { .. }
                                | DirectionProposalDisposition::Duplicate { .. },
                                Some(value),
                            ) if value.is_finite() => {}
                            (
                                DirectionProposalDisposition::Admitted { .. }
                                | DirectionProposalDisposition::Duplicate { .. },
                                _,
                            ) => {
                                return Err(CampaignTapeError::new(
                                    format!("events[{event_index}].score"),
                                    "admitted or duplicate novel candidate requires a finite verified score",
                                ));
                            }
                            (
                                DirectionProposalDisposition::Rejected { .. }
                                | DirectionProposalDisposition::Inconclusive { .. },
                                None,
                            ) => {}
                            (
                                DirectionProposalDisposition::Rejected { .. }
                                | DirectionProposalDisposition::Inconclusive { .. },
                                Some(_),
                            ) => {
                                return Err(CampaignTapeError::new(
                                    format!("events[{event_index}].score"),
                                    "rejected or inconclusive novel candidate cannot publish a score",
                                ));
                            }
                        }
                        new_claims += 1;
                        match disposition {
                            DirectionProposalDisposition::Admitted { canonical_direction_id } => {
                                validate_id("canonical_direction_id", canonical_direction_id)?;
                                if known_directions.contains(canonical_direction_id)
                                    || !pending_new_directions.insert(canonical_direction_id.clone())
                                {
                                    return Err(CampaignTapeError::new(
                                        format!("events[{event_index}].disposition"),
                                        "new admission must create a unique canonical direction",
                                    ));
                                }
                                *direction_claims.entry(canonical_direction_id.clone()).or_default() += 1;
                            }
                            DirectionProposalDisposition::Duplicate { existing_direction_id } => {
                                validate_id("existing_direction_id", existing_direction_id)?;
                                if !known_directions.contains(existing_direction_id)
                                    && !pending_new_directions.contains(existing_direction_id)
                                {
                                    return Err(CampaignTapeError::new(
                                        format!("events[{event_index}].disposition"),
                                        "duplicate must map to an already known or same-round admitted direction",
                                    ));
                                }
                                *direction_claims.entry(existing_direction_id.clone()).or_default() += 1;
                            }
                            DirectionProposalDisposition::Rejected { reason_code }
                            | DirectionProposalDisposition::Inconclusive { reason_code } => {
                                validate_id("reason_code", reason_code)?;
                            }
                        }
                    }
                    (
                        ArmClaim::NewDirection,
                        CampaignEventPayload::DirectionProposal {
                            proposal_sha256,
                            descriptor_sha256,
                            parent_candidate_sha256,
                            disposition,
                        },
                    ) => {
                        validate_sha256("proposal_sha256", proposal_sha256)?;
                        validate_sha256("descriptor_sha256", descriptor_sha256)?;
                        if let Some(parent) = parent_candidate_sha256 {
                            validate_sha256("parent_candidate_sha256", parent)?;
                        }
                        new_claims += 1;
                        match disposition {
                            DirectionProposalDisposition::Admitted { canonical_direction_id } => {
                                validate_id("canonical_direction_id", canonical_direction_id)?;
                                if known_directions.contains(canonical_direction_id)
                                    || !pending_new_directions.insert(canonical_direction_id.clone())
                                {
                                    return Err(CampaignTapeError::new(
                                        format!("events[{event_index}].disposition"),
                                        "new admission must create a unique canonical direction",
                                    ));
                                }
                            }
                            DirectionProposalDisposition::Duplicate { existing_direction_id } => {
                                validate_id("existing_direction_id", existing_direction_id)?;
                                if !known_directions.contains(existing_direction_id)
                                    && !pending_new_directions.contains(existing_direction_id)
                                {
                                    return Err(CampaignTapeError::new(
                                        format!("events[{event_index}].disposition"),
                                        "duplicate must map to an already known or same-round admitted direction",
                                    ));
                                }
                            }
                            DirectionProposalDisposition::Rejected { reason_code }
                            | DirectionProposalDisposition::Inconclusive { reason_code } => {
                                validate_id("reason_code", reason_code)?;
                            }
                        }
                    }
                    _ => {
                        return Err(CampaignTapeError::new(
                            format!("events[{event_index}].payload"),
                            "candidate evaluations must claim a direction; proposals must claim NEW",
                        ));
                    }
                }
            }

            let direction_cap = if round_id == 1 {
                self.config.first_round_direction_claim_cap
            } else {
                self.config.steady_state_direction_claim_cap
            } as usize;
            for (direction_id, claims) in direction_claims {
                if claims > direction_cap {
                    return Err(CampaignTapeError::new(
                        format!("round[{round_id}].claims[{direction_id}]"),
                        "direction claim cap exceeded",
                    ));
                }
            }
            if new_claims > self.config.new_direction_claim_cap as usize {
                return Err(CampaignTapeError::new(
                    format!("round[{round_id}].new_claims"),
                    "NEW-direction claim cap exceeded",
                ));
            }
            known_directions.extend(pending_new_directions);
            previous_round = round_id;
        }

        let totals = aggregate_resource_usage(&self.events)?;
        validate_budget(&totals, &self.config.resource_budget)
    }
}

/// Ensure all compared policies share the immutable problem, candidate stream and resource
/// envelope. Each policy must still have its own distinct tape and policy ID.
pub fn validate_matched_tape_headers(tapes: &[CampaignTape]) -> Result<(), CampaignTapeError> {
    let first = tapes.first().ok_or_else(|| {
        CampaignTapeError::new("tapes", "at least one campaign tape is required")
    })?;
    first.config.validate()?;
    first.validate_integrity()?;
    let mut tape_ids = BTreeSet::new();
    let mut policy_ids = BTreeSet::new();
    for (index, tape) in tapes.iter().enumerate() {
        tape.validate_integrity()?;
        if tape.config != first.config {
            return Err(CampaignTapeError::new(
                format!("tapes[{index}].config"),
                "matched policies must use identical task, evaluator, candidate stream, worker and resource envelopes",
            ));
        }
        if !tape_ids.insert(&tape.tape_id) || !policy_ids.insert(&tape.policy_id) {
            return Err(CampaignTapeError::new(
                format!("tapes[{index}]"),
                "tape IDs and policy IDs must be unique in a matched comparison",
            ));
        }
    }
    Ok(())
}

fn aggregate_resource_usage(events: &[CampaignEvent]) -> Result<ResourceTotals, CampaignTapeError> {
    if events.is_empty() {
        return Err(CampaignTapeError::new("events", "cannot aggregate an empty tape"));
    }
    let mut totals = ResourceTotals {
        action_units: 0,
        wall_millis: Some(0),
        cpu_millis: Some(0),
        gpu_millis: Some(0),
        host_memory_peak_mib: None,
        gpu_memory_peak_mib: None,
        energy_mwh: Some(0),
        input_tokens: Some(0),
        output_tokens: Some(0),
        tool_calls: Some(0),
        evaluator_calls: Some(0),
        verifier_calls: Some(0),
    };
    let mut rounds: BTreeMap<u64, Vec<&ResourceUsage>> = BTreeMap::new();
    for event in events {
        let usage = &event.resource_usage;
        totals.action_units = totals.action_units.checked_add(usage.action_units)
            .ok_or_else(|| CampaignTapeError::new("totals.action_units", "overflow"))?;
        add_optional(&mut totals.wall_millis, usage.wall_millis, "wall_millis")?;
        add_optional(&mut totals.cpu_millis, usage.cpu_millis, "cpu_millis")?;
        add_optional(&mut totals.gpu_millis, usage.gpu_millis, "gpu_millis")?;
        add_optional(&mut totals.energy_mwh, usage.energy_mwh, "energy_mwh")?;
        add_optional(&mut totals.input_tokens, usage.input_tokens, "input_tokens")?;
        add_optional(&mut totals.output_tokens, usage.output_tokens, "output_tokens")?;
        add_optional(&mut totals.tool_calls, usage.tool_calls, "tool_calls")?;
        add_optional(&mut totals.evaluator_calls, usage.evaluator_calls, "evaluator_calls")?;
        add_optional(&mut totals.verifier_calls, usage.verifier_calls, "verifier_calls")?;
        rounds.entry(event.round_id).or_default().push(usage);
    }
    totals.host_memory_peak_mib = peak_round_memory(&rounds, |usage| usage.host_memory_peak_mib)?;
    totals.gpu_memory_peak_mib = peak_round_memory(&rounds, |usage| usage.gpu_memory_peak_mib)?;
    Ok(totals)
}

fn add_optional(
    total: &mut Option<u64>,
    value: Option<u64>,
    field: &str,
) -> Result<(), CampaignTapeError> {
    *total = match (*total, value) {
        (Some(left), Some(right)) => Some(left.checked_add(right).ok_or_else(|| {
            CampaignTapeError::new(format!("totals.{field}"), "resource sum overflow")
        })?),
        _ => None,
    };
    Ok(())
}

fn peak_round_memory(
    rounds: &BTreeMap<u64, Vec<&ResourceUsage>>,
    field: impl Fn(&ResourceUsage) -> Option<u64>,
) -> Result<Option<u64>, CampaignTapeError> {
    let mut peak = 0_u64;
    for (round_id, workers) in rounds {
        let mut round_total = Some(0_u64);
        for usage in workers {
            let value = field(usage);
            round_total = match (round_total, value) {
                (Some(left), Some(right)) => Some(left.checked_add(right).ok_or_else(|| {
                    CampaignTapeError::new(
                        format!("round[{round_id}].memory_peak"),
                        "concurrent worker memory sum overflow",
                    )
                })?),
                _ => None,
            };
        }
        let Some(round_total) = round_total else {
            return Ok(None);
        };
        peak = peak.max(round_total);
    }
    Ok(Some(peak))
}

fn validate_budget(
    totals: &ResourceTotals,
    budget: &ResourceBudget,
) -> Result<(), CampaignTapeError> {
    if totals.action_units > budget.max_action_units {
        return Err(CampaignTapeError::new(
            "resource_budget.action_units",
            "hard action budget exceeded",
        ));
    }
    for (name, observed, limit) in [
        ("wall_millis", totals.wall_millis, budget.max_wall_millis),
        ("cpu_millis", totals.cpu_millis, budget.max_cpu_millis),
        ("gpu_millis", totals.gpu_millis, budget.max_gpu_millis),
        ("host_memory_peak_mib", totals.host_memory_peak_mib, budget.max_host_memory_peak_mib),
        ("gpu_memory_peak_mib", totals.gpu_memory_peak_mib, budget.max_gpu_memory_peak_mib),
        ("energy_mwh", totals.energy_mwh, budget.max_energy_mwh),
        ("input_tokens", totals.input_tokens, budget.max_input_tokens),
        ("output_tokens", totals.output_tokens, budget.max_output_tokens),
        ("tool_calls", totals.tool_calls, budget.max_tool_calls),
        ("evaluator_calls", totals.evaluator_calls, budget.max_evaluator_calls),
        ("verifier_calls", totals.verifier_calls, budget.max_verifier_calls),
    ] {
        if let Some(limit) = limit {
            let observed = observed.ok_or_else(|| CampaignTapeError::new(
                format!("resource_budget.{name}"),
                "usage is unknown under an active hard cap",
            ))?;
            if observed > limit {
                return Err(CampaignTapeError::new(
                    format!("resource_budget.{name}"),
                    "hard resource cap exceeded",
                ));
            }
        }
    }
    Ok(())
}

fn validate_id(field: &str, value: &str) -> Result<(), CampaignTapeError> {
    if value.trim().is_empty() || value != value.trim() {
        return Err(CampaignTapeError::new(field, "must be non-empty and already trimmed"));
    }
    Ok(())
}

fn validate_sha256(field: &str, value: &str) -> Result<(), CampaignTapeError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CampaignTapeError::new(field, "must be a 64-character hexadecimal SHA-256"));
    }
    Ok(())
}

fn domain_digest(domain: &[u8], payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(payload);
    let bytes = hasher.finalize();
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Validation failure with a stable field path and an explicit reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignTapeError {
    pub path: String,
    pub message: String,
}

impl CampaignTapeError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for CampaignTapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for CampaignTapeError {}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn config() -> CampaignTapeConfig {
        CampaignTapeConfig {
            task_recipe_sha256: A.to_owned(),
            evaluator_sha256: B.to_owned(),
            comparison_profile_sha256: C.to_owned(),
            candidate_stream_sha256: D.to_owned(),
            worker_count: 2,
            first_round_direction_claim_cap: 1,
            steady_state_direction_claim_cap: 2,
            new_direction_claim_cap: 1,
            initial_direction_ids: vec!["A".to_owned(), "B".to_owned(), "C".to_owned()],
            resource_budget: ResourceBudget {
                max_action_units: 10,
                max_wall_millis: Some(1_000),
                max_cpu_millis: Some(2_000),
                max_gpu_millis: None,
                max_host_memory_peak_mib: Some(8_192),
                max_gpu_memory_peak_mib: None,
                max_energy_mwh: None,
                max_input_tokens: None,
                max_output_tokens: None,
                max_tool_calls: None,
                max_evaluator_calls: None,
                max_verifier_calls: None,
            },
        }
    }

    fn usage() -> ResourceUsage {
        ResourceUsage {
            action_units: 1,
            wall_millis: Some(10),
            cpu_millis: Some(20),
            gpu_millis: None,
            host_memory_peak_mib: Some(100),
            gpu_memory_peak_mib: None,
            energy_mwh: Some(1),
            input_tokens: Some(10),
            output_tokens: Some(5),
            tool_calls: Some(1),
            evaluator_calls: Some(1),
            verifier_calls: Some(1),
        }
    }

    fn evaluation_event(event_id: &str, round: u64, worker: &str, cutoff: u64, direction: &str) -> CampaignEvent {
        CampaignEvent {
            event_id: event_id.to_owned(),
            sequence_number: 0,
            round_id: round,
            worker_id: worker.to_owned(),
            policy_visible_through_sequence: cutoff,
            claimed_arm: ArmClaim::Direction { direction_id: direction.to_owned() },
            task_recipe_sha256: A.to_owned(),
            evaluator_sha256: B.to_owned(),
            comparison_profile_sha256: C.to_owned(),
            payload: CampaignEventPayload::CandidateEvaluation {
                candidate_sha256: D.to_owned(),
                parent_candidate_sha256: A.to_owned(),
                semantic_delta_sha256: B.to_owned(),
                direction_id: direction.to_owned(),
                disposition: CandidateDisposition::Verified,
                score: Some(0.75),
            },
            resource_usage: usage(),
            previous_event_sha256: String::new(),
            event_sha256: String::new(),
        }
    }

    fn proposal_event(event_id: &str, round: u64, worker: &str, cutoff: u64, disposition: DirectionProposalDisposition) -> CampaignEvent {
        CampaignEvent {
            event_id: event_id.to_owned(),
            sequence_number: 0,
            round_id: round,
            worker_id: worker.to_owned(),
            policy_visible_through_sequence: cutoff,
            claimed_arm: ArmClaim::NewDirection,
            task_recipe_sha256: A.to_owned(),
            evaluator_sha256: B.to_owned(),
            comparison_profile_sha256: C.to_owned(),
            payload: CampaignEventPayload::DirectionProposal {
                proposal_sha256: D.to_owned(),
                descriptor_sha256: A.to_owned(),
                parent_candidate_sha256: Some(B.to_owned()),
                disposition,
            },
            resource_usage: usage(),
            previous_event_sha256: String::new(),
            event_sha256: String::new(),
        }
    }

    fn novel_candidate_event(
        event_id: &str,
        round: u64,
        worker: &str,
        cutoff: u64,
        disposition: DirectionProposalDisposition,
        score: Option<f64>,
    ) -> CampaignEvent {
        CampaignEvent {
            event_id: event_id.to_owned(),
            sequence_number: 0,
            round_id: round,
            worker_id: worker.to_owned(),
            policy_visible_through_sequence: cutoff,
            claimed_arm: ArmClaim::NewDirection,
            task_recipe_sha256: A.to_owned(),
            evaluator_sha256: B.to_owned(),
            comparison_profile_sha256: C.to_owned(),
            payload: CampaignEventPayload::NovelCandidateEvaluation {
                candidate_sha256: D.to_owned(),
                parent_candidate_sha256: A.to_owned(),
                semantic_delta_sha256: B.to_owned(),
                proposal_sha256: C.to_owned(),
                descriptor_sha256: D.to_owned(),
                disposition,
                score,
            },
            resource_usage: usage(),
            previous_event_sha256: String::new(),
            event_sha256: String::new(),
        }
    }

    fn novel_candidate_event(
        event_id: &str,
        round: u64,
        worker: &str,
        cutoff: u64,
        disposition: DirectionProposalDisposition,
        score: Option<f64>,
    ) -> CampaignEvent {
        CampaignEvent {
            event_id: event_id.to_owned(),
            sequence_number: 0,
            round_id: round,
            worker_id: worker.to_owned(),
            policy_visible_through_sequence: cutoff,
            claimed_arm: ArmClaim::NewDirection,
            task_recipe_sha256: A.to_owned(),
            evaluator_sha256: B.to_owned(),
            comparison_profile_sha256: C.to_owned(),
            payload: CampaignEventPayload::NovelCandidateEvaluation {
                candidate_sha256: D.to_owned(),
                parent_candidate_sha256: A.to_owned(),
                semantic_delta_sha256: B.to_owned(),
                proposal_sha256: C.to_owned(),
                descriptor_sha256: D.to_owned(),
                disposition,
                score,
            },
            resource_usage: usage(),
            previous_event_sha256: String::new(),
            event_sha256: String::new(),
        }
    }

    fn tape(policy_id: &str) -> CampaignTape {
        CampaignTape::new(
            format!("tape-{policy_id}"),
            policy_id,
            A,
            config(),
        ).unwrap()
    }

    #[test]
    fn sealed_tape_round_trips_integrity_and_chains_events() {
        let mut tape = tape("ucb");
        tape.append(evaluation_event("e1", 1, "w1", 0, "A")).unwrap();
        tape.append(evaluation_event("e2", 2, "w1", 1, "A")).unwrap();
        let digest = tape.seal().unwrap();
        assert_eq!(digest.len(), 64);
        assert_eq!(tape.events[1].previous_event_sha256, tape.events[0].event_sha256);
        tape.validate_integrity().unwrap();
    }

    #[test]
    fn event_content_mutation_is_detected() {
        let mut tape = tape("greedy");
        tape.append(evaluation_event("e1", 1, "w1", 0, "A")).unwrap();
        tape.seal().unwrap();
        if let CampaignEventPayload::CandidateEvaluation { score, .. } = &mut tape.events[0].payload {
            *score = Some(0.99);
        }
        assert!(tape.validate_integrity().is_err());
    }

    #[test]
    fn verified_and_failed_evaluations_have_distinct_score_contracts() {
        let mut tape = tape("greedy");
        let mut failed = evaluation_event("e1", 1, "w1", 0, "A");
        if let CampaignEventPayload::CandidateEvaluation { disposition, score, .. } = &mut failed.payload {
            *disposition = CandidateDisposition::Crashed;
            *score = Some(0.2);
        }
        tape.append(failed).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn same_round_batch_cannot_see_other_batch_results() {
        let mut tape = tape("batch");
        tape.append(evaluation_event("e1", 1, "w1", 0, "A")).unwrap();
        let second = evaluation_event("e2", 1, "w2", 1, "B");
        tape.append(second).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn one_worker_cannot_claim_two_actions_in_one_round() {
        let mut tape = tape("workers");
        tape.append(evaluation_event("e1", 1, "w1", 0, "A")).unwrap();
        tape.append(evaluation_event("e2", 1, "w1", 0, "B")).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn a_new_direction_is_not_evaluable_until_the_next_round() {
        let mut tape = tape("new-direction");
        tape.append(proposal_event(
            "e1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Admitted { canonical_direction_id: "D".to_owned() },
        )).unwrap();
        tape.append(evaluation_event("e2", 1, "w2", 0, "D")).unwrap();
        assert!(tape.seal().is_err());

        let mut valid = tape("new-direction-next-round");
        valid.append(proposal_event(
            "p1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Admitted { canonical_direction_id: "D".to_owned() },
        )).unwrap();
        valid.append(evaluation_event("v1", 2, "w1", 1, "D")).unwrap();
        valid.seal().unwrap();
    }

    #[test]
    fn duplicate_proposal_must_map_to_a_known_canonical_direction() {
        let mut tape = tape("duplicate");
        tape.append(proposal_event(
            "p1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Duplicate { existing_direction_id: "missing".to_owned() },
        )).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn same_round_duplicate_can_map_to_a_direction_admitted_in_that_batch() {
        let mut batch_config = config();
        batch_config.new_direction_claim_cap = 2;
        let mut tape = CampaignTape::new("tape-same-round-duplicate", "same-round-duplicate", A, batch_config).unwrap();
        tape.append(proposal_event(
            "p1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Admitted { canonical_direction_id: "D".to_owned() },
        )).unwrap();
        tape.append(proposal_event(
            "p2",
            1,
            "w2",
            0,
            DirectionProposalDisposition::Duplicate { existing_direction_id: "D".to_owned() },
        )).unwrap();
        tape.seal().unwrap();
    }

    #[test]
    fn unknown_resource_under_hard_cap_fails_closed() {
        let mut tape = tape("unknown-cost");
        let mut event = evaluation_event("e1", 1, "w1", 0, "A");
        event.resource_usage.cpu_millis = None;
        tape.append(event).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn unknown_unbounded_resource_remains_unknown_not_zero() {
        let mut tape = tape("unknown-unbounded");
        tape.append(evaluation_event("e1", 1, "w1", 0, "A")).unwrap();
        tape.seal().unwrap();
        let totals = tape.totals().unwrap();
        assert_eq!(totals.gpu_millis, None);
        assert_eq!(totals.gpu_memory_peak_mib, None);
    }

    #[test]
    fn concurrent_memory_is_summed_within_round_then_maximized_across_rounds() {
        let mut tape = tape("memory");
        tape.append(evaluation_event("r1w1", 1, "w1", 0, "A")).unwrap();
        let mut second = evaluation_event("r1w2", 1, "w2", 0, "B");
        second.resource_usage.host_memory_peak_mib = Some(200);
        tape.append(second).unwrap();
        let mut third = evaluation_event("r2w1", 2, "w1", 2, "A");
        third.resource_usage.host_memory_peak_mib = Some(250);
        tape.append(third).unwrap();
        tape.seal().unwrap();
        assert_eq!(tape.totals().unwrap().host_memory_peak_mib, Some(300));
    }

    #[test]
    fn matched_tapes_require_identical_resource_and_candidate_envelopes() {
        let mut left = tape("left");
        left.append(evaluation_event("l1", 1, "w1", 0, "A")).unwrap();
        left.seal().unwrap();
        let mut right = tape("right");
        right.append(evaluation_event("r1", 1, "w1", 0, "A")).unwrap();
        right.seal().unwrap();
        validate_matched_tape_headers(&[left.clone(), right.clone()]).unwrap();

        right.config.candidate_stream_sha256 = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_owned();
        right.tape_sha256.clear();
        right.seal().unwrap();
        assert!(validate_matched_tape_headers(&[left, right]).is_err());
    }

    #[test]
    fn proposal_and_evaluation_cannot_swap_claim_arm_types() {
        let mut tape = tape("claim-types");
        let mut event = proposal_event(
            "p1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Rejected { reason_code: "not_novel".to_owned() },
        );
        event.claimed_arm = ArmClaim::Direction { direction_id: "A".to_owned() };
        tape.append(event).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn configured_direction_claim_cap_is_enforced() {
        let mut tape = tape("claims");
        tape.append(evaluation_event("e1", 1, "w1", 0, "A")).unwrap();
        tape.append(evaluation_event("e2", 1, "w2", 0, "A")).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn resource_sum_overflow_is_rejected_not_downgraded_to_unknown() {
        let mut tape = tape("overflow");
        let mut first = evaluation_event("o1", 1, "w1", 0, "A");
        first.resource_usage.wall_millis = Some(u64::MAX);
        tape.append(first).unwrap();
        let mut second = evaluation_event("o2", 2, "w1", 1, "B");
        second.resource_usage.wall_millis = Some(1);
        tape.append(second).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn novel_candidate_admission_is_one_charged_action_and_direction_opens_next_round() {
        let mut tape = tape("novel-admit");
        tape.append(novel_candidate_event(
            "n1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Admitted {
                canonical_direction_id: "D".to_owned(),
            },
            Some(0.81),
        )).unwrap();
        tape.append(evaluation_event("e2", 2, "w1", 1, "D")).unwrap();
        tape.seal().unwrap();
        let totals = tape.totals().unwrap();
        assert_eq!(totals.action_units, 2);
        assert_eq!(tape.events.len(), 2);
    }

    #[test]
    fn novel_candidate_duplicate_maps_to_existing_arm_with_one_charge() {
        let mut tape = tape("novel-duplicate");
        tape.append(novel_candidate_event(
            "n1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Duplicate {
                existing_direction_id: "A".to_owned(),
            },
            Some(0.77),
        )).unwrap();
        tape.seal().unwrap();
        assert_eq!(tape.totals().unwrap().action_units, 1);
    }

    #[test]
    fn novel_candidate_admission_and_duplicate_can_share_a_batch() {
        let mut batch_config = config();
        batch_config.new_direction_claim_cap = 2;
        let mut tape = CampaignTape::new(
            "tape-novel-same-batch",
            "novel_same_batch",
            A,
            batch_config,
        ).unwrap();
        tape.append(novel_candidate_event(
            "n1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Admitted {
                canonical_direction_id: "D".to_owned(),
            },
            Some(0.80),
        )).unwrap();
        tape.append(novel_candidate_event(
            "n2",
            1,
            "w2",
            0,
            DirectionProposalDisposition::Duplicate {
                existing_direction_id: "D".to_owned(),
            },
            Some(0.82),
        )).unwrap();
        tape.seal().unwrap();
        assert_eq!(tape.totals().unwrap().action_units, 2);
    }

    #[test]
    fn duplicate_novel_candidate_counts_toward_the_existing_direction_claim_cap() {
        let mut campaign_config = config();
        campaign_config.worker_count = 3;
        let mut tape = CampaignTape::new(
            "tape-novel-duplicate-cap",
            "novel_duplicate_cap",
            A,
            campaign_config,
        ).unwrap();
        tape.append(evaluation_event("e1", 1, "bootstrap", 0, "A")).unwrap();
        tape.append(evaluation_event("e2", 2, "w1", 1, "A")).unwrap();
        tape.append(evaluation_event("e3", 2, "w2", 1, "A")).unwrap();
        tape.append(novel_candidate_event(
            "n4",
            2,
            "w3",
            1,
            DirectionProposalDisposition::Duplicate {
                existing_direction_id: "A".to_owned(),
            },
            Some(0.78),
        )).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn rejected_and_inconclusive_novel_candidates_cannot_publish_a_score() {
        for (policy, disposition) in [
            ("novel-rejected", DirectionProposalDisposition::Rejected {
                reason_code: "not_novel".to_owned(),
            }),
            ("novel-inconclusive", DirectionProposalDisposition::Inconclusive {
                reason_code: "insufficient_evidence".to_owned(),
            }),
        ] {
            let mut tape = tape(policy);
            tape.append(novel_candidate_event(
                "n1", 1, "w1", 0, disposition, None,
            )).unwrap();
            tape.seal().unwrap();
            assert_eq!(tape.totals().unwrap().action_units, 1);
        }
    }

    #[test]
    fn rejected_novel_candidate_with_score_is_invalid() {
        let mut tape = tape("novel-rejected-with-score");
        tape.append(novel_candidate_event(
            "n1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Rejected {
                reason_code: "not_novel".to_owned(),
            },
            Some(0.95),
        )).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn novel_candidate_admission_is_one_charged_action_and_direction_opens_next_round() {
        let mut tape = tape("novel-admit");
        tape.append(novel_candidate_event(
            "n1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Admitted {
                canonical_direction_id: "D".to_owned(),
            },
            Some(0.81),
        )).unwrap();
        tape.append(evaluation_event("e2", 2, "w1", 1, "D")).unwrap();
        tape.seal().unwrap();
        let totals = tape.totals().unwrap();
        assert_eq!(totals.action_units, 2);
        assert_eq!(tape.events.len(), 2);
    }

    #[test]
    fn novel_candidate_duplicate_maps_to_existing_arm_with_one_charge() {
        let mut tape = tape("novel-duplicate");
        tape.append(novel_candidate_event(
            "n1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Duplicate {
                existing_direction_id: "A".to_owned(),
            },
            Some(0.77),
        )).unwrap();
        tape.seal().unwrap();
        assert_eq!(tape.totals().unwrap().action_units, 1);
    }

    #[test]
    fn novel_candidate_admission_and_duplicate_can_share_a_batch() {
        let mut batch_config = config();
        batch_config.new_direction_claim_cap = 2;
        let mut tape = CampaignTape::new(
            "tape-novel-same-batch",
            "novel_same_batch",
            A,
            batch_config,
        ).unwrap();
        tape.append(novel_candidate_event(
            "n1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Admitted {
                canonical_direction_id: "D".to_owned(),
            },
            Some(0.80),
        )).unwrap();
        tape.append(novel_candidate_event(
            "n2",
            1,
            "w2",
            0,
            DirectionProposalDisposition::Duplicate {
                existing_direction_id: "D".to_owned(),
            },
            Some(0.82),
        )).unwrap();
        tape.seal().unwrap();
        assert_eq!(tape.totals().unwrap().action_units, 2);
    }

    #[test]
    fn rejected_and_inconclusive_novel_candidates_cannot_publish_a_score() {
        for (policy, disposition) in [
            ("novel-rejected", DirectionProposalDisposition::Rejected {
                reason_code: "not_novel".to_owned(),
            }),
            ("novel-inconclusive", DirectionProposalDisposition::Inconclusive {
                reason_code: "insufficient_evidence".to_owned(),
            }),
        ] {
            let mut tape = tape(policy);
            tape.append(novel_candidate_event(
                "n1", 1, "w1", 0, disposition, None,
            )).unwrap();
            tape.seal().unwrap();
            assert_eq!(tape.totals().unwrap().action_units, 1);
        }
    }

    #[test]
    fn rejected_novel_candidate_with_score_is_invalid() {
        let mut tape = tape("novel-rejected-with-score");
        tape.append(novel_candidate_event(
            "n1",
            1,
            "w1",
            0,
            DirectionProposalDisposition::Rejected {
                reason_code: "not_novel".to_owned(),
            },
            Some(0.95),
        )).unwrap();
        assert!(tape.seal().is_err());
    }

    #[test]
    fn hash_chain_detects_event_reordering() {
        let mut tape = tape("reorder");
        tape.append(evaluation_event("e1", 1, "w1", 0, "A")).unwrap();
        tape.append(evaluation_event("e2", 2, "w1", 1, "B")).unwrap();
        tape.seal().unwrap();
        tape.events.swap(0, 1);
        assert!(tape.validate_integrity().is_err());
    }
}
