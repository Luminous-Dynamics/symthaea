//! Four-seat execution adapter for the published Coral direction-board scheduler.
//!
//! The scheduler remains advisory: this module records actual worker claims and validates
//! seats, crowding caps and correspondence with evaluator-owned batch events. It does not
//! force agents to follow the allocation, and it is not a claim that all Coral agent runtime
//! orchestration details have been reproduced.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::coral_baseline::{
    CoralAllocation, CoralArm, CoralBaselineConfig, CoralDirectionBoard,
};
use super::event_tape::{
    ActionIntent, ComparisonMode, EventTape, ExecutionContract, ExecutionMode,
};

const SCHEMA: &str = "symthaea.architecture-engineering.coral-four-seat-round";
const VERSION: &str = "0.1.0";
pub const CORAL_AGENT_COUNT: u32 = 4;
pub const CORAL_MAX_AGENTS_PER_DIRECTION: u32 = 2;
pub const CORAL_FIRST_ROUND_AGENTS_PER_DIRECTION: u32 = 1;

/// Stable error for a Coral seat-round contract violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoralExecutionError {
    pub path: String,
    pub message: String,
}

impl CoralExecutionError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for CoralExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for CoralExecutionError {}

/// The action a worker actually claimed after consulting the advisory board.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoralSeatClaim {
    pub worker_id: u32,
    pub event_id: String,
    pub arm: CoralArm,
}

/// One four-seat round, frozen against a pre-round Coral board snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoralSeatRound {
    pub schema: String,
    pub version: String,
    pub round_id: u64,
    pub batch_label: String,
    /// Content-derived batch ID; event-tape events must use this exact value.
    pub batch_id: String,
    pub source_board_sha256: String,
    /// Formula-level advisory allocation captured before workers made their choices.
    pub advisory_allocations: Vec<CoralAllocation>,
    /// Actual worker claims. These are not rewritten to match the advisory allocation.
    pub claims: Vec<CoralSeatClaim>,
    pub round_sha256: String,
}

#[derive(Serialize)]
struct UnsignedRound<'a> {
    schema: &'a str,
    version: &'a str,
    round_id: u64,
    batch_label: &'a str,
    source_board_sha256: &'a str,
    advisory_allocations: &'a [CoralAllocation],
    claims: &'a [CoralSeatClaim],
}

/// The fixed execution semantics used by the four-seat Coral reference adapter.
///
/// Keep this separate from CoralBaselineConfig: the latter defines the published formula,
/// whereas this contract defines worker/round execution and event-tape batch semantics.
pub fn coral_four_seat_execution_contract() -> ExecutionContract {
    ExecutionContract {
        mode: ExecutionMode::Batched,
        worker_count: CORAL_AGENT_COUNT,
        seats_per_round: CORAL_AGENT_COUNT,
        first_round_claim_cap: Some(CORAL_FIRST_ROUND_AGENTS_PER_DIRECTION),
        steady_state_claim_cap: Some(CORAL_MAX_AGENTS_PER_DIRECTION),
    }
}

impl CoralSeatRound {
    /// Freeze a four-agent round from a pre-round board snapshot and actual worker claims.
    ///
    /// The board recommendation is captured, but it is not treated as mandatory. Workers may
    /// choose different available arms; only seat completeness and published direction caps
    /// are enforced here.
    pub fn new(
        source_board: &CoralDirectionBoard,
        round_id: u64,
        batch_id: &str,
        claims: Vec<CoralSeatClaim>,
    ) -> Result<Self, CoralExecutionError> {
        validate_reference_board(source_board)?;
        if round_id == 0 {
            return Err(CoralExecutionError::new("round_id", "round IDs begin at one"));
        }
        if batch_id.trim().is_empty() {
            return Err(CoralExecutionError::new("batch_id", "must be non-empty"));
        }
        let advisory_allocations = source_board
            .recommend()
            .map_err(|error| CoralExecutionError::new("advisory_allocations", error.to_string()))?;
        let source_board_sha256 = digest_serializable(source_board)?;
        let mut round = Self {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            round_id,
            batch_label: batch_id.to_owned(),
            batch_id: String::new(),
            source_board_sha256,
            advisory_allocations,
            claims,
            round_sha256: String::new(),
        };
        round.round_sha256 = round.compute_digest()?;
        round.batch_id = round.derived_batch_id();
        round.validate_against_board(source_board)?;
        Ok(round)
    }

    /// Validate the frozen board snapshot, advisory allocation and actual seat claims.
    pub fn validate_against_board(
        &self,
        source_board: &CoralDirectionBoard,
    ) -> Result<(), CoralExecutionError> {
        validate_reference_board(source_board)?;
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(CoralExecutionError::new(
                "schema",
                "unsupported Coral seat-round schema/version",
            ));
        }
        if self.round_id == 0 || self.batch_label.trim().is_empty() {
            return Err(CoralExecutionError::new(
                "round",
                "round ID and batch label are required",
            ));
        }
        validate_sha256(&self.source_board_sha256, "source_board_sha256")?;
        if digest_serializable(source_board)? != self.source_board_sha256 {
            return Err(CoralExecutionError::new(
                "source_board_sha256",
                "pre-round Coral board snapshot does not match the frozen round",
            ));
        }
        let expected_advisory = source_board
            .recommend()
            .map_err(|error| CoralExecutionError::new("advisory_allocations", error.to_string()))?;
        if expected_advisory != self.advisory_allocations {
            return Err(CoralExecutionError::new(
                "advisory_allocations",
                "captured advisory allocation differs from the source board recommendation",
            ));
        }
        validate_seat_claims(source_board, self.round_id, &self.claims)?;
        validate_sha256(&self.round_sha256, "round_sha256")?;
        if self.compute_digest()? != self.round_sha256 {
            return Err(CoralExecutionError::new(
                "round_sha256",
                "round digest does not match its frozen claims and advisory allocation",
            ));
        }
        if self.batch_id != self.derived_batch_id() {
            return Err(CoralExecutionError::new(
                "batch_id",
                "batch ID does not bind the frozen round digest",
            ));
        }
        Ok(())
    }

    /// Check every worker claim against exactly one event in its event-tape batch.
    ///
    /// The supplied board is the immutable pre-round snapshot used when the claims were made.
    /// EventTape performs the independent prefix-visibility and four-seat schedule checks.
    pub fn validate_event_batch(
        &self,
        source_board: &CoralDirectionBoard,
        tape: &EventTape,
    ) -> Result<(), CoralExecutionError> {
        self.validate_against_board(source_board)?;
        tape.validate_integrity()
            .map_err(|error| CoralExecutionError::new("event_tape.integrity", error.to_string()))?;

        let header = tape.header();
        if header.policy.has_hidden_oracle_access {
            return Err(CoralExecutionError::new(
                "event_tape.header.policy",
                "the Coral reference policy adapter cannot claim hidden oracle access",
            ));
        }
        let expected_config = serde_json::to_value(&source_board.config).map_err(|error| {
            CoralExecutionError::new(
                "event_tape.header.policy.config",
                format!("Coral policy config serialization failed: {error}"),
            )
        })?;
        if header.policy.algorithm_id != "coral-direction-board-v0.1"
            || header.policy.config != expected_config
        {
            return Err(CoralExecutionError::new(
                "event_tape.header.policy",
                "policy algorithm/config does not identify the frozen Coral reference scheduler",
            ));
        }
        if header.comparison_mode != ComparisonMode::MechanismMatched {
            return Err(CoralExecutionError::new(
                "event_tape.header.comparison_mode",
                "Coral reference adapter requires a mechanism-matched tape",
            ));
        }
        if header.execution != coral_four_seat_execution_contract() {
            return Err(CoralExecutionError::new(
                "event_tape.header.execution",
                "event tape does not use the four-seat Coral execution contract",
            ));
        }

        let batch_events: Vec<_> = tape
            .events()
            .iter()
            .filter(|envelope| {
                envelope.payload.round_id == self.round_id
                    && envelope.payload.batch_id.as_deref() == Some(self.batch_id.as_str())
            })
            .collect();
        if batch_events.len() != CORAL_AGENT_COUNT as usize {
            return Err(CoralExecutionError::new(
                "event_tape.batch",
                format!(
                    "expected exactly {CORAL_AGENT_COUNT} events for batch {:?}, found {}",
                    self.batch_id,
                    batch_events.len()
                ),
            ));
        }

        for claim in &self.claims {
            let event = batch_events
                .iter()
                .find(|envelope| envelope.payload.event_id == claim.event_id)
                .ok_or_else(|| {
                    CoralExecutionError::new(
                        format!("claims[worker={}].event_id", claim.worker_id),
                        "no matching event exists in the named batch",
                    )
                })?;
            let payload = &event.payload;
            if payload.worker_id != claim.worker_id {
                return Err(CoralExecutionError::new(
                    format!("claims[worker={}].worker_id", claim.worker_id),
                    "event worker ID differs from the frozen seat claim",
                ));
            }
            let action_arm = match &payload.intent {
                ActionIntent::EvaluateCandidate { direction_id, .. } => {
                    CoralArm::Direction(direction_id.clone())
                }
                ActionIntent::ProposeDirection { .. } => CoralArm::New,
            };
            if action_arm != claim.arm {
                return Err(CoralExecutionError::new(
                    format!("claims[worker={}].arm", claim.worker_id),
                    "event action kind/direction differs from the frozen worker claim",
                ));
            }
        }
        Ok(())
    }

    /// Difference between recommended and actually claimed seats, by arm.
    ///
    /// Non-zero values are descriptive, not failures: the board is advisory.
    pub fn advisory_deviation(&self) -> BTreeMap<String, i64> {
        let mut recommended = BTreeMap::<String, i64>::new();
        let mut claimed = BTreeMap::<String, i64>::new();
        for allocation in &self.advisory_allocations {
            *recommended.entry(arm_key(&allocation.arm)).or_default() +=
                allocation.wanted_agents as i64;
        }
        for claim in &self.claims {
            *claimed.entry(arm_key(&claim.arm)).or_default() += 1;
        }
        recommended
            .keys()
            .chain(claimed.keys())
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|arm| {
                let delta = claimed.get(&arm).copied().unwrap_or(0)
                    - recommended.get(&arm).copied().unwrap_or(0);
                (arm, delta)
            })
            .collect()
    }

    fn compute_digest(&self) -> Result<String, CoralExecutionError> {
        digest_serializable(&UnsignedRound {
            schema: &self.schema,
            version: &self.version,
            round_id: self.round_id,
            batch_label: &self.batch_label,
            source_board_sha256: &self.source_board_sha256,
            advisory_allocations: &self.advisory_allocations,
            claims: &self.claims,
        })
    }

    fn derived_batch_id(&self) -> String {
        format!("coral-seat-round-{}-{}", self.round_id, &self.round_sha256[..16])
    }
}

fn validate_reference_board(board: &CoralDirectionBoard) -> Result<(), CoralExecutionError> {
    board
        .validate_integrity()
        .map_err(|error| CoralExecutionError::new("source_board", error.to_string()))?;
    let config = &board.config;
    if config.window_size != 32
        || config.prior_n != 4
        || config.infancy_k != 3
        || config.kappa.to_bits() != 0.05f64.to_bits()
        || config.agent_count != CORAL_AGENT_COUNT as usize
        || config.max_agents_per_direction != CORAL_MAX_AGENTS_PER_DIRECTION as usize
        || config.first_round_agents_per_direction
            != CORAL_FIRST_ROUND_AGENTS_PER_DIRECTION as usize
    {
        return Err(CoralExecutionError::new(
            "source_board.config",
            "four-seat reference adapter requires the frozen Coral paper configuration W=32, prior_n=4, infancy_k=3, kappa=0.05, agents=4, steady cap=2, first-round cap=1",
        ));
    }
    Ok(())
}

fn validate_seat_claims(
    board: &CoralDirectionBoard,
    round_id: u64,
    claims: &[CoralSeatClaim],
) -> Result<(), CoralExecutionError> {
    if claims.len() != CORAL_AGENT_COUNT as usize {
        return Err(CoralExecutionError::new(
            "claims",
            format!(
                "a completed Coral round requires exactly {CORAL_AGENT_COUNT} worker claims"
            ),
        ));
    }
    let mut worker_ids = BTreeSet::new();
    let mut event_ids = BTreeSet::new();
    let mut direction_claims = BTreeMap::<String, u32>::new();
    for (index, claim) in claims.iter().enumerate() {
        if claim.worker_id >= CORAL_AGENT_COUNT {
            return Err(CoralExecutionError::new(
                format!("claims[{index}].worker_id"),
                "worker ID must be in 0..4",
            ));
        }
        if !worker_ids.insert(claim.worker_id) {
            return Err(CoralExecutionError::new(
                format!("claims[{index}].worker_id"),
                "worker may claim at most one seat in a round",
            ));
        }
        if claim.event_id.trim().is_empty() || !event_ids.insert(&claim.event_id) {
            return Err(CoralExecutionError::new(
                format!("claims[{index}].event_id"),
                "event IDs must be non-empty and unique within the round",
            ));
        }
        if let CoralArm::Direction(direction_id) = &claim.arm {
            board
                .validate_claim_capacity(
                    direction_id,
                    *direction_claims.get(direction_id).unwrap_or(&0) as usize,
                    round_id == 1,
                )
                .map_err(|error| {
                    CoralExecutionError::new(
                        format!("claims[{index}].arm"),
                        error.to_string(),
                    )
                })?;
            let count = direction_claims.entry(direction_id.clone()).or_default();
            *count += 1;
        }
    }
    if worker_ids.len() != CORAL_AGENT_COUNT as usize {
        return Err(CoralExecutionError::new(
            "claims",
            "every worker seat 0..3 must be represented exactly once",
        ));
    }
    Ok(())
}

fn arm_key(arm: &CoralArm) -> String {
    match arm {
        CoralArm::Direction(direction_id) => format!("direction:{direction_id}"),
        CoralArm::New => "new".to_owned(),
    }
}

fn validate_sha256(value: &str, path: &str) -> Result<(), CoralExecutionError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CoralExecutionError::new(
            path,
            "expected a lowercase 64-character SHA-256 digest",
        ));
    }
    Ok(())
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, CoralExecutionError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        CoralExecutionError::new(
            "serialization",
            format!("deterministic Coral round encoding failed: {error}"),
        )
    })?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board() -> CoralDirectionBoard {
        let mut board = CoralDirectionBoard::new(CoralBaselineConfig::default()).unwrap();
        for direction in ["A", "B", "C", "D"] {
            board.register_direction(direction).unwrap();
        }
        board
    }

    fn claims(arms: [CoralArm; 4]) -> Vec<CoralSeatClaim> {
        arms.into_iter()
            .enumerate()
            .map(|(worker_id, arm)| CoralSeatClaim {
                worker_id: worker_id as u32,
                event_id: format!("round-1-worker-{worker_id}"),
                arm,
            })
            .collect()
    }

    #[test]
    fn reference_execution_contract_is_exactly_four_sequential_seats_in_a_batch() {
        let contract = coral_four_seat_execution_contract();
        assert_eq!(contract.mode, ExecutionMode::Batched);
        assert_eq!(contract.worker_count, 4);
        assert_eq!(contract.seats_per_round, 4);
        assert_eq!(contract.first_round_claim_cap, Some(1));
        assert_eq!(contract.steady_state_claim_cap, Some(2));
    }

    #[test]
    fn round_accepts_four_distinct_workers_and_keeps_board_advisory() {
        let board = board();
        let round = CoralSeatRound::new(
            &board,
            1,
            "coral-round-1",
            claims([
                CoralArm::Direction("A".to_owned()),
                CoralArm::Direction("B".to_owned()),
                CoralArm::Direction("C".to_owned()),
                CoralArm::New,
            ]),
        )
        .unwrap();
        round.validate_against_board(&board).unwrap();
        assert_eq!(round.claims.len(), 4);
        assert!(round.advisory_deviation().values().any(|delta| *delta != 0));
    }

    #[test]
    fn first_round_cap_is_one_agent_per_direction() {
        let board = board();
        assert!(CoralSeatRound::new(
            &board,
            1,
            "coral-round-1",
            claims([
                CoralArm::Direction("A".to_owned()),
                CoralArm::Direction("A".to_owned()),
                CoralArm::Direction("C".to_owned()),
                CoralArm::New,
            ]),
        )
        .is_err());
    }

    #[test]
    fn later_round_cap_is_two_agents_per_direction() {
        let board = board();
        assert!(CoralSeatRound::new(
            &board,
            2,
            "coral-round-2",
            claims([
                CoralArm::Direction("A".to_owned()),
                CoralArm::Direction("A".to_owned()),
                CoralArm::Direction("A".to_owned()),
                CoralArm::New,
            ]),
        )
        .is_err());
        assert!(CoralSeatRound::new(
            &board,
            2,
            "coral-round-2",
            claims([
                CoralArm::Direction("A".to_owned()),
                CoralArm::Direction("A".to_owned()),
                CoralArm::Direction("C".to_owned()),
                CoralArm::New,
            ]),
        )
        .is_ok());
    }

    #[test]
    fn duplicate_workers_and_incomplete_rounds_are_rejected() {
        let board = board();
        let mut invalid = claims([
            CoralArm::Direction("A".to_owned()),
            CoralArm::Direction("B".to_owned()),
            CoralArm::Direction("C".to_owned()),
            CoralArm::New,
        ]);
        invalid[3].worker_id = 2;
        assert!(CoralSeatRound::new(&board, 1, "coral-round-1", invalid).is_err());
        assert!(CoralSeatRound::new(
            &board,
            1,
            "coral-round-1",
            claims([
                CoralArm::Direction("A".to_owned()),
                CoralArm::Direction("B".to_owned()),
                CoralArm::Direction("C".to_owned()),
                CoralArm::New,
            ])[..3]
                .to_vec(),
        )
        .is_err());
    }

    #[test]
    fn reference_adapter_refuses_modified_coral_scheduler_parameters() {
        let mut config = CoralBaselineConfig::default();
        config.kappa = 0.10;
        let mut changed = CoralDirectionBoard::new(config).unwrap();
        changed.register_direction("A").unwrap();
        changed.register_direction("B").unwrap();
        changed.register_direction("C").unwrap();
        changed.register_direction("D").unwrap();
        assert!(CoralSeatRound::new(
            &changed,
            1,
            "coral-round-1",
            claims([
                CoralArm::Direction("A".to_owned()),
                CoralArm::Direction("B".to_owned()),
                CoralArm::Direction("C".to_owned()),
                CoralArm::New,
            ]),
        )
        .is_err());
    }

    #[test]
    fn round_digest_detects_claim_tampering() {
        let board = board();
        let mut round = CoralSeatRound::new(
            &board,
            1,
            "coral-round-1",
            claims([
                CoralArm::Direction("A".to_owned()),
                CoralArm::Direction("B".to_owned()),
                CoralArm::Direction("C".to_owned()),
                CoralArm::New,
            ]),
        )
        .unwrap();
        round.claims[0].arm = CoralArm::New;
        assert!(round.validate_against_board(&board).is_err());
    }
}
