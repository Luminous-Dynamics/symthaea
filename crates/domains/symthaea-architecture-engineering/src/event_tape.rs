//! Versioned, append-only event tape for matched-budget research search.
//!
//! The tape is evaluator-owned. Policies choose intents; the evaluator records outcomes,
//! verifier decisions, visibility boundaries and complete resource receipts. Hash chaining
//! detects mutation, while the replay interface checks that a policy makes the same choice
//! from exactly the history it was permitted to observe.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const TAPE_SCHEMA: &str = "symthaea.ar5b4.event-tape";
const TAPE_VERSION: &str = "0.1.0";

/// Which comparison question this tape belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonMode {
    /// Every policy receives the same evaluator-owned proposal stream.
    MechanismMatched,
    /// Each complete system may use its own proposal and orchestration strategy.
    BestPracticalSystem,
}

/// Whether decisions are sequential or made against a shared batch snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Sequential,
    Batched,
}

/// Execution semantics are part of the experiment, not an implementation detail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionContract {
    pub mode: ExecutionMode,
    pub worker_count: u32,
    /// Number of advisory action seats required for a complete batch round.
    pub seats_per_round: u32,
    /// Maximum claims per canonical direction during the first round.
    pub first_round_claim_cap: Option<u32>,
    /// Maximum claims per canonical direction in later rounds.
    pub steady_state_claim_cap: Option<u32>,
}

/// The exact policy identity and frozen configuration used by one tape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicySnapshot {
    pub policy_id: String,
    pub algorithm_id: String,
    pub config: Value,
    pub config_sha256: String,
    /// True only for an explicitly labelled upper-bound/oracle policy.
    pub has_hidden_oracle_access: bool,
}

/// Immutable experiment identity and comparison contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventTapeHeader {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub comparison_mode: ComparisonMode,
    pub proposal_source_id: String,
    /// Required for mechanism-matched comparisons; identifies the shared proposal stream.
    pub proposal_stream_sha256: Option<String>,
    pub task_recipe_sha256: String,
    pub comparison_profile_sha256: String,
    pub evaluator_sha256: String,
    pub verifier_sha256: String,
    pub budget_limit_microunits: u64,
    pub initial_direction_ids: Vec<String>,
    pub execution: ExecutionContract,
    pub policy: PolicySnapshot,
}

/// A resource measurement is either known or explicitly unknown. Unknown is never zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum MeterValue {
    Measured { value: u64 },
    Unknown { reason: String },
}

impl MeterValue {
    fn measured_value(&self) -> Option<u64> {
        match self {
            Self::Measured { value } => Some(*value),
            Self::Unknown { .. } => None,
        }
    }

    fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }
}

/// Complete resource receipt for one paid action.
///
/// All fields must be reported. A zero is a measured zero; an unavailable measurement must
/// use Unknown and will prevent the tape from qualifying for matched-budget comparisons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceReceipt {
    pub host_cpu_millis: MeterValue,
    pub wall_millis: MeterValue,
    pub accelerator_millis: MeterValue,
    pub peak_memory_bytes: MeterValue,
    pub energy_millijoules: MeterValue,
    pub input_tokens: MeterValue,
    pub output_tokens: MeterValue,
    pub tool_calls: MeterValue,
    pub evaluator_calls: MeterValue,
    pub verifier_calls: MeterValue,
    /// Evaluator-owned normalized charge; the sum is checked against the campaign ceiling.
    pub charged_budget_microunits: MeterValue,
}

impl ResourceReceipt {
    fn has_unknown_measurement(&self) -> bool {
        [
            &self.host_cpu_millis,
            &self.wall_millis,
            &self.accelerator_millis,
            &self.peak_memory_bytes,
            &self.energy_millijoules,
            &self.input_tokens,
            &self.output_tokens,
            &self.tool_calls,
            &self.evaluator_calls,
            &self.verifier_calls,
            &self.charged_budget_microunits,
        ]
        .iter()
        .any(|meter| meter.is_unknown())
    }

    fn validate_unknown_reasons(&self, path: &str) -> Result<(), EventTapeError> {
        let fields = [
            ("host_cpu_millis", &self.host_cpu_millis),
            ("wall_millis", &self.wall_millis),
            ("accelerator_millis", &self.accelerator_millis),
            ("peak_memory_bytes", &self.peak_memory_bytes),
            ("energy_millijoules", &self.energy_millijoules),
            ("input_tokens", &self.input_tokens),
            ("output_tokens", &self.output_tokens),
            ("tool_calls", &self.tool_calls),
            ("evaluator_calls", &self.evaluator_calls),
            ("verifier_calls", &self.verifier_calls),
            ("charged_budget_microunits", &self.charged_budget_microunits),
        ];
        for (field, meter) in fields {
            if let MeterValue::Unknown { reason } = meter {
                if reason.trim().is_empty() {
                    return Err(EventTapeError::new(
                        format!("{path}.{field}"),
                        "unknown measurements require a non-empty reason",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// What the policy selected. Candidate generation itself may be shared or system-native.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionIntent {
    EvaluateCandidate {
        candidate_sha256: String,
        parent_candidate_sha256: Option<String>,
        semantic_delta_sha256: String,
        direction_id: String,
    },
    ProposeDirection {
        descriptor_sha256: String,
        parent_direction_id: Option<String>,
        semantic_delta_sha256: String,
    },
}

/// Candidate evaluation result recorded by the evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateDisposition {
    Verified,
    Rejected,
    Inconclusive,
    Crashed,
    TimedOut,
}

/// Verifier-owned canonical mapping for a direction proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "disposition", rename_all = "snake_case")]
pub enum ProposalDisposition {
    Admitted { canonical_direction_id: String },
    Duplicate { canonical_direction_id: String },
    Rejected,
    Inconclusive,
}

/// Outcome of the selected action. Metadata is repeated on each event so the evidence remains
/// self-describing even when a tape is extracted from a larger campaign.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionOutcome {
    CandidateEvaluation {
        task_recipe_sha256: String,
        comparison_profile_sha256: String,
        evaluator_sha256: String,
        disposition: CandidateDisposition,
        score: Option<f64>,
        uncertainty: Option<f64>,
        evidence_sha256: Option<String>,
    },
    DirectionVerification {
        task_recipe_sha256: String,
        comparison_profile_sha256: String,
        verifier_sha256: String,
        verifier_receipt_sha256: String,
        disposition: ProposalDisposition,
    },
}

/// One evaluator-owned action record before hash-chain envelope fields are added.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionEvent {
    pub event_id: String,
    pub policy_id: String,
    pub worker_id: u32,
    pub round_id: u64,
    pub batch_id: Option<String>,
    /// Number of prior event records visible at decision time.
    pub observed_through_sequence: u64,
    /// Filled by EventTape::append; binds the exact history visible to this decision.
    pub visible_history_sha256: String,
    /// Optional reference for retries. Retry work is charged again in its own receipt.
    pub retry_of_event_id: Option<String>,
    pub intent: ActionIntent,
    pub outcome: ActionOutcome,
    pub resources: ResourceReceipt,
}

/// Hash-chained event envelope. Sequence numbers begin at one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub sequence: u64,
    pub previous_event_sha256: String,
    pub event_sha256: String,
    pub payload: ActionEvent,
}

/// The complete replayable record for one policy/campaign.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventTape {
    header: EventTapeHeader,
    events: Vec<EventEnvelope>,
}

/// Summary produced by structural validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventTapeSummary {
    pub event_count: usize,
    pub candidate_evaluations: usize,
    pub direction_proposals: usize,
    pub admitted_directions: usize,
    pub duplicate_proposals: usize,
    pub rejected_proposals: usize,
    pub inconclusive_proposals: usize,
    /// None means at least one action's normalized charge is unknown.
    pub total_charged_budget_microunits: Option<u64>,
    pub fully_metered: bool,
    pub registered_direction_ids: Vec<String>,
}

/// A deterministic validation or replay failure with a stable location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventTapeError {
    pub path: String,
    pub message: String,
}

impl EventTapeError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for EventTapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for EventTapeError {}

impl EventTape {
    /// Create an empty tape after validating its immutable header.
    pub fn new(header: EventTapeHeader) -> Result<Self, EventTapeError> {
        validate_header(&header)?;
        Ok(Self {
            header,
            events: Vec::new(),
        })
    }

    /// Append one event, calculate its visible-history digest and chain hash, then validate.
    /// A failed append is rolled back; callers never receive a partially accepted event.
    /// Read the frozen experiment header.
    pub fn header(&self) -> &EventTapeHeader {
        &self.header
    }

    /// Read the immutable accepted event prefix.
    pub fn events(&self) -> &[EventEnvelope] {
        &self.events
    }

    pub fn append(&mut self, mut payload: ActionEvent) -> Result<(), EventTapeError> {
        let visible_count = usize::try_from(payload.observed_through_sequence).map_err(|_| {
            EventTapeError::new(
                "event.observed_through_sequence",
                "visible sequence does not fit this platform",
            )
        })?;
        if visible_count > self.events.len() {
            return Err(EventTapeError::new(
                "event.observed_through_sequence",
                "a decision cannot observe events that have not yet been appended",
            ));
        }
        payload.visible_history_sha256 = digest_serializable(&self.events[..visible_count])?;

        let sequence = u64::try_from(self.events.len())
            .map_err(|_| EventTapeError::new("events", "event count overflow"))?
            .checked_add(1)
            .ok_or_else(|| EventTapeError::new("events", "sequence overflow"))?;
        let previous_event_sha256 = match self.events.last() {
            Some(event) => event.event_sha256.clone(),
            None => digest_serializable(&self.header)?,
        };
        let event_sha256 = hash_event(sequence, &previous_event_sha256, &payload)?;
        self.events.push(EventEnvelope {
            sequence,
            previous_event_sha256,
            event_sha256,
            payload,
        });

        if let Err(error) = self.validate_integrity() {
            self.events.pop();
            return Err(error);
        }
        Ok(())
    }

    /// Validate schema, hash chain, visibility, verifier mappings, schedule and budget ceiling.
    /// Explicitly unknown meters are retained as unknown and do not become zero.
    pub fn validate_integrity(&self) -> Result<EventTapeSummary, EventTapeError> {
        validate_header(&self.header)?;

        let mut registered: BTreeSet<String> =
            self.header.initial_direction_ids.iter().cloned().collect();
        let mut event_ids = BTreeSet::new();
        let mut total_charge = 0u64;
        let mut charge_is_complete = true;
        let mut fully_metered = true;
        let mut candidate_evaluations = 0usize;
        let mut direction_proposals = 0usize;
        let mut admitted_directions = 0usize;
        let mut duplicate_proposals = 0usize;
        let mut rejected_proposals = 0usize;
        let mut inconclusive_proposals = 0usize;
        let mut claims_per_round_direction = BTreeMap::<(u64, String), u32>::new();

        let mut seen_batches = BTreeSet::<String>::new();
        let mut active_batch: Option<BatchState> = None;
        let mut previous_batch_round = 0u64;

        for (index, envelope) in self.events.iter().enumerate() {
            let sequence = u64::try_from(index)
                .map_err(|_| EventTapeError::new("events", "event index overflow"))?
                .checked_add(1)
                .ok_or_else(|| EventTapeError::new("events", "sequence overflow"))?;
            let path = format!("events[{index}]");

            if envelope.sequence != sequence {
                return Err(EventTapeError::new(
                    format!("{path}.sequence"),
                    format!("expected sequence {sequence}, found {}", envelope.sequence),
                ));
            }
            let expected_previous = if index == 0 {
                digest_serializable(&self.header)?
            } else {
                self.events[index - 1].event_sha256.clone()
            };
            if envelope.previous_event_sha256 != expected_previous {
                return Err(EventTapeError::new(
                    format!("{path}.previous_event_sha256"),
                    "hash chain does not point to the preceding event",
                ));
            }
            validate_sha256(
                &envelope.event_sha256,
                &format!("{path}.event_sha256"),
            )?;
            let expected_hash = hash_event(
                envelope.sequence,
                &envelope.previous_event_sha256,
                &envelope.payload,
            )?;
            if envelope.event_sha256 != expected_hash {
                return Err(EventTapeError::new(
                    format!("{path}.event_sha256"),
                    "event payload or chain metadata was modified",
                ));
            }

            let payload = &envelope.payload;
            validate_non_empty(&payload.event_id, &format!("{path}.event_id"))?;
            if !event_ids.insert(payload.event_id.clone()) {
                return Err(EventTapeError::new(
                    format!("{path}.event_id"),
                    "duplicate event ID",
                ));
            }
            if payload.policy_id != self.header.policy.policy_id {
                return Err(EventTapeError::new(
                    format!("{path}.policy_id"),
                    "event policy ID does not match the frozen tape policy",
                ));
            }
            validate_sha256(
                &payload.visible_history_sha256,
                &format!("{path}.visible_history_sha256"),
            )?;
            if payload.observed_through_sequence >= sequence {
                return Err(EventTapeError::new(
                    format!("{path}.observed_through_sequence"),
                    "an action cannot observe its own outcome or a future event",
                ));
            }
            let visible_count = usize::try_from(payload.observed_through_sequence).map_err(|_| {
                EventTapeError::new(
                    format!("{path}.observed_through_sequence"),
                    "visible sequence does not fit this platform",
                )
            })?;
            let expected_visible_hash =
                digest_serializable(&self.events[..visible_count])?;
            if payload.visible_history_sha256 != expected_visible_hash {
                return Err(EventTapeError::new(
                    format!("{path}.visible_history_sha256"),
                    "visible-history digest does not match the policy-visible prefix",
                ));
            }

            validate_schedule_event(
                &self.header.execution,
                envelope,
                &mut active_batch,
                &mut seen_batches,
                &mut previous_batch_round,
                &path,
            )?;

            if let Some(retry_id) = &payload.retry_of_event_id {
                validate_non_empty(retry_id, &format!("{path}.retry_of_event_id"))?;
                let prior = self.events[..index]
                    .iter()
                    .find(|event| event.payload.event_id == *retry_id)
                    .ok_or_else(|| {
                        EventTapeError::new(
                            format!("{path}.retry_of_event_id"),
                            "retry must reference an earlier event in this tape",
                        )
                    })?;
                if !same_work(&payload.intent, &prior.payload.intent) {
                    return Err(EventTapeError::new(
                        format!("{path}.retry_of_event_id"),
                        "retry target does not match the original candidate or proposal",
                    ));
                }
            }

            payload
                .resources
                .validate_unknown_reasons(&format!("{path}.resources"))?;
            if payload.resources.has_unknown_measurement() {
                fully_metered = false;
            }
            match payload.resources.charged_budget_microunits.measured_value() {
                Some(charge) => {
                    total_charge = total_charge.checked_add(charge).ok_or_else(|| {
                        EventTapeError::new(
                            format!("{path}.resources.charged_budget_microunits"),
                            "total normalized charge overflow",
                        )
                    })?;
                }
                None => {
                    charge_is_complete = false;
                    fully_metered = false;
                }
            }

            match (&payload.intent, &payload.outcome) {
                (
                    ActionIntent::EvaluateCandidate {
                        candidate_sha256,
                        parent_candidate_sha256,
                        semantic_delta_sha256,
                        direction_id,
                    },
                    ActionOutcome::CandidateEvaluation {
                        task_recipe_sha256,
                        comparison_profile_sha256,
                        evaluator_sha256,
                        disposition,
                        score,
                        uncertainty,
                        evidence_sha256,
                    },
                ) => {
                    candidate_evaluations += 1;
                    validate_sha256(candidate_sha256, &format!("{path}.intent.candidate_sha256"))?;
                    if let Some(parent) = parent_candidate_sha256 {
                        validate_sha256(parent, &format!("{path}.intent.parent_candidate_sha256"))?;
                    }
                    validate_sha256(
                        semantic_delta_sha256,
                        &format!("{path}.intent.semantic_delta_sha256"),
                    )?;
                    validate_registered_direction(direction_id, &registered, &path)?;
                    validate_event_versions(
                        task_recipe_sha256,
                        comparison_profile_sha256,
                        evaluator_sha256,
                        &self.header,
                        &path,
                        false,
                    )?;
                    match disposition {
                        CandidateDisposition::Verified => {
                            let value = score.ok_or_else(|| {
                                EventTapeError::new(
                                    format!("{path}.outcome.score"),
                                    "verified evaluations require a score",
                                )
                            })?;
                            if !value.is_finite() {
                                return Err(EventTapeError::new(
                                    format!("{path}.outcome.score"),
                                    "score must be finite",
                                ));
                            }
                            if let Some(value) = uncertainty {
                                if !value.is_finite() || *value < 0.0 {
                                    return Err(EventTapeError::new(
                                        format!("{path}.outcome.uncertainty"),
                                        "uncertainty must be finite and non-negative",
                                    ));
                                }
                            }
                            let evidence = evidence_sha256.as_deref().ok_or_else(|| {
                                EventTapeError::new(
                                    format!("{path}.outcome.evidence_sha256"),
                                    "verified evaluations require an evidence artifact digest",
                                )
                            })?;
                            validate_sha256(evidence, &format!("{path}.outcome.evidence_sha256"))?;
                        }
                        _ => {
                            if score.is_some() || uncertainty.is_some() || evidence_sha256.is_some() {
                                return Err(EventTapeError::new(
                                    format!("{path}.outcome"),
                                    "non-verified evaluations cannot publish score, uncertainty or evidence",
                                ));
                            }
                        }
                    }
                    let key = (payload.round_id, direction_id.clone());
                    let count = claims_per_round_direction.entry(key).or_default();
                    *count = count.checked_add(1).ok_or_else(|| {
                        EventTapeError::new(&path, "direction claim count overflow")
                    })?;
                }
                (
                    ActionIntent::ProposeDirection {
                        descriptor_sha256,
                        parent_direction_id,
                        semantic_delta_sha256,
                    },
                    ActionOutcome::DirectionVerification {
                        task_recipe_sha256,
                        comparison_profile_sha256,
                        verifier_sha256,
                        verifier_receipt_sha256,
                        disposition,
                    },
                ) => {
                    direction_proposals += 1;
                    validate_sha256(descriptor_sha256, &format!("{path}.intent.descriptor_sha256"))?;
                    validate_sha256(
                        semantic_delta_sha256,
                        &format!("{path}.intent.semantic_delta_sha256"),
                    )?;
                    if let Some(parent) = parent_direction_id {
                        validate_registered_direction(parent, &registered, &path)?;
                    }
                    validate_event_versions(
                        task_recipe_sha256,
                        comparison_profile_sha256,
                        verifier_sha256,
                        &self.header,
                        &path,
                        true,
                    )?;
                    validate_sha256(
                        verifier_receipt_sha256,
                        &format!("{path}.outcome.verifier_receipt_sha256"),
                    )?;
                    match disposition {
                        ProposalDisposition::Admitted {
                            canonical_direction_id,
                        } => {
                            validate_non_empty(
                                canonical_direction_id,
                                &format!("{path}.outcome.canonical_direction_id"),
                            )?;
                            if !registered.insert(canonical_direction_id.clone()) {
                                return Err(EventTapeError::new(
                                    format!("{path}.outcome.canonical_direction_id"),
                                    "verifier admitted a direction ID that already exists",
                                ));
                            }
                            admitted_directions += 1;
                        }
                        ProposalDisposition::Duplicate {
                            canonical_direction_id,
                        } => {
                            validate_registered_direction(
                                canonical_direction_id,
                                &registered,
                                &path,
                            )?;
                            duplicate_proposals += 1;
                        }
                        ProposalDisposition::Rejected => rejected_proposals += 1,
                        ProposalDisposition::Inconclusive => inconclusive_proposals += 1,
                    }
                }
                _ => {
                    return Err(EventTapeError::new(
                        format!("{path}.outcome.kind"),
                        "action intent and outcome kinds do not match",
                    ));
                }
            }
        }

        if total_charge > self.header.budget_limit_microunits {
            return Err(EventTapeError::new(
                "events",
                format!(
                    "charged budget {total_charge} exceeds campaign limit {}",
                    self.header.budget_limit_microunits
                ),
            ));
        }
        validate_claim_caps(
            &claims_per_round_direction,
            &self.header.execution,
        )?;

        let total_charged_budget_microunits =
            if charge_is_complete { Some(total_charge) } else { None };
        let mut registered_direction_ids: Vec<String> = registered.into_iter().collect();
        registered_direction_ids.sort_unstable();

        Ok(EventTapeSummary {
            event_count: self.events.len(),
            candidate_evaluations,
            direction_proposals,
            admitted_directions,
            duplicate_proposals,
            rejected_proposals,
            inconclusive_proposals,
            total_charged_budget_microunits,
            fully_metered,
            registered_direction_ids,
        })
    }

    /// Qualification validation: every resource dimension and every normalized charge must
    /// be measured. Use validate_integrity for exploratory tapes that intentionally retain
    /// explicit unknowns; such tapes cannot be used in matched-budget claims.
    pub fn validate(&self) -> Result<EventTapeSummary, EventTapeError> {
        if self.events.is_empty() {
            return Err(EventTapeError::new(
                "events",
                "an empty tape cannot qualify as an executed campaign",
            ));
        }
        let summary = self.validate_integrity()?;
        if self.header.execution.mode == ExecutionMode::Batched {
            if let Some(last_event) = self.events.last() {
                let last_batch_id = last_event.payload.batch_id.as_deref().ok_or_else(|| {
                    EventTapeError::new("events", "batched tape ends without a batch ID")
                })?;
                let completed_seats = self
                    .events
                    .iter()
                    .rev()
                    .take_while(|event| event.payload.batch_id.as_deref() == Some(last_batch_id))
                    .count();
                if completed_seats != self.header.execution.seats_per_round as usize {
                    return Err(EventTapeError::new(
                        "events.last_batch",
                        format!(
                            "last batch has {completed_seats} actions; expected {} advisory seats",
                            self.header.execution.seats_per_round
                        ),
                    ));
                }
            }
        }
        if !summary.fully_metered || summary.total_charged_budget_microunits.is_none() {
            return Err(EventTapeError::new(
                "events.resources",
                "tape is structurally valid but unqualified: one or more resource measurements are unknown",
            ));
        }
        Ok(summary)
    }

    /// Recompute every policy decision using exactly the prefix and worker context recorded
    /// in the tape. The adapter must be a deterministic replay of the named policy/config.
    pub fn verify_policy_replay<A: PolicyReplay>(
        &self,
        adapter: &A,
    ) -> Result<(), EventTapeError> {
        self.validate()?;
        if adapter.policy_id() != self.header.policy.policy_id {
            return Err(EventTapeError::new(
                "replay.policy_id",
                "replay adapter policy ID differs from the frozen tape policy",
            ));
        }
        if adapter.config_sha256() != self.header.policy.config_sha256 {
            return Err(EventTapeError::new(
                "replay.config_sha256",
                "replay adapter config digest differs from the frozen policy snapshot",
            ));
        }
        for (index, envelope) in self.events.iter().enumerate() {
            let payload = &envelope.payload;
            let visible_count = usize::try_from(payload.observed_through_sequence).map_err(|_| {
                EventTapeError::new(
                    format!("events[{index}].observed_through_sequence"),
                    "visible sequence does not fit this platform",
                )
            })?;
            let context = DecisionContext {
                round_id: payload.round_id,
                worker_id: payload.worker_id,
                batch_id: payload.batch_id.clone(),
                execution_mode: self.header.execution.mode,
            };
            let selected = adapter
                .select_action(&self.events[..visible_count], &context)
                .map_err(|message| {
                    EventTapeError::new(format!("events[{index}].replay"), message)
                })?;
            let recorded = match &payload.intent {
                ActionIntent::EvaluateCandidate { direction_id, .. } => {
                    PolicySelection::EvaluateDirection {
                        direction_id: direction_id.clone(),
                    }
                }
                ActionIntent::ProposeDirection { .. } => PolicySelection::ProposeDirection,
            };
            if selected != recorded {
                return Err(EventTapeError::new(
                    format!("events[{index}].replay"),
                    format!("policy replay selected {selected:?}, recorded {recorded:?}"),
                ));
            }
        }
        Ok(())
    }
}

/// Minimal action decision that a policy adapter must reproduce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicySelection {
    EvaluateDirection { direction_id: String },
    ProposeDirection,
}

/// Context that can legitimately affect a policy's choice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionContext {
    pub round_id: u64,
    pub worker_id: u32,
    pub batch_id: Option<String>,
    pub execution_mode: ExecutionMode,
}

/// Implement this for each policy to reproduce its choices from a visible event prefix.
/// This is intentionally immutable: all policy state must be derivable from the supplied tape.
pub trait PolicyReplay {
    /// Exact policy identifier from the frozen tape header.
    fn policy_id(&self) -> &str;

    /// SHA-256 of the policy's frozen serialized configuration.
    fn config_sha256(&self) -> &str;

    fn select_action(
        &self,
        visible_events: &[EventEnvelope],
        context: &DecisionContext,
    ) -> Result<PolicySelection, String>;
}

#[derive(Debug)]
struct BatchState {
    id: String,
    round_id: u64,
    visible_through_sequence: u64,
    workers: BTreeSet<u32>,
}

fn validate_schedule_event(
    contract: &ExecutionContract,
    envelope: &EventEnvelope,
    active_batch: &mut Option<BatchState>,
    seen_batches: &mut BTreeSet<String>,
    previous_batch_round: &mut u64,
    path: &str,
) -> Result<(), EventTapeError> {
    let payload = &envelope.payload;
    match contract.mode {
        ExecutionMode::Sequential => {
            if payload.batch_id.is_some()
                || payload.worker_id != 0
                || contract.worker_count != 1
                || payload.round_id != envelope.sequence
                || payload.observed_through_sequence != envelope.sequence - 1
            {
                return Err(EventTapeError::new(
                    format!("{path}.schedule"),
                    "sequential events require worker 0, no batch, one worker, round=sequence and full prior visibility",
                ));
            }
        }
        ExecutionMode::Batched => {
            let batch_id = payload.batch_id.as_deref().ok_or_else(|| {
                EventTapeError::new(
                    format!("{path}.batch_id"),
                    "batched execution requires a batch ID",
                )
            })?;
            validate_non_empty(batch_id, &format!("{path}.batch_id"))?;
            if contract.worker_count < 2 || payload.worker_id >= contract.worker_count {
                return Err(EventTapeError::new(
                    format!("{path}.worker_id"),
                    "batched execution requires at least two workers and a valid worker index",
                ));
            }

            let active_matches = active_batch
                .as_ref()
                .is_some_and(|batch| batch.id == batch_id);
            if !active_matches {
                if let Some(previous) = active_batch.take() {
                    if previous.workers.len() != contract.seats_per_round as usize {
                        return Err(EventTapeError::new(
                            format!("{path}.batch_id"),
                            format!(
                                "previous batch completed {} of {} advisory seats",
                                previous.workers.len(), contract.seats_per_round
                            ),
                        ));
                    }
                    seen_batches.insert(previous.id);
                }
                if seen_batches.contains(batch_id) {
                    return Err(EventTapeError::new(
                        format!("{path}.batch_id"),
                        "events for one batch must be contiguous",
                    ));
                }
                let expected_round = previous_batch_round.checked_add(1).ok_or_else(|| {
                    EventTapeError::new(format!("{path}.round_id"), "round overflow")
                })?;
                if payload.round_id != expected_round {
                    return Err(EventTapeError::new(
                        format!("{path}.round_id"),
                        format!("expected batch round {expected_round}, found {}", payload.round_id),
                    ));
                }
                if payload.observed_through_sequence != envelope.sequence - 1 {
                    return Err(EventTapeError::new(
                        format!("{path}.observed_through_sequence"),
                        "the first event in a batch must see all and only completed prior batches",
                    ));
                }
                let mut workers = BTreeSet::new();
                workers.insert(payload.worker_id);
                *active_batch = Some(BatchState {
                    id: batch_id.to_owned(),
                    round_id: payload.round_id,
                    visible_through_sequence: payload.observed_through_sequence,
                    workers,
                });
                *previous_batch_round = payload.round_id;
            } else {
                let batch = active_batch.as_mut().ok_or_else(|| {
                    EventTapeError::new(format!("{path}.batch_id"), "active batch state missing")
                })?;
                if payload.round_id != batch.round_id
                    || payload.observed_through_sequence != batch.visible_through_sequence
                {
                    return Err(EventTapeError::new(
                        format!("{path}.schedule"),
                        "all workers in a batch must share one round and one pre-batch visibility cutoff",
                    ));
                }
                if !batch.workers.insert(payload.worker_id) {
                    return Err(EventTapeError::new(
                        format!("{path}.worker_id"),
                        "a worker may submit only one action in a batch round",
                    ));
                }
            }
            if let Some(batch) = active_batch.as_ref() {
                if batch.workers.len() > contract.worker_count as usize {
                    return Err(EventTapeError::new(
                        format!("{path}.batch_id"),
                        "batch contains more actions than configured workers",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_claim_caps(
    claims: &BTreeMap<(u64, String), u32>,
    contract: &ExecutionContract,
) -> Result<(), EventTapeError> {
    for ((round_id, direction_id), count) in claims {
        let cap = if *round_id == 1 {
            contract.first_round_claim_cap
        } else {
            contract.steady_state_claim_cap
        };
        if let Some(cap) = cap {
            if *count > cap {
                return Err(EventTapeError::new(
                    format!("rounds[{round_id}].direction[{direction_id}]"),
                    format!("direction claim count {count} exceeds configured cap {cap}"),
                ));
            }
        }
    }
    Ok(())
}

fn validate_event_versions(
    task_recipe_sha256: &str,
    comparison_profile_sha256: &str,
    evaluator_or_verifier_sha256: &str,
    header: &EventTapeHeader,
    path: &str,
    is_proposal: bool,
) -> Result<(), EventTapeError> {
    validate_sha256(task_recipe_sha256, &format!("{path}.outcome.task_recipe_sha256"))?;
    validate_sha256(
        comparison_profile_sha256,
        &format!("{path}.outcome.comparison_profile_sha256"),
    )?;
    validate_sha256(
        evaluator_or_verifier_sha256,
        &format!("{path}.outcome.engine_sha256"),
    )?;
    let expected_engine = if is_proposal {
        &header.verifier_sha256
    } else {
        &header.evaluator_sha256
    };
    if task_recipe_sha256 != header.task_recipe_sha256
        || comparison_profile_sha256 != header.comparison_profile_sha256
        || evaluator_or_verifier_sha256 != expected_engine
    {
        return Err(EventTapeError::new(
            format!("{path}.outcome"),
            "task, comparison profile or evaluator/verifier digest differs from the frozen header",
        ));
    }
    Ok(())
}

fn validate_header(header: &EventTapeHeader) -> Result<(), EventTapeError> {
    if header.schema != TAPE_SCHEMA {
        return Err(EventTapeError::new(
            "header.schema",
            format!("expected {TAPE_SCHEMA:?}"),
        ));
    }
    if header.version != TAPE_VERSION {
        return Err(EventTapeError::new(
            "header.version",
            format!("expected {TAPE_VERSION:?}"),
        ));
    }
    validate_non_empty(&header.campaign_id, "header.campaign_id")?;
    validate_non_empty(&header.proposal_source_id, "header.proposal_source_id")?;
    validate_sha256(&header.task_recipe_sha256, "header.task_recipe_sha256")?;
    validate_sha256(
        &header.comparison_profile_sha256,
        "header.comparison_profile_sha256",
    )?;
    validate_sha256(&header.evaluator_sha256, "header.evaluator_sha256")?;
    validate_sha256(&header.verifier_sha256, "header.verifier_sha256")?;
    if header.budget_limit_microunits == 0 {
        return Err(EventTapeError::new(
            "header.budget_limit_microunits",
            "budget limit must be positive",
        ));
    }
    if header.comparison_mode == ComparisonMode::MechanismMatched {
        let stream = header.proposal_stream_sha256.as_deref().ok_or_else(|| {
            EventTapeError::new(
                "header.proposal_stream_sha256",
                "mechanism-matched comparisons require a shared proposal-stream digest",
            )
        })?;
        validate_sha256(stream, "header.proposal_stream_sha256")?;
    } else if let Some(stream) = &header.proposal_stream_sha256 {
        validate_sha256(stream, "header.proposal_stream_sha256")?;
    }

    if header.execution.worker_count == 0 {
        return Err(EventTapeError::new(
            "header.execution.worker_count",
            "worker count must be positive",
        ));
    }
    match header.execution.mode {
        ExecutionMode::Sequential
            if header.execution.worker_count != 1 || header.execution.seats_per_round != 1 => {
            return Err(EventTapeError::new(
                "header.execution.worker_count",
                "sequential mode requires exactly one worker",
            ));
        }
        ExecutionMode::Batched
            if header.execution.worker_count < 2
                || header.execution.seats_per_round == 0
                || header.execution.seats_per_round > header.execution.worker_count => {
            return Err(EventTapeError::new(
                "header.execution.worker_count",
                "batched mode requires at least two workers",
            ));
        }
        _ => {}
    }
    for (name, cap) in [
        ("first_round_claim_cap", header.execution.first_round_claim_cap),
        ("steady_state_claim_cap", header.execution.steady_state_claim_cap),
    ] {
        if cap == Some(0) {
            return Err(EventTapeError::new(
                format!("header.execution.{name}"),
                "claim cap must be positive or absent",
            ));
        }
    }

    validate_non_empty(&header.policy.policy_id, "header.policy.policy_id")?;
    validate_non_empty(&header.policy.algorithm_id, "header.policy.algorithm_id")?;
    if !header.policy.config.is_object() {
        return Err(EventTapeError::new(
            "header.policy.config",
            "frozen policy config must be a JSON object",
        ));
    }
    validate_sha256(&header.policy.config_sha256, "header.policy.config_sha256")?;
    let config_sha256 = digest_serializable(&header.policy.config)?;
    if config_sha256 != header.policy.config_sha256 {
        return Err(EventTapeError::new(
            "header.policy.config_sha256",
            "policy config digest does not match the frozen config object",
        ));
    }

    let mut direction_ids = BTreeSet::new();
    if header.initial_direction_ids.is_empty() {
        return Err(EventTapeError::new(
            "header.initial_direction_ids",
            "at least one initial canonical direction is required",
        ));
    }
    for (index, direction_id) in header.initial_direction_ids.iter().enumerate() {
        validate_non_empty(direction_id, &format!("header.initial_direction_ids[{index}]"))?;
        if !direction_ids.insert(direction_id) {
            return Err(EventTapeError::new(
                format!("header.initial_direction_ids[{index}]"),
                "duplicate initial direction ID",
            ));
        }
    }
    Ok(())
}

fn validate_registered_direction(
    direction_id: &str,
    registered: &BTreeSet<String>,
    path: &str,
) -> Result<(), EventTapeError> {
    validate_non_empty(direction_id, &format!("{path}.direction_id"))?;
    if !registered.contains(direction_id) {
        return Err(EventTapeError::new(
            format!("{path}.direction_id"),
            format!("direction {direction_id:?} has not been admitted by the verifier"),
        ));
    }
    Ok(())
}

fn validate_non_empty(value: &str, path: &str) -> Result<(), EventTapeError> {
    if value.trim().is_empty() {
        return Err(EventTapeError::new(path, "value must be non-empty"));
    }
    Ok(())
}

fn validate_sha256(value: &str, path: &str) -> Result<(), EventTapeError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(EventTapeError::new(
            path,
            "expected a lowercase 64-character SHA-256 digest",
        ));
    }
    Ok(())
}

fn same_work(left: &ActionIntent, right: &ActionIntent) -> bool {
    match (left, right) {
        (
            ActionIntent::EvaluateCandidate {
                candidate_sha256: left,
                ..
            },
            ActionIntent::EvaluateCandidate {
                candidate_sha256: right,
                ..
            },
        ) => left == right,
        (
            ActionIntent::ProposeDirection {
                descriptor_sha256: left,
                ..
            },
            ActionIntent::ProposeDirection {
                descriptor_sha256: right,
                ..
            },
        ) => left == right,
        _ => false,
    }
}

#[derive(Serialize)]
struct EventHashInput<'a> {
    sequence: u64,
    previous_event_sha256: &'a str,
    payload: &'a ActionEvent,
}

fn hash_event(
    sequence: u64,
    previous_event_sha256: &str,
    payload: &ActionEvent,
) -> Result<String, EventTapeError> {
    digest_serializable(&EventHashInput {
        sequence,
        previous_event_sha256,
        payload,
    })
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, EventTapeError> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        EventTapeError::new("serialization", format!("deterministic JSON encoding failed: {error}"))
    })?;
    Ok(digest_bytes(&encoded))
}

fn digest_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const H5: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn meter(value: u64) -> MeterValue {
        MeterValue::Measured { value }
    }

    fn resources(charge: u64) -> ResourceReceipt {
        ResourceReceipt {
            host_cpu_millis: meter(10),
            wall_millis: meter(2),
            accelerator_millis: meter(0),
            peak_memory_bytes: meter(1024),
            energy_millijoules: meter(2),
            input_tokens: meter(100),
            output_tokens: meter(20),
            tool_calls: meter(1),
            evaluator_calls: meter(1),
            verifier_calls: meter(1),
            charged_budget_microunits: meter(charge),
        }
    }

    fn header(mode: ExecutionMode, worker_count: u32, budget: u64) -> EventTapeHeader {
        let config = json!({"exploration": 0.55, "seed": 17});
        EventTapeHeader {
            schema: TAPE_SCHEMA.to_owned(),
            version: TAPE_VERSION.to_owned(),
            campaign_id: "campaign-test-01".to_owned(),
            comparison_mode: ComparisonMode::MechanismMatched,
            proposal_source_id: "frozen-proposal-generator-v1".to_owned(),
            proposal_stream_sha256: Some(H5.to_owned()),
            task_recipe_sha256: H1.to_owned(),
            comparison_profile_sha256: H2.to_owned(),
            evaluator_sha256: H3.to_owned(),
            verifier_sha256: H4.to_owned(),
            budget_limit_microunits: budget,
            initial_direction_ids: vec!["A".to_owned(), "B".to_owned()],
            execution: ExecutionContract {
                mode,
                worker_count,
                seats_per_round: if mode == ExecutionMode::Sequential { 1 } else { worker_count },
                first_round_claim_cap: None,
                steady_state_claim_cap: None,
            },
            policy: PolicySnapshot {
                policy_id: "direction-board".to_owned(),
                algorithm_id: "symthaea-direction-board-v0.1".to_owned(),
                config_sha256: digest_serializable(&config).unwrap(),
                config,
                has_hidden_oracle_access: false,
            },
        }
    }

    fn event(
        event_id: &str,
        worker_id: u32,
        round_id: u64,
        batch_id: Option<&str>,
        visible_through_sequence: u64,
        intent: ActionIntent,
        outcome: ActionOutcome,
        charge: u64,
    ) -> ActionEvent {
        ActionEvent {
            event_id: event_id.to_owned(),
            policy_id: "direction-board".to_owned(),
            worker_id,
            round_id,
            batch_id: batch_id.map(str::to_owned),
            observed_through_sequence,
            visible_history_sha256: String::new(),
            retry_of_event_id: None,
            intent,
            outcome,
            resources: resources(charge),
        }
    }

    fn verified_evaluation(direction_id: &str, candidate_sha256: &str) -> (ActionIntent, ActionOutcome) {
        (
            ActionIntent::EvaluateCandidate {
                candidate_sha256: candidate_sha256.to_owned(),
                parent_candidate_sha256: None,
                semantic_delta_sha256: H2.to_owned(),
                direction_id: direction_id.to_owned(),
            },
            ActionOutcome::CandidateEvaluation {
                task_recipe_sha256: H1.to_owned(),
                comparison_profile_sha256: H2.to_owned(),
                evaluator_sha256: H3.to_owned(),
                disposition: CandidateDisposition::Verified,
                score: Some(0.91),
                uncertainty: Some(0.01),
                evidence_sha256: Some(H5.to_owned()),
            },
        )
    }

    #[test]
    fn sequential_tape_hashes_history_and_conserves_charged_budget() {
        let mut tape = EventTape::new(header(ExecutionMode::Sequential, 1, 20)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H5);
        tape.append(event("e1", 0, 1, None, 0, intent, outcome, 7)).unwrap();
        let (intent, outcome) = verified_evaluation("B", H4);
        tape.append(event("e2", 0, 2, None, 1, intent, outcome, 8)).unwrap();

        let summary = tape.validate().unwrap();
        assert_eq!(summary.event_count, 2);
        assert_eq!(summary.candidate_evaluations, 2);
        assert_eq!(summary.total_charged_budget_microunits, Some(15));
        assert!(summary.fully_metered);
    }

    #[test]
    fn duplicate_proposal_maps_to_an_existing_canonical_direction() {
        let mut tape = EventTape::new(header(ExecutionMode::Sequential, 1, 20)).unwrap();
        tape.append(event(
            "p1",
            0,
            1,
            None,
            0,
            ActionIntent::ProposeDirection {
                descriptor_sha256: H5.to_owned(),
                parent_direction_id: Some("A".to_owned()),
                semantic_delta_sha256: H2.to_owned(),
            },
            ActionOutcome::DirectionVerification {
                task_recipe_sha256: H1.to_owned(),
                comparison_profile_sha256: H2.to_owned(),
                verifier_sha256: H4.to_owned(),
                verifier_receipt_sha256: H3.to_owned(),
                disposition: ProposalDisposition::Duplicate {
                    canonical_direction_id: "A".to_owned(),
                },
            },
            3,
        ))
        .unwrap();

        let summary = tape.validate().unwrap();
        assert_eq!(summary.duplicate_proposals, 1);
        assert_eq!(summary.admitted_directions, 0);
        assert_eq!(summary.registered_direction_ids, vec!["A", "B"]);
    }

    #[test]
    fn admitted_direction_becomes_evaluable_only_after_verifier_admission() {
        let mut tape = EventTape::new(header(ExecutionMode::Sequential, 1, 20)).unwrap();
        tape.append(event(
            "p1",
            0,
            1,
            None,
            0,
            ActionIntent::ProposeDirection {
                descriptor_sha256: H5.to_owned(),
                parent_direction_id: Some("A".to_owned()),
                semantic_delta_sha256: H2.to_owned(),
            },
            ActionOutcome::DirectionVerification {
                task_recipe_sha256: H1.to_owned(),
                comparison_profile_sha256: H2.to_owned(),
                verifier_sha256: H4.to_owned(),
                verifier_receipt_sha256: H3.to_owned(),
                disposition: ProposalDisposition::Admitted {
                    canonical_direction_id: "C".to_owned(),
                },
            },
            3,
        ))
        .unwrap();
        let (intent, outcome) = verified_evaluation("C", H3);
        tape.append(event("e2", 0, 2, None, 1, intent, outcome, 4)).unwrap();

        let summary = tape.validate().unwrap();
        assert_eq!(summary.admitted_directions, 1);
        assert!(summary.registered_direction_ids.contains(&"C".to_owned()));
    }

    #[test]
    fn unknown_meter_is_retained_but_cannot_qualify() {
        let mut tape = EventTape::new(header(ExecutionMode::Sequential, 1, 20)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H5);
        let mut action = event("e1", 0, 1, None, 0, intent, outcome, 7);
        action.resources.energy_millijoules = MeterValue::Unknown {
            reason: "runner energy counter unavailable".to_owned(),
        };
        tape.append(action).unwrap();

        let summary = tape.validate_integrity().unwrap();
        assert!(!summary.fully_metered);
        assert!(tape.validate().is_err());
    }

    #[test]
    fn unknown_budget_charge_is_not_silently_counted_as_zero() {
        let mut tape = EventTape::new(header(ExecutionMode::Sequential, 1, 20)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H5);
        let mut action = event("e1", 0, 1, None, 0, intent, outcome, 7);
        action.resources.charged_budget_microunits = MeterValue::Unknown {
            reason: "billing receipt unavailable".to_owned(),
        };
        tape.append(action).unwrap();

        assert_eq!(
            tape.validate_integrity()
                .unwrap()
                .total_charged_budget_microunits,
            None
        );
        assert!(tape.validate().is_err());
    }

    #[test]
    fn batch_workers_share_the_same_pre_batch_visibility_cutoff() {
        let mut tape = EventTape::new(header(ExecutionMode::Batched, 2, 20)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H5);
        tape.append(event(
            "w0",
            0,
            1,
            Some("batch-1"),
            0,
            intent,
            outcome,
            5,
        ))
        .unwrap();
        let (intent, outcome) = verified_evaluation("B", H4);
        tape.append(event(
            "w1",
            1,
            1,
            Some("batch-1"),
            0,
            intent,
            outcome,
            5,
        ))
        .unwrap();
        assert_eq!(tape.validate().unwrap().event_count, 2);
    }

    #[test]
    fn same_batch_cannot_observe_a_sibling_workers_result() {
        let mut tape = EventTape::new(header(ExecutionMode::Batched, 2, 20)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H5);
        tape.append(event(
            "w0",
            0,
            1,
            Some("batch-1"),
            0,
            intent,
            outcome,
            5,
        ))
        .unwrap();
        let (intent, outcome) = verified_evaluation("B", H4);
        let result = tape.append(event(
            "w1",
            1,
            1,
            Some("batch-1"),
            1,
            intent,
            outcome,
            5,
        ));
        assert!(result.is_err());
        assert_eq!(tape.events.len(), 1);
    }

    #[test]
    fn budget_overrun_rolls_back_the_append() {
        let mut tape = EventTape::new(header(ExecutionMode::Sequential, 1, 10)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H5);
        tape.append(event("e1", 0, 1, None, 0, intent, outcome, 6)).unwrap();
        let (intent, outcome) = verified_evaluation("B", H4);
        assert!(tape.append(event("e2", 0, 2, None, 1, intent, outcome, 5)).is_err());
        assert_eq!(tape.events.len(), 1);
        assert_eq!(
            tape.validate().unwrap().total_charged_budget_microunits,
            Some(6)
        );
    }

    #[test]
    fn event_tampering_breaks_the_hash_chain() {
        let mut tape = EventTape::new(header(ExecutionMode::Sequential, 1, 20)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H5);
        tape.append(event("e1", 0, 1, None, 0, intent, outcome, 5)).unwrap();
        if let ActionIntent::EvaluateCandidate { direction_id, .. } =
            &mut tape.events[0].payload.intent
        {
            *direction_id = "B".to_owned();
        }
        assert!(tape.validate_integrity().is_err());
    }

    struct AlwaysEvaluateA {
        config_sha256: String,
    }

    impl PolicyReplay for AlwaysEvaluateA {
        fn policy_id(&self) -> &str {
            "direction-board"
        }

        fn config_sha256(&self) -> &str {
            &self.config_sha256
        }

        fn select_action(
            &self,
            _visible_events: &[EventEnvelope],
            _context: &DecisionContext,
        ) -> Result<PolicySelection, String> {
            Ok(PolicySelection::EvaluateDirection {
                direction_id: "A".to_owned(),
            })
        }
    }

    #[test]
    fn policy_replay_uses_the_recorded_visible_prefix() {
        let mut tape = EventTape::new(header(ExecutionMode::Sequential, 1, 20)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H5);
        tape.append(event("e1", 0, 1, None, 0, intent, outcome, 5)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H4);
        tape.append(event("e2", 0, 2, None, 1, intent, outcome, 5)).unwrap();

        tape.verify_policy_replay(&AlwaysEvaluateA {
            config_sha256: tape.header.policy.config_sha256.clone(),
        })
        .unwrap();
    }

    #[test]
    fn claim_caps_are_enforced_per_round_and_canonical_direction() {
        let mut h = header(ExecutionMode::Batched, 2, 20);
        h.execution.first_round_claim_cap = Some(1);
        let mut tape = EventTape::new(h).unwrap();

        let (intent, outcome) = verified_evaluation("A", H5);
        tape.append(event("w0", 0, 1, Some("batch-1"), 0, intent, outcome, 3))
            .unwrap();
        let (intent, outcome) = verified_evaluation("A", H4);
        assert!(tape
            .append(event("w1", 1, 1, Some("batch-1"), 0, intent, outcome, 3))
            .is_err());
        assert_eq!(tape.events.len(), 1);
    }
    #[test]
    fn incomplete_final_batch_cannot_qualify() {
        let mut tape = EventTape::new(header(ExecutionMode::Batched, 4, 20)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H5);
        tape.append(event("w0", 0, 1, Some("batch-1"), 0, intent, outcome, 5))
            .unwrap();
        assert!(tape.validate_integrity().is_ok());
        assert!(tape.validate().is_err(), "one-of-four partial batch cannot qualify");
    }

    #[test]
    fn previous_batch_must_be_complete_before_next_round() {
        let mut tape = EventTape::new(header(ExecutionMode::Batched, 2, 20)).unwrap();
        let (intent, outcome) = verified_evaluation("A", H5);
        tape.append(event("w0", 0, 1, Some("batch-1"), 0, intent, outcome, 5))
            .unwrap();
        let (intent, outcome) = verified_evaluation("B", H4);
        assert!(tape
            .append(event("w0-round2", 0, 2, Some("batch-2"), 1, intent, outcome, 5))
            .is_err());
        assert_eq!(tape.events().len(), 1);
    }

    #[test]
    fn empty_tape_cannot_qualify_as_an_experiment() {
        let tape = EventTape::new(header(ExecutionMode::Sequential, 1, 20)).unwrap();
        assert!(tape.validate_integrity().is_ok());
        assert!(tape.validate().is_err());
    }

}
