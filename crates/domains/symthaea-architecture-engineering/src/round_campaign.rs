//! Shared round/action/cost/evidence contract for matched scheduler campaigns.
//!
//! One round is planned from a frozen pre-round snapshot. All seats execute against that
//! snapshot, and feedback is committed only after the round closes. Hashes are content
//! references, not signatures or proof of verifier honesty.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const ROUND_SCHEMA: &str = "symthaea.architecture-engineering.scheduler-round";
const ROUND_VERSION: &str = "0.1.0";
const ROUND_PLAN_DOMAIN: &[u8] = b"symthaea.scheduler-round-plan.v0.1\0";
const ROUND_CLOSE_DOMAIN: &[u8] = b"symthaea.scheduler-round-close.v0.1\0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundContractError {
    pub path: String,
    pub message: String,
}

impl RoundContractError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for RoundContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for RoundContractError {}

/// Resource dimensions that can be charged to a scheduler action.
/// A present zero is measured zero; an absent key is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostDimension {
    WallTimeMillis,
    CpuTimeMillis,
    GpuTimeMillis,
    PeakHostMemoryBytes,
    PeakGpuMemoryBytes,
    EnergyMicrojoules,
    InputTokens,
    OutputTokens,
    ToolCalls,
    EvaluatorCalls,
    CandidateExecutions,
}

/// Common claim-cap profile. The matched scheduler track applies the same profile
/// to every policy; a faithful-system ablation may use a policy's native caps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimCapProfile {
    pub max_evaluations_per_direction_per_round: usize,
    pub max_new_proposals_per_round: usize,
}

impl ClaimCapProfile {
    pub fn validate(&self, seat_count: usize) -> Result<(), RoundContractError> {
        if self.max_evaluations_per_direction_per_round == 0
            || self.max_new_proposals_per_round == 0
        {
            return Err(RoundContractError::new(
                "claim_caps",
                "both claim limits must be positive",
            ));
        }
        if self.max_evaluations_per_direction_per_round > seat_count
            || self.max_new_proposals_per_round > seat_count
        {
            return Err(RoundContractError::new(
                "claim_caps",
                "a claim limit cannot exceed the number of seats",
            ));
        }
        Ok(())
    }
}

/// Proposal and evaluation are separate paid actions by design. A novel proposal
/// still requires a separately charged evaluation before it can accrue a score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum RoundAction {
    EvaluateDirection(String),
    ProposeDirection(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundSeat {
    pub action_id: String,
    /// Digest of the immutable candidate or proposal work packet.
    pub work_item_sha256: String,
    pub action: RoundAction,
}

/// Immutable plan for one synchronous comparison round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundPlan {
    pub schema: String,
    pub version: String,
    pub round_id: String,
    pub round_index: u64,
    pub policy_id: String,
    pub seed: u64,
    /// Previous round-close digest; absent only for the first round.
    pub parent_close_sha256: Option<String>,
    /// Digest of all policy-visible evidence before any seat in this round executes.
    pub pre_round_snapshot_sha256: String,
    /// Digest of the resource-cost profile pinned for the whole campaign.
    pub cost_profile_sha256: String,
    /// Every dimension in this set must be measured on every action receipt.
    pub required_cost_dimensions: BTreeSet<CostDimension>,
    pub claim_caps: ClaimCapProfile,
    pub seats: Vec<RoundSeat>,
    pub plan_sha256: String,
}

impl RoundPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        round_id: impl Into<String>,
        round_index: u64,
        policy_id: impl Into<String>,
        seed: u64,
        parent_close_sha256: Option<String>,
        pre_round_snapshot_sha256: impl Into<String>,
        cost_profile_sha256: impl Into<String>,
        required_cost_dimensions: BTreeSet<CostDimension>,
        claim_caps: ClaimCapProfile,
        seats: Vec<RoundSeat>,
    ) -> Result<Self, RoundContractError> {
        let mut plan = Self {
            schema: ROUND_SCHEMA.to_owned(),
            version: ROUND_VERSION.to_owned(),
            round_id: round_id.into(),
            round_index,
            policy_id: policy_id.into(),
            seed,
            parent_close_sha256,
            pre_round_snapshot_sha256: pre_round_snapshot_sha256.into(),
            cost_profile_sha256: cost_profile_sha256.into(),
            required_cost_dimensions,
            claim_caps,
            seats,
            plan_sha256: String::new(),
        };
        plan.plan_sha256 = plan.compute_digest()?;
        plan.validate_integrity()?;
        Ok(plan)
    }

    fn compute_digest(&self) -> Result<String, RoundContractError> {
        let mut unsigned = self.clone();
        unsigned.plan_sha256.clear();
        let bytes = serde_json::to_vec(&unsigned)
            .map_err(|error| RoundContractError::new("plan", error.to_string()))?;
        Ok(domain_digest(ROUND_PLAN_DOMAIN, &bytes))
    }

    pub fn validate_integrity(&self) -> Result<(), RoundContractError> {
        if self.schema != ROUND_SCHEMA || self.version != ROUND_VERSION {
            return Err(RoundContractError::new(
                "schema",
                "unsupported scheduler round schema/version",
            ));
        }
        validate_identifier("round_id", &self.round_id)?;
        validate_identifier("policy_id", &self.policy_id)?;
        match (self.round_index, self.parent_close_sha256.as_deref()) {
            (0, None) => {}
            (0, Some(_)) => return Err(RoundContractError::new("parent_close_sha256", "first round cannot name a parent")),
            (_, Some(parent)) => validate_sha256("parent_close_sha256", parent)?,
            (_, None) => return Err(RoundContractError::new("parent_close_sha256", "non-first round must extend a prior close")),
        }
        validate_sha256("pre_round_snapshot_sha256", &self.pre_round_snapshot_sha256)?;
        validate_sha256("cost_profile_sha256", &self.cost_profile_sha256)?;
        validate_sha256("plan_sha256", &self.plan_sha256)?;
        if self.seats.is_empty() {
            return Err(RoundContractError::new("seats", "round must contain at least one seat"));
        }
        if self.required_cost_dimensions.is_empty() {
            return Err(RoundContractError::new(
                "required_cost_dimensions",
                "at least one cost dimension must be pinned",
            ));
        }
        self.claim_caps.validate(self.seats.len())?;

        let mut action_ids = BTreeSet::new();
        let mut direction_claims: BTreeMap<&str, usize> = BTreeMap::new();
        let mut proposal_ids = BTreeSet::new();
        for (index, seat) in self.seats.iter().enumerate() {
            validate_identifier(&format!("seats[{index}].action_id"), &seat.action_id)?;
            validate_sha256(&format!("seats[{index}].work_item_sha256"), &seat.work_item_sha256)?;
            if !action_ids.insert(seat.action_id.as_str()) {
                return Err(RoundContractError::new(
                    format!("seats[{index}].action_id"),
                    "duplicate action ID within round",
                ));
            }
            match &seat.action {
                RoundAction::EvaluateDirection(direction_id) => {
                    validate_identifier(&format!("seats[{index}].direction_id"), direction_id)?;
                    let count = direction_claims.entry(direction_id).or_default();
                    *count += 1;
                    if *count > self.claim_caps.max_evaluations_per_direction_per_round {
                        return Err(RoundContractError::new(
                            format!("seats[{index}].action"),
                            "per-direction evaluation claim cap exceeded",
                        ));
                    }
                }
                RoundAction::ProposeDirection(proposal_id) => {
                    validate_identifier(&format!("seats[{index}].proposal_id"), proposal_id)?;
                    if !proposal_ids.insert(proposal_id.as_str()) {
                        return Err(RoundContractError::new(
                            format!("seats[{index}].proposal_id"),
                            "duplicate proposal ID within round",
                        ));
                    }
                }
            }
        }
        if proposal_ids.len() > self.claim_caps.max_new_proposals_per_round {
            return Err(RoundContractError::new(
                "seats",
                "per-round new-direction proposal cap exceeded",
            ));
        }
        if self.compute_digest()? != self.plan_sha256 {
            return Err(RoundContractError::new(
                "plan_sha256",
                "does not match the round plan contents",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationDisposition {
    Verified,
    Rejected,
    Inconclusive,
    Failed,
    TimedOut,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalDisposition {
    Admitted,
    Duplicate,
    Rejected,
    Inconclusive,
    Failed,
    TimedOut,
}

/// Evaluator/verifier result for one action. Scores use a frozen normalized [0,1] metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action_kind", rename_all = "snake_case")]
pub enum ActionOutcome {
    Evaluation {
        disposition: EvaluationDisposition,
        score: Option<f64>,
        resolved_direction_id: String,
    },
    Proposal {
        disposition: ProposalDisposition,
        resolved_direction_id: Option<String>,
    },
}

/// One result receipt is required for every planned seat, including failed and timed-out work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionResultReceipt {
    pub round_id: String,
    pub action_id: String,
    /// Must equal the pre-round snapshot; same-round results cannot affect another seat.
    pub observed_snapshot_sha256: String,
    pub result_digest_sha256: String,
    pub verifier_receipt_sha256: String,
    pub outcome: ActionOutcome,
}

/// One cost receipt is required for every planned seat, including failures and retries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionCostReceipt {
    pub round_id: String,
    pub action_id: String,
    pub cost_profile_sha256: String,
    /// Discrete campaign budget charged. At least one unit for every executed seat.
    pub charged_budget_units: u64,
    /// Missing required dimensions mean unknown, never zero.
    pub resource_costs: BTreeMap<CostDimension, u64>,
}

/// Closed-round result set. Receipts are validated against the immutable plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundCloseReceipt {
    pub round_id: String,
    pub plan_sha256: String,
    pub results: Vec<ActionResultReceipt>,
    pub costs: Vec<ActionCostReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedRoundSummary {
    pub action_count: usize,
    pub charged_budget_units: u64,
    pub verified_evaluations: usize,
    pub admitted_directions: usize,
    pub duplicate_proposals: usize,
    pub rejected_actions: usize,
    pub inconclusive_actions: usize,
    pub failed_actions: usize,
    pub timed_out_actions: usize,
}

impl RoundCloseReceipt {
    pub fn validate_against(
        &self,
        plan: &RoundPlan,
    ) -> Result<ValidatedRoundSummary, RoundContractError> {
        plan.validate_integrity()?;
        if self.round_id != plan.round_id || self.plan_sha256 != plan.plan_sha256 {
            return Err(RoundContractError::new(
                "round_close",
                "round ID or plan digest does not bind to the supplied plan",
            ));
        }
        if self.results.len() != plan.seats.len() || self.costs.len() != plan.seats.len() {
            return Err(RoundContractError::new(
                "round_close",
                "exactly one result and one cost receipt are required per planned seat",
            ));
        }

        let seats: BTreeMap<&str, &RoundSeat> = plan
            .seats
            .iter()
            .map(|seat| (seat.action_id.as_str(), seat))
            .collect();
        let mut seen_results = BTreeSet::new();
        let mut seen_costs = BTreeSet::new();
        let mut summary = ValidatedRoundSummary {
            action_count: plan.seats.len(),
            charged_budget_units: 0,
            verified_evaluations: 0,
            admitted_directions: 0,
            duplicate_proposals: 0,
            rejected_actions: 0,
            inconclusive_actions: 0,
            failed_actions: 0,
            timed_out_actions: 0,
        };

        for (index, receipt) in self.results.iter().enumerate() {
            let path = format!("results[{index}]");
            if receipt.round_id != plan.round_id {
                return Err(RoundContractError::new(
                    format!("{path}.round_id"),
                    "receipt belongs to a different round",
                ));
            }
            let seat = seats.get(receipt.action_id.as_str()).ok_or_else(|| {
                RoundContractError::new(
                    format!("{path}.action_id"),
                    "receipt does not correspond to a planned seat",
                )
            })?;
            if !seen_results.insert(receipt.action_id.as_str()) {
                return Err(RoundContractError::new(
                    format!("{path}.action_id"),
                    "duplicate result receipt",
                ));
            }
            if receipt.observed_snapshot_sha256 != plan.pre_round_snapshot_sha256 {
                return Err(RoundContractError::new(
                    format!("{path}.observed_snapshot_sha256"),
                    "result was not evaluated against the frozen pre-round snapshot",
                ));
            }
            validate_sha256(&format!("{path}.result_digest_sha256"), &receipt.result_digest_sha256)?;
            validate_sha256(
                &format!("{path}.verifier_receipt_sha256"),
                &receipt.verifier_receipt_sha256,
            )?;
            match (&seat.action, &receipt.outcome) {
                (
                    RoundAction::EvaluateDirection(expected_direction),
                    ActionOutcome::Evaluation {
                        disposition,
                        score,
                        resolved_direction_id,
                    },
                ) => {
                    validate_identifier(&format!("{path}.resolved_direction_id"), resolved_direction_id)?;
                    if expected_direction != resolved_direction_id {
                        return Err(RoundContractError::new(
                            format!("{path}.resolved_direction_id"),
                            "evaluation was attributed to a different direction than planned",
                        ));
                    }
                    match disposition {
                        EvaluationDisposition::Verified => {
                            let value = score.ok_or_else(|| {
                                RoundContractError::new(
                                    format!("{path}.score"),
                                    "verified evaluation requires a score",
                                )
                            })?;
                            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                                return Err(RoundContractError::new(
                                    format!("{path}.score"),
                                    "score must be finite and normalized to [0,1]",
                                ));
                            }
                            summary.verified_evaluations += 1;
                        }
                        EvaluationDisposition::Rejected | EvaluationDisposition::Inconclusive => {
                            if score.is_some() {
                                return Err(RoundContractError::new(
                                    format!("{path}.score"),
                                    "rejected/inconclusive evaluation cannot carry a score",
                                ));
                            }
                            if *disposition == EvaluationDisposition::Rejected {
                                summary.rejected_actions += 1;
                            } else {
                                summary.inconclusive_actions += 1;
                            }
                        }
                        EvaluationDisposition::Failed => {
                            if score.is_some() {
                                return Err(RoundContractError::new(
                                    format!("{path}.score"),
                                    "failed evaluation cannot carry a score",
                                ));
                            }
                            summary.failed_actions += 1;
                        }
                        EvaluationDisposition::TimedOut => {
                            if score.is_some() {
                                return Err(RoundContractError::new(
                                    format!("{path}.score"),
                                    "timed-out evaluation cannot carry a score",
                                ));
                            }
                            summary.timed_out_actions += 1;
                        }
                    }
                }
                (
                    RoundAction::ProposeDirection(_),
                    ActionOutcome::Proposal {
                        disposition,
                        resolved_direction_id,
                    },
                ) => match disposition {
                    ProposalDisposition::Admitted => {
                        let id = resolved_direction_id.as_deref().ok_or_else(|| {
                            RoundContractError::new(
                                format!("{path}.resolved_direction_id"),
                                "admitted proposal requires a canonical direction ID",
                            )
                        })?;
                        validate_identifier(&format!("{path}.resolved_direction_id"), id)?;
                        summary.admitted_directions += 1;
                    }
                    ProposalDisposition::Duplicate => {
                        summary.duplicate_proposals += 1;
                        let id = resolved_direction_id.as_deref().ok_or_else(|| {
                            RoundContractError::new(
                                format!("{path}.resolved_direction_id"),
                                "duplicate proposal must resolve to an existing direction",
                            )
                        })?;
                        validate_identifier(&format!("{path}.resolved_direction_id"), id)?;
                    }
                    ProposalDisposition::Rejected | ProposalDisposition::Inconclusive => {
                        if resolved_direction_id.is_some() {
                            return Err(RoundContractError::new(
                                format!("{path}.resolved_direction_id"),
                                "rejected/inconclusive proposal cannot resolve to a direction",
                            ));
                        }
                        if *disposition == ProposalDisposition::Rejected {
                            summary.rejected_actions += 1;
                        } else {
                            summary.inconclusive_actions += 1;
                        }
                    }
                    ProposalDisposition::Failed => {
                        if resolved_direction_id.is_some() {
                            return Err(RoundContractError::new(
                                format!("{path}.resolved_direction_id"),
                                "failed proposal cannot resolve to a direction",
                            ));
                        }
                        summary.failed_actions += 1;
                    }
                    ProposalDisposition::TimedOut => {
                        if resolved_direction_id.is_some() {
                            return Err(RoundContractError::new(
                                format!("{path}.resolved_direction_id"),
                                "timed-out proposal cannot resolve to a direction",
                            ));
                        }
                        summary.timed_out_actions += 1;
                    }
                },
                _ => {
                    return Err(RoundContractError::new(
                        format!("{path}.outcome"),
                        "outcome action kind does not match the planned action",
                    ));
                }
            }
        }

        for (index, receipt) in self.costs.iter().enumerate() {
            let path = format!("costs[{index}]");
            if receipt.round_id != plan.round_id {
                return Err(RoundContractError::new(
                    format!("{path}.round_id"),
                    "cost receipt belongs to a different round",
                ));
            }
            if !seats.contains_key(receipt.action_id.as_str()) {
                return Err(RoundContractError::new(
                    format!("{path}.action_id"),
                    "cost receipt does not correspond to a planned seat",
                ));
            }
            if !seen_costs.insert(receipt.action_id.as_str()) {
                return Err(RoundContractError::new(
                    format!("{path}.action_id"),
                    "duplicate cost receipt",
                ));
            }
            if receipt.cost_profile_sha256 != plan.cost_profile_sha256 {
                return Err(RoundContractError::new(
                    format!("{path}.cost_profile_sha256"),
                    "cost profile differs from the frozen campaign profile",
                ));
            }
            if receipt.charged_budget_units == 0 {
                return Err(RoundContractError::new(
                    format!("{path}.charged_budget_units"),
                    "every executed action must charge at least one budget unit",
                ));
            }
            for dimension in &plan.required_cost_dimensions {
                if !receipt.resource_costs.contains_key(dimension) {
                    return Err(RoundContractError::new(
                        format!("{path}.resource_costs.{dimension:?}"),
                        "required resource dimension is unknown; missing is not zero",
                    ));
                }
            }
            summary.charged_budget_units = summary
                .charged_budget_units
                .checked_add(receipt.charged_budget_units)
                .ok_or_else(|| RoundContractError::new("costs", "charged budget sum overflow"))?;
        }

        if seen_results.len() != seats.len() || seen_costs.len() != seats.len() {
            return Err(RoundContractError::new(
                "round_close",
                "one or more planned action IDs are missing a result or cost receipt",
            ));
        }
        if summary.charged_budget_units < summary.action_count as u64 {
            return Err(RoundContractError::new(
                "charged_budget_units",
                "total charged budget cannot be lower than executed seats",
            ));
        }
        Ok(summary)
    }

    pub fn digest(&self) -> Result<String, RoundContractError> {
        // Seats may finish asynchronously. Canonicalize by action ID so equivalent sets
        // of receipts produce the same close digest independent of arrival order.
        let mut canonical = self.clone();
        canonical.results.sort_by(|left, right| left.action_id.cmp(&right.action_id));
        canonical.costs.sort_by(|left, right| left.action_id.cmp(&right.action_id));
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| RoundContractError::new("round_close", error.to_string()))?;
        Ok(domain_digest(ROUND_CLOSE_DOMAIN, &bytes))
    }
}

/// Immutable per-round ledger entry used to reconstruct campaign totals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundLedgerEntry {
    pub round_id: String,
    pub round_index: u64,
    pub parent_close_sha256: Option<String>,
    pub plan_sha256: String,
    pub close_sha256: String,
    pub action_count: u64,
    pub charged_budget_units: u64,
}

/// Append-only campaign chain. Aggregate totals are reconstructed from its entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundLedger {
    pub campaign_id: String,
    pub next_round_index: u64,
    pub last_close_sha256: Option<String>,
    pub total_action_count: u64,
    pub total_charged_budget_units: u64,
    pub closed_round_ids: BTreeSet<String>,
    pub entries: Vec<RoundLedgerEntry>,
}

impl RoundLedger {
    pub fn new(campaign_id: impl Into<String>) -> Result<Self, RoundContractError> {
        let campaign_id = campaign_id.into();
        validate_identifier("campaign_id", &campaign_id)?;
        Ok(Self {
            campaign_id,
            next_round_index: 0,
            last_close_sha256: None,
            total_action_count: 0,
            total_charged_budget_units: 0,
            closed_round_ids: BTreeSet::new(),
            entries: Vec::new(),
        })
    }

    /// Rebuild indexes and totals from the complete ordered entry list.
    pub fn validate_integrity(&self) -> Result<(), RoundContractError> {
        validate_identifier("campaign_id", &self.campaign_id)?;
        if self.next_round_index != self.entries.len() as u64 {
            return Err(RoundContractError::new(
                "next_round_index",
                "does not equal the number of closed round entries",
            ));
        }
        let mut ids = BTreeSet::new();
        let mut expected_parent: Option<String> = None;
        let mut actions = 0u64;
        let mut charged = 0u64;
        for (index, entry) in self.entries.iter().enumerate() {
            validate_identifier(&format!("entries[{index}].round_id"), &entry.round_id)?;
            if entry.round_index != index as u64 {
                return Err(RoundContractError::new(
                    format!("entries[{index}].round_index"),
                    "round indexes must be contiguous and ordered",
                ));
            }
            if !ids.insert(entry.round_id.clone()) {
                return Err(RoundContractError::new(
                    format!("entries[{index}].round_id"),
                    "duplicate round ID in ledger",
                ));
            }
            match (expected_parent.as_deref(), entry.parent_close_sha256.as_deref()) {
                (None, None) => {}
                (Some(expected), Some(actual)) if expected == actual => {}
                _ => {
                    return Err(RoundContractError::new(
                        format!("entries[{index}].parent_close_sha256"),
                        "entry does not extend the preceding close digest",
                    ));
                }
            }
            validate_sha256(&format!("entries[{index}].plan_sha256"), &entry.plan_sha256)?;
            validate_sha256(&format!("entries[{index}].close_sha256"), &entry.close_sha256)?;
            if entry.action_count == 0 || entry.charged_budget_units < entry.action_count {
                return Err(RoundContractError::new(
                    format!("entries[{index}]"),
                    "action count must be positive and charged budget cannot be lower",
                ));
            }
            actions = actions
                .checked_add(entry.action_count)
                .ok_or_else(|| RoundContractError::new("total_action_count", "overflow"))?;
            charged = charged
                .checked_add(entry.charged_budget_units)
                .ok_or_else(|| RoundContractError::new("total_charged_budget_units", "overflow"))?;
            expected_parent = Some(entry.close_sha256.clone());
        }

        if ids != self.closed_round_ids {
            return Err(RoundContractError::new(
                "closed_round_ids",
                "round ID index does not reconcile with ledger entries",
            ));
        }
        if actions != self.total_action_count || charged != self.total_charged_budget_units {
            return Err(RoundContractError::new(
                "totals",
                "stored action or budget totals do not reconcile with round entries",
            ));
        }
        if expected_parent != self.last_close_sha256 {
            return Err(RoundContractError::new(
                "last_close_sha256",
                "ledger tip does not match the final round entry",
            ));
        }
        Ok(())
    }

    pub fn append(
        &mut self,
        plan: &RoundPlan,
        close: &RoundCloseReceipt,
    ) -> Result<ValidatedRoundSummary, RoundContractError> {
        self.validate_integrity()?;
        plan.validate_integrity()?;
        if plan.round_index != self.next_round_index {
            return Err(RoundContractError::new(
                "round_index",
                "rounds must be appended in strict contiguous order",
            ));
        }
        if self.closed_round_ids.contains(&plan.round_id) {
            return Err(RoundContractError::new("round_id", "round ID was already closed"));
        }
        match (&self.last_close_sha256, plan.parent_close_sha256.as_deref()) {
            (None, None) => {}
            (Some(expected), Some(actual)) if expected == actual => {}
            _ => {
                return Err(RoundContractError::new(
                    "parent_close_sha256",
                    "round does not extend the current ledger tip",
                ));
            }
        }
        let summary = close.validate_against(plan)?;
        let close_digest = close.digest()?;
        let entry = RoundLedgerEntry {
            round_id: plan.round_id.clone(),
            round_index: plan.round_index,
            parent_close_sha256: plan.parent_close_sha256.clone(),
            plan_sha256: plan.plan_sha256.clone(),
            close_sha256: close_digest.clone(),
            action_count: summary.action_count as u64,
            charged_budget_units: summary.charged_budget_units,
        };
        self.total_action_count = self
            .total_action_count
            .checked_add(entry.action_count)
            .ok_or_else(|| RoundContractError::new("total_action_count", "overflow"))?;
        self.total_charged_budget_units = self
            .total_charged_budget_units
            .checked_add(entry.charged_budget_units)
            .ok_or_else(|| RoundContractError::new("total_charged_budget_units", "overflow"))?;
        self.closed_round_ids.insert(plan.round_id.clone());
        self.entries.push(entry);
        self.last_close_sha256 = Some(close_digest);
        self.next_round_index += 1;
        self.validate_integrity()?;
        Ok(summary)
    }
}

fn validate_identifier(field: &str, value: &str) -> Result<(), RoundContractError> {
    if value.trim().is_empty()
        || value.len() > 128
        || value.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(RoundContractError::new(
            field,
            "must be non-empty, at most 128 bytes, and contain no control characters",
        ));
    }
    Ok(())
}

fn validate_sha256(field: &str, value: &str) -> Result<(), RoundContractError> {
    let valid = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !valid {
        return Err(RoundContractError::new(
            field,
            "must be a lowercase 64-character SHA-256 hex digest",
        ));
    }
    Ok(())
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        format!("{:064x}", value.len() + 1)
    }

    fn plan() -> RoundPlan {
        let caps = ClaimCapProfile {
            max_evaluations_per_direction_per_round: 2,
            max_new_proposals_per_round: 2,
        };
        let required = [CostDimension::WallTimeMillis, CostDimension::CandidateExecutions]
            .into_iter()
            .collect();
        RoundPlan::new(
            "round-0",
            0,
            "coral-faithful",
            17,
            None,
            digest("snapshot"),
            digest("cost-profile"),
            required,
            caps,
            vec![
                RoundSeat {
                    action_id: "a-1".to_owned(),
                    work_item_sha256: digest("work-a1"),
                    action: RoundAction::EvaluateDirection("A".to_owned()),
                },
                RoundSeat {
                    action_id: "b-1".to_owned(),
                    work_item_sha256: digest("work-b1"),
                    action: RoundAction::EvaluateDirection("B".to_owned()),
                },
                RoundSeat {
                    action_id: "c-1".to_owned(),
                    work_item_sha256: digest("work-c1"),
                    action: RoundAction::ProposeDirection("proposal-1".to_owned()),
                },
                RoundSeat {
                    action_id: "c-2".to_owned(),
                    work_item_sha256: digest("work-c2"),
                    action: RoundAction::EvaluateDirection("C".to_owned()),
                },
            ],
        )
        .unwrap()
    }

    fn close_for(plan: &RoundPlan) -> RoundCloseReceipt {
        let results = plan
            .seats
            .iter()
            .map(|seat| {
                let outcome = match &seat.action {
                    RoundAction::EvaluateDirection(direction_id) => ActionOutcome::Evaluation {
                        disposition: EvaluationDisposition::Verified,
                        score: Some(0.75),
                        resolved_direction_id: direction_id.clone(),
                    },
                    RoundAction::ProposeDirection(_) => ActionOutcome::Proposal {
                        disposition: ProposalDisposition::Admitted,
                        resolved_direction_id: Some("D".to_owned()),
                    },
                };
                ActionResultReceipt {
                    round_id: plan.round_id.clone(),
                    action_id: seat.action_id.clone(),
                    observed_snapshot_sha256: plan.pre_round_snapshot_sha256.clone(),
                    result_digest_sha256: digest(&seat.action_id),
                    verifier_receipt_sha256: digest(&format!("verifier:{}", seat.action_id)),
                    outcome,
                }
            })
            .collect();
        let costs = plan
            .seats
            .iter()
            .map(|seat| ActionCostReceipt {
                round_id: plan.round_id.clone(),
                action_id: seat.action_id.clone(),
                cost_profile_sha256: plan.cost_profile_sha256.clone(),
                charged_budget_units: 1,
                resource_costs: [
                    (CostDimension::WallTimeMillis, 10),
                    (CostDimension::CandidateExecutions, 1),
                ]
                .into_iter()
                .collect(),
            })
            .collect();
        RoundCloseReceipt {
            round_id: plan.round_id.clone(),
            plan_sha256: plan.plan_sha256.clone(),
            results,
            costs,
        }
    }

    #[test]
    fn round_plan_seals_and_validates_exact_seat_actions() {
        let plan = plan();
        plan.validate_integrity().unwrap();
        assert_eq!(plan.seats.len(), 4);
    }

    #[test]
    fn duplicate_action_ids_fail_closed() {
        let mut plan = plan();
        plan.seats[1].action_id = plan.seats[0].action_id.clone();
        plan.plan_sha256 = plan.compute_digest().unwrap();
        assert!(plan.validate_integrity().is_err());
    }

    #[test]
    fn per_direction_claim_cap_is_enforced_before_execution() {
        let mut plan = plan();
        plan.seats[1].action = RoundAction::EvaluateDirection("A".to_owned());
        plan.seats[3].action = RoundAction::EvaluateDirection("A".to_owned());
        plan.plan_sha256 = plan.compute_digest().unwrap();
        assert!(plan.validate_integrity().is_err());
    }

    #[test]
    fn duplicate_proposal_ids_fail_closed() {
        let mut plan = plan();
        plan.seats[3].action = RoundAction::ProposeDirection("proposal-1".to_owned());
        plan.plan_sha256 = plan.compute_digest().unwrap();
        assert!(plan.validate_integrity().is_err());
    }

    #[test]
    fn round_close_requires_one_result_and_cost_per_seat() {
        let plan = plan();
        let mut close = close_for(&plan);
        close.results.pop();
        assert!(close.validate_against(&plan).is_err());

        let mut close = close_for(&plan);
        close.costs.pop();
        assert!(close.validate_against(&plan).is_err());
    }

    #[test]
    fn missing_required_cost_dimension_is_unknown_not_zero() {
        let plan = plan();
        let mut close = close_for(&plan);
        close.costs[0]
            .resource_costs
            .remove(&CostDimension::CandidateExecutions);
        assert!(close.validate_against(&plan).is_err());
    }

    #[test]
    fn stale_snapshot_receipt_is_rejected_to_prevent_feedback_leakage() {
        let plan = plan();
        let mut close = close_for(&plan);
        close.results[0].observed_snapshot_sha256 = digest("later-snapshot");
        assert!(close.validate_against(&plan).is_err());
    }

    #[test]
    fn action_outcome_must_match_planned_action_kind() {
        let plan = plan();
        let mut close = close_for(&plan);
        close.results[0].outcome = ActionOutcome::Proposal {
            disposition: ProposalDisposition::Rejected,
            resolved_direction_id: None,
        };
        assert!(close.validate_against(&plan).is_err());
    }

    #[test]
    fn failures_are_charged_and_preserved_in_the_summary() {
        let plan = plan();
        let mut close = close_for(&plan);
        close.results[0].outcome = ActionOutcome::Evaluation {
            disposition: EvaluationDisposition::Failed,
            score: None,
            resolved_direction_id: "A".to_owned(),
        };
        let summary = close.validate_against(&plan).unwrap();
        assert_eq!(summary.failed_actions, 1);
        assert_eq!(summary.charged_budget_units, 4);
    }

    #[test]
    fn ledger_requires_contiguous_rounds_and_parent_close_chain() {
        let plan = plan();
        let close = close_for(&plan);
        let mut ledger = RoundLedger::new("campaign-1").unwrap();
        let summary = ledger.append(&plan, &close).unwrap();
        assert_eq!(summary.verified_evaluations, 3);
        assert_eq!(summary.admitted_directions, 1);
        assert_eq!(ledger.total_action_count, 4);
        assert_eq!(ledger.total_charged_budget_units, 4);
        ledger.validate_integrity().unwrap();

        let parent = ledger.last_close_sha256.clone().unwrap();
        let next = RoundPlan::new(
            "round-1",
            1,
            "coral-faithful",
            17,
            Some(parent.clone()),
            digest("snapshot-1"),
            digest("cost-profile"),
            [
                CostDimension::WallTimeMillis,
                CostDimension::CandidateExecutions,
            ]
            .into_iter()
            .collect(),
            ClaimCapProfile {
                max_evaluations_per_direction_per_round: 1,
                max_new_proposals_per_round: 1,
            },
            vec![RoundSeat {
                action_id: "a-2".to_owned(),
                work_item_sha256: digest("work-a2"),
                action: RoundAction::EvaluateDirection("A".to_owned()),
            }],
        )
        .unwrap();
        let next_close = RoundCloseReceipt {
            round_id: next.round_id.clone(),
            plan_sha256: next.plan_sha256.clone(),
            results: vec![ActionResultReceipt {
                round_id: next.round_id.clone(),
                action_id: "a-2".to_owned(),
                observed_snapshot_sha256: next.pre_round_snapshot_sha256.clone(),
                result_digest_sha256: digest("next-result"),
                verifier_receipt_sha256: digest("next-verifier"),
                outcome: ActionOutcome::Evaluation {
                    disposition: EvaluationDisposition::Verified,
                    score: Some(0.8),
                    resolved_direction_id: "A".to_owned(),
                },
            }],
            costs: vec![ActionCostReceipt {
                round_id: next.round_id.clone(),
                action_id: "a-2".to_owned(),
                cost_profile_sha256: next.cost_profile_sha256.clone(),
                charged_budget_units: 1,
                resource_costs: [
                    (CostDimension::WallTimeMillis, 9),
                    (CostDimension::CandidateExecutions, 1),
                ]
                .into_iter()
                .collect(),
            }],
        };
        ledger.append(&next, &next_close).unwrap();
        assert_eq!(ledger.next_round_index, 2);
        assert_eq!(ledger.entries.len(), 2);
        ledger.validate_integrity().unwrap();
        let mut altered_totals = ledger.clone();
        altered_totals.total_action_count += 1;
        assert!(altered_totals.validate_integrity().is_err());
        assert!(ledger.append(&next, &next_close).is_err());
    }

    #[test]
    fn close_digest_is_independent_of_async_receipt_arrival_order() {
        let plan = plan();
        let close = close_for(&plan);
        let mut reordered = close.clone();
        reordered.results.reverse();
        reordered.costs.reverse();
        assert_eq!(close.digest().unwrap(), reordered.digest().unwrap());
    }

    #[test]
    fn duplicate_proposals_have_a_distinct_auditable_count() {
        let plan = plan();
        let mut close = close_for(&plan);
        close.results[2].outcome = ActionOutcome::Proposal {
            disposition: ProposalDisposition::Duplicate,
            resolved_direction_id: Some("A".to_owned()),
        };
        let summary = close.validate_against(&plan).unwrap();
        assert_eq!(summary.admitted_directions, 0);
        assert_eq!(summary.duplicate_proposals, 1);
    }

    #[test]
    fn unknown_or_zero_action_charge_is_rejected() {
        let plan = plan();
        let mut close = close_for(&plan);
        close.costs[0].charged_budget_units = 0;
        assert!(close.validate_against(&plan).is_err());
    }
}
