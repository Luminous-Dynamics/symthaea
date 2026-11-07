//! Deterministic, advisory-only research direction allocation.
//!
//! The proposer never mints direction identity. Every admitted direction is
//! verifier-assigned and bound to a canonical descriptor fingerprint. The
//! board recommends how to distribute a fixed evaluation budget; it does not
//! schedule workers, execute candidates, control a promotion gate, or prove
//! that the verifier's semantic classifications are correct.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const SCHEMA_VERSION: &str = "ar5b-direction-board-campaign-v0";
const RESERVED_NEW_ID: &str = "NEW";
const MAX_ALLOCATION_SLOTS: usize = 10_000;
const BASIS_POINTS: f64 = 10_000.0;
const EPSILON: f64 = 1e-9;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DirectionStatus {
    Qualified,
    Provisional,
    Quarantined,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationDisposition {
    Confirmed,
    Rejected,
    Inconclusive,
    Invalid,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub evaluation_id: String,
    pub task_family_digest: String,
    pub disposition: ObservationDisposition,
    /// Paired change relative to this direction's own local record.
    /// Inconclusive and invalid observations must declare zero effect.
    pub delta_vs_local_best: f64,
    /// All attempts consume cost, including invalid and inconclusive attempts.
    pub cost_units: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Direction {
    /// Actual in-flight allocation from the independently supplied worker snapshot.
    pub active_evaluations: usize,
    pub active_cost_units: f64,
    /// Canonical identifier assigned by the independent direction verifier.
    pub direction_id: String,
    /// Verifier-issued fingerprint of capability × solution × construction × scope.
    pub canonical_fingerprint: String,
    pub target_capability_digest: String,
    pub solution_object_digest: String,
    pub construction_principle_digest: String,
    pub scope_assumptions_digest: String,
    pub verifier_assignment_ref: String,
    pub protected_floor_audit_ref: String,
    pub status: DirectionStatus,
    pub protected_capability_floor_passed: bool,
    /// Current best score within this direction; never replaced by the global best.
    pub local_best_score: f64,
    pub estimated_cost_per_evaluation: f64,
    /// Maximum share of the *next* advisory budget, in basis points.
    pub max_share_bps: u16,
    /// Complete visible history for this direction, including negative/invalid attempts.
    pub history: Vec<Observation>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NewDirectionOption {
    /// Actual in-flight verifier-intake allocation from the active worker snapshot.
    pub active_evaluations: usize,
    pub active_cost_units: f64,
    /// The option allocates budget to verifier intake, not to a candidate-minted ID.
    pub enabled: bool,
    pub prior_gain_per_cost: f64,
    pub estimated_cost_per_evaluation: f64,
    pub max_share_bps: u16,
    pub verifier_intake_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    /// Total campaign envelope before active and new recommendations.
    pub total_cost_units: f64,
    /// Uncommitted portion of the total envelope available to this advisory allocation.
    pub available_cost_units: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoardConfig {
    pub exploration_weight: f64,
    pub prior_strength: f64,
    /// Used only if no qualified direction has paired observations.
    pub empty_history_prior_gain_per_cost: f64,
    pub max_allocation_slots: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Campaign {
    pub schema_version: String,
    pub campaign_id: String,
    pub budget: Budget,
    pub config: BoardConfig,
    /// Immutable snapshot reference for actual in-flight work; not minted by this board.
    pub active_allocation_snapshot_ref: String,
    pub directions: Vec<Direction>,
    pub new_direction_option: NewDirectionOption,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ScoreBreakdown {
    pub candidate_id: String,
    pub raw_gain_per_cost: f64,
    pub shrunk_gain_per_cost: f64,
    pub exploration_bonus: f64,
    pub potential: f64,
    pub completed_evaluations: usize,
    pub active_evaluations: usize,
    pub active_cost_units: f64,
    pub valid_paired_observations: usize,
    pub virtual_allocations: usize,
    pub eligible: bool,
    pub ineligible_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DecisionStep {
    pub step: usize,
    pub available_before: f64,
    pub scores: Vec<ScoreBreakdown>,
    pub selected_candidate_id: String,
    pub selected_cost_units: f64,
    pub available_after: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Recommendation {
    pub candidate_id: String,
    pub recommended_evaluations: usize,
    pub recommended_cost_units: f64,
    pub initial_potential: f64,
    pub estimated_cost_per_evaluation: f64,
    pub max_share_bps: u16,
    pub local_best_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BoardReport {
    pub schema_version: String,
    pub campaign_id: String,
    pub mode: String,
    pub total_budget_units: f64,
    pub active_budget_cost_units: f64,
    pub reserved_or_unavailable_budget_units: f64,
    pub active_allocation_snapshot_ref: String,
    pub available_budget_units: f64,
    pub allocated_budget_units: f64,
    pub unallocated_budget_units: f64,
    pub decision_limit_reached: bool,
    pub recommendations: Vec<Recommendation>,
    pub decision_trace: Vec<DecisionStep>,
    pub warnings: Vec<String>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, PartialEq)]
struct CandidateState {
    candidate_id: String,
    estimated_cost_per_evaluation: f64,
    max_share_bps: u16,
    local_best_score: Option<f64>,
    completed_evaluations: usize,
    active_evaluations: usize,
    active_cost_units: f64,
    valid_paired_observations: usize,
    raw_gain_per_cost: f64,
    shrunk_gain_per_cost: f64,
    max_budget_cost: f64,
    virtual_allocations: usize,
    allocated_cost: f64,
    initial_potential: Option<f64>,
}

impl CandidateState {
    fn score(
        &self,
        global_completed_evaluations: usize,
        global_virtual_allocations: usize,
        exploration_weight: f64,
    ) -> f64 {
        let observed =
            self.completed_evaluations + self.active_evaluations + self.virtual_allocations;
        let total = global_completed_evaluations + global_virtual_allocations;
        let numerator = ((total as f64) + 2.0).ln().max(0.0);
        let denominator = (observed as f64) + 1.0;
        let exploration = exploration_weight * (numerator / denominator).sqrt()
            / self.estimated_cost_per_evaluation;
        self.shrunk_gain_per_cost + exploration
    }

    fn eligibility(&self, remaining_budget: f64) -> Result<(), String> {
        if self.allocated_cost + self.estimated_cost_per_evaluation
            > self.max_budget_cost
        {
            return Err("direction budget-share cap reached".to_owned());
        }
        if self.estimated_cost_per_evaluation > remaining_budget {
            return Err("remaining campaign budget is below one evaluation cost".to_owned());
        }
        Ok(())
    }
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn finite(value: f64) -> bool {
    value.is_finite()
}

fn positive(value: f64) -> bool {
    finite(value) && value > 0.0
}

fn summarize_history(history: &[Observation]) -> (usize, usize, f64, f64) {
    let mut all_cost = 0.0;
    let mut valid_delta = 0.0;
    let mut valid_count = 0usize;
    for observation in history {
        all_cost += observation.cost_units;
        match observation.disposition {
            ObservationDisposition::Confirmed | ObservationDisposition::Rejected => {
                valid_delta += observation.delta_vs_local_best;
                valid_count += 1;
            }
            ObservationDisposition::Inconclusive | ObservationDisposition::Invalid => {}
        }
    }
    let total_count = history.len();
    let rate = if all_cost > 0.0 {
        valid_delta / all_cost
    } else {
        0.0
    };
    (total_count, valid_count, rate, all_cost)
}

fn campaign_prior(campaign: &Campaign) -> f64 {
    let mut delta = 0.0;
    let mut cost = 0.0;
    let mut has_observations = false;
    for direction in &campaign.directions {
        for observation in &direction.history {
            has_observations = true;
            cost += observation.cost_units;
            if matches!(
                observation.disposition,
                ObservationDisposition::Confirmed | ObservationDisposition::Rejected
            ) {
                delta += observation.delta_vs_local_best;
            }
        }
    }
    if has_observations && cost > 0.0 {
        delta / cost
    } else {
        campaign.config.empty_history_prior_gain_per_cost
    }
}

pub fn validate_campaign(campaign: &Campaign) -> Vec<String> {
    let mut issues = Vec::new();
    if campaign.schema_version != SCHEMA_VERSION {
        issues.push("SCHEMA_VERSION: unsupported campaign schema".to_owned());
    }
    if !non_empty(&campaign.campaign_id) {
        issues.push("CAMPAIGN_ID: campaign_id must be non-empty".to_owned());
    }
    if !positive(campaign.budget.total_cost_units) {
        issues.push("BUDGET: total_cost_units must be finite and positive".to_owned());
    }
    if !finite(campaign.budget.available_cost_units) || campaign.budget.available_cost_units < 0.0 {
        issues.push("BUDGET: available_cost_units must be finite and non-negative".to_owned());
    }
    if !non_empty(&campaign.active_allocation_snapshot_ref) {
        issues.push("ACTIVE_ALLOCATION_SNAPSHOT: an independent snapshot reference is required".to_owned());
    }
    let active_cost_from_directions: f64 = campaign
        .directions
        .iter()
        .map(|direction| direction.active_cost_units)
        .sum();
    let active_new_cost = campaign.new_direction_option.active_cost_units;
    let total_active_cost = active_cost_from_directions + active_new_cost;
    if !finite(total_active_cost)
        || total_active_cost + campaign.budget.available_cost_units
            > campaign.budget.total_cost_units + EPSILON
    {
        issues.push("BUDGET_DOUBLE_COUNT: active plus available budget exceeds total campaign envelope".to_owned());
    }
    if !finite(campaign.config.exploration_weight) || campaign.config.exploration_weight < 0.0 {
        issues.push("EXPLORATION_WEIGHT: must be finite and non-negative".to_owned());
    }
    if !positive(campaign.config.prior_strength) {
        issues.push("PRIOR_STRENGTH: must be finite and positive".to_owned());
    }
    if !finite(campaign.config.empty_history_prior_gain_per_cost) {
        issues.push("EMPTY_HISTORY_PRIOR: must be finite".to_owned());
    }
    if campaign.config.max_allocation_slots == 0
        || campaign.config.max_allocation_slots > MAX_ALLOCATION_SLOTS
    {
        issues.push(format!(
            "MAX_ALLOCATION_SLOTS: must be between 1 and {MAX_ALLOCATION_SLOTS}"
        ));
    }
    if campaign.directions.is_empty() && !campaign.new_direction_option.enabled {
        issues.push("NO_CANDIDATES: at least one qualified direction or NEW option is required".to_owned());
    }

    let mut direction_ids = BTreeSet::new();
    let mut fingerprints = BTreeSet::new();
    let mut evaluation_ids = BTreeSet::new();

    for direction in &campaign.directions {
        if !non_empty(&direction.direction_id) {
            issues.push("DIRECTION_ID: direction_id must be non-empty".to_owned());
        }
        if direction.direction_id == RESERVED_NEW_ID {
            issues.push("RESERVED_DIRECTION_ID: NEW is a board option, not a canonical direction ID".to_owned());
        }
        if !direction_ids.insert(direction.direction_id.clone()) {
            issues.push(format!("DUPLICATE_DIRECTION_ID: {}", direction.direction_id));
        }
        if !is_sha256(&direction.canonical_fingerprint) {
            issues.push(format!("CANONICAL_FINGERPRINT: invalid SHA-256 for {}", direction.direction_id));
        } else if !fingerprints.insert(direction.canonical_fingerprint.clone()) {
            issues.push(format!(
                "DUPLICATE_CANONICAL_FINGERPRINT: {}",
                direction.canonical_fingerprint
            ));
        }
        for (name, digest) in [
            ("target_capability_digest", &direction.target_capability_digest),
            ("solution_object_digest", &direction.solution_object_digest),
            ("construction_principle_digest", &direction.construction_principle_digest),
            ("scope_assumptions_digest", &direction.scope_assumptions_digest),
        ] {
            if !is_sha256(digest) {
                issues.push(format!("{name}: invalid SHA-256 for {}", direction.direction_id));
            }
        }
        if !non_empty(&direction.verifier_assignment_ref) {
            issues.push(format!("VERIFIER_ASSIGNMENT: missing for {}", direction.direction_id));
        }
        if !non_empty(&direction.protected_floor_audit_ref) {
            issues.push(format!("PROTECTED_FLOOR_AUDIT: missing for {}", direction.direction_id));
        }
        if direction.status != DirectionStatus::Qualified {
            issues.push(format!(
                "DIRECTION_NOT_QUALIFIED: {} must be verifier-qualified before allocation",
                direction.direction_id
            ));
        }
        if !direction.protected_capability_floor_passed {
            issues.push(format!(
                "PROTECTED_FLOOR_FAILED: {} cannot receive budget",
                direction.direction_id
            ));
        }
        if !finite(direction.local_best_score) {
            issues.push(format!("LOCAL_BEST: non-finite score for {}", direction.direction_id));
        }
        if !positive(direction.estimated_cost_per_evaluation) {
            issues.push(format!("EVALUATION_COST: invalid cost for {}", direction.direction_id));
        }
        if !finite(direction.active_cost_units) || direction.active_cost_units < 0.0 {
            issues.push(format!("ACTIVE_COST: invalid active cost for {}", direction.direction_id));
        }
        if (direction.active_evaluations == 0) != (direction.active_cost_units == 0.0) {
            issues.push(format!("ACTIVE_ALLOCATION_MISMATCH: count/cost mismatch for {}", direction.direction_id));
        }
        if direction.max_share_bps == 0 || direction.max_share_bps > 10_000 {
            issues.push(format!("MAX_SHARE: invalid basis-point cap for {}", direction.direction_id));
        }

        for observation in &direction.history {
            if !non_empty(&observation.evaluation_id) {
                issues.push(format!("EVALUATION_ID: empty for {}", direction.direction_id));
            }
            if !evaluation_ids.insert(observation.evaluation_id.clone()) {
                issues.push(format!("DUPLICATE_EVALUATION_ID: {}", observation.evaluation_id));
            }
            if !is_sha256(&observation.task_family_digest) {
                issues.push(format!(
                    "TASK_FAMILY_DIGEST: invalid for evaluation {}",
                    observation.evaluation_id
                ));
            }
            if !finite(observation.delta_vs_local_best) {
                issues.push(format!(
                    "OBSERVATION_DELTA: non-finite for {}",
                    observation.evaluation_id
                ));
            }
            if !positive(observation.cost_units) {
                issues.push(format!(
                    "OBSERVATION_COST: invalid for {}",
                    observation.evaluation_id
                ));
            }
            match observation.disposition {
                ObservationDisposition::Confirmed if observation.delta_vs_local_best <= 0.0 => {
                    issues.push(format!(
                        "CONFIRMED_DELTA: must be positive for {}",
                        observation.evaluation_id
                    ));
                }
                ObservationDisposition::Rejected if observation.delta_vs_local_best > 0.0 => {
                    issues.push(format!(
                        "REJECTED_DELTA: must be non-positive for {}",
                        observation.evaluation_id
                    ));
                }
                ObservationDisposition::Inconclusive | ObservationDisposition::Invalid
                    if observation.delta_vs_local_best != 0.0 =>
                {
                    issues.push(format!(
                        "UNSUPPORTED_DELTA: inconclusive/invalid observation must have zero effect for {}",
                        observation.evaluation_id
                    ));
                }
                _ => {}
            }
        }
    }

    let new_option = &campaign.new_direction_option;
    if !finite(new_option.active_cost_units) || new_option.active_cost_units < 0.0 {
        issues.push("NEW_ACTIVE_COST: invalid active allocation cost".to_owned());
    }
    if (new_option.active_evaluations == 0) != (new_option.active_cost_units == 0.0) {
        issues.push("NEW_ACTIVE_ALLOCATION_MISMATCH: count/cost mismatch".to_owned());
    }
    if !new_option.enabled && new_option.active_evaluations > 0 {
        issues.push("NEW_ACTIVE_WHEN_DISABLED: disabled NEW option cannot have active evaluations".to_owned());
    }
    if new_option.enabled {
        if !finite(new_option.prior_gain_per_cost) {
            issues.push("NEW_PRIOR: prior_gain_per_cost must be finite".to_owned());
        }
        if !positive(new_option.estimated_cost_per_evaluation) {
            issues.push("NEW_COST: estimated cost must be finite and positive".to_owned());
        }
        if new_option.max_share_bps == 0 || new_option.max_share_bps > 10_000 {
            issues.push("NEW_MAX_SHARE: enabled NEW option requires a cap in 1..=10000 bps".to_owned());
        }
        if !new_option
            .verifier_intake_ref
            .as_deref()
            .is_some_and(non_empty)
        {
            issues.push("NEW_VERIFIER_INTAKE: enabled NEW option requires a verifier-intake reference".to_owned());
        }
    }

    issues.sort();
    issues.dedup();
    issues
}

fn build_states(campaign: &Campaign, global_prior: f64) -> Vec<CandidateState> {
    let extra_candidate = if campaign.new_direction_option.enabled { 1 } else { 0 };
    let mut states = Vec::with_capacity(campaign.directions.len() + extra_candidate);
    for direction in &campaign.directions {
        let (total_count, valid_count, raw_rate, _) = summarize_history(&direction.history);
        let shrink_weight =
            valid_count as f64 / (valid_count as f64 + campaign.config.prior_strength);
        let shrunk = shrink_weight * raw_rate + (1.0 - shrink_weight) * global_prior;
        states.push(CandidateState {
            candidate_id: direction.direction_id.clone(),
            estimated_cost_per_evaluation: direction.estimated_cost_per_evaluation,
            max_share_bps: direction.max_share_bps,
            local_best_score: Some(direction.local_best_score),
            completed_evaluations: total_count,
            active_evaluations: direction.active_evaluations,
            active_cost_units: direction.active_cost_units,
            valid_paired_observations: valid_count,
            raw_gain_per_cost: raw_rate,
            shrunk_gain_per_cost: shrunk,
            max_budget_cost: campaign.budget.available_cost_units
                * (direction.max_share_bps as f64 / BASIS_POINTS),
            virtual_allocations: 0,
            allocated_cost: 0.0,
            initial_potential: None,
        });
    }
    if campaign.new_direction_option.enabled {
        let option = &campaign.new_direction_option;
        states.push(CandidateState {
            candidate_id: RESERVED_NEW_ID.to_owned(),
            estimated_cost_per_evaluation: option.estimated_cost_per_evaluation,
            max_share_bps: option.max_share_bps,
            local_best_score: None,
            completed_evaluations: 0,
            active_evaluations: option.active_evaluations,
            active_cost_units: option.active_cost_units,
            valid_paired_observations: 0,
            raw_gain_per_cost: option.prior_gain_per_cost,
            shrunk_gain_per_cost: option.prior_gain_per_cost,
            max_budget_cost: campaign.budget.available_cost_units
                * (option.max_share_bps as f64 / BASIS_POINTS),
            virtual_allocations: 0,
            allocated_cost: 0.0,
            initial_potential: None,
        });
    }
    states
}

fn make_score(
    state: &CandidateState,
    global_completed_evaluations: usize,
    global_virtual_allocations: usize,
    exploration_weight: f64,
    remaining_budget: f64,
) -> ScoreBreakdown {
    let potential = state.score(
        global_completed_evaluations,
        global_virtual_allocations,
        exploration_weight,
    );
    let eligibility = state.eligibility(remaining_budget);
    ScoreBreakdown {
        candidate_id: state.candidate_id.clone(),
        raw_gain_per_cost: state.raw_gain_per_cost,
        shrunk_gain_per_cost: state.shrunk_gain_per_cost,
        exploration_bonus: potential - state.shrunk_gain_per_cost,
        potential,
        completed_evaluations: state.completed_evaluations,
        active_evaluations: state.active_evaluations,
        active_cost_units: state.active_cost_units,
        valid_paired_observations: state.valid_paired_observations,
        virtual_allocations: state.virtual_allocations,
        eligible: eligibility.is_ok(),
        ineligible_reason: eligibility.err(),
    }
}

pub fn allocate(campaign: &Campaign) -> Result<BoardReport, Vec<String>> {
    let issues = validate_campaign(campaign);
    if !issues.is_empty() {
        return Err(issues);
    }

    let global_prior = campaign_prior(campaign);
    let mut states = build_states(campaign, global_prior);
    let global_completed_evaluations: usize = states
        .iter()
        .map(|state| state.completed_evaluations + state.active_evaluations)
        .sum();

    let mut remaining = campaign.budget.available_cost_units;
    let mut global_virtual_allocations = 0usize;
    let mut decision_trace = Vec::new();

    for step in 0..campaign.config.max_allocation_slots {
        let scores: Vec<ScoreBreakdown> = states
            .iter()
            .map(|state| {
                make_score(
                    state,
                    global_completed_evaluations,
                    global_virtual_allocations,
                    campaign.config.exploration_weight,
                    remaining,
                )
            })
            .collect();

        if scores.iter().any(|score| !finite(score.potential)) {
            return Err(vec![
                "DERIVED_SCORE: non-finite allocation score; reduce priors or exploration weight"
                    .to_owned(),
            ]);
        }

        let mut selected: Option<(usize, f64)> = None;
        for (index, score) in scores.iter().enumerate() {
            if !score.eligible {
                continue;
            }
            let replace = match selected {
                None => true,
                Some((best_index, best_potential)) => {
                    score.potential > best_potential
                        || (score.potential == best_potential
                            && score.candidate_id.as_str()
                                < states[best_index].candidate_id.as_str())
                }
            };
            if replace {
                selected = Some((index, score.potential));
            }
        }

        let Some((selected_index, selected_potential)) = selected else {
            break;
        };
        let selected_id = states[selected_index].candidate_id.clone();
        let selected_cost = states[selected_index].estimated_cost_per_evaluation;
        if states[selected_index].initial_potential.is_none() {
            states[selected_index].initial_potential = Some(selected_potential);
        }
        states[selected_index].virtual_allocations += 1;
        states[selected_index].allocated_cost += selected_cost;
        remaining = (remaining - selected_cost).max(0.0);
        global_virtual_allocations += 1;

        decision_trace.push(DecisionStep {
            step: step + 1,
            available_before: remaining + selected_cost,
            scores,
            selected_candidate_id: selected_id,
            selected_cost_units: selected_cost,
            available_after: remaining,
        });

        if remaining <= EPSILON {
            break;
        }
    }

    let mut recommendations: Vec<Recommendation> = states
        .iter()
        .filter(|state| state.virtual_allocations > 0)
        .map(|state| Recommendation {
            candidate_id: state.candidate_id.clone(),
            recommended_evaluations: state.virtual_allocations,
            recommended_cost_units: state.allocated_cost,
            active_evaluations: state.active_evaluations,
            active_cost_units: state.active_cost_units,
            initial_potential: state.initial_potential.unwrap_or(0.0),
            estimated_cost_per_evaluation: state.estimated_cost_per_evaluation,
            max_share_bps: state.max_share_bps,
            local_best_score: state.local_best_score,
        })
        .collect();
    recommendations.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let allocated = campaign.budget.available_cost_units - remaining;
    let active_budget_cost_units = states.iter().map(|state| state.active_cost_units).sum();
    let reserved_or_unavailable_budget_units =
        (campaign.budget.total_cost_units
            - active_budget_cost_units
            - campaign.budget.available_cost_units)
            .max(0.0);
    let decision_limit_reached =
        decision_trace.len() == campaign.config.max_allocation_slots && remaining > EPSILON;

    Ok(BoardReport {
        schema_version: SCHEMA_VERSION.to_owned(),
        campaign_id: campaign.campaign_id.clone(),
        mode: "advisory_only".to_owned(),
        total_budget_units: campaign.budget.total_cost_units,
        active_budget_cost_units,
        reserved_or_unavailable_budget_units,
        active_allocation_snapshot_ref: campaign.active_allocation_snapshot_ref.clone(),
        available_budget_units: campaign.budget.available_cost_units,
        allocated_budget_units: allocated,
        unallocated_budget_units: remaining,
        decision_limit_reached,
        recommendations,
        decision_trace,
        warnings: vec![
            "Direction identity is accepted from the verifier; fingerprint uniqueness does not prove semantic non-duplication."
                .to_owned(),
            "Allocation scores are advisory priors, not confidence intervals, causal effects, or promotion decisions."
                .to_owned(),
            "The NEW option buys verifier intake/evaluation capacity; it does not mint a direction ID or authorize a candidate."
                .to_owned(),
            "Protected capability-floor status and paired observations must be independently supplied and audited."
                .to_owned(),
        ],
        claim_ceiling: "A report recommends budget allocation across the supplied verifier-qualified directions under this frozen input and scoring rule. It does not establish direction novelty, causal validity, scientific discovery, or improved outer-loop research capability.".to_owned(),
    })
}

pub fn parse_and_allocate(input: &str) -> Result<BoardReport, String> {
    let campaign: Campaign =
        serde_json::from_str(input).map_err(|error| format!("invalid campaign JSON: {error}"))?;
    allocate(&campaign).map_err(|issues| issues.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const CAMPAIGN_JSON: &str =
        include_str!("../../../docs/research/fixtures/AR5B_DIRECTION_BOARD_SYNTHETIC_CAMPAIGN_V0_2026-11-07.json");

    fn fixture_campaign() -> Campaign {
        serde_json::from_str(CAMPAIGN_JSON).expect("campaign fixture parses")
    }

    fn as_value(campaign: &Campaign) -> Value {
        serde_json::to_value(campaign).expect("campaign serializes")
    }

    fn from_value(value: Value) -> Campaign {
        serde_json::from_value(value).expect("mutated campaign deserializes")
    }

    fn direction(id: &str, digest_char: char) -> Direction {
        let digest = digest_char.to_string().repeat(64);
        Direction {
            active_evaluations: 0,
            active_cost_units: 0.0,
            direction_id: id.to_owned(),
            canonical_fingerprint: digest.clone(),
            target_capability_digest: digest.clone(),
            solution_object_digest: digest.clone(),
            construction_principle_digest: digest.clone(),
            scope_assumptions_digest: digest,
            verifier_assignment_ref: format!("verifier://synthetic/{id}"),
            protected_floor_audit_ref: format!("audit://synthetic/{id}"),
            status: DirectionStatus::Qualified,
            protected_capability_floor_passed: true,
            local_best_score: 0.5,
            estimated_cost_per_evaluation: 10.0,
            max_share_bps: 10_000,
            history: Vec::new(),
        }
    }

    #[test]
    fn allocates_within_budget_and_reports_advisory_mode() {
        let report = allocate(&fixture_campaign()).expect("valid fixture allocates");
        assert_eq!(report.mode, "advisory_only");
        assert!(report.allocated_budget_units <= report.available_budget_units + EPSILON);
        assert!(
            (report.allocated_budget_units + report.unallocated_budget_units
                - report.available_budget_units)
                .abs()
                < EPSILON
        );
    }

    #[test]
    fn under_sampled_low_initial_yield_direction_receives_exploration_budget() {
        let report = allocate(&fixture_campaign()).expect("valid fixture allocates");
        let beta = report
            .recommendations
            .iter()
            .find(|item| item.candidate_id == "direction-beta");
        assert!(beta.is_some(), "low-initial-yield direction was starved");
        assert!(beta.unwrap().recommended_evaluations > 0);
    }

    #[test]
    fn verifier_gated_new_direction_option_receives_a_budget_feasible_slot() {
        let report = allocate(&fixture_campaign()).expect("valid fixture allocates");
        let new_option = report
            .recommendations
            .iter()
            .find(|item| item.candidate_id == RESERVED_NEW_ID);
        assert!(new_option.is_some(), "NEW intake option was never explored");
        assert!(new_option.unwrap().recommended_evaluations > 0);
    }

    #[test]
    fn new_direction_option_respects_its_explicit_share_cap() {
        let campaign = fixture_campaign();
        let report = allocate(&campaign).expect("valid fixture allocates");
        let new_cost = report
            .recommendations
            .iter()
            .find(|item| item.candidate_id == RESERVED_NEW_ID)
            .map(|item| item.recommended_cost_units)
            .unwrap_or(0.0);
        let cap = campaign.budget.available_cost_units
            * (campaign.new_direction_option.max_share_bps as f64 / BASIS_POINTS);
        assert!(new_cost <= cap + EPSILON);
    }

    #[test]
    fn rejects_duplicate_canonical_fingerprint() {
        let mut value = as_value(&fixture_campaign());
        let fingerprint = value["directions"][0]["canonical_fingerprint"].clone();
        value["directions"][1]["canonical_fingerprint"] = fingerprint;
        let report = validate_campaign(&from_value(value));
        assert!(report
            .iter()
            .any(|issue| issue.starts_with("DUPLICATE_CANONICAL_FINGERPRINT")));
    }

    #[test]
    fn rejects_unqualified_direction() {
        let mut value = as_value(&fixture_campaign());
        value["directions"][0]["status"] = Value::String("provisional".to_owned());
        let report = validate_campaign(&from_value(value));
        assert!(report
            .iter()
            .any(|issue| issue.starts_with("DIRECTION_NOT_QUALIFIED")));
    }

    #[test]
    fn rejects_failed_protected_capability_floor() {
        let mut value = as_value(&fixture_campaign());
        value["directions"][0]["protected_capability_floor_passed"] = Value::Bool(false);
        let report = validate_campaign(&from_value(value));
        assert!(report
            .iter()
            .any(|issue| issue.starts_with("PROTECTED_FLOOR_FAILED")));
    }

    #[test]
    fn reports_active_allocation_separately_from_advisory_recommendations() {
        let campaign = fixture_campaign();
        let report = allocate(&campaign).expect("valid fixture allocates");
        assert_eq!(report.active_allocation_snapshot_ref, campaign.active_allocation_snapshot_ref);
        assert_eq!(report.active_budget_cost_units, 20.0);
        assert_eq!(report.reserved_or_unavailable_budget_units, 30.0);
        let alpha = report
            .recommendations
            .iter()
            .find(|item| item.candidate_id == "direction-alpha")
            .expect("alpha receives advisory allocation");
        assert_eq!(alpha.active_evaluations, 1);
        assert_eq!(alpha.active_cost_units, 8.0);
        assert!(alpha.recommended_evaluations > 0);
    }

    #[test]
    fn rejects_active_plus_available_budget_over_total() {
        let mut value = as_value(&fixture_campaign());
        value["budget"]["total_cost_units"] = Value::from(50.0);
        let report = validate_campaign(&from_value(value));
        assert!(report
            .iter()
            .any(|issue| issue.starts_with("BUDGET_DOUBLE_COUNT")));
    }

    #[test]
    fn rejects_active_allocation_count_cost_mismatch() {
        let mut value = as_value(&fixture_campaign());
        value["directions"][0]["active_evaluations"] = Value::from(0);
        let report = validate_campaign(&from_value(value));
        assert!(report
            .iter()
            .any(|issue| issue.starts_with("ACTIVE_ALLOCATION_MISMATCH")));
    }

    #[test]
    fn rejects_invalid_observation_cost() {
        let mut value = as_value(&fixture_campaign());
        value["directions"][0]["history"][0]["cost_units"] = Value::from(0.0);
        let report = validate_campaign(&from_value(value));
        assert!(report
            .iter()
            .any(|issue| issue.starts_with("OBSERVATION_COST")));
    }

    #[test]
    fn inconclusive_attempt_cost_remains_in_gain_per_cost_denominator() {
        let history = vec![
            Observation {
                evaluation_id: "ok".to_owned(),
                task_family_digest: "a".repeat(64),
                disposition: ObservationDisposition::Confirmed,
                delta_vs_local_best: 1.0,
                cost_units: 10.0,
            },
            Observation {
                evaluation_id: "unknown".to_owned(),
                task_family_digest: "b".repeat(64),
                disposition: ObservationDisposition::Inconclusive,
                delta_vs_local_best: 0.0,
                cost_units: 30.0,
            },
        ];
        let (_, valid_count, rate, total_cost) = summarize_history(&history);
        assert_eq!(valid_count, 1);
        assert_eq!(total_cost, 40.0);
        assert!((rate - 0.025).abs() < EPSILON);
    }

    #[test]
    fn rejected_negative_evidence_is_not_dropped() {
        let history = vec![
            Observation {
                evaluation_id: "good".to_owned(),
                task_family_digest: "a".repeat(64),
                disposition: ObservationDisposition::Confirmed,
                delta_vs_local_best: 1.0,
                cost_units: 10.0,
            },
            Observation {
                evaluation_id: "bad".to_owned(),
                task_family_digest: "b".repeat(64),
                disposition: ObservationDisposition::Rejected,
                delta_vs_local_best: -2.0,
                cost_units: 10.0,
            },
        ];
        let (_, valid_count, rate, _) = summarize_history(&history);
        assert_eq!(valid_count, 2);
        assert!((rate + 0.05).abs() < EPSILON);
    }

    #[test]
    fn tie_break_is_deterministic_by_canonical_id() {
        let mut campaign = fixture_campaign();
        campaign.directions.clear();
        campaign.new_direction_option.enabled = false;
        let mut first = direction("direction-z", '1');
        let mut second = direction("direction-a", '2');
        first.max_share_bps = 10_000;
        second.max_share_bps = 10_000;
        first.estimated_cost_per_evaluation = 10.0;
        second.estimated_cost_per_evaluation = 10.0;
        campaign.directions = vec![first, second];
        campaign.budget.available_cost_units = 10.0;
        let report = allocate(&campaign).expect("tie campaign valid");
        assert_eq!(report.recommendations.len(), 1);
        assert_eq!(report.recommendations[0].candidate_id, "direction-a");
    }

    #[test]
    fn preserves_direction_local_best_separately_from_global_result() {
        let report = allocate(&fixture_campaign()).expect("valid fixture allocates");
        let source = &fixture_campaign();
        for recommendation in &report.recommendations {
            if recommendation.candidate_id != RESERVED_NEW_ID {
                let direction = source
                    .directions
                    .iter()
                    .find(|item| item.direction_id == recommendation.candidate_id)
                    .expect("known direction");
                assert_eq!(recommendation.local_best_score, Some(direction.local_best_score));
            } else {
                assert_eq!(recommendation.local_best_score, None);
            }
        }
    }

    #[test]
    fn disabled_new_option_never_receives_budget() {
        let mut campaign = fixture_campaign();
        campaign.new_direction_option.enabled = false;
        let report = allocate(&campaign).expect("known directions allocate");
        assert!(report
            .recommendations
            .iter()
            .all(|item| item.candidate_id != RESERVED_NEW_ID));
    }

    #[test]
    fn rejects_non_zero_effect_for_inconclusive_observation() {
        let mut value = as_value(&fixture_campaign());
        value["directions"][0]["history"][0]["disposition"] =
            Value::String("inconclusive".to_owned());
        value["directions"][0]["history"][0]["delta_vs_local_best"] = Value::from(0.1);
        let report = validate_campaign(&from_value(value));
        assert!(report
            .iter()
            .any(|issue| issue.starts_with("UNSUPPORTED_DELTA")));
    }
}
