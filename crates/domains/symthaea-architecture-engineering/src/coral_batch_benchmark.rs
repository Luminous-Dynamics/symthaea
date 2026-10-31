//! Four-seat batch adapter for the published Coral Discovery direction scheduler.
//!
//! This runs the Coral scheduler equations with the reference four-agent configuration.
//! It is not a reproduction of Coral's complete multi-agent execution/verifier system.
//! All four recommendations in a batch are fixed from the same pre-batch board snapshot;
//! results are recorded only after the batch assignment is complete.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::campaign_tape::{
    ArmClaim, CampaignEvent, CampaignEventPayload, CampaignTape, CampaignTapeConfig,
    CandidateDisposition, DirectionProposalDisposition, ResourceBudget, ResourceUsage,
};
use super::coral_baseline::{
    CoralAllocation, CoralArm, CoralBaselineConfig, CoralDirectionBoard,
};
use super::direction_board_benchmark::{DirectionLandscapeFamily, digest, payoff};

const PROFILE_SHA256: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const TARGET_SCORE: f64 = 0.90;

/// One worker seat selected from a single pre-batch Coral allocation snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoralBatchAction {
    pub action_id: String,
    pub round_id: u64,
    pub worker_id: String,
    pub assigned_arm: CoralArm,
    /// For NEW candidates, this is the verifier's canonical destination if admitted/duplicate.
    pub canonical_direction_id: Option<String>,
    /// Some only for NEW-arm verifier decisions; direction claims have no proposal verdict.
    pub disposition: Option<DirectionProposalDisposition>,
    pub score: Option<f64>,
    pub action_units: u64,
}

/// Coral's requested seat allocation and actually issued actions for one batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoralBatchRound {
    pub round_id: u64,
    pub visible_through_sequence: u64,
    pub requested_allocations: Vec<CoralAllocation>,
    pub issued_actions: Vec<CoralBatchAction>,
    pub idle_seats: usize,
}

/// Reproducible, four-seat Coral scheduler campaign. The full tape is included.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoralBatchCampaignRun {
    pub schema: String,
    pub version: String,
    pub landscape: DirectionLandscapeFamily,
    pub seed: u64,
    pub coral_config: CoralBaselineConfig,
    pub coral_config_sha256: String,
    /// Maximum charged action budget; claim caps may leave seats idle and usage below this.
    pub action_budget_units: u64,
    pub total_action_units: u64,
    pub bootstrap_actions: usize,
    /// Global best captured when the verifier opened D, before the admitting batch had results.
    pub new_direction_opening_global_best: Option<f64>,
    pub scheduled_rounds: Vec<CoralBatchRound>,
    pub best_direction: String,
    pub best_score: f64,
    pub target_score: f64,
    pub actions_to_target: Option<usize>,
    pub oracle_best_score: f64,
    pub simple_regret: f64,
    pub board_sha256: String,
    pub tape: CampaignTape,
    pub tape_sha256: String,
}

impl CoralBatchCampaignRun {
    pub fn validate_integrity(&self) -> Result<(), String> {
        if self.schema != "symthaea.coral-batch-campaign.run"
            || self.version != "0.1.0"
            || self.coral_config != CoralBaselineConfig::default()
            || self.coral_config_sha256 != config_digest(&self.coral_config)?
            || self.tape_sha256 != self.tape.tape_sha256
        {
            return Err("Coral run header/config/tape digest mismatch".to_owned());
        }
        self.tape.validate_integrity().map_err(|error| error.to_string())?;
        if self.tape.policy_id != "coral_discovery_four_seat_reference"
            || self.tape.config.worker_count != 4
            || self.tape.config.first_allocation_round != 2
            || self.tape.config.first_round_direction_claim_cap != 1
            || self.tape.config.steady_state_direction_claim_cap != 2
            || self.tape.config.new_direction_claim_cap != 4
        {
            return Err("Coral run does not preserve the four-seat reference contract".to_owned());
        }
        if self.bootstrap_actions != 3 {
            return Err("Coral run must retain the three shared bootstrap evaluations".to_owned());
        }
        let bootstrap_best = self.tape.events.iter()
            .take(self.bootstrap_actions)
            .filter_map(|event| match &event.payload {
                CampaignEventPayload::CandidateEvaluation { score, .. } => *score,
                _ => None,
            })
            .max_by(f64::total_cmp);
        if self.new_direction_opening_global_best != bootstrap_best
            && self.new_direction_opening_global_best.is_some()
        {
            return Err("new-direction opening best differs from the pre-allocation bootstrap best".to_owned());
        }
        let totals = self.tape.totals().map_err(|error| error.to_string())?;
        if totals.action_units != self.total_action_units
            || totals.action_units > self.action_budget_units
            || self.tape.events.len() != totals.action_units as usize
        {
            return Err("Coral event tape and action accounting do not reconcile".to_owned());
        }
        Ok(())
    }
}

/// Run the published Coral scheduler with its reference four-seat configuration.
///
/// The first three direction evaluations are a shared, policy-independent warm-up.
/// Thereafter, all four advisory seats are allocated from one pre-batch snapshot.
/// The action budget must allow at least one complete four-seat allocation batch.
pub fn run_coral_batch_campaign(
    landscape: DirectionLandscapeFamily,
    action_budget_units: u64,
    seed: u64,
) -> Result<CoralBatchCampaignRun, String> {
    let coral_config = CoralBaselineConfig::default();
    coral_config.validate().map_err(|error| error.to_string())?;
    if coral_config.agent_count != 4
        || coral_config.window_size != 32
        || coral_config.prior_n != 4
        || coral_config.infancy_k != 3
        || (coral_config.kappa - 0.05).abs() > f64::EPSILON
        || coral_config.max_agents_per_direction != 2
        || coral_config.first_round_agents_per_direction != 1
    {
        return Err("reference Coral configuration changed; refusing to label this exact Coral".to_owned());
    }
    if action_budget_units < 7 || (action_budget_units - 3) % 4 != 0 {
        return Err("action budget must be 3 bootstrap actions plus complete four-seat batches".to_owned());
    }

    let coral_config_sha256 = config_digest(&coral_config)?;
    let mut board = CoralDirectionBoard::new(coral_config.clone())
        .map_err(|error| error.to_string())?;
    let action_budget = action_budget_units;
    let task_recipe_sha256 = digest(&format!("coral-synthetic-task:v0.1:{landscape:?}"));
    let evaluator_sha256 = digest("coral-synthetic-evaluator:v0.1");
    let candidate_stream_sha256 = digest(&format!(
        "coral-synthetic-candidate-stream:v0.1:{landscape:?}:{seed}"
    ));
    let tape_config = CampaignTapeConfig {
        task_recipe_sha256,
        evaluator_sha256,
        comparison_profile_sha256: PROFILE_SHA256.to_owned(),
        candidate_stream_sha256,
        worker_count: 4,
        // Round 1 is common warm-up. Round 2 is Coral's first allocation round.
        first_allocation_round: 2,
        first_round_direction_claim_cap: coral_config.first_round_agents_per_direction as u16,
        steady_state_direction_claim_cap: coral_config.max_agents_per_direction as u16,
        // NEW is a separate arm. At most every configured agent can claim it per batch.
        new_direction_claim_cap: coral_config.agent_count as u16,
        initial_direction_ids: vec!["A".to_owned(), "B".to_owned(), "C".to_owned()],
        resource_budget: ResourceBudget {
            max_action_units: action_budget_units,
            max_wall_millis: None,
            max_cpu_millis: None,
            max_gpu_millis: None,
            max_host_memory_peak_mib: None,
            max_gpu_memory_peak_mib: None,
            max_energy_mwh: None,
            max_input_tokens: None,
            max_output_tokens: None,
            max_tool_calls: None,
            max_evaluator_calls: None,
            max_verifier_calls: None,
        },
    };
    let mut tape = CampaignTape::new(
        format!("coral-four-seat-{landscape:?}-{seed}").to_lowercase(),
        "coral_discovery_four_seat_reference",
        coral_config_sha256.clone(),
        tape_config,
    ).map_err(|error| error.to_string())?;

    let mut bootstrap_actions = 0usize;
    let mut direction_eval_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut best_observed: Option<(String, f64)> = None;
    let mut actions_to_target = None;

    // Common bootstrap: all policies should begin with the same A/B/C observations.
    // A zero baseline is explicit so that bootstrap improvements receive valid local credit.
    for (worker_index, direction_id) in ["A", "B", "C"].iter().enumerate() {
        board.register_direction(direction_id).map_err(|error| error.to_string())?;
        let score = payoff(landscape, direction_id, 0);
        let attempt_id = format!("coral-{seed}-bootstrap-{direction_id}");
        board.record_evaluation(
            &attempt_id,
            direction_id,
            Some(score),
            Some(0.0),
            Vec::new(),
            1,
        ).map_err(|error| error.to_string())?;
        direction_eval_counts.insert((*direction_id).to_owned(), 1);
        if best_observed.as_ref().is_none_or(|(_, best)| score > *best) {
            best_observed = Some(((*direction_id).to_owned(), score));
        }
        if actions_to_target.is_none() && score >= TARGET_SCORE {
            actions_to_target = Some(bootstrap_actions + 1);
        }
        append_candidate_event(
            &mut tape,
            1,
            format!("bootstrap-{worker_index}"),
            0,
            ArmClaim::Direction { direction_id: (*direction_id).to_owned() },
            landscape,
            seed,
            &attempt_id,
            *direction_id,
            0,
            score,
        )?;
        bootstrap_actions += 1;
    }

    let mut scheduled_rounds = Vec::new();
    let mut total_action_units = bootstrap_actions as u64;
    let mut scheduled_round = 0usize;
    let mut new_attempt_index = 0usize;

    while total_action_units < action_budget_units {
        scheduled_round += 1;
        let tape_round_id = scheduled_round as u64 + 1;
        let visible_through_sequence = tape.events.len() as u64;
        // One Coral recommendation call; no worker result from this batch is visible to
        // another worker's assignment in the same batch.
        let requested_allocations = board.recommend().map_err(|error| error.to_string())?;
        let mut selected_arms = Vec::with_capacity(coral_config.agent_count);
        for allocation in &requested_allocations {
            for _ in 0..allocation.wanted_agents {
                if selected_arms.len() < coral_config.agent_count {
                    selected_arms.push(allocation.arm.clone());
                }
            }
        }
        if selected_arms.len() != coral_config.agent_count {
            return Err("Coral largest-remainder allocator did not produce four seats".to_owned());
        }

        let remaining_budget = usize::try_from(action_budget_units - total_action_units)
            .map_err(|_| "remaining action budget does not fit usize".to_owned())?;
        if remaining_budget < coral_config.agent_count {
            // Claim caps can leave unspent budget. Do not turn a four-seat reference
            // configuration into a partial batch merely to consume the residual units.
            break;
        }
        let first_coral_allocation_round = scheduled_round == 1;
        let mut direction_claims: BTreeMap<String, usize> = BTreeMap::new();
        let round_parent_scores: BTreeMap<String, Option<f64>> = board.directions.iter()
            .map(|(direction_id, state)| {
                let best = state.evaluations.iter()
                    .filter_map(|evaluation| evaluation.score)
                    .max_by(f64::total_cmp);
                (direction_id.clone(), best.or(Some(0.0)))
            })
            .collect();
        let global_best_at_batch_start = board.global_best_score.or(Some(0.0));
        // The verifier canonicalizes/adopts a genuinely new direction before any batch
        // result is committed. Its opening-best snapshot must be the pre-batch global best.
        let mut new_direction_admission_pending = false;
        if landscape == DirectionLandscapeFamily::NewOperatorBreakthrough
            && !board.directions.contains_key("D")
            && selected_arms.iter().any(|arm| matches!(arm, CoralArm::New))
        {
            board.register_direction("D").map_err(|error| error.to_string())?;
            new_direction_admission_pending = true;
        }
        let mut round_actions = Vec::new();
        let mut staged_events = Vec::new();

        for (seat_index, assigned_arm) in selected_arms.iter().enumerate() {
            let worker_id = format!("coral-round-{scheduled_round}-worker-{}", seat_index + 1);
            let action_id = format!("coral-{seed}-round-{scheduled_round}-worker-{}", seat_index + 1);
            match assigned_arm {
                CoralArm::Direction(direction_id) => {
                    let currently_claimed = *direction_claims.get(direction_id).unwrap_or(&0);
                    if board.validate_claim_capacity(
                        direction_id,
                        currently_claimed,
                        first_coral_allocation_round,
                    ).is_err() {
                        continue;
                    }
                    *direction_claims.entry(direction_id.clone()).or_default() += 1;
                    if !board.directions.contains_key(direction_id) {
                        return Err(format!("Coral recommended unregistered direction {direction_id}"));
                    }
                    let parent_score = round_parent_scores.get(direction_id)
                        .copied()
                        .flatten()
                        .or(Some(0.0));
                    let eval_index = direction_eval_counts.get(direction_id).copied().unwrap_or(0);
                    let score = payoff(landscape, direction_id, eval_index);
                    let attempt_id = format!("{action_id}-evaluation");
                    board.record_evaluation(
                        &attempt_id,
                        direction_id,
                        Some(score),
                        parent_score,
                        Vec::new(),
                        1,
                    ).map_err(|error| error.to_string())?;
                    if best_observed.as_ref().is_none_or(|(_, best)| score > *best) {
                        best_observed = Some((direction_id.clone(), score));
                    }
                    let action_index = (total_action_units + round_actions.len() as u64 + 1) as usize;
                    if actions_to_target.is_none() && score >= TARGET_SCORE {
                        actions_to_target = Some(action_index);
                    }
                    let eval_index = direction_eval_counts.entry(direction_id.clone()).or_default();
                    *eval_index += 1;
                    staged_events.push(make_candidate_event(
                        tape_round_id,
                        worker_id.clone(),
                        visible_through_sequence,
                        ArmClaim::Direction { direction_id: direction_id.clone() },
                        landscape,
                        seed,
                        &attempt_id,
                        direction_id,
                        *eval_index - 1,
                        score,
                    ));
                    round_actions.push(CoralBatchAction {
                        action_id,
                        round_id: tape_round_id,
                        worker_id,
                        assigned_arm: assigned_arm.clone(),
                        canonical_direction_id: Some(direction_id.clone()),
                        disposition: None,
                        score: Some(score),
                        action_units: 1,
                    });
                }
                CoralArm::New => {
                    let attempt_id = format!("{action_id}-novel-attempt");
                    if new_direction_admission_pending {
                        new_direction_admission_pending = false;
                        let direction_id = "D";
                        let eval_index = direction_eval_counts.get(direction_id).copied().unwrap_or(0);
                        let score = payoff(landscape, direction_id, eval_index);
                        board.record_evaluation(
                            &attempt_id,
                            direction_id,
                            Some(score),
                            global_best_at_batch_start,
                            Vec::new(),
                            1,
                        ).map_err(|error| error.to_string())?;
                        *direction_eval_counts.entry(direction_id.to_owned()).or_default() += 1;
                        if best_observed.as_ref().is_none_or(|(_, best)| score > *best) {
                            best_observed = Some((direction_id.to_owned(), score));
                        }
                        let action_index = usize::try_from(
                            total_action_units + round_actions.len() as u64 + 1
                        ).map_err(|_| "action index does not fit usize".to_owned())?;
                        if actions_to_target.is_none() && score >= TARGET_SCORE {
                            actions_to_target = Some(action_index);
                        }
                        let disposition = DirectionProposalDisposition::Admitted {
                            canonical_direction_id: direction_id.to_owned(),
                        };
                        staged_events.push(make_novel_event(
                            tape_round_id,
                            worker_id.clone(),
                            visible_through_sequence,
                            landscape,
                            seed,
                            &attempt_id,
                            new_attempt_index,
                            disposition.clone(),
                            Some(score),
                        ));
                        round_actions.push(CoralBatchAction {
                            action_id,
                            round_id: tape_round_id,
                            worker_id,
                            assigned_arm: CoralArm::New,
                            canonical_direction_id: Some(direction_id.to_owned()),
                            disposition: Some(disposition),
                            score: Some(score),
                            action_units: 1,
                        });
                        new_attempt_index += 1;
                        continue;
                    }

                    if landscape == DirectionLandscapeFamily::NewOperatorBreakthrough
                        && board.directions.contains_key("D")
                    {
                        let direction_id = "D";
                        let parent_score = round_parent_scores.get(direction_id)
                            .copied()
                            .flatten()
                            .or(global_best_at_batch_start);
                        let eval_index = direction_eval_counts.get(direction_id).copied().unwrap_or(0);
                        let score = payoff(landscape, direction_id, eval_index);
                        board.record_evaluation(
                            &attempt_id,
                            direction_id,
                            Some(score),
                            parent_score,
                            Vec::new(),
                            1,
                        ).map_err(|error| error.to_string())?;
                        *direction_eval_counts.entry(direction_id.to_owned()).or_default() += 1;
                        if best_observed.as_ref().is_none_or(|(_, best)| score > *best) {
                            best_observed = Some((direction_id.to_owned(), score));
                        }
                        let action_index = usize::try_from(
                            total_action_units + round_actions.len() as u64 + 1
                        ).map_err(|_| "action index does not fit usize".to_owned())?;
                        if actions_to_target.is_none() && score >= TARGET_SCORE {
                            actions_to_target = Some(action_index);
                        }
                        let disposition = DirectionProposalDisposition::Duplicate {
                            existing_direction_id: direction_id.to_owned(),
                        };
                        staged_events.push(make_novel_event(
                            tape_round_id,
                            worker_id.clone(),
                            visible_through_sequence,
                            landscape,
                            seed,
                            &attempt_id,
                            new_attempt_index,
                            disposition.clone(),
                            Some(score),
                        ));
                        round_actions.push(CoralBatchAction {
                            action_id,
                            round_id: tape_round_id,
                            worker_id,
                            assigned_arm: CoralArm::New,
                            canonical_direction_id: Some(direction_id.to_owned()),
                            disposition: Some(disposition),
                            score: Some(score),
                            action_units: 1,
                        });
                        new_attempt_index += 1;
                    } else {
                        let disposition = DirectionProposalDisposition::Rejected {
                            reason_code: "synthetic_verifier_no_new_grammar_payoff".to_owned(),
                        };
                        staged_events.push(make_novel_event(
                            tape_round_id,
                            worker_id.clone(),
                            visible_through_sequence,
                            landscape,
                            seed,
                            &attempt_id,
                            new_attempt_index,
                            disposition.clone(),
                            None,
                        ));
                        round_actions.push(CoralBatchAction {
                            action_id,
                            round_id: tape_round_id,
                            worker_id,
                            assigned_arm: CoralArm::New,
                            canonical_direction_id: None,
                            disposition: Some(disposition),
                            score: None,
                            action_units: 1,
                        });
                        new_attempt_index += 1;
                    }
                }
            }
        }

        // Append the preselected batch only after all assignments are fixed. Event order is
        // canonical worker seat order and is used only to replay/attribute subsequent rewards.
        for event in staged_events {
            tape.append(event).map_err(|error| error.to_string())?;
        }
        let issued_actions = round_actions.len();
        total_action_units += issued_actions as u64;
        let idle_seats = coral_config.agent_count.saturating_sub(issued_actions);
        scheduled_rounds.push(CoralBatchRound {
            round_id: tape_round_id,
            visible_through_sequence,
            requested_allocations,
            issued_actions: round_actions,
            idle_seats,
        });
        if issued_actions == 0 {
            break;
        }
        if total_action_units >= action_budget_units {
            break;
        }
    }

    let new_direction_opening_global_best = board.directions.get("D")
        .and_then(|state| state.global_best_at_open);
    let board_bytes = serde_json::to_vec(&board)
        .map_err(|error| format!("Coral board serialization failed: {error}"))?;
    let board_sha256 = digest_bytes(&board_bytes);
    let tape_sha256 = tape.seal().map_err(|error| error.to_string())?;
    let (best_direction, best_score) = best_observed
        .ok_or_else(|| "Coral campaign did not record a verified score".to_owned())?;
    let oracle_best_score = oracle_best(landscape);
    let run = CoralBatchCampaignRun {
        schema: "symthaea.coral-batch-campaign.run".to_owned(),
        version: "0.1.0".to_owned(),
        landscape,
        seed,
        coral_config,
        coral_config_sha256,
        action_budget_units,
        total_action_units,
        bootstrap_actions,
        new_direction_opening_global_best,
        scheduled_rounds,
        best_direction,
        best_score,
        target_score: TARGET_SCORE,
        actions_to_target,
        oracle_best_score,
        simple_regret: (oracle_best_score - best_score).max(0.0),
        board_sha256,
        tape,
        tape_sha256,
    };
    run.validate_integrity()?;
    Ok(run)
}

fn append_candidate_event(
    tape: &mut CampaignTape,
    round_id: u64,
    worker_id: String,
    visible_through: u64,
    claimed_arm: ArmClaim,
    landscape: DirectionLandscapeFamily,
    seed: u64,
    attempt_id: &str,
    direction_id: &str,
    eval_index: usize,
    score: f64,
) -> Result<(), String> {
    tape.append(make_candidate_event(
        round_id, worker_id, visible_through, claimed_arm, landscape, seed,
        attempt_id, direction_id, eval_index, score,
    )).map_err(|error| error.to_string())
}

fn make_candidate_event(
    round_id: u64,
    worker_id: String,
    visible_through: u64,
    claimed_arm: ArmClaim,
    landscape: DirectionLandscapeFamily,
    seed: u64,
    attempt_id: &str,
    direction_id: &str,
    eval_index: usize,
    score: f64,
) -> CampaignEvent {
    let candidate = digest(&format!("coral-candidate:{landscape:?}:{seed}:{attempt_id}"));
    CampaignEvent {
        event_id: attempt_id.to_owned(),
        sequence_number: 0,
        round_id,
        worker_id,
        policy_visible_through_sequence: visible_through,
        claimed_arm,
        task_recipe_sha256: digest(&format!("coral-synthetic-task:v0.1:{landscape:?}")),
        evaluator_sha256: digest("coral-synthetic-evaluator:v0.1"),
        comparison_profile_sha256: PROFILE_SHA256.to_owned(),
        payload: CampaignEventPayload::CandidateEvaluation {
            candidate_sha256: candidate.clone(),
            parent_candidate_sha256: digest(&format!("coral-parent:{candidate}:{eval_index}")),
            semantic_delta_sha256: digest(&format!("coral-semantic-delta:{candidate}")),
            direction_id: direction_id.to_owned(),
            disposition: CandidateDisposition::Verified,
            score: Some(score),
        },
        resource_usage: unknown_usage(),
        previous_event_sha256: String::new(),
        event_sha256: String::new(),
    }
}

fn make_novel_event(
    round_id: u64,
    worker_id: String,
    visible_through: u64,
    landscape: DirectionLandscapeFamily,
    seed: u64,
    attempt_id: &str,
    proposal_number: usize,
    disposition: DirectionProposalDisposition,
    score: Option<f64>,
) -> CampaignEvent {
    let candidate = digest(&format!("coral-novel-candidate:{landscape:?}:{seed}:{attempt_id}"));
    CampaignEvent {
        event_id: attempt_id.to_owned(),
        sequence_number: 0,
        round_id,
        worker_id,
        policy_visible_through_sequence: visible_through,
        claimed_arm: ArmClaim::NewDirection,
        task_recipe_sha256: digest(&format!("coral-synthetic-task:v0.1:{landscape:?}")),
        evaluator_sha256: digest("coral-synthetic-evaluator:v0.1"),
        comparison_profile_sha256: PROFILE_SHA256.to_owned(),
        payload: CampaignEventPayload::NovelCandidateEvaluation {
            candidate_sha256: candidate.clone(),
            parent_candidate_sha256: digest(&format!("coral-parent:{candidate}")),
            semantic_delta_sha256: digest(&format!("coral-semantic-delta:{candidate}")),
            proposal_sha256: digest(&format!("coral-novel-proposal:{landscape:?}:{seed}:{proposal_number}")),
            descriptor_sha256: digest("coral-new-direction-descriptor:synthetic-v0.1"),
            disposition,
            score,
        },
        resource_usage: unknown_usage(),
        previous_event_sha256: String::new(),
        event_sha256: String::new(),
    }
}

fn unknown_usage() -> ResourceUsage {
    ResourceUsage {
        action_units: 1,
        wall_millis: None,
        cpu_millis: None,
        gpu_millis: None,
        host_memory_peak_mib: None,
        gpu_memory_peak_mib: None,
        energy_mwh: None,
        input_tokens: None,
        output_tokens: None,
        tool_calls: None,
        evaluator_calls: None,
        verifier_calls: None,
    }
}

fn config_digest(config: &CoralBaselineConfig) -> Result<String, String> {
    let bytes = serde_json::to_vec(config)
        .map_err(|error| format!("Coral config serialization failed: {error}"))?;
    Ok(digest_bytes(&bytes))
}

fn digest_bytes(value: &[u8]) -> String {
    let bytes = Sha256::digest(value);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn oracle_best(landscape: DirectionLandscapeFamily) -> f64 {
    let mut best = 0.0_f64;
    for direction in ["A", "B", "C", "D"] {
        if direction == "D" && landscape != DirectionLandscapeFamily::NewOperatorBreakthrough {
            continue;
        }
        for evaluation_index in 0..16 {
            best = best.max(payoff(landscape, direction, evaluation_index));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_reference_config_is_preserved_and_config_digest_is_bound() {
        let run = run_coral_batch_campaign(
            DirectionLandscapeFamily::NewOperatorBreakthrough,
            7,
            11,
        ).unwrap();
        let config = CoralBaselineConfig::default();
        assert_eq!(run.coral_config, config);
        assert_eq!(run.coral_config.window_size, 32);
        assert_eq!(run.coral_config.prior_n, 4);
        assert_eq!(run.coral_config.infancy_k, 3);
        assert_eq!(run.coral_config.kappa, 0.05);
        assert_eq!(run.coral_config.agent_count, 4);
        assert_eq!(run.coral_config.max_agents_per_direction, 2);
        assert_eq!(run.coral_config.first_round_agents_per_direction, 1);
        assert_eq!(run.coral_config_sha256, config_digest(&config).unwrap());
        run.validate_integrity().unwrap();
    }

    #[test]
    fn each_coral_batch_uses_one_shared_pre_batch_visibility_cutoff() {
        let run = run_coral_batch_campaign(
            DirectionLandscapeFamily::RankReversal,
            11,
            0xabc,
        ).unwrap();
        for round in &run.scheduled_rounds {
            for action in &action_events(&run.tape, round.round_id) {
                assert_eq!(action.policy_visible_through_sequence, round.visible_through_sequence);
                assert!(action.sequence_number > round.visible_through_sequence);
            }
        }
    }

    #[test]
    fn four_seat_rounds_never_claim_more_than_reference_direction_caps() {
        let run = run_coral_batch_campaign(
            DirectionLandscapeFamily::DelayedBreakthrough,
            15,
            0x123,
        ).unwrap();
        assert!(run.scheduled_rounds.iter().all(|round| round.issued_actions.len() <= 4));
        let mut claims: BTreeMap<(u64, String), usize> = BTreeMap::new();
        for event in &run.tape.events {
            if let ArmClaim::Direction { direction_id } = &event.claimed_arm {
                *claims.entry((event.round_id, direction_id.clone())).or_default() += 1;
            }
        }
        for ((round_id, _direction), count) in claims {
            let cap = if round_id <= 2 { 1 } else { 2 };
            assert!(count <= cap);
        }
    }

    #[test]
    fn new_arm_attempts_are_single_charged_scored_or_rejected_events() {
        let run = run_coral_batch_campaign(
            DirectionLandscapeFamily::NewOperatorBreakthrough,
            15,
            0x44,
        ).unwrap();
        for event in &run.tape.events {
            if let CampaignEventPayload::NovelCandidateEvaluation { disposition, score, .. } = &event.payload {
                match disposition {
                    DirectionProposalDisposition::Admitted { .. }
                    | DirectionProposalDisposition::Duplicate { .. } => assert!(score.is_some()),
                    DirectionProposalDisposition::Rejected { .. }
                    | DirectionProposalDisposition::Inconclusive { .. } => assert!(score.is_none()),
                }
            }
        }
        assert_eq!(run.tape.totals().unwrap().action_units, run.total_action_units);
    }

    #[test]
    fn no_new_direction_landscape_rejects_new_candidates_without_fake_scores() {
        let run = run_coral_batch_campaign(
            DirectionLandscapeFamily::NoNewDirection,
            15,
            0x55,
        ).unwrap();
        assert!(run.tape.events.iter().any(|event| matches!(
            &event.payload,
            CampaignEventPayload::NovelCandidateEvaluation {
                disposition: DirectionProposalDisposition::Rejected { .. },
                score: None,
                ..
            }
        )));
    }

    #[test]
    fn incomplete_four_seat_budget_is_rejected() {
        assert!(run_coral_batch_campaign(
            DirectionLandscapeFamily::DelayedBreakthrough,
            10,
            3,
        ).is_err());
    }
}

fn action_events(tape: &CampaignTape, round_id: u64) -> Vec<&CampaignEvent> {
    tape.events.iter().filter(|event| event.round_id == round_id).collect()
}
