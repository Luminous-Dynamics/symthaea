//! Per-individual multi-action decision-transfer evaluation.
//!
//! For one individual and one frozen decision point, compare the actions
//! selected from history-conditioned latent utility readouts and decoded
//! imagined-rollout utility predictions against independently realized
//! outcomes for the same complete action set.
//!
//! This is an evaluator-side finite scorecard. It does not authenticate
//! referenced receipts, identify causal effects, train a model, optimize a
//! policy, or perform statistical inference. The trusted launcher must verify
//! upstream branch/trajectory evidence and enforce #9437 isolation.
//!
//! Utilities are integer units on one frozen task/loss scale; higher is better.
//! Action rows are canonicalized by action ID before evaluation. Ties in a
//! predicted argmax select the lexicographically first action ID. Realized
//! epsilon-good membership is computed against the best realized utility.
//! Ranking components retain ties on both axes so Kendall tau-b can be
//! reconstructed without treating ties as concordant or discordant.

use super::individual_effect_retention::{BranchDisposition, NonApplicationReason};
use std::collections::BTreeSet;
use std::fmt;

pub const MAX_DECISION_TRANSFER_ACTIONS: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionTransferContext {
    pub decision_point_id: u64,
    pub individual_id_sha256: [u8; 32],
    pub factual_history_sha256: [u8; 32],
    /// Full admitted simulator/task branch-point snapshot identity, not merely
    /// a visible observation digest.
    pub branch_point_snapshot_sha256: [u8; 32],
    pub exogenous_forcing_manifest_sha256: [u8; 32],
    pub task_sha256: [u8; 32],
    pub loss_scale_sha256: [u8; 32],
    pub evaluation_split_sha256: [u8; 32],
    pub evaluator_profile_sha256: [u8; 32],
    pub model_configuration_sha256: [u8; 32],
    pub latent_representation_profile_sha256: [u8; 32],
    pub latent_state_sha256: [u8; 32],
    pub latent_readout_profile_sha256: [u8; 32],
    pub rollout_profile_sha256: [u8; 32],
    pub pairing_verifier_profile_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionTransferAction {
    pub action_id: String,
    pub run_id: u64,
    pub factual_history_sha256: [u8; 32],
    pub branch_point_snapshot_sha256: [u8; 32],
    pub exogenous_forcing_manifest_sha256: [u8; 32],
    pub model_configuration_sha256: [u8; 32],
    pub latent_representation_profile_sha256: [u8; 32],
    pub latent_state_sha256: [u8; 32],
    /// References are verified by the trusted evaluator launcher; nonzero
    /// digests alone do not authenticate the underlying artifacts.
    pub pairing_receipt_sha256: [u8; 32],
    pub trajectory_sha256: [u8; 32],
    pub disposition: BranchDisposition,
    pub latent_predicted_utility_units: i64,
    pub rollout_predicted_utility_units: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndividualDecisionTransferInput {
    pub context: DecisionTransferContext,
    /// Complete action set declared before inspecting evaluator outcomes.
    /// Input order need not be canonical; output rows are sorted by action ID.
    pub expected_action_ids: Vec<String>,
    /// Nonnegative tolerance on the same integer utility scale as each row.
    pub epsilon_units: i64,
    pub actions: Vec<DecisionTransferAction>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionTransferActionResult {
    pub action_id: String,
    pub run_id: u64,
    pub applied_at_us: u64,
    pub pairing_receipt_sha256: [u8; 32],
    pub trajectory_sha256: [u8; 32],
    pub realized_utility_units: i64,
    pub latent_predicted_utility_units: i64,
    pub rollout_predicted_utility_units: i64,
    pub realized_regret_units: i128,
    pub is_realized_top1: bool,
    pub is_realized_epsilon_good: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PairwiseRankingCounts {
    /// Predicted and realized orders agree, with neither side tied.
    pub concordant_pairs: u64,
    /// Predicted and realized orders disagree, with neither side tied.
    pub discordant_pairs: u64,
    /// Realized utility ties while the prediction orders the pair.
    pub realized_only_ties: u64,
    /// Predicted utility ties while realized utility orders the pair.
    pub predicted_only_ties: u64,
    /// Both realized and predicted utility tie.
    pub both_tied_pairs: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionSelection {
    pub action_id: String,
    pub predicted_utility_units: i64,
    pub realized_utility_units: i64,
    pub realized_regret_units: i128,
    /// True for any selected action attaining the exact realized maximum.
    pub is_realized_top1: bool,
    pub is_realized_epsilon_good: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndividualDecisionTransferCertificate {
    /// Complete frozen context is echoed so the result remains bound to the
    /// exact individual, history, snapshot, exogenous realization and model.
    pub context: DecisionTransferContext,
    pub decision_point_id: u64,
    pub individual_id_sha256: [u8; 32],
    pub action_count: u64,
    pub epsilon_units: i64,
    pub realized_best_utility_units: i64,
    /// Canonically ordered; all actions within epsilon of the realized maximum.
    pub realized_epsilon_good_action_ids: Vec<String>,
    pub latent_selection: DecisionSelection,
    pub rollout_selection: DecisionSelection,
    /// Positive means rollout selection has greater realized regret than the
    /// latent-readout selection.
    pub rollout_minus_latent_regret_units: i128,
    pub latent_ranking: PairwiseRankingCounts,
    pub rollout_ranking: PairwiseRankingCounts,
    /// Sorted by action ID for deterministic serialization.
    pub actions: Vec<DecisionTransferActionResult>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecisionTransferError {
    ZeroDecisionPointId,
    EmptyDigest { field: &'static str },
    NegativeEpsilon { epsilon_units: i64 },
    EmptyExpectedActionSet,
    TooManyActions { requested: usize, maximum: usize },
    TooFewActions { requested: usize },
    EmptyActionId,
    DuplicateExpectedActionId { action_id: String },
    DuplicateActionId { action_id: String },
    DuplicateRunId { run_id: u64 },
    ZeroRunId { action_id: String },
    ActionSetMismatch {
        missing_action_ids: Vec<String>,
        unexpected_action_ids: Vec<String>,
    },
    ProvenanceMismatch { action_id: String, field: &'static str },
    ActionNotApplied {
        action_id: String,
        terminal_at_us: u64,
        reason: NonApplicationReason,
    },
    ArithmeticOverflow,
}

impl fmt::Display for DecisionTransferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDecisionPointId => write!(formatter, "decision point ID must be nonzero"),
            Self::EmptyDigest { field } => write!(formatter, "required provenance digest {field} is empty"),
            Self::NegativeEpsilon { epsilon_units } => {
                write!(formatter, "epsilon must be nonnegative, received {epsilon_units}")
            }
            Self::EmptyExpectedActionSet => write!(formatter, "expected action set must not be empty"),
            Self::TooFewActions { requested } => write!(formatter, "decision-transfer requires at least two actions, received {requested}"),
            Self::TooManyActions { requested, maximum } => write!(
                formatter,
                "decision-transfer action count {requested} exceeds the bounded maximum {maximum}"
            ),
            Self::EmptyActionId => write!(formatter, "action IDs must be nonempty and trimmed"),
            Self::DuplicateExpectedActionId { action_id } => {
                write!(formatter, "expected action set repeats ID {action_id}")
            }
            Self::DuplicateActionId { action_id } => {
                write!(formatter, "decision-transfer rows repeat action ID {action_id}")
            }
            Self::DuplicateRunId { run_id } => write!(formatter, "decision-transfer rows repeat run ID {run_id}"),
            Self::ZeroRunId { action_id } => write!(formatter, "action {action_id} has zero run ID"),
            Self::ActionSetMismatch { missing_action_ids, unexpected_action_ids } => write!(
                formatter,
                "action rows differ from the declared complete action set (missing: {missing_action_ids:?}; unexpected: {unexpected_action_ids:?})"
            ),
            Self::ProvenanceMismatch { action_id, field } => {
                write!(formatter, "action {action_id} does not match frozen provenance field {field}")
            }
            Self::ActionNotApplied { action_id, terminal_at_us, reason } => write!(
                formatter,
                "action {action_id} was not applied at terminal time {terminal_at_us}: {reason:?}"
            ),
            Self::ArithmeticOverflow => write!(formatter, "decision-transfer arithmetic overflowed"),
        }
    }
}

impl std::error::Error for DecisionTransferError {}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn require_digest(
    digest: &[u8; 32],
    field: &'static str,
) -> Result<(), DecisionTransferError> {
    if digest_is_empty(digest) {
        return Err(DecisionTransferError::EmptyDigest { field });
    }
    Ok(())
}

fn validate_context(context: &DecisionTransferContext) -> Result<(), DecisionTransferError> {
    if context.decision_point_id == 0 {
        return Err(DecisionTransferError::ZeroDecisionPointId);
    }
    for (digest, field) in [
        (&context.individual_id_sha256, "individual_id_sha256"),
        (&context.factual_history_sha256, "factual_history_sha256"),
        (&context.branch_point_snapshot_sha256, "branch_point_snapshot_sha256"),
        (&context.exogenous_forcing_manifest_sha256, "exogenous_forcing_manifest_sha256"),
        (&context.task_sha256, "task_sha256"),
        (&context.loss_scale_sha256, "loss_scale_sha256"),
        (&context.evaluation_split_sha256, "evaluation_split_sha256"),
        (&context.evaluator_profile_sha256, "evaluator_profile_sha256"),
        (&context.model_configuration_sha256, "model_configuration_sha256"),
        (&context.latent_representation_profile_sha256, "latent_representation_profile_sha256"),
        (&context.latent_state_sha256, "latent_state_sha256"),
        (&context.latent_readout_profile_sha256, "latent_readout_profile_sha256"),
        (&context.rollout_profile_sha256, "rollout_profile_sha256"),
        (&context.pairing_verifier_profile_sha256, "pairing_verifier_profile_sha256"),
    ] {
        require_digest(digest, field)?;
    }
    Ok(())
}

fn validate_action_provenance(
    context: &DecisionTransferContext,
    action: &DecisionTransferAction,
) -> Result<(), DecisionTransferError> {
    let fields = [
        (
            &action.factual_history_sha256,
            &context.factual_history_sha256,
            "factual_history_sha256",
        ),
        (
            &action.branch_point_snapshot_sha256,
            &context.branch_point_snapshot_sha256,
            "branch_point_snapshot_sha256",
        ),
        (
            &action.exogenous_forcing_manifest_sha256,
            &context.exogenous_forcing_manifest_sha256,
            "exogenous_forcing_manifest_sha256",
        ),
        (
            &action.model_configuration_sha256,
            &context.model_configuration_sha256,
            "model_configuration_sha256",
        ),
        (
            &action.latent_representation_profile_sha256,
            &context.latent_representation_profile_sha256,
            "latent_representation_profile_sha256",
        ),
        (
            &action.latent_state_sha256,
            &context.latent_state_sha256,
            "latent_state_sha256",
        ),
    ];
    for (actual, expected, field) in fields {
        if actual != expected {
            return Err(DecisionTransferError::ProvenanceMismatch {
                action_id: action.action_id.clone(),
                field,
            });
        }
    }
    require_digest(&action.pairing_receipt_sha256, "pairing_receipt_sha256")?;
    require_digest(&action.trajectory_sha256, "trajectory_sha256")?;
    Ok(())
}

fn utility_of_disposition(
    action: &DecisionTransferAction,
) -> Result<(i64, u64), DecisionTransferError> {
    match action.disposition {
        BranchDisposition::Applied {
            applied_at_us,
            realized_utility_units,
        } => Ok((realized_utility_units, applied_at_us)),
        BranchDisposition::NotApplied {
            terminal_at_us,
            reason,
        } => Err(DecisionTransferError::ActionNotApplied {
            action_id: action.action_id.clone(),
            terminal_at_us,
            reason,
        }),
    }
}

fn select_action<F>(
    actions: &[DecisionTransferActionResult],
    predicted_utility: F,
    realized_best: i64,
    epsilon_units: i64,
) -> Result<DecisionSelection, DecisionTransferError>
where
    F: Fn(&DecisionTransferActionResult) -> i64,
{
    let mut selected: Option<&DecisionTransferActionResult> = None;
    for action in actions {
        let utility = predicted_utility(action);
        match selected {
            None => selected = Some(action),
            Some(current) if utility > predicted_utility(current) => selected = Some(action),
            // Actions are already sorted by ID. Keeping the first row gives a
            // stable lexicographically-smallest action on exact prediction ties.
            Some(_) => {}
        }
    }
    let selected = selected.ok_or(DecisionTransferError::EmptyExpectedActionSet)?;
    let regret = i128::from(realized_best)
        .checked_sub(i128::from(selected.realized_utility_units))
        .ok_or(DecisionTransferError::ArithmeticOverflow)?;
    let epsilon_good = i128::from(realized_best)
        .checked_sub(i128::from(selected.realized_utility_units))
        .ok_or(DecisionTransferError::ArithmeticOverflow)?
        <= i128::from(epsilon_units);
    Ok(DecisionSelection {
        action_id: selected.action_id.clone(),
        predicted_utility_units: predicted_utility(selected),
        realized_utility_units: selected.realized_utility_units,
        realized_regret_units: regret,
        is_realized_top1: selected.realized_utility_units == realized_best,
        is_realized_epsilon_good: epsilon_good,
    })
}

fn pairwise_ranking_counts<F>(
    actions: &[DecisionTransferActionResult],
    predicted_utility: F,
) -> Result<PairwiseRankingCounts, DecisionTransferError>
where
    F: Fn(&DecisionTransferActionResult) -> i64,
{
    let mut counts = PairwiseRankingCounts::default();
    for left_index in 0..actions.len() {
        for right_index in (left_index + 1)..actions.len() {
            let left = &actions[left_index];
            let right = &actions[right_index];
            let realized_order = left.realized_utility_units.cmp(&right.realized_utility_units);
            let predicted_order = predicted_utility(left).cmp(&predicted_utility(right));
            let target = match (realized_order, predicted_order) {
                (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => {
                    &mut counts.both_tied_pairs
                }
                (std::cmp::Ordering::Equal, _) => &mut counts.realized_only_ties,
                (_, std::cmp::Ordering::Equal) => &mut counts.predicted_only_ties,
                (left_order, right_order) if left_order == right_order => {
                    &mut counts.concordant_pairs
                }
                _ => &mut counts.discordant_pairs,
            };
            *target = (*target)
                .checked_add(1)
                .ok_or(DecisionTransferError::ArithmeticOverflow)?;
        }
    }
    Ok(counts)
}

pub fn evaluate_individual_decision_transfer(
    input: &IndividualDecisionTransferInput,
) -> Result<IndividualDecisionTransferCertificate, DecisionTransferError> {
    validate_context(&input.context)?;
    if input.epsilon_units < 0 {
        return Err(DecisionTransferError::NegativeEpsilon {
            epsilon_units: input.epsilon_units,
        });
    }
    if input.expected_action_ids.is_empty() {
        return Err(DecisionTransferError::EmptyExpectedActionSet);
    }
    if input.expected_action_ids.len() > MAX_DECISION_TRANSFER_ACTIONS
        || input.actions.len() > MAX_DECISION_TRANSFER_ACTIONS
    {
        return Err(DecisionTransferError::TooManyActions {
            requested: input.expected_action_ids.len().max(input.actions.len()),
            maximum: MAX_DECISION_TRANSFER_ACTIONS,
        });
    }

    let mut expected = BTreeSet::new();
    for action_id in &input.expected_action_ids {
        if action_id.trim().is_empty() || action_id.trim() != action_id.as_str() {
            return Err(DecisionTransferError::EmptyActionId);
        }
        if !expected.insert(action_id.clone()) {
            return Err(DecisionTransferError::DuplicateExpectedActionId {
                action_id: action_id.clone(),
            });
        }
    }

    if input.expected_action_ids.len() < 2 || input.actions.len() < 2 {
        return Err(DecisionTransferError::TooFewActions {
            requested: input.expected_action_ids.len().min(input.actions.len()),
        });
    }

    let mut actions = input.actions.clone();
    actions.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    let mut observed = BTreeSet::new();
    let mut run_ids = BTreeSet::new();
    let mut action_results = Vec::with_capacity(actions.len());

    for action in &actions {
        if action.action_id.trim().is_empty() || action.action_id.trim() != action.action_id.as_str() {
            return Err(DecisionTransferError::EmptyActionId);
        }
        if !observed.insert(action.action_id.clone()) {
            return Err(DecisionTransferError::DuplicateActionId {
                action_id: action.action_id.clone(),
            });
        }
        if action.run_id == 0 {
            return Err(DecisionTransferError::ZeroRunId {
                action_id: action.action_id.clone(),
            });
        }
        if !run_ids.insert(action.run_id) {
            return Err(DecisionTransferError::DuplicateRunId {
                run_id: action.run_id,
            });
        }
        validate_action_provenance(&input.context, action)?;
        let (realized_utility_units, applied_at_us) = utility_of_disposition(action)?;
        action_results.push(DecisionTransferActionResult {
            action_id: action.action_id.clone(),
            run_id: action.run_id,
            applied_at_us,
            pairing_receipt_sha256: action.pairing_receipt_sha256,
            trajectory_sha256: action.trajectory_sha256,
            realized_utility_units,
            latent_predicted_utility_units: action.latent_predicted_utility_units,
            rollout_predicted_utility_units: action.rollout_predicted_utility_units,
            realized_regret_units: 0,
            is_realized_top1: false,
            is_realized_epsilon_good: false,
        });
    }

    if expected != observed {
        let missing_action_ids = expected.difference(&observed).cloned().collect();
        let unexpected_action_ids = observed.difference(&expected).cloned().collect();
        return Err(DecisionTransferError::ActionSetMismatch {
            missing_action_ids,
            unexpected_action_ids,
        });
    }

    let realized_best = action_results
        .iter()
        .map(|action| action.realized_utility_units)
        .max()
        .ok_or(DecisionTransferError::EmptyExpectedActionSet)?;

    let mut realized_epsilon_good_action_ids = Vec::new();
    for action in &mut action_results {
        let regret = i128::from(realized_best)
            .checked_sub(i128::from(action.realized_utility_units))
            .ok_or(DecisionTransferError::ArithmeticOverflow)?;
        action.realized_regret_units = regret;
        action.is_realized_top1 = action.realized_utility_units == realized_best;
        action.is_realized_epsilon_good =
            regret <= i128::from(input.epsilon_units);
        if action.is_realized_epsilon_good {
            realized_epsilon_good_action_ids.push(action.action_id.clone());
        }
    }

    let latent_selection = select_action(
        &action_results,
        |action| action.latent_predicted_utility_units,
        realized_best,
        input.epsilon_units,
    )?;
    let rollout_selection = select_action(
        &action_results,
        |action| action.rollout_predicted_utility_units,
        realized_best,
        input.epsilon_units,
    )?;

    let rollout_minus_latent_regret_units = rollout_selection
        .realized_regret_units
        .checked_sub(latent_selection.realized_regret_units)
        .ok_or(DecisionTransferError::ArithmeticOverflow)?;

    let latent_ranking = pairwise_ranking_counts(
        &action_results,
        |action| action.latent_predicted_utility_units,
    )?;
    let rollout_ranking = pairwise_ranking_counts(
        &action_results,
        |action| action.rollout_predicted_utility_units,
    )?;

    let latent_selection = DecisionSelection {
        action_id: latent_selection.action_id,
        predicted_utility_units: latent_selection.predicted_utility_units,
        realized_utility_units: latent_selection.realized_utility_units,
        realized_regret_units: latent_selection.realized_regret_units,
        is_realized_top1: latent_selection.is_realized_top1,
        is_realized_epsilon_good: latent_selection.is_realized_epsilon_good,
    };
    let rollout_selection = DecisionSelection {
        action_id: rollout_selection.action_id,
        predicted_utility_units: rollout_selection.predicted_utility_units,
        realized_utility_units: rollout_selection.realized_utility_units,
        realized_regret_units: rollout_selection.realized_regret_units,
        is_realized_top1: rollout_selection.is_realized_top1,
        is_realized_epsilon_good: rollout_selection.is_realized_epsilon_good,
    };

    Ok(IndividualDecisionTransferCertificate {
        context: input.context.clone(),
        decision_point_id: input.context.decision_point_id,
        individual_id_sha256: input.context.individual_id_sha256,
        action_count: u64::try_from(action_results.len())
            .map_err(|_| DecisionTransferError::ArithmeticOverflow)?,
        epsilon_units: input.epsilon_units,
        realized_best_utility_units: realized_best,
        realized_epsilon_good_action_ids,
        latent_selection,
        rollout_selection,
        rollout_minus_latent_regret_units,
        latent_ranking,
        rollout_ranking,
        actions: action_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn context() -> DecisionTransferContext {
        DecisionTransferContext {
            decision_point_id: 7,
            individual_id_sha256: digest(1),
            factual_history_sha256: digest(2),
            branch_point_snapshot_sha256: digest(3),
            exogenous_forcing_manifest_sha256: digest(4),
            task_sha256: digest(5),
            loss_scale_sha256: digest(6),
            evaluation_split_sha256: digest(7),
            evaluator_profile_sha256: digest(8),
            model_configuration_sha256: digest(9),
            latent_representation_profile_sha256: digest(10),
            latent_state_sha256: digest(11),
            latent_readout_profile_sha256: digest(12),
            rollout_profile_sha256: digest(13),
            pairing_verifier_profile_sha256: digest(14),
        }
    }

    fn action(
        action_id: &str,
        run_id: u64,
        realized: i64,
        latent: i64,
        rollout: i64,
    ) -> DecisionTransferAction {
        DecisionTransferAction {
            action_id: action_id.to_owned(),
            run_id,
            factual_history_sha256: digest(2),
            branch_point_snapshot_sha256: digest(3),
            exogenous_forcing_manifest_sha256: digest(4),
            model_configuration_sha256: digest(9),
            latent_representation_profile_sha256: digest(10),
            latent_state_sha256: digest(11),
            pairing_receipt_sha256: digest((20 + run_id) as u8),
            trajectory_sha256: digest((40 + run_id) as u8),
            disposition: BranchDisposition::Applied {
                applied_at_us: 100 + run_id,
                realized_utility_units: realized,
            },
            latent_predicted_utility_units: latent,
            rollout_predicted_utility_units: rollout,
        }
    }

    fn input(actions: Vec<DecisionTransferAction>) -> IndividualDecisionTransferInput {
        IndividualDecisionTransferInput {
            context: context(),
            expected_action_ids: vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            epsilon_units: 1,
            actions,
        }
    }

    #[test]
    fn latent_can_preserve_decision_better_than_rollout() {
        let certificate = evaluate_individual_decision_transfer(&input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]))
        .unwrap();
        assert_eq!(certificate.latent_selection.action_id, "a");
        assert_eq!(certificate.rollout_selection.action_id, "b");
        assert_eq!(certificate.latent_selection.realized_regret_units, 0);
        assert_eq!(certificate.rollout_selection.realized_regret_units, 5);
        assert_eq!(certificate.rollout_minus_latent_regret_units, 5);
    }

    #[test]
    fn rollout_can_preserve_decision_better_than_latent() {
        let certificate = evaluate_individual_decision_transfer(&input(vec![
            action("a", 1, 10, 1, 9),
            action("b", 2, 5, 10, 1),
            action("c", 3, 0, 0, 0),
        ]))
        .unwrap();
        assert_eq!(certificate.latent_selection.action_id, "b");
        assert_eq!(certificate.rollout_selection.action_id, "a");
        assert_eq!(certificate.rollout_minus_latent_regret_units, -5);
    }

    #[test]
    fn shuffled_input_produces_identical_canonical_certificate() {
        let first = input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]);
        let mut shuffled = first.clone();
        shuffled.actions.reverse();
        shuffled.expected_action_ids.reverse();
        assert_eq!(
            evaluate_individual_decision_transfer(&first).unwrap(),
            evaluate_individual_decision_transfer(&shuffled).unwrap()
        );
    }

    #[test]
    fn epsilon_good_set_includes_all_actions_within_tolerance() {
        let certificate = evaluate_individual_decision_transfer(&input(vec![
            action("a", 1, 10, 10, 10),
            action("b", 2, 9, 9, 9),
            action("c", 3, 7, 7, 7),
        ]))
        .unwrap();
        assert_eq!(certificate.realized_epsilon_good_action_ids, vec!["a", "b"]);
        assert!(certificate.latent_selection.is_realized_epsilon_good);
    }

    #[test]
    fn ties_use_lexicographically_first_action_id_for_selection() {
        let certificate = evaluate_individual_decision_transfer(&input(vec![
            action("b", 2, 8, 4, 3),
            action("a", 1, 8, 4, 3),
            action("c", 3, 0, 0, 0),
        ]))
        .unwrap();
        assert_eq!(certificate.latent_selection.action_id, "a");
        assert_eq!(certificate.rollout_selection.action_id, "a");
        assert_eq!(certificate.realized_epsilon_good_action_ids, vec!["a", "b"]);
        assert_eq!(certificate.latent_ranking.both_tied_pairs, 1);
        assert_eq!(certificate.latent_ranking.realized_only_ties, 0);
    }

    #[test]
    fn predicted_ties_and_realized_ties_are_reported_separately() {
        let certificate = evaluate_individual_decision_transfer(&input(vec![
            action("a", 1, 10, 3, 3),
            action("b", 2, 5, 3, 1),
            action("c", 3, 5, 0, 0),
        ]))
        .unwrap();
        assert_eq!(certificate.latent_ranking.predicted_only_ties, 1);
        assert_eq!(certificate.latent_ranking.realized_only_ties, 1);
        assert_eq!(certificate.latent_ranking.both_tied_pairs, 0);
    }

    #[test]
    fn common_mode_utility_translation_preserves_choice_and_regret() {
        let original = evaluate_individual_decision_transfer(&input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]))
        .unwrap();
        let translated_input = input(vec![
            action("a", 1, 110, 109, 101),
            action("b", 2, 105, 101, 110),
            action("c", 3, 100, 100, 100),
        ]);
        let translated = evaluate_individual_decision_transfer(&translated_input).unwrap();
        assert_eq!(original.latent_selection.action_id, translated.latent_selection.action_id);
        assert_eq!(original.rollout_selection.action_id, translated.rollout_selection.action_id);
        assert_eq!(original.latent_selection.realized_regret_units, translated.latent_selection.realized_regret_units);
        assert_eq!(original.rollout_selection.realized_regret_units, translated.rollout_selection.realized_regret_units);
    }

    #[test]
    fn rejects_changed_exogenous_realization() {
        let mut candidate = input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]);
        candidate.actions[1].exogenous_forcing_manifest_sha256 = digest(99);
        assert!(matches!(
            evaluate_individual_decision_transfer(&candidate),
            Err(DecisionTransferError::ProvenanceMismatch {
                field: "exogenous_forcing_manifest_sha256",
                ..
            })
        ));
    }

    #[test]
    fn rejects_changed_full_branch_point_snapshot() {
        let mut candidate = input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]);
        candidate.actions[1].branch_point_snapshot_sha256 = digest(99);
        assert!(matches!(
            evaluate_individual_decision_transfer(&candidate),
            Err(DecisionTransferError::ProvenanceMismatch {
                field: "branch_point_snapshot_sha256",
                ..
            })
        ));
    }

    #[test]
    fn rejects_missing_or_unexpected_action_rows() {
        let missing = input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
        ]);
        assert!(matches!(
            evaluate_individual_decision_transfer(&missing),
            Err(DecisionTransferError::ActionSetMismatch { .. })
        ));
        let mut extra = input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]);
        extra.actions.push(action("d", 4, 0, 0, 0));
        assert!(matches!(
            evaluate_individual_decision_transfer(&extra),
            Err(DecisionTransferError::ActionSetMismatch { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_action_and_run_ids() {
        let duplicate_action = input(vec![
            action("a", 1, 10, 9, 1),
            action("a", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]);
        assert!(matches!(
            evaluate_individual_decision_transfer(&duplicate_action),
            Err(DecisionTransferError::DuplicateActionId { .. })
        ));
        let duplicate_run = input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 1, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]);
        assert!(matches!(
            evaluate_individual_decision_transfer(&duplicate_run),
            Err(DecisionTransferError::DuplicateRunId { .. })
        ));
    }

    #[test]
    fn rejects_non_applied_branches_without_zero_imputation() {
        let mut candidate = input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]);
        candidate.actions[2].disposition = BranchDisposition::NotApplied {
            terminal_at_us: 300,
            reason: NonApplicationReason::ExecutionFailure,
        };
        assert!(matches!(
            evaluate_individual_decision_transfer(&candidate),
            Err(DecisionTransferError::ActionNotApplied {
                action_id,
                reason: NonApplicationReason::ExecutionFailure,
                ..
            }) if action_id == "c"
        ));
    }

    #[test]
    fn rejects_missing_receipt_and_trajectory_digests() {
        let mut candidate = input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]);
        candidate.actions[0].pairing_receipt_sha256 = [0; 32];
        assert!(matches!(
            evaluate_individual_decision_transfer(&candidate),
            Err(DecisionTransferError::EmptyDigest {
                field: "pairing_receipt_sha256"
            })
        ));
    }

    #[test]
    fn rejects_negative_epsilon_and_zero_decision_id() {
        let mut candidate = input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]);
        candidate.epsilon_units = -1;
        assert!(matches!(
            evaluate_individual_decision_transfer(&candidate),
            Err(DecisionTransferError::NegativeEpsilon { .. })
        ));
        candidate.epsilon_units = 0;
        candidate.context.decision_point_id = 0;
        assert!(matches!(
            evaluate_individual_decision_transfer(&candidate),
            Err(DecisionTransferError::ZeroDecisionPointId)
        ));
    }

    #[test]
    fn extreme_signed_utilities_do_not_overflow() {
        let certificate = evaluate_individual_decision_transfer(&input(vec![
            action("a", 1, i64::MAX, i64::MAX, i64::MIN),
            action("b", 2, i64::MIN, i64::MIN, i64::MAX),
            action("c", 3, 0, 0, 0),
        ]))
        .unwrap();
        assert_eq!(
            certificate.latent_selection.realized_regret_units,
            0
        );
        assert_eq!(
            certificate.rollout_selection.realized_regret_units,
            i128::from(i64::MAX) - i128::from(i64::MIN)
        );
    }

    #[test]
    fn certificate_echoes_frozen_context_and_action_receipts() {
        let original = input(vec![
            action("a", 1, 10, 9, 1),
            action("b", 2, 5, 1, 10),
            action("c", 3, 0, 0, 0),
        ]);
        let certificate = evaluate_individual_decision_transfer(&original).unwrap();
        assert_eq!(certificate.context, original.context);
        assert_eq!(certificate.actions[0].action_id, "a");
        assert_eq!(certificate.actions[0].pairing_receipt_sha256, digest(21));
        assert_eq!(certificate.actions[0].trajectory_sha256, digest(41));
    }

    #[test]
    fn ranking_counts_sum_to_all_unordered_pairs() {
        let certificate = evaluate_individual_decision_transfer(&input(vec![
            action("a", 1, 10, 10, 10),
            action("b", 2, 5, 10, 5),
            action("c", 3, 5, 1, 1),
        ]))
        .unwrap();
        let total = certificate.latent_ranking.concordant_pairs
            + certificate.latent_ranking.discordant_pairs
            + certificate.latent_ranking.realized_only_ties
            + certificate.latent_ranking.predicted_only_ties
            + certificate.latent_ranking.both_tied_pairs;
        assert_eq!(total, 3);
    }
}
