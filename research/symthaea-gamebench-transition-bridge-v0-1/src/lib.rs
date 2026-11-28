//! Transition-aware GameBench causal-twin scoring.
//!
//! This crate owns only GameBench-specific matched-world scoring and the
//! structural validation of environment-produced world-mutation events.
//! It intentionally does not implement Bayesian updating, EVSI, a minimax
//! planner, a sequential policy oracle, a cryptographic receipt ledger, or a
//! process sandbox.
//!
//! Evaluator truth and candidate-visible views are separate data structures,
//! but Rust visibility is not a security boundary. Confirmatory evaluation
//! must isolate the environment executor, candidate, and evaluator process.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

const LOSS_TOLERANCE: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct TwinWorld {
    /// Evaluator-only stable world identifier.
    pub world_id: String,
    /// None marks an out-of-family challenge member.
    pub mechanism_variant_id: Option<String>,
    /// Digest of all declared nuisance factors that must remain matched.
    pub nuisance_digest: String,
    /// Digest of the initial candidate-visible observation.
    pub initial_observation_digest: String,
    /// Candidate-visible action vocabulary at the matched initial state.
    pub legal_action_ids: Vec<String>,
    /// Evaluator-only terminal loss for each legal action.
    pub loss_by_action: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CausalTwinManifest {
    pub challenge_id: String,
    pub mechanism_family_id: String,
    /// Public ontology of candidate mechanism hypotheses.
    pub mechanism_variant_ids: Vec<String>,
    /// Tolerance used only to identify tied optimal actions.
    pub tie_tolerance: f64,
    /// Includes evaluator-only truth. Never pass this object to a candidate.
    pub worlds: Vec<TwinWorld>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateView {
    pub challenge_id: String,
    pub initial_observation_digest: String,
    pub mechanism_hypothesis_ids: Vec<String>,
    pub legal_action_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateClaim {
    /// Mechanisms the candidate still considers compatible with its evidence.
    /// Empty means the candidate makes no mechanism-set claim.
    pub retained_mechanism_variant_ids: Vec<String>,
    /// None means the candidate abstains from a terminal action.
    pub selected_action_id: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionEquivalence {
    /// At least one action is optimal in every compatible in-family world.
    ActionEquivalent,
    /// No single action is optimal in every compatible in-family world.
    DecisionSensitive,
    /// An out-of-family world remains compatible or the support is unusable.
    Unresolved,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorldRegret {
    pub world_id: String,
    pub mechanism_variant_id: Option<String>,
    pub regret: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CausalTwinScore {
    pub action_equivalence: ActionEquivalence,
    pub compatible_world_count: usize,
    pub compatible_mechanism_variant_ids: Vec<String>,
    pub retained_true_mechanism_count: usize,
    pub mechanism_set_coverage: Option<f64>,
    /// A singleton claim is false-unique when evidence still supports another
    /// mechanism variant or an out-of-family world.
    pub false_unique_claim: bool,
    pub common_optimal_action_ids: Vec<String>,
    pub selected_action_is_common_optimal: Option<bool>,
    /// Unavailable when an out-of-family world remains compatible.
    pub worst_case_action_regret: Option<f64>,
    pub mean_action_regret: Option<f64>,
    pub per_world_regret: Vec<WorldRegret>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProtocolError {
    EmptyField(&'static str),
    InvalidManifest(&'static str),
    DuplicateIdentifier { field: &'static str, id: String },
    UnknownIdentifier { field: &'static str, id: String },
    InvalidLoss { action_id: String, value_bits: u64 },
    InvalidTieTolerance(u64),
    EmptyCompatibilitySet,
    DuplicateCompatibleWorld(String),
    InvalidCandidateHypothesis(String),
    InvalidCandidateAction(String),
    UnsupportedSideEffectClass,
    EpisodeMismatch { expected: String, actual: String },
    InterventionMismatch { expected: String, actual: String },
    ActionMismatch { expected: String, actual: String },
    InvocationAlreadyConsumed(String),
    StaleOrOutOfOrderTransition { expected_step: u64, actual_step: u64 },
    AppliedBeforeIssued { issued_at: u64, applied_at: u64 },
    ObservationBeforeApplication { applied_at: u64, observed_at: u64 },
    ObservationTooEarly { earliest: u64, observed_at: u64 },
    ObservationAfterDeadline { deadline: u64, observed_at: u64 },
    PreStateMismatch { expected: String, actual: String },
    CostAboveContract { ceiling: u64, actual: u64 },
    CostOverBudget { remaining: u64, actual: u64 },
    StepOverflow,
}

impl Display for ProtocolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::InvalidManifest(reason) => write!(f, "invalid causal-twin manifest: {reason}"),
            Self::DuplicateIdentifier { field, id } => {
                write!(f, "duplicate {field} identifier: {id}")
            }
            Self::UnknownIdentifier { field, id } => {
                write!(f, "unknown {field} identifier: {id}")
            }
            Self::InvalidLoss { action_id, value_bits } => {
                write!(f, "invalid loss for action {action_id}: {}", f64::from_bits(*value_bits))
            }
            Self::InvalidTieTolerance(value_bits) => {
                write!(f, "invalid tie tolerance: {}", f64::from_bits(*value_bits))
            }
            Self::EmptyCompatibilitySet => write!(f, "compatible-world set must not be empty"),
            Self::DuplicateCompatibleWorld(id) => {
                write!(f, "compatible-world set repeats world {id}")
            }
            Self::InvalidCandidateHypothesis(id) => {
                write!(f, "candidate claimed an undeclared mechanism hypothesis: {id}")
            }
            Self::InvalidCandidateAction(id) => {
                write!(f, "candidate selected an undeclared action: {id}")
            }
            Self::UnsupportedSideEffectClass => write!(
                f,
                "transition bridge requires an explicitly world-mutating intervention"
            ),
            Self::EpisodeMismatch { expected, actual } => {
                write!(f, "episode mismatch: expected {expected}, got {actual}")
            }
            Self::InterventionMismatch { expected, actual } => {
                write!(f, "intervention mismatch: expected {expected}, got {actual}")
            }
            Self::ActionMismatch { expected, actual } => {
                write!(f, "action mismatch: expected {expected}, got {actual}")
            }
            Self::InvocationAlreadyConsumed(id) => {
                write!(f, "invocation has already been consumed: {id}")
            }
            Self::StaleOrOutOfOrderTransition { expected_step, actual_step } => write!(
                f,
                "transition issued at step {actual_step}; current episode step is {expected_step}"
            ),
            Self::AppliedBeforeIssued { issued_at, applied_at } => {
                write!(f, "transition applied at {applied_at} before issue step {issued_at}")
            }
            Self::ObservationBeforeApplication { applied_at, observed_at } => write!(
                f,
                "observation at step {observed_at} precedes application at step {applied_at}"
            ),
            Self::ObservationTooEarly { earliest, observed_at } => write!(
                f,
                "observation at step {observed_at} arrived before earliest step {earliest}"
            ),
            Self::ObservationAfterDeadline { deadline, observed_at } => write!(
                f,
                "observation at step {observed_at} missed decision deadline {deadline}"
            ),
            Self::PreStateMismatch { expected, actual } => {
                write!(f, "transition pre-state mismatch: expected {expected}, got {actual}")
            }
            Self::CostAboveContract { ceiling, actual } => {
                write!(f, "charged cost {actual} exceeds contract ceiling {ceiling}")
            }
            Self::CostOverBudget { remaining, actual } => {
                write!(f, "charged cost {actual} exceeds remaining budget {remaining}")
            }
            Self::StepOverflow => write!(f, "step arithmetic overflow"),
        }
    }
}

impl Error for ProtocolError {}

impl CausalTwinManifest {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        require_nonempty("challenge_id", &self.challenge_id)?;
        require_nonempty("mechanism_family_id", &self.mechanism_family_id)?;

        if !self.tie_tolerance.is_finite()
            || !(0.0..=1.0).contains(&self.tie_tolerance)
        {
            return Err(ProtocolError::InvalidTieTolerance(
                self.tie_tolerance.to_bits(),
            ));
        }

        if self.worlds.len() < 2 {
            return Err(ProtocolError::InvalidManifest(
                "at least two matched worlds are required",
            ));
        }

        let mut declared_variants = BTreeSet::new();
        for variant_id in &self.mechanism_variant_ids {
            require_nonempty("mechanism_variant_id", variant_id)?;
            if !declared_variants.insert(variant_id.as_str()) {
                return Err(ProtocolError::DuplicateIdentifier {
                    field: "mechanism_variant",
                    id: variant_id.clone(),
                });
            }
        }
        if declared_variants.len() < 2 {
            return Err(ProtocolError::InvalidManifest(
                "at least two mechanism variants are required",
            ));
        }

        let first = &self.worlds[0];
        require_nonempty("nuisance_digest", &first.nuisance_digest)?;
        require_nonempty(
            "initial_observation_digest",
            &first.initial_observation_digest,
        )?;
        let expected_actions = checked_action_set(&first.legal_action_ids)?;

        let mut world_ids = BTreeSet::new();
        let mut seen_variants = BTreeSet::new();
        for world in &self.worlds {
            require_nonempty("world_id", &world.world_id)?;
            require_nonempty("nuisance_digest", &world.nuisance_digest)?;
            require_nonempty(
                "initial_observation_digest",
                &world.initial_observation_digest,
            )?;
            if !world_ids.insert(world.world_id.as_str()) {
                return Err(ProtocolError::DuplicateIdentifier {
                    field: "world",
                    id: world.world_id.clone(),
                });
            }
            if world.nuisance_digest != first.nuisance_digest {
                return Err(ProtocolError::InvalidManifest(
                    "matched twins differ in their nuisance digest",
                ));
            }
            if world.initial_observation_digest != first.initial_observation_digest {
                return Err(ProtocolError::InvalidManifest(
                    "matched twins do not share the same initial observation digest",
                ));
            }
            let action_set = checked_action_set(&world.legal_action_ids)?;
            if action_set != expected_actions {
                return Err(ProtocolError::InvalidManifest(
                    "matched twins do not share the same legal action set",
                ));
            }

            if let Some(variant_id) = &world.mechanism_variant_id {
                require_nonempty("world.mechanism_variant_id", variant_id)?;
                if !declared_variants.contains(variant_id.as_str()) {
                    return Err(ProtocolError::UnknownIdentifier {
                        field: "mechanism_variant",
                        id: variant_id.clone(),
                    });
                }
                seen_variants.insert(variant_id.as_str());
            }

            if world.loss_by_action.len() != expected_actions.len() {
                return Err(ProtocolError::InvalidManifest(
                    "each world must define exactly one loss for every legal action",
                ));
            }
            for action_id in &expected_actions {
                let loss = world.loss_by_action.get(action_id).ok_or_else(|| {
                    ProtocolError::UnknownIdentifier {
                        field: "world_loss_action",
                        id: action_id.clone(),
                    }
                })?;
                if !loss.is_finite() || !(0.0..=1.0).contains(loss) {
                    return Err(ProtocolError::InvalidLoss {
                        action_id: action_id.clone(),
                        value_bits: loss.to_bits(),
                    });
                }
            }
        }
        if seen_variants.len() < 2 {
            return Err(ProtocolError::InvalidManifest(
                "the manifest must instantiate at least two declared mechanism variants",
            ));
        }
        Ok(())
    }

    /// Returns only the candidate-visible initial view. The manifest itself
    /// contains evaluator truth and must not cross the process boundary.
    pub fn candidate_view(&self) -> Result<CandidateView, ProtocolError> {
        self.validate()?;
        let first = &self.worlds[0];
        let mut actions = first.legal_action_ids.clone();
        actions.sort();
        let mut hypotheses = self.mechanism_variant_ids.clone();
        hypotheses.sort();
        Ok(CandidateView {
            challenge_id: self.challenge_id.clone(),
            initial_observation_digest: first.initial_observation_digest.clone(),
            mechanism_hypothesis_ids: hypotheses,
            legal_action_ids: actions,
        })
    }
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), ProtocolError> {
    if value.trim().is_empty() {
        return Err(ProtocolError::EmptyField(field));
    }
    Ok(())
}

fn checked_action_set(actions: &[String]) -> Result<BTreeSet<String>, ProtocolError> {
    if actions.is_empty() {
        return Err(ProtocolError::InvalidManifest(
            "at least one legal action is required",
        ));
    }
    let mut set = BTreeSet::new();
    for action in actions {
        require_nonempty("legal_action_id", action)?;
        if !set.insert(action.clone()) {
            return Err(ProtocolError::DuplicateIdentifier {
                field: "legal_action",
                id: action.clone(),
            });
        }
    }
    Ok(set)
}

pub fn score_candidate_claim(
    manifest: &CausalTwinManifest,
    compatible_world_ids: &[String],
    claim: &CandidateClaim,
) -> Result<CausalTwinScore, ProtocolError> {
    manifest.validate()?;
    if compatible_world_ids.is_empty() {
        return Err(ProtocolError::EmptyCompatibilitySet);
    }

    let declared_variants: BTreeSet<&str> = manifest
        .mechanism_variant_ids
        .iter()
        .map(String::as_str)
        .collect();
    let mut claimed_variants = BTreeSet::new();
    for variant_id in &claim.retained_mechanism_variant_ids {
        if !declared_variants.contains(variant_id.as_str()) {
            return Err(ProtocolError::InvalidCandidateHypothesis(
                variant_id.clone(),
            ));
        }
        if !claimed_variants.insert(variant_id.as_str()) {
            return Err(ProtocolError::DuplicateIdentifier {
                field: "candidate_mechanism_hypothesis",
                id: variant_id.clone(),
            });
        }
    }
    if let Some(action_id) = &claim.selected_action_id {
        if !manifest.worlds[0]
            .legal_action_ids
            .iter()
            .any(|legal| legal == action_id)
        {
            return Err(ProtocolError::InvalidCandidateAction(
                action_id.clone(),
            ));
        }
    }

    let mut compatible_ids = BTreeSet::new();
    let mut compatible_worlds = Vec::with_capacity(compatible_world_ids.len());
    for world_id in compatible_world_ids {
        if !compatible_ids.insert(world_id.as_str()) {
            return Err(ProtocolError::DuplicateCompatibleWorld(
                world_id.clone(),
            ));
        }
        let world = manifest
            .worlds
            .iter()
            .find(|world| world.world_id == *world_id)
            .ok_or_else(|| ProtocolError::UnknownIdentifier {
                field: "compatible_world",
                id: world_id.clone(),
            })?;
        compatible_worlds.push(world);
    }

    let mut compatible_variants = BTreeSet::new();
    let mut out_of_family_possible = false;
    let mut common_optimal: Option<BTreeSet<String>> = None;
    let mut per_world_regret = Vec::new();
    let mut known_regrets = Vec::new();

    for world in &compatible_worlds {
        match &world.mechanism_variant_id {
            Some(variant_id) => {
                compatible_variants.insert(variant_id.clone());
            }
            None => out_of_family_possible = true,
        }

        let best_loss = world
            .loss_by_action
            .values()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let world_optimal: BTreeSet<String> = world
            .loss_by_action
            .iter()
            .filter(|(_, loss)| **loss <= best_loss + manifest.tie_tolerance)
            .map(|(action_id, _)| action_id.clone())
            .collect();

        if let Some(existing) = &mut common_optimal {
            existing.retain(|action_id| world_optimal.contains(action_id));
        } else {
            common_optimal = Some(world_optimal);
        }

        if let Some(action_id) = &claim.selected_action_id {
            let selected_loss = world.loss_by_action[action_id];
            let regret = (selected_loss - best_loss).max(0.0);
            per_world_regret.push(WorldRegret {
                world_id: world.world_id.clone(),
                mechanism_variant_id: world.mechanism_variant_id.clone(),
                regret,
            });
            if world.mechanism_variant_id.is_some() {
                known_regrets.push(regret);
            }
        }
    }

    let compatible_mechanism_variant_ids: Vec<String> =
        compatible_variants.iter().cloned().collect();
    let retained_true_mechanism_count = compatible_variants
        .iter()
        .filter(|variant| claimed_variants.contains(variant.as_str()))
        .count();
    let mechanism_set_coverage = if compatible_variants.is_empty() {
        None
    } else {
        Some(
            retained_true_mechanism_count as f64 / compatible_variants.len() as f64,
        )
    };

    let false_unique_claim = if claim.retained_mechanism_variant_ids.len() == 1 {
        let claimed = &claim.retained_mechanism_variant_ids[0];
        out_of_family_possible
            || compatible_variants.len() != 1
            || !compatible_variants.contains(claimed)
    } else {
        false
    };

    let common_optimal_action_ids: Vec<String> = common_optimal
        .unwrap_or_default()
        .into_iter()
        .collect();
    let action_equivalence = if out_of_family_possible {
        ActionEquivalence::Unresolved
    } else if common_optimal_action_ids.is_empty() {
        ActionEquivalence::DecisionSensitive
    } else {
        ActionEquivalence::ActionEquivalent
    };

    let selected_action_is_common_optimal = claim
        .selected_action_id
        .as_ref()
        .map(|action_id| common_optimal_action_ids.contains(action_id));

    let (worst_case_action_regret, mean_action_regret) =
        if out_of_family_possible || per_world_regret.is_empty() {
            (None, None)
        } else {
            let worst = known_regrets.iter().copied().fold(0.0, f64::max);
            let mean = known_regrets.iter().sum::<f64>() / known_regrets.len() as f64;
            (Some(worst), Some(mean))
        };

    Ok(CausalTwinScore {
        action_equivalence,
        compatible_world_count: compatible_worlds.len(),
        compatible_mechanism_variant_ids,
        retained_true_mechanism_count,
        mechanism_set_coverage,
        false_unique_claim,
        common_optimal_action_ids,
        selected_action_is_common_optimal,
        worst_case_action_regret,
        mean_action_regret,
        per_world_regret,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideEffectClass {
    ReadOnly,
    WorldMutating,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterventionContract {
    pub episode_id: String,
    pub intervention_id: String,
    pub action_id: String,
    pub side_effect_class: SideEffectClass,
    pub cost_ceiling_units: u64,
    pub minimum_result_delay_steps: u64,
    pub decision_deadline_step: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpisodeProgress {
    pub episode_id: String,
    pub current_step: u64,
    pub current_observation_digest: String,
    pub remaining_cost_units: u64,
    pub consumed_invocation_ids: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvironmentTransitionEvent {
    /// Produced by the environment executor, not by the candidate.
    pub episode_id: String,
    pub invocation_id: String,
    pub intervention_id: String,
    pub action_id: String,
    pub side_effect_class: SideEffectClass,
    pub issued_at_step: u64,
    pub applied_at_step: u64,
    pub observed_at_step: u64,
    pub before_observation_digest: String,
    pub after_observation_digest: String,
    pub outcome_id: String,
    pub actual_cost_units: u64,
    /// Source commitments are carried through; authenticity is checked by
    /// the independently permissioned executor/evaluator, not by this crate.
    pub environment_event_digest: String,
    pub transition_trace_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionReceipt {
    pub episode_id: String,
    pub invocation_id: String,
    pub intervention_id: String,
    pub action_id: String,
    pub issued_at_step: u64,
    pub applied_at_step: u64,
    pub observed_at_step: u64,
    pub before_observation_digest: String,
    pub after_observation_digest: String,
    pub outcome_id: String,
    pub charged_cost_units: u64,
    pub environment_event_digest: String,
    pub transition_trace_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionApplication {
    pub next_progress: EpisodeProgress,
    pub receipt: TransitionReceipt,
}

/// Validates and applies one environment-produced world mutation.
///
/// This checks protocol consistency, resource accounting and monotonic state
/// progression. It does not prove that the event digest is authentic or that
/// the simulator executed the claimed physics. The trusted environment runner
/// must independently verify its event stream before calling this function.
pub fn validate_and_apply_transition(
    contract: &InterventionContract,
    progress: &EpisodeProgress,
    event: &EnvironmentTransitionEvent,
) -> Result<TransitionApplication, ProtocolError> {
    require_nonempty("contract.episode_id", &contract.episode_id)?;
    require_nonempty("contract.intervention_id", &contract.intervention_id)?;
    require_nonempty("contract.action_id", &contract.action_id)?;
    require_nonempty("progress.episode_id", &progress.episode_id)?;
    require_nonempty(
        "progress.current_observation_digest",
        &progress.current_observation_digest,
    )?;
    require_nonempty("event.invocation_id", &event.invocation_id)?;
    require_nonempty("event.outcome_id", &event.outcome_id)?;
    require_nonempty(
        "event.before_observation_digest",
        &event.before_observation_digest,
    )?;
    require_nonempty(
        "event.after_observation_digest",
        &event.after_observation_digest,
    )?;
    require_nonempty("event.environment_event_digest", &event.environment_event_digest)?;
    require_nonempty("event.transition_trace_digest", &event.transition_trace_digest)?;

    if contract.side_effect_class != SideEffectClass::WorldMutating
        || event.side_effect_class != SideEffectClass::WorldMutating
    {
        return Err(ProtocolError::UnsupportedSideEffectClass);
    }
    if contract.episode_id != progress.episode_id {
        return Err(ProtocolError::EpisodeMismatch {
            expected: contract.episode_id.clone(),
            actual: progress.episode_id.clone(),
        });
    }
    if event.episode_id != progress.episode_id {
        return Err(ProtocolError::EpisodeMismatch {
            expected: progress.episode_id.clone(),
            actual: event.episode_id.clone(),
        });
    }
    if event.intervention_id != contract.intervention_id {
        return Err(ProtocolError::InterventionMismatch {
            expected: contract.intervention_id.clone(),
            actual: event.intervention_id.clone(),
        });
    }
    if event.action_id != contract.action_id {
        return Err(ProtocolError::ActionMismatch {
            expected: contract.action_id.clone(),
            actual: event.action_id.clone(),
        });
    }
    if progress
        .consumed_invocation_ids
        .contains(&event.invocation_id)
    {
        return Err(ProtocolError::InvocationAlreadyConsumed(
            event.invocation_id.clone(),
        ));
    }
    if event.issued_at_step != progress.current_step {
        return Err(ProtocolError::StaleOrOutOfOrderTransition {
            expected_step: progress.current_step,
            actual_step: event.issued_at_step,
        });
    }
    if event.applied_at_step < event.issued_at_step {
        return Err(ProtocolError::AppliedBeforeIssued {
            issued_at: event.issued_at_step,
            applied_at: event.applied_at_step,
        });
    }
    if event.observed_at_step < event.applied_at_step {
        return Err(ProtocolError::ObservationBeforeApplication {
            applied_at: event.applied_at_step,
            observed_at: event.observed_at_step,
        });
    }
    let earliest_observation = event
        .applied_at_step
        .checked_add(contract.minimum_result_delay_steps)
        .ok_or(ProtocolError::StepOverflow)?;
    if event.observed_at_step < earliest_observation {
        return Err(ProtocolError::ObservationTooEarly {
            earliest: earliest_observation,
            observed_at: event.observed_at_step,
        });
    }
    if event.observed_at_step > contract.decision_deadline_step {
        return Err(ProtocolError::ObservationAfterDeadline {
            deadline: contract.decision_deadline_step,
            observed_at: event.observed_at_step,
        });
    }
    if event.before_observation_digest != progress.current_observation_digest {
        return Err(ProtocolError::PreStateMismatch {
            expected: progress.current_observation_digest.clone(),
            actual: event.before_observation_digest.clone(),
        });
    }
    if event.actual_cost_units > contract.cost_ceiling_units {
        return Err(ProtocolError::CostAboveContract {
            ceiling: contract.cost_ceiling_units,
            actual: event.actual_cost_units,
        });
    }
    if event.actual_cost_units > progress.remaining_cost_units {
        return Err(ProtocolError::CostOverBudget {
            remaining: progress.remaining_cost_units,
            actual: event.actual_cost_units,
        });
    }

    let mut next_progress = progress.clone();
    next_progress.current_step = event.observed_at_step;
    next_progress.current_observation_digest = event.after_observation_digest.clone();
    next_progress.remaining_cost_units -= event.actual_cost_units;
    next_progress
        .consumed_invocation_ids
        .insert(event.invocation_id.clone());

    let receipt = TransitionReceipt {
        episode_id: event.episode_id.clone(),
        invocation_id: event.invocation_id.clone(),
        intervention_id: event.intervention_id.clone(),
        action_id: event.action_id.clone(),
        issued_at_step: event.issued_at_step,
        applied_at_step: event.applied_at_step,
        observed_at_step: event.observed_at_step,
        before_observation_digest: event.before_observation_digest.clone(),
        after_observation_digest: event.after_observation_digest.clone(),
        outcome_id: event.outcome_id.clone(),
        charged_cost_units: event.actual_cost_units,
        environment_event_digest: event.environment_event_digest.clone(),
        transition_trace_digest: event.transition_trace_digest.clone(),
    };

    Ok(TransitionApplication {
        next_progress,
        receipt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(
        world_id: &str,
        variant: Option<&str>,
        north_loss: f64,
        east_loss: f64,
    ) -> TwinWorld {
        TwinWorld {
            world_id: world_id.into(),
            mechanism_variant_id: variant.map(str::to_owned),
            nuisance_digest: "nuisance-sha256".into(),
            initial_observation_digest: "initial-observation-sha256".into(),
            legal_action_ids: vec!["north".into(), "east".into()],
            loss_by_action: BTreeMap::from([
                ("north".into(), north_loss),
                ("east".into(), east_loss),
            ]),
        }
    }

    fn manifest(decision_sensitive: bool) -> CausalTwinManifest {
        CausalTwinManifest {
            challenge_id: "switch-gate-001".into(),
            mechanism_family_id: "switch-topology".into(),
            mechanism_variant_ids: vec!["opens_north".into(), "opens_east".into()],
            tie_tolerance: LOSS_TOLERANCE,
            worlds: vec![
                world("world-north", Some("opens_north"), 0.0, 1.0),
                if decision_sensitive {
                    world("world-east", Some("opens_east"), 1.0, 0.0)
                } else {
                    world("world-east", Some("opens_east"), 0.0, 1.0)
                },
            ],
        }
    }

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn contract() -> InterventionContract {
        InterventionContract {
            episode_id: "episode-1".into(),
            intervention_id: "pull-switch".into(),
            action_id: "use-switch".into(),
            side_effect_class: SideEffectClass::WorldMutating,
            cost_ceiling_units: 2,
            minimum_result_delay_steps: 1,
            decision_deadline_step: 8,
        }
    }

    fn progress() -> EpisodeProgress {
        EpisodeProgress {
            episode_id: "episode-1".into(),
            current_step: 0,
            current_observation_digest: "obs-before".into(),
            remaining_cost_units: 3,
            consumed_invocation_ids: BTreeSet::new(),
        }
    }

    fn event() -> EnvironmentTransitionEvent {
        EnvironmentTransitionEvent {
            episode_id: "episode-1".into(),
            invocation_id: "invocation-1".into(),
            intervention_id: "pull-switch".into(),
            action_id: "use-switch".into(),
            side_effect_class: SideEffectClass::WorldMutating,
            issued_at_step: 0,
            applied_at_step: 0,
            observed_at_step: 1,
            before_observation_digest: "obs-before".into(),
            after_observation_digest: "obs-after".into(),
            outcome_id: "north-gate-open".into(),
            actual_cost_units: 2,
            environment_event_digest: "event-digest".into(),
            transition_trace_digest: "trace-digest".into(),
        }
    }

    #[test]
    fn matched_twins_share_initial_observation_and_candidate_view_hides_world_truth() {
        let manifest = manifest(true);
        manifest.validate().expect("valid matched twins");
        let view = manifest.candidate_view().expect("candidate view");
        assert_eq!(
            view.initial_observation_digest,
            "initial-observation-sha256"
        );
        assert_eq!(
            view.mechanism_hypothesis_ids,
            ids(&["opens_east", "opens_north"])
        );
        assert_eq!(view.legal_action_ids, ids(&["east", "north"]));
        assert!(!format!("{view:?}").contains("world-north"));
    }

    #[test]
    fn action_equivalence_does_not_require_mechanism_identification() {
        let manifest = manifest(false);
        let score = score_candidate_claim(
            &manifest,
            &ids(&["world-north", "world-east"]),
            &CandidateClaim {
                retained_mechanism_variant_ids: ids(&["opens_north", "opens_east"]),
                selected_action_id: Some("north".into()),
            },
        )
        .expect("score compatible worlds");

        assert_eq!(score.action_equivalence, ActionEquivalence::ActionEquivalent);
        assert_eq!(score.common_optimal_action_ids, ids(&["north"]));
        assert_eq!(score.selected_action_is_common_optimal, Some(true));
        assert_eq!(score.false_unique_claim, false);
        assert_eq!(score.worst_case_action_regret, Some(0.0));
    }

    #[test]
    fn decision_sensitive_twins_have_no_common_optimal_action() {
        let manifest = manifest(true);
        let score = score_candidate_claim(
            &manifest,
            &ids(&["world-north", "world-east"]),
            &CandidateClaim {
                retained_mechanism_variant_ids: ids(&["opens_north", "opens_east"]),
                selected_action_id: Some("north".into()),
            },
        )
        .expect("score decision-sensitive worlds");

        assert_eq!(score.action_equivalence, ActionEquivalence::DecisionSensitive);
        assert!(score.common_optimal_action_ids.is_empty());
        assert_eq!(score.selected_action_is_common_optimal, Some(false));
        assert_eq!(score.worst_case_action_regret, Some(1.0));
        assert_eq!(score.mean_action_regret, Some(0.5));
    }

    #[test]
    fn singleton_claim_on_ambiguous_support_is_false_unique() {
        let score = score_candidate_claim(
            &manifest(true),
            &ids(&["world-north", "world-east"]),
            &CandidateClaim {
                retained_mechanism_variant_ids: ids(&["opens_north"]),
                selected_action_id: None,
            },
        )
        .expect("score singleton claim");
        assert!(score.false_unique_claim);
        assert_eq!(score.mechanism_set_coverage, Some(0.5));
    }

    #[test]
    fn out_of_family_support_forces_unresolved() {
        let mut manifest = manifest(true);
        manifest.worlds.push(world("world-other", None, 0.0, 0.0));
        let score = score_candidate_claim(
            &manifest,
            &ids(&["world-north", "world-other"]),
            &CandidateClaim {
                retained_mechanism_variant_ids: ids(&["opens_north"]),
                selected_action_id: Some("north".into()),
            },
        )
        .expect("score open-world support");
        assert_eq!(score.action_equivalence, ActionEquivalence::Unresolved);
        assert!(score.false_unique_claim);
        assert_eq!(score.worst_case_action_regret, None);
    }

    #[test]
    fn manifest_rejects_mismatched_initial_observation() {
        let mut manifest = manifest(true);
        manifest.worlds[1].initial_observation_digest = "different-observation".into();
        assert!(matches!(
            manifest.validate(),
            Err(ProtocolError::InvalidManifest(_))
        ));
    }

    #[test]
    fn manifest_rejects_mismatched_nuisance_digest() {
        let mut manifest = manifest(true);
        manifest.worlds[1].nuisance_digest = "different-nuisance".into();
        assert!(matches!(
            manifest.validate(),
            Err(ProtocolError::InvalidManifest(_))
        ));
    }

    #[test]
    fn manifest_rejects_duplicate_action_ids() {
        let mut manifest = manifest(true);
        manifest.worlds[0].legal_action_ids = vec!["north".into(), "north".into()];
        assert!(matches!(
            manifest.validate(),
            Err(ProtocolError::DuplicateIdentifier { .. })
        ));
    }

    #[test]
    fn candidate_claim_rejects_unknown_action() {
        assert!(matches!(
            score_candidate_claim(
                &manifest(true),
                &ids(&["world-north"]),
                &CandidateClaim {
                    retained_mechanism_variant_ids: ids(&["opens_north"]),
                    selected_action_id: Some("teleport".into()),
                },
            ),
            Err(ProtocolError::InvalidCandidateAction(_))
        ));
    }

    #[test]
    fn candidate_claim_rejects_unknown_mechanism() {
        assert!(matches!(
            score_candidate_claim(
                &manifest(true),
                &ids(&["world-north"]),
                &CandidateClaim {
                    retained_mechanism_variant_ids: ids(&["opens_west"]),
                    selected_action_id: None,
                },
            ),
            Err(ProtocolError::InvalidCandidateHypothesis(_))
        ));
    }

    #[test]
    fn compatible_world_set_rejects_duplicates_and_unknowns() {
        let manifest = manifest(true);
        let claim = CandidateClaim {
            retained_mechanism_variant_ids: Vec::new(),
            selected_action_id: None,
        };
        assert!(matches!(
            score_candidate_claim(
                &manifest,
                &ids(&["world-north", "world-north"]),
                &claim
            ),
            Err(ProtocolError::DuplicateCompatibleWorld(_))
        ));
        assert!(matches!(
            score_candidate_claim(&manifest, &ids(&["missing-world"]), &claim),
            Err(ProtocolError::UnknownIdentifier { .. })
        ));
    }

    #[test]
    fn tied_optima_are_intersected_as_sets() {
        let mut manifest = manifest(false);
        manifest.worlds[0].loss_by_action.insert("east".into(), 0.0);
        let score = score_candidate_claim(
            &manifest,
            &ids(&["world-north", "world-east"]),
            &CandidateClaim {
                retained_mechanism_variant_ids: Vec::new(),
                selected_action_id: Some("east".into()),
            },
        )
        .expect("score tied actions");
        assert_eq!(
            score.common_optimal_action_ids,
            ids(&["east", "north"])
        );
        assert_eq!(score.selected_action_is_common_optimal, Some(true));
    }

    #[test]
    fn valid_world_mutation_advances_state_and_charges_actual_cost() {
        let result = validate_and_apply_transition(&contract(), &progress(), &event())
            .expect("valid transition");
        assert_eq!(result.next_progress.current_step, 1);
        assert_eq!(
            result.next_progress.current_observation_digest,
            "obs-after"
        );
        assert_eq!(result.next_progress.remaining_cost_units, 1);
        assert!(result
            .next_progress
            .consumed_invocation_ids
            .contains("invocation-1"));
        assert_eq!(result.receipt.charged_cost_units, 2);
    }

    #[test]
    fn transition_rejects_read_only_or_unknown_side_effect_class() {
        let mut read_only_contract = contract();
        read_only_contract.side_effect_class = SideEffectClass::ReadOnly;
        assert_eq!(
            validate_and_apply_transition(&read_only_contract, &progress(), &event()),
            Err(ProtocolError::UnsupportedSideEffectClass)
        );

        let mut unknown_event = event();
        unknown_event.side_effect_class = SideEffectClass::Unknown;
        assert_eq!(
            validate_and_apply_transition(&contract(), &progress(), &unknown_event),
            Err(ProtocolError::UnsupportedSideEffectClass)
        );
    }

    #[test]
    fn transition_rejects_replay_and_stale_pre_state() {
        let result = validate_and_apply_transition(&contract(), &progress(), &event())
            .expect("first transition");
        assert!(matches!(
            validate_and_apply_transition(&contract(), &result.next_progress, &event()),
            Err(ProtocolError::InvocationAlreadyConsumed(_))
        ));

        let mut stale = event();
        stale.before_observation_digest = "stale-before".into();
        assert!(matches!(
            validate_and_apply_transition(&contract(), &progress(), &stale),
            Err(ProtocolError::PreStateMismatch { .. })
        ));
    }

    #[test]
    fn transition_rejects_wrong_issue_step_and_wrong_action() {
        let mut wrong_step = event();
        wrong_step.issued_at_step = 1;
        assert!(matches!(
            validate_and_apply_transition(&contract(), &progress(), &wrong_step),
            Err(ProtocolError::StaleOrOutOfOrderTransition { .. })
        ));

        let mut wrong_action = event();
        wrong_action.action_id = "move-north".into();
        assert!(matches!(
            validate_and_apply_transition(&contract(), &progress(), &wrong_action),
            Err(ProtocolError::ActionMismatch { .. })
        ));
    }

    #[test]
    fn transition_rejects_early_and_late_observations() {
        let mut early = event();
        early.observed_at_step = 0;
        assert!(matches!(
            validate_and_apply_transition(&contract(), &progress(), &early),
            Err(ProtocolError::ObservationTooEarly { .. })
        ));

        let mut late = event();
        late.observed_at_step = 9;
        assert!(matches!(
            validate_and_apply_transition(&contract(), &progress(), &late),
            Err(ProtocolError::ObservationAfterDeadline { .. })
        ));
    }

    #[test]
    fn transition_rejects_over_contract_and_over_budget_costs() {
        let mut over_contract = event();
        over_contract.actual_cost_units = 3;
        assert!(matches!(
            validate_and_apply_transition(&contract(), &progress(), &over_contract),
            Err(ProtocolError::CostAboveContract { .. })
        ));

        let mut low_budget = progress();
        low_budget.remaining_cost_units = 1;
        assert!(matches!(
            validate_and_apply_transition(&contract(), &low_budget, &event()),
            Err(ProtocolError::CostOverBudget { .. })
        ));
    }

    #[test]
    fn transition_requires_source_commitments() {
        let mut missing_event_digest = event();
        missing_event_digest.environment_event_digest.clear();
        assert!(matches!(
            validate_and_apply_transition(&contract(), &progress(), &missing_event_digest),
            Err(ProtocolError::EmptyField("event.environment_event_digest"))
        ));

        let mut missing_trace_digest = event();
        missing_trace_digest.transition_trace_digest.clear();
        assert!(matches!(
            validate_and_apply_transition(&contract(), &progress(), &missing_trace_digest),
            Err(ProtocolError::EmptyField("event.transition_trace_digest"))
        ));
    }
}
