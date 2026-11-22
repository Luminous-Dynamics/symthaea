//! Evaluator-side value of an observation measured as decision-regret reduction.
//!
//! This module composes the finite hidden-world/action-loss contract in
//! decision_observability.rs. It does not create another loss oracle, sensor
//! policy, calibration service, or acquisition executor.
//!
//! For every evaluator-owned world, a frozen sensor model supplies a discrete
//! candidate-visible outcome likelihood and an availability time. Outcomes
//! delivered after the decision deadline are collapsed into the single
//! candidate-visible "no result by deadline" class.
//!
//! Two non-interchangeable quantities are reported:
//! - robust value: reduction in the deterministic minimax-regret floor under
//!   the worst possible candidate-visible outcome;
//! - expected value: reduction in Bayes expected regret under evaluator-owned,
//!   externally qualified world priors and sensor likelihoods.
//!
//! A noisy sensor can have positive expected value and zero robust value.
//! Hidden worlds, likelihoods, priors and loss tables are evaluator-only.
//! Package separation is not an OS isolation boundary; #9437 remains required.

use crate::decision_observability::{
    ActionLoss, CounterfactualWorld, DecisionObservabilityError,
    ObservationalEquivalenceClass, evaluate_decision_observability,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DECISION_REGRET_VOI_PPM: u32 = 1_000_000;
const PPM_U128: u128 = 1_000_000;
const EXPECTED_REGRET_DENOMINATOR: u128 = PPM_U128 * PPM_U128;
const MAX_OUTCOME_ROWS: usize = 65_536;
const NO_RESULT_LABEL: &str = "no_result_by_decision_deadline";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldPrior {
    /// Evaluator-only world identity from the base equivalence class.
    pub evaluator_world_id: u64,
    /// Externally qualified prior mass in parts per million; must be nonzero.
    pub prior_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SensorOutcomeLikelihood {
    /// Evaluator-only world identity.
    pub evaluator_world_id: u64,
    /// SHA-256 of the complete candidate-visible transcript after this result
    /// is appended, including the delivered measurement, masks, status and
    /// availability timestamp. Identical visible transcripts must share a
    /// digest across worlds.
    pub candidate_visible_outcome_sha256: [u8; 32],
    /// Conditional likelihood of this outcome in this world, in PPM.
    pub likelihood_ppm: u32,
    /// Absolute timestamp in the episode clock. None means no result is
    /// available. Results strictly after the deadline are also unavailable.
    pub available_at_us: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationModel {
    /// Stable identity of the exact sensor model and configuration.
    pub sensor_model_sha256: [u8; 32],
    /// Commitment to the separately qualified prior/probability artifact.
    /// A nonzero commitment binds provenance; it does not itself prove
    /// calibration. Expected-value claims still require the calibration owner.
    pub prior_commitment_sha256: [u8; 32],
    /// Decision deadline on the same absolute episode clock as availability.
    pub decision_deadline_us: u64,
    pub world_priors: Vec<WorldPrior>,
    pub outcome_likelihoods: Vec<SensorOutcomeLikelihood>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum CandidateVisibleOutcome {
    Delivered([u8; 32]),
    NoResultByDeadline,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationOutcomeCertificate {
    /// None denotes the single candidate-visible no-result-by-deadline class.
    pub candidate_visible_outcome_sha256: Option<[u8; 32]>,
    pub possible_world_count: u64,
    /// Evaluator-only worst-case-regret-optimal action for this outcome.
    pub minimax_regret_action_id: String,
    pub worst_case_regret_ppm: u32,
    /// Evaluator-only Bayes-regret-optimal action for this outcome.
    pub bayes_regret_action_id: String,
    /// Joint probability mass in PPM squared (prior x likelihood).
    pub joint_probability_mass_ppm_squared: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionRegretObservationCertificate {
    pub equivalence_class_id: u64,
    pub sensor_model_sha256: [u8; 32],
    pub prior_commitment_sha256: [u8; 32],
    pub decision_deadline_us: u64,
    pub baseline_worst_case_regret_ppm: u32,
    pub post_observation_worst_case_regret_ppm: u32,
    /// Guaranteed reduction across every possible candidate-visible outcome.
    pub guaranteed_regret_reduction_ppm: u32,
    /// Exact expected-regret numerators use EXPECTED_REGRET_DENOMINATOR.
    pub expected_regret_denominator: u128,
    pub baseline_expected_regret_numerator: u128,
    pub post_observation_expected_regret_numerator: u128,
    pub expected_regret_reduction_numerator: u128,
    /// Conservative integer lower bound on expected regret reduction in PPM.
    pub expected_regret_reduction_floor_ppm: u32,
    pub outcomes: Vec<ObservationOutcomeCertificate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecisionRegretVoiError {
    EmptySensorModelIdentity,
    EmptyPriorCommitment,
    EmptyPriorSet,
    DuplicatePriorWorld { world_id: u64 },
    PriorWorldSetMismatch,
    ZeroPriorProbability { world_id: u64 },
    PriorProbabilityOutOfRange { world_id: u64, probability_ppm: u32 },
    PriorMassNotOneMillion { actual_ppm: u64 },
    TooManyOutcomeRows,
    OutcomeWorldUnknown { world_id: u64 },
    OutcomeLikelihoodOutOfRange { world_id: u64, likelihood_ppm: u32 },
    OutcomeLikelihoodMassNotOneMillion { world_id: u64, actual_ppm: u64 },
    EmptyDeliveredOutcomeDigest { world_id: u64 },
    ExpectedRegretIncreased,
    ArithmeticOverflow,
    DecisionObservability(DecisionObservabilityError),
}

impl fmt::Display for DecisionRegretVoiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySensorModelIdentity => write!(formatter, "sensor-model digest must be nonzero"),
            Self::EmptyPriorCommitment => write!(formatter, "prior commitment digest must be nonzero"),
            Self::EmptyPriorSet => write!(formatter, "world-prior set must not be empty"),
            Self::DuplicatePriorWorld { world_id } => write!(formatter, "duplicate prior world ID {world_id}"),
            Self::PriorWorldSetMismatch => write!(formatter, "prior world IDs must exactly match the base equivalence class"),
            Self::ZeroPriorProbability { world_id } => write!(formatter, "world {world_id} has zero prior mass; remove it from the expected-value support or assign qualified positive mass"),
            Self::PriorProbabilityOutOfRange { world_id, probability_ppm } => write!(formatter, "world {world_id} prior {probability_ppm} exceeds one million PPM"),
            Self::PriorMassNotOneMillion { actual_ppm } => write!(formatter, "world priors sum to {actual_ppm} PPM rather than one million"),
            Self::TooManyOutcomeRows => write!(formatter, "sensor outcome rows exceed the bounded evaluator limit"),
            Self::OutcomeWorldUnknown { world_id } => write!(formatter, "sensor outcome references unknown world {world_id}"),
            Self::OutcomeLikelihoodOutOfRange { world_id, likelihood_ppm } => write!(formatter, "world {world_id} likelihood {likelihood_ppm} exceeds one million PPM"),
            Self::OutcomeLikelihoodMassNotOneMillion { world_id, actual_ppm } => write!(formatter, "sensor outcome likelihoods for world {world_id} sum to {actual_ppm} PPM rather than one million"),
            Self::EmptyDeliveredOutcomeDigest { world_id } => write!(formatter, "delivered outcome for world {world_id} has an empty candidate-visible transcript digest"),
            Self::ExpectedRegretIncreased => write!(formatter, "exact expected regret increased after conditioning on the supplied observation model"),
            Self::ArithmeticOverflow => write!(formatter, "decision-regret VOI arithmetic overflowed"),
            Self::DecisionObservability(error) => write!(formatter, "base decision-observability qualification failed: {error}"),
        }
    }
}

impl std::error::Error for DecisionRegretVoiError {}

impl From<DecisionObservabilityError> for DecisionRegretVoiError {
    fn from(value: DecisionObservabilityError) -> Self {
        Self::DecisionObservability(value)
    }
}

fn is_zero_digest(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn checked_add(total: u128, value: u128) -> Result<u128, DecisionRegretVoiError> {
    total.checked_add(value).ok_or(DecisionRegretVoiError::ArithmeticOverflow)
}

fn checked_mul(left: u128, right: u128) -> Result<u128, DecisionRegretVoiError> {
    left.checked_mul(right).ok_or(DecisionRegretVoiError::ArithmeticOverflow)
}

fn action_regrets(world: &CounterfactualWorld) -> Result<BTreeMap<String, u32>, DecisionRegretVoiError> {
    let best_loss = world
        .action_losses
        .iter()
        .map(|loss: &ActionLoss| loss.loss_ppm)
        .min()
        .ok_or(DecisionRegretVoiError::ArithmeticOverflow)?;
    Ok(world
        .action_losses
        .iter()
        .map(|loss| (loss.action_id.clone(), loss.loss_ppm.saturating_sub(best_loss)))
        .collect())
}

fn validate_priors(
    class: &ObservationalEquivalenceClass,
    priors: &[WorldPrior],
) -> Result<BTreeMap<u64, u32>, DecisionRegretVoiError> {
    if priors.is_empty() {
        return Err(DecisionRegretVoiError::EmptyPriorSet);
    }

    let world_ids: BTreeSet<u64> = class
        .alternatives
        .iter()
        .map(|world| world.evaluator_world_id)
        .collect();
    let mut weights = BTreeMap::new();
    let mut total = 0_u64;

    for prior in priors {
        if prior.prior_ppm == 0 {
            return Err(DecisionRegretVoiError::ZeroPriorProbability {
                world_id: prior.evaluator_world_id,
            });
        }
        if prior.prior_ppm > DECISION_REGRET_VOI_PPM {
            return Err(DecisionRegretVoiError::PriorProbabilityOutOfRange {
                world_id: prior.evaluator_world_id,
                probability_ppm: prior.prior_ppm,
            });
        }
        if weights.insert(prior.evaluator_world_id, prior.prior_ppm).is_some() {
            return Err(DecisionRegretVoiError::DuplicatePriorWorld {
                world_id: prior.evaluator_world_id,
            });
        }
        total = total
            .checked_add(u64::from(prior.prior_ppm))
            .ok_or(DecisionRegretVoiError::ArithmeticOverflow)?;
    }

    if weights.keys().copied().collect::<BTreeSet<_>>() != world_ids {
        return Err(DecisionRegretVoiError::PriorWorldSetMismatch);
    }
    if total != u64::from(DECISION_REGRET_VOI_PPM) {
        return Err(DecisionRegretVoiError::PriorMassNotOneMillion { actual_ppm: total });
    }
    Ok(weights)
}

fn normalize_visible_outcome(
    row: &SensorOutcomeLikelihood,
    deadline_us: u64,
) -> Result<CandidateVisibleOutcome, DecisionRegretVoiError> {
    match row.available_at_us {
        Some(available_at_us) if available_at_us <= deadline_us => {
            if is_zero_digest(&row.candidate_visible_outcome_sha256) {
                return Err(DecisionRegretVoiError::EmptyDeliveredOutcomeDigest {
                    world_id: row.evaluator_world_id,
                });
            }
            Ok(CandidateVisibleOutcome::Delivered(row.candidate_visible_outcome_sha256))
        }
        Some(_) | None => Ok(CandidateVisibleOutcome::NoResultByDeadline),
    }
}

fn validate_likelihoods(
    class: &ObservationalEquivalenceClass,
    model: &ObservationModel,
) -> Result<
    BTreeMap<u64, BTreeMap<CandidateVisibleOutcome, u32>>,
    DecisionRegretVoiError,
> {
    if model.outcome_likelihoods.len() > MAX_OUTCOME_ROWS {
        return Err(DecisionRegretVoiError::TooManyOutcomeRows);
    }

    let world_ids: BTreeSet<u64> = class
        .alternatives
        .iter()
        .map(|world| world.evaluator_world_id)
        .collect();
    let mut by_world: BTreeMap<u64, BTreeMap<CandidateVisibleOutcome, u32>> = BTreeMap::new();
    let mut totals: BTreeMap<u64, u64> = BTreeMap::new();

    for row in &model.outcome_likelihoods {
        if !world_ids.contains(&row.evaluator_world_id) {
            return Err(DecisionRegretVoiError::OutcomeWorldUnknown {
                world_id: row.evaluator_world_id,
            });
        }
        if row.likelihood_ppm > DECISION_REGRET_VOI_PPM {
            return Err(DecisionRegretVoiError::OutcomeLikelihoodOutOfRange {
                world_id: row.evaluator_world_id,
                likelihood_ppm: row.likelihood_ppm,
            });
        }

        let visible = normalize_visible_outcome(row, model.decision_deadline_us)?;
        let world_rows = by_world.entry(row.evaluator_world_id).or_default();
        let current = world_rows.get(&visible).copied().unwrap_or(0);
        let combined = current
            .checked_add(row.likelihood_ppm)
            .ok_or(DecisionRegretVoiError::ArithmeticOverflow)?;
        world_rows.insert(visible, combined);

        let total = totals.entry(row.evaluator_world_id).or_default();
        *total = total
            .checked_add(u64::from(row.likelihood_ppm))
            .ok_or(DecisionRegretVoiError::ArithmeticOverflow)?;
    }

    for world_id in &world_ids {
        let total = totals.get(world_id).copied().unwrap_or(0);
        if total != u64::from(DECISION_REGRET_VOI_PPM) {
            return Err(DecisionRegretVoiError::OutcomeLikelihoodMassNotOneMillion {
                world_id: *world_id,
                actual_ppm: total,
            });
        }
    }

    Ok(by_world)
}

fn baseline_expected_regret_numerator(
    class: &ObservationalEquivalenceClass,
    priors: &BTreeMap<u64, u32>,
) -> Result<u128, DecisionRegretVoiError> {
    let first = class
        .alternatives
        .first()
        .ok_or(DecisionRegretVoiError::EmptyPriorSet)?;
    let actions: BTreeSet<String> = first
        .action_losses
        .iter()
        .map(|item| item.action_id.clone())
        .collect();
    let mut by_action: BTreeMap<String, u128> =
        actions.iter().map(|action| (action.clone(), 0)).collect();

    for world in &class.alternatives {
        let prior = u128::from(
            *priors
                .get(&world.evaluator_world_id)
                .ok_or(DecisionRegretVoiError::PriorWorldSetMismatch)?,
        );
        let regrets = action_regrets(world)?;
        for action in &actions {
            let regret = u128::from(
                *regrets
                    .get(action)
                    .ok_or(DecisionRegretVoiError::ArithmeticOverflow)?,
            );
            let contribution = checked_mul(prior, regret)?;
            let current = by_action.get(action).copied().unwrap_or(0);
            by_action.insert(action.clone(), checked_add(current, contribution)?);
        }
    }

    let best_before = by_action
        .values()
        .copied()
        .min()
        .ok_or(DecisionRegretVoiError::ArithmeticOverflow)?;
    // Convert the prior-weighted PPM numerator (denominator 1e6) to the common
    // prior-times-likelihood denominator (1e12).
    checked_mul(best_before, PPM_U128)
}

fn bayes_action_for_outcome(
    class: &ObservationalEquivalenceClass,
    priors: &BTreeMap<u64, u32>,
    likelihoods: &BTreeMap<u64, BTreeMap<CandidateVisibleOutcome, u32>>,
    outcome: CandidateVisibleOutcome,
) -> Result<(String, u128, u64), DecisionRegretVoiError> {
    let first = class
        .alternatives
        .first()
        .ok_or(DecisionRegretVoiError::EmptyPriorSet)?;
    let actions: BTreeSet<String> = first
        .action_losses
        .iter()
        .map(|item| item.action_id.clone())
        .collect();
    let mut action_sums: BTreeMap<String, u128> =
        actions.iter().map(|action| (action.clone(), 0)).collect();
    let mut possible_world_count = 0_u64;

    for world in &class.alternatives {
        let likelihood = likelihoods
            .get(&world.evaluator_world_id)
            .and_then(|items| items.get(&outcome))
            .copied()
            .unwrap_or(0);
        if likelihood == 0 {
            continue;
        }
        possible_world_count = possible_world_count
            .checked_add(1)
            .ok_or(DecisionRegretVoiError::ArithmeticOverflow)?;
        let joint_mass = checked_mul(
            u128::from(
                *priors
                    .get(&world.evaluator_world_id)
                    .ok_or(DecisionRegretVoiError::PriorWorldSetMismatch)?,
            ),
            u128::from(likelihood),
        )?;
        let regrets = action_regrets(world)?;
        for action in &actions {
            let regret = u128::from(
                *regrets
                    .get(action)
                    .ok_or(DecisionRegretVoiError::ArithmeticOverflow)?,
            );
            let contribution = checked_mul(joint_mass, regret)?;
            let current = action_sums.get(action).copied().unwrap_or(0);
            action_sums.insert(action.clone(), checked_add(current, contribution)?);
        }
    }

    let (best_action, best_sum) = action_sums
        .iter()
        .min_by(|left, right| left.1.cmp(right.1).then_with(|| left.0.cmp(right.0)))
        .map(|(action, sum)| (action.clone(), *sum))
        .ok_or(DecisionRegretVoiError::ArithmeticOverflow)?;
    Ok((best_action, best_sum, possible_world_count))
}

fn robust_outcome_certificate(
    class: &ObservationalEquivalenceClass,
    possible_worlds: &BTreeSet<u64>,
) -> Result<(String, u32), DecisionRegretVoiError> {
    let reduced = ObservationalEquivalenceClass {
        equivalence_class_id: class.equivalence_class_id,
        alternatives: class
            .alternatives
            .iter()
            .filter(|world| possible_worlds.contains(&world.evaluator_world_id))
            .cloned()
            .collect(),
    };
    let certificate = evaluate_decision_observability(&reduced)?;
    Ok((
        certificate.minimax_regret_action_id,
        certificate.unavoidable_worst_case_regret_ppm,
    ))
}

/// Evaluate the robust and expected decision value of one sensor model.
///
/// The base class must already be evaluator-qualified. Priors and likelihoods
/// are evaluator-only inputs; prior_commitment_sha256 must identify a separate
/// calibration/provenance artifact before expected-value claims are made.
pub fn evaluate_decision_regret_observation(
    class: &ObservationalEquivalenceClass,
    model: &ObservationModel,
) -> Result<DecisionRegretObservationCertificate, DecisionRegretVoiError> {
    if is_zero_digest(&model.sensor_model_sha256) {
        return Err(DecisionRegretVoiError::EmptySensorModelIdentity);
    }
    if is_zero_digest(&model.prior_commitment_sha256) {
        return Err(DecisionRegretVoiError::EmptyPriorCommitment);
    }

    let baseline = evaluate_decision_observability(class)?;
    let priors = validate_priors(class, &model.world_priors)?;
    let likelihoods = validate_likelihoods(class, model)?;

    let baseline_common_numerator =
        baseline_expected_regret_numerator(class, &priors)?;
    let mut after_expected_numerator = 0_u128;

    let mut worlds_by_outcome: BTreeMap<CandidateVisibleOutcome, BTreeSet<u64>> =
        BTreeMap::new();
    let mut mass_by_outcome: BTreeMap<CandidateVisibleOutcome, u128> = BTreeMap::new();
    for (world_id, outcomes) in &likelihoods {
        let prior = u128::from(
            *priors
                .get(world_id)
                .ok_or(DecisionRegretVoiError::PriorWorldSetMismatch)?,
        );
        for (outcome, likelihood) in outcomes {
            if *likelihood == 0 {
                continue;
            }
            worlds_by_outcome
                .entry(outcome.clone())
                .or_default()
                .insert(*world_id);
            let joint_mass = checked_mul(prior, u128::from(*likelihood))?;
            let current = mass_by_outcome.get(outcome).copied().unwrap_or(0);
            mass_by_outcome.insert(outcome.clone(), checked_add(current, joint_mass)?);
        }
    }

    let mut post_worst_case_regret = 0_u32;
    let mut outcome_certificates = Vec::with_capacity(worlds_by_outcome.len());
    for (outcome, possible_worlds) in worlds_by_outcome {
        let (minimax_action, worst_case_regret) =
            robust_outcome_certificate(class, &possible_worlds)?;
        post_worst_case_regret = post_worst_case_regret.max(worst_case_regret);

        let (bayes_action, expected_numerator, expected_world_count) =
            bayes_action_for_outcome(class, &priors, &likelihoods, outcome)?;
        if expected_world_count != possible_worlds.len() as u64 {
            return Err(DecisionRegretVoiError::ArithmeticOverflow);
        }
        after_expected_numerator =
            checked_add(after_expected_numerator, expected_numerator)?;

        outcome_certificates.push(ObservationOutcomeCertificate {
            candidate_visible_outcome_sha256: match outcome {
                CandidateVisibleOutcome::Delivered(digest) => Some(digest),
                CandidateVisibleOutcome::NoResultByDeadline => None,
            },
            possible_world_count: possible_worlds.len() as u64,
            minimax_regret_action_id: minimax_action,
            worst_case_regret_ppm: worst_case_regret,
            bayes_regret_action_id: bayes_action,
            joint_probability_mass_ppm_squared: mass_by_outcome
                .get(&outcome)
                .copied()
                .unwrap_or(0),
        });
    }

    if after_expected_numerator > baseline_common_numerator {
        return Err(DecisionRegretVoiError::ExpectedRegretIncreased);
    }

    let expected_reduction = baseline_common_numerator - after_expected_numerator;
    let guaranteed_reduction = baseline
        .unavoidable_worst_case_regret_ppm
        .checked_sub(post_worst_case_regret)
        .ok_or(DecisionRegretVoiError::ArithmeticOverflow)?;

    Ok(DecisionRegretObservationCertificate {
        equivalence_class_id: class.equivalence_class_id,
        sensor_model_sha256: model.sensor_model_sha256,
        prior_commitment_sha256: model.prior_commitment_sha256,
        decision_deadline_us: model.decision_deadline_us,
        baseline_worst_case_regret_ppm: baseline.unavoidable_worst_case_regret_ppm,
        post_observation_worst_case_regret_ppm: post_worst_case_regret,
        guaranteed_regret_reduction_ppm: guaranteed_reduction,
        expected_regret_denominator: EXPECTED_REGRET_DENOMINATOR,
        baseline_expected_regret_numerator: baseline_common_numerator,
        post_observation_expected_regret_numerator: after_expected_numerator,
        expected_regret_reduction_numerator: expected_reduction,
        expected_regret_reduction_floor_ppm: u32::try_from(expected_reduction / EXPECTED_REGRET_DENOMINATOR)
            .map_err(|_| DecisionRegretVoiError::ArithmeticOverflow)?,
        outcomes: outcome_certificates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_observability::{CandidateViewIdentity, CounterfactualWorld};

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn view() -> CandidateViewIdentity {
        CandidateViewIdentity {
            observation_history_sha256: digest(1),
            recurrent_state_sha256: digest(2),
            model_configuration_sha256: digest(3),
            policy_seed: 17,
        }
    }

    fn world(
        world_id: u64,
        candidate_action: &str,
        losses: &[(&str, u32)],
    ) -> CounterfactualWorld {
        CounterfactualWorld {
            evaluator_world_id: world_id,
            hidden_world_sha256: digest(world_id as u8 + 10),
            candidate_view: view(),
            candidate_action_id: candidate_action.to_owned(),
            action_losses: losses
                .iter()
                .map(|(action, loss)| ActionLoss {
                    action_id: (*action).to_owned(),
                    loss_ppm: *loss,
                })
                .collect(),
        }
    }

    fn opposed_worlds() -> ObservationalEquivalenceClass {
        ObservationalEquivalenceClass {
            equivalence_class_id: 77,
            alternatives: vec![
                world(1, "left", &[("left", 0), ("right", 1_000_000)]),
                world(2, "left", &[("left", 1_000_000), ("right", 0)]),
            ],
        }
    }

    fn model(rows: Vec<SensorOutcomeLikelihood>, deadline: u64) -> ObservationModel {
        ObservationModel {
            sensor_model_sha256: digest(4),
            prior_commitment_sha256: digest(5),
            decision_deadline_us: deadline,
            world_priors: vec![
                WorldPrior {
                    evaluator_world_id: 1,
                    prior_ppm: 500_000,
                },
                WorldPrior {
                    evaluator_world_id: 2,
                    prior_ppm: 500_000,
                },
            ],
            outcome_likelihoods: rows,
        }
    }

    fn row(
        world_id: u64,
        outcome_digest: u8,
        likelihood_ppm: u32,
        available_at_us: Option<u64>,
    ) -> SensorOutcomeLikelihood {
        SensorOutcomeLikelihood {
            evaluator_world_id: world_id,
            candidate_visible_outcome_sha256: digest(outcome_digest),
            likelihood_ppm,
            available_at_us,
        }
    }

    #[test]
    fn perfect_observation_removes_decision_relevant_ambiguity() {
        let class = opposed_worlds();
        let model = model(
            vec![row(1, 10, 1_000_000, Some(100)), row(2, 20, 1_000_000, Some(100))],
            100,
        );
        let result = evaluate_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.baseline_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.post_observation_worst_case_regret_ppm, 0);
        assert_eq!(result.guaranteed_regret_reduction_ppm, 1_000_000);
        assert_eq!(result.expected_regret_reduction_floor_ppm, 500_000);
        assert_eq!(result.expected_regret_denominator, 1_000_000_000_000);
        assert_eq!(result.outcomes.len(), 2);
    }

    #[test]
    fn noisy_full_support_sensor_has_expected_value_but_no_worst_case_guarantee() {
        let class = opposed_worlds();
        let model = model(
            vec![
                row(1, 10, 800_000, Some(50)),
                row(1, 20, 200_000, Some(50)),
                row(2, 10, 200_000, Some(50)),
                row(2, 20, 800_000, Some(50)),
            ],
            100,
        );
        let result = evaluate_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.baseline_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.post_observation_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.guaranteed_regret_reduction_ppm, 0);
        assert_eq!(result.expected_regret_reduction_floor_ppm, 300_000);
        assert_eq!(result.baseline_expected_regret_numerator, 500_000_000_000_000_000);
        assert_eq!(result.post_observation_expected_regret_numerator, 200_000_000_000_000_000);
    }

    #[test]
    fn decision_irrelevant_sensor_outcomes_have_zero_value() {
        let class = ObservationalEquivalenceClass {
            equivalence_class_id: 78,
            alternatives: vec![
                world(1, "continue", &[("continue", 0), ("abstain", 100_000)]),
                world(2, "continue", &[("continue", 0), ("abstain", 200_000)]),
            ],
        };
        let model = model(
            vec![row(1, 30, 1_000_000, Some(40)), row(2, 40, 1_000_000, Some(40))],
            100,
        );
        let result = evaluate_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.baseline_worst_case_regret_ppm, 0);
        assert_eq!(result.post_observation_worst_case_regret_ppm, 0);
        assert_eq!(result.guaranteed_regret_reduction_ppm, 0);
        assert_eq!(result.expected_regret_reduction_floor_ppm, 0);
    }

    #[test]
    fn late_observation_is_the_same_candidate_view_at_the_deadline() {
        let class = opposed_worlds();
        let model = model(
            vec![row(1, 10, 1_000_000, Some(101)), row(2, 20, 1_000_000, Some(101))],
            100,
        );
        let result = evaluate_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(result.outcomes[0].candidate_visible_outcome_sha256, None);
        assert_eq!(result.post_observation_worst_case_regret_ppm, 1_000_000);
        assert_eq!(result.guaranteed_regret_reduction_ppm, 0);
        assert_eq!(result.expected_regret_reduction_floor_ppm, 0);
    }

    #[test]
    fn delivery_exactly_at_the_deadline_is_available() {
        let class = opposed_worlds();
        let model = model(
            vec![row(1, 10, 1_000_000, Some(100)), row(2, 20, 1_000_000, Some(100))],
            100,
        );
        let result = evaluate_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.post_observation_worst_case_regret_ppm, 0);
    }

    #[test]
    fn missing_observation_is_an_explicit_outcome_not_silent_exclusion() {
        let class = opposed_worlds();
        let model = model(
            vec![row(1, 0, 1_000_000, None), row(2, 0, 1_000_000, None)],
            100,
        );
        let result = evaluate_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(result.outcomes[0].candidate_visible_outcome_sha256, None);
        assert_eq!(result.outcomes[0].possible_world_count, 2);
    }

    #[test]
    fn rejects_unqualified_prior_probability_mass() {
        let class = opposed_worlds();
        let mut model = model(
            vec![row(1, 10, 1_000_000, Some(50)), row(2, 20, 1_000_000, Some(50))],
            100,
        );
        model.world_priors[1].prior_ppm = 400_000;
        assert_eq!(
            evaluate_decision_regret_observation(&class, &model),
            Err(DecisionRegretVoiError::PriorMassNotOneMillion {
                actual_ppm: 900_000
            })
        );
    }

    #[test]
    fn rejects_incomplete_sensor_likelihood_mass_per_world() {
        let class = opposed_worlds();
        let model = model(
            vec![row(1, 10, 1_000_000, Some(50)), row(2, 20, 900_000, Some(50))],
            100,
        );
        assert_eq!(
            evaluate_decision_regret_observation(&class, &model),
            Err(DecisionRegretVoiError::OutcomeLikelihoodMassNotOneMillion {
                world_id: 2,
                actual_ppm: 900_000
            })
        );
    }

    #[test]
    fn aggregates_latent_outcomes_with_identical_visible_transcripts() {
        let class = opposed_worlds();
        let model = model(
            vec![
                row(1, 10, 400_000, Some(50)),
                row(1, 10, 600_000, Some(50)),
                row(2, 20, 1_000_000, Some(50)),
            ],
            100,
        );
        let result = evaluate_decision_regret_observation(&class, &model).unwrap();
        assert_eq!(result.outcomes.len(), 2);
        assert_eq!(result.guaranteed_regret_reduction_ppm, 1_000_000);
    }

    #[test]
    fn rejects_outcomes_for_worlds_outside_the_base_class() {
        let class = opposed_worlds();
        let model = model(
            vec![row(1, 10, 1_000_000, Some(50)), row(3, 20, 1_000_000, Some(50))],
            100,
        );
        assert_eq!(
            evaluate_decision_regret_observation(&class, &model),
            Err(DecisionRegretVoiError::OutcomeWorldUnknown { world_id: 3 })
        );
    }

    #[test]
    fn rejects_missing_model_and_prior_provenance_commitments() {
        let class = opposed_worlds();
        let mut model = model(
            vec![row(1, 10, 1_000_000, Some(50)), row(2, 20, 1_000_000, Some(50))],
            100,
        );
        model.sensor_model_sha256 = [0; 32];
        assert_eq!(
            evaluate_decision_regret_observation(&class, &model),
            Err(DecisionRegretVoiError::EmptySensorModelIdentity)
        );
        model.sensor_model_sha256 = digest(4);
        model.prior_commitment_sha256 = [0; 32];
        assert_eq!(
            evaluate_decision_regret_observation(&class, &model),
            Err(DecisionRegretVoiError::EmptyPriorCommitment)
        );
    }
}
