//! Finite-action observational equivalence and separation-cost oracle.
//!
//! Evaluator-side exact bounded oracle. It computes response-signature
//! equivalence classes over a frozen finite action set, then searches for the
//! minimum-cost known-variance Gaussian replication schedule that separates
//! every non-equivalent pair at a Bonferroni family-wise alpha and target power.
//!
//! This is not a posterior, an adaptive action selector, a composite-null
//! test, or a physical measurement model.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use super::{normal_quantile, one_sided_power};

pub const MAX_ACTIONS: usize = 6;
pub const MAX_MECHANISMS: usize = 12;
pub const MAX_TOTAL_OBSERVATIONS: usize = 16;
const MAX_EXACT_SIGNATURE_BUCKET: f64 = 9_007_199_254_740_991.0;

/// Frozen action with known Gaussian noise, positive per-replicate cost and
/// evaluator-side response-signature quantum.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementAction {
    pub id: String,
    pub sigma: f64,
    pub cost: f64,
    pub response_quantum: f64,
}

/// Expected observable means in the same action order as the input actions.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateMechanism {
    pub id: String,
    pub expected_means: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MechanismEquivalenceClass {
    pub canonical_id: String,
    pub member_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionReplication {
    pub action_id: String,
    pub replicates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeparationDisposition {
    SingleEquivalenceClass,
    TargetAchievable,
    BudgetInsufficient,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorstSeparatedPair {
    pub left_mechanism_id: String,
    pub right_mechanism_id: String,
    pub information: f64,
    pub power: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IdentifiabilityEnvelopeReport {
    pub familywise_alpha: f64,
    pub target_power: f64,
    pub pairwise_alpha: f64,
    pub required_pairwise_information: f64,
    pub mechanism_count: usize,
    pub equivalence_class_count: usize,
    pub total_mechanism_pairs: usize,
    pub irreducible_equivalent_pair_count: usize,
    pub distinguishable_mechanism_pair_count: usize,
    pub irreducible_equivalent_pair_fraction: f64,
    pub equivalence_classes: Vec<MechanismEquivalenceClass>,
    pub max_total_observations: usize,
    pub disposition: SeparationDisposition,
    pub replications: Vec<ActionReplication>,
    pub total_observations: usize,
    pub total_measurement_cost: f64,
    pub worst_pair: Option<WorstSeparatedPair>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeparationError {
    EmptyActions,
    EmptyMechanisms,
    TooManyActions {
        actual: usize,
        maximum: usize,
    },
    TooManyMechanisms {
        actual: usize,
        maximum: usize,
    },
    ObservationCapTooLarge {
        actual: usize,
        maximum: usize,
    },
    EmptyActionId,
    EmptyMechanismId,
    DuplicateActionId(String),
    DuplicateMechanismId(String),
    InvalidSigma {
        action_id: String,
    },
    InvalidCost {
        action_id: String,
    },
    InvalidResponseQuantum {
        action_id: String,
    },
    InvalidAlpha,
    InvalidTargetPower,
    MechanismActionCountMismatch {
        mechanism_id: String,
        expected: usize,
        actual: usize,
    },
    NonFiniteMean {
        mechanism_id: String,
        action_id: String,
    },
    SignatureBucketOutOfRange {
        mechanism_id: String,
        action_id: String,
    },
    NumericalOverflow,
}

impl fmt::Display for SeparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyActions => write!(f, "at least one measurement action is required"),
            Self::EmptyMechanisms => write!(f, "at least one candidate mechanism is required"),
            Self::TooManyActions { actual, maximum } => {
                write!(f, "{actual} actions exceeds exact-search limit {maximum}")
            }
            Self::TooManyMechanisms { actual, maximum } => write!(
                f,
                "{actual} mechanisms exceeds exact-search limit {maximum}"
            ),
            Self::ObservationCapTooLarge { actual, maximum } => write!(
                f,
                "observation cap {actual} exceeds exact-search limit {maximum}"
            ),
            Self::EmptyActionId => write!(f, "action ID must not be empty"),
            Self::EmptyMechanismId => write!(f, "mechanism ID must not be empty"),
            Self::DuplicateActionId(id) => write!(f, "duplicate action ID: {id}"),
            Self::DuplicateMechanismId(id) => write!(f, "duplicate mechanism ID: {id}"),
            Self::InvalidSigma { action_id } => write!(
                f,
                "sigma for action {action_id} must be finite and positive"
            ),
            Self::InvalidCost { action_id } => {
                write!(f, "cost for action {action_id} must be finite and positive")
            }
            Self::InvalidResponseQuantum { action_id } => write!(
                f,
                "response quantum for action {action_id} must be finite and positive"
            ),
            Self::InvalidAlpha => {
                write!(f, "family-wise alpha or adjusted pairwise alpha is invalid")
            }
            Self::InvalidTargetPower => write!(
                f,
                "target power must be finite and strictly between family-wise alpha and one"
            ),
            Self::MechanismActionCountMismatch {
                mechanism_id,
                expected,
                actual,
            } => write!(
                f,
                "mechanism {mechanism_id} has {actual} action responses; expected {expected}"
            ),
            Self::NonFiniteMean {
                mechanism_id,
                action_id,
            } => write!(
                f,
                "mechanism {mechanism_id} has a non-finite mean for action {action_id}"
            ),
            Self::SignatureBucketOutOfRange {
                mechanism_id,
                action_id,
            } => write!(
                f,
                "response signature bucket is outside exact integer range for mechanism {mechanism_id}, action {action_id}"
            ),
            Self::NumericalOverflow => {
                write!(f, "information or measurement-cost calculation overflowed")
            }
        }
    }
}
impl Error for SeparationError {}

#[derive(Debug, Clone)]
struct CanonicalMechanism {
    id: String,
    means: Vec<f64>,
    signature: Vec<i64>,
    class_index: usize,
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    counts: Vec<usize>,
    observations: usize,
    cost: f64,
    worst_pair: WorstSeparatedPair,
    meets_target: bool,
}

fn canonical_signature(
    mechanism: &CandidateMechanism,
    actions: &[MeasurementAction],
) -> Result<Vec<i64>, SeparationError> {
    let mut signature = Vec::with_capacity(actions.len());
    for (action, mean) in actions.iter().zip(mechanism.expected_means.iter().copied()) {
        if !mean.is_finite() {
            return Err(SeparationError::NonFiniteMean {
                mechanism_id: mechanism.id.clone(),
                action_id: action.id.clone(),
            });
        }
        // Quantization is a numerical signature contract, independent of sigma.
        let scaled = mean / action.response_quantum;
        if !scaled.is_finite() || scaled.abs() > MAX_EXACT_SIGNATURE_BUCKET {
            return Err(SeparationError::SignatureBucketOutOfRange {
                mechanism_id: mechanism.id.clone(),
                action_id: action.id.clone(),
            });
        }
        signature.push(scaled.round() as i64);
    }
    Ok(signature)
}

fn validate_and_canonicalize(
    actions: Vec<MeasurementAction>,
    mechanisms: Vec<CandidateMechanism>,
    familywise_alpha: f64,
    target_power: f64,
    max_total_observations: usize,
) -> Result<(Vec<MeasurementAction>, Vec<CanonicalMechanism>), SeparationError> {
    if actions.is_empty() {
        return Err(SeparationError::EmptyActions);
    }
    if mechanisms.is_empty() {
        return Err(SeparationError::EmptyMechanisms);
    }
    if actions.len() > MAX_ACTIONS {
        return Err(SeparationError::TooManyActions {
            actual: actions.len(),
            maximum: MAX_ACTIONS,
        });
    }
    if mechanisms.len() > MAX_MECHANISMS {
        return Err(SeparationError::TooManyMechanisms {
            actual: mechanisms.len(),
            maximum: MAX_MECHANISMS,
        });
    }
    if max_total_observations > MAX_TOTAL_OBSERVATIONS {
        return Err(SeparationError::ObservationCapTooLarge {
            actual: max_total_observations,
            maximum: MAX_TOTAL_OBSERVATIONS,
        });
    }
    if !familywise_alpha.is_finite() || familywise_alpha <= 1.0e-12 || familywise_alpha >= 1.0 {
        return Err(SeparationError::InvalidAlpha);
    }
    if !target_power.is_finite() || target_power <= familywise_alpha || target_power >= 1.0 {
        return Err(SeparationError::InvalidTargetPower);
    }

    let mut action_ids = BTreeSet::new();
    for action in &actions {
        if action.id.trim().is_empty() {
            return Err(SeparationError::EmptyActionId);
        }
        if !action_ids.insert(action.id.clone()) {
            return Err(SeparationError::DuplicateActionId(action.id.clone()));
        }
        if !action.sigma.is_finite() || action.sigma <= 0.0 {
            return Err(SeparationError::InvalidSigma {
                action_id: action.id.clone(),
            });
        }
        if !action.cost.is_finite() || action.cost <= 0.0 {
            return Err(SeparationError::InvalidCost {
                action_id: action.id.clone(),
            });
        }
        if !action.response_quantum.is_finite() || action.response_quantum <= 0.0 {
            return Err(SeparationError::InvalidResponseQuantum {
                action_id: action.id.clone(),
            });
        }
    }

    let mut mechanism_ids = BTreeSet::new();
    for mechanism in &mechanisms {
        if mechanism.id.trim().is_empty() {
            return Err(SeparationError::EmptyMechanismId);
        }
        if !mechanism_ids.insert(mechanism.id.clone()) {
            return Err(SeparationError::DuplicateMechanismId(mechanism.id.clone()));
        }
        if mechanism.expected_means.len() != actions.len() {
            return Err(SeparationError::MechanismActionCountMismatch {
                mechanism_id: mechanism.id.clone(),
                expected: actions.len(),
                actual: mechanism.expected_means.len(),
            });
        }
    }

    let mut action_order = (0..actions.len()).collect::<Vec<_>>();
    action_order.sort_by(|a, b| actions[*a].id.cmp(&actions[*b].id));
    let canonical_actions = action_order
        .iter()
        .map(|i| actions[*i].clone())
        .collect::<Vec<_>>();

    let mut canonical_mechanisms = Vec::with_capacity(mechanisms.len());
    for mechanism in mechanisms {
        let means = action_order
            .iter()
            .map(|i| mechanism.expected_means[*i])
            .collect::<Vec<_>>();
        let canonical = CandidateMechanism {
            id: mechanism.id.clone(),
            expected_means: means.clone(),
        };
        let signature = canonical_signature(&canonical, &canonical_actions)?;
        canonical_mechanisms.push(CanonicalMechanism {
            id: mechanism.id,
            means,
            signature,
            class_index: 0,
        });
    }
    canonical_mechanisms.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((canonical_actions, canonical_mechanisms))
}

fn build_equivalence_classes(
    mechanisms: &mut [CanonicalMechanism],
) -> Vec<MechanismEquivalenceClass> {
    let mut classes: Vec<MechanismEquivalenceClass> = Vec::new();
    let mut signatures: Vec<Vec<i64>> = Vec::new();

    for mechanism in mechanisms.iter_mut() {
        let class_index =
            if let Some(index) = signatures.iter().position(|s| s == &mechanism.signature) {
                classes[index].member_ids.push(mechanism.id.clone());
                index
            } else {
                let index = classes.len();
                signatures.push(mechanism.signature.clone());
                classes.push(MechanismEquivalenceClass {
                    canonical_id: mechanism.id.clone(),
                    member_ids: vec![mechanism.id.clone()],
                });
                index
            };
        mechanism.class_index = class_index;
    }

    classes.sort_by(|a, b| a.canonical_id.cmp(&b.canonical_id));
    let index_by_id = classes
        .iter()
        .enumerate()
        .flat_map(|(index, class)| class.member_ids.iter().cloned().map(move |id| (id, index)))
        .collect::<std::collections::BTreeMap<_, _>>();
    for mechanism in mechanisms {
        mechanism.class_index = index_by_id[&mechanism.id];
    }
    classes
}

fn choose_two(n: usize) -> usize {
    n.saturating_mul(n.saturating_sub(1)) / 2
}

fn schedule_cost(
    counts: &[usize],
    actions: &[MeasurementAction],
) -> Result<(usize, f64), SeparationError> {
    let mut observations = 0_usize;
    let mut cost = 0.0;
    for (count, action) in counts.iter().copied().zip(actions) {
        observations = observations
            .checked_add(count)
            .ok_or(SeparationError::NumericalOverflow)?;
        cost += count as f64 * action.cost;
    }
    if !cost.is_finite() {
        return Err(SeparationError::NumericalOverflow);
    }
    Ok((observations, cost))
}

fn worst_pair_for_schedule(
    counts: &[usize],
    actions: &[MeasurementAction],
    mechanisms: &[CanonicalMechanism],
    class_count: usize,
    pairwise_alpha: f64,
) -> Result<Option<WorstSeparatedPair>, SeparationError> {
    if class_count <= 1 {
        return Ok(None);
    }
    let critical = normal_quantile(1.0 - pairwise_alpha);
    if !critical.is_finite() {
        return Err(SeparationError::NumericalOverflow);
    }

    let mut worst: Option<WorstSeparatedPair> = None;
    for i in 0..mechanisms.len() {
        for j in (i + 1)..mechanisms.len() {
            let left = &mechanisms[i];
            let right = &mechanisms[j];
            if left.class_index == right.class_index {
                continue;
            }

            let mut information = 0.0;
            for (action_index, action) in actions.iter().enumerate() {
                let standardized =
                    (left.means[action_index] - right.means[action_index]) / action.sigma;
                information += counts[action_index] as f64 * standardized * standardized;
            }
            if !information.is_finite() {
                return Err(SeparationError::NumericalOverflow);
            }
            let candidate = WorstSeparatedPair {
                left_mechanism_id: left.id.clone(),
                right_mechanism_id: right.id.clone(),
                information,
                power: one_sided_power(information.sqrt(), critical),
            };
            let replace = match &worst {
                None => true,
                Some(old) => {
                    candidate.information.total_cmp(&old.information).is_lt()
                        || (candidate.information == old.information
                            && (&candidate.left_mechanism_id, &candidate.right_mechanism_id)
                                < (&old.left_mechanism_id, &old.right_mechanism_id))
                }
            };
            if replace {
                worst = Some(candidate);
            }
        }
    }
    Ok(worst)
}

fn better_achievable(candidate: &SearchCandidate, old: &SearchCandidate) -> bool {
    candidate.cost.total_cmp(&old.cost).is_lt()
        || (candidate.cost == old.cost
            && (candidate.observations < old.observations
                || (candidate.observations == old.observations && candidate.counts < old.counts)))
}

fn better_fallback(candidate: &SearchCandidate, old: &SearchCandidate) -> bool {
    candidate
        .worst_pair
        .power
        .total_cmp(&old.worst_pair.power)
        .is_gt()
        || (candidate.worst_pair.power == old.worst_pair.power
            && (candidate.cost.total_cmp(&old.cost).is_lt()
                || (candidate.cost == old.cost
                    && (candidate.observations < old.observations
                        || (candidate.observations == old.observations
                            && candidate.counts < old.counts)))))
}

struct SearchState<'a> {
    actions: &'a [MeasurementAction],
    mechanisms: &'a [CanonicalMechanism],
    class_count: usize,
    pairwise_alpha: f64,
    required_information: f64,
    best_achievable: Option<SearchCandidate>,
    best_fallback: Option<SearchCandidate>,
}

impl SearchState<'_> {
    fn evaluate(&mut self, counts: &[usize]) -> Result<(), SeparationError> {
        let (observations, cost) = schedule_cost(counts, self.actions)?;
        let Some(worst_pair) = worst_pair_for_schedule(
            counts,
            self.actions,
            self.mechanisms,
            self.class_count,
            self.pairwise_alpha,
        )?
        else {
            return Ok(());
        };
        let candidate = SearchCandidate {
            counts: counts.to_vec(),
            observations,
            cost,
            meets_target: worst_pair.information + 1.0e-12 >= self.required_information,
            worst_pair,
        };
        if candidate.meets_target {
            if self
                .best_achievable
                .as_ref()
                .is_none_or(|old| better_achievable(&candidate, old))
            {
                self.best_achievable = Some(candidate);
            }
        } else if self
            .best_fallback
            .as_ref()
            .is_none_or(|old| better_fallback(&candidate, old))
        {
            self.best_fallback = Some(candidate);
        }
        Ok(())
    }

    fn enumerate(
        &mut self,
        index: usize,
        remaining: usize,
        counts: &mut [usize],
    ) -> Result<(), SeparationError> {
        if index == self.actions.len() {
            return self.evaluate(counts);
        }
        for count in 0..=remaining {
            counts[index] = count;
            self.enumerate(index + 1, remaining - count, counts)?;
        }
        Ok(())
    }
}

/// Compute action-space equivalence and the minimum-cost bounded
/// replication schedule for separating all non-equivalent mechanisms.
///
/// Response equivalence uses a deterministic per-action quantized signature:
/// round(expected_mean / response_quantum). This is a declared numerical
/// signature contract, separate from measurement sigma and practical effect
/// size. Search is exhaustive for at most six actions, twelve mechanisms and
/// sixteen total replicates.
///
/// Pairwise alpha is family-wise alpha divided by the number of cross-class
/// mechanism pairs (all mechanism pairs minus irreducible within-class pairs).
/// Under the simple known-variance Gaussian pairwise model, power is
/// Phi(sqrt(information) - Phi_inverse(1-alpha_pair)), with information
/// sum_a r[a] * delta[a]^2 / sigma[a]^2.
pub fn compute_identifiability_envelope(
    actions: Vec<MeasurementAction>,
    mechanisms: Vec<CandidateMechanism>,
    familywise_alpha: f64,
    target_power: f64,
    max_total_observations: usize,
) -> Result<IdentifiabilityEnvelopeReport, SeparationError> {
    let (actions, mut mechanisms) = validate_and_canonicalize(
        actions,
        mechanisms,
        familywise_alpha,
        target_power,
        max_total_observations,
    )?;
    let equivalence_classes = build_equivalence_classes(&mut mechanisms);
    let mechanism_count = mechanisms.len();
    let equivalence_class_count = equivalence_classes.len();
    let total_mechanism_pairs = choose_two(mechanism_count);
    let irreducible_equivalent_pair_count = equivalence_classes
        .iter()
        .map(|class| choose_two(class.member_ids.len()))
        .sum::<usize>();
    let irreducible_equivalent_pair_fraction = if total_mechanism_pairs == 0 {
        0.0
    } else {
        irreducible_equivalent_pair_count as f64 / total_mechanism_pairs as f64
    };

    if equivalence_class_count == 1 {
        return Ok(IdentifiabilityEnvelopeReport {
            familywise_alpha,
            target_power,
            pairwise_alpha: 0.0,
            required_pairwise_information: 0.0,
            mechanism_count,
            equivalence_class_count,
            total_mechanism_pairs,
            irreducible_equivalent_pair_count,
            distinguishable_mechanism_pair_count: 0,
            irreducible_equivalent_pair_fraction,
            equivalence_classes,
            max_total_observations,
            disposition: SeparationDisposition::SingleEquivalenceClass,
            replications: actions
                .iter()
                .map(|a| ActionReplication {
                    action_id: a.id.clone(),
                    replicates: 0,
                })
                .collect(),
            total_observations: 0,
            total_measurement_cost: 0.0,
            worst_pair: None,
        });
    }

    // Bonferroni counts every cross-class mechanism pair, not merely class
    // pairs: a class may contain multiple evaluator-equivalent mechanisms.
    let distinguishable_mechanism_pair_count =
        total_mechanism_pairs - irreducible_equivalent_pair_count;
    let pairwise_alpha = familywise_alpha / distinguishable_mechanism_pair_count as f64;
    if !pairwise_alpha.is_finite() || pairwise_alpha <= 1.0e-12 {
        return Err(SeparationError::InvalidAlpha);
    }
    let z_alpha = normal_quantile(1.0 - pairwise_alpha);
    let z_power = normal_quantile(target_power);
    let required_pairwise_information = (z_alpha + z_power).powi(2);
    if !required_pairwise_information.is_finite() || required_pairwise_information <= 0.0 {
        return Err(SeparationError::NumericalOverflow);
    }

    let mut state = SearchState {
        actions: &actions,
        mechanisms: &mechanisms,
        class_count: equivalence_class_count,
        pairwise_alpha,
        required_information: required_pairwise_information,
        best_achievable: None,
        best_fallback: None,
    };
    state.enumerate(0, max_total_observations, &mut vec![0; actions.len()])?;

    let (disposition, selected) = if let Some(candidate) = state.best_achievable {
        (SeparationDisposition::TargetAchievable, candidate)
    } else if let Some(candidate) = state.best_fallback {
        (SeparationDisposition::BudgetInsufficient, candidate)
    } else {
        return Err(SeparationError::NumericalOverflow);
    };

    Ok(IdentifiabilityEnvelopeReport {
        familywise_alpha,
        target_power,
        pairwise_alpha,
        required_pairwise_information,
        mechanism_count,
        equivalence_class_count,
        total_mechanism_pairs,
        irreducible_equivalent_pair_count,
        distinguishable_mechanism_pair_count,
        irreducible_equivalent_pair_fraction,
        equivalence_classes,
        max_total_observations,
        disposition,
        replications: actions
            .iter()
            .zip(selected.counts.iter().copied())
            .map(|(action, replicates)| ActionReplication {
                action_id: action.id.clone(),
                replicates,
            })
            .collect(),
        total_observations: selected.observations,
        total_measurement_cost: selected.cost,
        worst_pair: Some(selected.worst_pair),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(id: &str, sigma: f64, cost: f64, quantum: f64) -> MeasurementAction {
        MeasurementAction {
            id: id.into(),
            sigma,
            cost,
            response_quantum: quantum,
        }
    }
    fn mechanism(id: &str, means: &[f64]) -> CandidateMechanism {
        CandidateMechanism {
            id: id.into(),
            expected_means: means.to_vec(),
        }
    }

    #[test]
    fn action_space_aliases_remain_one_equivalence_class() {
        let report = compute_identifiability_envelope(
            vec![action("x", 0.1, 1.0, 1.0e-12), action("y", 0.2, 1.0, 1.0e-12)],
            vec![mechanism("m-a", &[1.0, 2.0]), mechanism("m-b", &[1.0, 2.0]), mechanism("m-c", &[1.0, 2.0])],
            0.05, 0.80, 16,
        ).unwrap();
        assert_eq!(report.disposition, SeparationDisposition::SingleEquivalenceClass);
        assert_eq!(report.equivalence_class_count, 1);
        assert_eq!(report.irreducible_equivalent_pair_count, 3);
        assert_eq!(report.equivalence_classes[0].member_ids, vec!["m-a", "m-b", "m-c"]);
        assert_eq!(report.total_observations, 0);
        assert!(report.worst_pair.is_none());
    }

    #[test]
    fn two_mechanisms_have_hand_calculable_minimum_replicate_count() {
        let report = compute_identifiability_envelope(
            vec![action("probe", 1.0, 1.0, 1.0e-12)],
            vec![mechanism("low", &[0.0]), mechanism("high", &[1.0])],
            0.05, 0.80, 16,
        ).unwrap();
        assert_eq!(report.disposition, SeparationDisposition::TargetAchievable);
        assert_eq!(report.pairwise_alpha, 0.05);
        assert_eq!(report.total_observations, 7);
        assert_eq!(report.replications[0].replicates, 7);
        assert!(report.worst_pair.unwrap().power > 0.80);
    }

    #[test]
    fn insufficient_budget_is_not_misreported_as_equivalence() {
        let report = compute_identifiability_envelope(
            vec![action("probe", 1.0, 1.0, 1.0e-12)],
            vec![mechanism("low", &[0.0]), mechanism("high", &[1.0])],
            0.05, 0.80, 6,
        ).unwrap();
        assert_eq!(report.disposition, SeparationDisposition::BudgetInsufficient);
        assert_eq!(report.equivalence_class_count, 2);
        assert_eq!(report.total_observations, 6);
        assert!(report.worst_pair.unwrap().power < 0.80);
    }

    #[test]
    fn complementary_action_splits_a_narrow_design_alias() {
        let narrow = compute_identifiability_envelope(
            vec![action("along-diagonal", 0.1, 1.0, 1.0e-12)],
            vec![mechanism("additive", &[1.0]), mechanism("interaction", &[1.0])],
            0.05, 0.80, 16,
        ).unwrap();
        let expanded = compute_identifiability_envelope(
            vec![action("along-diagonal", 0.1, 1.0, 1.0e-12), action("off-diagonal", 0.1, 1.0, 1.0e-12)],
            vec![mechanism("additive", &[1.0, 1.0]), mechanism("interaction", &[1.0, 2.0])],
            0.05, 0.80, 16,
        ).unwrap();
        assert_eq!(narrow.disposition, SeparationDisposition::SingleEquivalenceClass);
        assert_eq!(expanded.equivalence_class_count, 2);
        assert_eq!(expanded.disposition, SeparationDisposition::TargetAchievable);
        assert_eq!(expanded.total_observations, 1);
        assert_eq!(expanded.replications[0].replicates, 0);
        assert_eq!(expanded.replications[1].replicates, 1);
    }

    #[test]
    fn precise_expensive_action_can_beat_cheap_noisy_action() {
        let report = compute_identifiability_envelope(
            vec![action("cheap-noisy", 2.0, 1.0, 1.0e-12), action("precise-expensive", 0.5, 3.0, 1.0e-12)],
            vec![mechanism("a", &[0.0, 0.0]), mechanism("b", &[1.0, 1.0])],
            0.05, 0.80, 16,
        ).unwrap();
        assert_eq!(report.disposition, SeparationDisposition::TargetAchievable);
        assert_eq!(report.replications[0].replicates, 0);
        assert_eq!(report.replications[1].replicates, 2);
        assert_eq!(report.total_observations, 2);
        assert_eq!(report.total_measurement_cost, 6.0);
    }

    #[test]
    fn joint_schedule_separates_multiple_classes_that_no_single_action_can_resolve() {
        let report = compute_identifiability_envelope(
            vec![
                action("first", 0.5, 1.0, 1.0e-12),
                action("second", 0.5, 1.0, 1.0e-12),
            ],
            vec![
                mechanism("a", &[0.0, 0.0]),
                mechanism("b", &[1.0, 0.0]),
                mechanism("c", &[1.0, 1.0]),
            ],
            0.05, 0.80, 6,
        ).unwrap();
        assert_eq!(report.disposition, SeparationDisposition::TargetAchievable);
        assert_eq!(report.equivalence_class_count, 3);
        assert_eq!(report.total_observations, 6);
        assert_eq!(report.replications[0].replicates, 3);
        assert_eq!(report.replications[1].replicates, 3);
    }

    #[test]
    fn familywise_alpha_tightens_as_class_count_grows() {
        let actions = vec![action("probe", 1.0, 1.0, 1.0e-12)];
        let two = compute_identifiability_envelope(
            actions.clone(), vec![mechanism("a", &[0.0]), mechanism("b", &[1.0])],
            0.05, 0.80, 16,
        ).unwrap();
        let three = compute_identifiability_envelope(
            actions, vec![mechanism("a", &[0.0]), mechanism("b", &[1.0]), mechanism("c", &[2.0])],
            0.05, 0.80, 16,
        ).unwrap();
        assert!(three.pairwise_alpha < two.pairwise_alpha);
        assert!(three.required_pairwise_information > two.required_pairwise_information);
    }

    #[test]
    fn bonferroni_counts_cross_class_mechanism_pairs_not_only_class_pairs() {
        let report = compute_identifiability_envelope(
            vec![action("probe", 1.0, 1.0, 1.0e-12)],
            vec![
                mechanism("a1", &[0.0]),
                mechanism("a2", &[0.0]),
                mechanism("b1", &[1.0]),
                mechanism("b2", &[1.0]),
            ],
            0.05,
            0.80,
            16,
        )
        .unwrap();
        assert_eq!(report.equivalence_class_count, 2);
        assert_eq!(report.irreducible_equivalent_pair_count, 2);
        assert_eq!(report.distinguishable_mechanism_pair_count, 4);
        assert_eq!(report.pairwise_alpha, 0.0125);
    }

    #[test]
    fn report_is_invariant_to_input_order() {
        let first = compute_identifiability_envelope(
            vec![action("b", 0.5, 2.0, 1.0e-12), action("a", 1.0, 1.0, 1.0e-12)],
            vec![mechanism("z", &[1.0, 2.0]), mechanism("b", &[0.0, 0.0]), mechanism("a", &[0.0, 0.0])],
            0.05, 0.80, 10,
        ).unwrap();
        let replay = compute_identifiability_envelope(
            vec![action("a", 1.0, 1.0, 1.0e-12), action("b", 0.5, 2.0, 1.0e-12)],
            vec![mechanism("a", &[0.0, 0.0]), mechanism("z", &[2.0, 1.0]), mechanism("b", &[0.0, 0.0])],
            0.05, 0.80, 10,
        ).unwrap();
        assert_eq!(first, replay);
        assert_eq!(first.equivalence_classes[0].canonical_id, "a");
        assert_eq!(first.equivalence_classes[0].member_ids, vec!["a", "b"]);
    }

    #[test]
    fn response_quantization_is_separate_from_measurement_noise() {
        let merged = compute_identifiability_envelope(
            vec![action("probe", 0.001, 1.0, 0.1)],
            vec![mechanism("a", &[1.01]), mechanism("b", &[1.02])],
            0.05, 0.80, 16,
        ).unwrap();
        let split = compute_identifiability_envelope(
            vec![action("probe", 0.001, 1.0, 0.001)],
            vec![mechanism("a", &[1.01]), mechanism("b", &[1.02])],
            0.05, 0.80, 16,
        ).unwrap();
        assert_eq!(merged.disposition, SeparationDisposition::SingleEquivalenceClass);
        assert_eq!(split.equivalence_class_count, 2);
    }

    #[test]
    fn malformed_actions_and_mechanisms_fail_closed() {
        assert_eq!(
            compute_identifiability_envelope(vec![], vec![mechanism("a", &[])], 0.05, 0.80, 8).err(),
            Some(SeparationError::EmptyActions)
        );
        assert_eq!(
            compute_identifiability_envelope(
                vec![action("x", 0.0, 1.0, 1.0)], vec![mechanism("a", &[0.0])], 0.05, 0.80, 8
            ).err(),
            Some(SeparationError::InvalidSigma { action_id: "x".into() })
        );
        assert!(matches!(
            compute_identifiability_envelope(
                vec![action("x", 1.0, 1.0, 1.0)], vec![mechanism("a", &[f64::NAN])],
                0.05, 0.80, 8
            ),
            Err(SeparationError::NonFiniteMean { .. })
        ));
        assert_eq!(
            compute_identifiability_envelope(
                vec![action("x", 1.0, 1.0, 1.0)],
                vec![mechanism("a", &[0.0]), mechanism("a", &[1.0])],
                0.05, 0.80, 8
            ).err(),
            Some(SeparationError::DuplicateMechanismId("a".into()))
        );
        assert!(matches!(
            compute_identifiability_envelope(
                vec![action("x", 1.0, 1.0, 1.0)], vec![mechanism("a", &[0.0])],
                0.05, 0.80, MAX_TOTAL_OBSERVATIONS + 1
            ),
            Err(SeparationError::ObservationCapTooLarge { .. })
        ));
    }
}
