//! Bounded exact-rational k-adaptable policy portfolios for finite UMDP families.
//!
//! Scope: an explicitly enumerated finite family of finite-horizon, fully
//! observable UMDPs with common state/action spaces and initial distribution.
//! Policies are deterministic time-dependent Markov policies. This is an exact
//! reference, not a continuous-polytope portfolio solver or online selector.
//!
//! The objective is evaluated in the required order:
//! max_model (V*_model - max_policy_in_portfolio V_model(policy)).
//! The portfolio member is selected separately for each same model before
//! the outer worst-case maximum is taken.

use crate::semiring::CheckedRational;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

/// One finite-horizon MDP from an explicitly listed uncertainty family.
///
/// Shapes: initial[state], rewards[time][state][action], and
/// transitions[time][state][action][next_state] with H-1 transition layers
/// for H reward/decision layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteHorizonUmdpModel {
    initial: Vec<CheckedRational>,
    rewards: Vec<Vec<Vec<CheckedRational>>>,
    transitions: Vec<Vec<Vec<Vec<CheckedRational>>>>,
}

impl FiniteHorizonUmdpModel {
    pub fn new(
        initial: Vec<CheckedRational>,
        rewards: Vec<Vec<Vec<CheckedRational>>>,
        transitions: Vec<Vec<Vec<Vec<CheckedRational>>>>,
    ) -> Result<Self, PolicyPortfolioError> {
        if initial.is_empty() {
            return Err(PolicyPortfolioError::EmptyStateSpace);
        }
        if rewards.is_empty() {
            return Err(PolicyPortfolioError::ZeroHorizon);
        }
        let states = initial.len();
        let actions = rewards[0].first().map_or(0, Vec::len);
        if actions == 0 {
            return Err(PolicyPortfolioError::EmptyActionSpace);
        }
        for (state, probability) in initial.iter().enumerate() {
            if probability.is_negative()
                || compare_signed(*probability, CheckedRational::one()) == Ordering::Greater
            {
                return Err(PolicyPortfolioError::InvalidInitialProbability { state });
            }
        }
        if checked_sum(&initial)? != CheckedRational::one() {
            return Err(PolicyPortfolioError::InitialDistributionDoesNotSumToOne);
        }

        for (time, layer) in rewards.iter().enumerate() {
            if layer.len() != states {
                return Err(PolicyPortfolioError::RewardStateCountMismatch {
                    time, expected: states, actual: layer.len(),
                });
            }
            for (state, action_rewards) in layer.iter().enumerate() {
                if action_rewards.len() != actions {
                    return Err(PolicyPortfolioError::RewardActionCountMismatch {
                        time, state, expected: actions, actual: action_rewards.len(),
                    });
                }
            }
        }

        let expected_transition_layers = rewards.len()
            .checked_sub(1)
            .ok_or(PolicyPortfolioError::ZeroHorizon)?;
        let actual_reward_layers = transitions.len()
            .checked_add(1)
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if actual_reward_layers != rewards.len() {
            return Err(PolicyPortfolioError::TransitionHorizonMismatch {
                expected: expected_transition_layers,
                actual: transitions.len(),
            });
        }
        for (time, layer) in transitions.iter().enumerate() {
            if layer.len() != states {
                return Err(PolicyPortfolioError::TransitionStateCountMismatch {
                    time, expected: states, actual: layer.len(),
                });
            }
            for (state, action_rows) in layer.iter().enumerate() {
                if action_rows.len() != actions {
                    return Err(PolicyPortfolioError::TransitionActionCountMismatch {
                        time, state, expected: actions, actual: action_rows.len(),
                    });
                }
                for (action, row) in action_rows.iter().enumerate() {
                    if row.len() != states {
                        return Err(PolicyPortfolioError::TransitionDestinationCountMismatch {
                            time, state, action, expected: states, actual: row.len(),
                        });
                    }
                    for (destination, probability) in row.iter().enumerate() {
                        if probability.is_negative()
                            || compare_signed(*probability, CheckedRational::one()) == Ordering::Greater
                        {
                            return Err(PolicyPortfolioError::InvalidTransitionProbability {
                                time, state, action, destination,
                            });
                        }
                    }
                    if checked_sum(row)? != CheckedRational::one() {
                        return Err(PolicyPortfolioError::TransitionRowDoesNotSumToOne {
                            time, state, action,
                        });
                    }
                }
            }
        }
        Ok(Self { initial, rewards, transitions })
    }

    pub fn state_count(&self) -> usize { self.initial.len() }
    pub fn action_count(&self) -> usize { self.rewards[0][0].len() }
    pub fn horizon(&self) -> usize { self.rewards.len() }

    pub fn sha256(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"SYMPGM-FINITE-UMDP-MODEL\0\x01");
        hasher.update((self.state_count() as u64).to_le_bytes());
        hasher.update((self.action_count() as u64).to_le_bytes());
        hasher.update((self.horizon() as u64).to_le_bytes());
        for value in &self.initial { hash_rational(&mut hasher, *value); }
        for layer in &self.rewards {
            for state in layer {
                for reward in state { hash_rational(&mut hasher, *reward); }
            }
        }
        for layer in &self.transitions {
            for state in layer {
                for action in state {
                    for probability in action { hash_rational(&mut hasher, *probability); }
                }
            }
        }
        hasher.finalize().into()
    }
}

/// An ordered, explicitly enumerated finite family of environments.
/// Model ordering is part of the family identity. All models share one initial
/// distribution, state space, action space and finite horizon.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteUmdpModelFamily {
    models: Vec<FiniteHorizonUmdpModel>,
    model_digests: Vec<[u8; 32]>,
    digest: [u8; 32],
}

impl FiniteUmdpModelFamily {
    pub fn new(models: Vec<FiniteHorizonUmdpModel>) -> Result<Self, PolicyPortfolioError> {
        if models.is_empty() {
            return Err(PolicyPortfolioError::EmptyModelFamily);
        }
        let first = &models[0];
        for (index, model) in models.iter().enumerate().skip(1) {
            if model.state_count() != first.state_count()
                || model.action_count() != first.action_count()
                || model.horizon() != first.horizon()
            {
                return Err(PolicyPortfolioError::InconsistentModelDimensions { model: index });
            }
            if model.initial != first.initial {
                return Err(PolicyPortfolioError::InconsistentInitialDistribution { model: index });
            }
        }
        let mut model_digests = Vec::new();
        model_digests.try_reserve_exact(models.len()).map_err(|_| {
            PolicyPortfolioError::AllocationFailed { resource: "model digest list", entries: models.len() }
        })?;
        for (index, model) in models.iter().enumerate() {
            // Exact structural equality, not digest equality, defines duplicates.
            // The digest remains the compact external identity of each model.
            if models[..index].iter().any(|previous| previous == model) {
                return Err(PolicyPortfolioError::DuplicateModelIdentity);
            }
            model_digests.push(model.sha256());
        }
        let mut hasher = Sha256::new();
        hasher.update(b"SYMPGM-FINITE-UMDP-FAMILY\0\x01");
        hasher.update((models.len() as u64).to_le_bytes());
        for digest in &model_digests { hasher.update(digest); }
        let digest = hasher.finalize().into();
        Ok(Self { models, model_digests, digest })
    }

    pub fn models(&self) -> &[FiniteHorizonUmdpModel] { &self.models }
    pub fn model_digests(&self) -> &[[u8; 32]] { &self.model_digests }
    pub fn model_count(&self) -> usize { self.models.len() }
    pub fn state_count(&self) -> usize { self.models[0].state_count() }
    pub fn action_count(&self) -> usize { self.models[0].action_count() }
    pub fn horizon(&self) -> usize { self.models[0].horizon() }
    pub fn sha256(&self) -> [u8; 32] { self.digest }
}

/// Hard limits for policy enumeration, value evaluation, portfolio search and
/// complete tied-result storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyPortfolioBudget {
    pub max_policies: usize,
    pub max_model_policy_evaluations: usize,
    pub max_model_policy_value_entries: usize,
    pub max_dp_multiply_adds: usize,
    pub max_portfolio_candidates: usize,
    /// Total model-by-portfolio-member comparisons across all enumerated subsets.
    pub max_portfolio_member_comparisons: usize,
    pub max_optimal_portfolios: usize,
    pub max_stored_policy_indices: usize,
    pub max_stored_model_witnesses: usize,
    pub max_output_memory_bytes: usize,
    pub max_rational_bit_width: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortfolioModelWitness {
    pub model_index: usize,
    pub model_sha256: [u8; 32],
    pub model_optimal_value: CheckedRational,
    pub portfolio_value: CheckedRational,
    pub regret: CheckedRational,
    /// All members of the portfolio attaining its best value on this model.
    pub best_policy_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimalPolicyPortfolio {
    /// Canonical deterministic time-dependent Markov policy indices.
    pub policy_indices: Vec<usize>,
    pub model_witnesses: Vec<PortfolioModelWitness>,
    pub worst_case_model_indices: Vec<usize>,
}

/// Decoded deterministic time-dependent Markov policy.
///
/// Actions are stored in time-major, then state-major order. Index zero is
/// the first time/state slot; the policy index is the lexicographic rank of
/// the action vector when each action is a base-|A| digit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeterministicMarkovPolicy {
    pub index: usize,
    pub horizon: usize,
    pub state_count: usize,
    pub action_count: usize,
    pub actions: Vec<usize>,
}

impl DeterministicMarkovPolicy {
    pub fn action_at(&self, time: usize, state: usize) -> Option<usize> {
        if time >= self.horizon || state >= self.state_count {
            return None;
        }
        self.actions.get(time.checked_mul(self.state_count)?.checked_add(state)?).copied()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteFamilyPolicyPortfolioResult {
    pub model_family_sha256: [u8; 32],
    /// Versioned identity of the policy class and canonical policy-index encoding.
    pub policy_manifest_sha256: [u8; 32],
    pub policy_values_sha256: [u8; 32],
    pub decision_certificate_sha256: [u8; 32],
    pub discount: CheckedRational,
    pub horizon: usize,
    pub state_count: usize,
    pub action_count: usize,
    pub policy_count: usize,
    pub max_portfolio_size: usize,
    pub model_policy_evaluations: usize,
    pub portfolio_candidates_evaluated: usize,
    pub model_optimal_values: Vec<CheckedRational>,
    /// Model-major, canonical-policy-index order; retained for exact replay.
    pub model_policy_values: Vec<Vec<CheckedRational>>,
    pub minimax_regret: CheckedRational,
    pub optimal_portfolios: Vec<OptimalPolicyPortfolio>,
    pub enumeration_complete: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyPortfolioError {
    EmptyModelFamily,
    EmptyStateSpace,
    EmptyActionSpace,
    ZeroHorizon,
    InvalidDiscount,
    InvalidInitialProbability { state: usize },
    InitialDistributionDoesNotSumToOne,
    RewardStateCountMismatch { time: usize, expected: usize, actual: usize },
    RewardActionCountMismatch { time: usize, state: usize, expected: usize, actual: usize },
    TransitionHorizonMismatch { expected: usize, actual: usize },
    TransitionStateCountMismatch { time: usize, expected: usize, actual: usize },
    TransitionActionCountMismatch { time: usize, state: usize, expected: usize, actual: usize },
    TransitionDestinationCountMismatch { time: usize, state: usize, action: usize, expected: usize, actual: usize },
    InvalidTransitionProbability { time: usize, state: usize, action: usize, destination: usize },
    TransitionRowDoesNotSumToOne { time: usize, state: usize, action: usize },
    InconsistentModelDimensions { model: usize },
    InconsistentInitialDistribution { model: usize },
    DuplicateModelIdentity,
    PolicySpaceOverflow,
    InvalidPolicyIndex { index: usize, policy_count: usize },
    DimensionOverflow,
    BudgetExceeded { resource: &'static str, required: usize, limit: usize },
    AllocationFailed { resource: &'static str, entries: usize },
    ArithmeticOverflow,
    NegativeRegret { model: usize },
    EmptyPortfolio,
    PortfolioCertificateMismatch,
    PolicyMaskHorizonMismatch { expected: usize, actual: usize },
    PolicyMaskStateCountMismatch { time: usize, expected: usize, actual: usize },
    PolicyMaskActionCountMismatch { time: usize, state: usize, expected: usize, actual: usize },
    PolicyMaskEmptyActionSet { time: usize, state: usize },
    PolicyMaskSlotIndexOutOfBounds { time: usize, state: usize, horizon: usize, state_count: usize },
    PolicyMaskActionIndexOutOfBounds { time: usize, state: usize, action: usize, action_count: usize },
    PolicyMaskActionNotAllowed { time: usize, state: usize, action: usize },
    PolicyMaskCannotSplit { time: usize, state: usize },
    EmptyModelCluster,
    PolicyMaskModelIndexOutOfBounds { model_index: usize, model_count: usize },
    DuplicatePolicyMaskModelIndex { model_index: usize },
    PolicyMaskModelIndicesNotIncreasing { previous: usize, current: usize },
    PolicyMaskFamilyDigestMismatch,
    PolicyMaskCertificateMismatch,
}

impl fmt::Display for PolicyPortfolioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyModelFamily => write!(f, "finite UMDP model family is empty"),
            Self::EmptyStateSpace => write!(f, "UMDP state space is empty"),
            Self::EmptyActionSpace => write!(f, "UMDP action space is empty"),
            Self::ZeroHorizon => write!(f, "finite UMDP horizon must be positive"),
            Self::InvalidDiscount => write!(f, "discount must be a rational in [0,1]"),
            Self::InvalidInitialProbability { state } => write!(f, "invalid initial probability at state {state}"),
            Self::InitialDistributionDoesNotSumToOne => write!(f, "initial distribution must sum exactly to one"),
            Self::RewardStateCountMismatch { time, expected, actual } => write!(f, "reward layer {time} has {actual} states; expected {expected}"),
            Self::RewardActionCountMismatch { time, state, expected, actual } => write!(f, "reward layer {time}, state {state} has {actual} actions; expected {expected}"),
            Self::TransitionHorizonMismatch { expected, actual } => write!(f, "transition horizon is {actual}; expected {expected}"),
            Self::TransitionStateCountMismatch { time, expected, actual } => write!(f, "transition layer {time} has {actual} states; expected {expected}"),
            Self::TransitionActionCountMismatch { time, state, expected, actual } => write!(f, "transition layer {time}, state {state} has {actual} actions; expected {expected}"),
            Self::TransitionDestinationCountMismatch { time, state, action, expected, actual } => write!(f, "transition row ({time},{state},{action}) has {actual} destinations; expected {expected}"),
            Self::InvalidTransitionProbability { time, state, action, destination } => write!(f, "invalid transition probability ({time},{state},{action},{destination})"),
            Self::TransitionRowDoesNotSumToOne { time, state, action } => write!(f, "transition row ({time},{state},{action}) does not sum to one"),
            Self::InconsistentModelDimensions { model } => write!(f, "model {model} has inconsistent dimensions"),
            Self::InconsistentInitialDistribution { model } => write!(f, "model {model} has a different initial distribution"),
            Self::DuplicateModelIdentity => write!(f, "duplicate environment identity in finite family"),
            Self::PolicySpaceOverflow => write!(f, "deterministic Markov policy space overflowed usize"),
            Self::InvalidPolicyIndex { index, policy_count } => write!(f, "policy index {index} is outside 0..{policy_count}"),
            Self::DimensionOverflow => write!(f, "policy portfolio dimension/count overflowed"),
            Self::BudgetExceeded { resource, required, limit } => write!(f, "policy portfolio {resource} requires {required}; configured limit is {limit}"),
            Self::AllocationFailed { resource, entries } => write!(f, "unable to reserve {resource} for {entries} entries"),
            Self::ArithmeticOverflow => write!(f, "checked rational overflow in policy portfolio evaluation"),
            Self::NegativeRegret { model } => write!(f, "portfolio value exceeds model-optimal value for model {model}"),
            Self::EmptyPortfolio => write!(f, "portfolio must contain at least one policy"),
            Self::PortfolioCertificateMismatch => write!(f, "portfolio replay differs from the claimed result"),
            Self::PolicyMaskHorizonMismatch { expected, actual } => write!(f, "policy action mask has horizon {actual}; expected {expected}"),
            Self::PolicyMaskStateCountMismatch { time, expected, actual } => write!(f, "policy action mask at time {time} has {actual} states; expected {expected}"),
            Self::PolicyMaskActionCountMismatch { time, state, expected, actual } => write!(f, "policy action mask ({time},{state}) has {actual} actions; expected {expected}"),
            Self::PolicyMaskEmptyActionSet { time, state } => write!(f, "policy action mask permits no action at ({time},{state})"),
            Self::PolicyMaskSlotIndexOutOfBounds { time, state, horizon, state_count } => write!(f, "policy action-mask slot ({time},{state}) is outside horizon {horizon} and state count {state_count}"),
            Self::PolicyMaskActionIndexOutOfBounds { time, state, action, action_count } => write!(f, "policy action-mask action {action} at ({time},{state}) is outside 0..{action_count}"),
            Self::PolicyMaskActionNotAllowed { time, state, action } => write!(f, "cannot branch on disallowed action {action} at ({time},{state})"),
            Self::PolicyMaskCannotSplit { time, state } => write!(f, "action mask at ({time},{state}) has no nonempty force/forbid split"),
            Self::EmptyModelCluster => write!(f, "policy-mask cluster must contain at least one model"),
            Self::PolicyMaskModelIndexOutOfBounds { model_index, model_count } => write!(f, "cluster model index {model_index} is outside 0..{model_count}"),
            Self::DuplicatePolicyMaskModelIndex { model_index } => write!(f, "cluster repeats model index {model_index}"),
            Self::PolicyMaskModelIndicesNotIncreasing { previous, current } => write!(f, "cluster model indices must be strictly increasing: {previous} then {current}"),
            Self::PolicyMaskFamilyDigestMismatch => write!(f, "policy action mask belongs to a different model family"),
            Self::PolicyMaskCertificateMismatch => write!(f, "policy action-mask bound replay differs from the claimed result"),
        }
    }
}
impl Error for PolicyPortfolioError {}

fn add(a: CheckedRational, b: CheckedRational) -> Result<CheckedRational, PolicyPortfolioError> {
    a.checked_add(b).map_err(|_| PolicyPortfolioError::ArithmeticOverflow)
}
fn mul(a: CheckedRational, b: CheckedRational) -> Result<CheckedRational, PolicyPortfolioError> {
    a.checked_mul(b).map_err(|_| PolicyPortfolioError::ArithmeticOverflow)
}
fn neg(a: CheckedRational) -> Result<CheckedRational, PolicyPortfolioError> {
    let n = a.numerator().checked_neg().ok_or(PolicyPortfolioError::ArithmeticOverflow)?;
    CheckedRational::new(n, a.denominator()).map_err(|_| PolicyPortfolioError::ArithmeticOverflow)
}
fn sub(a: CheckedRational, b: CheckedRational) -> Result<CheckedRational, PolicyPortfolioError> {
    add(a, neg(b)?)
}
fn sum(values: &[CheckedRational]) -> Result<CheckedRational, PolicyPortfolioError> {
    values.iter().try_fold(CheckedRational::zero(), |acc, value| add(acc, *value))
}

fn bit_width(value: CheckedRational) -> usize {
    let n = value.numerator().unsigned_abs();
    let d = value.denominator() as u128;
    let nb = if n == 0 { 0 } else { (u128::BITS - n.leading_zeros()) as usize };
    let db = (u128::BITS - d.leading_zeros()) as usize;
    nb.max(db)
}

/// Enforce the configured rational-width limit on every generated intermediate,
/// not only on the source model inputs.
fn check_rational_width(
    value: CheckedRational,
    maximum: usize,
) -> Result<CheckedRational, PolicyPortfolioError> {
    let required = bit_width(value);
    if required > maximum {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "rational bit width",
            required,
            limit: maximum,
        });
    }
    Ok(value)
}

fn add_bounded(
    left: CheckedRational,
    right: CheckedRational,
    maximum: usize,
) -> Result<CheckedRational, PolicyPortfolioError> {
    check_rational_width(add(left, right)?, maximum)
}

fn mul_bounded(
    left: CheckedRational,
    right: CheckedRational,
    maximum: usize,
) -> Result<CheckedRational, PolicyPortfolioError> {
    check_rational_width(mul(left, right)?, maximum)
}

fn sub_bounded(
    left: CheckedRational,
    right: CheckedRational,
    maximum: usize,
) -> Result<CheckedRational, PolicyPortfolioError> {
    check_rational_width(sub(left, right)?, maximum)
}

fn compare_nonnegative(mut an: u128, mut ad: u128, mut bn: u128, mut bd: u128) -> Ordering {
    let mut reverse = false;
    loop {
        let ai = an / ad;
        let bi = bn / bd;
        if ai != bi {
            let order = ai.cmp(&bi);
            return if reverse { order.reverse() } else { order };
        }
        let ar = an % ad;
        let br = bn % bd;
        match (ar == 0, br == 0) {
            (true, true) => return Ordering::Equal,
            (true, false) => return if reverse { Ordering::Greater } else { Ordering::Less },
            (false, true) => return if reverse { Ordering::Less } else { Ordering::Greater },
            (false, false) => {
                an = ad; ad = ar; bn = bd; bd = br; reverse = !reverse;
            }
        }
    }
}

fn compare_signed(a: CheckedRational, b: CheckedRational) -> Ordering {
    match (a.is_negative(), b.is_negative()) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => compare_nonnegative(
            a.numerator() as u128, a.denominator() as u128,
            b.numerator() as u128, b.denominator() as u128,
        ),
        (true, true) => compare_nonnegative(
            a.numerator().unsigned_abs(), a.denominator() as u128,
            b.numerator().unsigned_abs(), b.denominator() as u128,
        ).reverse(),
    }
}

fn checked_policy_count(states: usize, horizon: usize, actions: usize) -> Result<usize, PolicyPortfolioError> {
    let exponent = states.checked_mul(horizon).ok_or(PolicyPortfolioError::PolicySpaceOverflow)?;
    // Exponentiation by squaring avoids a linear-time loop for large dimensions
    // (notably the one-action policy space, whose cardinality is always one).
    let mut power = exponent;
    let mut factor = actions;
    let mut count = 1usize;
    while power > 0 {
        if power & 1 == 1 {
            count = count.checked_mul(factor).ok_or(PolicyPortfolioError::PolicySpaceOverflow)?;
        }
        power >>= 1;
        if power > 0 {
            factor = factor.checked_mul(factor).ok_or(PolicyPortfolioError::PolicySpaceOverflow)?;
        }
    }
    Ok(count)
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

fn binomial(n: usize, k: usize) -> Result<usize, PolicyPortfolioError> {
    if k > n { return Ok(0); }
    let k = k.min(n - k);
    let mut value = 1u128;
    for i in 1..=k {
        let mut numerator = (n - k + i) as u128;
        let mut denominator = i as u128;
        let first_gcd = gcd_u128(numerator, denominator);
        numerator /= first_gcd;
        denominator /= first_gcd;
        let second_gcd = gcd_u128(value, denominator);
        value /= second_gcd;
        denominator /= second_gcd;
        if denominator != 1 {
            return Err(PolicyPortfolioError::DimensionOverflow);
        }
        value = value.checked_mul(numerator).ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if value > usize::MAX as u128 {
            return Err(PolicyPortfolioError::DimensionOverflow);
        }
    }
    Ok(value as usize)
}

fn portfolio_count(policies: usize, k: usize) -> Result<usize, PolicyPortfolioError> {
    let mut total = 0usize;
    for size in 1..=k.min(policies) {
        total = total.checked_add(binomial(policies, size)?)
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    }
    Ok(total)
}

fn decode_policy(index: usize, horizon: usize, states: usize, actions: usize, output: &mut [usize]) {
    let mut remaining = index;
    for slot in (0..(horizon * states)).rev() {
        output[slot] = remaining % actions;
        remaining /= actions;
    }
}

/// Decode the canonical policy index used by the exhaustive solver.
pub fn decode_deterministic_markov_policy(
    index: usize,
    horizon: usize,
    state_count: usize,
    action_count: usize,
) -> Result<DeterministicMarkovPolicy, PolicyPortfolioError> {
    if horizon == 0 {
        return Err(PolicyPortfolioError::ZeroHorizon);
    }
    if state_count == 0 {
        return Err(PolicyPortfolioError::EmptyStateSpace);
    }
    if action_count == 0 {
        return Err(PolicyPortfolioError::EmptyActionSpace);
    }
    let policy_count = checked_policy_count(state_count, horizon, action_count)?;
    if index >= policy_count {
        return Err(PolicyPortfolioError::InvalidPolicyIndex { index, policy_count });
    }
    let entries = horizon.checked_mul(state_count)
        .ok_or(PolicyPortfolioError::PolicySpaceOverflow)?;
    let mut actions = Vec::new();
    actions.try_reserve_exact(entries).map_err(|_| {
        PolicyPortfolioError::AllocationFailed { resource: "decoded policy actions", entries }
    })?;
    actions.resize(entries, 0usize);
    decode_policy(index, horizon, state_count, action_count, &mut actions);
    Ok(DeterministicMarkovPolicy {
        index,
        horizon,
        state_count,
        action_count,
        actions,
    })
}

fn evaluate_policy(
    model: &FiniteHorizonUmdpModel,
    discount: CheckedRational,
    policy: &[usize],
    current: &mut [CheckedRational],
    next: &mut [CheckedRational],
    max_rational_bit_width: usize,
) -> Result<CheckedRational, PolicyPortfolioError> {
    let states = model.state_count();
    let horizon = model.horizon();
    next.fill(CheckedRational::zero());
    current.fill(CheckedRational::zero());
    for time in (0..horizon).rev() {
        for state in 0..states {
            let action = policy[time * states + state];
            let mut continuation = CheckedRational::zero();
            if time + 1 < horizon {
                for destination in 0..states {
                    continuation = add_bounded(
                        continuation,
                        mul_bounded(model.transitions[time][state][action][destination], next[destination], max_rational_bit_width)?,
                        max_rational_bit_width,
                    )?;
                }
                continuation = mul_bounded(discount, continuation, max_rational_bit_width)?;
            }
            current[state] = add_bounded(model.rewards[time][state][action], continuation, max_rational_bit_width)?;
        }
        current.swap_with_slice(next);
    }
    let mut initial_value = CheckedRational::zero();
    for state in 0..states {
        initial_value = add_bounded(initial_value, mul_bounded(model.initial[state], next[state], max_rational_bit_width)?, max_rational_bit_width)?;
    }
    Ok(initial_value)
}

fn optimal_value(
    model: &FiniteHorizonUmdpModel,
    discount: CheckedRational,
    current: &mut [CheckedRational],
    next: &mut [CheckedRational],
    max_rational_bit_width: usize,
) -> Result<CheckedRational, PolicyPortfolioError> {
    let states = model.state_count();
    let actions = model.action_count();
    let horizon = model.horizon();
    next.fill(CheckedRational::zero());
    current.fill(CheckedRational::zero());
    for time in (0..horizon).rev() {
        for state in 0..states {
            let mut best: Option<CheckedRational> = None;
            for action in 0..actions {
                let mut continuation = CheckedRational::zero();
                if time + 1 < horizon {
                    for destination in 0..states {
                        continuation = add_bounded(
                            continuation,
                            mul_bounded(model.transitions[time][state][action][destination], next[destination], max_rational_bit_width)?,
                            max_rational_bit_width,
                        )?;
                    }
                    continuation = mul_bounded(discount, continuation, max_rational_bit_width)?;
                }
                let candidate = add_bounded(model.rewards[time][state][action], continuation, max_rational_bit_width)?;
                if best.map_or(true, |value| compare_signed(candidate, value) == Ordering::Greater) {
                    best = Some(candidate);
                }
            }
            current[state] = best.ok_or(PolicyPortfolioError::EmptyActionSpace)?;
        }
        current.swap_with_slice(next);
    }
    let mut initial_value = CheckedRational::zero();
    for state in 0..states {
        initial_value = add_bounded(initial_value, mul_bounded(model.initial[state], next[state], max_rational_bit_width)?, max_rational_bit_width)?;
    }
    Ok(initial_value)
}

/// Solve the finite-family minimax-regret portfolio problem by exact exhaustive
/// enumeration. Policies are deterministic and time-dependent Markov policies;
/// models are fixed and enumerated, not sampled or independently resampled per
/// policy/time step.
pub fn solve_finite_family_policy_portfolio(
    family: &FiniteUmdpModelFamily,
    discount: CheckedRational,
    max_portfolio_size: usize,
    budget: PolicyPortfolioBudget,
) -> Result<FiniteFamilyPolicyPortfolioResult, PolicyPortfolioError> {
    if discount.is_negative()
        || compare_signed(discount, CheckedRational::one()) == Ordering::Greater
    {
        return Err(PolicyPortfolioError::InvalidDiscount);
    }
    let models = family.model_count();
    let states = family.state_count();
    let actions = family.action_count();
    let horizon = family.horizon();
    let policy_count = checked_policy_count(states, horizon, actions)?;
    if policy_count > budget.max_policies {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "policies", required: policy_count, limit: budget.max_policies,
        });
    }
    let k = max_portfolio_size.min(policy_count);
    if k == 0 { return Err(PolicyPortfolioError::EmptyPortfolio); }
    let candidate_count = portfolio_count(policy_count, k)?;
    if candidate_count > budget.max_portfolio_candidates {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "portfolio candidates", required: candidate_count, limit: budget.max_portfolio_candidates,
        });
    }
    let mut portfolio_member_comparisons = 0usize;
    for size in 1..=k {
        let combinations = binomial(policy_count, size)?;
        let comparisons = models.checked_mul(size)
            .and_then(|value| value.checked_mul(combinations))
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        portfolio_member_comparisons = portfolio_member_comparisons.checked_add(comparisons)
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    }
    if portfolio_member_comparisons > budget.max_portfolio_member_comparisons {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "portfolio member comparisons",
            required: portfolio_member_comparisons,
            limit: budget.max_portfolio_member_comparisons,
        });
    }

    for model in family.models() {
        for value in model.initial.iter()
            .chain(model.rewards.iter().flatten().flatten())
            .chain(model.transitions.iter().flatten().flatten().flatten())
        {
            let required = bit_width(*value);
            if required > budget.max_rational_bit_width {
                return Err(PolicyPortfolioError::BudgetExceeded {
                    resource: "rational bit width", required, limit: budget.max_rational_bit_width,
                });
            }
        }
    }
    let discount_bits = bit_width(discount);
    if discount_bits > budget.max_rational_bit_width {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "rational bit width", required: discount_bits, limit: budget.max_rational_bit_width,
        });
    }

    let evaluations = models.checked_mul(policy_count).ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if evaluations > budget.max_model_policy_evaluations {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "model-policy evaluations", required: evaluations, limit: budget.max_model_policy_evaluations,
        });
    }
    if evaluations > budget.max_model_policy_value_entries {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "model-policy value entries", required: evaluations, limit: budget.max_model_policy_value_entries,
        });
    }

    let transition_stages = horizon.saturating_sub(1);
    // Exact upper counts of rational multiply/add operations in each policy
    // replay: transition dot products, discounting, reward addition, and the
    // initial-distribution dot product.
    let policy_transition_ops = transition_stages.checked_mul(states)
        .and_then(|value| value.checked_mul(states))
        .and_then(|value| value.checked_mul(2))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let policy_step_ops = horizon.checked_mul(states)
        .and_then(|value| value.checked_add(transition_stages.checked_mul(states)?))
        .and_then(|value| value.checked_add(states.checked_mul(2)?))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let policy_ops = policy_transition_ops.checked_add(policy_step_ops)
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;

    // Bellman optimum replay considers every action at every (time,state)
    // pair; max comparisons are additional work but not rational multiply/adds.
    let optimal_transition_ops = transition_stages.checked_mul(states)
        .and_then(|value| value.checked_mul(actions))
        .and_then(|value| value.checked_mul(states))
        .and_then(|value| value.checked_mul(2))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let optimal_step_ops = horizon.checked_mul(states)
        .and_then(|value| value.checked_mul(actions))
        .and_then(|value| value.checked_add(transition_stages.checked_mul(states)?.checked_mul(actions)?))
        .and_then(|value| value.checked_add(states.checked_mul(2)?))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let optimal_ops = optimal_transition_ops.checked_add(optimal_step_ops)
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let total_ops = evaluations.checked_mul(policy_ops)
        .and_then(|value| value.checked_add(models.checked_mul(optimal_ops)?))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if total_ops > budget.max_dp_multiply_adds {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "DP multiply-add work", required: total_ops, limit: budget.max_dp_multiply_adds,
        });
    }

    let value_bytes = evaluations.checked_mul(std::mem::size_of::<CheckedRational>())
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let policy_entries = horizon.checked_mul(states).ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let policy_buffer_bytes = policy_entries.checked_mul(std::mem::size_of::<usize>())
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<usize>>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let dp_buffer_bytes = states.checked_mul(2)
        .and_then(|value| value.checked_mul(std::mem::size_of::<CheckedRational>()))
        .and_then(|value| value.checked_add(2 * std::mem::size_of::<Vec<CheckedRational>>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let combination_buffer_bytes = k.checked_mul(std::mem::size_of::<usize>())
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<usize>>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let model_value_headers = models.checked_mul(std::mem::size_of::<Vec<CheckedRational>>())
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<Vec<CheckedRational>>>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let model_optimum_bytes = models.checked_mul(std::mem::size_of::<CheckedRational>())
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<CheckedRational>>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    // Peak temporary candidate state: one witness per model, its tied-best
    // policy-index vector (capacity k), and the worst-case model-index vector.
    let candidate_witness_bytes = models.checked_mul(std::mem::size_of::<PortfolioModelWitness>())
        .and_then(|value| value.checked_add(models.checked_mul(k.checked_mul(std::mem::size_of::<usize>())?)?))
        .and_then(|value| value.checked_add(models.checked_mul(std::mem::size_of::<usize>())?))
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<PortfolioModelWitness>>()))
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<usize>>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let base_peak_bytes = value_bytes
        .checked_add(model_value_headers)
        .and_then(|value| value.checked_add(model_optimum_bytes))
        .and_then(|value| value.checked_add(policy_buffer_bytes))
        .and_then(|value| value.checked_add(dp_buffer_bytes))
        .and_then(|value| value.checked_add(combination_buffer_bytes))
        .and_then(|value| value.checked_add(candidate_witness_bytes))
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<OptimalPolicyPortfolio>>()))
        .and_then(|value| value.checked_add(std::mem::size_of::<FiniteFamilyPolicyPortfolioResult>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if base_peak_bytes > budget.max_output_memory_bytes {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "estimated peak memory bytes",
            required: base_peak_bytes,
            limit: budget.max_output_memory_bytes,
        });
    }

    let mut policy = Vec::new();
    policy.try_reserve_exact(policy_entries).map_err(|_| PolicyPortfolioError::AllocationFailed {
        resource: "policy action buffer", entries: policy_entries,
    })?;
    policy.resize(policy_entries, 0usize);

    let mut policy_values: Vec<Vec<CheckedRational>> = Vec::new();
    policy_values.try_reserve_exact(models).map_err(|_| PolicyPortfolioError::AllocationFailed {
        resource: "model policy-value rows", entries: models,
    })?;
    let mut model_optima = Vec::new();
    model_optima.try_reserve_exact(models).map_err(|_| PolicyPortfolioError::AllocationFailed {
        resource: "model optimal values", entries: models,
    })?;

    for model in family.models() {
        let mut row = Vec::new();
        row.try_reserve_exact(policy_count).map_err(|_| PolicyPortfolioError::AllocationFailed {
            resource: "policy values for one model", entries: policy_count,
        })?;
        let mut current = Vec::new();
        current.try_reserve_exact(states).map_err(|_| PolicyPortfolioError::AllocationFailed {
            resource: "DP current vector", entries: states,
        })?;
        current.resize(states, CheckedRational::zero());
        let mut next = Vec::new();
        next.try_reserve_exact(states).map_err(|_| PolicyPortfolioError::AllocationFailed {
            resource: "DP next vector", entries: states,
        })?;
        next.resize(states, CheckedRational::zero());

        model_optima.push(optimal_value(model, discount, &mut current, &mut next, budget.max_rational_bit_width)?);
        for policy_index in 0..policy_count {
            decode_policy(policy_index, horizon, states, actions, &mut policy);
            row.push(evaluate_policy(model, discount, &policy, &mut current, &mut next, budget.max_rational_bit_width)?);
        }
        policy_values.push(row);
    }

    let policy_manifest_sha256 = policy_manifest_sha256(states, actions, horizon, policy_count);
    let policy_values_sha256 = hash_policy_values(
        family.sha256(),
        policy_manifest_sha256,
        discount,
        states,
        actions,
        horizon,
        &model_optima,
        &policy_values,
    );
    let mut incumbent: Option<CheckedRational> = None;
    let mut optimal_portfolios = Vec::new();
    let mut stored_policy_indices = 0usize;
    let mut stored_witnesses = 0usize;
    let mut output_bytes = base_peak_bytes;
    let mut candidates_evaluated = 0usize;

    for size in 1..=k {
        let mut combination = Vec::new();
        combination.try_reserve_exact(size).map_err(|_| PolicyPortfolioError::AllocationFailed {
            resource: "portfolio combination", entries: size,
        })?;
        combination.extend(0..size);

        loop {
            candidates_evaluated = candidates_evaluated.checked_add(1)
                .ok_or(PolicyPortfolioError::DimensionOverflow)?;
            let mut witnesses = Vec::new();
            witnesses.try_reserve_exact(models).map_err(|_| PolicyPortfolioError::AllocationFailed {
                resource: "portfolio model witnesses", entries: models,
            })?;
            let mut worst_regret = CheckedRational::zero();
            let mut worst_models = Vec::new();
            worst_models.try_reserve_exact(models).map_err(|_| PolicyPortfolioError::AllocationFailed {
                resource: "worst-case model indices", entries: models,
            })?;

            for model_index in 0..models {
                let mut portfolio_value: Option<CheckedRational> = None;
                let mut best_members = Vec::new();
                best_members.try_reserve_exact(size).map_err(|_| PolicyPortfolioError::AllocationFailed {
                    resource: "best portfolio members", entries: size,
                })?;
                for &policy_index in &combination {
                    let candidate = policy_values[model_index][policy_index];
                    match portfolio_value {
                        None => { portfolio_value = Some(candidate); best_members.push(policy_index); }
                        Some(best) => match compare_signed(candidate, best) {
                            Ordering::Greater => { portfolio_value = Some(candidate); best_members.clear(); best_members.push(policy_index); }
                            Ordering::Equal => best_members.push(policy_index),
                            Ordering::Less => {}
                        },
                    }
                }
                let portfolio_value = portfolio_value.ok_or(PolicyPortfolioError::EmptyPortfolio)?;
                let regret = sub_bounded(model_optima[model_index], portfolio_value, budget.max_rational_bit_width)?;
                if regret.is_negative() {
                    return Err(PolicyPortfolioError::NegativeRegret { model: model_index });
                }
                if compare_signed(regret, worst_regret) == Ordering::Greater {
                    worst_regret = regret;
                    worst_models.clear();
                    worst_models.push(model_index);
                } else if compare_signed(regret, worst_regret) == Ordering::Equal {
                    worst_models.push(model_index);
                }
                witnesses.push(PortfolioModelWitness {
                    model_index,
                    model_sha256: family.model_digests[model_index],
                    model_optimal_value: model_optima[model_index],
                    portfolio_value,
                    regret,
                    best_policy_indices: best_members,
                });
            }

            let order = incumbent.map_or(Ordering::Less, |value| compare_signed(worst_regret, value));
            if order == Ordering::Less {
                incumbent = Some(worst_regret);
                // Drop the old tied-output allocation as well as its elements;
                // retaining Vec capacity would invalidate the peak-memory model.
                optimal_portfolios = Vec::new();
                stored_policy_indices = 0;
                stored_witnesses = 0;
                output_bytes = base_peak_bytes;
            }
            if order == Ordering::Less || order == Ordering::Equal {
                let new_portfolios = optimal_portfolios.len() + 1;
                if new_portfolios > budget.max_optimal_portfolios {
                    return Err(PolicyPortfolioError::BudgetExceeded {
                        resource: "tied optimal portfolios", required: new_portfolios, limit: budget.max_optimal_portfolios,
                    });
                }
                let model_best_indices = witnesses.iter().try_fold(0usize, |total, witness| {
                    total.checked_add(witness.best_policy_indices.len())
                        .ok_or(PolicyPortfolioError::DimensionOverflow)
                })?;
                let next_stored_indices = stored_policy_indices
                    .checked_add(size)
                    .and_then(|value| value.checked_add(model_best_indices))
                    .and_then(|value| value.checked_add(worst_models.len()))
                    .ok_or(PolicyPortfolioError::DimensionOverflow)?;
                if next_stored_indices > budget.max_stored_policy_indices {
                    return Err(PolicyPortfolioError::BudgetExceeded {
                        resource: "stored portfolio and witness indices",
                        required: next_stored_indices,
                        limit: budget.max_stored_policy_indices,
                    });
                }
                let new_witnesses = stored_witnesses.checked_add(models).ok_or(PolicyPortfolioError::DimensionOverflow)?;
                if new_witnesses > budget.max_stored_model_witnesses {
                    return Err(PolicyPortfolioError::BudgetExceeded {
                        resource: "stored model witnesses", required: new_witnesses, limit: budget.max_stored_model_witnesses,
                    });
                }
                let estimate = size.checked_mul(std::mem::size_of::<usize>())
                    .and_then(|value| value.checked_add(std::mem::size_of::<Vec<usize>>()))
                    .and_then(|value| value.checked_add(models.checked_mul(std::mem::size_of::<PortfolioModelWitness>())?))
                    .and_then(|value| value.checked_add(std::mem::size_of::<Vec<PortfolioModelWitness>>()))
                    .and_then(|value| value.checked_add(models.checked_mul(size.checked_mul(std::mem::size_of::<usize>())?)?))
                    .and_then(|value| value.checked_add(worst_models.len().checked_mul(std::mem::size_of::<usize>())?))
                    .and_then(|value| value.checked_add(std::mem::size_of::<Vec<usize>>()))
                    .and_then(|value| value.checked_add(std::mem::size_of::<OptimalPolicyPortfolio>()))
                    .ok_or(PolicyPortfolioError::DimensionOverflow)?;
                let new_bytes = output_bytes.checked_add(estimate).ok_or(PolicyPortfolioError::DimensionOverflow)?;
                if new_bytes > budget.max_output_memory_bytes {
                    return Err(PolicyPortfolioError::BudgetExceeded {
                        resource: "estimated peak memory bytes",
                        required: new_bytes,
                        limit: budget.max_output_memory_bytes,
                    });
                }

                let mut indices = Vec::new();
                indices.try_reserve_exact(size).map_err(|_| PolicyPortfolioError::AllocationFailed {
                    resource: "stored portfolio indices", entries: size,
                })?;
                indices.extend_from_slice(&combination);
                let mut stored_models = Vec::new();
                stored_models.try_reserve_exact(models).map_err(|_| PolicyPortfolioError::AllocationFailed {
                    resource: "stored portfolio model witnesses", entries: models,
                })?;
                stored_models.extend(witnesses);
                let mut stored_worst = Vec::new();
                stored_worst.try_reserve_exact(worst_models.len()).map_err(|_| PolicyPortfolioError::AllocationFailed {
                    resource: "stored worst-case model indices", entries: worst_models.len(),
                })?;
                stored_worst.extend_from_slice(&worst_models);
                optimal_portfolios.try_reserve(1).map_err(|_| PolicyPortfolioError::AllocationFailed {
                    resource: "optimal portfolio outputs", entries: 1,
                })?;
                optimal_portfolios.push(OptimalPolicyPortfolio {
                    policy_indices: indices,
                    model_witnesses: stored_models,
                    worst_case_model_indices: stored_worst,
                });
                stored_policy_indices = next_stored_indices;
                stored_witnesses = new_witnesses;
                output_bytes = new_bytes;
            }

            let mut cursor = size;
            while cursor > 0 && combination[cursor - 1] == policy_count - size + cursor - 1 {
                cursor -= 1;
            }
            if cursor == 0 { break; }
            combination[cursor - 1] += 1;
            for index in cursor..size {
                combination[index] = combination[index - 1] + 1;
            }
        }
    }

    let minimax_regret = incumbent.ok_or(PolicyPortfolioError::EmptyPortfolio)?;
    let decision_certificate_sha256 = hash_portfolio_decision(
        family.sha256(),
        policy_manifest_sha256,
        policy_values_sha256,
        discount,
        policy_count,
        k,
        evaluations,
        candidates_evaluated,
        minimax_regret,
        &optimal_portfolios,
    );

    Ok(FiniteFamilyPolicyPortfolioResult {
        model_family_sha256: family.sha256(),
        policy_manifest_sha256,
        policy_values_sha256,
        decision_certificate_sha256,
        discount,
        horizon,
        state_count: states,
        action_count: actions,
        policy_count,
        max_portfolio_size: k,
        model_policy_evaluations: evaluations,
        portfolio_candidates_evaluated: candidates_evaluated,
        model_optimal_values: model_optima,
        model_policy_values: policy_values,
        minimax_regret,
        optimal_portfolios,
        enumeration_complete: candidates_evaluated == candidate_count,
    })
}

/// Full exact recomputation of values, all bounded portfolios and the tie set.
/// This verifier certifies only the declared finite family and policy class.
pub fn verify_finite_family_policy_portfolio_result(
    family: &FiniteUmdpModelFamily,
    budget: PolicyPortfolioBudget,
    claimed: &FiniteFamilyPolicyPortfolioResult,
) -> Result<(), PolicyPortfolioError> {
    let replay = solve_finite_family_policy_portfolio(
        family, claimed.discount, claimed.max_portfolio_size, budget,
    )?;
    if replay == *claimed && claimed.enumeration_complete {
        Ok(())
    } else {
        Err(PolicyPortfolioError::PortfolioCertificateMismatch)
    }
}

fn policy_manifest_sha256(
    states: usize,
    actions: usize,
    horizon: usize,
    policy_count: usize,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"SYMPGM-FINITE-UMDP-POLICY-MANIFEST\0\x01");
    h.update(b"deterministic-time-dependent-markov;time-major-state-major;lexicographic-base-actions");
    for value in [states, actions, horizon, policy_count] {
        h.update((value as u64).to_le_bytes());
    }
    h.finalize().into()
}

fn hash_policy_values(
    family: [u8; 32],
    policy_manifest: [u8; 32],
    discount: CheckedRational,
    states: usize,
    actions: usize,
    horizon: usize,
    optima: &[CheckedRational],
    values: &[Vec<CheckedRational>],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"SYMPGM-FINITE-UMDP-POLICY-VALUES\0\x02");
    h.update(family);
    h.update(policy_manifest);
    hash_rational(&mut h, discount);
    for value in [states, actions, horizon] { h.update((value as u64).to_le_bytes()); }
    h.update((optima.len() as u64).to_le_bytes());
    for value in optima { hash_rational(&mut h, *value); }
    h.update((values.len() as u64).to_le_bytes());
    for row in values {
        h.update((row.len() as u64).to_le_bytes());
        for value in row { hash_rational(&mut h, *value); }
    }
    h.finalize().into()
}

fn hash_portfolio_decision(
    family: [u8; 32],
    policy_manifest: [u8; 32],
    values_digest: [u8; 32],
    discount: CheckedRational,
    policy_count: usize,
    k: usize,
    evaluations: usize,
    candidates: usize,
    regret: CheckedRational,
    portfolios: &[OptimalPolicyPortfolio],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"SYMPGM-FINITE-UMDP-PORTFOLIO-DECISION\0\x02");
    h.update(family);
    h.update(policy_manifest);
    h.update(values_digest);
    hash_rational(&mut h, discount);
    for value in [policy_count, k, evaluations, candidates] { h.update((value as u64).to_le_bytes()); }
    hash_rational(&mut h, regret);
    h.update((portfolios.len() as u64).to_le_bytes());
    for portfolio in portfolios {
        h.update((portfolio.policy_indices.len() as u64).to_le_bytes());
        for index in &portfolio.policy_indices { h.update((*index as u64).to_le_bytes()); }
        h.update((portfolio.model_witnesses.len() as u64).to_le_bytes());
        for witness in &portfolio.model_witnesses {
            h.update((witness.model_index as u64).to_le_bytes());
            h.update(witness.model_sha256);
            hash_rational(&mut h, witness.model_optimal_value);
            hash_rational(&mut h, witness.portfolio_value);
            hash_rational(&mut h, witness.regret);
            h.update((witness.best_policy_indices.len() as u64).to_le_bytes());
            for index in &witness.best_policy_indices { h.update((*index as u64).to_le_bytes()); }
        }
        h.update((portfolio.worst_case_model_indices.len() as u64).to_le_bytes());
        for index in &portfolio.worst_case_model_indices { h.update((*index as u64).to_le_bytes()); }
    }
    h.finalize().into()
}

fn hash_rational(hasher: &mut Sha256, value: CheckedRational) {
    hasher.update(value.numerator().to_le_bytes());
    hasher.update(value.denominator().to_le_bytes());
}

pub fn policy_portfolio_sha256_hex(digest: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}


/// Hard caps for exact action-mask relaxation and its source-bound lower-bound
/// witness. The DP operation estimate covers both unrestricted and masked
/// Bellman evaluation for every model in the cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyActionMaskBudget {
    pub max_model_count: usize,
    pub max_mask_entries: usize,
    pub max_dp_multiply_adds: usize,
    pub max_rational_bit_width: usize,
    pub max_output_memory_bytes: usize,
}

/// A nonempty allowed-action set for every (time,state) decision slot.
///
/// The mask is bound to one finite model family. Splitting a slot into
/// "force action a" and "forbid action a" produces disjoint children whose
/// union is exactly the parent policy set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteUmdpPolicyActionMask {
    family_sha256: [u8; 32],
    allowed: Vec<Vec<Vec<bool>>>,
    sha256: [u8; 32],
}

impl FiniteUmdpPolicyActionMask {
    pub fn new(
        family: &FiniteUmdpModelFamily,
        allowed: Vec<Vec<Vec<bool>>>,
    ) -> Result<Self, PolicyPortfolioError> {
        let horizon = family.horizon();
        let states = family.state_count();
        let actions = family.action_count();
        if allowed.len() != horizon {
            return Err(PolicyPortfolioError::PolicyMaskHorizonMismatch {
                expected: horizon,
                actual: allowed.len(),
            });
        }
        for (time, layer) in allowed.iter().enumerate() {
            if layer.len() != states {
                return Err(PolicyPortfolioError::PolicyMaskStateCountMismatch {
                    time,
                    expected: states,
                    actual: layer.len(),
                });
            }
            for (state, action_set) in layer.iter().enumerate() {
                if action_set.len() != actions {
                    return Err(PolicyPortfolioError::PolicyMaskActionCountMismatch {
                        time,
                        state,
                        expected: actions,
                        actual: action_set.len(),
                    });
                }
                if !action_set.iter().any(|allowed| *allowed) {
                    return Err(PolicyPortfolioError::PolicyMaskEmptyActionSet {
                        time,
                        state,
                    });
                }
            }
        }
        let family_sha256 = family.sha256();
        let sha256 = canonical_policy_action_mask_sha256(
            family_sha256,
            horizon,
            states,
            actions,
            &allowed,
        );
        Ok(Self { family_sha256, allowed, sha256 })
    }

    pub fn sha256(&self) -> [u8; 32] { self.sha256 }
    pub fn horizon(&self) -> usize { self.allowed.len() }
    pub fn state_count(&self) -> usize { self.allowed.first().map_or(0, Vec::len) }
    pub fn action_count(&self) -> usize {
        self.allowed.first().and_then(|layer| layer.first()).map_or(0, Vec::len)
    }

    pub fn allows(&self, time: usize, state: usize, action: usize) -> Option<bool> {
        self.allowed.get(time)
            .and_then(|layer| layer.get(state))
            .and_then(|row| row.get(action))
            .copied()
    }

    pub fn allowed_actions(&self, time: usize, state: usize) -> Option<&[bool]> {
        self.allowed.get(time).and_then(|layer| layer.get(state)).map(Vec::as_slice)
    }

    /// Partition the current policy set into two exhaustive, disjoint children:
    /// one forces the selected action and the other forbids it.
    pub fn split_on_action(
        &self,
        family: &FiniteUmdpModelFamily,
        time: usize,
        state: usize,
        action: usize,
        budget: PolicyActionMaskBudget,
    ) -> Result<(Self, Self), PolicyPortfolioError> {
        if self.family_sha256 != family.sha256() {
            return Err(PolicyPortfolioError::PolicyMaskFamilyDigestMismatch);
        }
        if time >= self.horizon() || state >= self.state_count() {
            return Err(PolicyPortfolioError::PolicyMaskSlotIndexOutOfBounds {
                time,
                state,
                horizon: self.horizon(),
                state_count: self.state_count(),
            });
        }
        if action >= self.action_count() {
            return Err(PolicyPortfolioError::PolicyMaskActionIndexOutOfBounds {
                time,
                state,
                action,
                action_count: self.action_count(),
            });
        }
        let mask_entries = self.horizon().checked_mul(self.state_count())
            .and_then(|value| value.checked_mul(self.action_count()))
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if mask_entries > budget.max_mask_entries {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "action-mask split entries",
                required: mask_entries,
                limit: budget.max_mask_entries,
            });
        }
        let one_mask_bytes = mask_entries
            .checked_mul(std::mem::size_of::<bool>())
            .and_then(|value| value.checked_add(self.horizon().checked_mul(std::mem::size_of::<Vec<Vec<bool>>>())?))
            .and_then(|value| value.checked_add(self.horizon().checked_mul(self.state_count())?.checked_mul(std::mem::size_of::<Vec<bool>>())?))
            .and_then(|value| value.checked_add(std::mem::size_of::<FiniteUmdpPolicyActionMask>()))
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        let split_bytes = one_mask_bytes.checked_mul(2)
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if split_bytes > budget.max_output_memory_bytes {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "action-mask child split bytes",
                required: split_bytes,
                limit: budget.max_output_memory_bytes,
            });
        }
        if !self.allowed[time][state][action] {
            return Err(PolicyPortfolioError::PolicyMaskActionNotAllowed {
                time,
                state,
                action,
            });
        }
        if self.allowed[time][state].iter().filter(|allowed| **allowed).count() < 2 {
            return Err(PolicyPortfolioError::PolicyMaskCannotSplit { time, state });
        }

        let mut forced = try_clone_action_sets(&self.allowed)?;
        let mut forbidden = try_clone_action_sets(&self.allowed)?;
        for candidate in 0..self.action_count() {
            forced[time][state][candidate] = candidate == action;
        }
        forbidden[time][state][action] = false;

        Ok((
            Self::new(family, forced)?,
            Self::new(family, forbidden)?,
        ))
    }
}

fn try_clone_action_sets(
    source: &[Vec<Vec<bool>>],
) -> Result<Vec<Vec<Vec<bool>>>, PolicyPortfolioError> {
    let mut output = Vec::new();
    output.try_reserve_exact(source.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "policy action-mask time layers",
            entries: source.len(),
        }
    })?;
    for layer in source {
        let mut cloned_layer = Vec::new();
        cloned_layer.try_reserve_exact(layer.len()).map_err(|_| {
            PolicyPortfolioError::AllocationFailed {
                resource: "policy action-mask state rows",
                entries: layer.len(),
            }
        })?;
        for row in layer {
            let mut cloned_row = Vec::new();
            cloned_row.try_reserve_exact(row.len()).map_err(|_| {
                PolicyPortfolioError::AllocationFailed {
                    resource: "policy action-mask action entries",
                    entries: row.len(),
                }
            })?;
            cloned_row.extend_from_slice(row);
            cloned_layer.push(cloned_row);
        }
        output.push(cloned_layer);
    }
    Ok(output)
}

fn canonical_policy_action_mask_sha256(
    family_sha256: [u8; 32],
    horizon: usize,
    states: usize,
    actions: usize,
    allowed: &[Vec<Vec<bool>>],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"SYMPGM-FINITE-UMDP-ACTION-MASK\0\x01");
    hasher.update(family_sha256);
    for value in [horizon, states, actions] {
        hasher.update((value as u64).to_le_bytes());
    }
    for layer in allowed {
        for row in layer {
            for allowed in row {
                hasher.update([u8::from(*allowed)]);
            }
        }
    }
    hasher.finalize().into()
}

/// Per-model values used to justify an exact cluster regret lower bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyMaskModelBound {
    pub model_index: usize,
    pub model_sha256: [u8; 32],
    pub unrestricted_optimal_value: CheckedRational,
    /// Optimal value when every decision slot is restricted by the action mask.
    pub masked_optimal_value: CheckedRational,
    /// Unrestricted optimum minus masked optimum, a lower bound on any policy
    /// in this cluster that obeys the action mask.
    pub regret_gap_lower_bound: CheckedRational,
}

/// Exact lower bound for all common policies in one model cluster that obey
/// the supplied time/state action mask.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPolicyMaskClusterLowerBound {
    pub model_family_sha256: [u8; 32],
    pub action_mask_sha256: [u8; 32],
    pub discount: CheckedRational,
    pub cluster_model_indices: Vec<usize>,
    pub per_model: Vec<PolicyMaskModelBound>,
    pub lower_bound: CheckedRational,
    pub dp_multiply_adds: usize,
    pub certificate_sha256: [u8; 32],
}

/// Resource limits for exact single-cluster action-mask branch-and-bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactClusterPolicySearchBudget {
    pub per_node: PolicyActionMaskBudget,
    pub max_search_nodes: usize,
    pub max_total_dp_multiply_adds: usize,
    pub max_frontier_masks: usize,
    pub max_frontier_memory_bytes: usize,
    /// Peak estimate includes the active mask, queued masks, one masked-DP
    /// evaluation's scratch, and the returned certificate vectors.
    pub max_total_memory_bytes: usize,
}

/// Aggregate fail-closed limits for the outer finite-family partition search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactPolicyPortfolioSearchBudget {
    pub per_cluster: ExactClusterPolicySearchBudget,
    pub max_outer_search_nodes: usize,
    pub max_total_inner_dp_multiply_adds: usize,
    pub max_frontier_nodes: usize,
    pub max_frontier_memory_bytes: usize,
    pub max_cached_cluster_solutions: usize,
    pub max_cached_cluster_bytes: usize,
    pub max_total_memory_bytes: usize,
}

/// Complete exact minimax-regret portfolio result for an explicit finite family.
/// Each cluster contributes one deterministic time-dependent Markov policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedFiniteFamilyPolicyPortfolioOptimum {
    pub model_family_sha256: [u8; 32],
    pub policy_manifest_sha256: [u8; 32],
    pub discount: CheckedRational,
    pub max_portfolio_size: usize,
    /// Canonical restricted-growth partition; model indices are sorted per cluster.
    pub partition: Vec<Vec<usize>>,
    pub cluster_solutions: Vec<VerifiedClusterPolicyOptimum>,
    pub model_optimal_values: Vec<CheckedRational>,
    /// Model-major, selected-cluster-policy order.
    pub model_policy_values: Vec<Vec<CheckedRational>>,
    pub model_portfolio_values: Vec<CheckedRational>,
    pub model_regrets: Vec<CheckedRational>,
    pub worst_case_model_indices: Vec<usize>,
    pub minimax_regret: CheckedRational,
    /// Equal to the upper bound only after the complete outer frontier is closed.
    pub global_lower_bound: CheckedRational,
    pub global_upper_bound: CheckedRational,
    pub outer_search_nodes: usize,
    pub outer_pruned_nodes: usize,
    pub complete_partitions_evaluated: usize,
    pub cached_cluster_solutions: usize,
    pub total_cluster_search_nodes: usize,
    pub total_dp_multiply_adds: usize,
    pub partition_transcript_sha256: [u8; 32],
    pub certificate_sha256: [u8; 32],
    pub optimality_proven: bool,
}

/// Exact minimax-regret policy for one model cluster. Equal-regret policies
/// resolve to the lexicographically least time-major/state-major action vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedClusterPolicyOptimum {
    pub model_family_sha256: [u8; 32],
    pub discount: CheckedRational,
    pub cluster_model_indices: Vec<usize>,
    pub policy_actions: Vec<usize>,
    pub policy_index: usize,
    pub model_optimal_values: Vec<CheckedRational>,
    pub model_policy_values: Vec<CheckedRational>,
    pub model_regrets: Vec<CheckedRational>,
    pub worst_case_model_indices: Vec<usize>,
    pub minimax_regret: CheckedRational,
    pub search_nodes: usize,
    pub pruned_nodes: usize,
    pub dp_multiply_adds: usize,
    pub certificate_sha256: [u8; 32],
}

/// Evaluate one model under a per-time/state allowed-action mask using exact
/// finite-horizon backward dynamic programming.
fn masked_optimal_value(
    model: &FiniteHorizonUmdpModel,
    discount: CheckedRational,
    mask: &FiniteUmdpPolicyActionMask,
    max_rational_bit_width: usize,
) -> Result<CheckedRational, PolicyPortfolioError> {
    let states = model.state_count();
    let horizon = model.horizon();
    let mut current = Vec::new();
    current.try_reserve_exact(states).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "masked DP current vector",
            entries: states,
        }
    })?;
    current.resize(states, CheckedRational::zero());
    let mut next = Vec::new();
    next.try_reserve_exact(states).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "masked DP next vector",
            entries: states,
        }
    })?;
    next.resize(states, CheckedRational::zero());

    for time in (0..horizon).rev() {
        for state in 0..states {
            let mut best: Option<CheckedRational> = None;
            for action in 0..model.action_count() {
                if !mask.allowed[time][state][action] {
                    continue;
                }
                let mut continuation = CheckedRational::zero();
                if time + 1 < horizon {
                    for destination in 0..states {
                        continuation = add_bounded(
                            continuation,
                            mul_bounded(
                                model.transitions[time][state][action][destination],
                                next[destination],
                                max_rational_bit_width,
                            )?,
                            max_rational_bit_width,
                        )?;
                    }
                    continuation = mul_bounded(
                        discount,
                        continuation,
                        max_rational_bit_width,
                    )?;
                }
                let candidate = add_bounded(
                    model.rewards[time][state][action],
                    continuation,
                    max_rational_bit_width,
                )?;
                if best.map_or(true, |value| compare_signed(candidate, value) == Ordering::Greater) {
                    best = Some(candidate);
                }
            }
            current[state] = best.ok_or(
                PolicyPortfolioError::PolicyMaskEmptyActionSet { time, state }
            )?;
        }
        current.swap_with_slice(&mut next);
    }

    let mut initial_value = CheckedRational::zero();
    for state in 0..states {
        initial_value = add_bounded(
            initial_value,
            mul_bounded(
                model.initial[state],
                next[state],
                max_rational_bit_width,
            )?,
            max_rational_bit_width,
        )?;
    }
    Ok(initial_value)
}

/// Conservative deterministic rational multiply/add count for one
/// unrestricted plus one masked Bellman DP on every model in the cluster.
fn policy_mask_dp_multiply_adds(
    family: &FiniteUmdpModelFamily,
    model_count: usize,
) -> Result<usize, PolicyPortfolioError> {
    let horizon = family.horizon();
    let states = family.state_count();
    let actions = family.action_count();
    let transition_ops = horizon.saturating_sub(1)
        .checked_mul(states)
        .and_then(|value| value.checked_mul(actions))
        .and_then(|value| value.checked_mul(states))
        .and_then(|value| value.checked_mul(2))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let step_ops = horizon.checked_mul(states)
        .and_then(|value| value.checked_mul(actions))
        .and_then(|value| value.checked_add(
            horizon.saturating_sub(1).checked_mul(states)?.checked_mul(actions)?
        ))
        .and_then(|value| value.checked_add(states.checked_mul(2)?))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    transition_ops.checked_add(step_ops)
        .and_then(|value| value.checked_mul(model_count))
        .and_then(|value| value.checked_mul(2))
        .ok_or(PolicyPortfolioError::DimensionOverflow)
}

/// Compute an exact rational lower bound for the best common policy in a
/// finite model cluster under the supplied action mask.
///
/// For each model M, masked DP computes U_M^max(A), the best value any
/// policy satisfying the mask can attain in that model. Every common policy
/// pi obeying the mask has V_M^pi <= U_M^max(A), hence
/// V_M^* - V_M^pi >= V_M^* - U_M^max(A). Taking the maximum over models gives
/// a sound lower bound on cluster minimax regret.
///
/// At a leaf mask (one action per time/state), the masked value is exactly
/// the value of the unique common deterministic Markov policy, so the lower
/// bound equals its exact cluster regret. No big-M reward penalty is used.
pub fn compute_policy_mask_cluster_lower_bound(
    family: &FiniteUmdpModelFamily,
    cluster_model_indices: &[usize],
    discount: CheckedRational,
    mask: &FiniteUmdpPolicyActionMask,
    budget: PolicyActionMaskBudget,
) -> Result<VerifiedPolicyMaskClusterLowerBound, PolicyPortfolioError> {
    if discount.is_negative() || compare_signed(discount, CheckedRational::one()) == Ordering::Greater {
        return Err(PolicyPortfolioError::InvalidDiscount);
    }
    if mask.family_sha256 != family.sha256() {
        return Err(PolicyPortfolioError::PolicyMaskFamilyDigestMismatch);
    }
    if cluster_model_indices.is_empty() {
        return Err(PolicyPortfolioError::EmptyModelCluster);
    }
    if cluster_model_indices.len() > budget.max_model_count {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "action-mask cluster model count",
            required: cluster_model_indices.len(),
            limit: budget.max_model_count,
        });
    }
    for (position, &model_index) in cluster_model_indices.iter().enumerate() {
        if model_index >= family.model_count() {
            return Err(PolicyPortfolioError::PolicyMaskModelIndexOutOfBounds {
                model_index,
                model_count: family.model_count(),
            });
        }
        if position > 0 {
            let previous = cluster_model_indices[position - 1];
            if model_index == previous {
                return Err(PolicyPortfolioError::DuplicatePolicyMaskModelIndex { model_index });
            }
            if model_index < previous {
                return Err(PolicyPortfolioError::PolicyMaskModelIndicesNotIncreasing {
                    previous,
                    current: model_index,
                });
            }
        }
    }

    let states = family.state_count();
    let actions = family.action_count();
    let horizon = family.horizon();
    let mask_entries = horizon.checked_mul(states)
        .and_then(|value| value.checked_mul(actions))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if mask_entries > budget.max_mask_entries {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "action-mask entries",
            required: mask_entries,
            limit: budget.max_mask_entries,
        });
    }

    let dp_multiply_adds = policy_mask_dp_multiply_adds(
        family,
        cluster_model_indices.len(),
    )?;
    if dp_multiply_adds > budget.max_dp_multiply_adds {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "action-mask Bellman multiply-add work",
            required: dp_multiply_adds,
            limit: budget.max_dp_multiply_adds,
        });
    }

    let width = bit_width(discount);
    if width > budget.max_rational_bit_width {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "rational bit width",
            required: width,
            limit: budget.max_rational_bit_width,
        });
    }
    for &model_index in cluster_model_indices {
        let model = &family.models[model_index];
        for value in model.initial.iter()
            .chain(model.rewards.iter().flatten().flatten())
            .chain(model.transitions.iter().flatten().flatten().flatten())
        {
            let required = bit_width(*value);
            if required > budget.max_rational_bit_width {
                return Err(PolicyPortfolioError::BudgetExceeded {
                    resource: "rational bit width",
                    required,
                    limit: budget.max_rational_bit_width,
                });
            }
        }
    }

    let index_bytes = cluster_model_indices.len()
        .checked_mul(std::mem::size_of::<usize>())
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let per_model_bytes = cluster_model_indices.len()
        .checked_mul(std::mem::size_of::<PolicyMaskModelBound>())
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    // The unrestricted current/next vectors remain live while the masked
    // evaluator allocates its own current/next pair.
    let dp_scratch_bytes = states.checked_mul(4)
        .and_then(|value| value.checked_mul(std::mem::size_of::<CheckedRational>()))
        .and_then(|value| value.checked_add(4 * std::mem::size_of::<Vec<CheckedRational>>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let output_bytes = index_bytes
        .checked_add(per_model_bytes)
        .and_then(|value| value.checked_add(dp_scratch_bytes))
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<usize>>()))
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<PolicyMaskModelBound>>()))
        .and_then(|value| value.checked_add(std::mem::size_of::<VerifiedPolicyMaskClusterLowerBound>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if output_bytes > budget.max_output_memory_bytes {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "action-mask lower-bound output bytes",
            required: output_bytes,
            limit: budget.max_output_memory_bytes,
        });
    }

    let mut per_model = Vec::new();
    per_model.try_reserve_exact(cluster_model_indices.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "action-mask per-model proof values",
            entries: cluster_model_indices.len(),
        }
    })?;
    let mut lower_bound = CheckedRational::zero();
    for &model_index in cluster_model_indices {
        let model = &family.models[model_index];
        let mut current = Vec::new();
        current.try_reserve_exact(states).map_err(|_| {
            PolicyPortfolioError::AllocationFailed {
                resource: "unrestricted DP current vector",
                entries: states,
            }
        })?;
        current.resize(states, CheckedRational::zero());
        let mut next = Vec::new();
        next.try_reserve_exact(states).map_err(|_| {
            PolicyPortfolioError::AllocationFailed {
                resource: "unrestricted DP next vector",
                entries: states,
            }
        })?;
        next.resize(states, CheckedRational::zero());

        let unrestricted_optimal_value = optimal_value(
            model,
            discount,
            &mut current,
            &mut next,
            budget.max_rational_bit_width,
        )?;
        let masked_value = masked_optimal_value(
            model,
            discount,
            mask,
            budget.max_rational_bit_width,
        )?;
        let regret_gap_lower_bound = sub_bounded(
            unrestricted_optimal_value,
            masked_value,
            budget.max_rational_bit_width,
        )?;
        if regret_gap_lower_bound.is_negative() {
            return Err(PolicyPortfolioError::NegativeRegret { model: model_index });
        }
        if compare_signed(regret_gap_lower_bound, lower_bound) == Ordering::Greater {
            lower_bound = regret_gap_lower_bound;
        }
        per_model.push(PolicyMaskModelBound {
            model_index,
            model_sha256: family.model_digests[model_index],
            unrestricted_optimal_value,
            masked_optimal_value: masked_value,
            regret_gap_lower_bound,
        });
    }

    let model_family_sha256 = family.sha256();
    let certificate_sha256 = canonical_policy_mask_bound_sha256(
        model_family_sha256,
        mask.sha256,
        discount,
        cluster_model_indices,
        &per_model,
        lower_bound,
        dp_multiply_adds,
    );
    let mut stored_cluster_indices = Vec::new();
    stored_cluster_indices.try_reserve_exact(cluster_model_indices.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "stored action-mask cluster indices",
            entries: cluster_model_indices.len(),
        }
    })?;
    stored_cluster_indices.extend_from_slice(cluster_model_indices);

    Ok(VerifiedPolicyMaskClusterLowerBound {
        model_family_sha256,
        action_mask_sha256: mask.sha256,
        discount,
        cluster_model_indices: stored_cluster_indices,
        per_model,
        lower_bound,
        dp_multiply_adds,
        certificate_sha256,
    })
}

fn policy_action_mask_structural_bytes(
    horizon: usize,
    states: usize,
    actions: usize,
) -> Result<usize, PolicyPortfolioError> {
    let entries = horizon.checked_mul(states)
        .and_then(|value| value.checked_mul(actions))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    entries.checked_mul(std::mem::size_of::<bool>())
        .and_then(|value| value.checked_add(horizon.checked_mul(std::mem::size_of::<Vec<Vec<bool>>>())?))
        .and_then(|value| value.checked_add(horizon.checked_mul(states)?.checked_mul(std::mem::size_of::<Vec<bool>>())?))
        .and_then(|value| value.checked_add(std::mem::size_of::<FiniteUmdpPolicyActionMask>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)
}

fn full_policy_action_mask(
    family: &FiniteUmdpModelFamily,
    budget: PolicyActionMaskBudget,
) -> Result<FiniteUmdpPolicyActionMask, PolicyPortfolioError> {
    let horizon = family.horizon();
    let states = family.state_count();
    let actions = family.action_count();
    let entries = horizon.checked_mul(states)
        .and_then(|value| value.checked_mul(actions))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if entries > budget.max_mask_entries {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "action-mask entries",
            required: entries,
            limit: budget.max_mask_entries,
        });
    }
    let bytes = policy_action_mask_structural_bytes(horizon, states, actions)?;
    if bytes > budget.max_output_memory_bytes {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "full action-mask bytes",
            required: bytes,
            limit: budget.max_output_memory_bytes,
        });
    }

    let mut layers = Vec::new();
    layers.try_reserve_exact(horizon).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "full action-mask layers",
            entries: horizon,
        }
    })?;
    for _ in 0..horizon {
        let mut layer = Vec::new();
        layer.try_reserve_exact(states).map_err(|_| {
            PolicyPortfolioError::AllocationFailed {
                resource: "full action-mask state rows",
                entries: states,
            }
        })?;
        for _ in 0..states {
            let mut row = Vec::new();
            row.try_reserve_exact(actions).map_err(|_| {
                PolicyPortfolioError::AllocationFailed {
                    resource: "full action-mask action entries",
                    entries: actions,
                }
            })?;
            row.resize(actions, true);
            layer.push(row);
        }
        layers.push(layer);
    }
    FiniteUmdpPolicyActionMask::new(family, layers)
}

fn policy_mask_is_leaf(mask: &FiniteUmdpPolicyActionMask) -> bool {
    (0..mask.horizon()).all(|time| {
        (0..mask.state_count()).all(|state| {
            mask.allowed[time][state].iter().filter(|allowed| **allowed).count() == 1
        })
    })
}


fn policy_index_from_actions(
    actions: &[usize],
    action_count: usize,
) -> Result<usize, PolicyPortfolioError> {
    let mut index = 0usize;
    for &action in actions {
        if action >= action_count {
            return Err(PolicyPortfolioError::DimensionOverflow);
        }
        index = index.checked_mul(action_count)
            .and_then(|value| value.checked_add(action))
            .ok_or(PolicyPortfolioError::PolicySpaceOverflow)?;
    }
    Ok(index)
}

fn evaluate_cluster_policy(
    family: &FiniteUmdpModelFamily,
    cluster_model_indices: &[usize],
    discount: CheckedRational,
    policy_actions: &[usize],
    model_optimal_values: &[CheckedRational],
    max_rational_bit_width: usize,
) -> Result<(Vec<CheckedRational>, Vec<CheckedRational>, Vec<usize>, CheckedRational), PolicyPortfolioError> {
    let mut values = Vec::new();
    values.try_reserve_exact(cluster_model_indices.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "cluster incumbent policy values",
            entries: cluster_model_indices.len(),
        }
    })?;
    let mut regrets = Vec::new();
    regrets.try_reserve_exact(cluster_model_indices.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "cluster incumbent model regrets",
            entries: cluster_model_indices.len(),
        }
    })?;
    let mut worst_models = Vec::new();
    worst_models.try_reserve_exact(cluster_model_indices.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "cluster incumbent worst-model indices",
            entries: cluster_model_indices.len(),
        }
    })?;

    let states = family.state_count();
    let mut worst_regret = CheckedRational::zero();
    for (position, &model_index) in cluster_model_indices.iter().enumerate() {
        let mut current = Vec::new();
        current.try_reserve_exact(states).map_err(|_| {
            PolicyPortfolioError::AllocationFailed {
                resource: "cluster incumbent DP current vector",
                entries: states,
            }
        })?;
        current.resize(states, CheckedRational::zero());
        let mut next = Vec::new();
        next.try_reserve_exact(states).map_err(|_| {
            PolicyPortfolioError::AllocationFailed {
                resource: "cluster incumbent DP next vector",
                entries: states,
            }
        })?;
        next.resize(states, CheckedRational::zero());

        let value = evaluate_policy(
            &family.models[model_index],
            discount,
            policy_actions,
            &mut current,
            &mut next,
            max_rational_bit_width,
        )?;
        let regret = sub_bounded(
            model_optimal_values[position],
            value,
            max_rational_bit_width,
        )?;
        if regret.is_negative() {
            return Err(PolicyPortfolioError::NegativeRegret { model: model_index });
        }
        if compare_signed(regret, worst_regret) == Ordering::Greater {
            worst_regret = regret;
            worst_models.clear();
            worst_models.push(model_index);
        } else if regret == worst_regret {
            worst_models.push(model_index);
        }
        values.push(value);
        regrets.push(regret);
    }
    Ok((values, regrets, worst_models, worst_regret))
}

fn canonical_policy_mask_bound_sha256(
    model_family_sha256: [u8; 32],
    action_mask_sha256: [u8; 32],
    discount: CheckedRational,
    cluster_model_indices: &[usize],
    per_model: &[PolicyMaskModelBound],
    lower_bound: CheckedRational,
    dp_multiply_adds: usize,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"SYMPGM-FINITE-UMDP-ACTION-MASK-BOUND\0\x01");
    hasher.update(model_family_sha256);
    hasher.update(action_mask_sha256);
    hash_rational(&mut hasher, discount);
    hasher.update((cluster_model_indices.len() as u64).to_le_bytes());
    for index in cluster_model_indices {
        hasher.update((*index as u64).to_le_bytes());
    }
    hasher.update((per_model.len() as u64).to_le_bytes());
    for witness in per_model {
        hasher.update((witness.model_index as u64).to_le_bytes());
        hasher.update(witness.model_sha256);
        hash_rational(&mut hasher, witness.unrestricted_optimal_value);
        hash_rational(&mut hasher, witness.masked_optimal_value);
        hash_rational(&mut hasher, witness.regret_gap_lower_bound);
    }
    hash_rational(&mut hasher, lower_bound);
    hasher.update((dp_multiply_adds as u64).to_le_bytes());
    hasher.finalize().into()
}

/// Recompute and compare every per-model Bellman value and digest in a claimed
/// action-mask cluster lower-bound witness.
pub fn verify_policy_mask_cluster_lower_bound(
    family: &FiniteUmdpModelFamily,
    mask: &FiniteUmdpPolicyActionMask,
    budget: PolicyActionMaskBudget,
    claimed: &VerifiedPolicyMaskClusterLowerBound,
) -> Result<(), PolicyPortfolioError> {
    let replay = compute_policy_mask_cluster_lower_bound(
        family,
        &claimed.cluster_model_indices,
        claimed.discount,
        mask,
        budget,
    )?;
    if replay == *claimed {
        Ok(())
    } else {
        Err(PolicyPortfolioError::PolicyMaskCertificateMismatch)
    }
}


fn fixed_policy_dp_multiply_adds(
    family: &FiniteUmdpModelFamily,
    model_count: usize,
) -> Result<usize, PolicyPortfolioError> {
    let horizon = family.horizon();
    let states = family.state_count();
    let transition_dot = horizon.saturating_sub(1)
        .checked_mul(states)
        .and_then(|v| v.checked_mul(states))
        .and_then(|v| v.checked_mul(2))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let step_ops = horizon.saturating_sub(1).checked_mul(states)
        .and_then(|v| v.checked_add(horizon.checked_mul(states)?))
        .and_then(|v| v.checked_add(states.checked_mul(2)?))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    transition_dot.checked_add(step_ops)
        .and_then(|v| v.checked_mul(model_count))
        .ok_or(PolicyPortfolioError::DimensionOverflow)
}

/// Exact single-cluster minimax-regret branch-and-bound over action masks.
/// Budget exhaustion is an error: no incomplete incumbent is returned as
/// an optimum. Equal-regret policies resolve to the lexicographically least
/// time-major/state-major action vector.
pub fn solve_exact_cluster_policy_bnb(
    family: &FiniteUmdpModelFamily,
    cluster_model_indices: &[usize],
    discount: CheckedRational,
    budget: ExactClusterPolicySearchBudget,
) -> Result<VerifiedClusterPolicyOptimum, PolicyPortfolioError> {
    if discount.is_negative() || compare_signed(discount, CheckedRational::one()) == Ordering::Greater {
        return Err(PolicyPortfolioError::InvalidDiscount);
    }
    if cluster_model_indices.is_empty() {
        return Err(PolicyPortfolioError::EmptyModelCluster);
    }
    if cluster_model_indices.len() > budget.per_node.max_model_count {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "action-mask cluster model count",
            required: cluster_model_indices.len(),
            limit: budget.per_node.max_model_count,
        });
    }
    for (position, &model_index) in cluster_model_indices.iter().enumerate() {
        if model_index >= family.model_count() {
            return Err(PolicyPortfolioError::PolicyMaskModelIndexOutOfBounds {
                model_index,
                model_count: family.model_count(),
            });
        }
        if position > 0 {
            let previous = cluster_model_indices[position - 1];
            if model_index == previous {
                return Err(PolicyPortfolioError::DuplicatePolicyMaskModelIndex { model_index });
            }
            if model_index < previous {
                return Err(PolicyPortfolioError::PolicyMaskModelIndicesNotIncreasing { previous, current: model_index });
            }
        }
    }

    let _policy_count = checked_policy_count(family.state_count(), family.horizon(), family.action_count())?;
    if budget.max_search_nodes == 0 {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "cluster B&B search nodes", required: 1, limit: 0,
        });
    }
    if budget.max_frontier_masks == 0 {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "cluster B&B frontier masks", required: 1, limit: 0,
        });
    }

    let horizon = family.horizon();
    let states = family.state_count();
    let actions = family.action_count();
    let mask_bytes = policy_action_mask_structural_bytes(horizon, states, actions)?;
    let policy_entries = horizon.checked_mul(states).ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let result_bytes = policy_entries.checked_mul(std::mem::size_of::<usize>())
        .and_then(|v| v.checked_add(cluster_model_indices.len().checked_mul(3 * std::mem::size_of::<CheckedRational>())?))
        .and_then(|v| v.checked_add(cluster_model_indices.len().checked_mul(2 * std::mem::size_of::<usize>())?))
        .and_then(|v| v.checked_add(6 * std::mem::size_of::<Vec<usize>>()))
        .and_then(|v| v.checked_add(3 * std::mem::size_of::<Vec<CheckedRational>>()))
        .and_then(|v| v.checked_add(std::mem::size_of::<VerifiedClusterPolicyOptimum>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let root_frontier_bytes = mask_bytes.checked_add(std::mem::size_of::<Vec<FiniteUmdpPolicyActionMask>>())
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if root_frontier_bytes > budget.max_frontier_memory_bytes {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "cluster B&B frontier memory bytes",
            required: root_frontier_bytes,
            limit: budget.max_frontier_memory_bytes,
        });
    }
    let root_peak = root_frontier_bytes
        .checked_add(budget.per_node.max_output_memory_bytes)
        .and_then(|v| v.checked_add(result_bytes))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if root_peak > budget.max_total_memory_bytes {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "cluster B&B root peak memory bytes",
            required: root_peak,
            limit: budget.max_total_memory_bytes,
        });
    }

    let full_mask = full_policy_action_mask(family, budget.per_node)?;
    let root_bound = compute_policy_mask_cluster_lower_bound(
        family, cluster_model_indices, discount, &full_mask, budget.per_node,
    )?;
    let mut dp_multiply_adds = root_bound.dp_multiply_adds;
    let incumbent_ops = fixed_policy_dp_multiply_adds(family, cluster_model_indices.len())?;
    dp_multiply_adds = dp_multiply_adds.checked_add(incumbent_ops)
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if dp_multiply_adds > budget.max_total_dp_multiply_adds {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "cluster B&B total DP multiply-add work",
            required: dp_multiply_adds,
            limit: budget.max_total_dp_multiply_adds,
        });
    }

    let mut model_optimal_values = Vec::new();
    model_optimal_values.try_reserve_exact(cluster_model_indices.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed { resource: "cluster B&B model-optimal values", entries: cluster_model_indices.len() }
    })?;
    model_optimal_values.extend(root_bound.per_model.iter().map(|w| w.unrestricted_optimal_value));

    // Policy index zero (all first actions) is a valid incumbent and is also
    // the lexicographically least policy, so equal-regret later leaves cannot
    // improve the deterministic tie-break.
    let mut policy_actions = Vec::new();
    policy_actions.try_reserve_exact(policy_entries).map_err(|_| {
        PolicyPortfolioError::AllocationFailed { resource: "cluster B&B incumbent policy actions", entries: policy_entries }
    })?;
    policy_actions.resize(policy_entries, 0usize);
    let (mut policy_values, mut model_regrets, mut worst_case_model_indices, mut minimax_regret) =
        evaluate_cluster_policy(
            family, cluster_model_indices, discount, &policy_actions,
            &model_optimal_values, budget.per_node.max_rational_bit_width,
        )?;
    let mut policy_index = 0usize;

    let mut frontier = Vec::new();
    frontier.try_reserve_exact(1).map_err(|_| {
        PolicyPortfolioError::AllocationFailed { resource: "cluster B&B frontier", entries: 1 }
    })?;
    frontier.push(full_mask);
    let mut pending_root_bound = Some(root_bound);
    let mut search_nodes = 0usize;
    let mut pruned_nodes = 0usize;

    while let Some(mask) = frontier.pop() {
        search_nodes = search_nodes.checked_add(1).ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if search_nodes > budget.max_search_nodes {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "cluster B&B search nodes",
                required: search_nodes,
                limit: budget.max_search_nodes,
            });
        }

        let active_masks = frontier.len().checked_add(1).ok_or(PolicyPortfolioError::DimensionOverflow)?;
        let active_frontier_bytes = active_masks.checked_mul(mask_bytes)
            .and_then(|v| v.checked_add(std::mem::size_of::<Vec<FiniteUmdpPolicyActionMask>>()))
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if active_frontier_bytes > budget.max_frontier_memory_bytes {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "cluster B&B frontier memory bytes",
                required: active_frontier_bytes,
                limit: budget.max_frontier_memory_bytes,
            });
        }
        let active_peak = active_frontier_bytes
            .checked_add(budget.per_node.max_output_memory_bytes)
            .and_then(|v| v.checked_add(result_bytes))
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if active_peak > budget.max_total_memory_bytes {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "cluster B&B active peak memory bytes",
                required: active_peak,
                limit: budget.max_total_memory_bytes,
            });
        }

        let node_bound = match pending_root_bound.take() {
            Some(bound) => bound,
            None => {
                let node_work = policy_mask_dp_multiply_adds(family, cluster_model_indices.len())?;
                let required_work = dp_multiply_adds.checked_add(node_work)
                    .ok_or(PolicyPortfolioError::DimensionOverflow)?;
                if required_work > budget.max_total_dp_multiply_adds {
                    return Err(PolicyPortfolioError::BudgetExceeded {
                        resource: "cluster B&B total DP multiply-add work",
                        required: required_work,
                        limit: budget.max_total_dp_multiply_adds,
                    });
                }
                let bound = compute_policy_mask_cluster_lower_bound(
                    family, cluster_model_indices, discount, &mask, budget.per_node,
                )?;
                dp_multiply_adds = required_work;
                bound
            }
        };

        if compare_signed(node_bound.lower_bound, minimax_regret) != Ordering::Less {
            pruned_nodes = pruned_nodes.checked_add(1).ok_or(PolicyPortfolioError::DimensionOverflow)?;
            continue;
        }

        if policy_mask_is_leaf(&mask) {
            // A leaf contains one allowed action per slot. Reuse the incumbent
            // output buffers to avoid a second copy of all model witnesses.
            for time in 0..mask.horizon() {
                for state in 0..mask.state_count() {
                    let action = mask.allowed[time][state].iter()
                        .position(|allowed| *allowed)
                        .ok_or(PolicyPortfolioError::PolicyMaskEmptyActionSet { time, state })?;
                    policy_actions[time * states + state] = action;
                }
            }
            policy_index = policy_index_from_actions(&policy_actions, actions)?;
            policy_values.clear();
            model_regrets.clear();
            worst_case_model_indices.clear();
            for witness in &node_bound.per_model {
                policy_values.push(witness.masked_optimal_value);
                model_regrets.push(witness.regret_gap_lower_bound);
                if witness.regret_gap_lower_bound == node_bound.lower_bound {
                    worst_case_model_indices.push(witness.model_index);
                }
            }
            minimax_regret = node_bound.lower_bound;
            continue;
        }

        let mut branch_slot = None;
        for time in 0..mask.horizon() {
            for state in 0..mask.state_count() {
                if mask.allowed[time][state].iter().filter(|allowed| **allowed).count() > 1 {
                    branch_slot = Some((time, state));
                    break;
                }
            }
            if branch_slot.is_some() { break; }
        }
        let (time, state) = branch_slot.ok_or(PolicyPortfolioError::PolicyMaskCertificateMismatch)?;
        let action = mask.allowed[time][state].iter().position(|allowed| *allowed)
            .ok_or(PolicyPortfolioError::PolicyMaskEmptyActionSet { time, state })?;

        let next_count = frontier.len().checked_add(2).ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if next_count > budget.max_frontier_masks {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "cluster B&B frontier masks",
                required: next_count,
                limit: budget.max_frontier_masks,
            });
        }
        let split_mask_count = frontier.len().checked_add(3).ok_or(PolicyPortfolioError::DimensionOverflow)?;
        let split_frontier_bytes = split_mask_count.checked_mul(mask_bytes)
            .and_then(|v| v.checked_add(std::mem::size_of::<Vec<FiniteUmdpPolicyActionMask>>()))
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if split_frontier_bytes > budget.max_frontier_memory_bytes {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "cluster B&B split frontier memory bytes",
                required: split_frontier_bytes,
                limit: budget.max_frontier_memory_bytes,
            });
        }
        let split_peak = split_frontier_bytes
            .checked_add(budget.per_node.max_output_memory_bytes)
            .and_then(|v| v.checked_add(result_bytes))
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if split_peak > budget.max_total_memory_bytes {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "cluster B&B split peak memory bytes",
                required: split_peak,
                limit: budget.max_total_memory_bytes,
            });
        }

        let (forced, forbidden) = mask.split_on_action(
            family, time, state, action, budget.per_node,
        )?;
        frontier.try_reserve_exact(2).map_err(|_| {
            PolicyPortfolioError::AllocationFailed { resource: "cluster B&B child frontier", entries: 2 }
        })?;
        // LIFO: visit the force-action (lower-index) child first.
        frontier.push(forbidden);
        frontier.push(forced);
    }

    let model_family_sha256 = family.sha256();
    let certificate_sha256 = canonical_cluster_policy_optimum_sha256(
        model_family_sha256, discount, cluster_model_indices, &policy_actions, policy_index,
        &model_optimal_values, &policy_values, &model_regrets, &worst_case_model_indices,
        minimax_regret, search_nodes, pruned_nodes, dp_multiply_adds,
    );
    let mut stored_cluster_indices = Vec::new();
    stored_cluster_indices.try_reserve_exact(cluster_model_indices.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed { resource: "stored cluster B&B model indices", entries: cluster_model_indices.len() }
    })?;
    stored_cluster_indices.extend_from_slice(cluster_model_indices);

    Ok(VerifiedClusterPolicyOptimum {
        model_family_sha256,
        discount,
        cluster_model_indices: stored_cluster_indices,
        policy_actions,
        policy_index,
        model_optimal_values,
        model_policy_values: policy_values,
        model_regrets,
        worst_case_model_indices,
        minimax_regret,
        search_nodes,
        pruned_nodes,
        dp_multiply_adds,
        certificate_sha256,
    })
}

fn canonical_cluster_policy_optimum_sha256(
    family_sha256: [u8; 32],
    discount: CheckedRational,
    cluster_model_indices: &[usize],
    policy_actions: &[usize],
    policy_index: usize,
    model_optimal_values: &[CheckedRational],
    model_policy_values: &[CheckedRational],
    model_regrets: &[CheckedRational],
    worst_case_model_indices: &[usize],
    minimax_regret: CheckedRational,
    search_nodes: usize,
    pruned_nodes: usize,
    dp_multiply_adds: usize,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"SYMPGM-FINITE-UMDP-CLUSTER-POLICY-BNB\0\x01");
    hasher.update(family_sha256);
    hash_rational(&mut hasher, discount);
    for indices in [cluster_model_indices, policy_actions, worst_case_model_indices] {
        hasher.update((indices.len() as u64).to_le_bytes());
        for index in indices { hasher.update((*index as u64).to_le_bytes()); }
    }
    hasher.update((policy_index as u64).to_le_bytes());
    for values in [model_optimal_values, model_policy_values, model_regrets] {
        hasher.update((values.len() as u64).to_le_bytes());
        for value in values { hash_rational(&mut hasher, *value); }
    }
    hash_rational(&mut hasher, minimax_regret);
    for value in [search_nodes, pruned_nodes, dp_multiply_adds] {
        hasher.update((value as u64).to_le_bytes());
    }
    hasher.finalize().into()
}

/// Replay the exact cluster solver and compare the complete policy, witnesses,
/// search counters and source-bound certificate.
pub fn verify_exact_cluster_policy_bnb(
    family: &FiniteUmdpModelFamily,
    budget: ExactClusterPolicySearchBudget,
    claimed: &VerifiedClusterPolicyOptimum,
) -> Result<(), PolicyPortfolioError> {
    let replay = solve_exact_cluster_policy_bnb(
        family, &claimed.cluster_model_indices, claimed.discount, budget,
    )?;
    if replay == *claimed {
        Ok(())
    } else {
        Err(PolicyPortfolioError::PolicyMaskCertificateMismatch)
    }
}


#[derive(Clone, Debug, PartialEq, Eq)]
struct OuterPartitionSearchNode {
    next_model_index: usize,
    clusters: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClusterPolicyCacheEntry {
    model_indices: Vec<usize>,
    solution: VerifiedClusterPolicyOptimum,
}

fn try_clone_usize_vec(source: &[usize]) -> Result<Vec<usize>, PolicyPortfolioError> {
    let mut output = Vec::new();
    output.try_reserve_exact(source.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer policy-portfolio index copy",
            entries: source.len(),
        }
    })?;
    output.extend_from_slice(source);
    Ok(output)
}

fn try_clone_partition(
    source: &[Vec<usize>],
) -> Result<Vec<Vec<usize>>, PolicyPortfolioError> {
    let mut output = Vec::new();
    output.try_reserve_exact(source.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer partition cluster list",
            entries: source.len(),
        }
    })?;
    for cluster in source {
        output.push(try_clone_usize_vec(cluster)?);
    }
    Ok(output)
}

fn try_clone_cluster_policy_optimum(
    source: &VerifiedClusterPolicyOptimum,
) -> Result<VerifiedClusterPolicyOptimum, PolicyPortfolioError> {
    Ok(VerifiedClusterPolicyOptimum {
        model_family_sha256: source.model_family_sha256,
        discount: source.discount,
        cluster_model_indices: try_clone_usize_vec(&source.cluster_model_indices)?,
        policy_actions: try_clone_usize_vec(&source.policy_actions)?,
        policy_index: source.policy_index,
        model_optimal_values: clone_rational_vec(&source.model_optimal_values, "cluster optimum values")?,
        model_policy_values: clone_rational_vec(&source.model_policy_values, "cluster policy values")?,
        model_regrets: clone_rational_vec(&source.model_regrets, "cluster policy regrets")?,
        worst_case_model_indices: try_clone_usize_vec(&source.worst_case_model_indices)?,
        minimax_regret: source.minimax_regret,
        search_nodes: source.search_nodes,
        pruned_nodes: source.pruned_nodes,
        dp_multiply_adds: source.dp_multiply_adds,
        certificate_sha256: source.certificate_sha256,
    })
}

fn clone_rational_vec(
    source: &[CheckedRational],
    resource: &'static str,
) -> Result<Vec<CheckedRational>, PolicyPortfolioError> {
    let mut output = Vec::new();
    output.try_reserve_exact(source.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed { resource, entries: source.len() }
    })?;
    output.extend_from_slice(source);
    Ok(output)
}

fn outer_partition_node_bytes(
    node: &OuterPartitionSearchNode,
) -> Result<usize, PolicyPortfolioError> {
    let cluster_headers = node.clusters.len()
        .checked_mul(std::mem::size_of::<Vec<usize>>())
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let member_entries = node.clusters.iter().try_fold(0usize, |total, cluster| {
        total.checked_add(cluster.len()).ok_or(PolicyPortfolioError::DimensionOverflow)
    })?;
    std::mem::size_of::<OuterPartitionSearchNode>()
        .checked_add(std::mem::size_of::<Vec<Vec<usize>>>())
        .and_then(|value| value.checked_add(cluster_headers))
        .and_then(|value| value.checked_add(member_entries.checked_mul(std::mem::size_of::<usize>())?))
        .ok_or(PolicyPortfolioError::DimensionOverflow)
}

fn outer_frontier_bytes(
    frontier: &[OuterPartitionSearchNode],
) -> Result<usize, PolicyPortfolioError> {
    frontier.iter().try_fold(std::mem::size_of::<Vec<OuterPartitionSearchNode>>(), |total, node| {
        total.checked_add(outer_partition_node_bytes(node)?)
            .ok_or(PolicyPortfolioError::DimensionOverflow)
    })
}

fn cluster_policy_optimum_bytes(
    solution: &VerifiedClusterPolicyOptimum,
) -> Result<usize, PolicyPortfolioError> {
    let models = solution.cluster_model_indices.len();
    let policy_entries = solution.policy_actions.len();
    let bytes = models.checked_mul(
        3 * std::mem::size_of::<CheckedRational>() + 2 * std::mem::size_of::<usize>(),
    ).and_then(|value| value.checked_add(policy_entries.checked_mul(std::mem::size_of::<usize>())?))
        .and_then(|value| value.checked_add(6 * std::mem::size_of::<Vec<usize>>()))
        .and_then(|value| value.checked_add(3 * std::mem::size_of::<Vec<CheckedRational>>()))
        .and_then(|value| value.checked_add(std::mem::size_of::<VerifiedClusterPolicyOptimum>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    Ok(bytes)
}

fn cluster_cache_entry_bytes(
    entry: &ClusterPolicyCacheEntry,
) -> Result<usize, PolicyPortfolioError> {
    let key_bytes = entry.model_indices.len()
        .checked_mul(std::mem::size_of::<usize>())
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<usize>>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    key_bytes.checked_add(cluster_policy_optimum_bytes(&entry.solution)?)
        .and_then(|value| value.checked_add(std::mem::size_of::<ClusterPolicyCacheEntry>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)
}

fn cluster_cache_bytes(
    cache: &[ClusterPolicyCacheEntry],
) -> Result<usize, PolicyPortfolioError> {
    cache.iter().try_fold(
        std::mem::size_of::<Vec<ClusterPolicyCacheEntry>>(),
        |total, entry| total.checked_add(cluster_cache_entry_bytes(entry)?)
            .ok_or(PolicyPortfolioError::DimensionOverflow),
    )
}

fn portfolio_output_bytes(
    family: &FiniteUmdpModelFamily,
    partition: &[Vec<usize>],
    cluster_solutions: &[VerifiedClusterPolicyOptimum],
) -> Result<usize, PolicyPortfolioError> {
    let models = family.model_count();
    let policies = partition.len();
    let partition_bytes = partition.iter().try_fold(
        std::mem::size_of::<Vec<Vec<usize>>>() + policies * std::mem::size_of::<Vec<usize>>(),
        |total, cluster| total.checked_add(
            cluster.len().checked_mul(std::mem::size_of::<usize>())?
        ).ok_or(PolicyPortfolioError::DimensionOverflow),
    )?;
    let cluster_bytes = cluster_solutions.iter().try_fold(
        policies.checked_mul(std::mem::size_of::<VerifiedClusterPolicyOptimum>())
            .ok_or(PolicyPortfolioError::DimensionOverflow)?,
        |total, solution| total.checked_add(cluster_policy_optimum_bytes(solution)?)
            .ok_or(PolicyPortfolioError::DimensionOverflow),
    )?;
    let model_value_bytes = models.checked_mul(
        3 * std::mem::size_of::<CheckedRational>()
            + std::mem::size_of::<usize>()
            + policies.checked_mul(std::mem::size_of::<CheckedRational>())
                .ok_or(PolicyPortfolioError::DimensionOverflow)?,
    ).and_then(|value| value.checked_add(models.checked_mul(std::mem::size_of::<Vec<CheckedRational>>())?))
        .and_then(|value| value.checked_add(3 * std::mem::size_of::<Vec<CheckedRational>>()))
        .and_then(|value| value.checked_add(std::mem::size_of::<Vec<usize>>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    partition_bytes.checked_add(cluster_bytes)
        .and_then(|value| value.checked_add(model_value_bytes))
        .and_then(|value| value.checked_add(std::mem::size_of::<VerifiedFiniteFamilyPolicyPortfolioOptimum>()))
        .ok_or(PolicyPortfolioError::DimensionOverflow)
}

fn hash_outer_search_node(
    hasher: &mut Sha256,
    node: &OuterPartitionSearchNode,
    cluster_certificates: &[[u8; 32]],
    lower_bound: CheckedRational,
    event: u8,
) {
    hasher.update((node.next_model_index as u64).to_le_bytes());
    hasher.update((node.clusters.len() as u64).to_le_bytes());
    for cluster in &node.clusters {
        hasher.update((cluster.len() as u64).to_le_bytes());
        for model_index in cluster {
            hasher.update((*model_index as u64).to_le_bytes());
        }
    }
    hasher.update((cluster_certificates.len() as u64).to_le_bytes());
    for digest in cluster_certificates {
        hasher.update(digest);
    }
    hash_rational(hasher, lower_bound);
    hasher.update([event]);
}

fn portfolio_optimum_certificate_sha256(
    result: &VerifiedFiniteFamilyPolicyPortfolioOptimum,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"SYMPGM-FINITE-UMDP-OUTER-PARTITION-BNB\0\x01");
    hasher.update(result.model_family_sha256);
    hasher.update(result.policy_manifest_sha256);
    hash_rational(&mut hasher, result.discount);
    hasher.update((result.max_portfolio_size as u64).to_le_bytes());
    hasher.update((result.partition.len() as u64).to_le_bytes());
    for (cluster, solution) in result.partition.iter().zip(&result.cluster_solutions) {
        hasher.update((cluster.len() as u64).to_le_bytes());
        for model_index in cluster {
            hasher.update((*model_index as u64).to_le_bytes());
        }
        hasher.update(solution.certificate_sha256);
    }
    for values in [
        &result.model_optimal_values,
        &result.model_portfolio_values,
        &result.model_regrets,
    ] {
        hasher.update((values.len() as u64).to_le_bytes());
        for value in values {
            hash_rational(&mut hasher, *value);
        }
    }
    hasher.update((result.model_policy_values.len() as u64).to_le_bytes());
    for row in &result.model_policy_values {
        hasher.update((row.len() as u64).to_le_bytes());
        for value in row {
            hash_rational(&mut hasher, *value);
        }
    }
    hasher.update((result.worst_case_model_indices.len() as u64).to_le_bytes());
    for index in &result.worst_case_model_indices {
        hasher.update((*index as u64).to_le_bytes());
    }
    hash_rational(&mut hasher, result.minimax_regret);
    hash_rational(&mut hasher, result.global_lower_bound);
    hash_rational(&mut hasher, result.global_upper_bound);
    for value in [
        result.outer_search_nodes,
        result.outer_pruned_nodes,
        result.complete_partitions_evaluated,
        result.cached_cluster_solutions,
        result.total_cluster_search_nodes,
        result.total_dp_multiply_adds,
    ] {
        hasher.update((value as u64).to_le_bytes());
    }
    hasher.update(result.partition_transcript_sha256);
    hasher.update([u8::from(result.optimality_proven)]);
    hasher.finalize().into()
}

/// Exact outer restricted-growth partition search for a finite family.
///
/// The seed is the valid one-cluster partition. Every outer node assigns a
/// prefix of the ordered model family to at most k nonempty clusters. Existing
/// cluster identities are considered in ascending order, followed by one
/// optional new cluster; this is a canonical restricted-growth traversal and
/// visits each unlabeled set partition exactly once.
///
/// For a partial cluster C, adding models cannot decrease its optimal
/// common-policy minimax regret. Therefore the maximum of the exact inner
/// cluster optima over current clusters is an admissible lower bound for every
/// completion. Nodes are pruned only when that exact bound is >= the incumbent.
///
/// The complete outer frontier is exhausted before an optimum is returned.
/// Every cached cluster solution is an exact inner action-mask B&B certificate.
/// The result is checked by recomputing each selected policy on every model;
/// the portfolio regret must equal the partition objective. Budget exhaustion
/// returns an error, never an unproved incumbent.
pub fn solve_exact_finite_family_policy_portfolio_bnb(
    family: &FiniteUmdpModelFamily,
    discount: CheckedRational,
    max_portfolio_size: usize,
    budget: ExactPolicyPortfolioSearchBudget,
) -> Result<VerifiedFiniteFamilyPolicyPortfolioOptimum, PolicyPortfolioError> {
    let model_count = family.model_count();
    if max_portfolio_size == 0 || max_portfolio_size > model_count {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "outer portfolio size (must be in 1..=model_count)",
            required: max_portfolio_size,
            limit: model_count,
        });
    }
    if discount.is_negative() || compare_signed(discount, CheckedRational::one()) == Ordering::Greater {
        return Err(PolicyPortfolioError::InvalidDiscount);
    }
    if budget.max_outer_search_nodes == 0 {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "outer partition search nodes",
            required: 1,
            limit: 0,
        });
    }
    if budget.max_frontier_nodes == 0 {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "outer partition frontier nodes",
            required: 1,
            limit: 0,
        });
    }
    if budget.max_cached_cluster_solutions == 0 {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "outer cached cluster solutions",
            required: 1,
            limit: 0,
        });
    }

    let policy_count = checked_policy_count(
        family.state_count(),
        family.horizon(),
        family.action_count(),
    )?;
    let policy_manifest = policy_manifest_sha256(
        family.state_count(),
        family.action_count(),
        family.horizon(),
        policy_count,
    );
    let mut all_models = Vec::new();
    all_models.try_reserve_exact(model_count).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer complete model-index list",
            entries: model_count,
        }
    })?;
    all_models.extend(0..model_count);

    // Seed a feasible upper bound using the exact single-cluster solver.
    let mut seed_inner_budget = budget.per_cluster;
    seed_inner_budget.max_total_dp_multiply_adds = seed_inner_budget.max_total_dp_multiply_adds
        .min(budget.max_total_inner_dp_multiply_adds);
    let seed_solution = solve_exact_cluster_policy_bnb(
        family,
        &all_models,
        discount,
        seed_inner_budget,
    )?;
    let seed_dp_work = seed_solution.dp_multiply_adds;
    let seed_search_nodes = seed_solution.search_nodes;
    let seed_entry = ClusterPolicyCacheEntry {
        model_indices: try_clone_usize_vec(&all_models)?,
        solution: seed_solution,
    };
    let mut cache = Vec::new();
    cache.try_reserve_exact(1).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer cluster solution cache",
            entries: 1,
        }
    })?;
    let mut cached_bytes = cluster_cache_entry_bytes(&seed_entry)?
        .checked_add(std::mem::size_of::<Vec<ClusterPolicyCacheEntry>>())
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if cached_bytes > budget.max_cached_cluster_bytes {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "outer cached cluster bytes",
            required: cached_bytes,
            limit: budget.max_cached_cluster_bytes,
        });
    }
    cache.push(seed_entry);
    let mut total_dp_multiply_adds = seed_dp_work;
    let mut total_cluster_search_nodes = seed_search_nodes;
    let mut best_upper_bound = cache[0].solution.minimax_regret;
    let mut best_partition = Vec::new();
    best_partition.try_reserve_exact(1).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer incumbent partition",
            entries: 1,
        }
    })?;
    best_partition.push(try_clone_usize_vec(&all_models)?);

    let mut frontier = Vec::new();
    frontier.try_reserve_exact(1).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer partition frontier",
            entries: 1,
        }
    })?;
    frontier.push(OuterPartitionSearchNode {
        next_model_index: 0,
        clusters: Vec::new(),
    });

    let mut transcript = Sha256::new();
    transcript.update(b"SYMPGM-FINITE-UMDP-OUTER-PARTITION-TRANSCRIPT\0\x01");
    transcript.update(family.sha256());
    transcript.update(policy_manifest);
    hash_rational(&mut transcript, discount);
    transcript.update((max_portfolio_size as u64).to_le_bytes());

    let mut outer_search_nodes = 0usize;
    let mut outer_pruned_nodes = 0usize;
    let mut complete_partitions_evaluated = 0usize;

    while let Some(node) = frontier.pop() {
        outer_search_nodes = outer_search_nodes.checked_add(1)
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if outer_search_nodes > budget.max_outer_search_nodes {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "outer partition search nodes",
                required: outer_search_nodes,
                limit: budget.max_outer_search_nodes,
            });
        }

        let mut node_lower_bound = CheckedRational::zero();
        let mut cluster_certificates = Vec::new();
        cluster_certificates.try_reserve_exact(node.clusters.len()).map_err(|_| {
            PolicyPortfolioError::AllocationFailed {
                resource: "outer node cluster certificate list",
                entries: node.clusters.len(),
            }
        })?;

        for cluster in &node.clusters {
            let cache_index = match cache.iter().position(|entry| entry.model_indices == *cluster) {
                Some(index) => index,
                None => {
                    if cache.len() >= budget.max_cached_cluster_solutions {
                        return Err(PolicyPortfolioError::BudgetExceeded {
                            resource: "outer cached cluster solutions",
                            required: cache.len() + 1,
                            limit: budget.max_cached_cluster_solutions,
                        });
                    }
                    let remaining_work = budget.max_total_inner_dp_multiply_adds
                        .checked_sub(total_dp_multiply_adds)
                        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
                    let mut inner_budget = budget.per_cluster;
                    inner_budget.max_total_dp_multiply_adds = inner_budget.max_total_dp_multiply_adds
                        .min(remaining_work);
                    let frontier_bytes = outer_frontier_bytes(&frontier)?;
                    let active_node_bytes = outer_partition_node_bytes(&node)?;
                    let projected_cache = cached_bytes
                        .checked_add(budget.per_cluster.max_total_memory_bytes)
                        .and_then(|value| value.checked_add(frontier_bytes))
                        .and_then(|value| value.checked_add(active_node_bytes))
                        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
                    if projected_cache > budget.max_total_memory_bytes {
                        return Err(PolicyPortfolioError::BudgetExceeded {
                            resource: "outer memory including active inner cluster solve",
                            required: projected_cache,
                            limit: budget.max_total_memory_bytes,
                        });
                    }

                    let solution = solve_exact_cluster_policy_bnb(
                        family,
                        cluster,
                        discount,
                        inner_budget,
                    )?;
                    total_dp_multiply_adds = total_dp_multiply_adds
                        .checked_add(solution.dp_multiply_adds)
                        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
                    if total_dp_multiply_adds > budget.max_total_inner_dp_multiply_adds {
                        return Err(PolicyPortfolioError::BudgetExceeded {
                            resource: "outer total inner-cluster DP multiply-add work",
                            required: total_dp_multiply_adds,
                            limit: budget.max_total_inner_dp_multiply_adds,
                        });
                    }
                    total_cluster_search_nodes = total_cluster_search_nodes
                        .checked_add(solution.search_nodes)
                        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
                    let entry = ClusterPolicyCacheEntry {
                        model_indices: try_clone_usize_vec(cluster)?,
                        solution,
                    };
                    let entry_bytes = cluster_cache_entry_bytes(&entry)?;
                    cached_bytes = cached_bytes.checked_add(entry_bytes)
                        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
                    if cached_bytes > budget.max_cached_cluster_bytes {
                        return Err(PolicyPortfolioError::BudgetExceeded {
                            resource: "outer cached cluster bytes",
                            required: cached_bytes,
                            limit: budget.max_cached_cluster_bytes,
                        });
                    }
                    cache.try_reserve_exact(1).map_err(|_| {
                        PolicyPortfolioError::AllocationFailed {
                            resource: "outer cluster solution cache entry",
                            entries: 1,
                        }
                    })?;
                    cache.push(entry);
                    cache.len() - 1
                }
            };
            let solution = &cache[cache_index].solution;
            if compare_signed(solution.minimax_regret, node_lower_bound) == Ordering::Greater {
                node_lower_bound = solution.minimax_regret;
            }
            cluster_certificates.push(solution.certificate_sha256);
        }

        if node.next_model_index == model_count {
            complete_partitions_evaluated = complete_partitions_evaluated.checked_add(1)
                .ok_or(PolicyPortfolioError::DimensionOverflow)?;
            if compare_signed(node_lower_bound, best_upper_bound) == Ordering::Less {
                best_upper_bound = node_lower_bound;
                best_partition = try_clone_partition(&node.clusters)?;
            }
            hash_outer_search_node(
                &mut transcript,
                &node,
                &cluster_certificates,
                node_lower_bound,
                2,
            );
            continue;
        }

        if compare_signed(node_lower_bound, best_upper_bound) != Ordering::Less {
            outer_pruned_nodes = outer_pruned_nodes.checked_add(1)
                .ok_or(PolicyPortfolioError::DimensionOverflow)?;
            hash_outer_search_node(
                &mut transcript,
                &node,
                &cluster_certificates,
                node_lower_bound,
                1,
            );
            continue;
        }

        hash_outer_search_node(
            &mut transcript,
            &node,
            &cluster_certificates,
            node_lower_bound,
            0,
        );

        let can_create_cluster = node.clusters.len() < max_portfolio_size;
        let child_count = node.clusters.len()
            .checked_add(usize::from(can_create_cluster))
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        let next_frontier_count = frontier.len().checked_add(child_count)
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if next_frontier_count > budget.max_frontier_nodes {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "outer partition frontier nodes",
                required: next_frontier_count,
                limit: budget.max_frontier_nodes,
            });
        }

        let frontier_bytes = outer_frontier_bytes(&frontier)?;
        let current_node_bytes = outer_partition_node_bytes(&node)?;
        let mut split_frontier_bytes = frontier_bytes
            .checked_add(current_node_bytes)
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        let mut children = Vec::new();
        children.try_reserve_exact(child_count).map_err(|_| {
            PolicyPortfolioError::AllocationFailed {
                resource: "outer partition child staging list",
                entries: child_count,
            }
        })?;

        // Push child choices in reverse so DFS visits existing cluster 0,
        // existing cluster 1, ..., then the new-cluster choice. This is the
        // canonical restricted-growth-string lexicographic order.
        for choice in (0..child_count).rev() {
            let mut child_clusters = try_clone_partition(&node.clusters)?;
            if choice < node.clusters.len() {
                child_clusters[choice].try_reserve_exact(1).map_err(|_| {
                    PolicyPortfolioError::AllocationFailed {
                        resource: "outer partition cluster append",
                        entries: 1,
                    }
                })?;
                child_clusters[choice].push(node.next_model_index);
            } else {
                let mut new_cluster = Vec::new();
                new_cluster.try_reserve_exact(1).map_err(|_| {
                    PolicyPortfolioError::AllocationFailed {
                        resource: "outer new cluster",
                        entries: 1,
                    }
                })?;
                new_cluster.push(node.next_model_index);
                child_clusters.try_reserve_exact(1).map_err(|_| {
                    PolicyPortfolioError::AllocationFailed {
                        resource: "outer partition new cluster slot",
                        entries: 1,
                    }
                })?;
                child_clusters.push(new_cluster);
            }
            let child = OuterPartitionSearchNode {
                next_model_index: node.next_model_index + 1,
                clusters: child_clusters,
            };
            split_frontier_bytes = split_frontier_bytes
                .checked_add(outer_partition_node_bytes(&child)?)
                .ok_or(PolicyPortfolioError::DimensionOverflow)?;
            children.push(child);
        }

        if split_frontier_bytes > budget.max_frontier_memory_bytes {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "outer partition split frontier bytes",
                required: split_frontier_bytes,
                limit: budget.max_frontier_memory_bytes,
            });
        }
        let split_total_memory = split_frontier_bytes
            .checked_add(cached_bytes)
            .and_then(|value| value.checked_add(budget.per_cluster.max_total_memory_bytes))
            .ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if split_total_memory > budget.max_total_memory_bytes {
            return Err(PolicyPortfolioError::BudgetExceeded {
                resource: "outer partition split total memory bytes",
                required: split_total_memory,
                limit: budget.max_total_memory_bytes,
            });
        }

        frontier.try_reserve_exact(child_count).map_err(|_| {
            PolicyPortfolioError::AllocationFailed {
                resource: "outer partition frontier children",
                entries: child_count,
            }
        })?;
        for child in children {
            frontier.push(child);
        }
    }

    // The one-cluster seed is always feasible; complete frontier exhaustion
    // plus exact monotone pruning proves global lower bound == incumbent upper.
    let best_cache_indices: Vec<usize> = best_partition.iter().map(|cluster| {
        cache.iter().position(|entry| entry.model_indices == *cluster)
            .ok_or(PolicyPortfolioError::PolicyMaskCertificateMismatch)
    }).collect::<Result<_, _>>()?;
    let mut output_cluster_solutions = Vec::new();
    output_cluster_solutions.try_reserve_exact(best_partition.len()).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer selected cluster certificates",
            entries: best_partition.len(),
        }
    })?;
    for index in best_cache_indices {
        output_cluster_solutions.push(try_clone_cluster_policy_optimum(&cache[index].solution)?);
    }

    let all_model_cache_index = cache.iter().position(|entry| entry.model_indices == all_models)
        .ok_or(PolicyPortfolioError::PolicyMaskCertificateMismatch)?;
    let model_optimal_values = clone_rational_vec(
        &cache[all_model_cache_index].solution.model_optimal_values,
        "outer model-optimal values",
    )?;
    let output_bytes = portfolio_output_bytes(
        family,
        &best_partition,
        &output_cluster_solutions,
    )?;
    let final_memory = cached_bytes.checked_add(output_bytes)
        .and_then(|value| value.checked_add(family.state_count()
            .checked_mul(2 * std::mem::size_of::<CheckedRational>())?))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if final_memory > budget.max_total_memory_bytes {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "outer final portfolio output memory bytes",
            required: final_memory,
            limit: budget.max_total_memory_bytes,
        });
    }

    let portfolio_evaluation_count = model_count.checked_mul(best_partition.len())
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let portfolio_eval_ops = fixed_policy_dp_multiply_adds(family, portfolio_evaluation_count)?;
    let total_work_with_portfolio_replay = total_dp_multiply_adds
        .checked_add(portfolio_eval_ops)
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    if total_work_with_portfolio_replay > budget.max_total_inner_dp_multiply_adds {
        return Err(PolicyPortfolioError::BudgetExceeded {
            resource: "outer total DP work including cross-model portfolio replay",
            required: total_work_with_portfolio_replay,
            limit: budget.max_total_inner_dp_multiply_adds,
        });
    }

    let mut model_policy_values = Vec::new();
    model_policy_values.try_reserve_exact(model_count).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer model-policy value rows",
            entries: model_count,
        }
    })?;
    let mut model_portfolio_values = Vec::new();
    model_portfolio_values.try_reserve_exact(model_count).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer model portfolio values",
            entries: model_count,
        }
    })?;
    let mut model_regrets = Vec::new();
    model_regrets.try_reserve_exact(model_count).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer model portfolio regrets",
            entries: model_count,
        }
    })?;
    let mut worst_case_model_indices = Vec::new();
    worst_case_model_indices.try_reserve_exact(model_count).map_err(|_| {
        PolicyPortfolioError::AllocationFailed {
            resource: "outer worst-case model indices",
            entries: model_count,
        }
    })?;

    let states = family.state_count();
    for model_index in 0..model_count {
        let model = &family.models[model_index];
        let mut row = Vec::new();
        row.try_reserve_exact(output_cluster_solutions.len()).map_err(|_| {
            PolicyPortfolioError::AllocationFailed {
                resource: "outer model policy evaluation row",
                entries: output_cluster_solutions.len(),
            }
        })?;
        let mut portfolio_value: Option<CheckedRational> = None;
        for cluster_solution in &output_cluster_solutions {
            let mut current = Vec::new();
            current.try_reserve_exact(states).map_err(|_| {
                PolicyPortfolioError::AllocationFailed {
                    resource: "outer portfolio replay current DP",
                    entries: states,
                }
            })?;
            current.resize(states, CheckedRational::zero());
            let mut next = Vec::new();
            next.try_reserve_exact(states).map_err(|_| {
                PolicyPortfolioError::AllocationFailed {
                    resource: "outer portfolio replay next DP",
                    entries: states,
                }
            })?;
            next.resize(states, CheckedRational::zero());
            let value = evaluate_policy(
                model,
                discount,
                &cluster_solution.policy_actions,
                &mut current,
                &mut next,
                budget.per_cluster.per_node.max_rational_bit_width,
            )?;
            row.push(value);
            if portfolio_value.map_or(true, |best| compare_signed(value, best) == Ordering::Greater) {
                portfolio_value = Some(value);
            }
        }
        let portfolio_value = portfolio_value.ok_or(PolicyPortfolioError::EmptyPortfolio)?;
        let regret = sub_bounded(
            model_optimal_values[model_index],
            portfolio_value,
            budget.per_cluster.per_node.max_rational_bit_width,
        )?;
        if regret.is_negative() {
            return Err(PolicyPortfolioError::NegativeRegret { model: model_index });
        }
        if compare_signed(regret, best_upper_bound) == Ordering::Greater {
            worst_case_model_indices.clear();
            worst_case_model_indices.push(model_index);
        } else if regret == best_upper_bound {
            worst_case_model_indices.push(model_index);
        }
        model_policy_values.push(row);
        model_portfolio_values.push(portfolio_value);
        model_regrets.push(regret);
    }
    let portfolio_regret = model_regrets.iter().copied().fold(
        CheckedRational::zero(),
        |worst, regret| if compare_signed(regret, worst) == Ordering::Greater { regret } else { worst },
    );
    if portfolio_regret != best_upper_bound {
        return Err(PolicyPortfolioError::PortfolioCertificateMismatch);
    }

    let partition_transcript_sha256: [u8; 32] = transcript.finalize().into();
    let mut result = VerifiedFiniteFamilyPolicyPortfolioOptimum {
        model_family_sha256: family.sha256(),
        policy_manifest_sha256: policy_manifest,
        discount,
        max_portfolio_size,
        partition: best_partition,
        cluster_solutions: output_cluster_solutions,
        model_optimal_values,
        model_policy_values,
        model_portfolio_values,
        model_regrets,
        worst_case_model_indices,
        minimax_regret: portfolio_regret,
        global_lower_bound: portfolio_regret,
        global_upper_bound: best_upper_bound,
        outer_search_nodes,
        outer_pruned_nodes,
        complete_partitions_evaluated,
        cached_cluster_solutions: cache.len(),
        total_cluster_search_nodes,
        total_dp_multiply_adds: total_work_with_portfolio_replay,
        partition_transcript_sha256,
        certificate_sha256: [0; 32],
        optimality_proven: true,
    };
    result.certificate_sha256 = portfolio_optimum_certificate_sha256(&result);
    Ok(result)
}

/// Independently replay the complete outer partition search, every exact
/// cached inner cluster solve, and cross-model portfolio coverage.
pub fn verify_exact_finite_family_policy_portfolio_bnb(
    family: &FiniteUmdpModelFamily,
    budget: ExactPolicyPortfolioSearchBudget,
    claimed: &VerifiedFiniteFamilyPolicyPortfolioOptimum,
) -> Result<(), PolicyPortfolioError> {
    let replay = solve_exact_finite_family_policy_portfolio_bnb(
        family,
        claimed.discount,
        claimed.max_portfolio_size,
        budget,
    )?;
    if replay == *claimed
        && portfolio_optimum_certificate_sha256(&replay) == claimed.certificate_sha256
    {
        Ok(())
    } else {
        Err(PolicyPortfolioError::PortfolioCertificateMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i128, d: i128) -> CheckedRational {
        CheckedRational::new(n, d).unwrap()
    }

    fn budget(policies: usize, candidates: usize) -> PolicyPortfolioBudget {
        PolicyPortfolioBudget {
            max_policies: policies,
            max_model_policy_evaluations: 10_000,
            max_model_policy_value_entries: 10_000,
            max_dp_multiply_adds: 1_000_000,
            max_portfolio_candidates: candidates,
            max_portfolio_member_comparisons: 1_000_000,
            max_optimal_portfolios: 100,
            max_stored_policy_indices: 1_000,
            max_stored_model_witnesses: 1_000,
            max_output_memory_bytes: 1_000_000,
            max_rational_bit_width: 64,
        }
    }

    fn one_state_model(rewards: &[i128]) -> FiniteHorizonUmdpModel {
        FiniteHorizonUmdpModel::new(
            vec![q(1, 1)],
            vec![vec![rewards.iter().map(|reward| q(*reward, 1)).collect()]],
            vec![],
        ).unwrap()
    }

    fn policy_mask_budget() -> PolicyActionMaskBudget {
        PolicyActionMaskBudget {
            max_model_count: 8,
            max_mask_entries: 10_000,
            max_dp_multiply_adds: 1_000_000,
            max_rational_bit_width: 64,
            max_output_memory_bytes: 1_000_000,
        }
    }

    fn cluster_search_budget() -> ExactClusterPolicySearchBudget {
        ExactClusterPolicySearchBudget {
            per_node: policy_mask_budget(),
            max_search_nodes: 100,
            max_total_dp_multiply_adds: 1_000_000,
            max_frontier_masks: 100,
            max_frontier_memory_bytes: 1_000_000,
            max_total_memory_bytes: 2_000_000,
        }
    }

    #[test]
    fn exact_action_mask_bound_is_zero_for_full_mask_and_exact_at_leaf() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();
        let full = FiniteUmdpPolicyActionMask::new(
            &family,
            vec![vec![vec![true, true]]],
        ).unwrap();
        let full_bound = compute_policy_mask_cluster_lower_bound(
            &family, &[0, 1], q(1, 2), &full, policy_mask_budget(),
        ).unwrap();
        assert_eq!(full_bound.lower_bound, q(0, 1));
        assert_eq!(
            policy_portfolio_sha256_hex(full.sha256()),
            "2a46bf3456a75f5b317ebc1a5a7621b322f369b0cb1cb617a775336d157b7c2a"
        );
        assert_eq!(
            policy_portfolio_sha256_hex(full_bound.certificate_sha256),
            "a943abfaa5fd8c57bb251b24f6346719a85058db74a4373c6cd930fbb60b1c13"
        );
        assert_eq!(full_bound.dp_multiply_adds, 16);
        assert_eq!(
            full_bound.per_model.iter().map(|item| (
                item.unrestricted_optimal_value,
                item.masked_optimal_value,
                item.regret_gap_lower_bound,
            )).collect::<Vec<_>>(),
            vec![(q(1,1), q(1,1), q(0,1)), (q(1,1), q(1,1), q(0,1))]
        );
        verify_policy_mask_cluster_lower_bound(
            &family, &full, policy_mask_budget(), &full_bound,
        ).unwrap();

        let fixed_action_zero = FiniteUmdpPolicyActionMask::new(
            &family,
            vec![vec![vec![true, false]]],
        ).unwrap();
        let leaf_bound = compute_policy_mask_cluster_lower_bound(
            &family, &[0, 1], q(1, 2), &fixed_action_zero, policy_mask_budget(),
        ).unwrap();
        assert_eq!(leaf_bound.lower_bound, q(1, 1));
        assert_eq!(
            leaf_bound.per_model.iter().map(|item| item.regret_gap_lower_bound).collect::<Vec<_>>(),
            vec![q(0,1), q(1,1)]
        );

        let mut mutated = leaf_bound.clone();
        mutated.per_model[1].masked_optimal_value = q(1, 1);
        assert_eq!(
            verify_policy_mask_cluster_lower_bound(
                &family, &fixed_action_zero, policy_mask_budget(), &mutated,
            ),
            Err(PolicyPortfolioError::PolicyMaskCertificateMismatch)
        );
    }

    #[test]
    fn force_forbid_children_are_exhaustive_disjoint_and_source_bound() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();
        let parent = FiniteUmdpPolicyActionMask::new(
            &family,
            vec![vec![vec![true, true]]],
        ).unwrap();
        let (forced, forbidden) = parent.split_on_action(
            &family, 0, 0, 0, policy_mask_budget(),
        ).unwrap();
        assert_eq!(forced.allows(0, 0, 0), Some(true));
        assert_eq!(forced.allows(0, 0, 1), Some(false));
        assert_eq!(forbidden.allows(0, 0, 0), Some(false));
        assert_eq!(forbidden.allows(0, 0, 1), Some(true));
        for action in 0..2 {
            let parent_allowed = parent.allows(0, 0, action).unwrap();
            let forced_allowed = forced.allows(0, 0, action).unwrap();
            let forbidden_allowed = forbidden.allows(0, 0, action).unwrap();
            assert_eq!(forced_allowed && forbidden_allowed, false);
            assert_eq!(forced_allowed || forbidden_allowed, parent_allowed);
        }
        let forced_bound = compute_policy_mask_cluster_lower_bound(
            &family, &[0, 1], q(0,1), &forced, policy_mask_budget(),
        ).unwrap();
        let forbidden_bound = compute_policy_mask_cluster_lower_bound(
            &family, &[0, 1], q(0,1), &forbidden, policy_mask_budget(),
        ).unwrap();
        assert_eq!(forced_bound.lower_bound, q(1,1));
        assert_eq!(forbidden_bound.lower_bound, q(1,1));

        assert_eq!(
            parent.split_on_action(&family, 0, 0, 0, PolicyActionMaskBudget {
                max_model_count: 8,
                max_mask_entries: 1,
                max_dp_multiply_adds: 1_000_000,
                max_rational_bit_width: 64,
                max_output_memory_bytes: 1_000_000,
            }),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "action-mask split entries",
                required: 2,
                limit: 1,
            })
        );
    }

    #[test]
    fn masked_bellman_bound_handles_time_varying_actions_and_discount_exactly() {
        let model = FiniteHorizonUmdpModel::new(
            vec![q(1,1)],
            vec![
                vec![vec![q(1,1), q(0,1)]],
                vec![vec![q(0,1), q(1,1)]],
            ],
            vec![
                vec![
                    vec![vec![q(1,1)], vec![q(1,1)]],
                ],
            ],
        ).unwrap();
        let family = FiniteUmdpModelFamily::new(vec![model]).unwrap();
        let mask = FiniteUmdpPolicyActionMask::new(
            &family,
            vec![
                vec![vec![false, true]],
                vec![vec![false, true]],
            ],
        ).unwrap();
        let lower = compute_policy_mask_cluster_lower_bound(
            &family, &[0], q(1,2), &mask, policy_mask_budget(),
        ).unwrap();
        assert_eq!(lower.per_model[0].unrestricted_optimal_value, q(3,2));
        assert_eq!(lower.per_model[0].masked_optimal_value, q(1,2));
        assert_eq!(lower.lower_bound, q(1,1));
    }

    #[test]
    fn action_mask_validation_family_binding_and_budget_fail_closed() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();
        assert!(matches!(
            FiniteUmdpPolicyActionMask::new(&family, vec![vec![vec![false, false]]]),
            Err(PolicyPortfolioError::PolicyMaskEmptyActionSet { time: 0, state: 0 })
        ));
        let full = FiniteUmdpPolicyActionMask::new(
            &family, vec![vec![vec![true, true]]],
        ).unwrap();
        assert_eq!(
            full.split_on_action(&family, 0, 0, 0, policy_mask_budget()),
            Ok((
                FiniteUmdpPolicyActionMask::new(&family, vec![vec![vec![true, false]]]).unwrap(),
                FiniteUmdpPolicyActionMask::new(&family, vec![vec![vec![false, true]]]).unwrap(),
            ))
        );
        assert_eq!(
            full.split_on_action(&family, 0, 0, 1, PolicyActionMaskBudget {
                max_model_count: 8,
                max_mask_entries: 2,
                max_dp_multiply_adds: 1_000_000,
                max_rational_bit_width: 64,
                max_output_memory_bytes: 1,
            }),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "action-mask child split bytes",
                required: 2 * (2 * std::mem::size_of::<bool>()
                    + std::mem::size_of::<Vec<Vec<bool>>>()
                    + std::mem::size_of::<Vec<bool>>()
                    + std::mem::size_of::<FiniteUmdpPolicyActionMask>()),
                limit: 1,
            })
        );

        let other_family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[2, 0]),
            one_state_model(&[0, 2]),
        ]).unwrap();
        assert_eq!(
            compute_policy_mask_cluster_lower_bound(
                &other_family, &[0,1], q(1,2), &full, policy_mask_budget(),
            ),
            Err(PolicyPortfolioError::PolicyMaskFamilyDigestMismatch)
        );
    }

    #[test]
    fn action_mask_cluster_preflight_rejects_bad_indices_and_work_caps() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();
        let mask = FiniteUmdpPolicyActionMask::new(
            &family, vec![vec![vec![true, true]]],
        ).unwrap();
        assert_eq!(
            compute_policy_mask_cluster_lower_bound(
                &family, &[], q(1,2), &mask, policy_mask_budget(),
            ),
            Err(PolicyPortfolioError::EmptyModelCluster)
        );
        assert_eq!(
            compute_policy_mask_cluster_lower_bound(
                &family, &[2], q(1,2), &mask, policy_mask_budget(),
            ),
            Err(PolicyPortfolioError::PolicyMaskModelIndexOutOfBounds {
                model_index: 2,
                model_count: 2,
            })
        );
        assert_eq!(
            compute_policy_mask_cluster_lower_bound(
                &family, &[0,0], q(1,2), &mask, policy_mask_budget(),
            ),
            Err(PolicyPortfolioError::DuplicatePolicyMaskModelIndex { model_index: 0 })
        );
        assert_eq!(
            compute_policy_mask_cluster_lower_bound(
                &family, &[1,0], q(1,2), &mask, policy_mask_budget(),
            ),
            Err(PolicyPortfolioError::PolicyMaskModelIndicesNotIncreasing {
                previous: 1,
                current: 0,
            })
        );
        let mut operation_tight = policy_mask_budget();
        operation_tight.max_dp_multiply_adds = 15;
        assert_eq!(
            compute_policy_mask_cluster_lower_bound(
                &family, &[0,1], q(1,2), &mask, operation_tight,
            ),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "action-mask Bellman multiply-add work",
                required: 16,
                limit: 15,
            })
        );
        let mut mask_tight = policy_mask_budget();
        mask_tight.max_mask_entries = 1;
        assert_eq!(
            compute_policy_mask_cluster_lower_bound(
                &family, &[0,1], q(1,2), &mask, mask_tight,
            ),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "action-mask entries",
                required: 2,
                limit: 1,
            })
        );
        let mut memory_tight = policy_mask_budget();
        memory_tight.max_output_memory_bytes = 1;
        assert!(matches!(
            compute_policy_mask_cluster_lower_bound(
                &family, &[0,1], q(1,2), &mask, memory_tight,
            ),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "action-mask lower-bound output bytes",
                ..
            })
        ));
    }

    #[test]
    fn action_mask_generated_rational_width_and_leaf_splits_fail_closed() {
        let model = FiniteHorizonUmdpModel::new(
            vec![q(1,1)],
            vec![
                vec![vec![q(0,1)]],
                vec![vec![q(0,1)]],
                vec![vec![q(1,1)]],
            ],
            vec![
                vec![vec![vec![q(1,1)]]],
                vec![vec![vec![q(1,1)]]],
            ],
        ).unwrap();
        let family = FiniteUmdpModelFamily::new(vec![model]).unwrap();
        let mask = FiniteUmdpPolicyActionMask::new(
            &family,
            vec![
                vec![vec![true]],
                vec![vec![true]],
                vec![vec![true]],
            ],
        ).unwrap();
        let mut width_tight = policy_mask_budget();
        width_tight.max_rational_bit_width = 2;
        assert_eq!(
            compute_policy_mask_cluster_lower_bound(
                &family, &[0], q(1,2), &mask, width_tight,
            ),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "rational bit width",
                required: 3,
                limit: 2,
            })
        );

        assert_eq!(
            mask.split_on_action(&family, 0, 0, 0, policy_mask_budget()),
            Err(PolicyPortfolioError::PolicyMaskCannotSplit { time: 0, state: 0 })
        );
        assert_eq!(
            mask.split_on_action(&family, 3, 0, 0, policy_mask_budget()),
            Err(PolicyPortfolioError::PolicyMaskSlotIndexOutOfBounds {
                time: 3,
                state: 0,
                horizon: 3,
                state_count: 1,
            })
        );

        let two_actions = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
        ]).unwrap();
        let fixed = FiniteUmdpPolicyActionMask::new(
            &two_actions, vec![vec![vec![true, false]]],
        ).unwrap();
        assert_eq!(
            fixed.split_on_action(&two_actions, 0, 0, 1, policy_mask_budget()),
            Err(PolicyPortfolioError::PolicyMaskActionNotAllowed {
                time: 0,
                state: 0,
                action: 1,
            })
        );
    }

    #[test]
    fn exact_cluster_policy_bnb_proves_specialist_minimax_regret_and_replays() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();
        let result = solve_exact_cluster_policy_bnb(
            &family, &[0, 1], q(1,2), cluster_search_budget(),
        ).unwrap();
        assert_eq!(result.policy_index, 0);
        assert_eq!(result.policy_actions, vec![0]);
        assert_eq!(result.model_optimal_values, vec![q(1,1), q(1,1)]);
        assert_eq!(result.model_policy_values, vec![q(1,1), q(0,1)]);
        assert_eq!(result.model_regrets, vec![q(0,1), q(1,1)]);
        assert_eq!(result.worst_case_model_indices, vec![1]);
        assert_eq!(result.minimax_regret, q(1,1));
        assert_eq!(result.search_nodes, 3);
        assert_eq!(result.pruned_nodes, 2);
        assert_eq!(result.dp_multiply_adds, 54);
        assert_eq!(
            policy_portfolio_sha256_hex(result.certificate_sha256),
            "dc24a2ac4e0c89b98609a3e3a8fd82847e949947c375e642617410144ac237e6"
        );
        verify_exact_cluster_policy_bnb(&family, cluster_search_budget(), &result).unwrap();

        let mut mutated = result.clone();
        mutated.model_regrets[1] = q(0,1);
        assert_eq!(
            verify_exact_cluster_policy_bnb(&family, cluster_search_budget(), &mutated),
            Err(PolicyPortfolioError::PolicyMaskCertificateMismatch)
        );
    }

    #[test]
    fn exact_cluster_policy_bnb_discovers_better_policy_than_initial_incumbent() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[0, 1]),
        ]).unwrap();
        let result = solve_exact_cluster_policy_bnb(
            &family, &[0], q(1,2), cluster_search_budget(),
        ).unwrap();
        assert_eq!(result.policy_index, 1);
        assert_eq!(result.policy_actions, vec![1]);
        assert_eq!(result.model_optimal_values, vec![q(1,1)]);
        assert_eq!(result.model_policy_values, vec![q(1,1)]);
        assert_eq!(result.model_regrets, vec![q(0,1)]);
        assert_eq!(result.minimax_regret, q(0,1));
        assert_eq!(result.search_nodes, 3);
        assert_eq!(result.pruned_nodes, 1);
        assert_eq!(result.dp_multiply_adds, 27);
        verify_exact_cluster_policy_bnb(&family, cluster_search_budget(), &result).unwrap();
    }

    #[test]
    fn exact_cluster_policy_bnb_fails_closed_on_search_work_and_frontier_caps() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();

        let mut node_tight = cluster_search_budget();
        node_tight.max_search_nodes = 1;
        assert_eq!(
            solve_exact_cluster_policy_bnb(&family, &[0,1], q(1,2), node_tight),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "cluster B&B search nodes",
                required: 2,
                limit: 1,
            })
        );

        let mut work_tight = cluster_search_budget();
        work_tight.max_total_dp_multiply_adds = 21;
        assert_eq!(
            solve_exact_cluster_policy_bnb(&family, &[0,1], q(1,2), work_tight),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "cluster B&B total DP multiply-add work",
                required: 22,
                limit: 21,
            })
        );

        let mut frontier_tight = cluster_search_budget();
        frontier_tight.max_frontier_masks = 1;
        assert_eq!(
            solve_exact_cluster_policy_bnb(&family, &[0,1], q(1,2), frontier_tight),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "cluster B&B frontier masks",
                required: 2,
                limit: 1,
            })
        );
    }

    #[test]
    fn invalid_initial_and_transition_probabilities_are_rejected() {
        let invalid_initial = FiniteHorizonUmdpModel::new(
            vec![q(-1, 2), q(3, 2)],
            vec![vec![vec![q(0, 1)], vec![q(0, 1)]]],
            vec![],
        );
        assert_eq!(
            invalid_initial,
            Err(PolicyPortfolioError::InvalidInitialProbability { state: 0 })
        );

        let initial_not_normalized = FiniteHorizonUmdpModel::new(
            vec![q(1, 2), q(1, 4)],
            vec![vec![vec![q(0, 1)], vec![q(0, 1)]]],
            vec![],
        );
        assert_eq!(
            initial_not_normalized,
            Err(PolicyPortfolioError::InitialDistributionDoesNotSumToOne)
        );

        let negative_transition = FiniteHorizonUmdpModel::new(
            vec![q(1, 1)],
            vec![
                vec![vec![q(0, 1)]],
                vec![vec![q(0, 1)]],
            ],
            vec![vec![vec![vec![q(-1, 1)]]]],
        );
        assert_eq!(
            negative_transition,
            Err(PolicyPortfolioError::InvalidTransitionProbability {
                time: 0,
                state: 0,
                action: 0,
                destination: 0,
            })
        );
    }

    #[test]
    fn action_shapes_and_family_initial_distributions_are_checked() {
        let inconsistent_rewards = FiniteHorizonUmdpModel::new(
            vec![q(1, 1), q(0, 1)],
            vec![vec![vec![q(0, 1)], vec![q(0, 1), q(1, 1)]]],
            vec![],
        );
        assert_eq!(
            inconsistent_rewards,
            Err(PolicyPortfolioError::RewardActionCountMismatch {
                time: 0,
                state: 1,
                expected: 1,
                actual: 2,
            })
        );

        let model_a = FiniteHorizonUmdpModel::new(
            vec![q(1, 1), q(0, 1)],
            vec![vec![vec![q(1, 1)], vec![q(0, 1)]]],
            vec![],
        ).unwrap();
        let model_b = FiniteHorizonUmdpModel::new(
            vec![q(0, 1), q(1, 1)],
            vec![vec![vec![q(0, 1)], vec![q(1, 1)]]],
            vec![],
        ).unwrap();
        assert_eq!(
            FiniteUmdpModelFamily::new(vec![model_a, model_b]),
            Err(PolicyPortfolioError::InconsistentInitialDistribution { model: 1 })
        );
    }

    #[test]
    fn policy_count_and_binomial_preflights_are_checked_and_logarithmic() {
        assert_eq!(checked_policy_count(usize::MAX, 1, 1), Ok(1));
        assert_eq!(
            checked_policy_count(usize::MAX, 2, 2),
            Err(PolicyPortfolioError::PolicySpaceOverflow)
        );
        assert_eq!(binomial(usize::MAX, 1), Ok(usize::MAX));
        assert_eq!(
            binomial(100, 50),
            Err(PolicyPortfolioError::DimensionOverflow)
        );
    }

    #[test]
    fn policy_index_decoding_and_known_answer_hashes_are_canonical() {
        let decoded = decode_deterministic_markov_policy(2, 1, 2, 2).unwrap();
        assert_eq!(decoded.actions, vec![1, 0]);
        assert_eq!(decoded.action_at(0, 0), Some(1));
        assert_eq!(decoded.action_at(0, 1), Some(0));
        assert_eq!(decoded.action_at(1, 0), None);
        assert_eq!(
            decode_deterministic_markov_policy(4, 1, 2, 2),
            Err(PolicyPortfolioError::InvalidPolicyIndex { index: 4, policy_count: 4 })
        );

        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();
        let result = solve_finite_family_policy_portfolio(
            &family, q(1, 2), 2, budget(2, 3),
        ).unwrap();
        assert_eq!(
            policy_portfolio_sha256_hex(result.policy_manifest_sha256),
            "b94e4bd36c0c1ed1c42e1de6f8bf7937a671a08c1e0b2f8b105e4a7262481b2e"
        );
        assert_eq!(
            policy_portfolio_sha256_hex(result.policy_values_sha256),
            "51fd183735d11af0356fd806260b5fe717ae67a36593054f50720abb8760796b"
        );
        assert_eq!(
            policy_portfolio_sha256_hex(result.decision_certificate_sha256),
            "22d3c82a607b2d3dc1861e7ceeb0ddd94a6e652f3deb06a68d1326e3d59b9c2c"
        );
    }

    #[test]
    fn rational_width_budget_covers_generated_dynamic_programming_intermediates() {
        let model = FiniteHorizonUmdpModel::new(
            vec![q(1, 1), q(0, 1)],
            vec![
                vec![vec![q(0, 1)], vec![q(0, 1)]],
                vec![vec![q(0, 1)], vec![q(1, 1)]],
            ],
            vec![vec![
                vec![vec![q(1, 2), q(1, 2)]],
                vec![vec![q(0, 1), q(1, 1)]],
            ]],
        ).unwrap();
        let family = FiniteUmdpModelFamily::new(vec![model]).unwrap();
        let mut tight = budget(1, 1);
        tight.max_rational_bit_width = 2;
        assert_eq!(
            solve_finite_family_policy_portfolio(&family, q(1, 2), 1, tight),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "rational bit width",
                required: 3,
                limit: 2,
            })
        );
    }

    #[test]
    fn peak_memory_and_all_nested_policy_indices_are_preflighted() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();
        let mut memory_tight = budget(2, 3);
        memory_tight.max_output_memory_bytes = 1;
        assert!(matches!(
            solve_finite_family_policy_portfolio(&family, q(1, 2), 2, memory_tight),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "estimated peak memory bytes",
                ..
            })
        ));

        let mut index_tight = budget(2, 3);
        index_tight.max_stored_policy_indices = 5;
        assert_eq!(
            solve_finite_family_policy_portfolio(&family, q(1, 2), 2, index_tight),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "stored portfolio and witness indices",
                required: 6,
                limit: 5,
            })
        );
    }

    #[test]
    fn two_specialist_models_have_zero_regret_with_two_policy_portfolio() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();
        let result = solve_finite_family_policy_portfolio(&family, q(1, 2), 2, budget(2, 3)).unwrap();
        assert_eq!(result.policy_count, 2);
        assert_eq!(result.model_policy_evaluations, 4);
        assert_eq!(result.portfolio_candidates_evaluated, 3);
        assert_eq!(result.model_optimal_values, vec![q(1, 1), q(1, 1)]);
        assert_eq!(result.minimax_regret, q(0, 1));
        assert_eq!(result.optimal_portfolios.len(), 1);
        assert_eq!(result.optimal_portfolios[0].policy_indices, vec![0, 1]);
        assert_eq!(result.optimal_portfolios[0].worst_case_model_indices, vec![0, 1]);
        assert_eq!(result.optimal_portfolios[0].model_witnesses[0].best_policy_indices, vec![0]);
        assert_eq!(result.optimal_portfolios[0].model_witnesses[1].best_policy_indices, vec![1]);
        assert!(result.enumeration_complete);
        verify_finite_family_policy_portfolio_result(&family, budget(2, 3), &result).unwrap();
    }

    #[test]
    fn singleton_regret_is_one_but_portfolio_regret_is_zero() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();
        let single = solve_finite_family_policy_portfolio(&family, q(1, 2), 1, budget(2, 2)).unwrap();
        assert_eq!(single.minimax_regret, q(1, 1));
        assert_eq!(single.optimal_portfolios.len(), 2);
        let pair = solve_finite_family_policy_portfolio(&family, q(1, 2), 2, budget(2, 3)).unwrap();
        assert_eq!(pair.minimax_regret, q(0, 1));
        assert!(pair.minimax_regret < single.minimax_regret);
    }

    #[test]
    fn all_tied_optimal_portfolios_are_preserved() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0, 1, 0]),
            one_state_model(&[0, 1, 0, 1]),
        ]).unwrap();
        let result = solve_finite_family_policy_portfolio(&family, q(1, 2), 2, budget(4, 10)).unwrap();
        assert_eq!(result.minimax_regret, q(0, 1));
        assert_eq!(result.optimal_portfolios.len(), 4);
        assert_eq!(
            result.optimal_portfolios.iter().map(|p| p.policy_indices.clone()).collect::<Vec<_>>(),
            vec![vec![0, 1], vec![0, 3], vec![1, 2], vec![2, 3]]
        );
    }

    #[test]
    fn family_identity_binds_model_order_and_replay_rejects_mutation() {
        let a = one_state_model(&[1, 0]);
        let b = one_state_model(&[0, 1]);
        let family = FiniteUmdpModelFamily::new(vec![a.clone(), b.clone()]).unwrap();
        let reversed = FiniteUmdpModelFamily::new(vec![b, a]).unwrap();
        assert_ne!(family.sha256(), reversed.sha256());
        let result = solve_finite_family_policy_portfolio(&family, q(1, 2), 2, budget(2, 3)).unwrap();
        let mut changed = result.clone();
        changed.optimal_portfolios[0].policy_indices.swap(0, 1);
        assert_eq!(
            verify_finite_family_policy_portfolio_result(&family, budget(2, 3), &changed),
            Err(PolicyPortfolioError::PortfolioCertificateMismatch)
        );
    }

    #[test]
    fn policy_and_portfolio_caps_reject_before_enumeration() {
        let family = FiniteUmdpModelFamily::new(vec![
            one_state_model(&[1, 0]),
            one_state_model(&[0, 1]),
        ]).unwrap();
        assert_eq!(
            solve_finite_family_policy_portfolio(&family, q(1, 2), 2, budget(1, 3)),
            Err(PolicyPortfolioError::BudgetExceeded { resource: "policies", required: 2, limit: 1 })
        );
        assert_eq!(
            solve_finite_family_policy_portfolio(&family, q(1, 2), 2, budget(2, 2)),
            Err(PolicyPortfolioError::BudgetExceeded { resource: "portfolio candidates", required: 3, limit: 2 })
        );
        let mut comparison_tight = budget(2, 3);
        comparison_tight.max_portfolio_member_comparisons = 7;
        assert_eq!(
            solve_finite_family_policy_portfolio(&family, q(1, 2), 2, comparison_tight),
            Err(PolicyPortfolioError::BudgetExceeded {
                resource: "portfolio member comparisons",
                required: 8,
                limit: 7,
            })
        );
    }

    #[test]
    fn malformed_stochastic_rows_and_model_dimensions_are_rejected() {
        let invalid = FiniteHorizonUmdpModel::new(
            vec![q(1, 1)],
            vec![vec![vec![q(0, 1), q(1, 1)]]],
            vec![vec![vec![vec![q(1, 2)], vec![q(1, 2)]]]],
        );
        assert!(matches!(invalid, Err(PolicyPortfolioError::TransitionRowDoesNotSumToOne { .. })));
        let a = one_state_model(&[1, 0]);
        let b = FiniteHorizonUmdpModel::new(vec![q(1, 1)], vec![vec![vec![q(1, 1)]]], vec![]).unwrap();
        assert_eq!(
            FiniteUmdpModelFamily::new(vec![a, b]),
            Err(PolicyPortfolioError::InconsistentModelDimensions { model: 1 })
        );
    }

    #[test]
    fn discount_horizon_and_duplicate_environment_controls() {
        let family = FiniteUmdpModelFamily::new(vec![one_state_model(&[1, 0])]).unwrap();
        assert_eq!(
            solve_finite_family_policy_portfolio(&family, q(3, 2), 1, budget(2, 2)),
            Err(PolicyPortfolioError::InvalidDiscount)
        );
        assert!(matches!(
            FiniteHorizonUmdpModel::new(vec![q(1, 1)], vec![], vec![]),
            Err(PolicyPortfolioError::ZeroHorizon)
        ));
        let duplicate = one_state_model(&[1, 0]);
        assert_eq!(
            FiniteUmdpModelFamily::new(vec![duplicate.clone(), duplicate]),
            Err(PolicyPortfolioError::DuplicateModelIdentity)
        );
    }

    #[test]
    fn multi_step_exact_policy_evaluation_uses_discounted_transition_value() {
        let model = FiniteHorizonUmdpModel::new(
            vec![q(1, 1), q(0, 1)],
            vec![
                vec![vec![q(0, 1)], vec![q(0, 1)]],
                vec![vec![q(0, 1)], vec![q(1, 1)]],
            ],
            vec![vec![
                vec![vec![q(1, 2), q(1, 2)]],
                vec![vec![q(0, 1), q(1, 1)]],
            ]],
        ).unwrap();
        let family = FiniteUmdpModelFamily::new(vec![model]).unwrap();
        let result = solve_finite_family_policy_portfolio(&family, q(1, 2), 1, budget(1, 1)).unwrap();
        assert_eq!(result.model_optimal_values, vec![q(1, 4)]);
        assert_eq!(result.model_policy_values, vec![vec![q(1, 4)]]);
    }

    #[test]
    fn rational_bit_width_is_preflighted() {
        let family = FiniteUmdpModelFamily::new(vec![one_state_model(&[1, 0])]).unwrap();
        let mut tight = budget(2, 2);
        tight.max_rational_bit_width = 1;
        assert_eq!(
            solve_finite_family_policy_portfolio(&family, q(1, 2), 1, tight),
            Err(PolicyPortfolioError::BudgetExceeded { resource: "rational bit width", required: 2, limit: 1 })
        );
    }
}
