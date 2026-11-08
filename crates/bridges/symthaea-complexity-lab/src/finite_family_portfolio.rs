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

        if transitions.len() + 1 != rewards.len() {
            return Err(PolicyPortfolioError::TransitionHorizonMismatch {
                expected: rewards.len() - 1,
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
        for model in &models {
            let digest = model.sha256();
            if model_digests.contains(&digest) {
                return Err(PolicyPortfolioError::DuplicateModelIdentity);
            }
            model_digests.push(digest);
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
    let mut count = 1usize;
    for _ in 0..exponent {
        count = count.checked_mul(actions).ok_or(PolicyPortfolioError::PolicySpaceOverflow)?;
    }
    Ok(count)
}

fn binomial(n: usize, k: usize) -> Result<usize, PolicyPortfolioError> {
    if k > n { return Ok(0); }
    let k = k.min(n - k);
    let mut value = 1u128;
    for i in 1..=k {
        let numerator = (n - k + i) as u128;
        let denominator = i as u128;
        let product = value.checked_mul(numerator).ok_or(PolicyPortfolioError::DimensionOverflow)?;
        if product % denominator != 0 {
            return Err(PolicyPortfolioError::DimensionOverflow);
        }
        value = product / denominator;
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

    let policy_ops = horizon.checked_mul(states).and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_add(horizon.saturating_sub(1).checked_mul(states)?.checked_mul(states)?.checked_mul(2)?))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let optimal_ops = horizon.checked_mul(states).and_then(|v| v.checked_mul(actions)).and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_add(horizon.saturating_sub(1).checked_mul(states)?.checked_mul(actions)?.checked_mul(states)?.checked_mul(2)?))
        .ok_or(PolicyPortfolioError::DimensionOverflow)?;
    let total_ops = evaluations.checked_mul(policy_ops)
        .and_then(|v| v.checked_add(models.checked_mul(optimal_ops)?))
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
        let mut next = current.clone();

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
                optimal_portfolios.clear();
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
