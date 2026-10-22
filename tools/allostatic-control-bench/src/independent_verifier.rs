//! Independent checker for finite-horizon interval safety witnesses.
//!
//! This checker deliberately does not call the planner's transition, viability,
//! or recursive-search helpers. Given a first action plus a complete open-loop
//! continuation, it directly propagates the declared interval state through each
//! action and checks every intermediate constraint. This is a structurally
//! separate implementation over shared typed input data—not a formal proof of
//! the real plant, and not a cryptographically authenticated evidence receipt.

use super::robust_viability::{
    ActionEffectBounds, DynamicsBounds, Interval, StateEnvelope, ViabilityLimits,
};
use super::Action;
use super::multistep_viability::MAX_CERTIFIED_HORIZON;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationFailure {
    EmptySequence,
    HorizonMismatch,
    HorizonOutOfRange,
    RecoveryAuthorityMissing,
    ReserveConstraintViolated,
    LoadConstraintViolated,
    InvalidTransitionArithmetic,
}

impl VerificationFailure {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmptySequence => "empty_sequence",
            Self::HorizonMismatch => "horizon_mismatch",
            Self::HorizonOutOfRange => "horizon_out_of_range",
            Self::RecoveryAuthorityMissing => "recovery_authority_missing",
            Self::ReserveConstraintViolated => "reserve_constraint_violated",
            Self::LoadConstraintViolated => "load_constraint_violated",
            Self::InvalidTransitionArithmetic => "invalid_transition_arithmetic",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerificationReceipt {
    initial_state: StateEnvelope,
    final_state: StateEnvelope,
    steps_verified: usize,
    first_action: Action,
}

impl VerificationReceipt {
    pub const fn initial_state(self) -> StateEnvelope {
        self.initial_state
    }

    pub const fn final_state(self) -> StateEnvelope {
        self.final_state
    }

    pub const fn steps_verified(self) -> usize {
        self.steps_verified
    }

    pub const fn first_action(self) -> Action {
        self.first_action
    }
}

fn add_and_clamp(state: Interval, delta: Interval) -> Option<Interval> {
    // Recompute from primitive endpoints rather than reusing the planner's
    // interval-add / interval-clamp / transition helpers.
    let lower_sum = state.lower() + delta.lower();
    let upper_sum = state.upper() + delta.upper();
    if !lower_sum.is_finite() || !upper_sum.is_finite() || lower_sum > upper_sum {
        return None;
    }
    let clamped_lower = lower_sum.clamp(0.0, 1.0);
    let clamped_upper = upper_sum.clamp(0.0, 1.0);
    Interval::new(clamped_lower, clamped_upper)
}

fn independently_apply_effect(
    state: StateEnvelope,
    effect: ActionEffectBounds,
) -> Option<StateEnvelope> {
    let next_reserve = add_and_clamp(state.reserve(), effect.reserve_delta())?;
    let next_load = add_and_clamp(state.load(), effect.load_delta())?;
    StateEnvelope::new(next_reserve, next_load)
}

/// Verify a complete open-loop witness using direct interval propagation.
///
/// `expected_horizon` must equal the supplied sequence length and be inside the
/// supported finite-horizon range. The witness is accepted only if every
/// intermediate interval satisfies both constraints. A safe terminal state
/// cannot excuse an unsafe intermediate state. The function does not search for
/// a continuation; that work belongs to the untrusted candidate generator.
pub fn verify_sequence(
    initial_state: StateEnvelope,
    actions: &[Action],
    dynamics: DynamicsBounds,
    limits: ViabilityLimits,
    shield_can_recover: bool,
    expected_horizon: usize,
) -> Result<VerificationReceipt, VerificationFailure> {
    if expected_horizon == 0 || expected_horizon > MAX_CERTIFIED_HORIZON {
        return Err(VerificationFailure::HorizonOutOfRange);
    }
    if actions.is_empty() {
        return Err(VerificationFailure::EmptySequence);
    }
    if actions.len() != expected_horizon {
        return Err(VerificationFailure::HorizonMismatch);
    }

    let initial_reserve = initial_state.reserve();
    let initial_load = initial_state.load();
    if initial_reserve.lower() < limits.min_reserve() {
        return Err(VerificationFailure::ReserveConstraintViolated);
    }
    if initial_load.upper() > limits.max_load() {
        return Err(VerificationFailure::LoadConstraintViolated);
    }

    let mut state = initial_state;
    for action in actions {
        if *action == Action::EmergencyRecover && !shield_can_recover {
            return Err(VerificationFailure::RecoveryAuthorityMissing);
        }

        let effect = dynamics.effect_for_action(*action);
        state = independently_apply_effect(state, effect)
            .ok_or(VerificationFailure::InvalidTransitionArithmetic)?;

        // These checks occur after every transition, before the next action.
        if state.reserve().lower() < limits.min_reserve() {
            return Err(VerificationFailure::ReserveConstraintViolated);
        }
        if state.load().upper() > limits.max_load() {
            return Err(VerificationFailure::LoadConstraintViolated);
        }
    }

    Ok(VerificationReceipt {
        initial_state,
        final_state: state,
        steps_verified: actions.len(),
        first_action: actions[0],
    })
}

/// Unforgeable-by-construction permission token: the fields are private and
/// this type is emitted only after the lease and the full witness both verify.
/// The host integration should require this token at the action-application API.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerifiedActionPermit {
    action: Action,
    receipt: VerificationReceipt,
    issued_at_step: u64,
    expires_at_step: u64,
    evidence_digest: [u8; 32],
}

impl VerifiedActionPermit {
    pub const fn action(self) -> Action {
        self.action
    }

    pub const fn receipt(self) -> VerificationReceipt {
        self.receipt
    }

    pub const fn issued_at_step(self) -> u64 {
        self.issued_at_step
    }

    pub const fn expires_at_step(self) -> u64 {
        self.expires_at_step
    }

    pub const fn evidence_digest(self) -> [u8; 32] {
        self.evidence_digest
    }
}

/// Failure returned when a leased action is checked against its current
/// context and an independent sequence witness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeasedVerificationFailure {
    LeaseInvalid(super::certificate_lease::CertificateRejection),
    WitnessActionMismatch,
    WitnessHorizonMismatch,
    WitnessRejected(VerificationFailure),
}

/// Verify both certificate freshness and the action's complete interval witness.
/// This is the composition point between lifecycle validity and independent
/// transition checking. The caller must apply the action only on Ok(receipt).
pub fn verify_leased_witness(
    lease: super::certificate_lease::CertificateLease,
    current_context: super::certificate_lease::CertificateContext,
    initial_state: StateEnvelope,
    actions: &[Action],
    dynamics: DynamicsBounds,
    limits: ViabilityLimits,
    shield_can_recover: bool,
) -> Result<VerifiedActionPermit, LeasedVerificationFailure> {
    lease
        .validate(current_context)
        .map_err(LeasedVerificationFailure::LeaseInvalid)?;
    if actions.first().copied() != Some(lease.action()) {
        return Err(LeasedVerificationFailure::WitnessActionMismatch);
    }
    if actions.len() != lease.horizon() {
        return Err(LeasedVerificationFailure::WitnessHorizonMismatch);
    }
    let receipt = verify_sequence(
        initial_state,
        actions,
        dynamics,
        limits,
        shield_can_recover,
        lease.horizon(),
    )
    .map_err(LeasedVerificationFailure::WitnessRejected)?;

    Ok(VerifiedActionPermit {
        action: lease.action(),
        receipt,
        issued_at_step: lease.issued_at_step(),
        expires_at_step: lease.expires_at_step(),
        evidence_digest: lease.evidence_digest(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multistep_viability::evaluate_horizon_action_set;
    use crate::robust_viability::{
        evaluate_action_set, ActionEffectBounds, Interval, StateEnvelope,
    };

    fn interval(lower: f64, upper: f64) -> Interval {
        Interval::new(lower, upper).expect("valid test interval")
    }

    fn effects(reserve: (f64, f64), load: (f64, f64)) -> ActionEffectBounds {
        ActionEffectBounds::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
            .expect("valid test effects")
    }

    fn state(reserve: (f64, f64), load: (f64, f64)) -> StateEnvelope {
        StateEnvelope::new(interval(reserve.0, reserve.1), interval(load.0, load.1))
            .expect("valid test state")
    }

    fn dynamics() -> DynamicsBounds {
        DynamicsBounds::new(
            effects((-0.02, -0.01), (0.01, 0.02)),
            effects((0.02, 0.03), (-0.02, -0.01)),
            effects((0.08, 0.10), (-0.10, -0.08)),
        )
    }

    fn limits() -> ViabilityLimits {
        ViabilityLimits::new(0.15, 0.88).expect("valid limits")
    }

    #[test]
    fn independent_checker_accepts_a_safe_multistep_witness() {
        let initial = state((0.50, 0.50), (0.50, 0.50));
        let receipt = verify_sequence(
            initial,
            &[Action::Maintain, Action::Maintain, Action::Push],
            dynamics(),
            limits(),
            true,
            3,
        )
        .expect("safe full sequence");
        assert_eq!(receipt.steps_verified(), 3);
        assert_eq!(receipt.first_action(), Action::Maintain);
        assert!(receipt.final_state().reserve().lower() >= limits().min_reserve());
        assert!(receipt.final_state().load().upper() <= limits().max_load());
    }

    #[test]
    fn safe_terminal_state_does_not_excuse_unsafe_intermediate_state() {
        let initial = state((0.16, 0.16), (0.87, 0.87));
        let dynamics = DynamicsBounds::new(
            effects((-0.005, -0.005), (0.09, 0.10)),
            effects((0.02, 0.03), (0.03, 0.04)),
            effects((0.02, 0.03), (-0.30, -0.25)),
        );
        assert_eq!(
            verify_sequence(
                initial,
                &[Action::Push, Action::EmergencyRecover],
                dynamics,
                limits(),
                true,
                2,
            ),
            Err(VerificationFailure::LoadConstraintViolated)
        );
    }

    #[test]
    fn recovery_authority_is_checked_at_each_step() {
        let initial = state((0.50, 0.50), (0.50, 0.50));
        assert_eq!(
            verify_sequence(initial, &[Action::Maintain, Action::EmergencyRecover], dynamics(),
                limits(), false, 2),
            Err(VerificationFailure::RecoveryAuthorityMissing)
        );
    }

    #[test]
    fn malformed_length_and_horizon_fail_closed() {
        let initial = state((0.50, 0.50), (0.50, 0.50));
        assert_eq!(
            verify_sequence(initial, &[], dynamics(), limits(), true, 1),
            Err(VerificationFailure::EmptySequence)
        );
        assert_eq!(
            verify_sequence(initial, &[Action::Maintain], dynamics(), limits(), true, 2),
            Err(VerificationFailure::HorizonMismatch)
        );
        assert_eq!(
            verify_sequence(initial, &[Action::Maintain], dynamics(), limits(), true, 9),
            Err(VerificationFailure::HorizonOutOfRange)
        );
    }

    #[test]
    fn initial_state_must_already_satisfy_the_contract() {
        let unsafe_initial = state((0.14, 0.18), (0.80, 0.82));
        assert_eq!(
            verify_sequence(unsafe_initial, &[Action::EmergencyRecover], dynamics(),
                limits(), true, 1),
            Err(VerificationFailure::ReserveConstraintViolated)
        );
    }

    fn all_sequences(length: usize, prefix: &mut Vec<Action>, output: &mut Vec<Vec<Action>>) {
        if prefix.len() == length {
            output.push(prefix.clone());
            return;
        }
        for action in [Action::Push, Action::Maintain, Action::EmergencyRecover] {
            prefix.push(action);
            all_sequences(length, prefix, output);
            prefix.pop();
        }
    }

    #[test]
    fn planner_action_set_matches_independent_exhaustive_witness_search_on_fixture_grid() {
        let states = [
            state((0.22, 0.22), (0.80, 0.80)),
            state((0.30, 0.34), (0.70, 0.74)),
            state((0.50, 0.50), (0.50, 0.50)),
        ];
        let panels = [
            DynamicsBounds::new(
                effects((-0.02, -0.01), (0.01, 0.02)),
                effects((0.02, 0.03), (-0.02, -0.01)),
                effects((0.08, 0.10), (-0.10, -0.08)),
            ),
            DynamicsBounds::new(
                effects((-0.04, -0.02), (0.04, 0.08)),
                effects((0.01, 0.03), (-0.04, -0.01)),
                effects((0.05, 0.10), (-0.15, -0.08)),
            ),
        ];
        for initial in states {
            for model in panels {
                for authority in [false, true] {
                    for horizon in 1..=4 {
                        let planner = evaluate_horizon_action_set(
                            initial, model, limits(), authority, horizon
                        ).expect("small horizon");
                        let mut sequences = Vec::new();
                        all_sequences(horizon, &mut Vec::new(), &mut sequences);
                        for first_action in [Action::Push, Action::Maintain, Action::EmergencyRecover] {
                            let mut found = false;
                            for sequence in &sequences {
                                if sequence[0] != first_action {
                                    continue;
                                }
                                if verify_sequence(initial, sequence, model, limits(), authority, horizon).is_ok() {
                                    found = true;
                                    break;
                                }
                            }
                            assert_eq!(
                                planner.contains(first_action),
                                found,
                                "independent mismatch: horizon={horizon} authority={authority} first={first_action:?}"
                            );
                        }
                    }
                    // One-step evaluator is a second candidate, checked independently.
                    let one_step = evaluate_action_set(initial, model, limits(), authority);
                    let independent: Vec<bool> = [Action::Push, Action::Maintain, Action::EmergencyRecover]
                        .iter()
                        .map(|action| {
                            verify_sequence(initial, &[*action], model, limits(), authority, 1).is_ok()
                        })
                        .collect();
                    assert_eq!(one_step.push(), independent[0]);
                    assert_eq!(one_step.maintain(), independent[1]);
                    assert_eq!(one_step.emergency_recover(), independent[2]);
                }
            }
        }
    }
}
