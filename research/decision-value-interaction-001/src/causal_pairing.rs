//! Causal pairing validation for the compute × observation tournament.
//!
//! This module validates counterfactual trajectory lineage after the four
//! decision-regret arms have been executed. It composes with the generic
//! open-loop/action-coupled replay distinction in Symthaea #8764. It is not a
//! simulator, a transition-model validator, a causal-identification engine,
//! or an action-authority boundary.
//!
//! Open-loop action-invariant arms may share one physical state trajectory.
//! Closed-loop action-coupled arms must carry arm-local driver executions;
//! they share the initial state and exogenous forcing commitment, but their
//! endogenous state and observation trajectories may diverge.
//!
//! A digest is a commitment to supplied evidence, not proof that the evidence
//! is physically true or complete. Independent execution isolation and replay
//! audit remain required (#9437, #8732, and the Symtropy scenario owners).

use crate::{
    InteractionError, PairedRunIdentity, TreatmentArm,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// The causal regime represented by a four-arm experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CausalReplayRegime {
    /// Actions do not alter the physical source stream. The four arms must
    /// share the same underlying physical state trajectory.
    OpenLoopActionInvariant,
    /// Actions affect future state/observations. Each arm needs an independent
    /// driver-executed trajectory under common exogenous forcing.
    ClosedLoopActionCoupled,
    /// Logged-policy evaluation is a separate estimand and is not certified
    /// by this simulator-trajectory pairing contract.
    OfflineLoggedPolicyEvaluation,
}

/// Provenance of a world trajectory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrajectoryOrigin {
    /// The named driver executed and recorded this arm's own transitions.
    ArmLocalDriverExecution,
    /// This arm reuses a recorded future world trajectory.
    SharedRecordedFutureTrajectory,
}

/// Independently named random-number domains.
///
/// The same domain commitments are shared across the four paired arms.
/// Implementations must key exogenous values by stable event/tick/channel
/// identity (or an independently frozen equivalent), not by sequential PRNG
/// draw order that can desynchronize when branches consume different counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RandomnessDomainCommitments {
    pub world_disturbance_sha256: [u8; 32],
    pub policy_rng_sha256: [u8; 32],
    pub sensor_noise_sha256: [u8; 32],
    pub evaluator_rng_sha256: [u8; 32],
}

/// Shared experiment roots. All four treatment arms must match exactly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CausalPairingIdentity {
    /// The existing 13-field paired identity from the regret evaluator.
    pub paired: PairedRunIdentity,
    /// Digest of the transition/dynamics model used for all arms.
    pub dynamics_model_sha256: [u8; 32],
    /// Digest of simulator, engine build and relevant execution profile.
    pub simulator_build_sha256: [u8; 32],
    /// Digest of the frozen schedule of exogenous disturbances.
    pub exogenous_schedule_sha256: [u8; 32],
    /// Digest of the clock and tick-to-time mapping.
    pub clock_profile_sha256: [u8; 32],
    /// First world tick in every arm-local trajectory.
    pub initial_world_tick: u64,
    /// Separate commitments for policy, world, sensor and evaluator RNG.
    pub randomness_domains: RandomnessDomainCommitments,
}

/// One committed world transition at a single integer world tick.
///
/// An empty exogenous input or observation list must still have a canonical,
/// non-zero digest for its typed empty bundle. The action digest similarly
/// binds an explicit no-op when no command is applied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrajectoryStepCommitment {
    pub world_tick: u64,
    pub pre_state_sha256: [u8; 32],
    pub applied_action_sha256: [u8; 32],
    /// Digest of the exogenous event bundle for this tick, not endogenous
    /// actor responses or the arm's resulting observation stream.
    pub exogenous_input_sha256: [u8; 32],
    pub post_state_sha256: [u8; 32],
    /// Digest of the ordered observations delivered/available to this arm
    /// for this tick, including a typed empty bundle where applicable.
    pub observation_delivery_sha256: [u8; 32],
}

/// Driver and independent replay evidence for one treatment arm.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CausalTrajectoryReceipt {
    pub identity: CausalPairingIdentity,
    pub regime: CausalReplayRegime,
    pub origin: TrajectoryOrigin,
    pub arm: TreatmentArm,
    pub steps: Vec<TrajectoryStepCommitment>,
    pub driver_execution_receipt_sha256: [u8; 32],
    pub independent_replay_audit_sha256: [u8; 32],
    pub independent_replay_audit_passed: bool,
    /// True only when the driver reports that the complete declared episode
    /// (including a terminal disposition) is represented by this receipt.
    pub complete: bool,
}

/// Successful structural pairing result. This is not a physical-fidelity or
/// causal-identification certificate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CausalPairingCertificate {
    pub regime: CausalReplayRegime,
    pub episode_id: u64,
    pub hidden_world_id: u64,
    pub exogenous_schedule_sha256: [u8; 32],
    /// Number of integer world ticks present in every arm.
    pub matched_tick_count: u64,
    /// Last state digest recorded by each arm, in canonical treatment order:
    /// baseline, compute-only, observation-only, joint.
    pub terminal_state_sha256: [[u8; 32]; 4],
    /// Origins in canonical treatment order.
    pub trajectory_origins: [TrajectoryOrigin; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CausalDigestField {
    DynamicsModel,
    SimulatorBuild,
    ExogenousSchedule,
    ClockProfile,
    WorldDisturbanceRngDomain,
    PolicyRngDomain,
    SensorNoiseRngDomain,
    EvaluatorRngDomain,
    DriverExecutionReceipt,
    IndependentReplayAudit,
    PreState,
    AppliedAction,
    ExogenousInput,
    PostState,
    ObservationDelivery,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CausalPairingError {
    WrongReceiptCount { actual: usize },
    DuplicateArm { arm: TreatmentArm },
    MissingArm { arm: TreatmentArm },
    PairedIdentity(InteractionError),
    CommonIdentityMismatch,
    RegimeMismatch,
    MixedTrajectoryOrigins,
    OfflineLoggedPolicyEvaluationUnsupported,
    SharedRecordedTrajectoryForbiddenForClosedLoop { arm: TreatmentArm },
    IncompleteReceipt { arm: TreatmentArm },
    IndependentReplayAuditFailed { arm: TreatmentArm },
    EmptyTrajectory { arm: TreatmentArm },
    EmptyDigest {
        arm: Option<TreatmentArm>,
        field: CausalDigestField,
    },
    RandomnessDomainsNotSeparated,
    InitialStateMismatch { arm: TreatmentArm },
    TickOverflow { arm: TreatmentArm },
    NonContiguousTick {
        arm: TreatmentArm,
        expected: u64,
        actual: u64,
    },
    ExogenousInputMismatch { world_tick: u64 },
    OpenLoopTrajectoryMismatch { world_tick: u64 },
    ConflictingDeterministicTransition { world_tick: u64 },
    ArithmeticOverflow,
}

impl fmt::Display for CausalPairingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongReceiptCount { actual } => write!(
                formatter,
                "expected four causal trajectory receipts, received {actual}"
            ),
            Self::DuplicateArm { arm } => {
                write!(formatter, "causal trajectory receipt duplicates arm {arm:?}")
            }
            Self::MissingArm { arm } => {
                write!(formatter, "causal trajectory receipt is missing arm {arm:?}")
            }
            Self::PairedIdentity(error) => {
                write!(formatter, "paired decision-regret identity is invalid: {error}")
            }
            Self::CommonIdentityMismatch => {
                write!(formatter, "causal experiment roots differ across treatment arms")
            }
            Self::RegimeMismatch => {
                write!(formatter, "treatment arms declare different causal replay regimes")
            }
            Self::MixedTrajectoryOrigins => {
                write!(formatter, "treatment arms mix trajectory provenance modes")
            }
            Self::OfflineLoggedPolicyEvaluationUnsupported => write!(
                formatter,
                "offline logged-policy evaluation requires its own support/estimand contract"
            ),
            Self::SharedRecordedTrajectoryForbiddenForClosedLoop { arm } => write!(
                formatter,
                "closed-loop action-coupled arm {arm:?} cannot reuse a shared recorded future trajectory"
            ),
            Self::IncompleteReceipt { arm } => {
                write!(formatter, "trajectory receipt for arm {arm:?} is incomplete")
            }
            Self::IndependentReplayAuditFailed { arm } => write!(
                formatter,
                "independent replay audit did not pass for arm {arm:?}"
            ),
            Self::EmptyTrajectory { arm } => {
                write!(formatter, "trajectory for arm {arm:?} has no steps")
            }
            Self::EmptyDigest { arm, field } => {
                write!(formatter, "empty {field:?} digest in causal pairing receipt for {arm:?}")
            }
            Self::RandomnessDomainsNotSeparated => write!(
                formatter,
                "world, policy, sensor and evaluator RNG domains must be non-empty and distinct"
            ),
            Self::InitialStateMismatch { arm } => write!(
                formatter,
                "first transition in arm {arm:?} does not start at the paired initial state"
            ),
            Self::TickOverflow { arm } => {
                write!(formatter, "world-tick arithmetic overflowed for arm {arm:?}")
            }
            Self::NonContiguousTick {
                arm,
                expected,
                actual,
            } => write!(
                formatter,
                "arm {arm:?} expected world tick {expected}, received {actual}"
            ),
            Self::ExogenousInputMismatch { world_tick } => write!(
                formatter,
                "paired arms disagree on exogenous input at world tick {world_tick}"
            ),
            Self::OpenLoopTrajectoryMismatch { world_tick } => write!(
                formatter,
                "action-invariant arms have different physical state trajectories at tick {world_tick}"
            ),
            Self::ConflictingDeterministicTransition { world_tick } => write!(
                formatter,
                "identical pre-state/action/exogenous inputs produce conflicting next states at tick {world_tick}"
            ),
            Self::ArithmeticOverflow => {
                write!(formatter, "causal pairing certificate arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for CausalPairingError {}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn arm_index(arm: TreatmentArm) -> usize {
    match arm {
        TreatmentArm::Baseline => 0,
        TreatmentArm::ComputeOnly => 1,
        TreatmentArm::ObservationOnly => 2,
        TreatmentArm::Joint => 3,
    }
}

fn require_digest(
    digest: &[u8; 32],
    arm: Option<TreatmentArm>,
    field: CausalDigestField,
) -> Result<(), CausalPairingError> {
    if digest_is_empty(digest) {
        Err(CausalPairingError::EmptyDigest { arm, field })
    } else {
        Ok(())
    }
}

fn validate_common_identity(
    identity: &CausalPairingIdentity,
) -> Result<(), CausalPairingError> {
    crate::validate_identity(&identity.paired).map_err(CausalPairingError::PairedIdentity)?;

    for (digest, field) in [
        (&identity.dynamics_model_sha256, CausalDigestField::DynamicsModel),
        (&identity.simulator_build_sha256, CausalDigestField::SimulatorBuild),
        (&identity.exogenous_schedule_sha256, CausalDigestField::ExogenousSchedule),
        (&identity.clock_profile_sha256, CausalDigestField::ClockProfile),
        (
            &identity.randomness_domains.world_disturbance_sha256,
            CausalDigestField::WorldDisturbanceRngDomain,
        ),
        (
            &identity.randomness_domains.policy_rng_sha256,
            CausalDigestField::PolicyRngDomain,
        ),
        (
            &identity.randomness_domains.sensor_noise_sha256,
            CausalDigestField::SensorNoiseRngDomain,
        ),
        (
            &identity.randomness_domains.evaluator_rng_sha256,
            CausalDigestField::EvaluatorRngDomain,
        ),
    ] {
        require_digest(digest, None, field)?;
    }

    let domains = [
        identity.randomness_domains.world_disturbance_sha256,
        identity.randomness_domains.policy_rng_sha256,
        identity.randomness_domains.sensor_noise_sha256,
        identity.randomness_domains.evaluator_rng_sha256,
    ];
    if domains.iter().copied().collect::<BTreeSet<_>>().len() != domains.len() {
        return Err(CausalPairingError::RandomnessDomainsNotSeparated);
    }
    Ok(())
}

fn validate_steps(receipt: &CausalTrajectoryReceipt) -> Result<(), CausalPairingError> {
    if receipt.steps.is_empty() {
        return Err(CausalPairingError::EmptyTrajectory { arm: receipt.arm });
    }
    if !receipt.complete {
        return Err(CausalPairingError::IncompleteReceipt { arm: receipt.arm });
    }
    if !receipt.independent_replay_audit_passed {
        return Err(CausalPairingError::IndependentReplayAuditFailed { arm: receipt.arm });
    }
    require_digest(
        &receipt.driver_execution_receipt_sha256,
        Some(receipt.arm),
        CausalDigestField::DriverExecutionReceipt,
    )?;
    require_digest(
        &receipt.independent_replay_audit_sha256,
        Some(receipt.arm),
        CausalDigestField::IndependentReplayAudit,
    )?;

    let mut expected_tick = receipt.identity.initial_world_tick;
    for (index, step) in receipt.steps.iter().enumerate() {
        if step.world_tick != expected_tick {
            return Err(CausalPairingError::NonContiguousTick {
                arm: receipt.arm,
                expected: expected_tick,
                actual: step.world_tick,
            });
        }
        for (digest, field) in [
            (&step.pre_state_sha256, CausalDigestField::PreState),
            (&step.applied_action_sha256, CausalDigestField::AppliedAction),
            (&step.exogenous_input_sha256, CausalDigestField::ExogenousInput),
            (&step.post_state_sha256, CausalDigestField::PostState),
            (
                &step.observation_delivery_sha256,
                CausalDigestField::ObservationDelivery,
            ),
        ] {
            require_digest(digest, Some(receipt.arm), field)?;
        }
        if index == 0 && step.pre_state_sha256 != receipt.identity.paired.initial_state_sha256 {
            return Err(CausalPairingError::InitialStateMismatch { arm: receipt.arm });
        }
        expected_tick = expected_tick
            .checked_add(1)
            .ok_or(CausalPairingError::TickOverflow { arm: receipt.arm })?;
    }
    Ok(())
}

/// Validate a complete set of four paired causal trajectories.
///
/// This validates structural pairing, trajectory provenance, common
/// exogenous forcing, and deterministic transition consistency. It does not
/// establish that the registered dynamics are physically accurate, that the
/// exogenous schedule is complete, or that counterfactual outcomes are
/// identified from observational data.
pub fn validate_causal_pairing(
    receipts: &[CausalTrajectoryReceipt],
) -> Result<CausalPairingCertificate, CausalPairingError> {
    if receipts.len() != 4 {
        return Err(CausalPairingError::WrongReceiptCount {
            actual: receipts.len(),
        });
    }

    let mut by_arm: BTreeMap<TreatmentArm, &CausalTrajectoryReceipt> = BTreeMap::new();
    for receipt in receipts {
        if by_arm.insert(receipt.arm, receipt).is_some() {
            return Err(CausalPairingError::DuplicateArm { arm: receipt.arm });
        }
    }

    let ordered_arms = [
        TreatmentArm::Baseline,
        TreatmentArm::ComputeOnly,
        TreatmentArm::ObservationOnly,
        TreatmentArm::Joint,
    ];
    let mut ordered = Vec::with_capacity(4);
    for arm in ordered_arms {
        ordered.push(
            by_arm
                .get(&arm)
                .copied()
                .ok_or(CausalPairingError::MissingArm { arm })?,
        );
    }

    let baseline = ordered[0];
    validate_common_identity(&baseline.identity)?;
    let regime = baseline.regime;
    if regime == CausalReplayRegime::OfflineLoggedPolicyEvaluation {
        return Err(CausalPairingError::OfflineLoggedPolicyEvaluationUnsupported);
    }

    for receipt in &ordered {
        validate_common_identity(&receipt.identity)?;
        crate::validate_pair(&baseline.identity.paired, &receipt.identity.paired)
            .map_err(CausalPairingError::PairedIdentity)?;
        if receipt.identity != baseline.identity {
            return Err(CausalPairingError::CommonIdentityMismatch);
        }
        if receipt.regime != regime {
            return Err(CausalPairingError::RegimeMismatch);
        }
        if receipt.origin != baseline.origin {
            return Err(CausalPairingError::MixedTrajectoryOrigins);
        }
        if regime == CausalReplayRegime::ClosedLoopActionCoupled
            && receipt.origin == TrajectoryOrigin::SharedRecordedFutureTrajectory
        {
            return Err(
                CausalPairingError::SharedRecordedTrajectoryForbiddenForClosedLoop {
                    arm: receipt.arm,
                },
            );
        }
        validate_steps(receipt)?;
    }

    // All arms begin at the same initial world tick. Every overlapping tick
    // must receive the same exogenous event bundle, even when endogenous state
    // and observations diverge.
    let common_tick_count = ordered
        .iter()
        .map(|receipt| receipt.steps.len())
        .min()
        .ok_or(CausalPairingError::ArithmeticOverflow)?;
    let matched_tick_count =
        u64::try_from(common_tick_count).map_err(|_| CausalPairingError::ArithmeticOverflow)?;

    // A fully specified transition input must have one deterministic output.
    // This also catches accidental RNG stream desynchronization when the same
    // keyed exogenous event is presented with the same state and action.
    let mut transition_outputs: BTreeMap<
        (u64, [u8; 32], [u8; 32], [u8; 32]),
        [u8; 32],
    > = BTreeMap::new();

    let baseline_steps: BTreeMap<u64, &TrajectoryStepCommitment> = baseline
        .steps
        .iter()
        .map(|step| (step.world_tick, step))
        .collect();

    for receipt in &ordered {
        for step in &receipt.steps {
            let baseline_step = baseline_steps.get(&step.world_tick).copied();
            if let Some(reference) = baseline_step {
                if step.exogenous_input_sha256 != reference.exogenous_input_sha256 {
                    return Err(CausalPairingError::ExogenousInputMismatch {
                        world_tick: step.world_tick,
                    });
                }
            }

            let key = (
                step.world_tick,
                step.pre_state_sha256,
                step.applied_action_sha256,
                step.exogenous_input_sha256,
            );
            if let Some(previous_output) = transition_outputs.insert(key, step.post_state_sha256) {
                if previous_output != step.post_state_sha256 {
                    return Err(CausalPairingError::ConflictingDeterministicTransition {
                        world_tick: step.world_tick,
                    });
                }
            }
        }
    }

    if regime == CausalReplayRegime::OpenLoopActionInvariant {
        let reference_steps = &baseline.steps;
        for receipt in &ordered[1..] {
            if receipt.steps.len() != reference_steps.len() {
                return Err(CausalPairingError::OpenLoopTrajectoryMismatch {
                    world_tick: baseline.identity.initial_world_tick,
                });
            }
        }
        for index in 0..reference_steps.len() {
            let reference = &reference_steps[index];
            for receipt in &ordered[1..] {
                let candidate = &receipt.steps[index];
                if candidate.world_tick != reference.world_tick
                    || candidate.pre_state_sha256 != reference.pre_state_sha256
                    || candidate.post_state_sha256 != reference.post_state_sha256
                    || candidate.exogenous_input_sha256 != reference.exogenous_input_sha256
                {
                    return Err(CausalPairingError::OpenLoopTrajectoryMismatch {
                        world_tick: reference.world_tick,
                    });
                }
            }
        }
    }

    let terminal_state_sha256 = [
        ordered[0].steps.last().map(|step| step.post_state_sha256),
        ordered[1].steps.last().map(|step| step.post_state_sha256),
        ordered[2].steps.last().map(|step| step.post_state_sha256),
        ordered[3].steps.last().map(|step| step.post_state_sha256),
    ]
    .map(|digest| digest.ok_or(CausalPairingError::ArithmeticOverflow))
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?
    .try_into()
    .map_err(|_| CausalPairingError::ArithmeticOverflow)?;

    let trajectory_origins = [
        ordered[0].origin,
        ordered[1].origin,
        ordered[2].origin,
        ordered[3].origin,
    ];

    Ok(CausalPairingCertificate {
        regime,
        episode_id: baseline.identity.paired.episode_id,
        hidden_world_id: baseline.identity.paired.hidden_world_id,
        exogenous_schedule_sha256: baseline.identity.exogenous_schedule_sha256,
        matched_tick_count,
        terminal_state_sha256,
        trajectory_origins,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn paired_identity() -> PairedRunIdentity {
        PairedRunIdentity {
            episode_id: 41,
            hidden_world_id: 7,
            hidden_world_sha256: digest(1),
            initial_state_sha256: digest(2),
            task_sha256: digest(3),
            loss_scale_sha256: digest(4),
            evaluator_profile_sha256: digest(5),
            execution_environment_sha256: digest(6),
            action_domain_sha256: digest(7),
            resource_budget_sha256: digest(8),
            model_configuration_sha256: digest(9),
            candidate_policy_sha256: digest(10),
            policy_seed: 99,
        }
    }

    fn identity() -> CausalPairingIdentity {
        CausalPairingIdentity {
            paired: paired_identity(),
            dynamics_model_sha256: digest(11),
            simulator_build_sha256: digest(12),
            exogenous_schedule_sha256: digest(13),
            clock_profile_sha256: digest(14),
            initial_world_tick: 100,
            randomness_domains: RandomnessDomainCommitments {
                world_disturbance_sha256: digest(15),
                policy_rng_sha256: digest(16),
                sensor_noise_sha256: digest(17),
                evaluator_rng_sha256: digest(18),
            },
        }
    }

    fn step(
        tick: u64,
        pre: u8,
        action: u8,
        exogenous: u8,
        post: u8,
        observation: u8,
    ) -> TrajectoryStepCommitment {
        TrajectoryStepCommitment {
            world_tick: tick,
            pre_state_sha256: digest(pre),
            applied_action_sha256: digest(action),
            exogenous_input_sha256: digest(exogenous),
            post_state_sha256: digest(post),
            observation_delivery_sha256: digest(observation),
        }
    }

    fn receipts(
        regime: CausalReplayRegime,
        origin: TrajectoryOrigin,
        divergent: bool,
    ) -> Vec<CausalTrajectoryReceipt> {
        let arms = [
            TreatmentArm::Baseline,
            TreatmentArm::ComputeOnly,
            TreatmentArm::ObservationOnly,
            TreatmentArm::Joint,
        ];
        arms
            .into_iter()
            .enumerate()
            .map(|(index, arm)| {
                let branch = if divergent { index as u8 } else { 0 };
                let first_post = if divergent { 30 + branch } else { 30 };
                let second_post = if divergent { 40 + branch } else { 40 };
                CausalTrajectoryReceipt {
                    identity: identity(),
                    regime,
                    origin,
                    arm,
                    steps: vec![
                        step(100, 2, 20 + index as u8, 50, first_post, 60 + index as u8),
                        step(
                            101,
                            first_post,
                            70 + index as u8,
                            51,
                            second_post,
                            80 + index as u8,
                        ),
                    ],
                    driver_execution_receipt_sha256: digest(90 + index as u8),
                    independent_replay_audit_sha256: digest(100 + index as u8),
                    independent_replay_audit_passed: true,
                    complete: true,
                }
            })
            .collect()
    }

    #[test]
    fn closed_loop_accepts_arm_local_divergent_trajectories() {
        let runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        let certificate = validate_causal_pairing(&runs).expect("valid paired causal arms");
        assert_eq!(certificate.matched_tick_count, 2);
        assert_ne!(certificate.terminal_state_sha256[0], certificate.terminal_state_sha256[1]);
    }

    #[test]
    fn closed_loop_rejects_shared_recorded_future_trajectory() {
        let runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::SharedRecordedFutureTrajectory,
            false,
        );
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::SharedRecordedTrajectoryForbiddenForClosedLoop {
                arm: TreatmentArm::Baseline
            })
        ));
    }

    #[test]
    fn open_loop_accepts_shared_underlying_state_trajectory() {
        let runs = receipts(
            CausalReplayRegime::OpenLoopActionInvariant,
            TrajectoryOrigin::SharedRecordedFutureTrajectory,
            false,
        );
        let certificate = validate_causal_pairing(&runs).expect("valid action-invariant arms");
        assert_eq!(certificate.matched_tick_count, 2);
        assert!(certificate
            .terminal_state_sha256
            .iter()
            .all(|state| *state == digest(40)));
    }

    #[test]
    fn closed_loop_null_case_may_have_identical_trajectories() {
        let runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            false,
        );
        let certificate = validate_causal_pairing(&runs).expect("valid null causal case");
        assert!(certificate
            .terminal_state_sha256
            .iter()
            .all(|state| *state == digest(40)));
    }

    #[test]
    fn open_loop_rejects_physical_state_divergence() {
        let mut runs = receipts(
            CausalReplayRegime::OpenLoopActionInvariant,
            TrajectoryOrigin::ArmLocalDriverExecution,
            false,
        );
        runs[1].steps[0].post_state_sha256 = digest(77);
        runs[1].steps[1].pre_state_sha256 = digest(77);
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::OpenLoopTrajectoryMismatch { world_tick: 100 })
        ));
    }

    #[test]
    fn exogenous_forcing_must_match_on_every_overlapping_tick() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[2].steps[1].exogenous_input_sha256 = digest(222);
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::ExogenousInputMismatch { world_tick: 101 })
        ));
    }

    #[test]
    fn each_arm_must_start_from_the_paired_initial_state() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[3].steps[0].pre_state_sha256 = digest(199);
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::InitialStateMismatch {
                arm: TreatmentArm::Joint
            })
        ));
    }

    #[test]
    fn each_arm_trajectory_must_be_contiguous() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[1].steps[1].world_tick = 104;
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::NonContiguousTick {
                arm: TreatmentArm::ComputeOnly,
                expected: 101,
                actual: 104
            })
        ));
    }

    #[test]
    fn paired_identity_mismatch_is_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[1].identity.paired.initial_state_sha256 = digest(211);
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::PairedIdentity(
                InteractionError::IdentityMismatch {
                    field: crate::IdentityField::InitialStateDigest
                }
            ))
        ));
    }

    #[test]
    fn differing_rng_domains_are_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[0].identity.randomness_domains.sensor_noise_sha256 = digest(15);
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::RandomnessDomainsNotSeparated)
        ));
    }

    #[test]
    fn offline_logged_policy_evaluation_is_not_mislabelled_as_simulator_pairing() {
        let runs = receipts(
            CausalReplayRegime::OfflineLoggedPolicyEvaluation,
            TrajectoryOrigin::SharedRecordedFutureTrajectory,
            false,
        );
        assert_eq!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::OfflineLoggedPolicyEvaluationUnsupported)
        );
    }

    #[test]
    fn failed_replay_audit_is_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[0].independent_replay_audit_passed = false;
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::IndependentReplayAuditFailed {
                arm: TreatmentArm::Baseline
            })
        ));
    }

    #[test]
    fn truncated_trajectory_is_not_certified_complete() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[0].complete = false;
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::IncompleteReceipt {
                arm: TreatmentArm::Baseline
            })
        ));
    }

    #[test]
    fn repeated_tick_and_state_action_input_cannot_have_two_outputs() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            false,
        );
        runs[1].steps[0].applied_action_sha256 = runs[0].steps[0].applied_action_sha256;
        runs[1].steps[0].post_state_sha256 = digest(222);
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::ConflictingDeterministicTransition { world_tick: 100 })
        ));
    }

    #[test]
    fn mixed_trajectory_origins_are_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::OpenLoopActionInvariant,
            TrajectoryOrigin::ArmLocalDriverExecution,
            false,
        );
        runs[3].origin = TrajectoryOrigin::SharedRecordedFutureTrajectory;
        assert_eq!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::MixedTrajectoryOrigins)
        );
    }
}
