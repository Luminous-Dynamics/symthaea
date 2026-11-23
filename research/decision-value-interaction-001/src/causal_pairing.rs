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
    ApplicationDisposition, DecisionValueInteractionCertificate, InteractionError,
    NonApplicationReason, PairedRunIdentity, TreatmentArm,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

type TransitionInputKey = (u64, [u8; 32], [u8; 32], [u8; 32]);
type TransitionOutputDigest = [u8; 32];

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

/// One evidence item actually delivered to a policy before a decision.
///
/// `available_at_us` is when the evidence became available to the candidate
/// under the registered clock profile; `delivered_at_us` is when the driver
/// made it visible to this arm. The event ID is a stable opaque identifier.
/// Payload hashing and canonical event bytes remain owned by the generic
/// receipt/artifact layer (#9437/#8715).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeliveredEvidenceRef {
    pub event_id_sha256: [u8; 32],
    pub available_at_us: u64,
    pub delivered_at_us: u64,
}

/// One committed world transition at a single integer world tick.
///
/// An empty exogenous input or observation list must still have a canonical,
/// non-zero digest for its typed empty bundle. The action digest similarly
/// binds an explicit no-op when no command is applied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrajectoryStepCommitment {
    pub world_tick: u64,
    /// Driver-clock time when this arm committed the decision/action proposal.
    pub decision_at_us: u64,
    /// Driver-clock time when the proposed action was actually applied.
    pub action_applied_at_us: u64,
    /// Evidence made visible to this arm at or before this decision.
    pub delivered_evidence: Vec<DeliveredEvidenceRef>,
    /// Evidence IDs the policy claims to have consumed for this decision.
    /// Every ID must occur in this arm's delivered prefix, including earlier
    /// ticks; availability after decision time is never admissible.
    pub consumed_evidence_ids: Vec<[u8; 32]>,
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

/// Which action/outcome from this trajectory is scored by the paired regret
/// evaluator. This prevents a valid causal trace from being paired with a
/// regret value from a different action or terminal disposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoredOutcomeBinding {
    Applied {
        world_tick: u64,
        action_sha256: [u8; 32],
        applied_at_us: u64,
    },
    NotApplied {
        terminal_at_us: u64,
        reason: NonApplicationReason,
    },
}

/// Driver and independent replay evidence for one treatment arm.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CausalTrajectoryReceipt {
    pub identity: CausalPairingIdentity,
    pub regime: CausalReplayRegime,
    pub origin: TrajectoryOrigin,
    pub arm: TreatmentArm,
    pub steps: Vec<TrajectoryStepCommitment>,
    pub scored_outcome: ScoredOutcomeBinding,
    pub driver_execution_receipt_sha256: [u8; 32],
    pub independent_replay_audit_sha256: [u8; 32],
    pub independent_replay_audit_passed: bool,
    /// Digest of the complete Symtropy correspondence-conditioned stochastic
    /// draw bundle (hypothesis-scoped semantic addresses plus Philox receipts).
    /// The referenced bundle must be independently verified by the existing
    /// Symtropy correspondence-receipt verifier; this module does not resolve
    /// PRNG outputs or infer event correspondence.
    pub stochastic_draw_bundle_sha256: [u8; 32],
    /// True only when the driver reports that the complete declared episode
    /// (including a terminal disposition and complete draw correspondence set)
    /// is represented by this receipt.
    pub complete: bool,
}

/// Successful structural pairing result. This is not a physical-fidelity or
/// causal-identification certificate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CausalPairingCertificate {
    pub regime: CausalReplayRegime,
    /// The full 13-field pairing root used by all four arms.
    pub paired_run_identity: PairedRunIdentity,
    /// Scored outcome bindings in baseline, compute, observation, joint order.
    pub scored_outcome_bindings: [ScoredOutcomeBinding; 4],
    /// Number of decisions checked across all four arms.
    pub validated_decision_count: u64,
    /// Number of evidence references consumed across all four arms.
    pub consumed_evidence_reference_count: u64,
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
    EvidenceEventIdentity,
    StochasticDrawBundle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CausalPairingError {
    WrongReceiptCount {
        actual: usize,
    },
    DuplicateArm {
        arm: TreatmentArm,
    },
    MissingArm {
        arm: TreatmentArm,
    },
    PairedIdentity(InteractionError),
    CommonIdentityMismatch,
    RegimeMismatch,
    MixedTrajectoryOrigins,
    OfflineLoggedPolicyEvaluationUnsupported,
    SharedRecordedTrajectoryForbiddenForClosedLoop {
        arm: TreatmentArm,
    },
    IncompleteReceipt {
        arm: TreatmentArm,
    },
    IndependentReplayAuditFailed {
        arm: TreatmentArm,
    },
    EmptyTrajectory {
        arm: TreatmentArm,
    },
    EmptyDigest {
        arm: Option<TreatmentArm>,
        field: CausalDigestField,
    },
    RandomnessDomainsNotSeparated,
    InitialStateMismatch {
        arm: TreatmentArm,
    },
    StateChainMismatch {
        arm: TreatmentArm,
        world_tick: u64,
    },
    ScoredActionTickMissing {
        arm: TreatmentArm,
        world_tick: u64,
    },
    ScoredActionDigestMismatch {
        arm: TreatmentArm,
        world_tick: u64,
    },
    ScoredActionTimeMismatch {
        arm: TreatmentArm,
        expected_us: u64,
        actual_us: u64,
    },
    ScoredTerminalBeforeDecision {
        arm: TreatmentArm,
        terminal_at_us: u64,
        last_decision_at_us: u64,
    },
    ScoredOutcomeDispositionMismatch {
        arm: TreatmentArm,
    },
    ScoredOutcomeIdentityMismatch {
        field: crate::IdentityField,
    },
    DecisionValueInteraction(InteractionError),
    TickOverflow {
        arm: TreatmentArm,
    },
    DecisionClockRegression {
        arm: TreatmentArm,
        previous_us: u64,
        current_us: u64,
    },
    ActionAppliedBeforeDecision {
        arm: TreatmentArm,
        decision_at_us: u64,
        action_applied_at_us: u64,
    },
    EvidenceAvailableAfterDelivery {
        arm: TreatmentArm,
        event_id_sha256: [u8; 32],
    },
    EvidenceDeliveredAfterDecision {
        arm: TreatmentArm,
        event_id_sha256: [u8; 32],
    },
    DuplicateEvidenceDelivery {
        arm: TreatmentArm,
        event_id_sha256: [u8; 32],
    },
    DuplicateConsumedEvidence {
        arm: TreatmentArm,
        event_id_sha256: [u8; 32],
    },
    ConsumedEvidenceNotDelivered {
        arm: TreatmentArm,
        event_id_sha256: [u8; 32],
    },
    NonContiguousTick {
        arm: TreatmentArm,
        expected: u64,
        actual: u64,
    },
    ExogenousInputMismatch {
        world_tick: u64,
    },
    OpenLoopTrajectoryMismatch {
        world_tick: u64,
    },
    ConflictingDeterministicTransition {
        world_tick: u64,
    },
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
                write!(
                    formatter,
                    "causal trajectory receipt duplicates arm {arm:?}"
                )
            }
            Self::MissingArm { arm } => {
                write!(
                    formatter,
                    "causal trajectory receipt is missing arm {arm:?}"
                )
            }
            Self::PairedIdentity(error) => {
                write!(
                    formatter,
                    "paired decision-regret identity is invalid: {error}"
                )
            }
            Self::CommonIdentityMismatch => {
                write!(
                    formatter,
                    "causal experiment roots differ across treatment arms"
                )
            }
            Self::RegimeMismatch => {
                write!(
                    formatter,
                    "treatment arms declare different causal replay regimes"
                )
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
                write!(
                    formatter,
                    "trajectory receipt for arm {arm:?} is incomplete"
                )
            }
            Self::IndependentReplayAuditFailed { arm } => write!(
                formatter,
                "independent replay audit did not pass for arm {arm:?}"
            ),
            Self::EmptyTrajectory { arm } => {
                write!(formatter, "trajectory for arm {arm:?} has no steps")
            }
            Self::EmptyDigest { arm, field } => {
                write!(
                    formatter,
                    "empty {field:?} digest in causal pairing receipt for {arm:?}"
                )
            }
            Self::RandomnessDomainsNotSeparated => write!(
                formatter,
                "world, policy, sensor and evaluator RNG domains must be non-empty and distinct"
            ),
            Self::InitialStateMismatch { arm } => write!(
                formatter,
                "first transition in arm {arm:?} does not start at the paired initial state"
            ),
            Self::StateChainMismatch { arm, world_tick } => write!(
                formatter,
                "arm {arm:?} has a discontinuous world-state chain at tick {world_tick}"
            ),
            Self::ScoredActionTickMissing { arm, world_tick } => write!(
                formatter,
                "scored action for arm {arm:?} references missing world tick {world_tick}"
            ),
            Self::ScoredActionDigestMismatch { arm, world_tick } => write!(
                formatter,
                "scored action digest for arm {arm:?} does not match its trajectory at tick {world_tick}"
            ),
            Self::ScoredActionTimeMismatch {
                arm,
                expected_us,
                actual_us,
            } => write!(
                formatter,
                "scored action time for arm {arm:?} is {actual_us}, trajectory records {expected_us}"
            ),
            Self::ScoredTerminalBeforeDecision {
                arm,
                terminal_at_us,
                last_decision_at_us,
            } => write!(
                formatter,
                "terminal disposition for arm {arm:?} at {terminal_at_us} precedes final decision at {last_decision_at_us}"
            ),
            Self::ScoredOutcomeDispositionMismatch { arm } => write!(
                formatter,
                "scored outcome disposition for arm {arm:?} differs from its paired regret outcome"
            ),
            Self::ScoredOutcomeIdentityMismatch { field } => write!(
                formatter,
                "scored regret outcome does not match causal trajectory identity field {field:?}"
            ),
            Self::DecisionValueInteraction(error) => {
                write!(
                    formatter,
                    "paired decision-regret evaluation failed: {error}"
                )
            }
            Self::TickOverflow { arm } => {
                write!(
                    formatter,
                    "world-tick arithmetic overflowed for arm {arm:?}"
                )
            }
            Self::DecisionClockRegression {
                arm,
                previous_us,
                current_us,
            } => write!(
                formatter,
                "arm {arm:?} decision clock regressed from {previous_us} to {current_us} microseconds"
            ),
            Self::ActionAppliedBeforeDecision {
                arm,
                decision_at_us,
                action_applied_at_us,
            } => write!(
                formatter,
                "arm {arm:?} action applied at {action_applied_at_us} before decision at {decision_at_us} microseconds"
            ),
            Self::EvidenceAvailableAfterDelivery {
                arm,
                event_id_sha256,
            } => write!(
                formatter,
                "arm {arm:?} evidence {event_id_sha256:02x?} was delivered before it became available"
            ),
            Self::EvidenceDeliveredAfterDecision {
                arm,
                event_id_sha256,
            } => write!(
                formatter,
                "arm {arm:?} evidence {event_id_sha256:02x?} was delivered after the decision"
            ),
            Self::DuplicateEvidenceDelivery {
                arm,
                event_id_sha256,
            } => write!(
                formatter,
                "arm {arm:?} evidence {event_id_sha256:02x?} was delivered more than once"
            ),
            Self::DuplicateConsumedEvidence {
                arm,
                event_id_sha256,
            } => write!(
                formatter,
                "arm {arm:?} decision references evidence {event_id_sha256:02x?} more than once"
            ),
            Self::ConsumedEvidenceNotDelivered {
                arm,
                event_id_sha256,
            } => write!(
                formatter,
                "arm {arm:?} consumed evidence {event_id_sha256:02x?} outside its delivered prefix"
            ),
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
                write!(
                    formatter,
                    "causal pairing certificate arithmetic overflowed"
                )
            }
        }
    }
}

impl std::error::Error for CausalPairingError {}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
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

fn validate_common_identity(identity: &CausalPairingIdentity) -> Result<(), CausalPairingError> {
    crate::validate_identity(&identity.paired).map_err(CausalPairingError::PairedIdentity)?;

    for (digest, field) in [
        (
            &identity.dynamics_model_sha256,
            CausalDigestField::DynamicsModel,
        ),
        (
            &identity.simulator_build_sha256,
            CausalDigestField::SimulatorBuild,
        ),
        (
            &identity.exogenous_schedule_sha256,
            CausalDigestField::ExogenousSchedule,
        ),
        (
            &identity.clock_profile_sha256,
            CausalDigestField::ClockProfile,
        ),
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct StepValidationSummary {
    decision_count: u64,
    consumed_evidence_reference_count: u64,
}

fn validate_scored_outcome_binding(
    receipt: &CausalTrajectoryReceipt,
) -> Result<(), CausalPairingError> {
    match receipt.scored_outcome {
        ScoredOutcomeBinding::Applied {
            world_tick,
            action_sha256,
            applied_at_us,
        } => {
            let step = receipt
                .steps
                .iter()
                .find(|step| step.world_tick == world_tick)
                .ok_or(CausalPairingError::ScoredActionTickMissing {
                    arm: receipt.arm,
                    world_tick,
                })?;
            if step.applied_action_sha256 != action_sha256 {
                return Err(CausalPairingError::ScoredActionDigestMismatch {
                    arm: receipt.arm,
                    world_tick,
                });
            }
            if step.action_applied_at_us != applied_at_us {
                return Err(CausalPairingError::ScoredActionTimeMismatch {
                    arm: receipt.arm,
                    expected_us: step.action_applied_at_us,
                    actual_us: applied_at_us,
                });
            }
        }
        ScoredOutcomeBinding::NotApplied { terminal_at_us, .. } => {
            let last_decision_at_us = receipt
                .steps
                .last()
                .map(|step| step.decision_at_us)
                .ok_or(CausalPairingError::EmptyTrajectory { arm: receipt.arm })?;
            if terminal_at_us < last_decision_at_us {
                return Err(CausalPairingError::ScoredTerminalBeforeDecision {
                    arm: receipt.arm,
                    terminal_at_us,
                    last_decision_at_us,
                });
            }
        }
    }
    Ok(())
}

fn validate_steps(
    receipt: &CausalTrajectoryReceipt,
) -> Result<StepValidationSummary, CausalPairingError> {
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
    require_digest(
        &receipt.stochastic_draw_bundle_sha256,
        Some(receipt.arm),
        CausalDigestField::StochasticDrawBundle,
    )?;

    let mut expected_tick = receipt.identity.initial_world_tick;
    let mut previous_decision_at_us = None;
    let mut delivered_evidence = BTreeSet::new();
    let mut summary = StepValidationSummary::default();

    for (index, step) in receipt.steps.iter().enumerate() {
        if step.world_tick != expected_tick {
            return Err(CausalPairingError::NonContiguousTick {
                arm: receipt.arm,
                expected: expected_tick,
                actual: step.world_tick,
            });
        }
        if let Some(previous_us) = previous_decision_at_us
            && step.decision_at_us < previous_us
        {
            return Err(CausalPairingError::DecisionClockRegression {
                arm: receipt.arm,
                previous_us,
                current_us: step.decision_at_us,
            });
        }
        if step.action_applied_at_us < step.decision_at_us {
            return Err(CausalPairingError::ActionAppliedBeforeDecision {
                arm: receipt.arm,
                decision_at_us: step.decision_at_us,
                action_applied_at_us: step.action_applied_at_us,
            });
        }
        previous_decision_at_us = Some(step.decision_at_us);

        for (digest, field) in [
            (&step.pre_state_sha256, CausalDigestField::PreState),
            (
                &step.applied_action_sha256,
                CausalDigestField::AppliedAction,
            ),
            (
                &step.exogenous_input_sha256,
                CausalDigestField::ExogenousInput,
            ),
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
        if index > 0 && step.pre_state_sha256 != receipt.steps[index - 1].post_state_sha256 {
            return Err(CausalPairingError::StateChainMismatch {
                arm: receipt.arm,
                world_tick: step.world_tick,
            });
        }

        for evidence in &step.delivered_evidence {
            require_digest(
                &evidence.event_id_sha256,
                Some(receipt.arm),
                CausalDigestField::EvidenceEventIdentity,
            )?;
            if evidence.available_at_us > evidence.delivered_at_us {
                return Err(CausalPairingError::EvidenceAvailableAfterDelivery {
                    arm: receipt.arm,
                    event_id_sha256: evidence.event_id_sha256,
                });
            }
            if evidence.delivered_at_us > step.decision_at_us {
                return Err(CausalPairingError::EvidenceDeliveredAfterDecision {
                    arm: receipt.arm,
                    event_id_sha256: evidence.event_id_sha256,
                });
            }
            if !delivered_evidence.insert(evidence.event_id_sha256) {
                return Err(CausalPairingError::DuplicateEvidenceDelivery {
                    arm: receipt.arm,
                    event_id_sha256: evidence.event_id_sha256,
                });
            }
        }

        let mut consumed_this_decision = BTreeSet::new();
        for event_id in &step.consumed_evidence_ids {
            require_digest(
                event_id,
                Some(receipt.arm),
                CausalDigestField::EvidenceEventIdentity,
            )?;
            if !consumed_this_decision.insert(*event_id) {
                return Err(CausalPairingError::DuplicateConsumedEvidence {
                    arm: receipt.arm,
                    event_id_sha256: *event_id,
                });
            }
            if !delivered_evidence.contains(event_id) {
                return Err(CausalPairingError::ConsumedEvidenceNotDelivered {
                    arm: receipt.arm,
                    event_id_sha256: *event_id,
                });
            }
            summary.consumed_evidence_reference_count = summary
                .consumed_evidence_reference_count
                .checked_add(1)
                .ok_or(CausalPairingError::ArithmeticOverflow)?;
        }
        summary.decision_count = summary
            .decision_count
            .checked_add(1)
            .ok_or(CausalPairingError::ArithmeticOverflow)?;

        if index + 1 < receipt.steps.len() {
            expected_tick = expected_tick
                .checked_add(1)
                .ok_or(CausalPairingError::TickOverflow { arm: receipt.arm })?;
        }
    }
    Ok(summary)
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

    let mut validated_decision_count = 0_u64;
    let mut consumed_evidence_reference_count = 0_u64;

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
        let step_summary = validate_steps(receipt)?;
        validate_scored_outcome_binding(receipt)?;
        validated_decision_count = validated_decision_count
            .checked_add(step_summary.decision_count)
            .ok_or(CausalPairingError::ArithmeticOverflow)?;
        consumed_evidence_reference_count = consumed_evidence_reference_count
            .checked_add(step_summary.consumed_evidence_reference_count)
            .ok_or(CausalPairingError::ArithmeticOverflow)?;
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
    let mut transition_outputs: BTreeMap<TransitionInputKey, TransitionOutputDigest> =
        BTreeMap::new();

    let mut exogenous_by_tick: BTreeMap<u64, [u8; 32]> = BTreeMap::new();

    for receipt in &ordered {
        for step in &receipt.steps {
            if exogenous_by_tick
                .insert(step.world_tick, step.exogenous_input_sha256)
                .is_some_and(|reference| reference != step.exogenous_input_sha256)
            {
                return Err(CausalPairingError::ExogenousInputMismatch {
                    world_tick: step.world_tick,
                });
            }

            let key = (
                step.world_tick,
                step.pre_state_sha256,
                step.applied_action_sha256,
                step.exogenous_input_sha256,
            );
            if transition_outputs
                .insert(key, step.post_state_sha256)
                .is_some_and(|previous_output| previous_output != step.post_state_sha256)
            {
                return Err(CausalPairingError::ConflictingDeterministicTransition {
                    world_tick: step.world_tick,
                });
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
        for receipt in &ordered[1..] {
            for (reference, candidate) in reference_steps.iter().zip(&receipt.steps) {
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

    let scored_outcome_bindings = [
        ordered[0].scored_outcome,
        ordered[1].scored_outcome,
        ordered[2].scored_outcome,
        ordered[3].scored_outcome,
    ];

    Ok(CausalPairingCertificate {
        regime,
        paired_run_identity: baseline.identity.paired.clone(),
        scored_outcome_bindings,
        validated_decision_count,
        consumed_evidence_reference_count,
        episode_id: baseline.identity.paired.episode_id,
        hidden_world_id: baseline.identity.paired.hidden_world_id,
        exogenous_schedule_sha256: baseline.identity.exogenous_schedule_sha256,
        matched_tick_count,
        terminal_state_sha256,
        trajectory_origins,
    })
}

/// Combined certificate joining four causal trajectories to the exact
/// four-arm decision-regret outcomes scored at action-application time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CausalDecisionValueCertificate {
    pub causal_pairing: CausalPairingCertificate,
    pub decision_value: DecisionValueInteractionCertificate,
}

/// Validate that the causal trajectory receipts and the scored decision-regret
/// outcomes refer to the same paired identity, treatment arm, action digest,
/// application timestamp, or terminal non-application disposition.
///
/// This is the join boundary between #13375 and #13281. It prevents a caller
/// from attaching a valid regret vector to unrelated but individually valid
/// trajectory receipts.
pub fn validate_causal_pairing_with_interaction(
    receipts: &[CausalTrajectoryReceipt],
    outcomes: &[crate::ArmOutcome],
) -> Result<CausalDecisionValueCertificate, CausalPairingError> {
    let decision_value = crate::evaluate_decision_value_interaction(outcomes)
        .map_err(CausalPairingError::DecisionValueInteraction)?;
    let causal_pairing = validate_causal_pairing(receipts)?;

    for outcome in outcomes {
        crate::validate_pair(&causal_pairing.paired_run_identity, &outcome.identity).map_err(
            |error| match error {
                InteractionError::IdentityMismatch { field } => {
                    CausalPairingError::ScoredOutcomeIdentityMismatch { field }
                }
                other => CausalPairingError::DecisionValueInteraction(other),
            },
        )?;

        let index = arm_index_for_pairing(outcome.arm);
        let binding = causal_pairing.scored_outcome_bindings[index];
        let disposition_matches = match (binding, outcome.disposition) {
            (
                ScoredOutcomeBinding::Applied { applied_at_us, .. },
                ApplicationDisposition::Applied { at_us },
            ) => applied_at_us == at_us,
            (
                ScoredOutcomeBinding::NotApplied {
                    terminal_at_us: expected_terminal_at_us,
                    reason: expected_reason,
                },
                ApplicationDisposition::NotApplied {
                    terminal_at_us,
                    reason,
                },
            ) => expected_terminal_at_us == terminal_at_us && expected_reason == reason,
            _ => false,
        };
        if !disposition_matches {
            return Err(CausalPairingError::ScoredOutcomeDispositionMismatch { arm: outcome.arm });
        }
    }

    Ok(CausalDecisionValueCertificate {
        causal_pairing,
        decision_value,
    })
}

fn arm_index_for_pairing(arm: TreatmentArm) -> usize {
    match arm {
        TreatmentArm::Baseline => 0,
        TreatmentArm::ComputeOnly => 1,
        TreatmentArm::ObservationOnly => 2,
        TreatmentArm::Joint => 3,
    }
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
            decision_at_us: (tick - 99) * 1_000,
            action_applied_at_us: (tick - 99) * 1_000 + 100,
            delivered_evidence: Vec::new(),
            consumed_evidence_ids: Vec::new(),
            pre_state_sha256: digest(pre),
            applied_action_sha256: digest(action),
            exogenous_input_sha256: digest(exogenous),
            post_state_sha256: digest(post),
            observation_delivery_sha256: digest(observation),
        }
    }

    fn arm_outcomes(regrets: [u32; 4]) -> Vec<crate::ArmOutcome> {
        let arms = [
            TreatmentArm::Baseline,
            TreatmentArm::ComputeOnly,
            TreatmentArm::ObservationOnly,
            TreatmentArm::Joint,
        ];
        arms.into_iter()
            .enumerate()
            .map(|(index, arm)| crate::ArmOutcome {
                identity: paired_identity(),
                arm,
                realized_regret_ppm: regrets[index],
                disposition: ApplicationDisposition::Applied { at_us: 2_100 },
            })
            .collect()
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
        arms.into_iter()
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
                    scored_outcome: ScoredOutcomeBinding::Applied {
                        world_tick: 101,
                        action_sha256: digest(70 + index as u8),
                        applied_at_us: 2_100,
                    },
                    driver_execution_receipt_sha256: digest(90 + index as u8),
                    independent_replay_audit_sha256: digest(100 + index as u8),
                    stochastic_draw_bundle_sha256: digest(120 + index as u8),
                    independent_replay_audit_passed: true,
                    complete: true,
                }
            })
            .collect()
    }

    #[test]
    fn causal_pairing_and_decision_regret_join_on_same_actions() {
        let runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        let certificate = validate_causal_pairing_with_interaction(
            &runs,
            &arm_outcomes([400_000, 350_000, 380_000, 100_000]),
        )
        .expect("causal trajectories and regret outcomes share the same actions");
        assert_eq!(certificate.decision_value.interaction_ppm, 230_000);
        assert_eq!(
            certificate.causal_pairing.paired_run_identity,
            certificate.decision_value.pairing_identity
        );
    }

    #[test]
    fn scored_action_with_missing_world_tick_is_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[1].scored_outcome = ScoredOutcomeBinding::Applied {
            world_tick: 999,
            action_sha256: digest(71),
            applied_at_us: 2_100,
        };
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::ScoredActionTickMissing {
                arm: TreatmentArm::ComputeOnly,
                world_tick: 999
            })
        ));
    }

    #[test]
    fn scored_action_digest_must_match_trajectory_step() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[2].scored_outcome = ScoredOutcomeBinding::Applied {
            world_tick: 101,
            action_sha256: digest(222),
            applied_at_us: 2_100,
        };
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::ScoredActionDigestMismatch {
                arm: TreatmentArm::ObservationOnly,
                world_tick: 101
            })
        ));
    }

    #[test]
    fn scored_action_application_time_must_match_trajectory() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[3].scored_outcome = ScoredOutcomeBinding::Applied {
            world_tick: 101,
            action_sha256: digest(73),
            applied_at_us: 2_101,
        };
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::ScoredActionTimeMismatch {
                arm: TreatmentArm::Joint,
                expected_us: 2_100,
                actual_us: 2_101
            })
        ));
    }

    #[test]
    fn regret_identity_must_match_causal_trajectory_identity() {
        let runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        let mut outcomes = arm_outcomes([400_000, 350_000, 380_000, 100_000]);
        for outcome in &mut outcomes {
            outcome.identity.policy_seed = 100;
        }
        assert!(matches!(
            validate_causal_pairing_with_interaction(&runs, &outcomes),
            Err(CausalPairingError::ScoredOutcomeIdentityMismatch {
                field: crate::IdentityField::PolicySeed
            })
        ));
    }

    #[test]
    fn regret_disposition_must_match_scored_outcome_binding() {
        let runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        let mut outcomes = arm_outcomes([400_000, 350_000, 380_000, 100_000]);
        outcomes[2].disposition = ApplicationDisposition::Applied { at_us: 2_101 };
        assert!(matches!(
            validate_causal_pairing_with_interaction(&runs, &outcomes),
            Err(CausalPairingError::ScoredOutcomeDispositionMismatch {
                arm: TreatmentArm::ObservationOnly
            })
        ));
    }

    #[test]
    fn matching_non_application_terminal_disposition_can_be_joined() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[3].scored_outcome = ScoredOutcomeBinding::NotApplied {
            terminal_at_us: 2_300,
            reason: NonApplicationReason::DeadlineMiss,
        };
        let mut outcomes = arm_outcomes([400_000, 350_000, 380_000, 1_000_000]);
        outcomes[3].disposition = ApplicationDisposition::NotApplied {
            terminal_at_us: 2_300,
            reason: NonApplicationReason::DeadlineMiss,
        };
        let certificate = validate_causal_pairing_with_interaction(&runs, &outcomes)
            .expect("matching non-application outcome remains scored");
        assert_eq!(certificate.decision_value.realized_regret_ppm[3], 1_000_000);
    }

    #[test]
    fn mismatched_non_application_reason_is_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[0].scored_outcome = ScoredOutcomeBinding::NotApplied {
            terminal_at_us: 2_300,
            reason: NonApplicationReason::DeadlineMiss,
        };
        let mut outcomes = arm_outcomes([1_000_000, 350_000, 380_000, 100_000]);
        outcomes[0].disposition = ApplicationDisposition::NotApplied {
            terminal_at_us: 2_300,
            reason: NonApplicationReason::ExecutionFailure,
        };
        assert!(matches!(
            validate_causal_pairing_with_interaction(&runs, &outcomes),
            Err(CausalPairingError::ScoredOutcomeDispositionMismatch {
                arm: TreatmentArm::Baseline
            })
        ));
    }

    #[test]
    fn non_application_cannot_precede_final_decision() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[0].scored_outcome = ScoredOutcomeBinding::NotApplied {
            terminal_at_us: 1_999,
            reason: NonApplicationReason::DeadlineMiss,
        };
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::ScoredTerminalBeforeDecision {
                arm: TreatmentArm::Baseline,
                terminal_at_us: 1_999,
                last_decision_at_us: 2_000
            })
        ));
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
        assert_ne!(
            certificate.terminal_state_sha256[0],
            certificate.terminal_state_sha256[1]
        );
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
            Err(
                CausalPairingError::SharedRecordedTrajectoryForbiddenForClosedLoop {
                    arm: TreatmentArm::Baseline
                }
            )
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
        assert!(
            certificate
                .terminal_state_sha256
                .iter()
                .all(|state| *state == digest(40))
        );
    }

    #[test]
    fn closed_loop_null_case_may_have_identical_trajectories() {
        let runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            false,
        );
        let certificate = validate_causal_pairing(&runs).expect("valid null causal case");
        assert!(
            certificate
                .terminal_state_sha256
                .iter()
                .all(|state| *state == digest(40))
        );
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
    fn missing_correspondence_draw_bundle_receipt_is_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[1].stochastic_draw_bundle_sha256 = digest(0);
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::EmptyDigest {
                arm: Some(TreatmentArm::ComputeOnly),
                field: CausalDigestField::StochasticDrawBundle
            })
        ));
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
        runs[1].steps[1].pre_state_sha256 = digest(222);
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::ConflictingDeterministicTransition { world_tick: 100 })
        ));
    }

    #[test]
    fn closed_loop_checks_exogenous_overlap_even_beyond_baseline_termination() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[0].steps.truncate(1);
        runs[0].scored_outcome = ScoredOutcomeBinding::Applied {
            world_tick: 100,
            action_sha256: digest(20),
            applied_at_us: 1_100,
        };
        runs[2].steps[1].exogenous_input_sha256 = digest(222);
        runs[3].steps[1].exogenous_input_sha256 = digest(222);
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::ExogenousInputMismatch { world_tick: 101 })
        ));
    }

    #[test]
    fn discontinuous_state_chain_is_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[2].steps[1].pre_state_sha256 = digest(233);
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::StateChainMismatch {
                arm: TreatmentArm::ObservationOnly,
                world_tick: 101
            })
        ));
    }

    #[test]
    fn duplicate_treatment_arm_is_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[3].arm = TreatmentArm::Baseline;
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::DuplicateArm {
                arm: TreatmentArm::Baseline
            })
        ));
    }

    #[test]
    fn incomplete_four_arm_set_is_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs.truncate(3);
        assert_eq!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::WrongReceiptCount { actual: 3 })
        );
    }

    fn evidence(id: u8, available_at_us: u64, delivered_at_us: u64) -> DeliveredEvidenceRef {
        DeliveredEvidenceRef {
            event_id_sha256: digest(id),
            available_at_us,
            delivered_at_us,
        }
    }

    #[test]
    fn certificate_counts_decisions_and_consumed_evidence_references() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        for run in &mut runs {
            run.steps[0].delivered_evidence = vec![evidence(201, 900, 950)];
            run.steps[0].consumed_evidence_ids = vec![digest(201)];
            run.steps[1].consumed_evidence_ids = vec![digest(201)];
        }
        let certificate = validate_causal_pairing(&runs).expect("decision-time evidence is valid");
        assert_eq!(certificate.validated_decision_count, 8);
        assert_eq!(certificate.consumed_evidence_reference_count, 8);
    }

    #[test]
    fn prior_delivered_evidence_can_be_reused_at_later_decisions() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        for run in &mut runs {
            run.steps[0].delivered_evidence = vec![evidence(202, 800, 900)];
            run.steps[1].consumed_evidence_ids = vec![digest(202)];
        }
        assert!(validate_causal_pairing(&runs).is_ok());
    }

    #[test]
    fn future_evidence_cannot_be_consumed_before_delivery() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[0].steps[0].consumed_evidence_ids = vec![digest(203)];
        runs[0].steps[1].delivered_evidence = vec![evidence(203, 1_900, 1_950)];
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::ConsumedEvidenceNotDelivered {
                arm: TreatmentArm::Baseline,
                event_id_sha256
            }) if event_id_sha256 == digest(203)
        ));
    }

    #[test]
    fn evidence_delivered_after_decision_is_rejected() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[1].steps[0].delivered_evidence = vec![evidence(204, 900, 1_001)];
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::EvidenceDeliveredAfterDecision {
                arm: TreatmentArm::ComputeOnly,
                event_id_sha256
            }) if event_id_sha256 == digest(204)
        ));
    }

    #[test]
    fn evidence_cannot_be_delivered_before_it_is_available() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[2].steps[0].delivered_evidence = vec![evidence(205, 1_001, 1_000)];
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::EvidenceAvailableAfterDelivery {
                arm: TreatmentArm::ObservationOnly,
                event_id_sha256
            }) if event_id_sha256 == digest(205)
        ));
    }

    #[test]
    fn an_event_cannot_be_delivered_twice_to_one_arm() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[3].steps[0].delivered_evidence = vec![evidence(206, 800, 900)];
        runs[3].steps[1].delivered_evidence = vec![evidence(206, 800, 1_900)];
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::DuplicateEvidenceDelivery {
                arm: TreatmentArm::Joint,
                event_id_sha256
            }) if event_id_sha256 == digest(206)
        ));
    }

    #[test]
    fn one_decision_cannot_claim_the_same_evidence_twice() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[0].steps[0].delivered_evidence = vec![evidence(207, 800, 900)];
        runs[0].steps[0].consumed_evidence_ids = vec![digest(207), digest(207)];
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::DuplicateConsumedEvidence {
                arm: TreatmentArm::Baseline,
                event_id_sha256
            }) if event_id_sha256 == digest(207)
        ));
    }

    #[test]
    fn decision_clock_must_not_regress() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[2].steps[1].decision_at_us = 999;
        runs[2].steps[1].action_applied_at_us = 1_100;
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::DecisionClockRegression {
                arm: TreatmentArm::ObservationOnly,
                previous_us: 1_000,
                current_us: 999
            })
        ));
    }

    #[test]
    fn action_cannot_be_applied_before_its_decision() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[3].steps[0].action_applied_at_us = 999;
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::ActionAppliedBeforeDecision {
                arm: TreatmentArm::Joint,
                decision_at_us: 1_000,
                action_applied_at_us: 999
            })
        ));
    }

    #[test]
    fn evidence_event_identity_cannot_be_empty() {
        let mut runs = receipts(
            CausalReplayRegime::ClosedLoopActionCoupled,
            TrajectoryOrigin::ArmLocalDriverExecution,
            true,
        );
        runs[1].steps[0].delivered_evidence = vec![evidence(0, 800, 900)];
        assert!(matches!(
            validate_causal_pairing(&runs),
            Err(CausalPairingError::EmptyDigest {
                arm: Some(TreatmentArm::ComputeOnly),
                field: CausalDigestField::EvidenceEventIdentity
            })
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
