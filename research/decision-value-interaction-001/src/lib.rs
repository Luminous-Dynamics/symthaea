//! Paired evaluator reference for decision-regret interaction between compute and observation.
//!
//! This crate evaluates exactly four paired treatment outcomes for one frozen
//! episode and hidden-world commitment. It does not estimate loss, create
//! candidate policies, choose observations, allocate resources, or authorize
//! actions. Regret is supplied by an independent evaluator at each arm's
//! actual action-application time.
//!
//! The result is a per-world factorial contrast. Episode-level statistical
//! inference, resource/exposure factorial attribution, and joint-plan
//! feasibility remain owned by the benchmark protocol and #13176/#9882.
//!
//! Evaluator records and this crate must remain outside candidate policy
//! processes. Package separation alone is not a security boundary; #9437
//! process/UID isolation, protected storage, evaluator-owned signing keys and
//! replay-resistant receipts remain necessary.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;

pub mod cluster_adapter;

/// Normalized regret scale used by the paired benchmark.
pub const DECISION_REGRET_PPM: u32 = 1_000_000;

/// The four arms of the 2x2 additional-compute x additional-observation design.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TreatmentArm {
    /// No additional compute block and no additional observation.
    Baseline,
    /// Additional compute using the original authorized evidence prefix.
    ComputeOnly,
    /// One additional observation without the treatment compute block.
    ObservationOnly,
    /// Additional observation plus the declared additional compute block.
    Joint,
}

impl TreatmentArm {
    fn index(self) -> usize {
        match self {
            Self::Baseline => 0,
            Self::ComputeOnly => 1,
            Self::ObservationOnly => 2,
            Self::Joint => 3,
        }
    }
}

/// Why a treatment arm did not produce an applied action.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NonApplicationReason {
    DeadlineMiss,
    NoQualifiedAction,
    RejectedByConsumer,
    ExecutionFailure,
}

/// Terminal disposition of one treatment arm.
///
/// NotApplied remains a scored arm, but its penalty must be supplied by the
/// frozen task contract. This crate never invents a no-output penalty.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplicationDisposition {
    Applied {
        at_us: u64,
    },
    NotApplied {
        terminal_at_us: u64,
        reason: NonApplicationReason,
    },
}

/// Identity shared by all four counterfactual treatment runs.
///
/// Application time is deliberately not part of this identity: additional
/// compute or observation may delay action application, and that difference is
/// part of the measured treatment effect. Observation transcripts may also
/// differ through the declared observation treatment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedRunIdentity {
    pub episode_id: u64,
    pub hidden_world_id: u64,
    pub hidden_world_sha256: [u8; 32],
    pub initial_state_sha256: [u8; 32],
    pub task_sha256: [u8; 32],
    pub loss_scale_sha256: [u8; 32],
    pub evaluator_profile_sha256: [u8; 32],
    /// Digest of the frozen simulator/plant/hardware execution environment.
    pub execution_environment_sha256: [u8; 32],
    /// Digest of the common candidate action domain and action semantics.
    pub action_domain_sha256: [u8; 32],
    /// Digest of the common hard resource/deadline budget envelope.
    pub resource_budget_sha256: [u8; 32],
    pub model_configuration_sha256: [u8; 32],
    pub candidate_policy_sha256: [u8; 32],
    pub policy_seed: u64,
}

/// One evaluator-measured outcome for a single treatment arm.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArmOutcome {
    pub identity: PairedRunIdentity,
    pub arm: TreatmentArm,
    /// Nonnegative evaluator score in [0, 1,000,000] PPM.
    ///
    /// For Applied, this is action regret at the actual application time.
    /// For NotApplied, this is the task-contract-defined terminal no-output
    /// regret penalty. It must not be imputed as zero or dropped.
    pub realized_regret_ppm: u32,
    /// Applied time or explicit terminal no-application disposition.
    pub disposition: ApplicationDisposition,
}

/// Sign of the compute-observation interaction on the declared regret scale.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InteractionSign {
    /// The joint treatment reduces more regret than isolated benefits predict.
    Complementary,
    /// The joint treatment is additive relative to isolated effects.
    AdditiveOrNeutral,
    /// The joint treatment is less useful than isolated effects predict.
    Antagonistic,
}

/// Exact paired decision-regret contrasts for one episode and hidden world.
///
/// All signed benefits are retained. Negative benefit means that the treatment
/// increased regret. Positive interaction means complementarity; negative
/// interaction means redundancy/antagonism. These quantities are not resource
/// savings and must not be added to #13176's exposure/resource dividends.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionValueInteractionCertificate {
    /// The exact thirteen-field identity validated across all four treatment arms.
    pub pairing_identity: PairedRunIdentity,
    pub episode_id: u64,
    pub hidden_world_id: u64,
    pub hidden_world_sha256: [u8; 32],
    /// Regrets in [baseline, compute-only, observation-only, joint] order.
    pub realized_regret_ppm: [u32; 4],
    /// Application/terminal dispositions in [baseline, compute-only, observation-only, joint] order.
    pub dispositions: [ApplicationDisposition; 4],
    /// Baseline regret minus compute-only regret.
    pub compute_benefit_ppm: i64,
    /// Baseline regret minus observation-only regret.
    pub observation_benefit_ppm: i64,
    /// Baseline regret minus joint-treatment regret.
    pub joint_benefit_ppm: i64,
    /// R10 + R01 - R00 - R11.
    pub interaction_ppm: i64,
    pub interaction_sign: InteractionSign,
}

/// Identity component used when reporting a pairing failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentityField {
    EpisodeId,
    HiddenWorldId,
    HiddenWorldDigest,
    InitialStateDigest,
    TaskDigest,
    LossScaleDigest,
    EvaluatorProfileDigest,
    ExecutionEnvironmentDigest,
    ActionDomainDigest,
    ResourceBudgetDigest,
    ModelConfigurationDigest,
    CandidatePolicyDigest,
    PolicySeed,
}

/// Errors are explicit so incomplete/mismatched arms cannot be silently scored.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InteractionError {
    WrongOutcomeCount { actual: usize },
    DuplicateArm { arm: TreatmentArm },
    MissingArm { arm: TreatmentArm },
    EmptyIdentityDigest { field: IdentityField },
    IdentityMismatch { field: IdentityField },
    RegretOutOfRange { arm: TreatmentArm, regret_ppm: u32 },
    ArithmeticOverflow,
}

impl fmt::Display for InteractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongOutcomeCount { actual } => {
                write!(
                    formatter,
                    "expected exactly four treatment outcomes, received {actual}"
                )
            }
            Self::DuplicateArm { arm } => {
                write!(formatter, "treatment arm {arm:?} appears more than once")
            }
            Self::MissingArm { arm } => {
                write!(formatter, "required treatment arm {arm:?} is missing")
            }
            Self::EmptyIdentityDigest { field } => {
                write!(formatter, "paired-run identity digest {field:?} is empty")
            }
            Self::IdentityMismatch { field } => {
                write!(
                    formatter,
                    "paired treatment runs disagree on identity field {field:?}"
                )
            }
            Self::RegretOutOfRange { arm, regret_ppm } => {
                write!(
                    formatter,
                    "arm {arm:?} has regret {regret_ppm} outside [0, {DECISION_REGRET_PPM}]"
                )
            }
            Self::ArithmeticOverflow => write!(
                formatter,
                "decision-value interaction arithmetic overflowed"
            ),
        }
    }
}

impl std::error::Error for InteractionError {}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn validate_identity(identity: &PairedRunIdentity) -> Result<(), InteractionError> {
    let digests = [
        (
            IdentityField::HiddenWorldDigest,
            &identity.hidden_world_sha256,
        ),
        (
            IdentityField::InitialStateDigest,
            &identity.initial_state_sha256,
        ),
        (IdentityField::TaskDigest, &identity.task_sha256),
        (IdentityField::LossScaleDigest, &identity.loss_scale_sha256),
        (
            IdentityField::EvaluatorProfileDigest,
            &identity.evaluator_profile_sha256,
        ),
        (
            IdentityField::ExecutionEnvironmentDigest,
            &identity.execution_environment_sha256,
        ),
        (
            IdentityField::ActionDomainDigest,
            &identity.action_domain_sha256,
        ),
        (
            IdentityField::ResourceBudgetDigest,
            &identity.resource_budget_sha256,
        ),
        (
            IdentityField::ModelConfigurationDigest,
            &identity.model_configuration_sha256,
        ),
        (
            IdentityField::CandidatePolicyDigest,
            &identity.candidate_policy_sha256,
        ),
    ];
    for (field, digest) in digests {
        if digest_is_empty(digest) {
            return Err(InteractionError::EmptyIdentityDigest { field });
        }
    }
    Ok(())
}

fn validate_pair(
    reference: &PairedRunIdentity,
    candidate: &PairedRunIdentity,
) -> Result<(), InteractionError> {
    let checks = [
        (
            IdentityField::EpisodeId,
            reference.episode_id == candidate.episode_id,
        ),
        (
            IdentityField::HiddenWorldId,
            reference.hidden_world_id == candidate.hidden_world_id,
        ),
        (
            IdentityField::HiddenWorldDigest,
            reference.hidden_world_sha256 == candidate.hidden_world_sha256,
        ),
        (
            IdentityField::InitialStateDigest,
            reference.initial_state_sha256 == candidate.initial_state_sha256,
        ),
        (
            IdentityField::TaskDigest,
            reference.task_sha256 == candidate.task_sha256,
        ),
        (
            IdentityField::LossScaleDigest,
            reference.loss_scale_sha256 == candidate.loss_scale_sha256,
        ),
        (
            IdentityField::EvaluatorProfileDigest,
            reference.evaluator_profile_sha256 == candidate.evaluator_profile_sha256,
        ),
        (
            IdentityField::ExecutionEnvironmentDigest,
            reference.execution_environment_sha256 == candidate.execution_environment_sha256,
        ),
        (
            IdentityField::ActionDomainDigest,
            reference.action_domain_sha256 == candidate.action_domain_sha256,
        ),
        (
            IdentityField::ResourceBudgetDigest,
            reference.resource_budget_sha256 == candidate.resource_budget_sha256,
        ),
        (
            IdentityField::ModelConfigurationDigest,
            reference.model_configuration_sha256 == candidate.model_configuration_sha256,
        ),
        (
            IdentityField::CandidatePolicyDigest,
            reference.candidate_policy_sha256 == candidate.candidate_policy_sha256,
        ),
        (
            IdentityField::PolicySeed,
            reference.policy_seed == candidate.policy_seed,
        ),
    ];
    for (field, matches) in checks {
        if !matches {
            return Err(InteractionError::IdentityMismatch { field });
        }
    }
    Ok(())
}

fn signed_difference(lhs: u32, rhs: u32) -> Result<i64, InteractionError> {
    let difference = i128::from(lhs)
        .checked_sub(i128::from(rhs))
        .ok_or(InteractionError::ArithmeticOverflow)?;
    i64::try_from(difference).map_err(|_| InteractionError::ArithmeticOverflow)
}

/// Evaluate one complete paired compute x observation decision-regret contrast.
///
/// Exactly one outcome is required for each of the four treatment arms. All
/// paired identities must match. Each regret is in the same task-loss scale
/// and must lie in [0, 1_000_000] PPM. Arm application times may differ and are
/// returned in the certificate rather than normalized away.
pub fn evaluate_decision_value_interaction(
    outcomes: &[ArmOutcome],
) -> Result<DecisionValueInteractionCertificate, InteractionError> {
    if outcomes.len() != 4 {
        return Err(InteractionError::WrongOutcomeCount {
            actual: outcomes.len(),
        });
    }

    let mut by_arm: BTreeMap<TreatmentArm, &ArmOutcome> = BTreeMap::new();
    for outcome in outcomes {
        if outcome.realized_regret_ppm > DECISION_REGRET_PPM {
            return Err(InteractionError::RegretOutOfRange {
                arm: outcome.arm,
                regret_ppm: outcome.realized_regret_ppm,
            });
        }
        validate_identity(&outcome.identity)?;
        if by_arm.insert(outcome.arm, outcome).is_some() {
            return Err(InteractionError::DuplicateArm { arm: outcome.arm });
        }
    }

    let ordered = [
        TreatmentArm::Baseline,
        TreatmentArm::ComputeOnly,
        TreatmentArm::ObservationOnly,
        TreatmentArm::Joint,
    ];
    let mut regrets = [0_u32; 4];
    let mut dispositions = [ApplicationDisposition::Applied { at_us: 0 }; 4];

    let baseline =
        by_arm
            .get(&TreatmentArm::Baseline)
            .copied()
            .ok_or(InteractionError::MissingArm {
                arm: TreatmentArm::Baseline,
            })?;

    for arm in ordered {
        let outcome = by_arm
            .get(&arm)
            .copied()
            .ok_or(InteractionError::MissingArm { arm })?;
        validate_pair(&baseline.identity, &outcome.identity)?;
        let index = arm.index();
        regrets[index] = outcome.realized_regret_ppm;
        dispositions[index] = outcome.disposition;
    }

    let compute_benefit = signed_difference(regrets[0], regrets[1])?;
    let observation_benefit = signed_difference(regrets[0], regrets[2])?;
    let joint_benefit = signed_difference(regrets[0], regrets[3])?;

    let interaction = i128::from(regrets[1])
        .checked_add(i128::from(regrets[2]))
        .and_then(|sum| sum.checked_sub(i128::from(regrets[0])))
        .and_then(|sum| sum.checked_sub(i128::from(regrets[3])))
        .ok_or(InteractionError::ArithmeticOverflow)?;
    let interaction_ppm =
        i64::try_from(interaction).map_err(|_| InteractionError::ArithmeticOverflow)?;

    let interaction_sign = if interaction_ppm > 0 {
        InteractionSign::Complementary
    } else if interaction_ppm < 0 {
        InteractionSign::Antagonistic
    } else {
        InteractionSign::AdditiveOrNeutral
    };

    Ok(DecisionValueInteractionCertificate {
        pairing_identity: baseline.identity.clone(),
        episode_id: baseline.identity.episode_id,
        hidden_world_id: baseline.identity.hidden_world_id,
        hidden_world_sha256: baseline.identity.hidden_world_sha256,
        realized_regret_ppm: regrets,
        dispositions,
        compute_benefit_ppm: compute_benefit,
        observation_benefit_ppm: observation_benefit,
        joint_benefit_ppm: joint_benefit,
        interaction_ppm,
        interaction_sign,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type IdentityMutation = (IdentityField, fn(&mut PairedRunIdentity));

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn identity() -> PairedRunIdentity {
        PairedRunIdentity {
            episode_id: 41,
            hidden_world_id: 7,
            hidden_world_sha256: digest(1),
            initial_state_sha256: digest(2),
            task_sha256: digest(3),
            loss_scale_sha256: digest(4),
            evaluator_profile_sha256: digest(5),
            execution_environment_sha256: digest(8),
            action_domain_sha256: digest(9),
            resource_budget_sha256: digest(10),
            model_configuration_sha256: digest(6),
            candidate_policy_sha256: digest(7),
            policy_seed: 99,
        }
    }

    fn outcomes(regrets: [u32; 4]) -> Vec<ArmOutcome> {
        let arms = [
            TreatmentArm::Baseline,
            TreatmentArm::ComputeOnly,
            TreatmentArm::ObservationOnly,
            TreatmentArm::Joint,
        ];
        arms.into_iter()
            .enumerate()
            .map(|(index, arm)| ArmOutcome {
                identity: identity(),
                arm,
                realized_regret_ppm: regrets[index],
                disposition: ApplicationDisposition::Applied {
                    at_us: 100 + (index as u64 * 10),
                },
            })
            .collect()
    }

    #[test]
    fn positive_interaction_detects_complementarity() {
        let result =
            evaluate_decision_value_interaction(&outcomes([400_000, 350_000, 380_000, 100_000]))
                .expect("complete paired arms");

        assert_eq!(result.compute_benefit_ppm, 50_000);
        assert_eq!(result.observation_benefit_ppm, 20_000);
        assert_eq!(result.joint_benefit_ppm, 300_000);
        assert_eq!(result.interaction_ppm, 230_000);
        assert_eq!(result.interaction_sign, InteractionSign::Complementary);
    }

    #[test]
    fn zero_interaction_is_additive_or_neutral() {
        let result =
            evaluate_decision_value_interaction(&outcomes([400_000, 300_000, 250_000, 150_000]))
                .expect("complete paired arms");

        assert_eq!(result.compute_benefit_ppm, 100_000);
        assert_eq!(result.observation_benefit_ppm, 150_000);
        assert_eq!(result.joint_benefit_ppm, 250_000);
        assert_eq!(result.interaction_ppm, 0);
        assert_eq!(result.interaction_sign, InteractionSign::AdditiveOrNeutral);
    }

    #[test]
    fn negative_interaction_preserves_antagonism() {
        let result =
            evaluate_decision_value_interaction(&outcomes([400_000, 250_000, 300_000, 320_000]))
                .expect("complete paired arms");

        assert_eq!(result.compute_benefit_ppm, 150_000);
        assert_eq!(result.observation_benefit_ppm, 100_000);
        assert_eq!(result.joint_benefit_ppm, 80_000);
        assert_eq!(result.interaction_ppm, -170_000);
        assert_eq!(result.interaction_sign, InteractionSign::Antagonistic);
    }

    #[test]
    fn harmful_compute_and_joint_escalation_are_not_clipped() {
        let result =
            evaluate_decision_value_interaction(&outcomes([400_000, 550_000, 350_000, 600_000]))
                .expect("complete paired arms");

        assert_eq!(result.compute_benefit_ppm, -150_000);
        assert_eq!(result.observation_benefit_ppm, 50_000);
        assert_eq!(result.joint_benefit_ppm, -200_000);
        assert_eq!(result.interaction_ppm, -100_000);
        assert_eq!(result.interaction_sign, InteractionSign::Antagonistic);
    }

    #[test]
    fn input_order_does_not_change_treatment_assignment() {
        let mut rows = outcomes([400_000, 350_000, 380_000, 100_000]);
        rows.reverse();
        let result = evaluate_decision_value_interaction(&rows)
            .expect("arm IDs, not input order, define the experiment");
        assert_eq!(result.interaction_ppm, 230_000);
        assert_eq!(
            result.dispositions,
            [
                ApplicationDisposition::Applied { at_us: 100 },
                ApplicationDisposition::Applied { at_us: 110 },
                ApplicationDisposition::Applied { at_us: 120 },
                ApplicationDisposition::Applied { at_us: 130 },
            ]
        );
    }

    #[test]
    fn application_time_differences_are_preserved() {
        let mut rows = outcomes([400_000, 350_000, 380_000, 100_000]);
        rows[1].disposition = ApplicationDisposition::Applied { at_us: 250 };
        rows[3].disposition = ApplicationDisposition::Applied { at_us: 900 };
        let result = evaluate_decision_value_interaction(&rows)
            .expect("application times are treatment outcomes, not pairing keys");
        assert_eq!(
            result.dispositions,
            [
                ApplicationDisposition::Applied { at_us: 100 },
                ApplicationDisposition::Applied { at_us: 250 },
                ApplicationDisposition::Applied { at_us: 120 },
                ApplicationDisposition::Applied { at_us: 900 },
            ]
        );
        assert_eq!(result.interaction_ppm, 230_000);
    }

    #[test]
    fn non_application_is_retained_with_terminal_penalty() {
        let mut rows = outcomes([400_000, 1_000_000, 380_000, 100_000]);
        rows[1].disposition = ApplicationDisposition::NotApplied {
            terminal_at_us: 200,
            reason: NonApplicationReason::DeadlineMiss,
        };
        let result = evaluate_decision_value_interaction(&rows)
            .expect("task-defined no-output penalty remains in the paired design");
        assert_eq!(result.compute_benefit_ppm, -600_000);
        assert_eq!(result.joint_benefit_ppm, 300_000);
        assert_eq!(
            result.dispositions[1],
            ApplicationDisposition::NotApplied {
                terminal_at_us: 200,
                reason: NonApplicationReason::DeadlineMiss,
            }
        );
    }

    #[test]
    fn missing_arm_is_rejected() {
        let mut rows = outcomes([400_000, 350_000, 380_000, 100_000]);
        rows.pop();
        assert!(matches!(
            evaluate_decision_value_interaction(&rows),
            Err(InteractionError::WrongOutcomeCount { actual: 3 })
        ));
    }

    #[test]
    fn duplicate_arm_is_rejected() {
        let mut rows = outcomes([400_000, 350_000, 380_000, 100_000]);
        rows[3].arm = TreatmentArm::ComputeOnly;
        assert!(matches!(
            evaluate_decision_value_interaction(&rows),
            Err(InteractionError::DuplicateArm {
                arm: TreatmentArm::ComputeOnly
            })
        ));
    }

    #[test]
    fn hidden_world_digest_mismatch_is_rejected() {
        let mut rows = outcomes([400_000, 350_000, 380_000, 100_000]);
        rows[3].identity.hidden_world_sha256 = digest(88);
        assert!(matches!(
            evaluate_decision_value_interaction(&rows),
            Err(InteractionError::IdentityMismatch {
                field: IdentityField::HiddenWorldDigest
            })
        ));
    }

    #[test]
    fn candidate_policy_digest_mismatch_is_rejected() {
        let mut rows = outcomes([400_000, 350_000, 380_000, 100_000]);
        rows[2].identity.candidate_policy_sha256 = digest(91);
        assert!(matches!(
            evaluate_decision_value_interaction(&rows),
            Err(InteractionError::IdentityMismatch {
                field: IdentityField::CandidatePolicyDigest
            })
        ));
    }

    #[test]
    fn every_shared_run_identity_field_is_bound_across_all_four_arms() {
        fn change_episode_id(value: &mut PairedRunIdentity) {
            value.episode_id += 1;
        }
        fn change_world_id(value: &mut PairedRunIdentity) {
            value.hidden_world_id += 1;
        }
        fn change_hidden_world_digest(value: &mut PairedRunIdentity) {
            value.hidden_world_sha256 = digest(81);
        }
        fn change_initial_state_digest(value: &mut PairedRunIdentity) {
            value.initial_state_sha256 = digest(82);
        }
        fn change_task_digest(value: &mut PairedRunIdentity) {
            value.task_sha256 = digest(83);
        }
        fn change_loss_scale_digest(value: &mut PairedRunIdentity) {
            value.loss_scale_sha256 = digest(84);
        }
        fn change_evaluator_profile_digest(value: &mut PairedRunIdentity) {
            value.evaluator_profile_sha256 = digest(85);
        }
        fn change_execution_environment_digest(value: &mut PairedRunIdentity) {
            value.execution_environment_sha256 = digest(88);
        }
        fn change_action_domain_digest(value: &mut PairedRunIdentity) {
            value.action_domain_sha256 = digest(89);
        }
        fn change_resource_budget_digest(value: &mut PairedRunIdentity) {
            value.resource_budget_sha256 = digest(90);
        }
        fn change_model_configuration_digest(value: &mut PairedRunIdentity) {
            value.model_configuration_sha256 = digest(86);
        }
        fn change_candidate_policy_digest(value: &mut PairedRunIdentity) {
            value.candidate_policy_sha256 = digest(87);
        }
        fn change_policy_seed(value: &mut PairedRunIdentity) {
            value.policy_seed += 1;
        }

        let cases: [IdentityMutation; 13] = [
            (IdentityField::EpisodeId, change_episode_id),
            (IdentityField::HiddenWorldId, change_world_id),
            (IdentityField::HiddenWorldDigest, change_hidden_world_digest),
            (
                IdentityField::InitialStateDigest,
                change_initial_state_digest,
            ),
            (IdentityField::TaskDigest, change_task_digest),
            (IdentityField::LossScaleDigest, change_loss_scale_digest),
            (
                IdentityField::EvaluatorProfileDigest,
                change_evaluator_profile_digest,
            ),
            (
                IdentityField::ExecutionEnvironmentDigest,
                change_execution_environment_digest,
            ),
            (
                IdentityField::ActionDomainDigest,
                change_action_domain_digest,
            ),
            (
                IdentityField::ResourceBudgetDigest,
                change_resource_budget_digest,
            ),
            (
                IdentityField::ModelConfigurationDigest,
                change_model_configuration_digest,
            ),
            (
                IdentityField::CandidatePolicyDigest,
                change_candidate_policy_digest,
            ),
            (IdentityField::PolicySeed, change_policy_seed),
        ];

        for (expected_field, mutate) in cases {
            let mut rows = outcomes([400_000, 350_000, 380_000, 100_000]);
            mutate(&mut rows[2].identity);
            assert_eq!(
                evaluate_decision_value_interaction(&rows),
                Err(InteractionError::IdentityMismatch {
                    field: expected_field,
                }),
                "mismatched identity field {expected_field:?} must be rejected"
            );
        }
    }

    #[test]
    fn out_of_range_regret_is_rejected() {
        let mut rows = outcomes([400_000, 350_000, 380_000, 100_000]);
        rows[0].realized_regret_ppm = 1_000_001;
        assert!(matches!(
            evaluate_decision_value_interaction(&rows),
            Err(InteractionError::RegretOutOfRange {
                arm: TreatmentArm::Baseline,
                regret_ppm: 1_000_001
            })
        ));
    }

    #[test]
    fn empty_identity_digest_is_rejected() {
        let mut rows = outcomes([400_000, 350_000, 380_000, 100_000]);
        rows[0].identity.task_sha256 = [0; 32];
        assert!(matches!(
            evaluate_decision_value_interaction(&rows),
            Err(InteractionError::EmptyIdentityDigest {
                field: IdentityField::TaskDigest
            })
        ));
    }
}
