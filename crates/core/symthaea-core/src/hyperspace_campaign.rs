// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! HYPERSPACE-001 H0-H6 intervention campaign over the analytic evaluator.
//!
//! Oracle truth and planner observations are recorded separately. Analytic
//! blocked arms never use sampling failure as proof, and the target R4 planner
//! never receives the evaluator-only constructive escape witness.

use blake3::Hasher;
use thiserror::Error;

use crate::continuous_reachability::{
    AnalyticBoxValidityOracle, AxisAlignedBox, ContinuousPathReplay,
    ContinuousReachabilityError, ContinuousValidityOracle, EuclideanPlanningProblem,
    validate_euclidean_path,
};
use crate::hyperspace_benchmark::{
    FiniteWShellOracle, HyperspaceBenchmarkError, HyperspaceDimension,
    canonical_shell_problem, certify_radial_separation, evaluator_reference_escape_path,
    qualify_projection_trap,
};
use crate::reachability::UnknownReason;
use crate::sampling_reachability::{
    RrtConfig, SamplingReachabilityError, SamplingReachabilityResult, seeded_rrt,
};

/// Canonical HYPERSPACE-001 campaign arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyperspaceArm {
    /// R2 warm-up detour planning fixture.
    H0,
    /// R3 closed shell: analytically blocked.
    H1,
    /// R4 sufficient-w fixture: constructive existence truth.
    H2,
    /// R4 with w motion disabled: analytically blocked.
    H3,
    /// R4 with insufficient w range: analytically blocked.
    H4,
    /// R4 sufficient w range: target seeded RRT discovery arm.
    H5,
    /// Projection trap evaluated on the target-discovered H5 path.
    H6,
}

impl HyperspaceArm {
    fn tag(self) -> u8 {
        match self {
            Self::H0 => 0,
            Self::H1 => 1,
            Self::H2 => 2,
            Self::H3 => 3,
            Self::H4 => 4,
            Self::H5 => 5,
            Self::H6 => 6,
        }
    }
}

/// Independently established truth/evidence for one arm.
#[derive(Clone, Debug, PartialEq)]
pub enum ArmTruth {
    /// A known valid path independently establishes existence.
    ConstructivelyFeasible {
        /// Independently replayed path identity.
        path_identity: [u8; 32],
        /// Independently recomputed Euclidean path length.
        total_cost: f64,
    },
    /// An analytic radial-separation theorem establishes impossibility in the exact model.
    AnalyticallyBlocked {
        /// Independently verified certificate identity.
        certificate_identity: [u8; 32],
    },
    /// A full R4-valid path was independently shown to collide after xyz projection.
    ProjectionTrapEstablished {
        /// Exact projection-trap receipt identity.
        receipt_identity: [u8; 32],
        /// Exact full-dimensional planner path identity.
        full_path_identity: [u8; 32],
    },
    /// The requested evidence depends on a planner result that was not established.
    NotEstablished,
}

/// Observation made by the target planner, separate from oracle truth.
#[derive(Clone, Debug, PartialEq)]
pub enum PlannerObservation {
    /// A candidate path survived independent replay.
    Feasible {
        /// Exact replayed path identity.
        path_identity: [u8; 32],
        /// Exact recomputed path cost.
        total_cost: f64,
        /// Exact solver/config identity.
        solver_identity: [u8; 32],
        /// Executed expansion iterations.
        iterations: u32,
        /// Maximum absolute fourth coordinate used by the path when applicable.
        max_abs_w: Option<f64>,
    },
    /// The bounded search remained epistemically unresolved.
    Unknown {
        /// Proof-strength unknown class.
        reason: UnknownReason,
        /// Exact solver/config identity.
        solver_identity: [u8; 32],
        /// Executed expansion iterations.
        iterations: u32,
        /// Number of accepted tree nodes.
        accepted_nodes: u64,
    },
    /// Planner was deliberately not run for this arm.
    NotRun {
        /// Exact reason the target planner was omitted.
        reason: String,
    },
}

/// Exact receipt for one campaign arm.
#[derive(Clone, Debug, PartialEq)]
pub struct HyperspaceArmReceipt {
    arm: HyperspaceArm,
    problem_identity: [u8; 32],
    oracle_identity: [u8; 32],
    truth: ArmTruth,
    planner: PlannerObservation,
    identity: [u8; 32],
}

impl HyperspaceArmReceipt {
    /// Arm identifier.
    pub fn arm(&self) -> HyperspaceArm {
        self.arm
    }
    /// Exact problem identity.
    pub fn problem_identity(&self) -> [u8; 32] {
        self.problem_identity
    }
    /// Exact oracle identity.
    pub fn oracle_identity(&self) -> [u8; 32] {
        self.oracle_identity
    }
    /// Independently established oracle/evaluator truth.
    pub fn truth(&self) -> &ArmTruth {
        &self.truth
    }
    /// Target-planner observation.
    pub fn planner(&self) -> &PlannerObservation {
        &self.planner
    }
    /// Exact arm-receipt identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Exact deterministic campaign configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HyperspaceCampaignConfig {
    warmup_rrt: RrtConfig,
    blocked_rrt: RrtConfig,
    escape_rrt: RrtConfig,
}

impl HyperspaceCampaignConfig {
    /// Construct a campaign from exact solver profiles.
    pub fn new(warmup_rrt: RrtConfig, blocked_rrt: RrtConfig, escape_rrt: RrtConfig) -> Self {
        Self {
            warmup_rrt,
            blocked_rrt,
            escape_rrt,
        }
    }

    /// Named deterministic reference campaign.
    pub fn reference_v1() -> Self {
        Self {
            warmup_rrt: RrtConfig::new(7, 4_000, 0.35, 0.05, 0.5)
                .expect("reference warmup RRT config is valid"),
            blocked_rrt: RrtConfig::new(11, 2_000, 0.4, 0.05, 0.6)
                .expect("reference blocked RRT config is valid"),
            escape_rrt: RrtConfig::new(42, 5_000, 0.75, 0.10, 5.0)
                .expect("reference escape RRT config is valid"),
        }
    }

    /// Deterministic campaign identity.
    pub fn identity(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-hyperspace-001-campaign-config-v1\0");
        hasher.update(&self.warmup_rrt.identity());
        hasher.update(&self.blocked_rrt.identity());
        hasher.update(&self.escape_rrt.identity());
        *hasher.finalize().as_bytes()
    }
}

impl Default for HyperspaceCampaignConfig {
    fn default() -> Self {
        Self::reference_v1()
    }
}

/// Complete H0-H6 campaign receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct HyperspaceCampaignReceipt {
    config_identity: [u8; 32],
    arms: Vec<HyperspaceArmReceipt>,
    identity: [u8; 32],
}

impl HyperspaceCampaignReceipt {
    /// Exact campaign configuration identity.
    pub fn config_identity(&self) -> [u8; 32] {
        self.config_identity
    }

    /// Ordered H0-H6 receipts.
    pub fn arms(&self) -> &[HyperspaceArmReceipt] {
        &self.arms
    }

    /// Find one named arm.
    pub fn arm(&self, arm: HyperspaceArm) -> Option<&HyperspaceArmReceipt> {
        self.arms.iter().find(|receipt| receipt.arm == arm)
    }

    /// Whether target RRT independently established the H5 R4 route.
    pub fn h5_target_feasible(&self) -> bool {
        self.arm(HyperspaceArm::H5).is_some_and(|receipt| {
            matches!(&receipt.planner, PlannerObservation::Feasible { .. })
        })
    }

    /// Whether H6 projection-trap evidence was established on that target path.
    pub fn h6_projection_trap_established(&self) -> bool {
        self.arm(HyperspaceArm::H6).is_some_and(|receipt| {
            matches!(&receipt.truth, ArmTruth::ProjectionTrapEstablished { .. })
        })
    }

    /// Exact campaign receipt identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Fail-closed campaign errors.
#[derive(Debug, Error)]
pub enum HyperspaceCampaignError {
    /// A planner returned feasible in an independently certified blocked arm.
    #[error("false feasible planner result in analytically blocked arm {arm:?}")]
    FalseFeasible {
        /// Analytically blocked arm that produced a false feasible planner result.
        arm: HyperspaceArm,
    },
    /// An evaluator-only constructive witness failed independent replay.
    #[error("constructive truth witness failed replay for {arm:?}: {reason}")]
    ConstructiveWitnessInvalid {
        /// Campaign arm whose evaluator witness failed.
        arm: HyperspaceArm,
        /// Independent replay failure reason.
        reason: String,
    },
    /// Shared HYPERSPACE evaluator failed.
    #[error(transparent)]
    Hyperspace(#[from] HyperspaceBenchmarkError),
    /// Shared continuous planner/evaluator contract failed.
    #[error(transparent)]
    Continuous(#[from] ContinuousReachabilityError),
    /// Sampling planner failed structurally.
    #[error(transparent)]
    Sampling(#[from] SamplingReachabilityError),
}

/// Execute the exact H0-H6 campaign.
pub fn run_hyperspace_campaign(
    config: HyperspaceCampaignConfig,
) -> Result<HyperspaceCampaignReceipt, HyperspaceCampaignError> {
    let mut arms = Vec::with_capacity(7);

    let warmup_oracle = warmup_oracle()?;
    let warmup_problem = EuclideanPlanningProblem::new(
        2,
        vec![0.0, 0.0],
        vec![3.0, 0.0],
        &warmup_oracle,
    )?;
    let warmup_witness = vec![vec![0.0, 0.0], vec![1.5, 1.0], vec![3.0, 0.0]];
    let warmup_truth = constructive_truth(
        HyperspaceArm::H0,
        &warmup_problem,
        &warmup_oracle,
        &warmup_witness,
    )?;
    let warmup_planner = planner_observation(
        seeded_rrt(&warmup_problem, &warmup_oracle, config.warmup_rrt)?,
        None,
    );
    arms.push(build_arm(
        HyperspaceArm::H0,
        &warmup_problem,
        &warmup_oracle,
        warmup_truth,
        warmup_planner,
    ));

    let inner = 1.0;
    let outer = 2.0;
    let w_shell = 0.25;
    let xyz_bound = 4.0;
    let goal_x = 3.0;

    let h1_oracle = FiniteWShellOracle::new(
        HyperspaceDimension::R3,
        inner,
        outer,
        0.0,
        xyz_bound,
        0.0,
    )?;
    let h1_problem = canonical_shell_problem(&h1_oracle, goal_x)?;
    let h1_certificate = certify_radial_separation(&h1_problem, &h1_oracle)?;
    let h1_planner = planner_observation(
        seeded_rrt(&h1_problem, &h1_oracle, config.blocked_rrt)?,
        None,
    );
    reject_false_feasible(HyperspaceArm::H1, &h1_planner)?;
    arms.push(build_arm(
        HyperspaceArm::H1,
        &h1_problem,
        &h1_oracle,
        ArmTruth::AnalyticallyBlocked {
            certificate_identity: h1_certificate.identity(),
        },
        h1_planner,
    ));

    let escape_oracle = FiniteWShellOracle::new(
        HyperspaceDimension::R4,
        inner,
        outer,
        w_shell,
        xyz_bound,
        2.0,
    )?;
    let escape_problem = canonical_shell_problem(&escape_oracle, goal_x)?;
    let evaluator_path = evaluator_reference_escape_path(&escape_problem, &escape_oracle, 0.25)?;
    let h2_truth = constructive_truth(
        HyperspaceArm::H2,
        &escape_problem,
        &escape_oracle,
        &evaluator_path,
    )?;
    arms.push(build_arm(
        HyperspaceArm::H2,
        &escape_problem,
        &escape_oracle,
        h2_truth,
        PlannerObservation::NotRun {
            reason: "H2 is evaluator-only existence qualification; target discovery is isolated to H5"
                .to_string(),
        },
    ));

    let h3_oracle = FiniteWShellOracle::new(
        HyperspaceDimension::R4,
        inner,
        outer,
        w_shell,
        xyz_bound,
        0.0,
    )?;
    let h3_problem = canonical_shell_problem(&h3_oracle, goal_x)?;
    let h3_certificate = certify_radial_separation(&h3_problem, &h3_oracle)?;
    arms.push(build_arm(
        HyperspaceArm::H3,
        &h3_problem,
        &h3_oracle,
        ArmTruth::AnalyticallyBlocked {
            certificate_identity: h3_certificate.identity(),
        },
        PlannerObservation::NotRun {
            reason: "RRT V1 does not sample a zero-width w axis; oracle truth remains analytic"
                .to_string(),
        },
    ));

    let h4_oracle = FiniteWShellOracle::new(
        HyperspaceDimension::R4,
        inner,
        outer,
        w_shell,
        xyz_bound,
        0.20,
    )?;
    let h4_problem = canonical_shell_problem(&h4_oracle, goal_x)?;
    let h4_certificate = certify_radial_separation(&h4_problem, &h4_oracle)?;
    let h4_planner = planner_observation(
        seeded_rrt(&h4_problem, &h4_oracle, config.blocked_rrt)?,
        Some(3),
    );
    reject_false_feasible(HyperspaceArm::H4, &h4_planner)?;
    arms.push(build_arm(
        HyperspaceArm::H4,
        &h4_problem,
        &h4_oracle,
        ArmTruth::AnalyticallyBlocked {
            certificate_identity: h4_certificate.identity(),
        },
        h4_planner,
    ));

    let h5_result = seeded_rrt(&escape_problem, &escape_oracle, config.escape_rrt)?;
    let (h5_planner, discovered_path) = planner_observation_with_path(h5_result, Some(3));
    arms.push(build_arm(
        HyperspaceArm::H5,
        &escape_problem,
        &escape_oracle,
        constructive_truth(
            HyperspaceArm::H5,
            &escape_problem,
            &escape_oracle,
            &evaluator_path,
        )?,
        h5_planner.clone(),
    ));

    let h6_truth = if let Some(path) = discovered_path {
        let trap = qualify_projection_trap(&escape_problem, &escape_oracle, &path)?;
        ArmTruth::ProjectionTrapEstablished {
            receipt_identity: trap.identity(),
            full_path_identity: trap.full_path_identity(),
        }
    } else {
        ArmTruth::NotEstablished
    };
    arms.push(build_arm(
        HyperspaceArm::H6,
        &escape_problem,
        &escape_oracle,
        h6_truth,
        h5_planner,
    ));

    let config_identity = config.identity();
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-hyperspace-001-campaign-receipt-v1\0");
    hasher.update(&config_identity);
    for arm in &arms {
        hasher.update(&arm.identity);
    }
    let identity = *hasher.finalize().as_bytes();
    Ok(HyperspaceCampaignReceipt {
        config_identity,
        arms,
        identity,
    })
}

fn warmup_oracle() -> Result<AnalyticBoxValidityOracle, ContinuousReachabilityError> {
    AnalyticBoxValidityOracle::new(
        AxisAlignedBox::new(vec![-0.5, -2.0], vec![3.5, 2.0])?,
        vec![AxisAlignedBox::new(vec![1.25, -0.35], vec![1.75, 0.35])?],
        Vec::new(),
    )
}

fn constructive_truth<O: ContinuousValidityOracle>(
    arm: HyperspaceArm,
    problem: &EuclideanPlanningProblem,
    oracle: &O,
    path: &[Vec<f64>],
) -> Result<ArmTruth, HyperspaceCampaignError> {
    match validate_euclidean_path(problem, oracle, path)? {
        ContinuousPathReplay::Valid(validation) => Ok(ArmTruth::ConstructivelyFeasible {
            path_identity: validation.path_identity(),
            total_cost: validation.total_cost(),
        }),
        ContinuousPathReplay::Invalid { reason, .. } => {
            Err(HyperspaceCampaignError::ConstructiveWitnessInvalid { arm, reason })
        }
        ContinuousPathReplay::Unknown { reason, .. } => {
            Err(HyperspaceCampaignError::ConstructiveWitnessInvalid {
                arm,
                reason: format!("constructive witness replay remained unknown: {reason}"),
            })
        }
    }
}

fn planner_observation(
    result: SamplingReachabilityResult,
    w_axis: Option<usize>,
) -> PlannerObservation {
    planner_observation_with_path(result, w_axis).0
}

fn planner_observation_with_path(
    result: SamplingReachabilityResult,
    w_axis: Option<usize>,
) -> (PlannerObservation, Option<Vec<Vec<f64>>>) {
    match result {
        SamplingReachabilityResult::Feasible {
            path,
            validation,
            solver,
        } => {
            let max_abs_w = w_axis.map(|axis| {
                path.iter()
                    .map(|state| state[axis].abs())
                    .fold(0.0_f64, f64::max)
            });
            (
                PlannerObservation::Feasible {
                    path_identity: validation.path_identity(),
                    total_cost: validation.total_cost(),
                    solver_identity: solver.solver_identity(),
                    iterations: solver.iterations(),
                    max_abs_w,
                },
                Some(path),
            )
        }
        SamplingReachabilityResult::Unknown { reason, solver, .. } => (
            PlannerObservation::Unknown {
                reason,
                solver_identity: solver.solver_identity(),
                iterations: solver.iterations(),
                accepted_nodes: solver.accepted_nodes(),
            },
            None,
        ),
    }
}

fn reject_false_feasible(
    arm: HyperspaceArm,
    observation: &PlannerObservation,
) -> Result<(), HyperspaceCampaignError> {
    if matches!(observation, PlannerObservation::Feasible { .. }) {
        Err(HyperspaceCampaignError::FalseFeasible { arm })
    } else {
        Ok(())
    }
}

fn build_arm<O: ContinuousValidityOracle>(
    arm: HyperspaceArm,
    problem: &EuclideanPlanningProblem,
    oracle: &O,
    truth: ArmTruth,
    planner: PlannerObservation,
) -> HyperspaceArmReceipt {
    let problem_identity = problem.identity();
    let oracle_identity = oracle.profile().identity();
    let identity = hash_arm_receipt(arm, problem_identity, oracle_identity, &truth, &planner);
    HyperspaceArmReceipt {
        arm,
        problem_identity,
        oracle_identity,
        truth,
        planner,
        identity,
    }
}

fn hash_arm_receipt(
    arm: HyperspaceArm,
    problem_identity: [u8; 32],
    oracle_identity: [u8; 32],
    truth: &ArmTruth,
    planner: &PlannerObservation,
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-hyperspace-001-arm-receipt-v1\0");
    hasher.update(&[arm.tag()]);
    hasher.update(&problem_identity);
    hasher.update(&oracle_identity);
    hash_truth(&mut hasher, truth);
    hash_planner(&mut hasher, planner);
    *hasher.finalize().as_bytes()
}

fn hash_truth(hasher: &mut Hasher, truth: &ArmTruth) {
    match truth {
        ArmTruth::ConstructivelyFeasible {
            path_identity,
            total_cost,
        } => {
            hasher.update(&[0]);
            hasher.update(path_identity);
            hasher.update(&total_cost.to_bits().to_le_bytes());
        }
        ArmTruth::AnalyticallyBlocked {
            certificate_identity,
        } => {
            hasher.update(&[1]);
            hasher.update(certificate_identity);
        }
        ArmTruth::ProjectionTrapEstablished {
            receipt_identity,
            full_path_identity,
        } => {
            hasher.update(&[2]);
            hasher.update(receipt_identity);
            hasher.update(full_path_identity);
        }
        ArmTruth::NotEstablished => {
            hasher.update(&[3]);
        }
    }
}

fn hash_planner(hasher: &mut Hasher, planner: &PlannerObservation) {
    match planner {
        PlannerObservation::Feasible {
            path_identity,
            total_cost,
            solver_identity,
            iterations,
            max_abs_w,
        } => {
            hasher.update(&[0]);
            hasher.update(path_identity);
            hasher.update(&total_cost.to_bits().to_le_bytes());
            hasher.update(solver_identity);
            hasher.update(&iterations.to_le_bytes());
            match max_abs_w {
                Some(value) => {
                    hasher.update(&[1]);
                    hasher.update(&value.to_bits().to_le_bytes());
                }
                None => {
                    hasher.update(&[0]);
                }
            }
        }
        PlannerObservation::Unknown {
            reason,
            solver_identity,
            iterations,
            accepted_nodes,
        } => {
            hasher.update(&[1, unknown_reason_tag(reason)]);
            hasher.update(solver_identity);
            hasher.update(&iterations.to_le_bytes());
            hasher.update(&accepted_nodes.to_le_bytes());
        }
        PlannerObservation::NotRun { reason } => {
            hasher.update(&[2]);
            hasher.update(&(reason.len() as u64).to_le_bytes());
            hasher.update(reason.as_bytes());
        }
    }
}

fn unknown_reason_tag(reason: &UnknownReason) -> u8 {
    match reason {
        UnknownReason::BudgetExceeded => 0,
        UnknownReason::NoPathFound => 1,
        UnknownReason::ValidityOracleIncomplete => 2,
        UnknownReason::NumericalFailure => 3,
        UnknownReason::ModelOutOfDomain => 4,
        UnknownReason::ConstraintProjectionFailure => 5,
        UnknownReason::UnsupportedSpace => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_campaign_preserves_proof_strength_and_discovers_escape() {
        let receipt = run_hyperspace_campaign(HyperspaceCampaignConfig::reference_v1()).unwrap();

        for arm in [HyperspaceArm::H1, HyperspaceArm::H3, HyperspaceArm::H4] {
            let arm = receipt.arm(arm).unwrap();
            assert!(matches!(arm.truth(), ArmTruth::AnalyticallyBlocked { .. }));
            assert!(!matches!(arm.planner(), PlannerObservation::Feasible { .. }));
        }

        let h2 = receipt.arm(HyperspaceArm::H2).unwrap();
        assert!(matches!(h2.truth(), ArmTruth::ConstructivelyFeasible { .. }));
        assert!(matches!(h2.planner(), PlannerObservation::NotRun { .. }));

        assert!(
            receipt.h5_target_feasible(),
            "reference seeded RRT did not establish H5; exact campaign identity={:?}",
            receipt.identity()
        );
        assert!(
            receipt.h6_projection_trap_established(),
            "H6 projection trap was not established on the target-discovered path"
        );
        let h5 = receipt.arm(HyperspaceArm::H5).unwrap();
        let PlannerObservation::Feasible { max_abs_w, .. } = h5.planner() else {
            unreachable!("h5_target_feasible already checked")
        };
        assert!(max_abs_w.is_some_and(|value| value > 0.25));
    }

    #[test]
    fn campaign_identity_binds_solver_seed_and_budget() {
        let a = HyperspaceCampaignConfig::reference_v1();
        let b = HyperspaceCampaignConfig::new(
            RrtConfig::new(8, 4_000, 0.35, 0.05, 0.5).unwrap(),
            a.blocked_rrt,
            a.escape_rrt,
        );
        assert_ne!(a.identity(), b.identity());
    }
}
