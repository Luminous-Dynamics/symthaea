//! SLO-qualified mission productivity and evidence-preserving fleet comparison.
//!
//! This is an offline benchmark evaluator. It does not implement a scheduler,
//! robot controller, physical model, runtime assurance monitor, or actuator
//! authority. It keeps task classes and evidence domains separate and refuses
//! to turn Unknown measurements into zero-valued successes.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

const PPM: u128 = 1_000_000;
const MICROS_PER_GPU_HOUR: u128 = 3_600_000_000;
const NANOS_PER_ROBOT_HOUR: u128 = 3_600_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceState {
    Qualified,
    Unknown,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MatchIdentity {
    pub pair_id: String,
    pub robot_id: String,
    pub task_class: String,
    pub work_unit_id: String,
    pub workload_digest: String,
    pub scenario_digest: String,
    pub seed_id: String,
    pub initial_state_digest: String,
    pub model_digest: String,
    pub control_policy_digest: String,
    pub hardware_digest: String,
    pub observation_budget_digest: String,
    pub compute_budget_digest: String,
}

/// One independent mission/perturbation-sequence outcome for one scheduler arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionTrial {
    pub mission_id: String,
    pub arm_id: String,
    pub identity: MatchIdentity,
    /// Wall-clock mission duration; all failed and incomplete missions remain included.
    pub elapsed_ns: u64,
    /// GPU service consumed by this robot mission, including work that did not succeed.
    pub gpu_micros: u64,
    /// Task-class-specific useful work. Only SLO-qualified completed work is credited.
    pub useful_work_units: u64,
    pub task_completed: bool,
    pub terminal_safe: bool,
    pub hard_safety_violation: bool,
    pub critical_deadline_miss: bool,
    /// Longest interval without a valid action in this mission.
    pub longest_starvation_ticks: u64,
    /// Declared maximum acceptable no-valid-action interval for this robot/task.
    pub starvation_limit_ticks: u64,
    pub thermal_evidence: EvidenceState,
    pub recovery_evidence: EvidenceState,
    /// Upstream energy boundary state. Energy is never inferred from task or GPU metrics.
    pub energy_evidence: EvidenceState,
    /// Present exactly when energy_evidence is Qualified.
    pub measured_energy_joules: Option<u64>,
    /// Counts imported from the causal mission replay receipt.
    pub actuator_applications_examined: u64,
    pub actuator_applications_linked: u64,
    pub actuator_applications_causally_closed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationError {
    EmptyIdentity(&'static str),
    DuplicateMissionId(String),
    DuplicatePairId(String),
    WrongArm { expected: String, observed: String },
    InvalidDuration(String),
    InvalidGpuDemand(String),
    InvalidStarvationLimit(String),
    InvalidApplicationCounts(String),
    InvalidEnergyReceipt(String),
    InvalidClosureThreshold(u32),
    EmptyTrialSet,
    PairSetMismatch(String),
    PairIdentityMismatch(String),
    MetricOverflow,
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for EvaluationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScorecardContract {
    /// Minimum per-mission physical causal closure required for SLO credit.
    pub minimum_causal_closure_ppm: u32,
}

impl ScorecardContract {
    pub fn validate(self) -> Result<(), EvaluationError> {
        if self.minimum_causal_closure_ppm == 0
            || self.minimum_causal_closure_ppm > 1_000_000
        {
            return Err(EvaluationError::InvalidClosureThreshold(
                self.minimum_causal_closure_ppm,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalClosureTotals {
    pub applications_examined: u64,
    pub applications_linked: u64,
    pub applications_causally_closed: u64,
    pub closure_ppm: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskClassScorecard {
    pub task_class: String,
    pub work_unit_id: String,
    pub missions: u64,
    pub task_completions: u64,
    pub physical_safe_successes: u64,
    pub evidence_qualified_safe_successes: u64,
    pub slo_qualified_missions: u64,
    pub critical_deadline_misses: u64,
    pub hard_safety_violations: u64,
    pub service_floor_breaches: u64,
    pub productive_work_units: u128,
    pub gpu_micros: u128,
    pub elapsed_ns: u128,
    /// Fixed-point milli-work-units per GPU-hour; None if no GPU service was measured.
    pub work_units_per_gpu_hour_milli: Option<u64>,
    /// Fixed-point milli-work-units per robot-hour; uses all mission wall time.
    pub work_units_per_robot_hour_milli: Option<u64>,
    pub thermal_unknown_or_invalid: u64,
    pub recovery_unknown_or_invalid: u64,
    pub energy_qualified_missions: u64,
    pub energy_unknown_or_invalid_missions: u64,
    pub known_energy_joules: u128,
    pub causal_closure: CausalClosureTotals,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobotServiceScorecard {
    pub robot_id: String,
    pub missions: u64,
    pub physical_safe_successes: u64,
    pub slo_qualified_missions: u64,
    pub hard_safety_violations: u64,
    pub critical_deadline_misses: u64,
    pub service_floor_breaches: u64,
    pub longest_starvation_ticks: u64,
    pub causal_closure: CausalClosureTotals,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FleetScorecard {
    pub arm_id: String,
    pub missions: u64,
    pub task_completions: u64,
    pub physical_safe_successes: u64,
    pub evidence_qualified_safe_successes: u64,
    pub slo_qualified_missions: u64,
    pub critical_deadline_misses: u64,
    pub hard_safety_violations: u64,
    pub service_floor_breaches: u64,
    pub causal_closure: CausalClosureTotals,
    pub task_classes: BTreeMap<String, TaskClassScorecard>,
    pub robots: BTreeMap<String, RobotServiceScorecard>,
}

fn nonempty(value: &str, field: &'static str) -> Result<(), EvaluationError> {
    if value.trim().is_empty() {
        return Err(EvaluationError::EmptyIdentity(field));
    }
    Ok(())
}

fn checked_add(target: &mut u64, value: u64) -> Result<(), EvaluationError> {
    *target = target.checked_add(value).ok_or(EvaluationError::MetricOverflow)?;
    Ok(())
}

fn checked_add_u128(target: &mut u128, value: u128) -> Result<(), EvaluationError> {
    *target = target.checked_add(value).ok_or(EvaluationError::MetricOverflow)?;
    Ok(())
}

fn ppm(numerator: u64, denominator: u64) -> Option<u32> {
    if denominator == 0 {
        None
    } else {
        Some(((u128::from(numerator) * PPM) / u128::from(denominator)) as u32)
    }
}

fn closure_ppm(examined: u64, closed: u64) -> Option<u32> {
    ppm(closed, examined)
}

fn closure_is_qualified(
    trial: &MissionTrial,
    minimum_causal_closure_ppm: u32,
) -> bool {
    trial.actuator_applications_examined > 0
        && trial.actuator_applications_linked == trial.actuator_applications_examined
        && ppm(
            trial.actuator_applications_causally_closed,
            trial.actuator_applications_examined,
        )
        .is_some_and(|value| value >= minimum_causal_closure_ppm)
}

fn physical_safe_success(trial: &MissionTrial) -> bool {
    trial.task_completed && trial.terminal_safe && !trial.hard_safety_violation
}

fn evidence_qualified_safe_success(
    trial: &MissionTrial,
    contract: ScorecardContract,
) -> bool {
    physical_safe_success(trial)
        && trial.thermal_evidence == EvidenceState::Qualified
        && trial.recovery_evidence == EvidenceState::Qualified
        && closure_is_qualified(trial, contract.minimum_causal_closure_ppm)
}

fn slo_qualified(trial: &MissionTrial, contract: ScorecardContract) -> bool {
    evidence_qualified_safe_success(trial, contract)
        && !trial.critical_deadline_miss
        && trial.longest_starvation_ticks <= trial.starvation_limit_ticks
}

fn validate_trial(
    trial: &MissionTrial,
    expected_arm: &str,
    mission_ids: &mut BTreeSet<String>,
    pair_ids: &mut BTreeSet<String>,
) -> Result<(), EvaluationError> {
    if trial.arm_id != expected_arm {
        return Err(EvaluationError::WrongArm {
            expected: expected_arm.to_owned(),
            observed: trial.arm_id.clone(),
        });
    }
    for (name, value) in [
        ("mission_id", trial.mission_id.as_str()),
        ("arm_id", trial.arm_id.as_str()),
        ("pair_id", trial.identity.pair_id.as_str()),
        ("robot_id", trial.identity.robot_id.as_str()),
        ("task_class", trial.identity.task_class.as_str()),
        ("work_unit_id", trial.identity.work_unit_id.as_str()),
        ("workload_digest", trial.identity.workload_digest.as_str()),
        ("scenario_digest", trial.identity.scenario_digest.as_str()),
        ("seed_id", trial.identity.seed_id.as_str()),
        ("initial_state_digest", trial.identity.initial_state_digest.as_str()),
        ("model_digest", trial.identity.model_digest.as_str()),
        ("control_policy_digest", trial.identity.control_policy_digest.as_str()),
        ("hardware_digest", trial.identity.hardware_digest.as_str()),
        (
            "observation_budget_digest",
            trial.identity.observation_budget_digest.as_str(),
        ),
        (
            "compute_budget_digest",
            trial.identity.compute_budget_digest.as_str(),
        ),
    ] {
        nonempty(value, name)?;
    }
    if !mission_ids.insert(trial.mission_id.clone()) {
        return Err(EvaluationError::DuplicateMissionId(trial.mission_id.clone()));
    }
    if !pair_ids.insert(trial.identity.pair_id.clone()) {
        return Err(EvaluationError::DuplicatePairId(trial.identity.pair_id.clone()));
    }
    if trial.elapsed_ns == 0 {
        return Err(EvaluationError::InvalidDuration(trial.mission_id.clone()));
    }
    if trial.gpu_micros == 0 {
        return Err(EvaluationError::InvalidGpuDemand(trial.mission_id.clone()));
    }
    if trial.starvation_limit_ticks == 0 {
        return Err(EvaluationError::InvalidStarvationLimit(
            trial.mission_id.clone(),
        ));
    }
    if trial.actuator_applications_linked > trial.actuator_applications_examined
        || trial.actuator_applications_causally_closed > trial.actuator_applications_linked
    {
        return Err(EvaluationError::InvalidApplicationCounts(
            trial.mission_id.clone(),
        ));
    }
    match (trial.energy_evidence, trial.measured_energy_joules) {
        (EvidenceState::Qualified, Some(_)) => {}
        (EvidenceState::Qualified, None) | (_, Some(_)) => {
            return Err(EvaluationError::InvalidEnergyReceipt(
                trial.mission_id.clone(),
            ));
        }
        (EvidenceState::Unknown | EvidenceState::Invalid, None) => {}
    }
    Ok(())
}

#[derive(Default)]
struct ClassAccumulator {
    missions: u64,
    task_completions: u64,
    physical_safe_successes: u64,
    evidence_qualified_safe_successes: u64,
    slo_qualified_missions: u64,
    critical_deadline_misses: u64,
    hard_safety_violations: u64,
    service_floor_breaches: u64,
    productive_work_units: u128,
    gpu_micros: u128,
    elapsed_ns: u128,
    thermal_unknown_or_invalid: u64,
    recovery_unknown_or_invalid: u64,
    energy_qualified_missions: u64,
    energy_unknown_or_invalid_missions: u64,
    known_energy_joules: u128,
    applications_examined: u64,
    applications_linked: u64,
    applications_causally_closed: u64,
}

#[derive(Default)]
struct RobotAccumulator {
    missions: u64,
    physical_safe_successes: u64,
    slo_qualified_missions: u64,
    hard_safety_violations: u64,
    critical_deadline_misses: u64,
    service_floor_breaches: u64,
    longest_starvation_ticks: u64,
    applications_examined: u64,
    applications_linked: u64,
    applications_causally_closed: u64,
}

/// Aggregate independent mission outcomes without averaging unlike task classes.
pub fn evaluate_fleet(
    arm_id: &str,
    trials: &[MissionTrial],
    contract: ScorecardContract,
) -> Result<FleetScorecard, EvaluationError> {
    contract.validate()?;
    nonempty(arm_id, "arm_id")?;
    if trials.is_empty() {
        return Err(EvaluationError::EmptyTrialSet);
    }
    let mut mission_ids = BTreeSet::new();
    let mut pair_ids = BTreeSet::new();
    let mut task_work_units: BTreeMap<String, String> = BTreeMap::new();

    let mut total = FleetScorecard {
        arm_id: arm_id.to_owned(),
        missions: 0,
        task_completions: 0,
        physical_safe_successes: 0,
        evidence_qualified_safe_successes: 0,
        slo_qualified_missions: 0,
        critical_deadline_misses: 0,
        hard_safety_violations: 0,
        service_floor_breaches: 0,
        causal_closure: CausalClosureTotals {
            applications_examined: 0,
            applications_linked: 0,
            applications_causally_closed: 0,
            closure_ppm: None,
        },
        task_classes: BTreeMap::new(),
        robots: BTreeMap::new(),
    };
    let mut classes: BTreeMap<String, ClassAccumulator> = BTreeMap::new();
    let mut robots: BTreeMap<String, RobotAccumulator> = BTreeMap::new();

    for trial in trials {
        validate_trial(trial, arm_id, &mut mission_ids, &mut pair_ids)?;
        if let Some(existing_unit) = task_work_units.get(&trial.identity.task_class)
            && existing_unit != &trial.identity.work_unit_id
        {
            return Err(EvaluationError::PairIdentityMismatch(format!(
                "incomparable work units within task class {}",
                trial.identity.task_class
            )));
        }
        task_work_units
            .entry(trial.identity.task_class.clone())
            .or_insert_with(|| trial.identity.work_unit_id.clone());
        let physical_safe = physical_safe_success(trial);
        let evidence_safe = evidence_qualified_safe_success(trial, contract);
        let slo = slo_qualified(trial, contract);
        let starvation_breach =
            trial.longest_starvation_ticks > trial.starvation_limit_ticks;

        checked_add(&mut total.missions, 1)?;
        if trial.task_completed {
            checked_add(&mut total.task_completions, 1)?;
        }
        if physical_safe {
            checked_add(&mut total.physical_safe_successes, 1)?;
        }
        if evidence_safe {
            checked_add(&mut total.evidence_qualified_safe_successes, 1)?;
        }
        if slo {
            checked_add(&mut total.slo_qualified_missions, 1)?;
        }
        if trial.critical_deadline_miss {
            checked_add(&mut total.critical_deadline_misses, 1)?;
        }
        if trial.hard_safety_violation {
            checked_add(&mut total.hard_safety_violations, 1)?;
        }
        if starvation_breach {
            checked_add(&mut total.service_floor_breaches, 1)?;
        }
        checked_add(
            &mut total.causal_closure.applications_examined,
            trial.actuator_applications_examined,
        )?;
        checked_add(
            &mut total.causal_closure.applications_linked,
            trial.actuator_applications_linked,
        )?;
        checked_add(
            &mut total.causal_closure.applications_causally_closed,
            trial.actuator_applications_causally_closed,
        )?;

        let class = classes
            .entry(trial.identity.task_class.clone())
            .or_default();
        checked_add(&mut class.missions, 1)?;
        if trial.task_completed {
            checked_add(&mut class.task_completions, 1)?;
        }
        if physical_safe {
            checked_add(&mut class.physical_safe_successes, 1)?;
        }
        if evidence_safe {
            checked_add(&mut class.evidence_qualified_safe_successes, 1)?;
        }
        if slo {
            checked_add(&mut class.slo_qualified_missions, 1)?;
            checked_add_u128(
                &mut class.productive_work_units,
                u128::from(trial.useful_work_units),
            )?;
        }
        checked_add_u128(&mut class.gpu_micros, u128::from(trial.gpu_micros))?;
        checked_add_u128(&mut class.elapsed_ns, u128::from(trial.elapsed_ns))?;
        if trial.critical_deadline_miss {
            checked_add(&mut class.critical_deadline_misses, 1)?;
        }
        if trial.hard_safety_violation {
            checked_add(&mut class.hard_safety_violations, 1)?;
        }
        if starvation_breach {
            checked_add(&mut class.service_floor_breaches, 1)?;
        }
        if trial.thermal_evidence != EvidenceState::Qualified {
            checked_add(&mut class.thermal_unknown_or_invalid, 1)?;
        }
        if trial.recovery_evidence != EvidenceState::Qualified {
            checked_add(&mut class.recovery_unknown_or_invalid, 1)?;
        }
        if trial.energy_evidence == EvidenceState::Qualified {
            checked_add(&mut class.energy_qualified_missions, 1)?;
            checked_add_u128(
                &mut class.known_energy_joules,
                u128::from(trial.measured_energy_joules.unwrap_or(0)),
            )?;
        } else {
            checked_add(&mut class.energy_unknown_or_invalid_missions, 1)?;
        }
        checked_add(
            &mut class.applications_examined,
            trial.actuator_applications_examined,
        )?;
        checked_add(
            &mut class.applications_linked,
            trial.actuator_applications_linked,
        )?;
        checked_add(
            &mut class.applications_causally_closed,
            trial.actuator_applications_causally_closed,
        )?;

        let robot = robots
            .entry(trial.identity.robot_id.clone())
            .or_default();
        checked_add(&mut robot.missions, 1)?;
        if physical_safe {
            checked_add(&mut robot.physical_safe_successes, 1)?;
        }
        if slo {
            checked_add(&mut robot.slo_qualified_missions, 1)?;
        }
        if trial.hard_safety_violation {
            checked_add(&mut robot.hard_safety_violations, 1)?;
        }
        if trial.critical_deadline_miss {
            checked_add(&mut robot.critical_deadline_misses, 1)?;
        }
        if starvation_breach {
            checked_add(&mut robot.service_floor_breaches, 1)?;
        }
        robot.longest_starvation_ticks = robot
            .longest_starvation_ticks
            .max(trial.longest_starvation_ticks);
        checked_add(
            &mut robot.applications_examined,
            trial.actuator_applications_examined,
        )?;
        checked_add(
            &mut robot.applications_linked,
            trial.actuator_applications_linked,
        )?;
        checked_add(
            &mut robot.applications_causally_closed,
            trial.actuator_applications_causally_closed,
        )?;
    }

    total.causal_closure.closure_ppm = closure_ppm(
        total.causal_closure.applications_examined,
        total.causal_closure.applications_causally_closed,
    );
    for (task_class, a) in classes {
        let gpu_hour_rate = if a.gpu_micros == 0 {
            None
        } else {
            let milli = a
                .productive_work_units
                .checked_mul(MICROS_PER_GPU_HOUR * 1_000)
                .ok_or(EvaluationError::MetricOverflow)?
                / a.gpu_micros;
            Some(u64::try_from(milli).map_err(|_| EvaluationError::MetricOverflow)?)
        };
        let robot_hour_rate = if a.elapsed_ns == 0 {
            None
        } else {
            let milli = a
                .productive_work_units
                .checked_mul(NANOS_PER_ROBOT_HOUR * 1_000)
                .ok_or(EvaluationError::MetricOverflow)?
                / a.elapsed_ns;
            Some(u64::try_from(milli).map_err(|_| EvaluationError::MetricOverflow)?)
        };
        total.task_classes.insert(
            task_class.clone(),
            TaskClassScorecard {
                task_class,
                work_unit_id: trials
                    .iter()
                    .find(|t| t.identity.task_class == task_class)
                    .map(|t| t.identity.work_unit_id.clone())
                    .unwrap_or_default(),
                missions: a.missions,
                task_completions: a.task_completions,
                physical_safe_successes: a.physical_safe_successes,
                evidence_qualified_safe_successes: a.evidence_qualified_safe_successes,
                slo_qualified_missions: a.slo_qualified_missions,
                critical_deadline_misses: a.critical_deadline_misses,
                hard_safety_violations: a.hard_safety_violations,
                service_floor_breaches: a.service_floor_breaches,
                productive_work_units: a.productive_work_units,
                gpu_micros: a.gpu_micros,
                elapsed_ns: a.elapsed_ns,
                work_units_per_gpu_hour_milli: gpu_hour_rate,
                work_units_per_robot_hour_milli: robot_hour_rate,
                thermal_unknown_or_invalid: a.thermal_unknown_or_invalid,
                recovery_unknown_or_invalid: a.recovery_unknown_or_invalid,
                energy_qualified_missions: a.energy_qualified_missions,
                energy_unknown_or_invalid_missions: a.energy_unknown_or_invalid_missions,
                known_energy_joules: a.known_energy_joules,
                causal_closure: CausalClosureTotals {
                    applications_examined: a.applications_examined,
                    applications_linked: a.applications_linked,
                    applications_causally_closed: a.applications_causally_closed,
                    closure_ppm: closure_ppm(
                        a.applications_examined,
                        a.applications_causally_closed,
                    ),
                },
            },
        );
    }
    for (robot_id, a) in robots {
        total.robots.insert(
            robot_id.clone(),
            RobotServiceScorecard {
                robot_id,
                missions: a.missions,
                physical_safe_successes: a.physical_safe_successes,
                slo_qualified_missions: a.slo_qualified_missions,
                hard_safety_violations: a.hard_safety_violations,
                critical_deadline_misses: a.critical_deadline_misses,
                service_floor_breaches: a.service_floor_breaches,
                longest_starvation_ticks: a.longest_starvation_ticks,
                causal_closure: CausalClosureTotals {
                    applications_examined: a.applications_examined,
                    applications_linked: a.applications_linked,
                    applications_causally_closed: a.applications_causally_closed,
                    closure_ppm: closure_ppm(
                        a.applications_examined,
                        a.applications_causally_closed,
                    ),
                },
            },
        );
    }
    Ok(total)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComparisonContract {
    pub max_physical_safe_success_drop_ppm: u32,
    pub max_slo_safe_success_drop_ppm: u32,
    pub max_deadline_miss_increase_ppm: u32,
    pub max_hard_violation_increase_ppm: u32,
    pub max_service_floor_breach_increase_ppm: u32,
    pub max_causal_closure_drop_ppm: u32,
}

impl ComparisonContract {
    pub fn validate(self) -> Result<(), EvaluationError> {
        for value in [
            self.max_physical_safe_success_drop_ppm,
            self.max_slo_safe_success_drop_ppm,
            self.max_deadline_miss_increase_ppm,
            self.max_hard_violation_increase_ppm,
            self.max_service_floor_breach_increase_ppm,
            self.max_causal_closure_drop_ppm,
        ] {
            if value > 1_000_000 {
                return Err(EvaluationError::InvalidClosureThreshold(value));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateDelta {
    pub baseline_ppm: Option<u32>,
    pub candidate_ppm: Option<u32>,
    pub delta_ppm: Option<i64>,
}

fn rate_delta(
    baseline_count: u64,
    baseline_total: u64,
    candidate_count: u64,
    candidate_total: u64,
) -> RateDelta {
    let baseline = ppm(baseline_count, baseline_total);
    let candidate = ppm(candidate_count, candidate_total);
    let delta = baseline.zip(candidate).map(|(b, c)| i64::from(c) - i64::from(b));
    RateDelta {
        baseline_ppm: baseline,
        candidate_ppm: candidate,
        delta_ppm: delta,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskClassComparison {
    pub task_class: String,
    pub physical_safe_success: RateDelta,
    pub slo_qualified_safe_success: RateDelta,
    pub critical_deadline_miss: RateDelta,
    pub hard_safety_violation: RateDelta,
    pub service_floor_breach: RateDelta,
    pub causal_closure: RateDelta,
    pub baseline_work_units_per_gpu_hour_milli: Option<u64>,
    pub candidate_work_units_per_gpu_hour_milli: Option<u64>,
    pub productivity_delta_milli: Option<i128>,
    pub baseline_work_units_per_robot_hour_milli: Option<u64>,
    pub candidate_work_units_per_robot_hour_milli: Option<u64>,
    pub robot_hour_productivity_delta_milli: Option<i128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobotComparison {
    pub robot_id: String,
    pub physical_safe_success: RateDelta,
    pub slo_qualified_safe_success: RateDelta,
    pub hard_safety_violation: RateDelta,
    pub critical_deadline_miss: RateDelta,
    pub service_floor_breach: RateDelta,
    pub causal_closure: RateDelta,
    pub baseline_longest_starvation_ticks: u64,
    pub candidate_longest_starvation_ticks: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FleetComparison {
    pub baseline_arm_id: String,
    pub candidate_arm_id: String,
    pub matched_missions: u64,
    pub per_task_class: BTreeMap<String, TaskClassComparison>,
    pub per_robot: BTreeMap<String, RobotComparison>,
    /// Hard/non-inferiority conditions across every task class and robot.
    pub candidate_noninferior: bool,
    /// Pareto result over task-class productivity, not a pooled scalar utility.
    pub pareto_dominates_productivity: bool,
    pub productivity_improved_classes: Vec<String>,
    pub productivity_regressed_classes: Vec<String>,
}

/// Require one-to-one paired missions with identical frozen workload, plant,
/// policy, model, observation and compute budgets. Only the scheduler arm differs.
pub fn compare_fleet_arms(
    baseline_arm: &str,
    baseline: &[MissionTrial],
    candidate_arm: &str,
    candidate: &[MissionTrial],
    scorecard_contract: ScorecardContract,
    comparison_contract: ComparisonContract,
) -> Result<FleetComparison, EvaluationError> {
    comparison_contract.validate()?;
    let baseline_score = evaluate_fleet(baseline_arm, baseline, scorecard_contract)?;
    let candidate_score = evaluate_fleet(candidate_arm, candidate, scorecard_contract)?;

    let mut baseline_by_pair: BTreeMap<&str, &MissionTrial> = BTreeMap::new();
    let mut candidate_by_pair: BTreeMap<&str, &MissionTrial> = BTreeMap::new();
    for trial in baseline {
        baseline_by_pair.insert(trial.identity.pair_id.as_str(), trial);
    }
    for trial in candidate {
        candidate_by_pair.insert(trial.identity.pair_id.as_str(), trial);
    }
    if baseline_by_pair.len() != candidate_by_pair.len() {
        return Err(EvaluationError::PairSetMismatch("pair cardinality".to_owned()));
    }
    for (pair_id, baseline_trial) in &baseline_by_pair {
        let Some(candidate_trial) = candidate_by_pair.get(pair_id) else {
            return Err(EvaluationError::PairSetMismatch((*pair_id).to_owned()));
        };
        if baseline_trial.identity != candidate_trial.identity {
            return Err(EvaluationError::PairIdentityMismatch((*pair_id).to_owned()));
        }
    }

    let mut per_task_class = BTreeMap::new();
    let mut productivity_improved_classes = Vec::new();
    let mut productivity_regressed_classes = Vec::new();
    let mut candidate_noninferior = true;
    let mut pareto_dominates_productivity = true;
    let task_classes: BTreeSet<String> = baseline_score
        .task_classes
        .keys()
        .chain(candidate_score.task_classes.keys())
        .cloned()
        .collect();

    for class in task_classes {
        let b = baseline_score.task_classes.get(&class).ok_or_else(|| {
            EvaluationError::PairIdentityMismatch(format!("missing baseline task class {class}"))
        })?;
        let c = candidate_score.task_classes.get(&class).ok_or_else(|| {
            EvaluationError::PairIdentityMismatch(format!("missing candidate task class {class}"))
        })?;
        if b.work_unit_id != c.work_unit_id {
            return Err(EvaluationError::PairIdentityMismatch(format!(
                "incomparable work units for task class {class}"
            )));
        }
        let physical = rate_delta(
            b.physical_safe_successes,
            b.missions,
            c.physical_safe_successes,
            c.missions,
        );
        let slo = rate_delta(
            b.slo_qualified_missions,
            b.missions,
            c.slo_qualified_missions,
            c.missions,
        );
        let deadline = rate_delta(
            b.critical_deadline_misses,
            b.missions,
            c.critical_deadline_misses,
            c.missions,
        );
        let hard = rate_delta(
            b.hard_safety_violations,
            b.missions,
            c.hard_safety_violations,
            c.missions,
        );
        let service = rate_delta(
            b.service_floor_breaches,
            b.missions,
            c.service_floor_breaches,
            c.missions,
        );
        let closure = rate_delta(
            b.causal_closure.applications_causally_closed,
            b.causal_closure.applications_examined,
            c.causal_closure.applications_causally_closed,
            c.causal_closure.applications_examined,
        );
        let class_noninferior = physical.delta_ppm.is_some_and(|d| {
            d >= -(i64::from(comparison_contract.max_physical_safe_success_drop_ppm))
        }) && slo.delta_ppm.is_some_and(|d| {
            d >= -(i64::from(comparison_contract.max_slo_safe_success_drop_ppm))
        }) && deadline.delta_ppm.is_some_and(|d| {
            d <= i64::from(comparison_contract.max_deadline_miss_increase_ppm)
        }) && hard.delta_ppm.is_some_and(|d| {
            d <= i64::from(comparison_contract.max_hard_violation_increase_ppm)
        }) && service.delta_ppm.is_some_and(|d| {
            d <= i64::from(comparison_contract.max_service_floor_breach_increase_ppm)
        }) && closure.delta_ppm.is_some_and(|d| {
            d >= -(i64::from(comparison_contract.max_causal_closure_drop_ppm))
        });
        candidate_noninferior &= class_noninferior;
        let bp = b.work_units_per_gpu_hour_milli;
        let cp = c.work_units_per_gpu_hour_milli;
        let productivity_delta = bp.zip(cp).map(|(bv, cv)| i128::from(cv) - i128::from(bv));
        let br = b.work_units_per_robot_hour_milli;
        let cr = c.work_units_per_robot_hour_milli;
        let robot_hour_delta = br
            .zip(cr)
            .map(|(bv, cv)| i128::from(cv) - i128::from(bv));
        match productivity_delta {
            Some(d) if d > 0 => productivity_improved_classes.push(class.clone()),
            Some(d) if d < 0 => productivity_regressed_classes.push(class.clone()),
            _ => {}
        }
        if productivity_delta.is_none_or(|d| d < 0)
            || robot_hour_delta.is_none_or(|d| d < 0)
        {
            pareto_dominates_productivity = false;
        }
        per_task_class.insert(
            class.clone(),
            TaskClassComparison {
                task_class: class,
                physical_safe_success: physical,
                slo_qualified_safe_success: slo,
                hard_safety_violation: hard,
                critical_deadline_miss: deadline,
                hard_safety_violation: hard,
                service_floor_breach: service,
                causal_closure: closure,
                baseline_work_units_per_gpu_hour_milli: bp,
                candidate_work_units_per_gpu_hour_milli: cp,
                productivity_delta_milli: productivity_delta,
                baseline_work_units_per_robot_hour_milli: br,
                candidate_work_units_per_robot_hour_milli: cr,
                robot_hour_productivity_delta_milli: robot_hour_delta,
            },
        );
    }

    let robot_ids: BTreeSet<String> = baseline_score
        .robots
        .keys()
        .chain(candidate_score.robots.keys())
        .cloned()
        .collect();
    let mut per_robot = BTreeMap::new();
    for robot_id in robot_ids {
        let b = baseline_score.robots.get(&robot_id).ok_or_else(|| {
            EvaluationError::PairIdentityMismatch(format!("missing baseline robot {robot_id}"))
        })?;
        let c = candidate_score.robots.get(&robot_id).ok_or_else(|| {
            EvaluationError::PairIdentityMismatch(format!("missing candidate robot {robot_id}"))
        })?;
        let physical = rate_delta(
            b.physical_safe_successes,
            b.missions,
            c.physical_safe_successes,
            c.missions,
        );
        let slo = rate_delta(
            b.slo_qualified_missions,
            b.missions,
            c.slo_qualified_missions,
            c.missions,
        );
        let hard = rate_delta(
            b.hard_safety_violations,
            b.missions,
            c.hard_safety_violations,
            c.missions,
        );
        let deadline = rate_delta(
            b.critical_deadline_misses,
            b.missions,
            c.critical_deadline_misses,
            c.missions,
        );
        let service = rate_delta(
            b.service_floor_breaches,
            b.missions,
            c.service_floor_breaches,
            c.missions,
        );
        let closure = rate_delta(
            b.causal_closure.applications_causally_closed,
            b.causal_closure.applications_examined,
            c.causal_closure.applications_causally_closed,
            c.causal_closure.applications_examined,
        );
        let robot_ok = physical.delta_ppm.is_some_and(|d| {
            d >= -(i64::from(comparison_contract.max_physical_safe_success_drop_ppm))
        }) && slo.delta_ppm.is_some_and(|d| {
            d >= -(i64::from(comparison_contract.max_slo_safe_success_drop_ppm))
        }) && hard.delta_ppm.is_some_and(|d| {
            d <= i64::from(comparison_contract.max_hard_violation_increase_ppm)
        }) && deadline.delta_ppm.is_some_and(|d| {
            d <= i64::from(comparison_contract.max_deadline_miss_increase_ppm)
        }) && service.delta_ppm.is_some_and(|d| {
            d <= i64::from(comparison_contract.max_service_floor_breach_increase_ppm)
        }) && closure.delta_ppm.is_some_and(|d| {
            d >= -(i64::from(comparison_contract.max_causal_closure_drop_ppm))
        });
        candidate_noninferior &= robot_ok;
        per_robot.insert(
            robot_id.clone(),
            RobotComparison {
                robot_id,
                physical_safe_success: physical,
                slo_qualified_safe_success: slo,
                critical_deadline_miss: deadline,
                service_floor_breach: service,
                causal_closure: closure,
                baseline_longest_starvation_ticks: b.longest_starvation_ticks,
                candidate_longest_starvation_ticks: c.longest_starvation_ticks,
            },
        );
    }
    pareto_dominates_productivity &=
        candidate_noninferior && !productivity_improved_classes.is_empty();

    Ok(FleetComparison {
        baseline_arm_id: baseline_arm.to_owned(),
        candidate_arm_id: candidate_arm.to_owned(),
        matched_missions: baseline.len() as u64,
        per_task_class,
        per_robot,
        candidate_noninferior,
        pareto_dominates_productivity,
        productivity_improved_classes,
        productivity_regressed_classes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(pair: &str, robot: &str) -> MatchIdentity {
        MatchIdentity {
            pair_id: pair.to_owned(),
            robot_id: robot.to_owned(),
            task_class: "delivery".to_owned(),
            work_unit_id: "completed-package-v1".to_owned(),
            workload_digest: "workload-sha".to_owned(),
            scenario_digest: "scenario-sha".to_owned(),
            seed_id: pair.to_owned(),
            initial_state_digest: "initial-state-sha".to_owned(),
            model_digest: "model-sha".to_owned(),
            control_policy_digest: "local-control-sha".to_owned(),
            hardware_digest: "hardware-sha".to_owned(),
            observation_budget_digest: "observation-budget-sha".to_owned(),
            compute_budget_digest: "compute-budget-sha".to_owned(),
        }
    }

    fn trial(pair: &str, robot: &str, arm: &str) -> MissionTrial {
        MissionTrial {
            mission_id: format!("{arm}-{pair}"),
            arm_id: arm.to_owned(),
            identity: identity(pair, robot),
            elapsed_ns: 10_000_000_000,
            gpu_micros: 3_600_000_000,
            useful_work_units: 1,
            task_completed: true,
            terminal_safe: true,
            hard_safety_violation: false,
            critical_deadline_miss: false,
            longest_starvation_ticks: 2,
            starvation_limit_ticks: 10,
            thermal_evidence: EvidenceState::Qualified,
            recovery_evidence: EvidenceState::Qualified,
            energy_evidence: EvidenceState::Qualified,
            measured_energy_joules: Some(900),
            actuator_applications_examined: 10,
            actuator_applications_linked: 10,
            actuator_applications_causally_closed: 10,
        }
    }

    fn contract() -> ScorecardContract {
        ScorecardContract {
            minimum_causal_closure_ppm: 1_000_000,
        }
    }

    fn comparison_contract() -> ComparisonContract {
        ComparisonContract {
            max_physical_safe_success_drop_ppm: 0,
            max_slo_safe_success_drop_ppm: 0,
            max_deadline_miss_increase_ppm: 0,
            max_hard_violation_increase_ppm: 0,
            max_service_floor_breach_increase_ppm: 0,
            max_causal_closure_drop_ppm: 0,
        }
    }

    #[test]
    fn complete_trace_receives_physical_and_evidence_qualified_safe_success() {
        let score = evaluate_fleet("candidate", &[trial("p1", "r1", "candidate")], contract())
            .unwrap();
        assert_eq!(score.physical_safe_successes, 1);
        assert_eq!(score.evidence_qualified_safe_successes, 1);
        assert_eq!(score.slo_qualified_missions, 1);
        assert_eq!(
            score.causal_closure.closure_ppm,
            Some(1_000_000)
        );
    }

    #[test]
    fn unknown_thermal_or_recovery_evidence_cannot_become_safe_success() {
        let mut input = trial("p1", "r1", "candidate");
        input.thermal_evidence = EvidenceState::Unknown;
        let score = evaluate_fleet("candidate", &[input], contract()).unwrap();
        assert_eq!(score.physical_safe_successes, 1);
        assert_eq!(score.evidence_qualified_safe_successes, 0);
        assert_eq!(score.slo_qualified_missions, 0);
        assert_eq!(score.task_classes["delivery"].thermal_unknown_or_invalid, 1);
    }

    #[test]
    fn incomplete_causal_application_chain_cannot_earn_productivity_credit() {
        let mut input = trial("p1", "r1", "candidate");
        input.actuator_applications_causally_closed = 9;
        let score = evaluate_fleet("candidate", &[input], contract()).unwrap();
        assert_eq!(score.slo_qualified_missions, 0);
        assert_eq!(
            score.task_classes["delivery"].productive_work_units,
            0
        );
        assert_eq!(
            score.causal_closure.closure_ppm,
            Some(900_000)
        );
    }

    #[test]
    fn energy_unknown_remains_a_separate_incomplete_measurement() {
        let mut input = trial("p1", "r1", "candidate");
        input.energy_evidence = EvidenceState::Unknown;
        input.measured_energy_joules = None;
        let score = evaluate_fleet("candidate", &[input], contract()).unwrap();
        let class = &score.task_classes["delivery"];
        assert_eq!(class.energy_qualified_missions, 0);
        assert_eq!(class.energy_unknown_or_invalid_missions, 1);
        assert_eq!(class.known_energy_joules, 0);
    }

    #[test]
    fn unknown_energy_cannot_be_reported_as_zero_energy() {
        let mut input = trial("p1", "r1", "candidate");
        input.energy_evidence = EvidenceState::Unknown;
        assert_eq!(
            evaluate_fleet("candidate", &[input], contract()),
            Err(EvaluationError::InvalidEnergyReceipt(
                "candidate-p1".to_owned()
            ))
        );
    }

    #[test]
    fn gpu_hour_denominator_includes_failed_missions() {
        let good = trial("p1", "r1", "candidate");
        let mut failed = trial("p2", "r1", "candidate");
        failed.task_completed = false;
        failed.terminal_safe = false;
        failed.useful_work_units = 0;
        let score = evaluate_fleet("candidate", &[good, failed], contract()).unwrap();
        let class = &score.task_classes["delivery"];
        assert_eq!(class.productive_work_units, 1);
        assert_eq!(class.gpu_micros, 7_200_000_000);
        assert_eq!(class.elapsed_ns, 20_000_000_000);
        assert_eq!(class.work_units_per_gpu_hour_milli, Some(500));
        assert_eq!(class.work_units_per_robot_hour_milli, Some(180_000));
    }

    #[test]
    fn productivity_is_kept_separate_by_task_class_and_work_unit() {
        let mut second = trial("p2", "r2", "candidate");
        second.identity.task_class = "inspection".to_owned();
        second.identity.work_unit_id = "inspected-metre-v1".to_owned();
        let score = evaluate_fleet("candidate", &[trial("p1", "r1", "candidate"), second], contract())
            .unwrap();
        assert_eq!(score.task_classes.len(), 2);
        assert_eq!(
            score.task_classes["delivery"].work_unit_id,
            "completed-package-v1"
        );
        assert_eq!(
            score.task_classes["inspection"].work_unit_id,
            "inspected-metre-v1"
        );
    }

    #[test]
    fn longest_starvation_breach_blocks_slo_credit_but_not_physical_success() {
        let mut input = trial("p1", "r1", "candidate");
        input.longest_starvation_ticks = 11;
        let score = evaluate_fleet("candidate", &[input], contract()).unwrap();
        assert_eq!(score.physical_safe_successes, 1);
        assert_eq!(score.slo_qualified_missions, 0);
        assert_eq!(score.service_floor_breaches, 1);
    }

    #[test]
    fn invalid_application_counts_are_rejected() {
        let mut input = trial("p1", "r1", "candidate");
        input.actuator_applications_linked = 11;
        assert_eq!(
            evaluate_fleet("candidate", &[input], contract()),
            Err(EvaluationError::InvalidApplicationCounts(
                "candidate-p1".to_owned()
            ))
        );
    }

    #[test]
    fn duplicate_mission_and_pair_ids_are_rejected() {
        let a = trial("p1", "r1", "candidate");
        let mut b = trial("p1", "r2", "candidate");
        b.mission_id = "different-mission".to_owned();
        assert_eq!(
            evaluate_fleet("candidate", &[a, b], contract()),
            Err(EvaluationError::DuplicatePairId("p1".to_owned()))
        );
    }

    #[test]
    fn matched_arms_compare_per_task_and_per_robot_without_scalarization() {
        let baseline = vec![trial("p1", "r1", "baseline"), trial("p2", "r2", "baseline")];
        let candidate = vec![
            trial("p1", "r1", "candidate"),
            trial("p2", "r2", "candidate"),
        ];
        let report = compare_fleet_arms(
            "baseline",
            &baseline,
            "candidate",
            &candidate,
            contract(),
            comparison_contract(),
        )
        .unwrap();
        assert!(report.candidate_noninferior);
        assert_eq!(report.per_task_class.len(), 1);
        assert_eq!(report.per_robot.len(), 2);
        assert!(!report.pareto_dominates_productivity);
    }

    #[test]
    fn unmatched_missions_are_rejected_not_dropped() {
        let baseline = vec![trial("p1", "r1", "baseline")];
        let candidate = vec![trial("p2", "r1", "candidate")];
        assert_eq!(
            compare_fleet_arms(
                "baseline",
                &baseline,
                "candidate",
                &candidate,
                contract(),
                comparison_contract(),
            ),
            Err(EvaluationError::PairSetMismatch("pair cardinality".to_owned()))
        );
    }

    #[test]
    fn changed_initial_state_or_model_invalidates_a_pair() {
        let baseline = vec![trial("p1", "r1", "baseline")];
        let mut candidate_trial = trial("p1", "r1", "candidate");
        candidate_trial.identity.initial_state_digest = "other-initial-state".to_owned();
        assert_eq!(
            compare_fleet_arms(
                "baseline",
                &baseline,
                "candidate",
                &[candidate_trial],
                contract(),
                comparison_contract(),
            ),
            Err(EvaluationError::PairIdentityMismatch("p1".to_owned()))
        );
    }

    #[test]
    fn hard_safety_regression_fails_noninferiority_even_if_productivity_improves() {
        let baseline = vec![trial("p1", "r1", "baseline")];
        let mut candidate_trial = trial("p1", "r1", "candidate");
        candidate_trial.hard_safety_violation = true;
        candidate_trial.useful_work_units = 10;
        let report = compare_fleet_arms(
            "baseline",
            &baseline,
            "candidate",
            &[candidate_trial],
            contract(),
            comparison_contract(),
        )
        .unwrap();
        assert!(!report.candidate_noninferior);
        assert!(!report.pareto_dominates_productivity);
        assert_eq!(
            report.per_robot["r1"].hard_safety_violation.delta_ppm,
            Some(1_000_000)
        );
    }

    #[test]
    fn productivity_improvement_is_reported_only_after_noninferiority() {
        let baseline = vec![trial("p1", "r1", "baseline")];
        let mut candidate_trial = trial("p1", "r1", "candidate");
        candidate_trial.useful_work_units = 2;
        let report = compare_fleet_arms(
            "baseline",
            &baseline,
            "candidate",
            &[candidate_trial],
            contract(),
            comparison_contract(),
        )
        .unwrap();
        assert!(report.candidate_noninferior);
        assert!(report.pareto_dominates_productivity);
        assert_eq!(report.productivity_improved_classes, vec!["delivery"]);
    }

    #[test]
    fn closure_aggregation_is_weighted_by_application_count() {
        let mut many = trial("p1", "r1", "candidate");
        many.actuator_applications_examined = 100;
        many.actuator_applications_linked = 100;
        many.actuator_applications_causally_closed = 100;
        let mut one = trial("p2", "r2", "candidate");
        one.actuator_applications_examined = 1;
        one.actuator_applications_linked = 1;
        one.actuator_applications_causally_closed = 0;
        let score = evaluate_fleet("candidate", &[many, one], contract()).unwrap();
        assert_eq!(score.causal_closure.closure_ppm, Some(990_099));
    }

    #[test]
    fn comparison_rejects_missing_causal_closure_measurement() {
        let mut baseline = trial("p1", "r1", "baseline");
        baseline.actuator_applications_examined = 0;
        baseline.actuator_applications_linked = 0;
        baseline.actuator_applications_causally_closed = 0;
        let candidate = trial("p1", "r1", "candidate");
        let report = compare_fleet_arms(
            "baseline",
            &[baseline],
            "candidate",
            &[candidate],
            contract(),
            comparison_contract(),
        )
        .unwrap();
        assert!(!report.candidate_noninferior);
    }
}
