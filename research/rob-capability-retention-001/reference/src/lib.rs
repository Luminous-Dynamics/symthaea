//! Productive work versus terminal physical recoverability reference.
//!
//! This crate consumes externally qualified work, reserve and option-ledger
//! summaries. It does not calculate physical reserves, infer viability, score
//! economic value, authorize actuation, or replace cluster-aware inference.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

/// Fixed-point scale for work-rate reporting (milli accepted items / robot-hour).
pub const WORK_RATE_SCALE: u64 = 1_000;

/// Physical capability dimensions that may constrain continued operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReserveKind {
    /// Remaining battery-side energy reserve.
    EnergyReserveMilliWh,
    /// Conservative thermal headroom to a declared operating limit.
    ThermalHeadroomMilliC,
    /// Conservative usable actuator torque headroom.
    ActuatorTorqueHeadroomMilliNm,
    /// Qualified mechanical condition indicator (profile-defined PPM scale).
    MechanicalConditionPpm,
    /// Qualified sensor coverage / freshness eligibility (profile-defined PPM scale).
    SensorEligibilityPpm,
    /// Qualified future recovery/continuation options.
    RecoveryOptionCount,
    /// Remaining mission/deadline slack.
    DeadlineSlackMillis,
    /// Remaining service-free operating allowance under a declared duty profile.
    ServiceFreeAllowanceMinutes,
}

/// Whether one reserve dimension is required for this workcell or explicitly absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReserveApplicability {
    /// This component is in scope and has a hard, predeclared minimum floor.
    Required {
        /// Minimum conservative lower bound in the units implied by ReserveKind.
        minimum_floor: u64,
        /// Whether a valid calibration receipt is mandatory for this reserve measurement.
        calibration_required: bool,
    },
    /// This component does not apply to the declared embodiment/workcell.
    NotApplicable {
        /// Explicit rationale; an empty reason is invalid.
        reason: String,
    },
}

/// Provenance/qualification disposition supplied by the reserve/option evidence owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceDisposition {
    /// Evidence owner has qualified the receipt for this scope.
    Qualified,
    /// Evidence is missing or unresolved.
    Unknown,
    /// This dimension is explicitly outside the frozen applicability profile.
    NotApplicable,
    /// Evidence is contradictory or explicitly rejected.
    Rejected,
}

/// Frozen reserve floors for one comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservePolicy {
    /// Versioned reserve-policy identity digest.
    pub policy_digest: String,
    /// Exactly one applicability entry per reserve dimension.
    pub components: Vec<(ReserveKind, ReserveApplicability)>,
}

/// Exact comparison scope shared by candidate and baseline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityWorkScope {
    /// Benchmark/workcell identifier.
    pub workcell_id: String,
    /// Frozen work order / task mix manifest digest.
    pub work_manifest_digest: String,
    /// Exact robot article and configuration digest.
    pub robot_configuration_digest: String,
    /// Exact work acceptance evaluator and safety-specification digest.
    pub evaluator_digest: String,
    /// Scheduled shift, staffing, authority and resource boundary digest.
    pub operating_envelope_digest: String,
    /// Common resource-budget boundary digest.
    pub resource_budget_digest: String,
    /// Frozen terminal-reserve policy digest.
    pub reserve_policy_digest: String,
}

/// Exact identity of one compared policy/controller arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmManifest {
    /// Stable arm identity.
    pub arm_id: String,
    /// Model/controller artifact digest.
    pub model_artifact_digest: String,
    /// Policy/controller configuration digest.
    pub policy_digest: String,
    /// Runtime/harness digest.
    pub harness_digest: String,
    /// Exact common work scope.
    pub workcell_id: String,
    /// Frozen work manifest.
    pub work_manifest_digest: String,
    /// Physical article/configuration.
    pub robot_configuration_digest: String,
    /// Work evaluator and safety profile.
    pub evaluator_digest: String,
    /// Operating envelope.
    pub operating_envelope_digest: String,
    /// Resource boundary.
    pub resource_budget_digest: String,
    /// Reserve-policy binding.
    pub reserve_policy_digest: String,
}

/// Mutually exclusive terminal status for one scheduled work item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkItemDisposition {
    /// Independently inspected and accepted work.
    Accepted,
    /// Completed but rejected by the declared acceptance criteria.
    Rejected,
    /// Required rework; not counted as accepted output in this final tally.
    Reworked,
    /// Aborted before acceptance.
    Aborted,
    /// Held for human review or unresolved quality disposition.
    Held,
    /// Outcome could not be established.
    Unknown,
}

/// Complete scheduled-work denominator and final dispositions for one cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkTally {
    /// Total work items scheduled before execution.
    pub scheduled_items: u32,
    /// Final accepted, quality-qualified items.
    pub accepted_items: u32,
    /// Completed but rejected items.
    pub rejected_items: u32,
    /// Items requiring rework.
    pub reworked_items: u32,
    /// Aborted items.
    pub aborted_items: u32,
    /// Held items.
    pub held_items: u32,
    /// Unknown final dispositions.
    pub unknown_items: u32,
}

impl WorkTally {
    fn validate(&self, arm_id: &str, cluster_id: &str) -> Result<(), CapabilityRetentionError> {
        if self.scheduled_items == 0 {
            return Err(CapabilityRetentionError::InvalidWorkTally {
                arm_id: arm_id.to_owned(),
                cluster_id: cluster_id.to_owned(),
            });
        }
        let classified = self
            .accepted_items
            .checked_add(self.rejected_items)
            .and_then(|value| value.checked_add(self.reworked_items))
            .and_then(|value| value.checked_add(self.aborted_items))
            .and_then(|value| value.checked_add(self.held_items))
            .and_then(|value| value.checked_add(self.unknown_items))
            .ok_or_else(|| CapabilityRetentionError::InvalidWorkTally {
                arm_id: arm_id.to_owned(),
                cluster_id: cluster_id.to_owned(),
            })?;
        if classified != self.scheduled_items {
            return Err(CapabilityRetentionError::InvalidWorkTally {
                arm_id: arm_id.to_owned(),
                cluster_id: cluster_id.to_owned(),
            });
        }
        Ok(())
    }
}

/// Conservative terminal reserve observation from the owning physical evidence system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReserveObservation {
    /// Reserve dimension.
    pub kind: ReserveKind,
    /// Evidence disposition.
    pub disposition: EvidenceDisposition,
    /// Conservative lower bound in kind-specific integer units, when Qualified.
    pub terminal_lower_bound: Option<u64>,
    /// Source/metrology receipt identity.
    pub source_receipt_digest: Option<String>,
    /// Calibration identity, where the dimension requires calibration.
    pub calibration_receipt_digest: Option<String>,
    /// Required only for an explicitly NotApplicable reserve dimension.
    pub not_applicable_reason: Option<String>,
}

/// Summary receipt from the independently qualified recovery-topology owner.
///
/// This consumes #4511 outputs; it does not rebuild route graphs, disjoint
/// path packs, failure-domain dependencies or common-cause cut sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryTopologyReceipt {
    /// Evidence disposition supplied by the topology evaluator.
    pub disposition: EvidenceDisposition,
    /// Exact recovery-topology report digest.
    pub report_digest: Option<String>,
    /// Canonical physical recovery-route classes after alias collapse.
    pub recovery_route_class_count: Option<u32>,
    /// Failure-domain-independent routes, not merely action-distinct paths.
    pub failure_domain_independent_route_count: Option<u32>,
    /// Cardinality of the minimum declared common-cause failure-domain cut set.
    pub minimum_failure_domain_cut_set_cardinality: Option<u32>,
}

/// Option-set evidence supplied by the future-option ledger owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionSetReceipt {
    /// Evidence disposition from the option-ledger owner.
    pub disposition: EvidenceDisposition,
    /// Exact option-loss ledger receipt digest.
    pub ledger_receipt_digest: String,
    /// Predecessor physical/reserve state digest.
    pub predecessor_state_digest: String,
    /// Resulting physical/reserve state digest.
    pub result_state_digest: String,
    /// Exact starting qualified continuation-set digest.
    pub starting_option_set_digest: String,
    /// Exact terminal qualified continuation-set digest.
    pub terminal_option_set_digest: String,
    /// Count of terminal qualified continuations as reported by the option owner.
    pub terminal_qualified_option_count: u32,
    /// Separate topological recovery-redundancy evidence from #4511.
    pub recovery_topology: RecoveryTopologyReceipt,
}

/// Complete per-arm outcome and terminal reserve summary for one cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmClusterSummary {
    /// Final work-item dispositions.
    pub work: WorkTally,
    /// Receipt from the accepted-work ledger / inspection owner.
    pub work_outcome_receipt_digest: String,
    /// Receipt binding the scheduled time, human intervention and downtime accounting.
    pub operating_time_receipt_digest: String,
    /// Scheduled available robot minutes in the declared matched work period.
    pub available_robot_minutes: u32,
    /// Nonproductive downtime within the scheduled available period.
    pub nonproductive_downtime_minutes: u32,
    /// Human rescue, teleoperation and intervention minutes.
    pub human_intervention_minutes: u32,
    /// Electrical energy crossing the declared resource boundary, milli-Wh.
    pub electrical_energy_milli_wh: u64,
    /// Compute energy over the declared boundary, milli-joules.
    pub compute_energy_milli_joules: u64,
    /// Episodes with at least one independently adjudicated critical safety violation.
    pub critical_safety_violation_episodes: u32,
    /// One observation per reserve-policy component.
    pub reserves: Vec<ReserveObservation>,
    /// Exact option-set evidence from the option-loss owner.
    pub option_set: OptionSetReceipt,
}

/// One paired cluster and its candidate/baseline summaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairedCapabilityCluster {
    /// Stable independent-cluster identity.
    pub cluster_id: String,
    /// Exact paired work-item/scenario manifest digest.
    pub paired_work_manifest_digest: String,
    /// Receipt supporting independence and cluster construction.
    pub independence_receipt_digest: String,
    /// Independence disposition supplied by the experiment owner.
    pub independence_disposition: EvidenceDisposition,
    /// Existing qualified baseline.
    pub baseline: ArmClusterSummary,
    /// Candidate autonomy/assurance composition.
    pub candidate: ArmClusterSummary,
}

/// Complete capability-retention comparison input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRetentionInput {
    /// Frozen common scope.
    pub scope: CapabilityWorkScope,
    /// Frozen reserve floors/applicability.
    pub reserve_policy: ReservePolicy,
    /// Baseline policy identity.
    pub baseline_manifest: ArmManifest,
    /// Candidate policy identity.
    pub candidate_manifest: ArmManifest,
    /// One exact paired summary per independent cluster.
    pub clusters: Vec<PairedCapabilityCluster>,
}

/// Per-dimension terminal reserve comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReserveDelta {
    /// Reserve dimension.
    pub kind: ReserveKind,
    /// Baseline conservative lower bound, if Required and Qualified.
    pub baseline_lower_bound: Option<u64>,
    /// Candidate conservative lower bound, if Required and Qualified.
    pub candidate_lower_bound: Option<u64>,
    /// Candidate minus baseline, only where both values are available.
    pub candidate_minus_baseline: Option<i128>,
    /// Frozen hard floor, if Required.
    pub minimum_floor: Option<u64>,
    /// Whether the candidate breached the hard floor.
    pub candidate_below_floor: bool,
    /// Evidence disposition.
    pub disposition: EvidenceDisposition,
}

/// Per-cluster descriptive comparison; it is not an inferential result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterCapabilityReport {
    /// Independent cluster identity.
    pub cluster_id: String,
    /// Scheduled work-item count.
    pub scheduled_items: u32,
    /// Baseline and candidate accepted items.
    pub baseline_accepted_items: u32,
    pub candidate_accepted_items: u32,
    /// Baseline and candidate accepted items per available robot-hour, milli-units/hour.
    pub baseline_work_rate_milli_per_hour: u64,
    pub candidate_work_rate_milli_per_hour: u64,
    /// Candidate minus baseline work rate.
    pub work_rate_delta_milli_per_hour: i128,
    /// Human intervention minutes, kept separate from robot availability.
    pub baseline_human_intervention_minutes: u32,
    pub candidate_human_intervention_minutes: u32,
    /// Nonproductive downtime within the common scheduled time denominator.
    pub baseline_nonproductive_downtime_minutes: u32,
    pub candidate_nonproductive_downtime_minutes: u32,
    /// Energy/resource use, not collapsed into a universal cost scalar.
    pub baseline_electrical_energy_milli_wh: u64,
    pub candidate_electrical_energy_milli_wh: u64,
    pub baseline_compute_energy_milli_joules: u64,
    pub candidate_compute_energy_milli_joules: u64,
    /// Qualified terminal option counts from the upstream option-set receipts.
    pub baseline_terminal_option_count: Option<u32>,
    pub candidate_terminal_option_count: Option<u32>,
    /// Canonical recovery-route class counts from #4511.
    pub baseline_recovery_route_class_count: Option<u32>,
    pub candidate_recovery_route_class_count: Option<u32>,
    /// Failure-domain-independent routes; distinct from nominal route/action count.
    pub baseline_failure_domain_independent_route_count: Option<u32>,
    pub candidate_failure_domain_independent_route_count: Option<u32>,
    /// Minimum declared common-cause cut-set cardinality; None means not reported.
    pub baseline_minimum_cut_set_cardinality: Option<u32>,
    pub candidate_minimum_cut_set_cardinality: Option<u32>,
    /// Exact upstream topology report identities.
    pub baseline_topology_report_digest: Option<String>,
    pub candidate_topology_report_digest: Option<String>,
    /// Reserve comparison vector.
    pub reserve_deltas: Vec<ReserveDelta>,
    /// Critical safety violations remain separate and non-compensable.
    pub baseline_critical_safety_violation_episodes: u32,
    pub candidate_critical_safety_violation_episodes: u32,
    /// True when an unknown/censored outcome or unknown evidence prevents confirmation.
    pub evidence_complete: bool,
}

/// Evidence status for the full comparison. No statistical significance is implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityRetentionDisposition {
    /// All supplied evidence is structurally valid and complete; inspect per-cluster deltas.
    EvidenceComplete,
    /// One or more candidate critical safety violations were observed.
    CandidateCriticalSafetyObserved,
    /// One or more candidate terminal reserve floors were breached.
    CandidateReserveFloorBreached,
    /// Unknown/censored work or unresolved reserve/option/independence evidence.
    Inconclusive,
}

/// Full descriptive work-plus-capability-retention report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRetentionReport {
    /// Workcell identity.
    pub workcell_id: String,
    /// Baseline arm identity.
    pub baseline_arm_id: String,
    /// Candidate arm identity.
    pub candidate_arm_id: String,
    /// Overall evidence / hard-gate disposition.
    pub disposition: CapabilityRetentionDisposition,
    /// Canonically ordered independent-cluster summaries.
    pub clusters: Vec<ClusterCapabilityReport>,
    /// This reference has no actuation authority.
    pub actuation_authorized: bool,
}

/// Structural, scope or evidence-contract failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityRetentionError {
    /// Required identity or digest is empty.
    EmptyIdentity(&'static str),
    /// Scope and reserve-policy digest do not match.
    ReservePolicyMismatch,
    /// Duplicate reserve policy component.
    DuplicateReservePolicyComponent(ReserveKind),
    /// Missing required reserve policy component in an arm summary.
    MissingReserveObservation { arm_id: String, kind: ReserveKind },
    /// Duplicate reserve observation.
    DuplicateReserveObservation { arm_id: String, kind: ReserveKind },
    /// Reserve evidence contradicts its policy applicability.
    ReserveApplicabilityMismatch { arm_id: String, kind: ReserveKind },
    /// NotApplicable dimension lacks a reason or has an unexpected value.
    InvalidNotApplicableReserve { arm_id: String, kind: ReserveKind },
    /// Qualified reserve lacks a conservative bound or source receipt.
    IncompleteQualifiedReserve { arm_id: String, kind: ReserveKind },
    /// Explicitly rejected reserve evidence.
    RejectedReserveEvidence { arm_id: String, kind: ReserveKind },
    /// Arm does not match the common operating scope.
    ArmScopeMismatch { arm_id: String, field: &'static str },
    /// Candidate and baseline identifiers are not distinct.
    DuplicateArmId(String),
    /// Duplicate or empty cluster identity.
    DuplicateOrEmptyClusterId(String),
    /// Missing paired work or independence receipt.
    MissingClusterReceipt(String),
    /// A cluster was given duplicate or contradictory evidence identity.
    DuplicateClusterReceipt(String),
    /// Independence evidence was explicitly rejected.
    RejectedIndependenceEvidence(String),
    /// Paired work manifests differ between candidate and baseline.
    PairingMismatch(String),
    /// Work categories do not exactly cover the scheduled denominator.
    InvalidWorkTally { arm_id: String, cluster_id: String },
    /// No scheduled available robot time.
    ZeroAvailableRobotTime { arm_id: String, cluster_id: String },
    /// Candidate/baseline option sets did not start from the same qualified set.
    StartingOptionSetMismatch(String),
    /// Paired arms did not start from the same qualified physical state.
    StartingPhysicalStateMismatch(String),
    /// Paired arms used different scheduled robot-time denominators.
    ScheduledRobotTimeMismatch(String),
    /// Downtime exceeded the scheduled robot-time denominator.
    InvalidTimeAccounting { arm_id: String, cluster_id: String },
    /// Qualified reserve option count contradicts the qualified option-set receipt.
    OptionSetReserveCountMismatch { arm_id: String, cluster_id: String },
    /// Required option-set identity or ledger receipt is absent.
    MissingOptionSetReceipt(String),
    /// Option-ledger evidence was explicitly rejected.
    RejectedOptionSetEvidence(String),
    /// Recovery-topology evidence was explicitly rejected or is not applicable.
    RejectedRecoveryTopologyEvidence(String),
    /// Qualified recovery-topology evidence is missing a required identity/count.
    IncompleteQualifiedRecoveryTopology(String),
    /// Failure-domain-independent route count exceeds the canonical route count.
    InconsistentRecoveryTopology(String),
    /// A rate or aggregate arithmetic operation overflowed.
    ArithmeticOverflow,
}

fn require_nonempty(name: &'static str, value: &str) -> Result<(), CapabilityRetentionError> {
    if value.trim().is_empty() {
        return Err(CapabilityRetentionError::EmptyIdentity(name));
    }
    Ok(())
}

fn work_rate_milli_per_hour(
    accepted: u32,
    available_minutes: u32,
) -> Result<u64, CapabilityRetentionError> {
    if available_minutes == 0 {
        return Err(CapabilityRetentionError::ArithmeticOverflow);
    }
    let numerator = u128::from(accepted)
        .checked_mul(60_000)
        .ok_or(CapabilityRetentionError::ArithmeticOverflow)?;
    let rate = numerator / u128::from(available_minutes);
    u64::try_from(rate).map_err(|_| CapabilityRetentionError::ArithmeticOverflow)
}

fn validate_scope(
    scope: &CapabilityWorkScope,
    policy: &ReservePolicy,
) -> Result<(), CapabilityRetentionError> {
    for (name, value) in [
        ("workcell_id", scope.workcell_id.as_str()),
        ("work_manifest_digest", scope.work_manifest_digest.as_str()),
        (
            "robot_configuration_digest",
            scope.robot_configuration_digest.as_str(),
        ),
        ("evaluator_digest", scope.evaluator_digest.as_str()),
        (
            "operating_envelope_digest",
            scope.operating_envelope_digest.as_str(),
        ),
        (
            "resource_budget_digest",
            scope.resource_budget_digest.as_str(),
        ),
        (
            "reserve_policy_digest",
            scope.reserve_policy_digest.as_str(),
        ),
        (
            "reserve_policy.policy_digest",
            policy.policy_digest.as_str(),
        ),
    ] {
        require_nonempty(name, value)?;
    }
    if scope.reserve_policy_digest != policy.policy_digest {
        return Err(CapabilityRetentionError::ReservePolicyMismatch);
    }
    let mut kinds = BTreeSet::new();
    for (kind, applicability) in &policy.components {
        if !kinds.insert(*kind) {
            return Err(CapabilityRetentionError::DuplicateReservePolicyComponent(
                *kind,
            ));
        }
        if let ReserveApplicability::NotApplicable { reason } = applicability {
            if reason.trim().is_empty() {
                return Err(CapabilityRetentionError::InvalidNotApplicableReserve {
                    arm_id: "reserve-policy".into(),
                    kind: *kind,
                });
            }
        }
    }
    if kinds.is_empty()
        || !policy.components.iter().any(|(_, applicability)| {
            matches!(applicability, ReserveApplicability::Required { .. })
        })
    {
        return Err(CapabilityRetentionError::ReservePolicyMismatch);
    }
    Ok(())
}

fn validate_arm(
    scope: &CapabilityWorkScope,
    manifest: &ArmManifest,
) -> Result<(), CapabilityRetentionError> {
    for (name, value) in [
        ("arm_id", manifest.arm_id.as_str()),
        (
            "model_artifact_digest",
            manifest.model_artifact_digest.as_str(),
        ),
        ("policy_digest", manifest.policy_digest.as_str()),
        ("harness_digest", manifest.harness_digest.as_str()),
    ] {
        require_nonempty(name, value)?;
    }
    for (field, observed, expected) in [
        (
            "workcell_id",
            manifest.workcell_id.as_str(),
            scope.workcell_id.as_str(),
        ),
        (
            "work_manifest_digest",
            manifest.work_manifest_digest.as_str(),
            scope.work_manifest_digest.as_str(),
        ),
        (
            "robot_configuration_digest",
            manifest.robot_configuration_digest.as_str(),
            scope.robot_configuration_digest.as_str(),
        ),
        (
            "evaluator_digest",
            manifest.evaluator_digest.as_str(),
            scope.evaluator_digest.as_str(),
        ),
        (
            "operating_envelope_digest",
            manifest.operating_envelope_digest.as_str(),
            scope.operating_envelope_digest.as_str(),
        ),
        (
            "resource_budget_digest",
            manifest.resource_budget_digest.as_str(),
            scope.resource_budget_digest.as_str(),
        ),
        (
            "reserve_policy_digest",
            manifest.reserve_policy_digest.as_str(),
            scope.reserve_policy_digest.as_str(),
        ),
    ] {
        if observed != expected {
            return Err(CapabilityRetentionError::ArmScopeMismatch {
                arm_id: manifest.arm_id.clone(),
                field,
            });
        }
    }
    Ok(())
}

fn validate_recovery_topology(
    arm_id: &str,
    topology: &RecoveryTopologyReceipt,
) -> Result<(), CapabilityRetentionError> {
    match topology.disposition {
        EvidenceDisposition::Rejected | EvidenceDisposition::NotApplicable => {
            return Err(CapabilityRetentionError::RejectedRecoveryTopologyEvidence(
                arm_id.to_owned(),
            ));
        }
        EvidenceDisposition::Unknown => {
            if let Some(digest) = topology.report_digest.as_deref() {
                if digest.trim().is_empty() {
                    return Err(CapabilityRetentionError::IncompleteQualifiedRecoveryTopology(
                        arm_id.to_owned(),
                    ));
                }
            }
            return Ok(());
        }
        EvidenceDisposition::Qualified => {}
    }

    let report_digest = topology
        .report_digest
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    let route_count = topology.recovery_route_class_count;
    let independent_count = topology.failure_domain_independent_route_count;
    if report_digest.is_none() || route_count.is_none() || independent_count.is_none() {
        return Err(CapabilityRetentionError::IncompleteQualifiedRecoveryTopology(
            arm_id.to_owned(),
        ));
    }
    if independent_count.unwrap_or(0) > route_count.unwrap_or(0) {
        return Err(CapabilityRetentionError::InconsistentRecoveryTopology(
            arm_id.to_owned(),
        ));
    }
    Ok(())
}

fn validate_option_set(
    arm_id: &str,
    options: &OptionSetReceipt,
) -> Result<(), CapabilityRetentionError> {
    if matches!(
        options.disposition,
        EvidenceDisposition::Rejected | EvidenceDisposition::NotApplicable
    ) {
        return Err(CapabilityRetentionError::RejectedOptionSetEvidence(
            arm_id.to_owned(),
        ));
    }
    for value in [
        options.ledger_receipt_digest.as_str(),
        options.predecessor_state_digest.as_str(),
        options.result_state_digest.as_str(),
        options.starting_option_set_digest.as_str(),
        options.terminal_option_set_digest.as_str(),
    ] {
        if value.trim().is_empty() {
            return Err(CapabilityRetentionError::MissingOptionSetReceipt(
                arm_id.to_owned(),
            ));
        }
    }
    validate_recovery_topology(arm_id, &options.recovery_topology)?;
    Ok(())
}

fn validate_option_reserve_count(
    arm_id: &str,
    cluster_id: &str,
    summary: &ArmClusterSummary,
    reserves: &BTreeMap<ReserveKind, &ReserveObservation>,
) -> Result<(), CapabilityRetentionError> {
    if summary.option_set.disposition != EvidenceDisposition::Qualified {
        return Ok(());
    }
    if let Some(reserve) = reserves.get(&ReserveKind::RecoveryOptionCount) {
        if reserve.disposition == EvidenceDisposition::Qualified
            && reserve.terminal_lower_bound
                != Some(u64::from(
                    summary.option_set.terminal_qualified_option_count,
                ))
        {
            return Err(CapabilityRetentionError::OptionSetReserveCountMismatch {
                arm_id: arm_id.to_owned(),
                cluster_id: cluster_id.to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_reserves<'a>(
    arm_id: &str,
    policy: &ReservePolicy,
    observations: &'a [ReserveObservation],
) -> Result<BTreeMap<ReserveKind, &'a ReserveObservation>, CapabilityRetentionError> {
    let mut indexed = BTreeMap::new();
    for observation in observations {
        if indexed.insert(observation.kind, observation).is_some() {
            return Err(CapabilityRetentionError::DuplicateReserveObservation {
                arm_id: arm_id.to_owned(),
                kind: observation.kind,
            });
        }
    }

    let policy_map: BTreeMap<_, _> = policy.components.iter().map(|(k, v)| (*k, v)).collect();
    for (kind, applicability) in &policy_map {
        let observation = indexed.get(kind).ok_or_else(|| {
            CapabilityRetentionError::MissingReserveObservation {
                arm_id: arm_id.to_owned(),
                kind: *kind,
            }
        })?;
        match (applicability, observation.disposition) {
            (ReserveApplicability::Required { .. }, EvidenceDisposition::Rejected) => {
                return Err(CapabilityRetentionError::RejectedReserveEvidence {
                    arm_id: arm_id.to_owned(),
                    kind: *kind,
                });
            }
            (ReserveApplicability::Required { .. }, EvidenceDisposition::Qualified) => {
                if observation.terminal_lower_bound.is_none() {
                    return Err(CapabilityRetentionError::IncompleteQualifiedReserve {
                        arm_id: arm_id.to_owned(),
                        kind: *kind,
                    });
                }
                let receipt = observation
                    .source_receipt_digest
                    .as_deref()
                    .filter(|value| !value.trim().is_empty());
                if receipt.is_none() {
                    return Err(CapabilityRetentionError::IncompleteQualifiedReserve {
                        arm_id: arm_id.to_owned(),
                        kind: *kind,
                    });
                }
                if matches!(
                    applicability,
                    ReserveApplicability::Required {
                        calibration_required: true,
                        ..
                    }
                ) && !observation
                    .calibration_receipt_digest
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty())
                {
                    return Err(CapabilityRetentionError::IncompleteQualifiedReserve {
                        arm_id: arm_id.to_owned(),
                        kind: *kind,
                    });
                }
            }
            (ReserveApplicability::Required { .. }, EvidenceDisposition::Unknown) => {}
            (ReserveApplicability::Required { .. }, EvidenceDisposition::NotApplicable) => {
                return Err(CapabilityRetentionError::ReserveApplicabilityMismatch {
                    arm_id: arm_id.to_owned(),
                    kind: *kind,
                });
            }
            (
                ReserveApplicability::NotApplicable { reason },
                EvidenceDisposition::NotApplicable,
            ) if !reason.trim().is_empty()
                && observation.not_applicable_reason.as_deref() == Some(reason.as_str()) =>
            {
                if observation.terminal_lower_bound.is_some()
                    || observation.source_receipt_digest.is_some()
                    || observation.calibration_receipt_digest.is_some()
                {
                    return Err(CapabilityRetentionError::InvalidNotApplicableReserve {
                        arm_id: arm_id.to_owned(),
                        kind: *kind,
                    });
                }
            }
            (ReserveApplicability::NotApplicable { .. }, _) => {
                return Err(CapabilityRetentionError::ReserveApplicabilityMismatch {
                    arm_id: arm_id.to_owned(),
                    kind: *kind,
                });
            }
        }
        if let Some(calibration) = observation.calibration_receipt_digest.as_deref() {
            if calibration.trim().is_empty() {
                return Err(CapabilityRetentionError::IncompleteQualifiedReserve {
                    arm_id: arm_id.to_owned(),
                    kind: *kind,
                });
            }
        }
    }
    if indexed.len() != policy_map.len() {
        let unexpected = indexed
            .keys()
            .find(|kind| !policy_map.contains_key(kind))
            .copied()
            .unwrap_or(ReserveKind::EnergyReserveMilliWh);
        return Err(CapabilityRetentionError::ReserveApplicabilityMismatch {
            arm_id: arm_id.to_owned(),
            kind: unexpected,
        });
    }
    Ok(indexed)
}

fn reserve_deltas(
    policy: &ReservePolicy,
    baseline: &BTreeMap<ReserveKind, &ReserveObservation>,
    candidate: &BTreeMap<ReserveKind, &ReserveObservation>,
) -> Result<Vec<ReserveDelta>, CapabilityRetentionError> {
    let mut deltas = Vec::with_capacity(policy.components.len());
    for (kind, applicability) in &policy.components {
        match applicability {
            ReserveApplicability::NotApplicable { .. } => {
                deltas.push(ReserveDelta {
                    kind: *kind,
                    baseline_lower_bound: None,
                    candidate_lower_bound: None,
                    candidate_minus_baseline: None,
                    minimum_floor: None,
                    candidate_below_floor: false,
                    disposition: EvidenceDisposition::NotApplicable,
                });
            }
            ReserveApplicability::Required { minimum_floor, .. } => {
                let base = baseline.get(kind).ok_or(
                    CapabilityRetentionError::ReserveApplicabilityMismatch {
                        arm_id: "baseline".into(),
                        kind: *kind,
                    },
                )?;
                let cand = candidate.get(kind).ok_or(
                    CapabilityRetentionError::ReserveApplicabilityMismatch {
                        arm_id: "candidate".into(),
                        kind: *kind,
                    },
                )?;
                let disposition = if base.disposition == EvidenceDisposition::Unknown
                    || cand.disposition == EvidenceDisposition::Unknown
                {
                    EvidenceDisposition::Unknown
                } else {
                    EvidenceDisposition::Qualified
                };
                let baseline_lower = base.terminal_lower_bound;
                let candidate_lower = cand.terminal_lower_bound;
                let delta = baseline_lower
                    .zip(candidate_lower)
                    .map(|(left, right)| i128::from(right) - i128::from(left));
                deltas.push(ReserveDelta {
                    kind: *kind,
                    baseline_lower_bound: baseline_lower,
                    candidate_lower_bound: candidate_lower,
                    candidate_minus_baseline: delta,
                    minimum_floor: Some(*minimum_floor),
                    candidate_below_floor: disposition == EvidenceDisposition::Qualified
                        && candidate_lower.is_some_and(|value| value < *minimum_floor),
                    disposition,
                });
            }
        }
    }
    deltas.sort_by_key(|delta| delta.kind);
    Ok(deltas)
}

/// Compare accepted work together with the terminal physical capability vector.
///
/// This function is intentionally descriptive: it does not calculate a p-value,
/// confidence interval, statistical power, or universal capability score.
/// Independent-cluster inference remains owned by the paired comparison evaluator.
pub fn compare_work_and_terminal_capability(
    input: &CapabilityRetentionInput,
) -> Result<CapabilityRetentionReport, CapabilityRetentionError> {
    validate_scope(&input.scope, &input.reserve_policy)?;
    validate_arm(&input.scope, &input.baseline_manifest)?;
    validate_arm(&input.scope, &input.candidate_manifest)?;
    if input.baseline_manifest.arm_id == input.candidate_manifest.arm_id {
        return Err(CapabilityRetentionError::DuplicateArmId(
            input.baseline_manifest.arm_id.clone(),
        ));
    }
    if input.clusters.is_empty() {
        return Err(CapabilityRetentionError::DuplicateOrEmptyClusterId(
            "no-clusters".into(),
        ));
    }

    let mut cluster_ids = BTreeSet::new();
    let mut work_receipts = BTreeSet::new();
    let mut independence_receipts = BTreeSet::new();
    let mut reports = Vec::with_capacity(input.clusters.len());
    let mut unknown = false;
    let mut candidate_critical = false;
    let mut candidate_reserve_breach = false;

    for cluster in &input.clusters {
        require_nonempty("cluster_id", &cluster.cluster_id)?;
        require_nonempty(
            "paired_work_manifest_digest",
            &cluster.paired_work_manifest_digest,
        )
        .map_err(|_| CapabilityRetentionError::MissingClusterReceipt(cluster.cluster_id.clone()))?;
        require_nonempty(
            "independence_receipt_digest",
            &cluster.independence_receipt_digest,
        )
        .map_err(|_| CapabilityRetentionError::MissingClusterReceipt(cluster.cluster_id.clone()))?;

        if !cluster_ids.insert(cluster.cluster_id.clone()) {
            return Err(CapabilityRetentionError::DuplicateOrEmptyClusterId(
                cluster.cluster_id.clone(),
            ));
        }
        if !work_receipts.insert(cluster.paired_work_manifest_digest.clone())
            || !independence_receipts.insert(cluster.independence_receipt_digest.clone())
        {
            return Err(CapabilityRetentionError::DuplicateClusterReceipt(
                cluster.cluster_id.clone(),
            ));
        }
        if cluster.independence_disposition == EvidenceDisposition::Rejected {
            return Err(CapabilityRetentionError::RejectedIndependenceEvidence(
                cluster.cluster_id.clone(),
            ));
        }
        if cluster.independence_disposition == EvidenceDisposition::Unknown {
            unknown = true;
        }

        for (arm_id, summary) in [
            (&input.baseline_manifest.arm_id, &cluster.baseline),
            (&input.candidate_manifest.arm_id, &cluster.candidate),
        ] {
            summary.work.validate(arm_id, &cluster.cluster_id)?;
            require_nonempty(
                "work_outcome_receipt_digest",
                &summary.work_outcome_receipt_digest,
            )
            .map_err(|_| {
                CapabilityRetentionError::MissingClusterReceipt(cluster.cluster_id.clone())
            })?;
            require_nonempty(
                "operating_time_receipt_digest",
                &summary.operating_time_receipt_digest,
            )
            .map_err(|_| {
                CapabilityRetentionError::MissingClusterReceipt(cluster.cluster_id.clone())
            })?;
            if summary.available_robot_minutes == 0 {
                return Err(CapabilityRetentionError::ZeroAvailableRobotTime {
                    arm_id: arm_id.clone(),
                    cluster_id: cluster.cluster_id.clone(),
                });
            }
            if summary.nonproductive_downtime_minutes > summary.available_robot_minutes {
                return Err(CapabilityRetentionError::InvalidTimeAccounting {
                    arm_id: arm_id.clone(),
                    cluster_id: cluster.cluster_id.clone(),
                });
            }
            if summary.work.unknown_items > 0 || summary.work.held_items > 0 {
                unknown = true;
            }
            validate_option_set(arm_id, &summary.option_set)?;
        }

        if cluster.baseline.work.scheduled_items != cluster.candidate.work.scheduled_items {
            return Err(CapabilityRetentionError::PairingMismatch(
                cluster.cluster_id.clone(),
            ));
        }
        if cluster.baseline.available_robot_minutes != cluster.candidate.available_robot_minutes {
            return Err(CapabilityRetentionError::ScheduledRobotTimeMismatch(
                cluster.cluster_id.clone(),
            ));
        }
        if cluster.baseline.option_set.starting_option_set_digest
            != cluster.candidate.option_set.starting_option_set_digest
        {
            return Err(CapabilityRetentionError::StartingOptionSetMismatch(
                cluster.cluster_id.clone(),
            ));
        }
        if cluster.baseline.option_set.predecessor_state_digest
            != cluster.candidate.option_set.predecessor_state_digest
        {
            return Err(CapabilityRetentionError::StartingPhysicalStateMismatch(
                cluster.cluster_id.clone(),
            ));
        }
        if cluster.baseline.option_set.disposition == EvidenceDisposition::Unknown
            || cluster.candidate.option_set.disposition == EvidenceDisposition::Unknown
            || cluster.baseline.option_set.recovery_topology.disposition
                == EvidenceDisposition::Unknown
            || cluster.candidate.option_set.recovery_topology.disposition
                == EvidenceDisposition::Unknown
        {
            unknown = true;
        }

        let baseline_reserves = validate_reserves(
            &input.baseline_manifest.arm_id,
            &input.reserve_policy,
            &cluster.baseline.reserves,
        )?;
        let candidate_reserves = validate_reserves(
            &input.candidate_manifest.arm_id,
            &input.reserve_policy,
            &cluster.candidate.reserves,
        )?;
        if baseline_reserves
            .values()
            .chain(candidate_reserves.values())
            .any(|reserve| reserve.disposition == EvidenceDisposition::Unknown)
        {
            unknown = true;
        }

        validate_option_reserve_count(
            &input.baseline_manifest.arm_id,
            &cluster.cluster_id,
            &cluster.baseline,
            &baseline_reserves,
        )?;
        validate_option_reserve_count(
            &input.candidate_manifest.arm_id,
            &cluster.cluster_id,
            &cluster.candidate,
            &candidate_reserves,
        )?;
        let reserves = reserve_deltas(
            &input.reserve_policy,
            &baseline_reserves,
            &candidate_reserves,
        )?;
        if reserves.iter().any(|delta| delta.candidate_below_floor) {
            candidate_reserve_breach = true;
        }
        if cluster.candidate.critical_safety_violation_episodes > 0 {
            candidate_critical = true;
        }

        let baseline_rate = work_rate_milli_per_hour(
            cluster.baseline.work.accepted_items,
            cluster.baseline.available_robot_minutes,
        )?;
        let candidate_rate = work_rate_milli_per_hour(
            cluster.candidate.work.accepted_items,
            cluster.candidate.available_robot_minutes,
        )?;
        let baseline_option_count =
            if cluster.baseline.option_set.disposition == EvidenceDisposition::Qualified {
                Some(cluster.baseline.option_set.terminal_qualified_option_count)
            } else {
                None
            };
        let candidate_option_count =
            if cluster.candidate.option_set.disposition == EvidenceDisposition::Qualified {
                Some(cluster.candidate.option_set.terminal_qualified_option_count)
            } else {
                None
            };
        let baseline_topology = &cluster.baseline.option_set.recovery_topology;
        let candidate_topology = &cluster.candidate.option_set.recovery_topology;

        reports.push(ClusterCapabilityReport {
            cluster_id: cluster.cluster_id.clone(),
            scheduled_items: cluster.baseline.work.scheduled_items,
            baseline_accepted_items: cluster.baseline.work.accepted_items,
            candidate_accepted_items: cluster.candidate.work.accepted_items,
            baseline_work_rate_milli_per_hour: baseline_rate,
            candidate_work_rate_milli_per_hour: candidate_rate,
            work_rate_delta_milli_per_hour: i128::from(candidate_rate) - i128::from(baseline_rate),
            baseline_human_intervention_minutes: cluster.baseline.human_intervention_minutes,
            candidate_human_intervention_minutes: cluster.candidate.human_intervention_minutes,
            baseline_nonproductive_downtime_minutes: cluster
                .baseline
                .nonproductive_downtime_minutes,
            candidate_nonproductive_downtime_minutes: cluster
                .candidate
                .nonproductive_downtime_minutes,
            baseline_electrical_energy_milli_wh: cluster.baseline.electrical_energy_milli_wh,
            candidate_electrical_energy_milli_wh: cluster.candidate.electrical_energy_milli_wh,
            baseline_compute_energy_milli_joules: cluster.baseline.compute_energy_milli_joules,
            candidate_compute_energy_milli_joules: cluster.candidate.compute_energy_milli_joules,
            baseline_terminal_option_count: baseline_option_count,
            candidate_terminal_option_count: candidate_option_count,
            baseline_recovery_route_class_count: baseline_topology.recovery_route_class_count,
            candidate_recovery_route_class_count: candidate_topology.recovery_route_class_count,
            baseline_failure_domain_independent_route_count: baseline_topology
                .failure_domain_independent_route_count,
            candidate_failure_domain_independent_route_count: candidate_topology
                .failure_domain_independent_route_count,
            baseline_minimum_cut_set_cardinality: baseline_topology
                .minimum_failure_domain_cut_set_cardinality,
            candidate_minimum_cut_set_cardinality: candidate_topology
                .minimum_failure_domain_cut_set_cardinality,
            baseline_topology_report_digest: baseline_topology.report_digest.clone(),
            candidate_topology_report_digest: candidate_topology.report_digest.clone(),
            reserve_deltas: reserves,
            baseline_critical_safety_violation_episodes: cluster
                .baseline
                .critical_safety_violation_episodes,
            candidate_critical_safety_violation_episodes: cluster
                .candidate
                .critical_safety_violation_episodes,
            evidence_complete: cluster.independence_disposition == EvidenceDisposition::Qualified
                && cluster.baseline.option_set.disposition == EvidenceDisposition::Qualified
                && cluster.candidate.option_set.disposition == EvidenceDisposition::Qualified
                && baseline_topology.disposition == EvidenceDisposition::Qualified
                && candidate_topology.disposition == EvidenceDisposition::Qualified
                && cluster.baseline.work.unknown_items == 0
                && cluster.baseline.work.held_items == 0
                && cluster.candidate.work.unknown_items == 0
                && cluster.candidate.work.held_items == 0
                && reserves.iter().all(|delta| {
                    matches!(
                        delta.disposition,
                        EvidenceDisposition::Qualified | EvidenceDisposition::NotApplicable
                    )
                }),
        });
    }

    reports.sort_by(|a, b| a.cluster_id.cmp(&b.cluster_id));
    let disposition = if candidate_critical {
        CapabilityRetentionDisposition::CandidateCriticalSafetyObserved
    } else if candidate_reserve_breach {
        CapabilityRetentionDisposition::CandidateReserveFloorBreached
    } else if unknown {
        CapabilityRetentionDisposition::Inconclusive
    } else {
        CapabilityRetentionDisposition::EvidenceComplete
    };

    Ok(CapabilityRetentionReport {
        workcell_id: input.scope.workcell_id.clone(),
        baseline_arm_id: input.baseline_manifest.arm_id.clone(),
        candidate_arm_id: input.candidate_manifest.arm_id.clone(),
        disposition,
        clusters: reports,
        actuation_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reserve_policy() -> ReservePolicy {
        ReservePolicy {
            policy_digest: "reserve-policy-sha256".into(),
            components: vec![
                (
                    ReserveKind::EnergyReserveMilliWh,
                    ReserveApplicability::Required {
                        minimum_floor: 10_000,
                        calibration_required: true,
                    },
                ),
                (
                    ReserveKind::ThermalHeadroomMilliC,
                    ReserveApplicability::Required {
                        minimum_floor: 5_000,
                        calibration_required: true,
                    },
                ),
                (
                    ReserveKind::RecoveryOptionCount,
                    ReserveApplicability::Required {
                        minimum_floor: 1,
                        calibration_required: false,
                    },
                ),
                (
                    ReserveKind::SensorEligibilityPpm,
                    ReserveApplicability::NotApplicable {
                        reason: "No active sensor stack in this non-sensing fixture".into(),
                    },
                ),
            ],
        }
    }

    fn scope() -> CapabilityWorkScope {
        CapabilityWorkScope {
            workcell_id: "workcell-v1".into(),
            work_manifest_digest: "work-manifest-sha256".into(),
            robot_configuration_digest: "robot-config-sha256".into(),
            evaluator_digest: "acceptance-evaluator-sha256".into(),
            operating_envelope_digest: "operating-envelope-sha256".into(),
            resource_budget_digest: "resource-budget-sha256".into(),
            reserve_policy_digest: "reserve-policy-sha256".into(),
        }
    }

    fn manifest(arm_id: &str) -> ArmManifest {
        ArmManifest {
            arm_id: arm_id.into(),
            model_artifact_digest: format!("{arm_id}-model"),
            policy_digest: format!("{arm_id}-policy"),
            harness_digest: format!("{arm_id}-harness"),
            workcell_id: "workcell-v1".into(),
            work_manifest_digest: "work-manifest-sha256".into(),
            robot_configuration_digest: "robot-config-sha256".into(),
            evaluator_digest: "acceptance-evaluator-sha256".into(),
            operating_envelope_digest: "operating-envelope-sha256".into(),
            resource_budget_digest: "resource-budget-sha256".into(),
            reserve_policy_digest: "reserve-policy-sha256".into(),
        }
    }

    fn tally(accepted: u32, unknown: u32) -> WorkTally {
        WorkTally {
            scheduled_items: 10,
            accepted_items: accepted,
            rejected_items: 10 - accepted - unknown,
            reworked_items: 0,
            aborted_items: 0,
            held_items: 0,
            unknown_items: unknown,
        }
    }

    fn reserves(energy: u64, thermal: u64, options: u64) -> Vec<ReserveObservation> {
        vec![
            ReserveObservation {
                kind: ReserveKind::EnergyReserveMilliWh,
                disposition: EvidenceDisposition::Qualified,
                terminal_lower_bound: Some(energy),
                source_receipt_digest: Some("energy-receipt".into()),
                calibration_receipt_digest: Some("energy-calibration".into()),
                not_applicable_reason: None,
            },
            ReserveObservation {
                kind: ReserveKind::ThermalHeadroomMilliC,
                disposition: EvidenceDisposition::Qualified,
                terminal_lower_bound: Some(thermal),
                source_receipt_digest: Some("thermal-receipt".into()),
                calibration_receipt_digest: Some("thermal-calibration".into()),
                not_applicable_reason: None,
            },
            ReserveObservation {
                kind: ReserveKind::RecoveryOptionCount,
                disposition: EvidenceDisposition::Qualified,
                terminal_lower_bound: Some(options),
                source_receipt_digest: Some("option-reserve-receipt".into()),
                calibration_receipt_digest: None,
                not_applicable_reason: None,
            },
            ReserveObservation {
                kind: ReserveKind::SensorEligibilityPpm,
                disposition: EvidenceDisposition::NotApplicable,
                terminal_lower_bound: None,
                source_receipt_digest: None,
                calibration_receipt_digest: None,
                not_applicable_reason: Some(
                    "No active sensor stack in this non-sensing fixture".into(),
                ),
            },
        ]
    }

    fn topology() -> RecoveryTopologyReceipt {
        RecoveryTopologyReceipt {
            disposition: EvidenceDisposition::Qualified,
            report_digest: Some("recovery-topology-report-sha256".into()),
            recovery_route_class_count: Some(3),
            failure_domain_independent_route_count: Some(2),
            minimum_failure_domain_cut_set_cardinality: Some(1),
        }
    }

    fn option_set(terminal_digest: &str, count: u32) -> OptionSetReceipt {
        OptionSetReceipt {
            disposition: EvidenceDisposition::Qualified,
            ledger_receipt_digest: "option-ledger-receipt".into(),
            predecessor_state_digest: "initial-physical-state".into(),
            result_state_digest: format!("{terminal_digest}-state"),
            starting_option_set_digest: "starting-options".into(),
            terminal_option_set_digest: terminal_digest.into(),
            terminal_qualified_option_count: count,
            recovery_topology: topology(),
        }
    }

    fn arm_summary(
        accepted: u32,
        available_minutes: u32,
        energy: u64,
        thermal: u64,
        options: u64,
    ) -> ArmClusterSummary {
        ArmClusterSummary {
            work: tally(accepted, 0),
            work_outcome_receipt_digest: "accepted-work-receipt".into(),
            operating_time_receipt_digest: "operating-time-receipt".into(),
            available_robot_minutes: available_minutes,
            nonproductive_downtime_minutes: 0,
            human_intervention_minutes: 0,
            electrical_energy_milli_wh: 20_000,
            compute_energy_milli_joules: 100_000,
            critical_safety_violation_episodes: 0,
            reserves: reserves(energy, thermal, options),
            option_set: option_set("terminal-options", options as u32),
        }
    }

    fn input() -> CapabilityRetentionInput {
        CapabilityRetentionInput {
            scope: scope(),
            reserve_policy: reserve_policy(),
            baseline_manifest: manifest("baseline"),
            candidate_manifest: manifest("candidate"),
            clusters: vec![PairedCapabilityCluster {
                cluster_id: "cluster-A".into(),
                paired_work_manifest_digest: "paired-work-A".into(),
                independence_receipt_digest: "independence-A".into(),
                independence_disposition: EvidenceDisposition::Qualified,
                baseline: arm_summary(7, 60, 20_000, 9_000, 2),
                candidate: arm_summary(8, 60, 18_000, 8_000, 2),
            }],
        }
    }

    #[test]
    fn more_accepted_work_with_reserve_floors_preserved_is_reported_descriptively() {
        let report = compare_work_and_terminal_capability(&input()).unwrap();
        assert_eq!(
            report.disposition,
            CapabilityRetentionDisposition::EvidenceComplete
        );
        assert_eq!(report.clusters[0].baseline_accepted_items, 7);
        assert_eq!(report.clusters[0].candidate_accepted_items, 8);
        assert_eq!(report.clusters[0].work_rate_delta_milli_per_hour, 1_000);
        assert_eq!(
            report.clusters[0].baseline_terminal_option_count,
            Some(2)
        );
        assert_eq!(
            report.clusters[0].candidate_recovery_route_class_count,
            Some(3)
        );
        assert_eq!(
            report.clusters[0].candidate_failure_domain_independent_route_count,
            Some(2)
        );
        assert_eq!(
            report.clusters[0].candidate_minimum_cut_set_cardinality,
            Some(1)
        );
        assert!(!report.actuation_authorized);
    }

    #[test]
    fn unknown_topology_evidence_makes_comparison_inconclusive() {
        let mut audit = input();
        audit.clusters[0]
            .candidate
            .option_set
            .recovery_topology
            .disposition = EvidenceDisposition::Unknown;
        audit.clusters[0]
            .candidate
            .option_set
            .recovery_topology
            .recovery_route_class_count = None;
        audit.clusters[0]
            .candidate
            .option_set
            .recovery_topology
            .failure_domain_independent_route_count = None;
        let report = compare_work_and_terminal_capability(&audit).unwrap();
        assert_eq!(
            report.disposition,
            CapabilityRetentionDisposition::Inconclusive
        );
        assert!(!report.clusters[0].evidence_complete);
    }

    #[test]
    fn failure_domain_independent_routes_cannot_exceed_route_classes() {
        let mut audit = input();
        audit.clusters[0]
            .candidate
            .option_set
            .recovery_topology
            .failure_domain_independent_route_count = Some(4);
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::InconsistentRecoveryTopology(
                "candidate".into()
            ))
        );
    }

    #[test]
    fn qualified_topology_receipt_requires_digest_and_route_counts() {
        let mut audit = input();
        audit.clusters[0]
            .candidate
            .option_set
            .recovery_topology
            .report_digest = None;
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::IncompleteQualifiedRecoveryTopology(
                "candidate".into()
            ))
        );
    }

    #[test]
    fn rejected_topology_evidence_is_rejected() {
        let mut audit = input();
        audit.clusters[0]
            .candidate
            .option_set
            .recovery_topology
            .disposition = EvidenceDisposition::Rejected;
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::RejectedRecoveryTopologyEvidence(
                "candidate".into()
            ))
        );
    }

    #[test]
    fn more_work_but_thermal_headroom_below_floor_vetoes_positive_disposition() {
        let mut audit = input();
        audit.clusters[0].candidate.reserves = reserves(18_000, 4_000, 2);
        let report = compare_work_and_terminal_capability(&audit).unwrap();
        assert_eq!(
            report.disposition,
            CapabilityRetentionDisposition::CandidateReserveFloorBreached
        );
        assert!(report.clusters[0].reserve_deltas.iter().any(|delta| {
            delta.kind == ReserveKind::ThermalHeadroomMilliC && delta.candidate_below_floor
        }));
    }

    #[test]
    fn lower_energy_reserve_is_reported_without_being_hidden_by_more_work() {
        let report = compare_work_and_terminal_capability(&input()).unwrap();
        let energy = report.clusters[0]
            .reserve_deltas
            .iter()
            .find(|delta| delta.kind == ReserveKind::EnergyReserveMilliWh)
            .unwrap();
        assert_eq!(energy.candidate_minus_baseline, Some(-2_000));
        assert!(!energy.candidate_below_floor);
    }

    #[test]
    fn any_candidate_critical_safety_event_vetoes_positive_disposition() {
        let mut audit = input();
        audit.clusters[0]
            .candidate
            .critical_safety_violation_episodes = 1;
        let report = compare_work_and_terminal_capability(&audit).unwrap();
        assert_eq!(
            report.disposition,
            CapabilityRetentionDisposition::CandidateCriticalSafetyObserved
        );
    }

    #[test]
    fn unknown_work_outcomes_make_comparison_inconclusive() {
        let mut audit = input();
        audit.clusters[0].candidate.work = tally(7, 1);
        let report = compare_work_and_terminal_capability(&audit).unwrap();
        assert_eq!(
            report.disposition,
            CapabilityRetentionDisposition::Inconclusive
        );
    }

    #[test]
    fn unknown_required_reserve_provenance_makes_comparison_inconclusive() {
        let mut audit = input();
        audit.clusters[0].candidate.reserves[0].disposition = EvidenceDisposition::Unknown;
        audit.clusters[0].candidate.reserves[0].terminal_lower_bound = None;
        audit.clusters[0].candidate.reserves[0].source_receipt_digest = None;
        let report = compare_work_and_terminal_capability(&audit).unwrap();
        assert_eq!(
            report.disposition,
            CapabilityRetentionDisposition::Inconclusive
        );
    }

    #[test]
    fn unqualified_low_reserve_reading_cannot_trigger_a_hard_floor_veto() {
        let mut audit = input();
        audit.clusters[0].candidate.reserves[0].disposition = EvidenceDisposition::Unknown;
        audit.clusters[0].candidate.reserves[0].terminal_lower_bound = Some(1);
        audit.clusters[0].candidate.reserves[0].source_receipt_digest = None;
        let report = compare_work_and_terminal_capability(&audit).unwrap();
        assert_eq!(
            report.disposition,
            CapabilityRetentionDisposition::Inconclusive
        );
        let energy = report.clusters[0]
            .reserve_deltas
            .iter()
            .find(|delta| delta.kind == ReserveKind::EnergyReserveMilliWh)
            .unwrap();
        assert!(!energy.candidate_below_floor);
        assert_eq!(energy.disposition, EvidenceDisposition::Unknown);
    }

    #[test]
    fn calibration_required_reserve_without_calibration_receipt_is_rejected() {
        let mut audit = input();
        audit.clusters[0].candidate.reserves[1].calibration_receipt_digest = None;
        assert!(matches!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::IncompleteQualifiedReserve {
                arm_id,
                kind: ReserveKind::ThermalHeadroomMilliC
            }) if arm_id == "candidate"
        ));
    }

    #[test]
    fn missing_option_ledger_receipt_is_rejected() {
        let mut audit = input();
        audit.clusters[0]
            .candidate
            .option_set
            .ledger_receipt_digest
            .clear();
        assert!(matches!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::MissingOptionSetReceipt(_))
        ));
    }

    #[test]
    fn rejected_option_ledger_evidence_is_rejected() {
        let mut audit = input();
        audit.clusters[0].candidate.option_set.disposition = EvidenceDisposition::Rejected;
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::RejectedOptionSetEvidence(
                "candidate".into()
            ))
        );
    }

    #[test]
    fn starting_option_sets_must_be_identical_for_paired_comparison() {
        let mut audit = input();
        audit.clusters[0]
            .candidate
            .option_set
            .starting_option_set_digest = "different-start".into();
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::StartingOptionSetMismatch(
                "cluster-A".into()
            ))
        );
    }

    #[test]
    fn reserve_floor_policy_digest_must_match_the_frozen_scope() {
        let mut audit = input();
        audit.reserve_policy.policy_digest = "different-reserve-policy".into();
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::ReservePolicyMismatch)
        );
    }

    #[test]
    fn duplicate_reserve_policy_component_is_rejected() {
        let mut audit = input();
        audit
            .reserve_policy
            .components
            .push(audit.reserve_policy.components[0].clone());
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::DuplicateReservePolicyComponent(
                ReserveKind::EnergyReserveMilliWh
            ))
        );
    }

    #[test]
    fn not_applicable_reserve_requires_explicit_reason_and_no_measurement() {
        let mut audit = input();
        audit.clusters[0].candidate.reserves[3].not_applicable_reason = Some("".into());
        assert!(matches!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::ReserveApplicabilityMismatch {
                kind: ReserveKind::SensorEligibilityPpm,
                ..
            })
        ));
    }

    #[test]
    fn missing_reserve_component_is_rejected() {
        let mut audit = input();
        audit.clusters[0].candidate.reserves.pop();
        assert!(matches!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::MissingReserveObservation {
                kind: ReserveKind::SensorEligibilityPpm,
                ..
            })
        ));
    }

    #[test]
    fn duplicate_cluster_identity_is_rejected() {
        let mut audit = input();
        let mut second = audit.clusters[0].clone();
        second.paired_work_manifest_digest = "paired-work-B".into();
        second.independence_receipt_digest = "independence-B".into();
        audit.clusters.push(second);
        assert!(matches!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::DuplicateOrEmptyClusterId(cluster))
                if cluster == "cluster-A"
        ));
    }

    #[test]
    fn one_pair_receipt_cannot_be_reused_for_a_different_cluster() {
        let mut audit = input();
        let mut second = audit.clusters[0].clone();
        second.cluster_id = "cluster-B".into();
        second.paired_work_manifest_digest = "paired-work-B".into();
        audit.clusters.push(second);
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::DuplicateClusterReceipt(
                "cluster-B".into()
            ))
        );
    }

    #[test]
    fn candidate_and_baseline_must_share_work_scope() {
        let mut audit = input();
        audit.candidate_manifest.resource_budget_digest = "other-budget".into();
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::ArmScopeMismatch {
                arm_id: "candidate".into(),
                field: "resource_budget_digest",
            })
        );
    }

    #[test]
    fn work_tally_must_cover_every_scheduled_item() {
        let mut audit = input();
        audit.clusters[0].candidate.work.rejected_items -= 1;
        assert!(matches!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::InvalidWorkTally {
                arm_id,
                cluster_id
            }) if arm_id == "candidate" && cluster_id == "cluster-A"
        ));
    }

    #[test]
    fn paired_comparison_requires_the_same_scheduled_robot_time_denominator() {
        let mut audit = input();
        audit.clusters[0].candidate.available_robot_minutes = 30;
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::ScheduledRobotTimeMismatch(
                "cluster-A".into()
            ))
        );
    }

    #[test]
    fn downtime_is_reported_separately_from_accepted_work_and_robot_time() {
        let mut audit = input();
        audit.clusters[0].candidate.nonproductive_downtime_minutes = 12;
        let report = compare_work_and_terminal_capability(&audit).unwrap();
        assert_eq!(
            report.clusters[0].candidate_nonproductive_downtime_minutes,
            12
        );
        assert_eq!(report.clusters[0].candidate_work_rate_milli_per_hour, 8_000);
    }

    #[test]
    fn downtime_cannot_exceed_the_scheduled_robot_time() {
        let mut audit = input();
        audit.clusters[0].candidate.nonproductive_downtime_minutes = 61;
        assert!(matches!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::InvalidTimeAccounting {
                arm_id,
                cluster_id
            }) if arm_id == "candidate" && cluster_id == "cluster-A"
        ));
    }

    #[test]
    fn work_outcome_receipt_is_required() {
        let mut audit = input();
        audit.clusters[0]
            .candidate
            .work_outcome_receipt_digest
            .clear();
        assert!(matches!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::MissingClusterReceipt(cluster))
                if cluster == "cluster-A"
        ));
    }

    #[test]
    fn starting_physical_state_must_match_between_arms() {
        let mut audit = input();
        audit.clusters[0]
            .candidate
            .option_set
            .predecessor_state_digest = "different-physical-start".into();
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::StartingPhysicalStateMismatch(
                "cluster-A".into()
            ))
        );
    }

    #[test]
    fn qualified_option_count_must_match_the_recovery_reserve_vector() {
        let mut audit = input();
        audit.clusters[0].candidate.reserves[2].terminal_lower_bound = Some(1);
        assert_eq!(
            compare_work_and_terminal_capability(&audit),
            Err(CapabilityRetentionError::OptionSetReserveCountMismatch {
                arm_id: "candidate".into(),
                cluster_id: "cluster-A".into()
            })
        );
    }

    #[test]
    fn canonical_cluster_order_does_not_change_the_report() {
        let mut audit = input();
        let mut second = audit.clusters[0].clone();
        second.cluster_id = "cluster-B".into();
        second.paired_work_manifest_digest = "paired-work-B".into();
        second.independence_receipt_digest = "independence-B".into();
        second.baseline.option_set.ledger_receipt_digest = "ledger-B".into();
        second.candidate.option_set.ledger_receipt_digest = "ledger-B".into();
        audit.clusters.push(second);
        let report_a = compare_work_and_terminal_capability(&audit).unwrap();
        audit.clusters.reverse();
        let report_b = compare_work_and_terminal_capability(&audit).unwrap();
        assert_eq!(report_a, report_b);
    }

    #[test]
    fn work_rate_uses_scheduled_available_robot_minutes_not_powered_on_time() {
        let mut audit = input();
        audit.clusters[0].baseline.available_robot_minutes = 30;
        audit.clusters[0].candidate.available_robot_minutes = 30;
        let report = compare_work_and_terminal_capability(&audit).unwrap();
        assert_eq!(
            report.clusters[0].candidate_work_rate_milli_per_hour,
            16_000
        );
    }

    #[test]
    fn no_unknown_missing_or_censored_evidence_is_silently_counted_as_accepted() {
        let mut audit = input();
        audit.clusters[0].candidate.work.held_items = 1;
        audit.clusters[0].candidate.work.rejected_items -= 1;
        let report = compare_work_and_terminal_capability(&audit).unwrap();
        assert_eq!(
            report.disposition,
            CapabilityRetentionDisposition::Inconclusive
        );
        assert_eq!(report.clusters[0].candidate_accepted_items, 8);
    }
}
