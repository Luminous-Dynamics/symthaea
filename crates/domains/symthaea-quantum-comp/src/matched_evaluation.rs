//! Matched-cost evaluator for comparing trajectory-guided MIS methods.
//!
//! The evaluator refuses comparisons with mismatched instance identity, target,
//! protocol, verifier, solver budgets, or paired trial IDs. It retains failed and
//! unverified trials. Unknown stage costs remain unknown; they are never treated
//! as zero. A target hit requires a candidate checked by the declared independent
//! verifier, verified to be an independent set, and meeting the target cardinality.
//!
//! This is an evidence-accounting contract, not a solver and not a statistical
//! significance procedure. Trial pairs on one graph are repeated measurements;
//! graph-level generalization must be assessed across independent instances.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

/// Explicit boundary for an energy measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyScope {
    /// No energy measurement exists for this trial.
    NotMeasured,
    /// Energy measured at the optimization device only.
    DeviceOnly,
    /// Energy includes more than the device but not the complete declared system.
    PartialSystem,
    /// Energy covers the full declared system boundary.
    FullSystem,
}

/// Energy record with a mandatory measurement boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyRecord {
    /// Measured joules. None means no number was measured.
    pub joules: Option<f64>,
    /// Declared boundary covered by the measurement.
    pub scope: EnergyScope,
}

impl EnergyRecord {
    /// Construct a record explicitly stating that energy was not measured.
    pub const fn not_measured() -> Self {
        Self {
            joules: None,
            scope: EnergyScope::NotMeasured,
        }
    }

    fn validate(&self) -> Result<(), MatchedEvaluationError> {
        match (self.scope, self.joules) {
            (EnergyScope::NotMeasured, None) => Ok(()),
            (EnergyScope::NotMeasured, Some(_)) => {
                Err(MatchedEvaluationError::EnergyValueWithNotMeasuredScope)
            }
            (_, None) => Err(MatchedEvaluationError::EnergyScopeWithoutValue),
            (_, Some(joules)) if !joules.is_finite() || joules <= 0.0 => {
                Err(MatchedEvaluationError::InvalidEnergy(joules))
            }
            _ => Ok(()),
        }
    }
}

/// Stage-by-stage wall-clock cost in milliseconds.
///
/// Each field is Option-valued: None means unknown or not measured, while
/// Some(0.0) means the stage was measured and consumed zero milliseconds at the
/// chosen clock resolution. The all-in total is unavailable if any stage is
/// unknown. Zero must never be used as a placeholder for missing data.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StageCostsMs {
    /// Trajectory acquisition / device execution.
    pub acquisition_ms: Option<f64>,
    /// Graph reading, canonicalization, and preprocessing.
    pub preprocessing_ms: Option<f64>,
    /// Trajectory feature extraction or controller inference.
    pub feature_extraction_ms: Option<f64>,
    /// Data movement to and from a sampler or accelerator.
    pub transfer_ms: Option<f64>,
    /// Amortized compilation and initialization cost under a declared policy.
    pub compile_init_amortized_ms: Option<f64>,
    /// Downstream classical or quantum optimization.
    pub solve_ms: Option<f64>,
    /// Classical repair or reconstruction.
    pub repair_ms: Option<f64>,
    /// Independent feasibility / target verification.
    pub verification_ms: Option<f64>,
    /// Receipt construction, hashing, and output persistence.
    pub receipt_ms: Option<f64>,
}

impl StageCostsMs {
    fn values(&self) -> [Option<f64>; 9] {
        [
            self.acquisition_ms,
            self.preprocessing_ms,
            self.feature_extraction_ms,
            self.transfer_ms,
            self.compile_init_amortized_ms,
            self.solve_ms,
            self.repair_ms,
            self.verification_ms,
            self.receipt_ms,
        ]
    }

    /// Returns true when every required cost stage is measured.
    pub fn is_complete(&self) -> bool {
        self.values().iter().all(Option::is_some)
    }

    /// Returns the all-in cost in milliseconds, or None if any stage is unknown.
    ///
    /// Call validate_trial / MatchedArmReport::validate before treating this as
    /// audited evidence; a complete all-zero vector is invalid for an actual run.
    pub fn all_in_total_ms(&self) -> Option<f64> {
        let values = self.values();
        if values.iter().any(Option::is_none) {
            return None;
        }
        Some(values.into_iter().flatten().sum())
    }

    fn validate(&self, trial_id: &str) -> Result<(), MatchedEvaluationError> {
        for (stage, value) in [
            ("acquisition_ms", self.acquisition_ms),
            ("preprocessing_ms", self.preprocessing_ms),
            ("feature_extraction_ms", self.feature_extraction_ms),
            ("transfer_ms", self.transfer_ms),
            ("compile_init_amortized_ms", self.compile_init_amortized_ms),
            ("solve_ms", self.solve_ms),
            ("repair_ms", self.repair_ms),
            ("verification_ms", self.verification_ms),
            ("receipt_ms", self.receipt_ms),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value < 0.0 {
                    return Err(MatchedEvaluationError::InvalidStageCost {
                        trial_id: trial_id.to_owned(),
                        stage,
                        value,
                    });
                }
            }
        }

        if self.is_complete() {
            let total = self.all_in_total_ms().unwrap_or(f64::NAN);
            if !total.is_finite() || total <= 0.0 {
                return Err(MatchedEvaluationError::InvalidAllInCost {
                    trial_id: trial_id.to_owned(),
                    value: total,
                });
            }
        }
        Ok(())
    }
}

/// Operational termination state of one trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialTermination {
    /// Solver stopped under its declared normal stopping condition.
    Completed,
    /// Solver or sampler hit its time limit.
    TimedOut,
    /// Solver or sampler failed.
    Failed,
    /// Trial was cancelled or its budget was withdrawn.
    Aborted,
}

/// Independent verification result for a candidate returned by a trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateVerification {
    /// No candidate was returned.
    NotReturned,
    /// Candidate exists but has not been independently checked.
    Unverified,
    /// Independent verifier confirmed a feasible independent set.
    VerifiedIndependentSet,
    /// Independent verifier found at least one edge conflict.
    VerifiedConflict,
    /// The verifier failed or could not reach a trustworthy conclusion.
    VerificationFailed,
}

/// One attempted solver run, including failed and unverified outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchedTrialRecord {
    /// Unique identifier for this attempted run within an arm.
    pub trial_id: String,
    /// Pair identifier shared by the corresponding baseline/candidate trial.
    pub pair_id: String,
    /// Declared PRNG seed or reproducibility identifier for this pair.
    pub seed: u64,
    /// Operational termination state.
    pub termination: TrialTermination,
    /// Candidate size / objective value when available. For MIS this is cardinality.
    pub candidate_size: Option<usize>,
    /// Independent verification outcome.
    pub verification: CandidateVerification,
    /// Number of graph-edge conflicts reported by the verifier, if available.
    pub conflict_count: Option<usize>,
    /// Stage-by-stage time costs. Missing stages remain unknown.
    pub costs: StageCostsMs,
    /// Energy measurement and scope.
    pub energy: EnergyRecord,
}

impl MatchedTrialRecord {
    /// True only when independent feasibility was confirmed and the target is met.
    pub fn verified_target_hit(&self, target_size: usize) -> bool {
        self.verification == CandidateVerification::VerifiedIndependentSet
            && self.conflict_count == Some(0)
            && self.candidate_size.is_some_and(|size| size >= target_size)
    }

    fn validate(&self) -> Result<(), MatchedEvaluationError> {
        if self.trial_id.trim().is_empty() {
            return Err(MatchedEvaluationError::EmptyTrialId);
        }
        if self.pair_id.trim().is_empty() {
            return Err(MatchedEvaluationError::EmptyPairId(self.trial_id.clone()));
        }

        match self.verification {
            CandidateVerification::NotReturned => {
                if self.candidate_size.is_some() || self.conflict_count.is_some() {
                    return Err(MatchedEvaluationError::InconsistentCandidateRecord(
                        self.trial_id.clone(),
                    ));
                }
            }
            CandidateVerification::Unverified | CandidateVerification::VerificationFailed => {
                if self.candidate_size.is_none() || self.conflict_count.is_some() {
                    return Err(MatchedEvaluationError::InconsistentCandidateRecord(
                        self.trial_id.clone(),
                    ));
                }
            }
            CandidateVerification::VerifiedIndependentSet => {
                if self.candidate_size.is_none() || self.conflict_count != Some(0) {
                    return Err(MatchedEvaluationError::InconsistentCandidateRecord(
                        self.trial_id.clone(),
                    ));
                }
            }
            CandidateVerification::VerifiedConflict => {
                if self.candidate_size.is_none()
                    || self.conflict_count.is_none_or(|count| count == 0)
                {
                    return Err(MatchedEvaluationError::InconsistentCandidateRecord(
                        self.trial_id.clone(),
                    ));
                }
            }
        }

        self.costs.validate(&self.trial_id)?;
        self.energy.validate()?;
        Ok(())
    }
}

/// All run-level fields that must match for a fair paired comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchedArmReport {
    /// Identifier for this method/arm.
    pub arm_id: String,
    /// Human-readable solver / hybrid pipeline name.
    pub method_id: String,
    /// SHA-256 of canonical original graph bytes.
    pub instance_sha256: String,
    /// SHA-256 of the frozen comparison protocol.
    pub protocol_sha256: String,
    /// Declared cardinality target. This need not be a proven optimum.
    pub target_size: usize,
    /// Maximum solver budget per paired trial, in milliseconds.
    pub per_trial_solver_budget_ms: f64,
    /// Tuning budget assigned to this method, in milliseconds.
    pub tuning_budget_ms: f64,
    /// Source revision (40- or 64-character hexadecimal commit ID).
    pub source_revision: String,
    /// SHA-256 of the method configuration.
    pub configuration_sha256: String,
    /// SHA-256 of the independent verifier build / source.
    pub verifier_sha256: String,
    /// Explicit assertion that the verifier is independent of this solver's
    /// internal candidate-generation and acceptance logic.
    pub verifier_is_independent: bool,
    /// Every attempted run, including failure, timeout, and unverified candidate.
    pub trials: Vec<MatchedTrialRecord>,
}

impl MatchedArmReport {
    /// Validate identities, digests, budgets, and all trial records.
    pub fn validate(&self) -> Result<(), MatchedEvaluationError> {
        if self.arm_id.trim().is_empty() || self.method_id.trim().is_empty() {
            return Err(MatchedEvaluationError::EmptyArmIdentity);
        }
        validate_sha256("instance_sha256", &self.instance_sha256)?;
        validate_sha256("protocol_sha256", &self.protocol_sha256)?;
        validate_sha256("configuration_sha256", &self.configuration_sha256)?;
        validate_commit_id(&self.source_revision)?;
        validate_sha256("verifier_sha256", &self.verifier_sha256)?;

        if self.target_size == 0 {
            return Err(MatchedEvaluationError::ZeroTarget);
        }
        if !self.per_trial_solver_budget_ms.is_finite()
            || self.per_trial_solver_budget_ms <= 0.0
            || !self.tuning_budget_ms.is_finite()
            || self.tuning_budget_ms < 0.0
        {
            return Err(MatchedEvaluationError::InvalidBudget);
        }
        if !self.verifier_is_independent {
            return Err(MatchedEvaluationError::IndependentVerifierRequired);
        }
        if self.trials.is_empty() {
            return Err(MatchedEvaluationError::NoTrials(self.arm_id.clone()));
        }

        let mut trial_ids = HashSet::with_capacity(self.trials.len());
        let mut pair_ids = HashSet::with_capacity(self.trials.len());
        for trial in &self.trials {
            trial.validate()?;
            if !trial_ids.insert(trial.trial_id.clone()) {
                return Err(MatchedEvaluationError::DuplicateTrialId(
                    trial.trial_id.clone(),
                ));
            }
            if !pair_ids.insert(trial.pair_id.clone()) {
                return Err(MatchedEvaluationError::DuplicatePairId {
                    arm_id: self.arm_id.clone(),
                    pair_id: trial.pair_id.clone(),
                });
            }
        }
        Ok(())
    }
}

/// How the all-in runtime comparison ranked one paired run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeWinner {
    /// Candidate reached the same verified target faster.
    Candidate,
    /// Baseline reached the same verified target faster.
    Baseline,
    /// All-in times were equal within the declared tolerance.
    Tie,
}

/// Per-pair result. Missing cost or energy ratios are explicit, never zero.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchedPairResult {
    /// Pair ID present in both arms.
    pub pair_id: String,
    /// Whether the baseline independently verified a target hit.
    pub baseline_target_hit: bool,
    /// Whether the candidate independently verified a target hit.
    pub candidate_target_hit: bool,
    /// Baseline all-in wall time, only when all stages were measured.
    pub baseline_all_in_ms: Option<f64>,
    /// Candidate all-in wall time, only when all stages were measured.
    pub candidate_all_in_ms: Option<f64>,
    /// Baseline time divided by candidate time; only when both hit the target and
    /// both complete all-in costs are available.
    pub time_speedup_baseline_over_candidate: Option<f64>,
    /// Time winner when both methods hit the target and have complete costs.
    pub time_winner: Option<TimeWinner>,
    /// Baseline energy joules, if measured.
    pub baseline_energy_joules: Option<f64>,
    /// Candidate energy joules, if measured.
    pub candidate_energy_joules: Option<f64>,
    /// Baseline energy divided by candidate energy, only for matching measured
    /// energy scopes and paired verified target hits.
    pub energy_speedup_baseline_over_candidate: Option<f64>,
}

/// Machine-readable summary of a matched comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchedComparisonSummary {
    /// Baseline arm ID.
    pub baseline_arm_id: String,
    /// Candidate arm ID.
    pub candidate_arm_id: String,
    /// Number of complete pairs in this comparison.
    pub pair_count: usize,
    /// Number of pairs where the baseline reached its verified target.
    pub baseline_target_hits: usize,
    /// Number of pairs where the candidate reached its verified target.
    pub candidate_target_hits: usize,
    /// Number of pairs where both methods reached the target.
    pub both_target_hits: usize,
    /// Pairs where only the candidate reached the target.
    pub candidate_only_target_hits: usize,
    /// Pairs where only the baseline reached the target.
    pub baseline_only_target_hits: usize,
    /// Pairs where neither method reached the target.
    pub neither_target_hit: usize,
    /// Both-hit pairs with complete cost vectors for both methods.
    pub both_hit_costed_pairs: usize,
    /// Target-hit trials whose all-in cost cannot be computed because at least one
    /// stage is unknown.
    pub baseline_uncosted_target_hits: usize,
    /// Target-hit trials whose all-in cost cannot be computed because at least one
    /// stage is unknown.
    pub candidate_uncosted_target_hits: usize,
    /// Number of paired energy comparisons with matching measured boundaries.
    pub comparable_energy_pairs: usize,
    /// Both-hit pairs where measured energy boundaries differ.
    pub mismatched_energy_scope_pairs: usize,
    /// Median paired baseline/candidate time ratio, restricted to both-hit costed
    /// pairs. Values greater than 1 favor the candidate on runtime.
    pub median_time_speedup: Option<f64>,
    /// Median paired baseline/candidate energy ratio, restricted to both-hit pairs
    /// with matching measured boundaries. Values greater than 1 favor the candidate.
    pub median_energy_speedup: Option<f64>,
    /// Complete per-pair details so the summary is auditable.
    pub pairs: Vec<MatchedPairResult>,
}

/// Validation, accounting, or comparison error.
#[derive(Debug, Clone, PartialEq)]
pub enum MatchedEvaluationError {
    /// Arm identity is empty.
    EmptyArmIdentity,
    /// SHA-256 field is malformed.
    InvalidSha256 { field: &'static str },
    /// Source revision is not a 40- or 64-character hex commit ID.
    InvalidCommitId,
    /// MIS target is zero.
    ZeroTarget,
    /// A budget is invalid.
    InvalidBudget,
    /// Comparison requires a verifier independent from candidate generation.
    IndependentVerifierRequired,
    /// Arm has no attempted trials.
    NoTrials(String),
    /// Trial ID is empty.
    EmptyTrialId,
    /// Pair ID is empty.
    EmptyPairId(String),
    /// Trial ID repeats in an arm.
    DuplicateTrialId(String),
    /// Pair ID repeats in one arm.
    DuplicatePairId { arm_id: String, pair_id: String },
    /// Candidate size and verification fields are inconsistent.
    InconsistentCandidateRecord(String),
    /// A stage cost is negative or non-finite.
    InvalidStageCost { trial_id: String, stage: &'static str, value: f64 },
    /// Complete all-in measured cost is non-finite or non-positive.
    InvalidAllInCost { trial_id: String, value: f64 },
    /// A derived speed ratio overflowed or was non-finite.
    InvalidRatio { quantity: &'static str },
    /// Energy is present despite the scope being NotMeasured.
    EnergyValueWithNotMeasuredScope,
    /// A measured energy scope has no joule value.
    EnergyScopeWithoutValue,
    /// Measured joules are not finite and positive.
    InvalidEnergy(f64),
    /// The baseline and candidate describe different graph instances.
    InstanceMismatch,
    /// The arms use different protocol digests.
    ProtocolMismatch,
    /// The arms use different target cardinalities.
    TargetMismatch,
    /// The arms use different per-trial solver budgets.
    SolverBudgetMismatch,
    /// The arms use different tuning budgets.
    TuningBudgetMismatch,
    /// The arms use different independent verifier builds.
    VerifierMismatch,
    /// Pair IDs differ between arms; no silent inner-join is permitted.
    PairSetMismatch,
    /// A shared pair ID has different seed identity between arms.
    PairSeedMismatch(String),
    /// Time tie tolerance is negative or non-finite.
    InvalidTieTolerance,
}

impl fmt::Display for MatchedEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyArmIdentity => write!(f, "arm ID and method ID must not be empty"),
            Self::InvalidSha256 { field } => write!(f, "invalid SHA-256 digest in {field}"),
            Self::InvalidCommitId => write!(f, "source revision must be 40 or 64 hexadecimal characters"),
            Self::ZeroTarget => write!(f, "target size must be positive"),
            Self::InvalidBudget => write!(f, "budgets must be finite and valid"),
            Self::IndependentVerifierRequired => write!(f, "an independent verifier is required"),
            Self::NoTrials(arm) => write!(f, "arm {arm} has no trial records"),
            Self::EmptyTrialId => write!(f, "trial ID must not be empty"),
            Self::EmptyPairId(trial) => write!(f, "trial {trial} has an empty pair ID"),
            Self::DuplicateTrialId(id) => write!(f, "duplicate trial ID: {id}"),
            Self::DuplicatePairId { arm_id, pair_id } => {
                write!(f, "arm {arm_id} duplicates pair ID {pair_id}")
            }
            Self::InconsistentCandidateRecord(id) => {
                write!(f, "trial {id} has inconsistent candidate-verification fields")
            }
            Self::InvalidStageCost { trial_id, stage, value } => {
                write!(f, "trial {trial_id} has invalid {stage} value {value}")
            }
            Self::InvalidAllInCost { trial_id, value } => write!(
                f,
                "trial {trial_id} has invalid all-in cost {value}; it must be finite and positive"
            ),
            Self::InvalidRatio { quantity } => write!(
                f,
                "derived {quantity} ratio was not finite and positive"
            ),
            Self::EnergyValueWithNotMeasuredScope => {
                write!(f, "energy value supplied with NotMeasured scope")
            }
            Self::EnergyScopeWithoutValue => {
                write!(f, "measured energy scope supplied without a joule value")
            }
            Self::InvalidEnergy(joules) => {
                write!(f, "measured energy must be finite and positive, got {joules}")
            }
            Self::InstanceMismatch => write!(f, "baseline and candidate instance digests differ"),
            Self::ProtocolMismatch => write!(f, "baseline and candidate protocol digests differ"),
            Self::TargetMismatch => write!(f, "baseline and candidate target sizes differ"),
            Self::SolverBudgetMismatch => write!(f, "per-trial solver budgets differ"),
            Self::TuningBudgetMismatch => write!(f, "method tuning budgets differ"),
            Self::VerifierMismatch => write!(f, "independent verifier identity differs"),
            Self::PairSetMismatch => write!(f, "baseline and candidate pair-ID sets differ"),
            Self::PairSeedMismatch(id) => write!(f, "pair {id} uses different seed identities"),
            Self::InvalidTieTolerance => write!(f, "tie tolerance must be finite and non-negative"),
        }
    }
}

impl Error for MatchedEvaluationError {}

/// Compare a baseline and candidate under a strict paired protocol.
///
/// Every pair must be present in both arms with matching seeds. Target hits are
/// derived only from independently verified feasible candidates. Speedup ratios
/// are only computed when both methods hit the same target and have complete
/// positive all-in costs. A failed target remains in the hit-rate counts and does
/// not receive an invented runtime ratio.
pub fn compare_matched_arms(
    baseline: &MatchedArmReport,
    candidate: &MatchedArmReport,
    time_tie_tolerance_ms: f64,
) -> Result<MatchedComparisonSummary, MatchedEvaluationError> {
    baseline.validate()?;
    candidate.validate()?;
    if !time_tie_tolerance_ms.is_finite() || time_tie_tolerance_ms < 0.0 {
        return Err(MatchedEvaluationError::InvalidTieTolerance);
    }
    if baseline.instance_sha256 != candidate.instance_sha256 {
        return Err(MatchedEvaluationError::InstanceMismatch);
    }
    if baseline.protocol_sha256 != candidate.protocol_sha256 {
        return Err(MatchedEvaluationError::ProtocolMismatch);
    }
    if baseline.target_size != candidate.target_size {
        return Err(MatchedEvaluationError::TargetMismatch);
    }
    if baseline.per_trial_solver_budget_ms != candidate.per_trial_solver_budget_ms {
        return Err(MatchedEvaluationError::SolverBudgetMismatch);
    }
    if baseline.tuning_budget_ms != candidate.tuning_budget_ms {
        return Err(MatchedEvaluationError::TuningBudgetMismatch);
    }
    if baseline.verifier_sha256 != candidate.verifier_sha256 {
        return Err(MatchedEvaluationError::VerifierMismatch);
    }

    let baseline_by_pair: HashMap<&str, &MatchedTrialRecord> = baseline
        .trials
        .iter()
        .map(|trial| (trial.pair_id.as_str(), trial))
        .collect();
    let candidate_by_pair: HashMap<&str, &MatchedTrialRecord> = candidate
        .trials
        .iter()
        .map(|trial| (trial.pair_id.as_str(), trial))
        .collect();

    let baseline_ids: HashSet<&str> = baseline_by_pair.keys().copied().collect();
    let candidate_ids: HashSet<&str> = candidate_by_pair.keys().copied().collect();
    if baseline_ids != candidate_ids {
        return Err(MatchedEvaluationError::PairSetMismatch);
    }

    let mut pairs = Vec::with_capacity(baseline.trials.len());
    let mut baseline_target_hits = 0;
    let mut candidate_target_hits = 0;
    let mut both_target_hits = 0;
    let mut candidate_only_target_hits = 0;
    let mut baseline_only_target_hits = 0;
    let mut neither_target_hit = 0;
    let mut both_hit_costed_pairs = 0;
    let mut baseline_uncosted_target_hits = 0;
    let mut candidate_uncosted_target_hits = 0;
    let mut comparable_energy_pairs = 0;
    let mut mismatched_energy_scope_pairs = 0;
    let mut time_speedups = Vec::new();
    let mut energy_speedups = Vec::new();

    for baseline_trial in &baseline.trials {
        let candidate_trial = candidate_by_pair
            .get(baseline_trial.pair_id.as_str())
            .expect("pair sets checked above");

        if baseline_trial.seed != candidate_trial.seed {
            return Err(MatchedEvaluationError::PairSeedMismatch(
                baseline_trial.pair_id.clone(),
            ));
        }

        let baseline_hit = baseline_trial.verified_target_hit(baseline.target_size);
        let candidate_hit = candidate_trial.verified_target_hit(candidate.target_size);
        if baseline_hit {
            baseline_target_hits += 1;
        }
        if candidate_hit {
            candidate_target_hits += 1;
        }

        if baseline_hit && candidate_hit {
            both_target_hits += 1;
        } else if candidate_hit {
            candidate_only_target_hits += 1;
        } else if baseline_hit {
            baseline_only_target_hits += 1;
        } else {
            neither_target_hit += 1;
        }

        let baseline_all_in_ms = baseline_trial.costs.all_in_total_ms();
        let candidate_all_in_ms = candidate_trial.costs.all_in_total_ms();

        if baseline_hit && baseline_all_in_ms.is_none() {
            baseline_uncosted_target_hits += 1;
        }
        if candidate_hit && candidate_all_in_ms.is_none() {
            candidate_uncosted_target_hits += 1;
        }

        let (time_speedup, time_winner) = if baseline_hit && candidate_hit {
            match (baseline_all_in_ms, candidate_all_in_ms) {
                (Some(base_ms), Some(candidate_ms)) => {
                    both_hit_costed_pairs += 1;
                    let ratio = base_ms / candidate_ms;
                    if !ratio.is_finite() || ratio <= 0.0 {
                        return Err(MatchedEvaluationError::InvalidRatio {
                            quantity: "time-speedup",
                        });
                    }
                    time_speedups.push(ratio);
                    let winner = if candidate_ms + time_tie_tolerance_ms < base_ms {
                        Some(TimeWinner::Candidate)
                    } else if base_ms + time_tie_tolerance_ms < candidate_ms {
                        Some(TimeWinner::Baseline)
                    } else {
                        Some(TimeWinner::Tie)
                    };
                    (Some(ratio), winner)
                }
                _ => (None, None),
            }
        } else {
            (None, None)
        };

        let baseline_energy = baseline_trial.energy.joules;
        let candidate_energy = candidate_trial.energy.joules;
        let energy_speedup = if baseline_hit && candidate_hit {
            if baseline_trial.energy.scope == candidate_trial.energy.scope
                && baseline_trial.energy.scope != EnergyScope::NotMeasured
            {
                match (baseline_energy, candidate_energy) {
                    (Some(base_j), Some(candidate_j)) => {
                        comparable_energy_pairs += 1;
                        let ratio = base_j / candidate_j;
                        if !ratio.is_finite() || ratio <= 0.0 {
                            return Err(MatchedEvaluationError::InvalidRatio {
                                quantity: "energy-speedup",
                            });
                        }
                        energy_speedups.push(ratio);
                        Some(ratio)
                    }
                    _ => None,
                }
            } else {
                mismatched_energy_scope_pairs += 1;
                None
            }
        } else {
            None
        };

        pairs.push(MatchedPairResult {
            pair_id: baseline_trial.pair_id.clone(),
            baseline_target_hit: baseline_hit,
            candidate_target_hit: candidate_hit,
            baseline_all_in_ms,
            candidate_all_in_ms,
            time_speedup_baseline_over_candidate: time_speedup,
            time_winner,
            baseline_energy_joules: baseline_energy,
            candidate_energy_joules: candidate_energy,
            energy_speedup_baseline_over_candidate: energy_speedup,
        });
    }

    Ok(MatchedComparisonSummary {
        baseline_arm_id: baseline.arm_id.clone(),
        candidate_arm_id: candidate.arm_id.clone(),
        pair_count: pairs.len(),
        baseline_target_hits,
        candidate_target_hits,
        both_target_hits,
        candidate_only_target_hits,
        baseline_only_target_hits,
        neither_target_hit,
        both_hit_costed_pairs,
        baseline_uncosted_target_hits,
        candidate_uncosted_target_hits,
        comparable_energy_pairs,
        mismatched_energy_scope_pairs,
        median_time_speedup: median(&mut time_speedups),
        median_energy_speedup: median(&mut energy_speedups),
        pairs,
    })
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[middle - 1] + values[middle]) / 2.0)
    } else {
        Some(values[middle])
    }
}

fn validate_sha256(field: &'static str, value: &str) -> Result<(), MatchedEvaluationError> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(MatchedEvaluationError::InvalidSha256 { field });
    }
    Ok(())
}

fn validate_commit_id(value: &str) -> Result<(), MatchedEvaluationError> {
    if (value.len() == 40 || value.len() == 64)
        && value.bytes().all(|b| b.is_ascii_hexdigit())
    {
        Ok(())
    } else {
        Err(MatchedEvaluationError::InvalidCommitId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn costs(total_ms: f64) -> StageCostsMs {
        StageCostsMs {
            acquisition_ms: Some(total_ms * 0.1),
            preprocessing_ms: Some(total_ms * 0.1),
            feature_extraction_ms: Some(total_ms * 0.1),
            transfer_ms: Some(total_ms * 0.05),
            compile_init_amortized_ms: Some(total_ms * 0.05),
            solve_ms: Some(total_ms * 0.35),
            repair_ms: Some(total_ms * 0.1),
            verification_ms: Some(total_ms * 0.1),
            receipt_ms: Some(total_ms * 0.05),
        }
    }

    fn verified_trial(
        id: &str,
        pair: &str,
        seed: u64,
        size: usize,
        stage_costs: StageCostsMs,
        energy: EnergyRecord,
    ) -> MatchedTrialRecord {
        MatchedTrialRecord {
            trial_id: id.to_owned(),
            pair_id: pair.to_owned(),
            seed,
            termination: TrialTermination::Completed,
            candidate_size: Some(size),
            verification: CandidateVerification::VerifiedIndependentSet,
            conflict_count: Some(0),
            costs: stage_costs,
            energy,
        }
    }

    fn report(arm_id: &str, method_id: &str, trials: Vec<MatchedTrialRecord>) -> MatchedArmReport {
        MatchedArmReport {
            arm_id: arm_id.to_owned(),
            method_id: method_id.to_owned(),
            instance_sha256: "a".repeat(64),
            protocol_sha256: "b".repeat(64),
            target_size: 5,
            per_trial_solver_budget_ms: 1_000.0,
            tuning_budget_ms: 10_000.0,
            source_revision: "d".repeat(40),
            configuration_sha256: if arm_id == "baseline" {
                "e".repeat(64)
            } else {
                "f".repeat(64)
            },
            verifier_sha256: "c".repeat(64),
            verifier_is_independent: true,
            trials,
        }
    }

    fn measured_energy(joules: f64, scope: EnergyScope) -> EnergyRecord {
        EnergyRecord {
            joules: Some(joules),
            scope,
        }
    }

    #[test]
    fn computes_paired_speedup_only_for_both_verified_target_hits() {
        let baseline = report(
            "baseline",
            "classical",
            vec![
                verified_trial("b1", "pair-1", 1, 5, costs(100.0), measured_energy(10.0, EnergyScope::FullSystem)),
                MatchedTrialRecord {
                    trial_id: "b2".into(),
                    pair_id: "pair-2".into(),
                    seed: 2,
                    termination: TrialTermination::TimedOut,
                    candidate_size: None,
                    verification: CandidateVerification::NotReturned,
                    conflict_count: None,
                    costs: costs(1_000.0),
                    energy: EnergyRecord::not_measured(),
                },
            ],
        );
        let candidate = report(
            "candidate",
            "trajectory-hybrid",
            vec![
                verified_trial("c1", "pair-1", 1, 6, costs(50.0), measured_energy(5.0, EnergyScope::FullSystem)),
                verified_trial("c2", "pair-2", 2, 5, costs(80.0), EnergyRecord::not_measured()),
            ],
        );

        let result = compare_matched_arms(&baseline, &candidate, 0.0).unwrap();
        assert_eq!(result.pair_count, 2);
        assert_eq!(result.baseline_target_hits, 1);
        assert_eq!(result.candidate_target_hits, 2);
        assert_eq!(result.both_target_hits, 1);
        assert_eq!(result.candidate_only_target_hits, 1);
        assert_eq!(result.baseline_only_target_hits, 0);
        assert_eq!(result.neither_target_hit, 0);
        assert_eq!(result.both_hit_costed_pairs, 1);
        assert_eq!(result.median_time_speedup, Some(2.0));
        assert_eq!(result.median_energy_speedup, Some(2.0));
        assert_eq!(result.pairs[1].time_speedup_baseline_over_candidate, None);
    }

    #[test]
    fn missing_stage_cost_is_unknown_not_a_free_speedup() {
        let baseline_costs = costs(100.0);
        let mut candidate_costs = costs(50.0);
        candidate_costs.acquisition_ms = None;
        assert_eq!(candidate_costs.all_in_total_ms(), None);

        let baseline = report(
            "baseline",
            "classical",
            vec![verified_trial("b", "pair", 4, 5, baseline_costs, EnergyRecord::not_measured())],
        );
        let candidate = report(
            "candidate",
            "trajectory",
            vec![verified_trial("c", "pair", 4, 5, candidate_costs, EnergyRecord::not_measured())],
        );

        let result = compare_matched_arms(&baseline, &candidate, 0.0).unwrap();
        assert_eq!(result.both_target_hits, 1);
        assert_eq!(result.both_hit_costed_pairs, 0);
        assert_eq!(result.candidate_uncosted_target_hits, 1);
        assert_eq!(result.median_time_speedup, None);
        assert_eq!(result.pairs[0].candidate_all_in_ms, None);
    }

    #[test]
    fn unverified_candidate_above_target_is_not_a_target_hit() {
        let baseline = report(
            "baseline",
            "classical",
            vec![verified_trial("b", "pair", 1, 5, costs(100.0), EnergyRecord::not_measured())],
        );
        let candidate = report(
            "candidate",
            "trajectory",
            vec![MatchedTrialRecord {
                trial_id: "c".into(),
                pair_id: "pair".into(),
                seed: 1,
                termination: TrialTermination::Completed,
                candidate_size: Some(99),
                verification: CandidateVerification::Unverified,
                conflict_count: None,
                costs: costs(1.0),
                energy: EnergyRecord::not_measured(),
            }],
        );

        let result = compare_matched_arms(&baseline, &candidate, 0.0).unwrap();
        assert_eq!(result.baseline_target_hits, 1);
        assert_eq!(result.candidate_target_hits, 0);
        assert_eq!(result.baseline_only_target_hits, 1);
        assert_eq!(result.pairs[0].time_speedup_baseline_over_candidate, None);
    }

    #[test]
    fn rejects_energy_value_marked_not_measured() {
        let baseline = report(
            "baseline",
            "classical",
            vec![verified_trial(
                "b",
                "pair",
                1,
                5,
                costs(1.0),
                measured_energy(1.0, EnergyScope::NotMeasured),
            )],
        );
        assert!(matches!(
            baseline.validate(),
            Err(MatchedEvaluationError::EnergyValueWithNotMeasuredScope)
        ));
    }

    #[test]
    fn refuses_energy_comparison_across_different_boundaries() {
        let baseline = report(
            "baseline",
            "classical",
            vec![verified_trial("b", "pair", 1, 5, costs(100.0), measured_energy(2.0, EnergyScope::DeviceOnly))],
        );
        let candidate = report(
            "candidate",
            "trajectory",
            vec![verified_trial("c", "pair", 1, 5, costs(100.0), measured_energy(20.0, EnergyScope::FullSystem))],
        );
        let result = compare_matched_arms(&baseline, &candidate, 0.0).unwrap();
        assert_eq!(result.mismatched_energy_scope_pairs, 1);
        assert_eq!(result.median_energy_speedup, None);
        assert_eq!(result.pairs[0].energy_speedup_baseline_over_candidate, None);
    }

    #[test]
    fn rejects_mismatched_instance_target_budget_verifier_or_pair_ids() {
        let baseline = report(
            "baseline",
            "classical",
            vec![verified_trial("b", "pair-1", 1, 5, costs(100.0), EnergyRecord::not_measured())],
        );

        let mut candidate = report(
            "candidate",
            "trajectory",
            vec![verified_trial("c", "pair-1", 1, 5, costs(50.0), EnergyRecord::not_measured())],
        );
        candidate.instance_sha256 = "9".repeat(64);
        assert!(matches!(compare_matched_arms(&baseline, &candidate, 0.0), Err(MatchedEvaluationError::InstanceMismatch)));

        candidate.instance_sha256 = baseline.instance_sha256.clone();
        candidate.target_size = 6;
        assert!(matches!(compare_matched_arms(&baseline, &candidate, 0.0), Err(MatchedEvaluationError::TargetMismatch)));

        candidate.target_size = 5;
        candidate.per_trial_solver_budget_ms = 999.0;
        assert!(matches!(compare_matched_arms(&baseline, &candidate, 0.0), Err(MatchedEvaluationError::SolverBudgetMismatch)));

        candidate.per_trial_solver_budget_ms = baseline.per_trial_solver_budget_ms;
        candidate.verifier_sha256 = "8".repeat(64);
        assert!(matches!(compare_matched_arms(&baseline, &candidate, 0.0), Err(MatchedEvaluationError::VerifierMismatch)));

        candidate.verifier_sha256 = baseline.verifier_sha256.clone();
        candidate.trials[0].pair_id = "other-pair".into();
        assert!(matches!(compare_matched_arms(&baseline, &candidate, 0.0), Err(MatchedEvaluationError::PairSetMismatch)));
    }

    #[test]
    fn rejects_inconsistent_verified_candidate_and_zero_cost_records() {
        let mut trial = verified_trial("b", "pair", 1, 5, costs(1.0), EnergyRecord::not_measured());
        trial.conflict_count = Some(1);
        let bad = report("baseline", "classical", vec![trial]);
        assert!(matches!(
            bad.validate(),
            Err(MatchedEvaluationError::InconsistentCandidateRecord(_))
        ));

        let zero = StageCostsMs {
            acquisition_ms: Some(0.0),
            preprocessing_ms: Some(0.0),
            feature_extraction_ms: Some(0.0),
            transfer_ms: Some(0.0),
            compile_init_amortized_ms: Some(0.0),
            solve_ms: Some(0.0),
            repair_ms: Some(0.0),
            verification_ms: Some(0.0),
            receipt_ms: Some(0.0),
        };
        let bad_zero = report(
            "baseline",
            "classical",
            vec![verified_trial("b", "pair", 1, 5, zero, EnergyRecord::not_measured())],
        );
        assert!(matches!(
            bad_zero.validate(),
            Err(MatchedEvaluationError::InvalidAllInCost { .. })
        ));
    }

    #[test]
    fn rejects_different_seed_for_a_paired_run_and_bad_tie_tolerance() {
        let baseline = report(
            "baseline",
            "classical",
            vec![verified_trial("b", "pair", 1, 5, costs(100.0), EnergyRecord::not_measured())],
        );
        let candidate = report(
            "candidate",
            "trajectory",
            vec![verified_trial("c", "pair", 2, 5, costs(50.0), EnergyRecord::not_measured())],
        );
        assert!(matches!(
            compare_matched_arms(&baseline, &candidate, 0.0),
            Err(MatchedEvaluationError::PairSeedMismatch(_))
        ));
        assert!(matches!(
            compare_matched_arms(&baseline, &candidate, f64::NAN),
            Err(MatchedEvaluationError::InvalidTieTolerance)
        ));
    }
}
