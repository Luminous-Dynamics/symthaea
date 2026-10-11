use super::*;

/// Stable failure reasons for an independent holdout assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutonomousHoldoutFailure {
    InvalidInput,
    IdenticalSeeds,
    ZeroTrajectories,
    TrajectoryLimitExceeded,
    IncompleteCoverage,
    ThresholdExceeded,
}

/// Generalization evidence from fresh initial conditions. Passing this validator
/// is not symbolic equivalence, formal proof, or validation against reality.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AutonomousHoldoutValidation {
    pub discovery_seed: u64,
    pub holdout_seed: u64,
    pub requested_trajectories: usize,
    pub trajectories_checked: usize,
    pub max_lie_derivative_variance: f64,
    pub median_lie_derivative_variance: f64,
    pub threshold: f64,
    pub passed: bool,
    pub failure: Option<AutonomousHoldoutFailure>,
}

const MAX_AUTONOMOUS_HOLDOUT_TRAJECTORIES: usize = 4096;
const MAX_AUTONOMOUS_STEPS_PER_TRAJECTORY: f64 = 100_000.0;
const MAX_AUTONOMOUS_TOTAL_HOLDOUT_STEPS: usize = 5_000_000;
const MAX_AUTONOMOUS_STATE_DIMENSION: usize = 4096;
const MAX_AUTONOMOUS_STATE_UPDATES_PER_TRAJECTORY: f64 = 10_000_000.0;
const MAX_AUTONOMOUS_TOTAL_STATE_UPDATES: f64 = 50_000_000.0;

fn failed_holdout(
    discovery_seed: u64,
    holdout_seed: u64,
    requested_trajectories: usize,
    threshold: f64,
    trajectories_checked: usize,
    max_variance: f64,
    failure: AutonomousHoldoutFailure,
) -> AutonomousHoldoutValidation {
    AutonomousHoldoutValidation {
        discovery_seed,
        holdout_seed,
        requested_trajectories,
        trajectories_checked,
        max_lie_derivative_variance: max_variance,
        median_lie_derivative_variance: f64::MAX,
        threshold,
        passed: false,
        failure: Some(failure),
    }
}

fn checked_rk4_trajectory(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial: &[f64],
    t_end: f64,
    dt: f64,
) -> Option<Vec<Vec<f64>>> {
    let dimension = initial.len();
    if dimension == 0 || !t_end.is_finite() || t_end <= 0.0 || !dt.is_finite() || dt <= 0.0 {
        return None;
    }
    let estimated_steps = (t_end / dt).ceil();
    let estimated_state_updates = estimated_steps * dimension as f64;
    if !estimated_steps.is_finite()
        || estimated_steps > MAX_AUTONOMOUS_STEPS_PER_TRAJECTORY
        || dimension > MAX_AUTONOMOUS_STATE_DIMENSION
        || estimated_state_updates > MAX_AUTONOMOUS_STATE_UPDATES_PER_TRAJECTORY
    {
        return None;
    }
    let max_steps = estimated_steps as usize;
    let mut t = 0.0_f64;
    let mut state = initial.to_vec();
    let mut states = Vec::with_capacity(max_steps.saturating_add(1));
    states.push(state.clone());

    let derivative = |values: &[f64], time: f64| -> Option<Vec<f64>> {
        if values.len() != dimension || values.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let output = rhs(values, time);
        if output.len() != dimension || output.iter().any(|value| !value.is_finite()) {
            return None;
        }
        Some(output)
    };

    for _ in 0..max_steps {
        if t >= t_end {
            break;
        }
        let h = dt.min(t_end - t);
        if !h.is_finite() || h <= 0.0 || t + h <= t {
            return None;
        }
        let k1 = derivative(&state, t)?;
        let y2: Vec<f64> = (0..dimension).map(|i| state[i] + h * 0.5 * k1[i]).collect();
        let k2 = derivative(&y2, t + 0.5 * h)?;
        let y3: Vec<f64> = (0..dimension).map(|i| state[i] + h * 0.5 * k2[i]).collect();
        let k3 = derivative(&y3, t + 0.5 * h)?;
        let y4: Vec<f64> = (0..dimension).map(|i| state[i] + h * k3[i]).collect();
        let k4 = derivative(&y4, t + h)?;

        for i in 0..dimension {
            state[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        if state.iter().any(|value| !value.is_finite()) {
            return None;
        }
        t += h;
        states.push(state.clone());
    }

    if t < t_end {
        return None;
    }
    Some(states)
}

/// Assess a candidate on independently seeded, perturbed initial conditions.
///
/// The discovery seed is read from the exact search config rather than accepting
/// a second seed argument. Identical seeds, invalid inputs, insufficient coverage,
/// or invalid scores fail closed. A pass denotes numerical holdout generalization
/// only; it is not symbolic or formal proof.
pub fn validate_invariant_holdout(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    candidate: &Expr,
    initial_state: &[f64],
    var_names: &[&str],
    search_config: &RegressorConfig,
    t_max: f64,
    dt: f64,
    holdout_trajectories: usize,
    holdout_seed: u64,
    threshold: f64,
) -> AutonomousHoldoutValidation {
    let discovery_seed = search_config.seed;
    if discovery_seed == holdout_seed {
        return failed_holdout(
            discovery_seed,
            holdout_seed,
            holdout_trajectories,
            threshold,
            0,
            f64::MAX,
            AutonomousHoldoutFailure::IdenticalSeeds,
        );
    }
    if initial_state.is_empty()
        || initial_state.len() != var_names.len()
        || var_names.is_empty()
        || var_names.iter().any(|name| name.trim().is_empty())
        || var_names.iter().collect::<std::collections::HashSet<_>>().len() != var_names.len()
        || initial_state.iter().any(|value| !value.is_finite())
        || !t_max.is_finite()
        || t_max <= 0.0
        || !dt.is_finite()
        || dt <= 0.0
        || !(t_max / dt).is_finite()
        || !threshold.is_finite()
        || threshold < 0.0
    {
        return failed_holdout(
            discovery_seed,
            holdout_seed,
            holdout_trajectories,
            threshold,
            0,
            f64::MAX,
            AutonomousHoldoutFailure::InvalidInput,
        );
    }
    if holdout_trajectories == 0 {
        return failed_holdout(
            discovery_seed,
            holdout_seed,
            holdout_trajectories,
            threshold,
            0,
            f64::MAX,
            AutonomousHoldoutFailure::ZeroTrajectories,
        );
    }
    let estimated_steps = (t_max / dt).ceil();
    let state_updates_per_trajectory = estimated_steps * initial_state.len() as f64;
    let total_state_updates = state_updates_per_trajectory * holdout_trajectories as f64;
    let total_steps = estimated_steps * holdout_trajectories as f64;
    if holdout_trajectories > MAX_AUTONOMOUS_HOLDOUT_TRAJECTORIES
        || initial_state.len() > MAX_AUTONOMOUS_STATE_DIMENSION
        || estimated_steps > MAX_AUTONOMOUS_STEPS_PER_TRAJECTORY
        || state_updates_per_trajectory > MAX_AUTONOMOUS_STATE_UPDATES_PER_TRAJECTORY
        || total_steps > MAX_AUTONOMOUS_TOTAL_HOLDOUT_STEPS as f64
        || total_state_updates > MAX_AUTONOMOUS_TOTAL_STATE_UPDATES
    {
        return failed_holdout(
            discovery_seed,
            holdout_seed,
            holdout_trajectories,
            threshold,
            0,
            f64::MAX,
            AutonomousHoldoutFailure::TrajectoryLimitExceeded,
        );
    }
    let initial_derivative = rhs(initial_state, 0.0);
    if initial_derivative.len() != initial_state.len()
        || initial_derivative.iter().any(|value| !value.is_finite())
    {
        return failed_holdout(
            discovery_seed,
            holdout_seed,
            holdout_trajectories,
            threshold,
            0,
            f64::MAX,
            AutonomousHoldoutFailure::InvalidInput,
        );
    }

    let sample_orbit = |initial: &[f64]| -> Option<Vec<Vec<f64>>> {
        let states = checked_rk4_trajectory(rhs, initial, t_max, dt)?;
        let n_samples = 200.min(states.len());
        if n_samples < 20 {
            return None;
        }
        let step = states.len() / n_samples.max(1);
        let samples: Vec<Vec<f64>> = states
            .iter()
            .step_by(step.max(1))
            .take(n_samples)
            .cloned()
            .collect();
        if samples.len() < 20
            || samples.iter().any(|state| {
                state.len() != var_names.len() || state.iter().any(|value| !value.is_finite())
            })
        {
            return None;
        }
        Some(samples)
    };

    let mut prng = holdout_seed;
    let mut variances = Vec::with_capacity(holdout_trajectories);
    for _ in 0..holdout_trajectories {
        let mut initial = initial_state.to_vec();
        for value in &mut initial {
            prng = prng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (prng as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let scale = value.abs().max(0.05);
            *value += u * 0.25 * scale;
        }
        let Some(orbit) = sample_orbit(&initial) else {
            continue;
        };
        let variance = lie_derivative_variance(candidate, rhs, &orbit, var_names);
        // The fitness helper uses f64::MAX as an invalid-score sentinel.
        if variance.is_finite() && variance < f64::MAX {
            variances.push(variance);
        }
    }

    let checked = variances.len();
    let max_variance = variances.iter().copied().fold(0.0_f64, f64::max);
    if checked != holdout_trajectories {
        return failed_holdout(
            discovery_seed,
            holdout_seed,
            holdout_trajectories,
            threshold,
            checked,
            if checked == 0 { f64::MAX } else { max_variance },
            AutonomousHoldoutFailure::IncompleteCoverage,
        );
    }

    variances.sort_by(f64::total_cmp);
    let mid = variances.len() / 2;
    let median = if variances.len() % 2 == 0 {
        variances[mid - 1] / 2.0 + variances[mid] / 2.0
    } else {
        variances[mid]
    };
    let passed = max_variance <= threshold;
    AutonomousHoldoutValidation {
        discovery_seed,
        holdout_seed,
        requested_trajectories: holdout_trajectories,
        trajectories_checked: checked,
        max_lie_derivative_variance: max_variance,
        median_lie_derivative_variance: median,
        threshold,
        passed,
        failure: if passed {
            None
        } else {
            Some(AutonomousHoldoutFailure::ThresholdExceeded)
        },
    }
}

/// Receipt-construction failures. These are provenance contract violations, not
/// scientific verdicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutonomousDiscoveryReceiptError {
    InvalidSourceRevision,
    EmptySystemId,
    EmptySystemDefinition,
    EmptyCandidate,
    InvalidVariables,
    InvalidInitialState,
    InvalidTiming,
    IdenticalSeeds,
    HoldoutSeedMismatch,
    InvalidHoldoutRecord,
    ColdRunContainsPriors,
    ComposedCandidateNeedsParentReceipts,
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hash_field(hasher, value.as_bytes());
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn hash_usize(hasher: &mut blake3::Hasher, value: usize) {
    hash_u64(hasher, value as u64);
}

fn hash_f64(hasher: &mut blake3::Hasher, value: f64) {
    hash_u64(hasher, value.to_bits());
}

fn sanitized_config(config: &RegressorConfig, mode: AutonomousDiscoveryMode) -> RegressorConfig {
    let mut effective = config.clone();
    if mode == AutonomousDiscoveryMode::Cold {
        effective.disable_macro_seeds = true;
        effective.prior_composition_rate = 0.0;
        effective.prior_fragment_bonus = 1.0;
        effective.orthogonality_penalty = 1.0;
        effective.known_invariants.clear();
    }
    effective
}

fn search_config_digest(config: &RegressorConfig, extra_seed_templates: &[Expr]) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_string(&mut hasher, "symthaea-autonomous-search-config-v1");
    hash_usize(&mut hasher, config.population_size);
    hash_usize(&mut hasher, config.generations);
    hash_usize(&mut hasher, config.max_depth);
    hash_usize(&mut hasher, config.max_complexity);
    hash_f64(&mut hasher, config.lambda);
    hash_usize(&mut hasher, config.tournament_size);
    hash_f64(&mut hasher, config.mutation_rate);
    hash_u64(&mut hasher, config.seed);
    hasher.update(&[u8::from(config.disable_macro_seeds)]);
    hasher.update(&[u8::from(config.exclude_trig)]);
    hash_usize(&mut hasher, config.diverse_trajectory_count);
    hash_f64(&mut hasher, config.prior_composition_rate);
    hash_f64(&mut hasher, config.prior_fragment_bonus);
    hash_f64(&mut hasher, config.orthogonality_penalty);
    hash_f64(&mut hasher, config.orthogonality_threshold);
    hasher.update(&[u8::from(config.use_lie_fitness)]);
    hash_usize(&mut hasher, config.known_invariants.len());
    for invariant in &config.known_invariants {
        hash_string(&mut hasher, &format!("{}", invariant));
    }
    hash_usize(&mut hasher, extra_seed_templates.len());
    for template in extra_seed_templates {
        hash_string(&mut hasher, &format!("{}", template));
    }
    hasher.finalize().to_hex().to_string()
}

/// Identity receipt for one candidate and one independent holdout record.
/// It binds declared inputs and outputs; it does not prove execution occurred
/// honestly, certify the ODE implementation, or establish physical truth.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AutonomousDiscoveryReceiptV1 {
    pub schema_version: u32,
    pub source_revision: String,
    pub system_id: String,
    /// BLAKE3 of a canonical system definition, including RHS equations/constants.
    pub system_definition_blake3: String,
    pub discovery_mode: AutonomousDiscoveryMode,
    pub discovery_seed: u64,
    pub holdout_seed: u64,
    pub search_config_blake3: String,
    pub candidate_formula: String,
    pub candidate_blake3: String,
    pub variable_names: Vec<String>,
    pub initial_state: Vec<f64>,
    pub t_max: f64,
    pub dt: f64,
    pub requested_trajectories: usize,
    pub trajectories_checked: usize,
    pub max_lie_derivative_variance: f64,
    pub median_lie_derivative_variance: f64,
    pub threshold: f64,
    pub holdout_passed: bool,
    pub holdout_failure: Option<AutonomousHoldoutFailure>,
    pub receipt_blake3: String,
}

/// Construct a deterministic, domain-separated receipt.
///
/// Cold mode rejects explicit templates and hashes the sanitized config. Composed
/// candidates are rejected because a sound multi-parent lineage requires a
/// separate parent-receipt schema rather than a false single-run identity.
pub fn build_autonomous_discovery_receipt_v1(
    source_revision: &str,
    system_id: &str,
    system_definition_canonical: &str,
    config: &RegressorConfig,
    extra_seed_templates: &[Expr],
    candidate: &AutonomousInvariant,
    var_names: &[&str],
    initial_state: &[f64],
    t_max: f64,
    dt: f64,
    holdout: &AutonomousHoldoutValidation,
) -> Result<AutonomousDiscoveryReceiptV1, AutonomousDiscoveryReceiptError> {
    let source_revision = source_revision.to_ascii_lowercase();
    if !matches!(source_revision.len(), 40 | 64)
        || !source_revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(AutonomousDiscoveryReceiptError::InvalidSourceRevision);
    }
    if system_id.trim().is_empty() {
        return Err(AutonomousDiscoveryReceiptError::EmptySystemId);
    }
    if system_definition_canonical.trim().is_empty() {
        return Err(AutonomousDiscoveryReceiptError::EmptySystemDefinition);
    }
    // Hash the exact canonical UTF-8 bytes here instead of trusting a supplied
    // digest string. The caller must retain this definition as a separate artifact.
    let system_definition_blake3 =
        blake3::hash(system_definition_canonical.as_bytes()).to_hex().to_string();
    if candidate.formula_str.trim().is_empty()
        || candidate.formula_str != format!("{}", candidate.formula)
    {
        return Err(AutonomousDiscoveryReceiptError::EmptyCandidate);
    }
    if var_names.is_empty()
        || var_names.iter().any(|name| name.trim().is_empty())
        || var_names.iter().collect::<std::collections::HashSet<_>>().len() != var_names.len()
    {
        return Err(AutonomousDiscoveryReceiptError::InvalidVariables);
    }
    if initial_state.len() != var_names.len()
        || initial_state.iter().any(|value| !value.is_finite())
    {
        return Err(AutonomousDiscoveryReceiptError::InvalidInitialState);
    }
    if !t_max.is_finite() || t_max <= 0.0 || !dt.is_finite() || dt <= 0.0 {
        return Err(AutonomousDiscoveryReceiptError::InvalidTiming);
    }
    if holdout.discovery_seed != config.seed {
        return Err(AutonomousDiscoveryReceiptError::HoldoutSeedMismatch);
    }
    if candidate.discovery_mode == AutonomousDiscoveryMode::Composed {
        return Err(AutonomousDiscoveryReceiptError::ComposedCandidateNeedsParentReceipts);
    }
    if candidate.discovery_mode == AutonomousDiscoveryMode::Cold
        && !extra_seed_templates.is_empty()
    {
        return Err(AutonomousDiscoveryReceiptError::ColdRunContainsPriors);
    }
    let effective = sanitized_config(config, candidate.discovery_mode);
    let holdout_valid = holdout_record_is_consistent(holdout, t_max, dt, initial_state.len());
    if !holdout_valid {
        return Err(AutonomousDiscoveryReceiptError::InvalidHoldoutRecord);
    }

    let formula = candidate.formula_str.clone();
    let candidate_blake3 = blake3::hash(formula.as_bytes()).to_hex().to_string();
    let mut receipt = AutonomousDiscoveryReceiptV1 {
        schema_version: 1,
        source_revision,
        system_id: system_id.trim().to_string(),
        system_definition_blake3,
        discovery_mode: candidate.discovery_mode,
        discovery_seed: config.seed,
        holdout_seed: holdout.holdout_seed,
        search_config_blake3: search_config_digest(
            &effective,
            if candidate.discovery_mode == AutonomousDiscoveryMode::Cold {
                &[]
            } else {
                extra_seed_templates
            },
        ),
        candidate_formula: formula,
        candidate_blake3,
        variable_names: var_names.iter().map(|name| (*name).to_string()).collect(),
        initial_state: initial_state.to_vec(),
        t_max,
        dt,
        requested_trajectories: holdout.requested_trajectories,
        trajectories_checked: holdout.trajectories_checked,
        max_lie_derivative_variance: holdout.max_lie_derivative_variance,
        median_lie_derivative_variance: holdout.median_lie_derivative_variance,
        threshold: holdout.threshold,
        holdout_passed: holdout.passed,
        holdout_failure: holdout.failure,
        receipt_blake3: String::new(),
    };
    receipt.receipt_blake3 = compute_receipt_digest(&receipt);
    Ok(receipt)
}

/// Verify content integrity only. This is not a signature or execution attestation.
pub fn verify_autonomous_discovery_receipt_v1(receipt: &AutonomousDiscoveryReceiptV1) -> bool {
    let seed_pair_consistent = if receipt.discovery_seed == receipt.holdout_seed {
        !receipt.holdout_passed
            && receipt.holdout_failure == Some(AutonomousHoldoutFailure::IdenticalSeeds)
            && receipt.trajectories_checked == 0
    } else {
        receipt.holdout_failure != Some(AutonomousHoldoutFailure::IdenticalSeeds)
    };
    let holdout = AutonomousHoldoutValidation {
        discovery_seed: receipt.discovery_seed,
        holdout_seed: receipt.holdout_seed,
        requested_trajectories: receipt.requested_trajectories,
        trajectories_checked: receipt.trajectories_checked,
        max_lie_derivative_variance: receipt.max_lie_derivative_variance,
        median_lie_derivative_variance: receipt.median_lie_derivative_variance,
        threshold: receipt.threshold,
        passed: receipt.holdout_passed,
        failure: receipt.holdout_failure,
    };
    receipt.schema_version == 1
        && matches!(receipt.source_revision.len(), 40 | 64)
        && receipt.source_revision == receipt.source_revision.to_ascii_lowercase()
        && receipt.source_revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        && !receipt.system_id.trim().is_empty()
        && receipt.system_id == receipt.system_id.trim()
        && receipt.system_definition_blake3.len() == 64
        && receipt.system_definition_blake3 == receipt.system_definition_blake3.to_ascii_lowercase()
        && receipt.system_definition_blake3.bytes().all(|byte| byte.is_ascii_hexdigit())
        && receipt.search_config_blake3.len() == 64
        && receipt.search_config_blake3 == receipt.search_config_blake3.to_ascii_lowercase()
        && receipt.search_config_blake3.bytes().all(|byte| byte.is_ascii_hexdigit())
        && !receipt.candidate_formula.trim().is_empty()
        && !receipt.variable_names.is_empty()
        && receipt.variable_names.iter().all(|name| !name.trim().is_empty())
        && receipt.variable_names.iter().collect::<std::collections::HashSet<_>>().len()
            == receipt.variable_names.len()
        && receipt.initial_state.len() == receipt.variable_names.len()
        && receipt.initial_state.iter().all(|value| value.is_finite())
        && receipt.t_max.is_finite()
        && receipt.t_max > 0.0
        && receipt.dt.is_finite()
        && receipt.dt > 0.0
        && seed_pair_consistent
        && holdout_record_is_consistent(
            &holdout,
            receipt.t_max,
            receipt.dt,
            receipt.initial_state.len(),
        )
        && blake3::hash(receipt.candidate_formula.as_bytes()).to_hex().to_string()
            == receipt.candidate_blake3
        && compute_receipt_digest(receipt) == receipt.receipt_blake3
}

/// A failure must be recorded explicitly and agree with the census and metric
/// fields. In particular, a failed score above threshold cannot have no failure
/// code, and an incomplete census cannot be represented as threshold-only failure.
fn holdout_record_is_consistent(
    holdout: &AutonomousHoldoutValidation,
    t_max: f64,
    dt: f64,
    state_dimension: usize,
) -> bool {
    if !t_max.is_finite()
        || t_max <= 0.0
        || !dt.is_finite()
        || dt <= 0.0
        || !(t_max / dt).is_finite()
        || !holdout.threshold.is_finite()
        || holdout.threshold < 0.0
        || !holdout.max_lie_derivative_variance.is_finite()
        || holdout.max_lie_derivative_variance < 0.0
        || !holdout.median_lie_derivative_variance.is_finite()
        || holdout.median_lie_derivative_variance < 0.0
        || holdout.trajectories_checked > holdout.requested_trajectories
    {
        return false;
    }
    let estimated_steps = (t_max / dt).ceil();
    let state_updates_per_trajectory = estimated_steps * state_dimension as f64;
    let total_steps = estimated_steps * holdout.requested_trajectories as f64;
    let total_state_updates = state_updates_per_trajectory
        * holdout.requested_trajectories as f64;
    let budget_exceeded = holdout.requested_trajectories
        > MAX_AUTONOMOUS_HOLDOUT_TRAJECTORIES
        || state_dimension > MAX_AUTONOMOUS_STATE_DIMENSION
        || estimated_steps > MAX_AUTONOMOUS_STEPS_PER_TRAJECTORY
        || state_updates_per_trajectory > MAX_AUTONOMOUS_STATE_UPDATES_PER_TRAJECTORY
        || total_steps > MAX_AUTONOMOUS_TOTAL_HOLDOUT_STEPS as f64
        || total_state_updates > MAX_AUTONOMOUS_TOTAL_STATE_UPDATES;

    if holdout.passed {
        return !budget_exceeded
            && holdout.failure.is_none()
            && holdout.requested_trajectories > 0
            && holdout.trajectories_checked == holdout.requested_trajectories
            && holdout.max_lie_derivative_variance <= holdout.threshold;
    }
    match holdout.failure {
        Some(AutonomousHoldoutFailure::IdenticalSeeds) => {
            holdout.discovery_seed == holdout.holdout_seed
                && holdout.trajectories_checked == 0
        }
        Some(AutonomousHoldoutFailure::ZeroTrajectories) => {
            holdout.requested_trajectories == 0 && holdout.trajectories_checked == 0
        }
        Some(AutonomousHoldoutFailure::TrajectoryLimitExceeded) => budget_exceeded,
        Some(AutonomousHoldoutFailure::IncompleteCoverage) => {
            !budget_exceeded
                && holdout.requested_trajectories > 0
                && holdout.trajectories_checked < holdout.requested_trajectories
        }
        Some(AutonomousHoldoutFailure::ThresholdExceeded) => {
            !budget_exceeded
                && holdout.requested_trajectories > 0
                && holdout.trajectories_checked == holdout.requested_trajectories
                && holdout.max_lie_derivative_variance > holdout.threshold
        }
        Some(AutonomousHoldoutFailure::InvalidInput) | None => false,
    }
}

fn compute_receipt_digest(receipt: &AutonomousDiscoveryReceiptV1) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_string(&mut hasher, "symthaea-autonomous-discovery-receipt-v1");
    hash_u64(&mut hasher, receipt.schema_version as u64);
    hash_string(&mut hasher, &receipt.source_revision);
    hash_string(&mut hasher, &receipt.system_id);
    hash_string(&mut hasher, &receipt.system_definition_blake3);
    hash_string(&mut hasher, receipt.discovery_mode.as_str());
    hash_u64(&mut hasher, receipt.discovery_seed);
    hash_u64(&mut hasher, receipt.holdout_seed);
    hash_string(&mut hasher, &receipt.search_config_blake3);
    hash_string(&mut hasher, &receipt.candidate_formula);
    hash_string(&mut hasher, &receipt.candidate_blake3);
    hash_usize(&mut hasher, receipt.variable_names.len());
    for name in &receipt.variable_names {
        hash_string(&mut hasher, name);
    }
    hash_usize(&mut hasher, receipt.initial_state.len());
    for value in &receipt.initial_state {
        hash_f64(&mut hasher, *value);
    }
    hash_f64(&mut hasher, receipt.t_max);
    hash_f64(&mut hasher, receipt.dt);
    hash_usize(&mut hasher, receipt.requested_trajectories);
    hash_usize(&mut hasher, receipt.trajectories_checked);
    hash_f64(&mut hasher, receipt.max_lie_derivative_variance);
    hash_f64(&mut hasher, receipt.median_lie_derivative_variance);
    hash_f64(&mut hasher, receipt.threshold);
    hasher.update(&[u8::from(receipt.holdout_passed)]);
    let failure = receipt.holdout_failure
        .map(|value| format!("{:?}", value))
        .unwrap_or_else(|| "none".to_string());
    hash_string(&mut hasher, &failure);
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic_rhs(state: &[f64], _time: f64) -> Vec<f64> {
        vec![state[2], state[3], -state[0], -state[1]]
    }

    fn wrong_dimension_after_start(state: &[f64], time: f64) -> Vec<f64> {
        if time == 0.0 {
            vec![state[2], state[3], -state[0], -state[1]]
        } else {
            Vec::new()
        }
    }

    fn system_definition_canonical() -> &'static str {
        "system-definition-v1\nharmonic-oscillator-v1\nxdot=vx\nydot=vy\nvxdot=-x\nvydot=-y"
    }

    fn harmonic_energy() -> Expr {
        let square = |name: &str| Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var(name.to_string())),
            Box::new(Expr::Const(2.0)),
        );
        Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(0.5)),
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::BinOp(BinOp::Add, Box::new(square("x")), Box::new(square("y")))),
                Box::new(Expr::BinOp(BinOp::Add, Box::new(square("vx")), Box::new(square("vy")))),
            )),
        )
    }

    fn candidate(formula: Expr, mode: AutonomousDiscoveryMode) -> AutonomousInvariant {
        AutonomousInvariant {
            formula_str: format!("{}", formula),
            complexity: formula.complexity(),
            formula,
            variance: 0.0,
            mean_value: 1.0,
            symbolically_proven: false,
            discovery_mode: mode,
        }
    }

    #[test]
    fn identical_seeds_fail_closed() {
        let config = RegressorConfig { seed: 404, ..RegressorConfig::default() };
        let result = validate_invariant_holdout(
            harmonic_rhs, &harmonic_energy(), &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 404, 1e-2,
        );
        assert!(!result.passed);
        assert_eq!(result.failure, Some(AutonomousHoldoutFailure::IdenticalSeeds));
        assert_eq!(result.trajectories_checked, 0);
    }

    #[test]
    fn harmonic_energy_generalizes_to_fresh_initial_conditions() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let result = validate_invariant_holdout(
            harmonic_rhs, &harmonic_energy(), &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 4, 701, 1e-2,
        );
        assert!(result.passed, "unexpected holdout record: {:?}", result);
        assert_eq!(result.trajectories_checked, 4);
        assert_ne!(result.discovery_seed, result.holdout_seed);
    }

    #[test]
    fn malformed_inputs_and_zero_coverage_fail_closed() {
        let config = RegressorConfig { seed: 9, ..RegressorConfig::default() };
        let malformed = validate_invariant_holdout(
            harmonic_rhs, &harmonic_energy(), &[f64::NAN, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 4, 709, 1e-2,
        );
        assert!(!malformed.passed);
        assert_eq!(malformed.failure, Some(AutonomousHoldoutFailure::InvalidInput));

        let zero = validate_invariant_holdout(
            harmonic_rhs, &harmonic_energy(), &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 0, 709, 1e-2,
        );
        assert!(!zero.passed);
        assert_eq!(zero.failure, Some(AutonomousHoldoutFailure::ZeroTrajectories));
    }

    #[test]
    fn receipt_digest_binds_seeds_candidate_and_holdout_metrics() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            harmonic_rhs, &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 701, 1e-2,
        );
        let receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("well-formed receipt should be constructed");
        assert!(verify_autonomous_discovery_receipt_v1(&receipt));
        assert_eq!(receipt.discovery_seed, 13);
        assert_eq!(receipt.holdout_seed, 701);
        assert_eq!(receipt.discovery_mode, AutonomousDiscoveryMode::Cold);
        assert_eq!(
            receipt.system_definition_blake3,
            blake3::hash(system_definition_canonical().as_bytes()).to_hex().to_string()
        );
        let mut tampered = receipt.clone();
        tampered.holdout_seed += 1;
        assert!(!verify_autonomous_discovery_receipt_v1(&tampered));
    }

    #[test]
    fn receipt_rejects_cold_run_with_explicit_priors() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            harmonic_rhs, &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 2, 701, 1e-2,
        );
        let result = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            &config,
            &[Expr::Var("x".into())],
            &candidate, &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        );
        assert_eq!(result.unwrap_err(), AutonomousDiscoveryReceiptError::ColdRunContainsPriors);
    }

    #[test]
    fn composed_candidates_need_parent_receipts() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Composed);
        let holdout = validate_invariant_holdout(
            harmonic_rhs, &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 2, 701, 1e-2,
        );
        let result = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        );
        assert_eq!(
            result.unwrap_err(),
            AutonomousDiscoveryReceiptError::ComposedCandidateNeedsParentReceipts
        );
    }
    
    #[test]
    fn receipt_rejects_failed_holdout_without_failure_code() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let mut holdout = validate_invariant_holdout(
            harmonic_rhs, &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 701, 1e-2,
        );
        holdout.passed = false;
        holdout.failure = None;
        let result = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        );
        assert_eq!(result.unwrap_err(), AutonomousDiscoveryReceiptError::InvalidHoldoutRecord);
    }

    #[test]
    fn receipt_preserves_a_valid_negative_result() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let mut holdout = validate_invariant_holdout(
            harmonic_rhs, &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 701, 1e-2,
        );
        holdout.passed = false;
        holdout.failure = Some(AutonomousHoldoutFailure::ThresholdExceeded);
        holdout.max_lie_derivative_variance = 1.0;
        holdout.median_lie_derivative_variance = 0.5;
        holdout.threshold = 0.01;
        let receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("a fully scored negative result should remain recordable");
        assert!(!receipt.holdout_passed);
        assert_eq!(receipt.holdout_failure, Some(AutonomousHoldoutFailure::ThresholdExceeded));
        assert!(verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn receipt_preserves_identical_seed_failure_without_promoting_it() {
        let config = RegressorConfig { seed: 404, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            harmonic_rhs, &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 404, 1e-2,
        );
        let receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("an identical-seed failure should be archivally recordable");
        assert!(!receipt.holdout_passed);
        assert_eq!(receipt.holdout_failure, Some(AutonomousHoldoutFailure::IdenticalSeeds));
        assert!(verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn verifier_rejects_semantically_invalid_receipt_even_after_redigest() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            harmonic_rhs, &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 701, 1e-2,
        );
        let mut receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("well-formed receipt should be constructed");
        receipt.holdout_passed = false;
        receipt.holdout_failure = None;
        receipt.receipt_blake3 = compute_receipt_digest(&receipt);
        assert!(!verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn changing_system_definition_changes_receipt_identity() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            harmonic_rhs, &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 701, 1e-2,
        );
        let receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("well-formed receipt should be constructed");
        let mut altered = receipt.clone();
        altered.system_definition_blake3 = blake3::hash(b"a different rhs").to_hex().to_string();
        assert!(!verify_autonomous_discovery_receipt_v1(&altered));
    }


    #[test]
    fn wrong_rhs_dimension_on_intermediate_step_fails_closed_without_panicking() {
        let config = RegressorConfig { seed: 23, ..RegressorConfig::default() };
        let result = validate_invariant_holdout(
            wrong_dimension_after_start, &harmonic_energy(), &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 709, 1e-2,
        );
        assert!(!result.passed);
        assert_eq!(result.trajectories_checked, 0);
        assert_eq!(result.failure, Some(AutonomousHoldoutFailure::IncompleteCoverage));
    }

    #[test]
    fn oversized_state_dimension_is_rejected_before_rhs_execution() {
        fn constant_rhs(state: &[f64], _time: f64) -> Vec<f64> {
            vec![0.0; state.len()]
        }

        let config = RegressorConfig { seed: 23, ..RegressorConfig::default() };
        let initial_state = vec![1.0; MAX_AUTONOMOUS_STATE_DIMENSION + 1];
        let names: Vec<String> = (0..initial_state.len())
            .map(|index| format!("x{index}"))
            .collect();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let result = validate_invariant_holdout(
            constant_rhs,
            &Expr::Var("x0".into()),
            &initial_state,
            &name_refs,
            &config,
            1.0,
            0.01,
            1,
            709,
            1e-2,
        );
        assert!(!result.passed);
        assert_eq!(result.trajectories_checked, 0);
        assert_eq!(result.failure, Some(AutonomousHoldoutFailure::TrajectoryLimitExceeded));
    }

    #[test]
    fn oversized_holdout_budget_is_rejected_before_integration() {
        let config = RegressorConfig { seed: 23, ..RegressorConfig::default() };
        let result = validate_invariant_holdout(
            harmonic_rhs, &harmonic_energy(), &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 600.0, 0.01, 100, 709, 1e-2,
        );
        assert!(!result.passed);
        assert_eq!(result.trajectories_checked, 0);
        assert_eq!(result.failure, Some(AutonomousHoldoutFailure::TrajectoryLimitExceeded));
    }

    #[test]
    fn empty_system_definition_is_rejected() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            harmonic_rhs, &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 2, 701, 1e-2,
        );
        let result = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", "", &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        );
        assert_eq!(result.unwrap_err(), AutonomousDiscoveryReceiptError::EmptySystemDefinition);
    }

    #[test]
    fn verifier_rejects_malformed_search_digest_even_after_redigest() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            harmonic_rhs, &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 2, 701, 1e-2,
        );
        let mut receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("well-formed receipt should be constructed");
        receipt.search_config_blake3 = "not-a-digest".to_string();
        receipt.receipt_blake3 = compute_receipt_digest(&receipt);
        assert!(!verify_autonomous_discovery_receipt_v1(&receipt));
    }

}
