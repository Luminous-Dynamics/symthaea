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
    InsufficientGradientInformativeness,
    ThresholdExceeded,
}

/// A single evaluator paired with its declared, versioned system identity.
/// Holdout evaluation always invokes this stored evaluator. This improves
/// identity consistency, but does not prove the declaration equals the code.
#[derive(Clone)]
pub struct AutonomousHoldoutSystemV1 {
    system_id: String,
    canonical_definition: String,
    definition_blake3: String,
    rhs: fn(&[f64], f64) -> Vec<f64>,
}
impl AutonomousHoldoutSystemV1 {
    pub fn new(
        system_id: &str,
        canonical_definition: &str,
        rhs: fn(&[f64], f64) -> Vec<f64>,
    ) -> Result<Self, AutonomousDiscoveryReceiptError> {
        let (system_id, definition_blake3) =
            validate_system_identity(system_id, canonical_definition)?;
        Ok(Self {
            system_id,
            canonical_definition: canonical_definition.to_string(),
            definition_blake3,
            rhs,
        })
    }

    pub fn system_id(&self) -> &str {
        &self.system_id
    }

    pub fn canonical_definition(&self) -> &str {
        &self.canonical_definition
    }

    pub fn definition_blake3(&self) -> &str {
        &self.definition_blake3
    }
}
/// Generalization evidence from fresh initial conditions. Passing this validator
/// is not symbolic equivalence, formal proof, or validation against reality.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AutonomousHoldoutValidation {
    pub discovery_seed: u64,
    pub holdout_seed: u64,
    pub system_id: String,
    pub system_definition_blake3: String,
    pub requested_trajectories: usize,
    pub trajectories_checked: usize,
    /// Work spent integrating holdout orbits, including failed attempts.
    pub attempted_integration_steps: u64,
    pub completed_integration_steps: u64,
    pub integration_rhs_evaluations: u64,
    pub scoring_rhs_evaluations: u64,
    /// Integration plus candidate-score RHS calls.
    pub rhs_evaluations: u64,
    pub state_updates: u64,
    pub max_lie_derivative_variance: f64,
    pub median_lie_derivative_variance: f64,
    /// Minimum gradient-informativeness fraction across successfully scored orbits.
    pub min_gradient_informativeness_fraction: f64,
    /// Shared squared-gradient floor used to compute the preceding metric.
    pub gradient_informativeness_relative_squared_floor: f64,
    /// Required minimum fraction for a passing holdout.
    pub required_gradient_informativeness_fraction: f64,
    pub threshold: f64,
    pub passed: bool,
    pub failure: Option<AutonomousHoldoutFailure>,
    /// Fingerprint of the exact candidate and numerical inputs assessed. This is
    /// a context-binding digest, not an execution attestation or digital signature.
    pub assessment_context_blake3: String,
}

const MAX_AUTONOMOUS_HOLDOUT_TRAJECTORIES: usize = 4096;
const MAX_AUTONOMOUS_STEPS_PER_TRAJECTORY: f64 = 100_000.0;
const MAX_AUTONOMOUS_TOTAL_HOLDOUT_STEPS: usize = 5_000_000;
const MAX_AUTONOMOUS_STATE_DIMENSION: usize = 4096;
const MAX_AUTONOMOUS_STATE_UPDATES_PER_TRAJECTORY: f64 = 10_000_000.0;
const MAX_AUTONOMOUS_TOTAL_STATE_UPDATES: f64 = 50_000_000.0;
/// Require at least half of holdout orbits to pass the shared gradient-information floor.
const MIN_AUTONOMOUS_HOLDOUT_INFORMATIVE_FRACTION: f64 = 0.5;

fn failed_holdout(
    discovery_seed: u64,
    holdout_seed: u64,
    requested_trajectories: usize,
    threshold: f64,
    trajectories_checked: usize,
    max_variance: f64,
    min_gradient_informativeness_fraction: f64,
    failure: AutonomousHoldoutFailure,
) -> AutonomousHoldoutValidation {
    AutonomousHoldoutValidation {
        discovery_seed,
        holdout_seed,
        system_id: String::new(),
        system_definition_blake3: String::new(),
        requested_trajectories,
        trajectories_checked,
        attempted_integration_steps: 0,
        completed_integration_steps: 0,
        integration_rhs_evaluations: 0,
        scoring_rhs_evaluations: 0,
        rhs_evaluations: 0,
        state_updates: 0,
        max_lie_derivative_variance: max_variance,
        median_lie_derivative_variance: f64::MAX,
        min_gradient_informativeness_fraction,
        gradient_informativeness_relative_squared_floor:
            GRADIENT_INFORMATIVENESS_RELATIVE_SQUARED_FLOOR,
        required_gradient_informativeness_fraction:
            MIN_AUTONOMOUS_HOLDOUT_INFORMATIVE_FRACTION,
        threshold,
        passed: false,
        failure: Some(failure),
        assessment_context_blake3: String::new(),
    }
}

fn checked_rk4_trajectory(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    initial: &[f64],
    t_end: f64,
    dt: f64,
) -> (Option<Vec<Vec<f64>>>, super::dynamics::Rk4TrajectoryWorkV1) {
    use super::dynamics::Rk4TrajectoryWorkV1;

    let empty_work = Rk4TrajectoryWorkV1::default();
    let dimension = initial.len();
    if dimension == 0
        || !t_end.is_finite()
        || t_end <= 0.0
        || !dt.is_finite()
        || dt <= 0.0
    {
        return (None, empty_work);
    }
    let estimated_steps = (t_end / dt).ceil();
    let estimated_state_updates = estimated_steps * dimension as f64;
    if !estimated_steps.is_finite()
        || estimated_steps > MAX_AUTONOMOUS_STEPS_PER_TRAJECTORY
        || dimension > MAX_AUTONOMOUS_STATE_DIMENSION
        || estimated_state_updates > MAX_AUTONOMOUS_STATE_UPDATES_PER_TRAJECTORY
    {
        return (None, empty_work);
    }

    let (_times, states, work) =
        super::dynamics::rk4_trajectory_with_work(rhs, initial, t_end, dt);
    if !work.completed {
        return (None, work);
    }
    (Some(states), work)
}

/// Assess a candidate on independently seeded, perturbed initial conditions.
///
/// The discovery seed is read from the exact search config rather than accepting
/// a second seed argument. Identical seeds, invalid inputs, insufficient coverage,
/// or invalid scores fail closed. A pass denotes numerical holdout generalization
/// only; it is not symbolic or formal proof.
fn validate_invariant_holdout_inner(
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
            0.0,
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
            0.0,
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
            0.0,
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
            0.0,
            AutonomousHoldoutFailure::TrajectoryLimitExceeded,
        );
    }
    let sample_orbit = |initial: &[f64]| -> (Option<Vec<Vec<f64>>>, super::dynamics::Rk4TrajectoryWorkV1) {
        let (states, work) = checked_rk4_trajectory(rhs, initial, t_max, dt);
        let Some(states) = states else {
            return (None, work);
        };
        let n_samples = 200.min(states.len());
        if n_samples < 20 {
            return (None, work);
        }
        let samples = super::autonomous::sample_trajectory_evenly(&states, n_samples);
        if samples.len() < 20
            || samples.iter().any(|state| {
                state.len() != var_names.len() || state.iter().any(|value| !value.is_finite())
            })
        {
            return (None, work);
        }
        (Some(samples), work)
    };

    let mut prng = holdout_seed;
    let mut variances = Vec::with_capacity(holdout_trajectories);
    let mut gradient_fractions = Vec::with_capacity(holdout_trajectories);
    let mut attempted_integration_steps = 0u64;
    let mut completed_integration_steps = 0u64;
    let mut integration_rhs_evaluations = 0u64;
    let mut scoring_rhs_evaluations = 0u64;
    let mut rhs_evaluations = 0u64;
    let mut state_updates = 0u64;
    let mut invalid_rhs_at_start = false;
    for _ in 0..holdout_trajectories {
        let mut initial = initial_state.to_vec();
        for value in &mut initial {
            prng = prng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (prng as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let scale = value.abs().max(0.05);
            *value += u * 0.25 * scale;
        }
        let (orbit_result, work) = sample_orbit(&initial);
        attempted_integration_steps = attempted_integration_steps.saturating_add(work.attempted_steps);
        completed_integration_steps = completed_integration_steps.saturating_add(work.completed_steps);
        integration_rhs_evaluations =
            integration_rhs_evaluations.saturating_add(work.rhs_evaluations);
        rhs_evaluations = rhs_evaluations.saturating_add(work.rhs_evaluations);
        state_updates = state_updates.saturating_add(work.state_updates);
        if orbit_result.is_none()
            && work.attempted_steps == 1
            && work.completed_steps == 0
            && work.rhs_evaluations == 1
        {
            invalid_rhs_at_start = true;
        }
        let Some(orbit) = orbit_result else {
            continue;
        };
        let informativeness = gradient_informativeness_fraction(candidate, &orbit, var_names);
        let (variance, score_rhs_calls) =
            lie_derivative_variance_with_work(candidate, rhs, &orbit, var_names);
        scoring_rhs_evaluations =
            scoring_rhs_evaluations.saturating_add(score_rhs_calls);
        rhs_evaluations = rhs_evaluations.saturating_add(score_rhs_calls);
        // Invalid scores or invalid informativeness results do not count as a
        // successfully scored holdout trajectory.
        if variance.is_finite()
            && variance < f64::MAX
            && informativeness.is_finite()
            && (0.0..=1.0).contains(&informativeness)
        {
            variances.push(variance);
            gradient_fractions.push(informativeness);
        }
    }

    let checked = variances.len();
    let max_variance = variances.iter().copied().fold(0.0_f64, f64::max);
    let min_gradient_informativeness_fraction =
        gradient_fractions.iter().copied().fold(1.0_f64, f64::min);
    let min_gradient_informativeness_fraction = if gradient_fractions.is_empty() {
        0.0
    } else {
        min_gradient_informativeness_fraction
    };
    if checked != holdout_trajectories {
        let failure = if checked == 0 && invalid_rhs_at_start {
            AutonomousHoldoutFailure::InvalidInput
        } else {
            AutonomousHoldoutFailure::IncompleteCoverage
        };
        let mut result = failed_holdout(
            discovery_seed,
            holdout_seed,
            holdout_trajectories,
            threshold,
            checked,
            if checked == 0 { f64::MAX } else { max_variance },
            min_gradient_informativeness_fraction,
            failure,
        );
        result.attempted_integration_steps = attempted_integration_steps;
        result.completed_integration_steps = completed_integration_steps;
        result.integration_rhs_evaluations = integration_rhs_evaluations;
        result.scoring_rhs_evaluations = scoring_rhs_evaluations;
        result.rhs_evaluations = rhs_evaluations;
        result.state_updates = state_updates;
        return result;
    }

    variances.sort_by(f64::total_cmp);
    let mid = variances.len() / 2;
    let median = if variances.len() % 2 == 0 {
        variances[mid - 1] / 2.0 + variances[mid] / 2.0
    } else {
        variances[mid]
    };
    let gradient_informativeness_passed =
        min_gradient_informativeness_fraction >= MIN_AUTONOMOUS_HOLDOUT_INFORMATIVE_FRACTION;
    let passed = max_variance <= threshold && gradient_informativeness_passed;
    let failure = if !gradient_informativeness_passed {
        Some(AutonomousHoldoutFailure::InsufficientGradientInformativeness)
    } else if !passed {
        Some(AutonomousHoldoutFailure::ThresholdExceeded)
    } else {
        None
    };
    AutonomousHoldoutValidation {
        discovery_seed,
        holdout_seed,
        system_id: String::new(),
        system_definition_blake3: String::new(),
        requested_trajectories: holdout_trajectories,
        trajectories_checked: checked,
        attempted_integration_steps,
        completed_integration_steps,
        integration_rhs_evaluations,
        scoring_rhs_evaluations,
        rhs_evaluations,
        state_updates,
        max_lie_derivative_variance: max_variance,
        median_lie_derivative_variance: median,
        min_gradient_informativeness_fraction,
        gradient_informativeness_relative_squared_floor:
            GRADIENT_INFORMATIVENESS_RELATIVE_SQUARED_FLOOR,
        required_gradient_informativeness_fraction:
            MIN_AUTONOMOUS_HOLDOUT_INFORMATIVE_FRACTION,
        threshold,
        passed,
        failure,
        assessment_context_blake3: String::new(),
    }
}

/// Validate a candidate on holdout trajectories and bind its full evaluation
/// context: candidate formula, variables, initial state, integration settings,
/// seed pair, trajectory budget, threshold, and versioned system identity.
pub fn validate_invariant_holdout(
    system: &AutonomousHoldoutSystemV1,
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
    let mut result = validate_invariant_holdout_inner(
        system.rhs, candidate, initial_state, var_names, search_config, t_max, dt,
        holdout_trajectories, holdout_seed, threshold,
    );
    let variable_names: Vec<String> = var_names.iter().map(|name| (*name).to_string()).collect();
    result.system_id = system.system_id.clone();
    result.system_definition_blake3 = system.definition_blake3.clone();
    result.assessment_context_blake3 = assessment_context_digest(
        &format!("{}", candidate), &variable_names, initial_state, search_config.seed,
        holdout_seed, t_max, dt, holdout_trajectories, threshold,
        GRADIENT_INFORMATIVENESS_RELATIVE_SQUARED_FLOOR,
        MIN_AUTONOMOUS_HOLDOUT_INFORMATIVE_FRACTION,
        &system.system_id, &system.definition_blake3,
    );
    result
}

/// Receipt-construction failures. These are provenance contract violations, not
/// scientific verdicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutonomousDiscoveryReceiptError {
    InvalidSourceRevision,
    EmptySystemId,
    InvalidSystemDefinition,
    InvalidSearchConfig,
    EmptyCandidate,
    InvalidVariables,
    InvalidInitialState,
    InvalidTiming,
    IdenticalSeeds,
    HoldoutSeedMismatch,
    InvalidHoldoutRecord,
    UncontrolledRunContainsPriors,
    ComposedCandidateNeedsParentReceipts,
}
fn validate_system_identity(
    system_id: &str,
    canonical_definition: &str,
) -> Result<(String, String), AutonomousDiscoveryReceiptError> {
    let normalized_id = system_id.trim();
    if normalized_id.is_empty() {
        return Err(AutonomousDiscoveryReceiptError::EmptySystemId);
    }
    if !canonical_definition.starts_with("system-definition-v1\n")
        || canonical_definition.contains('\r')
    {
        return Err(AutonomousDiscoveryReceiptError::InvalidSystemDefinition);
    }

    let mut lines = canonical_definition.split('\n');
    let header = lines.next();
    let declared_id = lines.next();
    let has_body = lines.any(|line| !line.trim().is_empty());
    if header != Some("system-definition-v1")
        || declared_id != Some(normalized_id)
        || !has_body
    {
        return Err(AutonomousDiscoveryReceiptError::InvalidSystemDefinition);
    }

    let digest = blake3::hash(canonical_definition.as_bytes()).to_hex().to_string();
    Ok((normalized_id.to_string(), digest))
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

fn assessment_context_digest(
    candidate_formula: &str,
    variable_names: &[String],
    initial_state: &[f64],
    discovery_seed: u64,
    holdout_seed: u64,
    t_max: f64,
    dt: f64,
    requested_trajectories: usize,
    threshold: f64,
    gradient_informativeness_relative_squared_floor: f64,
    required_gradient_informativeness_fraction: f64,
    system_id: &str,
    system_definition_blake3: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_string(&mut hasher, "symthaea-autonomous-holdout-context-v3");
    hash_string(&mut hasher, candidate_formula);
    hash_string(&mut hasher, system_id);
    hash_string(&mut hasher, system_definition_blake3);
    hash_usize(&mut hasher, variable_names.len());
    for name in variable_names {
        hash_string(&mut hasher, name);
    }
    hash_usize(&mut hasher, initial_state.len());
    for value in initial_state {
        hash_f64(&mut hasher, *value);
    }
    hash_u64(&mut hasher, discovery_seed);
    hash_u64(&mut hasher, holdout_seed);
    hash_f64(&mut hasher, t_max);
    hash_f64(&mut hasher, dt);
    hash_usize(&mut hasher, requested_trajectories);
    hash_f64(&mut hasher, threshold);
    hash_f64(&mut hasher, gradient_informativeness_relative_squared_floor);
    hash_f64(&mut hasher, required_gradient_informativeness_fraction);
    hasher.finalize().to_hex().to_string()
}

fn sanitized_config(config: &RegressorConfig, mode: AutonomousDiscoveryMode) -> RegressorConfig {
    let mut effective = config.clone();
    if matches!(mode, AutonomousDiscoveryMode::Cold | AutonomousDiscoveryMode::RandomBaseline) {
        effective.disable_macro_seeds = true;
        effective.prior_composition_rate = 0.0;
        effective.prior_fragment_bonus = 1.0;
        effective.orthogonality_penalty = 1.0;
        effective.known_invariants.clear();
    }
    effective
}

/// Serializable effective search settings carried inside a receipt so an
/// independent verifier can inspect prior isolation instead of trusting only a hash.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AutonomousSearchConfigV1 {
    pub population_size: usize,
    pub generations: usize,
    pub max_depth: usize,
    pub max_complexity: usize,
    pub lambda: f64,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub seed: u64,
    pub disable_macro_seeds: bool,
    pub exclude_trig: bool,
    pub diverse_trajectory_count: usize,
    pub prior_composition_rate: f64,
    pub prior_fragment_bonus: f64,
    pub orthogonality_penalty: f64,
    pub orthogonality_threshold: f64,
    pub known_invariant_formulas: Vec<String>,
    pub use_lie_fitness: bool,
    pub extra_seed_template_formulas: Vec<String>,
}

fn snapshot_search_config(
    config: &RegressorConfig,
    extra_seed_templates: &[Expr],
) -> AutonomousSearchConfigV1 {
    AutonomousSearchConfigV1 {
        population_size: config.population_size,
        generations: config.generations,
        max_depth: config.max_depth,
        max_complexity: config.max_complexity,
        lambda: config.lambda,
        tournament_size: config.tournament_size,
        mutation_rate: config.mutation_rate,
        seed: config.seed,
        disable_macro_seeds: config.disable_macro_seeds,
        exclude_trig: config.exclude_trig,
        diverse_trajectory_count: config.diverse_trajectory_count,
        prior_composition_rate: config.prior_composition_rate,
        prior_fragment_bonus: config.prior_fragment_bonus,
        orthogonality_penalty: config.orthogonality_penalty,
        orthogonality_threshold: config.orthogonality_threshold,
        known_invariant_formulas: config
            .known_invariants
            .iter()
            .map(|invariant| format!("{}", invariant))
            .collect(),
        use_lie_fitness: config.use_lie_fitness,
        extra_seed_template_formulas: extra_seed_templates
            .iter()
            .map(|template| format!("{}", template))
            .collect(),
    }
}

fn search_config_digest(config: &AutonomousSearchConfigV1) -> String {
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
    hash_usize(&mut hasher, config.known_invariant_formulas.len());
    for formula in &config.known_invariant_formulas {
        hash_string(&mut hasher, formula);
    }
    hash_usize(&mut hasher, config.extra_seed_template_formulas.len());
    for formula in &config.extra_seed_template_formulas {
        hash_string(&mut hasher, formula);
    }
    hasher.finalize().to_hex().to_string()
}

fn search_config_is_consistent(
    config: &AutonomousSearchConfigV1,
    mode: AutonomousDiscoveryMode,
    discovery_seed: u64,
) -> bool {
    let scalars_finite = config.lambda.is_finite()
        && config.mutation_rate.is_finite()
        && config.prior_composition_rate.is_finite()
        && config.prior_fragment_bonus.is_finite()
        && config.orthogonality_penalty.is_finite()
        && config.orthogonality_threshold.is_finite();
    if !scalars_finite
        || config.population_size == 0
        || config.generations == 0
        || config.max_depth == 0
        || config.max_complexity == 0
        || config.tournament_size == 0
        || config.diverse_trajectory_count == 0
        || config.seed != discovery_seed
        || config.lambda < 0.0
        || config.orthogonality_penalty < 0.0
        || config.mutation_rate < 0.0
        || config.mutation_rate > 1.0
        || config.prior_composition_rate < 0.0
        || config.prior_composition_rate > 1.0
        || config.prior_fragment_bonus < 0.0
        || config.prior_fragment_bonus > 1.0
        || config.orthogonality_threshold < 0.0
        || config.orthogonality_threshold > 1.0
        || config.known_invariant_formulas.iter().any(|formula| formula.trim().is_empty())
        || config.extra_seed_template_formulas.iter().any(|formula| formula.trim().is_empty())
    {
        return false;
    }
    match mode {
        AutonomousDiscoveryMode::Cold | AutonomousDiscoveryMode::RandomBaseline => {
            config.disable_macro_seeds
                && config.prior_composition_rate == 0.0
                && config.prior_fragment_bonus == 1.0
                && config.orthogonality_penalty == 1.0
                && config.known_invariant_formulas.is_empty()
                && config.extra_seed_template_formulas.is_empty()
        }
        AutonomousDiscoveryMode::Primed => true,
        AutonomousDiscoveryMode::Composed => false,
    }
}

/// Identity receipt for one candidate and one independent holdout record.
/// It binds declared inputs and outputs; it does not prove execution occurred
/// honestly, certify the ODE implementation, or establish physical truth.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AutonomousDiscoveryReceiptV1 {
    pub schema_version: u32,
    pub source_revision: String,
    pub system_id: String,
    /// Exact versioned UTF-8 system declaration whose digest is recorded below.
    pub system_definition_canonical: String,
    /// BLAKE3 of the canonical system definition, including RHS equations/constants.
    pub system_definition_blake3: String,
    pub discovery_mode: AutonomousDiscoveryMode,
    pub discovery_seed: u64,
    pub holdout_seed: u64,
    pub search_config: AutonomousSearchConfigV1,
    pub search_config_blake3: String,
    pub candidate_formula: String,
    pub candidate_blake3: String,
    pub assessment_context_blake3: String,
    pub variable_names: Vec<String>,
    pub initial_state: Vec<f64>,
    pub t_max: f64,
    pub dt: f64,
    pub requested_trajectories: usize,
    pub trajectories_checked: usize,
    /// Exact holdout integration work, including failed trajectories.
    pub attempted_integration_steps: u64,
    pub completed_integration_steps: u64,
    pub integration_rhs_evaluations: u64,
    pub scoring_rhs_evaluations: u64,
    /// Integration plus candidate-score RHS calls.
    pub rhs_evaluations: u64,
    pub state_updates: u64,
    pub max_lie_derivative_variance: f64,
    pub median_lie_derivative_variance: f64,
    /// Minimum gradient-informativeness fraction across successfully scored orbits.
    pub min_gradient_informativeness_fraction: f64,
    /// Shared squared-gradient floor used to compute the preceding metric.
    pub gradient_informativeness_relative_squared_floor: f64,
    /// Required minimum fraction for a passing holdout.
    pub required_gradient_informativeness_fraction: f64,
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
    let (normalized_system_id, system_definition_blake3) =
        validate_system_identity(system_id, system_definition_canonical)?;
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
    if matches!(
        candidate.discovery_mode,
        AutonomousDiscoveryMode::Cold | AutonomousDiscoveryMode::RandomBaseline
    ) && !extra_seed_templates.is_empty()
    {
        return Err(AutonomousDiscoveryReceiptError::UncontrolledRunContainsPriors);
    }
    let effective = sanitized_config(config, candidate.discovery_mode);
    let effective_templates = if matches!(
        candidate.discovery_mode,
        AutonomousDiscoveryMode::Cold | AutonomousDiscoveryMode::RandomBaseline
    ) {
        &[]
    } else {
        extra_seed_templates
    };
    let search_config = snapshot_search_config(&effective, effective_templates);
    if !search_config_is_consistent(
        &search_config,
        candidate.discovery_mode,
        config.seed,
    ) {
        return Err(AutonomousDiscoveryReceiptError::InvalidSearchConfig);
    }
    let holdout_valid = holdout_record_is_consistent(holdout, t_max, dt, initial_state.len());
    if !holdout_valid {
        return Err(AutonomousDiscoveryReceiptError::InvalidHoldoutRecord);
    }

    let formula = candidate.formula_str.clone();
    let receipt_variable_names: Vec<String> =
        var_names.iter().map(|name| (*name).to_string()).collect();
    if holdout.system_id != normalized_system_id
        || holdout.system_definition_blake3 != system_definition_blake3 {
        return Err(AutonomousDiscoveryReceiptError::InvalidHoldoutRecord);
    }
    let expected_assessment_context = assessment_context_digest(
        &formula, &receipt_variable_names, initial_state, config.seed, holdout.holdout_seed,
        t_max, dt, holdout.requested_trajectories, holdout.threshold,
        holdout.gradient_informativeness_relative_squared_floor,
        holdout.required_gradient_informativeness_fraction,
        &normalized_system_id, &system_definition_blake3,
    );
    if holdout.assessment_context_blake3 != expected_assessment_context {
        return Err(AutonomousDiscoveryReceiptError::InvalidHoldoutRecord);
    }
    let candidate_blake3 = blake3::hash(formula.as_bytes()).to_hex().to_string();
    let mut receipt = AutonomousDiscoveryReceiptV1 {
        schema_version: 1,
        source_revision,
        system_id: normalized_system_id,
        system_definition_canonical: system_definition_canonical.to_string(),
        system_definition_blake3,
        discovery_mode: candidate.discovery_mode,
        discovery_seed: config.seed,
        holdout_seed: holdout.holdout_seed,
        search_config_blake3: search_config_digest(&search_config),
        search_config,
        candidate_formula: formula,
        candidate_blake3,
        assessment_context_blake3: holdout.assessment_context_blake3.clone(),
        variable_names: receipt_variable_names,
        initial_state: initial_state.to_vec(),
        t_max,
        dt,
        requested_trajectories: holdout.requested_trajectories,
        trajectories_checked: holdout.trajectories_checked,
        attempted_integration_steps: holdout.attempted_integration_steps,
        completed_integration_steps: holdout.completed_integration_steps,
        integration_rhs_evaluations: holdout.integration_rhs_evaluations,
        scoring_rhs_evaluations: holdout.scoring_rhs_evaluations,
        rhs_evaluations: holdout.rhs_evaluations,
        state_updates: holdout.state_updates,
        max_lie_derivative_variance: holdout.max_lie_derivative_variance,
        median_lie_derivative_variance: holdout.median_lie_derivative_variance,
        min_gradient_informativeness_fraction: holdout.min_gradient_informativeness_fraction,
        gradient_informativeness_relative_squared_floor:
            holdout.gradient_informativeness_relative_squared_floor,
        required_gradient_informativeness_fraction:
            holdout.required_gradient_informativeness_fraction,
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
    let system_identity_consistent =
        match validate_system_identity(&receipt.system_id, &receipt.system_definition_canonical) {
            Ok((system_id, definition_blake3)) => {
                system_id == receipt.system_id
                    && definition_blake3 == receipt.system_definition_blake3
            }
            Err(_) => false,
        };
    let holdout = AutonomousHoldoutValidation {
        discovery_seed: receipt.discovery_seed,
        holdout_seed: receipt.holdout_seed,
        requested_trajectories: receipt.requested_trajectories,
        trajectories_checked: receipt.trajectories_checked,
        attempted_integration_steps: receipt.attempted_integration_steps,
        completed_integration_steps: receipt.completed_integration_steps,
        integration_rhs_evaluations: receipt.integration_rhs_evaluations,
        scoring_rhs_evaluations: receipt.scoring_rhs_evaluations,
        rhs_evaluations: receipt.rhs_evaluations,
        state_updates: receipt.state_updates,
        max_lie_derivative_variance: receipt.max_lie_derivative_variance,
        median_lie_derivative_variance: receipt.median_lie_derivative_variance,
        min_gradient_informativeness_fraction: receipt.min_gradient_informativeness_fraction,
        gradient_informativeness_relative_squared_floor:
            receipt.gradient_informativeness_relative_squared_floor,
        required_gradient_informativeness_fraction:
            receipt.required_gradient_informativeness_fraction,
        threshold: receipt.threshold,
        passed: receipt.holdout_passed,
        failure: receipt.holdout_failure,
        system_id: receipt.system_id.clone(),
        system_definition_blake3: receipt.system_definition_blake3.clone(),
        assessment_context_blake3: receipt.assessment_context_blake3.clone(),
    };
    receipt.schema_version == 1
        && system_identity_consistent
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
        && search_config_is_consistent(
            &receipt.search_config,
            receipt.discovery_mode,
            receipt.discovery_seed,
        )
        && search_config_digest(&receipt.search_config) == receipt.search_config_blake3
        && receipt.discovery_mode != AutonomousDiscoveryMode::Composed
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
        && receipt.assessment_context_blake3 == assessment_context_digest(
            &receipt.candidate_formula, &receipt.variable_names, &receipt.initial_state,
            receipt.discovery_seed, receipt.holdout_seed, receipt.t_max, receipt.dt,
            receipt.requested_trajectories, receipt.threshold,
            receipt.gradient_informativeness_relative_squared_floor,
            receipt.required_gradient_informativeness_fraction,
            &receipt.system_id, &receipt.system_definition_blake3,
        )
        && compute_receipt_digest(receipt) == receipt.receipt_blake3
}

/// A failure must be recorded explicitly and agree with the census and metric
/// fields. In particular, a failed score above threshold cannot have no failure
/// code, and an incomplete census cannot be represented as threshold-only failure.
fn actual_rk4_step_count(t_end: f64, dt: f64) -> Option<u64> {
    if !t_end.is_finite() || t_end <= 0.0 || !dt.is_finite() || dt <= 0.0 {
        return None;
    }
    let estimated = (t_end / dt).ceil();
    if !estimated.is_finite() || estimated < 1.0 || estimated > MAX_AUTONOMOUS_STEPS_PER_TRAJECTORY {
        return None;
    }
    let max_steps = estimated as usize;
    let mut completed = 0u64;
    for step in 0..max_steps {
        let t = (step as f64 * dt).min(t_end);
        if t >= t_end {
            break;
        }
        let next_t = if step + 1 == max_steps {
            t_end
        } else {
            ((step + 1) as f64 * dt).min(t_end)
        };
        let h = next_t - t;
        if !next_t.is_finite() || !h.is_finite() || h <= 0.0 || t + h <= t {
            return None;
        }
        completed = completed.saturating_add(1);
        if next_t == t_end {
            break;
        }
    }
    (completed > 0).then_some(completed)
}

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
        || (holdout.discovery_seed == holdout.holdout_seed
            && holdout.failure != Some(AutonomousHoldoutFailure::IdenticalSeeds))
        || (holdout.discovery_seed != holdout.holdout_seed
            && holdout.failure == Some(AutonomousHoldoutFailure::IdenticalSeeds))
        || !holdout.threshold.is_finite()
        || holdout.threshold < 0.0
        || !holdout.max_lie_derivative_variance.is_finite()
        || holdout.max_lie_derivative_variance < 0.0
        || !holdout.median_lie_derivative_variance.is_finite()
        || holdout.median_lie_derivative_variance < 0.0
        || !holdout.min_gradient_informativeness_fraction.is_finite()
        || !(0.0..=1.0).contains(&holdout.min_gradient_informativeness_fraction)
        || holdout.gradient_informativeness_relative_squared_floor
            != GRADIENT_INFORMATIVENESS_RELATIVE_SQUARED_FLOOR
        || holdout.required_gradient_informativeness_fraction
            != MIN_AUTONOMOUS_HOLDOUT_INFORMATIVE_FRACTION
        || holdout.trajectories_checked > holdout.requested_trajectories
        || holdout.completed_integration_steps > holdout.attempted_integration_steps
        || holdout.state_updates
            < holdout.completed_integration_steps.saturating_mul(state_dimension as u64)
        || holdout.state_updates
            > holdout.attempted_integration_steps.saturating_mul(state_dimension as u64)
        || holdout.integration_rhs_evaluations < holdout.completed_integration_steps.saturating_mul(4)
        || holdout.integration_rhs_evaluations > holdout.attempted_integration_steps.saturating_mul(4)
        || holdout.rhs_evaluations
            != holdout.integration_rhs_evaluations.saturating_add(holdout.scoring_rhs_evaluations)
        || holdout.scoring_rhs_evaluations
            > (holdout.requested_trajectories as u64).saturating_mul(200)
    {
        return false;
    }
    let estimated_steps = (t_max / dt).ceil();
    // A preflight-rejected over-budget record has executed zero steps. Do not
    // run the bounded step-count loop for a request that should never integrate.
    let actual_steps = if estimated_steps > MAX_AUTONOMOUS_STEPS_PER_TRAJECTORY {
        0
    } else {
        match actual_rk4_step_count(t_max, dt) {
            Some(steps) => steps,
            None => return false,
        }
    };
    let state_updates_per_trajectory = estimated_steps * state_dimension as f64;
    let total_steps = estimated_steps * holdout.requested_trajectories as f64;
    let maximum_actual_steps = actual_steps.saturating_mul(holdout.requested_trajectories as u64);
    if holdout.attempted_integration_steps > maximum_actual_steps {
        return false;
    }
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
            && holdout.attempted_integration_steps == maximum_actual_steps
            && holdout.completed_integration_steps == holdout.attempted_integration_steps
            && holdout.state_updates
                == holdout.completed_integration_steps.saturating_mul(state_dimension as u64)
            && holdout.integration_rhs_evaluations
                == holdout.completed_integration_steps.saturating_mul(4)
            && holdout.max_lie_derivative_variance < f64::MAX
            && holdout.median_lie_derivative_variance <= holdout.max_lie_derivative_variance
            && holdout.max_lie_derivative_variance <= holdout.threshold
            && holdout.min_gradient_informativeness_fraction
                >= MIN_AUTONOMOUS_HOLDOUT_INFORMATIVE_FRACTION;
    }
    match holdout.failure {
        Some(AutonomousHoldoutFailure::IdenticalSeeds) => {
            holdout.discovery_seed == holdout.holdout_seed
                && holdout.trajectories_checked == 0
                && holdout.max_lie_derivative_variance == f64::MAX
                && holdout.median_lie_derivative_variance == f64::MAX
                && holdout.min_gradient_informativeness_fraction == 0.0
        }
        Some(AutonomousHoldoutFailure::ZeroTrajectories) => {
            holdout.requested_trajectories == 0
                && holdout.trajectories_checked == 0
                && holdout.max_lie_derivative_variance == f64::MAX
                && holdout.median_lie_derivative_variance == f64::MAX
                && holdout.min_gradient_informativeness_fraction == 0.0
        }
        Some(AutonomousHoldoutFailure::TrajectoryLimitExceeded) => {
            budget_exceeded
                && holdout.trajectories_checked == 0
                && holdout.max_lie_derivative_variance == f64::MAX
                && holdout.median_lie_derivative_variance == f64::MAX
                && holdout.min_gradient_informativeness_fraction == 0.0
        }
        Some(AutonomousHoldoutFailure::IncompleteCoverage) => {
            !budget_exceeded
                && holdout.requested_trajectories > 0
                && holdout.trajectories_checked < holdout.requested_trajectories
                && holdout.median_lie_derivative_variance == f64::MAX
                && if holdout.trajectories_checked == 0 {
                    holdout.max_lie_derivative_variance == f64::MAX
                        && holdout.min_gradient_informativeness_fraction == 0.0
                } else {
                    holdout.max_lie_derivative_variance < f64::MAX
                }
        }
        Some(AutonomousHoldoutFailure::InsufficientGradientInformativeness) => {
            !budget_exceeded
                && holdout.requested_trajectories > 0
                && holdout.trajectories_checked == holdout.requested_trajectories
                && holdout.max_lie_derivative_variance < f64::MAX
                && holdout.median_lie_derivative_variance
                    <= holdout.max_lie_derivative_variance
                && holdout.min_gradient_informativeness_fraction
                    < MIN_AUTONOMOUS_HOLDOUT_INFORMATIVE_FRACTION
        }
        Some(AutonomousHoldoutFailure::ThresholdExceeded) => {
            !budget_exceeded
                && holdout.requested_trajectories > 0
                && holdout.trajectories_checked == holdout.requested_trajectories
                && holdout.max_lie_derivative_variance < f64::MAX
                && holdout.median_lie_derivative_variance
                    <= holdout.max_lie_derivative_variance
                && holdout.max_lie_derivative_variance > holdout.threshold
                && holdout.min_gradient_informativeness_fraction
                    >= MIN_AUTONOMOUS_HOLDOUT_INFORMATIVE_FRACTION
        }
        Some(AutonomousHoldoutFailure::InvalidInput) => {
            holdout.requested_trajectories > 0
                && holdout.trajectories_checked == 0
                && holdout.max_lie_derivative_variance == f64::MAX
                && holdout.median_lie_derivative_variance == f64::MAX
                && holdout.min_gradient_informativeness_fraction == 0.0
        }
        None => false,
    }
}

fn compute_receipt_digest(receipt: &AutonomousDiscoveryReceiptV1) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_string(&mut hasher, "symthaea-autonomous-discovery-receipt-v1");
    hash_u64(&mut hasher, receipt.schema_version as u64);
    hash_string(&mut hasher, &receipt.source_revision);
    hash_string(&mut hasher, &receipt.system_id);
    hash_string(&mut hasher, &receipt.system_definition_canonical);
    hash_string(&mut hasher, &receipt.system_definition_blake3);
    hash_string(&mut hasher, receipt.discovery_mode.as_str());
    hash_u64(&mut hasher, receipt.discovery_seed);
    hash_u64(&mut hasher, receipt.holdout_seed);
    hash_string(&mut hasher, &receipt.search_config_blake3);
    hash_string(&mut hasher, &receipt.candidate_formula);
    hash_string(&mut hasher, &receipt.candidate_blake3);
    hash_string(&mut hasher, &receipt.assessment_context_blake3);
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
    hash_u64(&mut hasher, receipt.attempted_integration_steps);
    hash_u64(&mut hasher, receipt.completed_integration_steps);
    hash_u64(&mut hasher, receipt.integration_rhs_evaluations);
    hash_u64(&mut hasher, receipt.scoring_rhs_evaluations);
    hash_u64(&mut hasher, receipt.rhs_evaluations);
    hash_u64(&mut hasher, receipt.state_updates);
    hash_f64(&mut hasher, receipt.max_lie_derivative_variance);
    hash_f64(&mut hasher, receipt.median_lie_derivative_variance);
    hash_f64(&mut hasher, receipt.min_gradient_informativeness_fraction);
    hash_f64(&mut hasher, receipt.gradient_informativeness_relative_squared_floor);
    hash_f64(&mut hasher, receipt.required_gradient_informativeness_fraction);
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

    fn wrong_dimension_at_start(_state: &[f64], _time: f64) -> Vec<f64> {
        Vec::new()
    }

    fn system_definition_canonical() -> &'static str {
        "system-definition-v1\nharmonic-oscillator-v1\nxdot=vx\nydot=vy\nvxdot=-x\nvydot=-y"
    }

    fn harmonic_system() -> AutonomousHoldoutSystemV1 {
        AutonomousHoldoutSystemV1::new(
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            harmonic_rhs,
        )
        .expect("valid harmonic test system")
    }
    fn invalid_rhs_system() -> AutonomousHoldoutSystemV1 {
        AutonomousHoldoutSystemV1::new(
            "invalid-rhs-shape-v1",
            "system-definition-v1\ninvalid-rhs-shape-v1\nrhs(state, t) -> []",
            wrong_dimension_at_start,
        )
            .expect("valid invalid-shape identity")
    }
    fn wrong_dimension_after_start_system() -> AutonomousHoldoutSystemV1 {
        AutonomousHoldoutSystemV1::new(
            "wrong-dimension-after-start-v1",
            "system-definition-v1\nwrong-dimension-after-start-v1\nrhs changes shape after t=0",
            wrong_dimension_after_start,
        )
        .expect("valid test identity")
    }
    fn constant_rhs(state: &[f64], _time: f64) -> Vec<f64> {
        vec![0.0; state.len()]
    }

    fn constant_system() -> AutonomousHoldoutSystemV1 {
        AutonomousHoldoutSystemV1::new(
            "constant-flow-v1",
            "system-definition-v1\nconstant-flow-v1\nxdot=0",
            constant_rhs,
        )
        .expect("valid constant identity")
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
    fn receipt_verifier_rejects_resealed_pass_with_low_gradient_informativeness() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"],
            &config,
            4.0,
            0.01,
            3,
            701,
            1e-2,
        );
        let mut receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            &config,
            &[],
            &candidate,
            &["x", "y", "vx", "vy"],
            &[1.0, 0.0, 0.0, 1.0],
            4.0,
            0.01,
            &holdout,
        ).expect("well-formed receipt should be constructed");
        receipt.min_gradient_informativeness_fraction = 0.25;
        receipt.receipt_blake3 = compute_receipt_digest(&receipt);
        assert!(!verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn holdout_rejects_high_power_candidate_with_low_gradient_informativeness() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let high_power = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(10.0)),
        );
        let result = validate_invariant_holdout(
            &harmonic_system(),
            &high_power,
            &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"],
            &config,
            std::f64::consts::TAU,
            0.01,
            4,
            701,
            1e6,
        );
        assert!(!result.passed);
        assert_eq!(
            result.failure,
            Some(AutonomousHoldoutFailure::InsufficientGradientInformativeness)
        );
        assert_eq!(result.trajectories_checked, 4);
        assert!(
            result.min_gradient_informativeness_fraction
                < MIN_AUTONOMOUS_HOLDOUT_INFORMATIVE_FRACTION
        );
    }

    #[test]
    fn identical_seeds_fail_closed() {
        let config = RegressorConfig { seed: 404, ..RegressorConfig::default() };
        let result = validate_invariant_holdout(
            &harmonic_system(), &harmonic_energy(), &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 404, 1e-2,
        );
        assert!(!result.passed);
        assert_eq!(result.failure, Some(AutonomousHoldoutFailure::IdenticalSeeds));
        assert_eq!(result.trajectories_checked, 0);
    }

    #[test]
    fn checked_rk4_reaches_horizon_despite_repeated_addition_roundoff() {
        for (t_end, dt, expected_steps) in [
            (4.0, 0.01, 400usize),
            (1.0, 0.1, 10usize),
        ] {
            let orbit = checked_rk4_trajectory(
                harmonic_rhs,
                &[1.0, 0.0, 0.0, 1.0],
                t_end,
                dt,
            )
            .expect("a complete orbit must not fail due to accumulated time rounding");
            assert_eq!(orbit.len(), expected_steps + 1);
            assert!(orbit.iter().all(|state| {
                state.len() == 4 && state.iter().all(|value| value.is_finite())
            }));
        }
    }

    #[test]
    fn harmonic_energy_generalizes_to_fresh_initial_conditions() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let result = validate_invariant_holdout(
            &harmonic_system(), &harmonic_energy(), &[1.0, 0.0, 0.0, 1.0],
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
            &harmonic_system(), &harmonic_energy(), &[f64::NAN, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 4, 709, 1e-2,
        );
        assert!(!malformed.passed);
        assert_eq!(malformed.failure, Some(AutonomousHoldoutFailure::InvalidInput));

        let zero = validate_invariant_holdout(
            &harmonic_system(), &harmonic_energy(), &[1.0, 0.0, 0.0, 1.0],
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
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
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
    fn cold_receipt_contains_the_sanitized_effective_search_config() {
        let mut config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        config.known_invariants.push(Expr::Var("x".into()));
        config.orthogonality_penalty = 4.0;
        config.prior_composition_rate = 0.6;
        config.prior_fragment_bonus = 0.5;
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"],
            &config,
            4.0,
            0.01,
            2,
            701,
            1e-2,
        );
        let receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            &config,
            &[],
            &candidate,
            &["x", "y", "vx", "vy"],
            &[1.0, 0.0, 0.0, 1.0],
            4.0,
            0.01,
            &holdout,
        )
        .expect("cold mode must sanitize prior-dependent configuration");
        assert!(receipt.search_config.disable_macro_seeds);
        assert_eq!(receipt.search_config.prior_composition_rate, 0.0);
        assert_eq!(receipt.search_config.prior_fragment_bonus, 1.0);
        assert_eq!(receipt.search_config.orthogonality_penalty, 1.0);
        assert!(receipt.search_config.known_invariant_formulas.is_empty());
        assert!(receipt.search_config.extra_seed_template_formulas.is_empty());
        assert!(verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn receipt_rejects_cold_run_with_explicit_priors() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
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
        assert_eq!(result.unwrap_err(), AutonomousDiscoveryReceiptError::UncontrolledRunContainsPriors);
    }

    #[test]
    fn random_baseline_receipt_records_sanitized_prior_policy() {
        let mut config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        config.disable_macro_seeds = false;
        config.prior_composition_rate = 0.6;
        config.prior_fragment_bonus = 0.5;
        config.orthogonality_penalty = 4.0;
        config.known_invariants = vec![Expr::Var("x".into())];

        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::RandomBaseline);
        let holdout = validate_invariant_holdout(
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 2, 701, 1e-2,
        );
        let receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("random baseline receipt should be constructed with sanitized policy");

        assert_eq!(receipt.discovery_mode, AutonomousDiscoveryMode::RandomBaseline);
        assert!(receipt.search_config.disable_macro_seeds);
        assert_eq!(receipt.search_config.prior_composition_rate, 0.0);
        assert_eq!(receipt.search_config.prior_fragment_bonus, 1.0);
        assert_eq!(receipt.search_config.orthogonality_penalty, 1.0);
        assert!(receipt.search_config.known_invariant_formulas.is_empty());
        assert!(receipt.search_config.extra_seed_template_formulas.is_empty());
        assert!(verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn random_baseline_receipt_rejects_explicit_prior_templates() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::RandomBaseline);
        let holdout = validate_invariant_holdout(
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 2, 701, 1e-2,
        );
        let result = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[Expr::Var("x".into())], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        );
        assert_eq!(result.unwrap_err(), AutonomousDiscoveryReceiptError::UncontrolledRunContainsPriors);
    }

    #[test]
    fn receipt_binds_actual_holdout_work_counters() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 2, 701, 1e-2,
        );
        let mut receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("valid holdout work record should be receipted");
        assert!(verify_autonomous_discovery_receipt_v1(&receipt));
        assert_eq!(
            receipt.integration_rhs_evaluations,
            receipt.completed_integration_steps * 4
        );
        assert!(receipt.scoring_rhs_evaluations > 0);
        assert_eq!(
            receipt.rhs_evaluations,
            receipt.integration_rhs_evaluations + receipt.scoring_rhs_evaluations
        );
        assert_eq!(receipt.state_updates, receipt.completed_integration_steps * 4);

        receipt.rhs_evaluations += 1;
        receipt.receipt_blake3 = compute_receipt_digest(&receipt);
        assert!(!verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn composed_candidates_need_parent_receipts() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Composed);
        let holdout = validate_invariant_holdout(
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
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
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
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
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
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
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
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
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
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
    fn verifier_rejects_resealed_identical_seed_receipt_with_fabricated_metrics() {
        let config = RegressorConfig { seed: 404, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"],
            &config,
            4.0,
            0.01,
            3,
            404,
            1e-2,
        );
        let mut receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            &config,
            &[],
            &candidate,
            &["x", "y", "vx", "vy"],
            &[1.0, 0.0, 0.0, 1.0],
            4.0,
            0.01,
            &holdout,
        ).expect("identical-seed failure should remain recordable");

        receipt.max_lie_derivative_variance = 0.0;
        receipt.median_lie_derivative_variance = 0.0;
        receipt.min_gradient_informativeness_fraction = 1.0;
        receipt.receipt_blake3 = compute_receipt_digest(&receipt);
        assert!(!verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn verifier_rejects_resealed_receipt_with_changed_gradient_policy() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"],
            &config,
            4.0,
            0.01,
            3,
            701,
            1e-2,
        );
        let mut receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            &config,
            &[],
            &candidate,
            &["x", "y", "vx", "vy"],
            &[1.0, 0.0, 0.0, 1.0],
            4.0,
            0.01,
            &holdout,
        ).expect("well-formed receipt should be constructed");

        receipt.required_gradient_informativeness_fraction = 0.1;
        receipt.assessment_context_blake3 = assessment_context_digest(
            &receipt.candidate_formula,
            &receipt.variable_names,
            &receipt.initial_state,
            receipt.discovery_seed,
            receipt.holdout_seed,
            receipt.t_max,
            receipt.dt,
            receipt.requested_trajectories,
            receipt.threshold,
            receipt.gradient_informativeness_relative_squared_floor,
            receipt.required_gradient_informativeness_fraction,
            &receipt.system_id,
            &receipt.system_definition_blake3,
        );
        receipt.receipt_blake3 = compute_receipt_digest(&receipt);
        assert!(!verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn builder_rejects_rehashed_zero_trajectory_record_with_identical_seeds() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let initial = [1.0, 0.0, 0.0, 1.0];
        let names = ["x", "y", "vx", "vy"];
        let mut holdout = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &initial,
            &names,
            &config,
            4.0,
            0.01,
            0,
            701,
            1e-2,
        );
        assert_eq!(
            holdout.failure,
            Some(AutonomousHoldoutFailure::ZeroTrajectories)
        );

        holdout.holdout_seed = config.seed;
        let variable_names: Vec<String> =
            names.iter().map(|name| (*name).to_string()).collect();
        holdout.assessment_context_blake3 = assessment_context_digest(
            &format!("{}", formula),
            &variable_names,
            &initial,
            config.seed,
            holdout.holdout_seed,
            4.0,
            0.01,
            holdout.requested_trajectories,
            holdout.threshold,
            holdout.gradient_informativeness_relative_squared_floor,
            holdout.required_gradient_informativeness_fraction,
            &holdout.system_id,
            &holdout.system_definition_blake3,
        );

        let result = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            &config,
            &[],
            &candidate,
            &names,
            &initial,
            4.0,
            0.01,
            &holdout,
        );
        assert_eq!(
            result.unwrap_err(),
            AutonomousDiscoveryReceiptError::InvalidHoldoutRecord
        );
    }

    #[test]
    fn receipt_rejects_holdout_metrics_reused_with_different_initial_state() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 701, 1e-2,
        );
        let result = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[2.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        );
        assert_eq!(result.unwrap_err(), AutonomousDiscoveryReceiptError::InvalidHoldoutRecord);
    }

    #[test]
    fn system_identity_requires_matching_versioned_definition() {
        let result = AutonomousHoldoutSystemV1::new(
            "another-system-v1",
            system_definition_canonical(),
            harmonic_rhs,
        );
        assert!(matches!(
            result,
            Err(AutonomousDiscoveryReceiptError::InvalidSystemDefinition)
        ));
    }

    #[test]
    fn holdout_context_rejects_receipt_for_another_system() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let holdout = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"],
            &config,
            4.0,
            0.01,
            3,
            701,
            1e-2,
        );
        let result = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "unrelated-system-v1",
            "system-definition-v1\nunrelated-system-v1\nxdot=0",
            &config,
            &[],
            &candidate(formula, AutonomousDiscoveryMode::Cold),
            &["x", "y", "vx", "vy"],
            &[1.0, 0.0, 0.0, 1.0],
            4.0,
            0.01,
            &holdout,
        );
        assert!(matches!(
            result,
            Err(AutonomousDiscoveryReceiptError::InvalidHoldoutRecord)
        ));
    }

    #[test]
    fn verifier_rejects_resealed_receipt_with_mismatched_system_id() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"],
            &config,
            4.0,
            0.01,
            3,
            701,
            1e-2,
        );
        let mut receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            &config,
            &[],
            &candidate,
            &["x", "y", "vx", "vy"],
            &[1.0, 0.0, 0.0, 1.0],
            4.0,
            0.01,
            &holdout,
        ).expect("well-formed receipt should be constructed");

        receipt.system_id = "unrelated-system-v1".to_string();
        receipt.assessment_context_blake3 = assessment_context_digest(
            &receipt.candidate_formula,
            &receipt.variable_names,
            &receipt.initial_state,
            receipt.discovery_seed,
            receipt.holdout_seed,
            receipt.t_max,
            receipt.dt,
            receipt.requested_trajectories,
            receipt.threshold,
            receipt.gradient_informativeness_relative_squared_floor,
            receipt.required_gradient_informativeness_fraction,
            &receipt.system_id,
            &receipt.system_definition_blake3,
        );
        receipt.receipt_blake3 = compute_receipt_digest(&receipt);
        assert!(!verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn verifier_rejects_resealed_receipt_with_changed_initial_state() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 701, 1e-2,
        );
        let mut receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("well-formed receipt should be constructed");

        receipt.initial_state[0] = 2.0;
        receipt.receipt_blake3 = compute_receipt_digest(&receipt);
        assert!(!verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn changing_system_definition_changes_receipt_identity() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 3, 701, 1e-2,
        );
        let receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1", system_definition_canonical(), &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("well-formed receipt should be constructed");
        let mut altered = receipt.clone();
        altered.system_definition_canonical =
            "system-definition-v1\nharmonic-oscillator-v1\nxdot=0\nydot=0\nvxdot=0\nvydot=0"
                .to_string();
        altered.receipt_blake3 = compute_receipt_digest(&altered);
        assert!(!verify_autonomous_discovery_receipt_v1(&altered));
    }


    #[test]
    fn immediate_invalid_rhs_result_is_archived_as_a_failed_receipt() {
        let config = RegressorConfig { seed: 23, ..RegressorConfig::default() };
        let formula = Expr::Var("x".into());
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let initial = [1.0, 0.0, 0.0, 1.0];
        let names = ["x", "y", "vx", "vy"];
        let holdout = validate_invariant_holdout(
            &invalid_rhs_system(),
            &formula,
            &initial,
            &names,
            &config,
            4.0,
            0.01,
            3,
            709,
            1e-2,
        );
        assert!(!holdout.passed);
        assert_eq!(holdout.failure, Some(AutonomousHoldoutFailure::InvalidInput));
        assert_eq!(holdout.trajectories_checked, 0);

        let receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "invalid-rhs-shape-v1",
            "system-definition-v1\ninvalid-rhs-shape-v1\nrhs(state, t) -> []",
            &config,
            &[],
            &candidate,
            &names,
            &initial,
            4.0,
            0.01,
            &holdout,
        )
        .expect("the invalid-input result must remain archivable");
        assert!(!receipt.holdout_passed);
        assert_eq!(receipt.holdout_failure, Some(AutonomousHoldoutFailure::InvalidInput));
        assert!(verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn wrong_rhs_dimension_on_intermediate_step_fails_closed_without_panicking() {
        let config = RegressorConfig { seed: 23, ..RegressorConfig::default() };
        let result = validate_invariant_holdout(
            &wrong_dimension_after_start_system(), &harmonic_energy(), &[1.0, 0.0, 0.0, 1.0],
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
            &constant_system(),
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
            &harmonic_system(), &harmonic_energy(), &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 600.0, 0.01, 100, 709, 1e-2,
        );
        assert!(!result.passed);
        assert_eq!(result.trajectories_checked, 0);
        assert_eq!(result.failure, Some(AutonomousHoldoutFailure::TrajectoryLimitExceeded));
    }

    #[test]
    fn unversioned_or_noncanonical_system_definition_is_rejected() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"], &config, 4.0, 0.01, 2, 701, 1e-2,
        );
        let invalid_definitions = [
            "",
            "harmonic-oscillator-v1\nxdot=vx",
            "system-definition-v1\r\nmodel",
        ];
        for invalid_definition in invalid_definitions {
            let result = build_autonomous_discovery_receipt_v1(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "harmonic-oscillator-v1",
                invalid_definition,
                &config,
                &[],
                &candidate,
                &["x", "y", "vx", "vy"],
                &[1.0, 0.0, 0.0, 1.0],
                4.0,
                0.01,
                &holdout,
            );
            assert_eq!(
                result.unwrap_err(),
                AutonomousDiscoveryReceiptError::InvalidSystemDefinition
            );
        }
    }

    #[test]
    fn verifier_rejects_malformed_search_digest_even_after_redigest() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(), &formula, &[1.0, 0.0, 0.0, 1.0],
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

    #[test]
    fn verifier_rejects_cold_receipt_with_prior_snapshot_even_after_redigest() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let holdout = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"],
            &config,
            4.0,
            0.01,
            2,
            701,
            1e-2,
        );
        let mut receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            &config,
            &[],
            &candidate,
            &["x", "y", "vx", "vy"],
            &[1.0, 0.0, 0.0, 1.0],
            4.0,
            0.01,
            &holdout,
        )
        .expect("well-formed receipt should be constructed");
        receipt.search_config.known_invariant_formulas.push("x".to_string());
        receipt.search_config_blake3 = search_config_digest(&receipt.search_config);
        receipt.receipt_blake3 = compute_receipt_digest(&receipt);
        assert!(!verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn verifier_rejects_composed_receipt_even_after_redigest() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Primed);
        let holdout = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &[1.0, 0.0, 0.0, 1.0],
            &["x", "y", "vx", "vy"],
            &config,
            4.0,
            0.01,
            2,
            701,
            1e-2,
        );
        let mut receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            &config,
            &[],
            &candidate,
            &["x", "y", "vx", "vy"],
            &[1.0, 0.0, 0.0, 1.0],
            4.0,
            0.01,
            &holdout,
        )
        .expect("well-formed receipt should be constructed");
        receipt.discovery_mode = AutonomousDiscoveryMode::Composed;
        receipt.receipt_blake3 = compute_receipt_digest(&receipt);
        assert!(!verify_autonomous_discovery_receipt_v1(&receipt));
    }

    #[test]
    fn holdout_replay_and_receipt_json_roundtrip_are_deterministic() {
        let config = RegressorConfig { seed: 13, ..RegressorConfig::default() };
        let formula = harmonic_energy();
        let initial = [1.0, 0.0, 0.0, 1.0];
        let names = ["x", "y", "vx", "vy"];
        let first = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &initial,
            &names,
            &config,
            4.0,
            0.01,
            3,
            701,
            1e-2,
        );
        let replay = validate_invariant_holdout(
            &harmonic_system(),
            &formula,
            &initial,
            &names,
            &config,
            4.0,
            0.01,
            3,
            701,
            1e-2,
        );
        assert_eq!(first, replay);

        let candidate = candidate(formula.clone(), AutonomousDiscoveryMode::Cold);
        let receipt = build_autonomous_discovery_receipt_v1(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "harmonic-oscillator-v1",
            system_definition_canonical(),
            &config,
            &[],
            &candidate,
            &names,
            &initial,
            4.0,
            0.01,
            &first,
        )
        .expect("well-formed receipt should be constructed");
        let encoded = serde_json::to_string(&receipt)
            .expect("receipt should serialize to JSON");
        let decoded: AutonomousDiscoveryReceiptV1 = serde_json::from_str(&encoded)
            .expect("receipt should deserialize from JSON");
        assert_eq!(decoded, receipt);
        assert!(verify_autonomous_discovery_receipt_v1(&decoded));
    }

}
