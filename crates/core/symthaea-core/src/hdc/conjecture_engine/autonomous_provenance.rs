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
        || {
            let initial_derivative = rhs(initial_state, 0.0);
            initial_derivative.len() != initial_state.len()
                || initial_derivative.iter().any(|value| !value.is_finite())
        }
        || !t_max.is_finite()
        || t_max <= 0.0
        || !dt.is_finite()
        || dt <= 0.0
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
    if holdout_trajectories > MAX_AUTONOMOUS_HOLDOUT_TRAJECTORIES {
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

    let sample_orbit = |initial: &[f64]| -> Option<Vec<Vec<f64>>> {
        let (_times, states) = rk45_trajectory(rhs, initial, t_max, dt);
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
    if initial_state.len() != var_names.len() || initial_state.iter().any(|value| !value.is_finite()) {
        return Err(AutonomousDiscoveryReceiptError::InvalidInitialState);
    }
    if !t_max.is_finite() || t_max <= 0.0 || !dt.is_finite() || dt <= 0.0 {
        return Err(AutonomousDiscoveryReceiptError::InvalidTiming);
    }
    if config.seed == holdout.holdout_seed {
        return Err(AutonomousDiscoveryReceiptError::IdenticalSeeds);
    }
    if holdout.discovery_seed != config.seed {
        return Err(AutonomousDiscoveryReceiptError::HoldoutSeedMismatch);
    }
    if candidate.discovery_mode == AutonomousDiscoveryMode::Composed {
        return Err(AutonomousDiscoveryReceiptError::ComposedCandidateNeedsParentReceipts);
    }
    if candidate.discovery_mode == AutonomousDiscoveryMode::Cold && !extra_seed_templates.is_empty() {
        return Err(AutonomousDiscoveryReceiptError::ColdRunContainsPriors);
    }
    let effective = sanitized_config(config, candidate.discovery_mode);
    let holdout_valid = holdout.requested_trajectories > 0
        && holdout.trajectories_checked <= holdout.requested_trajectories
        && holdout.threshold.is_finite()
        && holdout.threshold >= 0.0
        && holdout.max_lie_derivative_variance.is_finite()
        && holdout.median_lie_derivative_variance.is_finite()
        && (holdout.passed
            == (holdout.failure.is_none()
                && holdout.trajectories_checked == holdout.requested_trajectories
                && holdout.max_lie_derivative_variance <= holdout.threshold));
    if !holdout_valid {
        return Err(AutonomousDiscoveryReceiptError::InvalidHoldoutRecord);
    }

    let formula = candidate.formula_str.clone();
    let candidate_blake3 = blake3::hash(formula.as_bytes()).to_hex().to_string();
    let mut receipt = AutonomousDiscoveryReceiptV1 {
        schema_version: 1,
        source_revision,
        system_id: system_id.trim().to_string(),
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
    receipt.schema_version == 1
        && receipt.discovery_seed != receipt.holdout_seed
        && blake3::hash(receipt.candidate_formula.as_bytes()).to_hex().to_string()
            == receipt.candidate_blake3
        && compute_receipt_digest(receipt) == receipt.receipt_blake3
}

fn compute_receipt_digest(receipt: &AutonomousDiscoveryReceiptV1) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_string(&mut hasher, "symthaea-autonomous-discovery-receipt-v1");
    hash_u64(&mut hasher, receipt.schema_version as u64);
    hash_string(&mut hasher, &receipt.source_revision);
    hash_string(&mut hasher, &receipt.system_id);
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
            "harmonic-oscillator-v1", &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        ).expect("well-formed receipt should be constructed");
        assert!(verify_autonomous_discovery_receipt_v1(&receipt));
        assert_eq!(receipt.discovery_seed, 13);
        assert_eq!(receipt.holdout_seed, 701);
        assert_eq!(receipt.discovery_mode, AutonomousDiscoveryMode::Cold);
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
            "harmonic-oscillator-v1", &config, &[Expr::Var("x".into())],
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
            "harmonic-oscillator-v1", &config, &[], &candidate,
            &["x", "y", "vx", "vy"], &[1.0, 0.0, 0.0, 1.0],
            4.0, 0.01, &holdout,
        );
        assert_eq!(result.unwrap_err(), AutonomousDiscoveryReceiptError::ComposedCandidateNeedsParentReceipts);
    }
}
