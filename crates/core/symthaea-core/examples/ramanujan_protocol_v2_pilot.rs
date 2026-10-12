//! Reproducible pilot for the Ramanujan Protocol v2 discovery conditions.
//!
//! Usage:
//! cargo run --locked -p symthaea-core --example ramanujan_protocol_v2_pilot -- \
//!   target/ramanujan-protocol-v2-pilot.jsonl [replicates=10] [first_seed=42]
//!
//! The output file is created with create_new semantics: existing evidence is
//! never overwritten. Every line contains a self-digest, exact discovery and
//! holdout work counters, candidate-level holdout receipts, and an analytic
//! quadratic-energy oracle classification. Passing this pilot is not a formal
//! proof and results should be aggregated across the preregistered seed batch.

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    process::Command,
};

use serde_json::{Value, json};
use symthaea_core::hdc::conjecture_engine::{
    AutonomousDiscoveryMode, AutonomousHoldoutSystemV1, AutonomousInvariant,
    RegressorConfig, build_autonomous_discovery_receipt_v1,
    discover_invariants_autonomous_cold_with_metrics,
    discover_invariants_autonomous_random_baseline_with_metrics,
    discover_invariants_autonomous_with_seed_templates_and_metrics,
    validate_invariant_holdout, verify_autonomous_discovery_receipt_v1,
    BinOp, Expr,
};

const PILOT_SCHEMA_VERSION: u32 = 1;
const PILOT_PROTOCOL_ID: &str = "ramanujan-protocol-v2-pilot-1";
const SYSTEM_ID: &str = "harmonic-oscillator-1d-v1";
const SYSTEM_DEFINITION: &str =
    "system-definition-v1\nharmonic-oscillator-1d-v1\nxdot=v\nvdot=-x\nunits=dimensionless\n";
const HOLDOUT_SEED_OFFSET: u64 = 0x9E3779B97F4A7C15;
const T_MAX: f64 = 4.0;
const DT: f64 = 0.01;
const HOLDOUT_TRAJECTORIES: usize = 3;
const HOLDOUT_THRESHOLD: f64 = 1e-2;
/// Exploratory instrumentation threshold, to be frozen before confirmatory runs.
const STEP_SIZE_STABILITY_RELATIVE_TOLERANCE: f64 = 0.10;
/// Absolute floor for near-zero Lie-variance comparisons at different step sizes.
const STEP_SIZE_STABILITY_ABSOLUTE_TOLERANCE: f64 = 1e-12;
const MAX_REPLICATES: usize = 100;

type Monomial = (u8, u8);
type Polynomial = BTreeMap<Monomial, f64>;

fn harmonic_rhs(state: &[f64], _time: f64) -> Vec<f64> {
    if state.len() != 2 {
        return Vec::new();
    }
    vec![state[1], -state[0]]
}

fn command_text(program: &str, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        return Err(format!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn source_revision() -> Result<String, Box<dyn Error>> {
    let revision = command_text("git", &["rev-parse", "HEAD"])?.to_ascii_lowercase();
    if !matches!(revision.len(), 40 | 64)
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("git HEAD must be a full 40- or 64-character hex revision".into());
    }
    let dirty = command_text("git", &["status", "--porcelain", "--untracked-files=all"])?;
    if !dirty.is_empty() {
        return Err("refusing evidence generation from a dirty or untracked working tree".into());
    }
    Ok(revision)
}

fn file_blake3(path: &str) -> Result<String, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn canonical_pilot_conditions() -> [AutonomousDiscoveryMode; 3] {
    [
        AutonomousDiscoveryMode::Cold,
        AutonomousDiscoveryMode::Primed,
        AutonomousDiscoveryMode::RandomBaseline,
    ]
}

fn run_order_for_replicate(replicate: usize) -> [AutonomousDiscoveryMode; 3] {
    let canonical = canonical_pilot_conditions();
    let orders = [
        canonical,
        [
            AutonomousDiscoveryMode::Primed,
            AutonomousDiscoveryMode::RandomBaseline,
            AutonomousDiscoveryMode::Cold,
        ],
        [
            AutonomousDiscoveryMode::RandomBaseline,
            AutonomousDiscoveryMode::Cold,
            AutonomousDiscoveryMode::Primed,
        ],
    ];
    orders[replicate % orders.len()]
}

fn mode_name(mode: AutonomousDiscoveryMode) -> &'static str {
    match mode {
        AutonomousDiscoveryMode::Cold => "cold",
        AutonomousDiscoveryMode::Primed => "primed",
        AutonomousDiscoveryMode::RandomBaseline => "random_baseline",
        AutonomousDiscoveryMode::Composed => "composed",
    }
}

#[derive(Debug, Clone, Copy)]
struct StepSizeStabilityAssessmentV1 {
    passed: bool,
    absolute_variance_difference: f64,
    relative_variance_difference: f64,
}

fn assess_step_size_stability(
    coarse_passed: bool,
    fine_passed: bool,
    coarse_variance: f64,
    fine_variance: f64,
) -> StepSizeStabilityAssessmentV1 {
    if !coarse_variance.is_finite()
        || !fine_variance.is_finite()
        || coarse_variance < 0.0
        || fine_variance < 0.0
        || coarse_variance >= f64::MAX
        || fine_variance >= f64::MAX
    {
        return StepSizeStabilityAssessmentV1 {
            passed: false,
            absolute_variance_difference: f64::MAX,
            relative_variance_difference: f64::MAX,
        };
    }

    let absolute_variance_difference = (coarse_variance - fine_variance).abs();
    let scale = coarse_variance.abs().max(fine_variance.abs());
    let relative_variance_difference =
        absolute_variance_difference / scale.max(STEP_SIZE_STABILITY_ABSOLUTE_TOLERANCE);
    let within_mixed_tolerance = absolute_variance_difference
        <= STEP_SIZE_STABILITY_ABSOLUTE_TOLERANCE
            + STEP_SIZE_STABILITY_RELATIVE_TOLERANCE * scale;

    StepSizeStabilityAssessmentV1 {
        passed: coarse_passed && fine_passed && within_mixed_tolerance,
        absolute_variance_difference,
        relative_variance_difference,
    }
}

fn empty_constant(value: f64) -> Polynomial {
    let mut result = Polynomial::new();
    if value != 0.0 {
        result.insert((0, 0), value);
    }
    result
}

fn normalize(poly: &mut Polynomial) {
    poly.retain(|_, coefficient| coefficient.abs() > 1e-12);
}

fn add_polynomials(left: &Polynomial, right: &Polynomial, right_scale: f64) -> Polynomial {
    let mut result = left.clone();
    for (monomial, coefficient) in right {
        *result.entry(*monomial).or_insert(0.0) += right_scale * coefficient;
    }
    normalize(&mut result);
    result
}

fn multiply_polynomials(left: &Polynomial, right: &Polynomial) -> Option<Polynomial> {
    let mut result = Polynomial::new();
    for ((left_x, left_v), left_coefficient) in left {
        for ((right_x, right_v), right_coefficient) in right {
            let x_power = *left_x + *right_x;
            let v_power = *left_v + *right_v;
            if x_power + v_power > 2 {
                return None;
            }
            *result.entry((x_power, v_power)).or_insert(0.0) +=
                left_coefficient * right_coefficient;
        }
    }
    normalize(&mut result);
    result.values().all(|coefficient| coefficient.is_finite()).then_some(result)
}

/// Convert only the small degree-2 polynomial grammar into an exact coefficient
/// map. Unsupported operations are not guessed at by evaluating a few points.
fn quadratic_polynomial(expression: &Expr) -> Option<Polynomial> {
    match expression {
        Expr::Const(value) if value.is_finite() => Some(empty_constant(*value)),
        Expr::Var(name) if name == "x" => {
            let mut result = Polynomial::new();
            result.insert((1, 0), 1.0);
            Some(result)
        }
        Expr::Var(name) if name == "v" => {
            let mut result = Polynomial::new();
            result.insert((0, 1), 1.0);
            Some(result)
        }
        Expr::BinOp(operation, left, right) => {
            let left_poly = quadratic_polynomial(left)?;
            match operation {
                BinOp::Add | BinOp::Sub => {
                    let right_poly = quadratic_polynomial(right)?;
                    Some(add_polynomials(
                        &left_poly,
                        &right_poly,
                        if matches!(operation, BinOp::Add) { 1.0 } else { -1.0 },
                    ))
                }
                BinOp::Mul => {
                    let right_poly = quadratic_polynomial(right)?;
                    multiply_polynomials(&left_poly, &right_poly)
                }
                BinOp::Div => {
                    let mut denominator = quadratic_polynomial(right)?;
                    normalize(&mut denominator);
                    if denominator.keys().any(|monomial| *monomial != (0, 0)) {
                        return None;
                    }
                    let divisor = *denominator.get(&(0, 0)).unwrap_or(&0.0);
                    if !divisor.is_finite() || divisor.abs() <= 1e-15 {
                        return None;
                    }
                    let mut result = left_poly
                        .into_iter()
                        .map(|(monomial, coefficient)| (monomial, coefficient / divisor))
                        .collect::<Polynomial>();
                    normalize(&mut result);
                    result.values().all(|coefficient| coefficient.is_finite()).then_some(result)
                }
                BinOp::Pow => {
                    let power = match right.as_ref() {
                        Expr::Const(power) if power.is_finite() => *power,
                        _ => return None,
                    };
                    if power == 0.0 {
                        return Some(empty_constant(1.0));
                    }
                    if power == 1.0 {
                        return Some(left_poly);
                    }
                    if power == 2.0 {
                        return multiply_polynomials(&left_poly, &left_poly);
                    }
                    None
                }
            }
        }
        Expr::Func(_, _) | Expr::Sum(_, _) => None,
        Expr::Var(_) | Expr::Const(_) => None,
    }
}

/// Analytic coefficient-map oracle for the dimensionless harmonic oscillator.
/// Accepts the quadratic family a*(x²+v²)+b above the explicit 1e-10 coefficient
/// floor, with 1e-10 relative coefficient tolerance. Coefficients below 1e-12
/// are normalized away. This classification is tolerance-qualified algebraic
/// form, not physical truth and not the holdout criterion.
/// Three-way oracle result: None means the expression falls outside the
/// oracle's supported degree-2 polynomial grammar and must not be counted as a
/// false discovery.
fn classify_harmonic_energy_oracle(candidate: &AutonomousInvariant) -> Option<bool> {
    let mut polynomial = quadratic_polynomial(&candidate.formula)?;
    normalize(&mut polynomial);
    let x_squared = *polynomial.get(&(2, 0)).unwrap_or(&0.0);
    let v_squared = *polynomial.get(&(0, 2)).unwrap_or(&0.0);
    let coefficient_scale = x_squared.abs().max(v_squared.abs()).max(1.0);
    let tolerance = 1e-10 * coefficient_scale;
    if x_squared.abs() <= 1e-10 || (x_squared - v_squared).abs() > tolerance {
        return Some(false);
    }
    Some(polynomial.iter().all(|(monomial, coefficient)| {
        matches!(monomial, (0, 0) | (2, 0) | (0, 2)) || coefficient.abs() <= tolerance
    }))
}

fn condition_run(
    mode: AutonomousDiscoveryMode,
    config: &RegressorConfig,
) -> symthaea_core::hdc::conjecture_engine::AutonomousDiscoveryRunV1 {
    match mode {
        AutonomousDiscoveryMode::Cold =>
            discover_invariants_autonomous_cold_with_metrics(
                harmonic_rhs, &[1.0, 0.0], &["x", "v"], None, config, T_MAX, DT,
            ),
        AutonomousDiscoveryMode::Primed =>
            discover_invariants_autonomous_with_seed_templates_and_metrics(
                harmonic_rhs, &[1.0, 0.0], &["x", "v"], None, config, T_MAX, DT, &[],
            ),
        AutonomousDiscoveryMode::RandomBaseline =>
            discover_invariants_autonomous_random_baseline_with_metrics(
                harmonic_rhs, &[1.0, 0.0], &["x", "v"], None, config, T_MAX, DT,
            ),
        AutonomousDiscoveryMode::Composed => unreachable!("pilot does not run composed mode"),
    }
}

fn build_run_record(
    mode: AutonomousDiscoveryMode,
    seed: u64,
    holdout_seed: u64,
    config: &RegressorConfig,
    source_revision: &str,
    rustc_version: &str,
    cargo_version: &str,
    host_parallelism: usize,
    replicate_order_index: usize,
    protocol_blake3: &str,
    cargo_lock_blake3: &str,
    workspace_manifest_blake3: &str,
    core_manifest_blake3: &str,
) -> Result<Value, Box<dyn Error>> {
    if config.seed != seed || seed == holdout_seed {
        return Err("paired-seed invariant failed before run".into());
    }
    let discovery = condition_run(mode, config);
    let system = AutonomousHoldoutSystemV1::new(SYSTEM_ID, SYSTEM_DEFINITION, harmonic_rhs)
        .map_err(|error| format!("system identity initialization failed: {error:?}"))?;

    let mut candidate_records = Vec::with_capacity(discovery.candidates.len());
    let mut holdout_attempted_steps = 0u64;
    let mut holdout_completed_steps = 0u64;
    let mut holdout_integration_rhs_evaluations = 0u64;
    let mut holdout_scoring_rhs_evaluations = 0u64;
    let mut holdout_rhs_evaluations = 0u64;
    let mut holdout_state_updates = 0u64;
    let mut holdout_passing_candidates = 0usize;
    let mut stable_holdout_candidates = 0usize;
    let mut holdout_stability_passes = 0usize;
    let mut oracle_supported_candidates = 0usize;
    let mut oracle_unsupported_candidates = 0usize;
    let mut oracle_equivalent_candidates = 0usize;
    let mut oracle_and_holdout_candidates = 0usize;
    let mut oracle_and_stable_holdout_candidates = 0usize;
    let mut holdout_false_discoveries = 0usize;
    let mut stable_holdout_false_discoveries = 0usize;
    let mut oracle_unsupported_holdout_candidates = 0usize;
    let mut oracle_unsupported_stable_holdout_candidates = 0usize;
    let mut receipt_errors = 0usize;

    for candidate in &discovery.candidates {
        let oracle_equivalence = classify_harmonic_energy_oracle(candidate);
        match oracle_equivalence {
            Some(true) => {
                oracle_supported_candidates += 1;
                oracle_equivalent_candidates += 1;
            }
            Some(false) => {
                oracle_supported_candidates += 1;
            }
            None => {
                oracle_unsupported_candidates += 1;
            }
        }
        let holdout = validate_invariant_holdout(
            &system,
            &candidate.formula,
            &[1.0, 0.0],
            &["x", "v"],
            config,
            T_MAX,
            DT,
            HOLDOUT_TRAJECTORIES,
            holdout_seed,
            HOLDOUT_THRESHOLD,
        );
        // Re-assess the same sealed initial-condition stream at half the step
        // size. A separate receipt binds the distinct numerical context.
        let stability_holdout = validate_invariant_holdout(
            &system,
            &candidate.formula,
            &[1.0, 0.0],
            &["x", "v"],
            config,
            T_MAX,
            DT / 2.0,
            HOLDOUT_TRAJECTORIES,
            holdout_seed,
            HOLDOUT_THRESHOLD,
        );
        holdout_attempted_steps = holdout_attempted_steps
            .saturating_add(holdout.attempted_integration_steps)
            .saturating_add(stability_holdout.attempted_integration_steps);
        holdout_completed_steps = holdout_completed_steps
            .saturating_add(holdout.completed_integration_steps)
            .saturating_add(stability_holdout.completed_integration_steps);
        holdout_integration_rhs_evaluations = holdout_integration_rhs_evaluations
            .saturating_add(holdout.integration_rhs_evaluations)
            .saturating_add(stability_holdout.integration_rhs_evaluations);
        holdout_scoring_rhs_evaluations = holdout_scoring_rhs_evaluations
            .saturating_add(holdout.scoring_rhs_evaluations)
            .saturating_add(stability_holdout.scoring_rhs_evaluations);
        holdout_rhs_evaluations = holdout_rhs_evaluations
            .saturating_add(holdout.rhs_evaluations)
            .saturating_add(stability_holdout.rhs_evaluations);
        holdout_state_updates = holdout_state_updates
            .saturating_add(holdout.state_updates)
            .saturating_add(stability_holdout.state_updates);

        let stability = assess_step_size_stability(
            holdout.passed,
            stability_holdout.passed,
            holdout.max_lie_derivative_variance,
            stability_holdout.max_lie_derivative_variance,
        );
        let absolute_step_size_difference = stability.absolute_variance_difference;
        let relative_step_size_difference = stability.relative_variance_difference;
        let step_size_stable = stability.passed;

        if holdout.passed {
            holdout_passing_candidates += 1;
            match oracle_equivalence {
                Some(true) => oracle_and_holdout_candidates += 1,
                Some(false) => holdout_false_discoveries += 1,
                None => oracle_unsupported_holdout_candidates += 1,
            }
        }
        if step_size_stable {
            holdout_stability_passes += 1;
            stable_holdout_candidates += 1;
            match oracle_equivalence {
                Some(true) => oracle_and_stable_holdout_candidates += 1,
                Some(false) => stable_holdout_false_discoveries += 1,
                None => oracle_unsupported_stable_holdout_candidates += 1,
            }
        }

        let receipt_value = match build_autonomous_discovery_receipt_v1(
            source_revision,
            SYSTEM_ID,
            SYSTEM_DEFINITION,
            config,
            &[],
            candidate,
            &["x", "v"],
            &[1.0, 0.0],
            T_MAX,
            DT,
            &holdout,
        ) {
            Ok(receipt) => {
                let verified = verify_autonomous_discovery_receipt_v1(&receipt);
                if !verified {
                    receipt_errors += 1;
                }
                json!({"verified": verified, "receipt": receipt})
            }
            Err(error) => {
                receipt_errors += 1;
                json!({"verified": false, "error": format!("{error:?}")})
            }
        };
        let stability_receipt_value = match build_autonomous_discovery_receipt_v1(
            source_revision,
            SYSTEM_ID,
            SYSTEM_DEFINITION,
            config,
            &[],
            candidate,
            &["x", "v"],
            &[1.0, 0.0],
            T_MAX,
            DT / 2.0,
            &stability_holdout,
        ) {
            Ok(receipt) => {
                let verified = verify_autonomous_discovery_receipt_v1(&receipt);
                if !verified {
                    receipt_errors += 1;
                }
                json!({"verified": verified, "receipt": receipt})
            }
            Err(error) => {
                receipt_errors += 1;
                json!({"verified": false, "error": format!("{error:?}")})
            }
        };

        candidate_records.push(json!({
            "formula": candidate.formula_str,
            "in_sample_lie_variance": candidate.variance,
            "in_sample_mean": candidate.mean_value,
            "complexity": candidate.complexity,
            "symbolically_proven": candidate.symbolically_proven,
            "discovery_mode": mode_name(candidate.discovery_mode),
            "oracle_equivalence": match oracle_equivalence {
                Some(true) => "equivalent_within_supported_grammar",
                Some(false) => "not_equivalent_within_supported_grammar",
                None => "unsupported_expression",
            },
            "oracle_equivalent_harmonic_energy": oracle_equivalence,
            "holdout": holdout,
            "step_size_stability": {
                "second_dt": DT / 2.0,
                "absolute_variance_difference": absolute_step_size_difference,
                "relative_variance_difference": relative_step_size_difference,
                "absolute_tolerance": STEP_SIZE_STABILITY_ABSOLUTE_TOLERANCE,
                "relative_tolerance": STEP_SIZE_STABILITY_RELATIVE_TOLERANCE,
                "passed": step_size_stable,
                "assessment": stability_holdout
            },
            "receipt": receipt_value,
            "stability_receipt": stability_receipt_value
        }));
    }

    Ok(json!({
        "record_schema_version": PILOT_SCHEMA_VERSION,
        "record_type": "ramanujan_protocol_v2_pilot_run",
        "protocol_id": PILOT_PROTOCOL_ID,
        "source_revision": source_revision,
        "toolchain": {
            "rustc": rustc_version,
            "cargo": cargo_version
        },
        "host_parallelism": host_parallelism,
        "replicate_order_index": replicate_order_index,
        "preregistration_blake3": protocol_blake3,
        "cargo_lock_blake3": cargo_lock_blake3,
        "workspace_manifest_blake3": workspace_manifest_blake3,
        "core_manifest_blake3": core_manifest_blake3,
        "host_os": env::consts::OS,
        "host_arch": env::consts::ARCH,
        "system": {
            "system_id": SYSTEM_ID,
            "system_definition_canonical": SYSTEM_DEFINITION,
            "system_definition_blake3": blake3::hash(SYSTEM_DEFINITION.as_bytes()).to_hex().to_string(),
            "oracle": "degree-2 polynomial coefficient map: a*(x^2+v^2)+b; |a|>1e-10, relative coefficient tolerance 1e-10, coefficients below 1e-12 normalized to zero; unsupported expressions fail closed"
        },
        "condition": mode_name(mode),
        "discovery_seed": seed,
        "holdout_seed": holdout_seed,
        "paired_replicate_id": seed,
        "discovery_holdout_seeds_distinct": seed != holdout_seed,
        "search_config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "max_depth": config.max_depth,
            "max_complexity": config.max_complexity,
            "lambda": config.lambda,
            "tournament_size": config.tournament_size,
            "mutation_rate": config.mutation_rate,
            "diverse_trajectory_count": config.diverse_trajectory_count,
            "use_lie_fitness": config.use_lie_fitness,
            "exclude_trig": config.exclude_trig
        },
        "integrator": {
            "method": "fixed-step RK4",
            "t_max": T_MAX,
            "dt": DT,
            "stability_dt": DT / 2.0,
            "step_size_stability_relative_tolerance": STEP_SIZE_STABILITY_RELATIVE_TOLERANCE,
            "requested_holdout_trajectories_per_candidate": HOLDOUT_TRAJECTORIES,
            "holdout_variance_threshold": HOLDOUT_THRESHOLD
        },
        "discovery_work": discovery.metrics,
        "holdout_work_across_returned_candidates": {
            "attempted_integration_steps": holdout_attempted_steps,
            "completed_integration_steps": holdout_completed_steps,
            "integration_rhs_evaluations": holdout_integration_rhs_evaluations,
            "scoring_rhs_evaluations": holdout_scoring_rhs_evaluations,
            "rhs_evaluations": holdout_rhs_evaluations,
            "state_updates": holdout_state_updates
        },
        "outcome": {
            "returned_candidates": discovery.candidates.len(),
            "holdout_passing_candidates": holdout_passing_candidates,
            "stable_holdout_candidates": stable_holdout_candidates,
            "holdout_stability_passes": holdout_stability_passes,
            "oracle_supported_candidates": oracle_supported_candidates,
            "oracle_unsupported_candidates": oracle_unsupported_candidates,
            "oracle_equivalent_candidates": oracle_equivalent_candidates,
            "oracle_and_holdout_candidates": oracle_and_holdout_candidates,
            "oracle_and_stable_holdout_candidates": oracle_and_stable_holdout_candidates,
            "holdout_false_discoveries_within_supported_oracle": holdout_false_discoveries,
            "stable_holdout_false_discoveries_within_supported_oracle": stable_holdout_false_discoveries,
            "oracle_unsupported_holdout_candidates": oracle_unsupported_holdout_candidates,
            "oracle_unsupported_stable_holdout_candidates": oracle_unsupported_stable_holdout_candidates,
            "run_recovery_success": oracle_and_stable_holdout_candidates > 0,
            "receipt_errors": receipt_errors
        },
        "candidates": candidate_records,
        "interpretation_limits": [
            "Pilot is a single-system instrument check, not the full preregistered evaluation.",
            "The polynomial oracle is exact within its supported degree-2 grammar; unsupported expression forms fail oracle classification.",
            "A holdout pass is numerical generalization evidence, not formal proof.",
            "Search wall time is recorded but should not be compared alone; exact work counters are primary for compute accounting."
        ]
    }))
}

fn append_hashed_record(output: &mut BufWriter<std::fs::File>, mut record: Value) -> Result<String, Box<dyn Error>> {
    if !record.is_object() {
        return Err("pilot run record unexpectedly not a JSON object".into());
    }
    record
        .as_object_mut()
        .expect("record object shape checked")
        .remove("record_blake3");
    let digest = blake3::hash(&serde_json::to_vec(&record)?).to_hex().to_string();
    record
        .as_object_mut()
        .expect("record object shape checked")
        .insert("record_blake3".into(), Value::String(digest.clone()));
    serde_json::to_writer(&mut *output, &record)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(digest)
}

fn parse_positive_usize(value: Option<&String>, default: usize, label: &str) -> Result<usize, Box<dyn Error>> {
    let Some(value) = value else {
        return Ok(default);
    };
    let parsed = value.parse::<usize>()?;
    if parsed == 0 {
        return Err(format!("{label} must be positive").into());
    }
    Ok(parsed)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        return Err(
            "usage: ramanujan_protocol_v2_pilot <new-output.jsonl> [replicates=10] [first_seed=42]"
                .into(),
        );
    }
    let output_path = Path::new(&args[0]);
    let replicates = parse_positive_usize(args.get(1), 10, "replicates")?;
    if replicates > MAX_REPLICATES {
        return Err(format!("replicates may not exceed {MAX_REPLICATES}").into());
    }
    let first_seed = args.get(2).map(|value| value.parse::<u64>()).transpose()?.unwrap_or(42);
    let revision = source_revision()?;
    let rustc_version = command_text("rustc", &["--version"])?;
    let cargo_version = command_text("cargo", &["--version"])?;
    let host_parallelism = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let protocol_blake3 = file_blake3("docs/planning/ramanujan-protocol-v2-evaluation-protocol.md")?;
    let cargo_lock_blake3 = file_blake3("Cargo.lock")?;
    let workspace_manifest_blake3 = file_blake3("Cargo.toml")?;
    let core_manifest_blake3 = file_blake3("crates/core/symthaea-core/Cargo.toml")?;

    if let Some(parent) = output_path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    if output_path.exists() {
        return Err(format!(
            "refusing to overwrite evidence file {}",
            output_path.display()
        ).into());
    }
    let file = OpenOptions::new().write(true).create_new(true).open(output_path)
        .map_err(|error| format!("cannot create evidence file {}: {error}", output_path.display()))?;
    let mut output = BufWriter::new(file);

    let canonical_conditions = canonical_pilot_conditions();
    let mut run_record_digests = Vec::with_capacity(replicates * canonical_conditions.len());
    let mut replicate_orders = Vec::with_capacity(replicates);
    for replicate in 0..replicates {
        let seed = first_seed.wrapping_add(replicate as u64);
        let holdout_seed = seed.wrapping_add(HOLDOUT_SEED_OFFSET);
        let run_order = run_order_for_replicate(replicate);
        replicate_orders.push(json!({
            "replicate_id": seed,
            "ordered_conditions": run_order.map(mode_name)
        }));
        for (replicate_order_index, mode) in run_order.into_iter().enumerate() {
            let config = RegressorConfig {
                seed,
                population_size: 80,
                generations: 8,
                max_depth: 4,
                max_complexity: 18,
                diverse_trajectory_count: 3,
                use_lie_fitness: true,
                ..RegressorConfig::for_autonomous_discovery()
            };
            let record = build_run_record(
                mode,
                seed,
                holdout_seed,
                &config,
                &revision,
                &rustc_version,
                &cargo_version,
                host_parallelism,
                replicate_order_index,
                &protocol_blake3,
                &cargo_lock_blake3,
                &workspace_manifest_blake3,
                &core_manifest_blake3,
            )?;
            run_record_digests.push(append_hashed_record(&mut output, record)?);
        }
    }

    let batch_manifest = json!({
        "record_schema_version": PILOT_SCHEMA_VERSION,
        "record_type": "ramanujan_protocol_v2_batch_manifest",
        "protocol_id": PILOT_PROTOCOL_ID,
        "source_revision": revision,
        "preregistration_blake3": protocol_blake3,
        "cargo_lock_blake3": cargo_lock_blake3,
        "workspace_manifest_blake3": workspace_manifest_blake3,
        "core_manifest_blake3": core_manifest_blake3,
        "toolchain": {
            "rustc": rustc_version,
            "cargo": cargo_version
        },
        "host_parallelism": host_parallelism,
        "replicates": replicates,
        "first_discovery_seed": first_seed,
        "holdout_seed_offset": HOLDOUT_SEED_OFFSET,
        "conditions": canonical_conditions.map(mode_name),
        "replicate_orders": replicate_orders,
        "run_record_count": run_record_digests.len(),
        "run_record_blake3s": run_record_digests
    });
    let batch_digest = append_hashed_record(&mut output, batch_manifest)?;

    eprintln!(
        "wrote {} immutable JSONL run records plus batch manifest ({batch_digest}) to {}",
        replicates * conditions.len(),
        output_path.display()
    );
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    fn square(variable: &str) -> Expr {
        Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var(variable.to_string())),
            Box::new(Expr::Const(2.0)),
        )
    }

    fn candidate(formula: Expr) -> AutonomousInvariant {
        AutonomousInvariant {
            formula_str: format!("{formula}"),
            formula,
            variance: 0.0,
            mean_value: 1.0,
            complexity: 3,
            symbolically_proven: false,
            discovery_mode: AutonomousDiscoveryMode::Cold,
        }
    }

    #[test]
    fn run_order_is_a_counterbalanced_three_order_latin_square() {
        let orders = [
            run_order_for_replicate(0),
            run_order_for_replicate(1),
            run_order_for_replicate(2),
        ];
        assert_eq!(orders[0], [
            AutonomousDiscoveryMode::Cold,
            AutonomousDiscoveryMode::Primed,
            AutonomousDiscoveryMode::RandomBaseline,
        ]);
        assert_eq!(orders[1], [
            AutonomousDiscoveryMode::Primed,
            AutonomousDiscoveryMode::RandomBaseline,
            AutonomousDiscoveryMode::Cold,
        ]);
        assert_eq!(orders[2], [
            AutonomousDiscoveryMode::RandomBaseline,
            AutonomousDiscoveryMode::Cold,
            AutonomousDiscoveryMode::Primed,
        ]);
        for mode in canonical_pilot_conditions() {
            for position in 0..3 {
                assert_eq!(
                    orders.iter().filter(|order| order[position] == mode).count(),
                    1,
                    "each condition must occupy each position once over three replicates"
                );
            }
        }
        assert_eq!(run_order_for_replicate(3), orders[0]);
    }

    #[test]
    fn oracle_accepts_scaled_and_offset_harmonic_energy() {
        let energy = Expr::BinOp(
            BinOp::Add,
            Box::new(square("x")),
            Box::new(square("v")),
        );
        let scaled_and_offset = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Const(7.0)),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Const(0.5)),
                Box::new(energy),
            )),
        );
        assert_eq!(classify_harmonic_energy_oracle(&candidate(scaled_and_offset)), Some(true));
    }

    #[test]
    fn oracle_rejects_anisotropic_quadratic_and_cross_term() {
        let anisotropic = Expr::BinOp(
            BinOp::Add,
            Box::new(square("x")),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Const(2.0)),
                Box::new(square("v")),
            )),
        );
        let cross_term = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("v".into())),
        );
        assert_eq!(classify_harmonic_energy_oracle(&candidate(anisotropic)), Some(false));
        assert_eq!(classify_harmonic_energy_oracle(&candidate(cross_term)), Some(false));
    }

    #[test]
    fn step_size_stability_uses_explicit_absolute_tolerance_near_zero() {
        let near_zero = assess_step_size_stability(true, true, 2e-15, 8e-15);
        assert!(near_zero.passed);
        assert!((near_zero.absolute_variance_difference - 6e-15).abs() < 1e-20);
    }

    #[test]
    fn step_size_stability_enforces_relative_tolerance_away_from_zero() {
        let stable = assess_step_size_stability(true, true, 5e-3, 5.3e-3);
        assert!(stable.passed);

        let unstable = assess_step_size_stability(true, true, 5e-3, 6e-3);
        assert!(!unstable.passed);
    }

    #[test]
    fn step_size_stability_never_passes_if_either_holdout_fails() {
        assert!(!assess_step_size_stability(false, true, 0.0, 0.0).passed);
        assert!(!assess_step_size_stability(true, false, 1e-14, 1e-14).passed);
        assert!(!assess_step_size_stability(true, true, f64::MAX, f64::MAX).passed);
    }

    #[test]
    fn oracle_rejects_unsupported_non_polynomial_form() {
        let expression = Expr::BinOp(
            BinOp::Add,
            Box::new(square("x")),
            Box::new(Expr::Func(
                symthaea_core::hdc::conjecture_engine::UnaryFn::Sin,
                Box::new(Expr::Var("v".into())),
            )),
        );
        assert_eq!(classify_harmonic_energy_oracle(&candidate(expression)), None);
    }
}
