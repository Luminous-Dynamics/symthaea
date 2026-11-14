use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[path = "../../../../../crates/domains/symthaea-frontier-physics/src/u1_mps.rs"]
mod u1_mps;

use u1_mps::{
    ComplexU1Mps, U1TeBDHamiltonianTerm, U1TeBDStrangStepper,
    U1TwoSiteHamiltonian, U1TwoSiteTruncationPolicy,
};

const TOLERANCE: f64 = 1.0e-12;

fn number(value: &Value, key: &str) -> Result<f64, String> {
    value[key]
        .as_f64()
        .ok_or_else(|| format!("missing numeric field {key}"))
}

fn integer(value: &Value, key: &str) -> Result<usize, String> {
    value[key]
        .as_u64()
        .map(|value| value as usize)
        .ok_or_else(|| format!("missing integer field {key}"))
}

fn complex_pair(value: &Value) -> Result<Complex64, String> {
    Ok(Complex64::new(number(value, "re")?, number(value, "im")?))
}

fn load_case(fixture_path: &PathBuf, case_id: &str) -> Result<(Value, Value), String> {
    let source = fs::read_to_string(fixture_path)
        .map_err(|error| format!("read fixture {}: {error}", fixture_path.display()))?;
    let fixture: Value = serde_json::from_str(&source)
        .map_err(|error| format!("parse fixture JSON: {error}"))?;
    let cases = fixture["cases"]
        .as_array()
        .ok_or_else(|| "fixture has no cases array".to_string())?;
    let case = cases
        .iter()
        .find(|case| case["id"].as_str() == Some(case_id))
        .cloned()
        .ok_or_else(|| format!("unknown benchmark case {case_id}"))?;
    Ok((fixture, case))
}

fn initial_bits(case: &Value) -> Result<Vec<bool>, String> {
    let values = case["initial_bits"]
        .as_array()
        .ok_or_else(|| "case has no initial_bits array".to_string())?;
    values
        .iter()
        .map(|value| match value.as_u64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err("initial_bits values must be 0 or 1".to_string()),
        })
        .collect()
}

fn make_terms(case: &Value) -> Result<Vec<U1TeBDHamiltonianTerm>, String> {
    let bonds = case["bonds"]
        .as_array()
        .ok_or_else(|| "case has no bonds array".to_string())?;
    let mut terms = Vec::with_capacity(bonds.len());
    for bond in bonds {
        let hamiltonian = U1TwoSiteHamiltonian::new(
            number(bond, "mu_left")?,
            number(bond, "mu_right")?,
            number(bond, "density_interaction")?,
            complex_pair(&bond["hopping"])?,
        )
        .map_err(|error| error.to_string())?;
        terms.push(U1TeBDHamiltonianTerm {
            bond_index: integer(bond, "bond_index")?,
            hamiltonian,
        });
    }
    terms.sort_by_key(|term| term.bond_index);
    Ok(terms)
}

fn build_policy(max_bond: usize, cutoff: f64) -> U1TwoSiteTruncationPolicy {
    U1TwoSiteTruncationPolicy {
        max_bond_dimension: max_bond,
        relative_singular_value_cutoff: cutoff,
        tolerance: TOLERANCE,
    }
}

fn initial_dense(bits: &[bool]) -> DVector<Complex64> {
    let mut basis = 0_usize;
    for bit in bits {
        basis = (basis << 1) | usize::from(*bit);
    }
    let mut state =
        DVector::from_element(1_usize << bits.len(), Complex64::new(0.0, 0.0));
    state[basis] = Complex64::new(1.0, 0.0);
    state
}

fn apply_dense_gate(
    state: &mut DVector<Complex64>,
    gate: &[[Complex64; 4]; 4],
    site_count: usize,
    bond: usize,
) {
    let dimension = state.len();
    let left_shift = site_count - 1 - bond;
    let right_shift = site_count - 2 - bond;
    let mut output =
        DVector::from_element(dimension, Complex64::new(0.0, 0.0));

    for basis in 0..dimension {
        let left_bit = (basis >> left_shift) & 1;
        let right_bit = (basis >> right_shift) & 1;
        let local_input = 2 * left_bit + right_bit;
        let base = basis & !(1_usize << left_shift) & !(1_usize << right_shift);
        for local_output in 0..4 {
            let output_left = (local_output >> 1) & 1;
            let output_right = local_output & 1;
            let target = base | (output_left << left_shift) | (output_right << right_shift);
            output[target] += gate[local_output][local_input] * state[basis];
        }
    }
    *state = output;
}

fn dense_discrete_oracle(
    bits: &[bool],
    terms: &[U1TeBDHamiltonianTerm],
    time_step: f64,
    steps: usize,
) -> Result<DVector<Complex64>, String> {
    let mut state = initial_dense(bits);
    for _ in 0..steps {
        for (parity, scale) in [(0_usize, 0.5_f64), (1, 1.0), (0, 0.5)] {
            for term in terms
                .iter()
                .filter(|term| term.bond_index % 2 == parity)
            {
                let gate = term
                    .hamiltonian
                    .to_gate(time_step * scale, TOLERANCE)
                    .map_err(|error| error.to_string())?;
                apply_dense_gate(&mut state, gate.matrix(), bits.len(), term.bond_index);
            }
        }
    }
    Ok(state)
}

fn local_hamiltonian(term: &U1TwoSiteHamiltonian) -> [[Complex64; 4]; 4] {
    let mut h = [[Complex64::new(0.0, 0.0); 4]; 4];
    h[1][1] = Complex64::new(term.mu_right, 0.0);
    h[2][2] = Complex64::new(term.mu_left, 0.0);
    h[3][3] = Complex64::new(
        term.mu_left + term.mu_right + term.density_interaction,
        0.0,
    );
    h[1][2] = term.hopping;
    h[2][1] = term.hopping.conj();
    h
}

fn dense_hamiltonian(
    site_count: usize,
    terms: &[U1TeBDHamiltonianTerm],
) -> DMatrix<Complex64> {
    let dimension = 1_usize << site_count;
    let mut full = DMatrix::from_element(
        dimension,
        dimension,
        Complex64::new(0.0, 0.0),
    );
    for term in terms {
        let local = local_hamiltonian(&term.hamiltonian);
        let left_shift = site_count - 1 - term.bond_index;
        let right_shift = site_count - 2 - term.bond_index;
        for basis in 0..dimension {
            let left_bit = (basis >> left_shift) & 1;
            let right_bit = (basis >> right_shift) & 1;
            let local_input = 2 * left_bit + right_bit;
            let base = basis & !(1_usize << left_shift) & !(1_usize << right_shift);
            for local_output in 0..4 {
                let output_left = (local_output >> 1) & 1;
                let output_right = local_output & 1;
                let target = base | (output_left << left_shift) | (output_right << right_shift);
                full[(target, basis)] += local[local_output][local_input];
            }
        }
    }
    full
}

/// Small-matrix scaling-and-squaring Taylor exponential for the dense oracle.
fn matrix_exponential(matrix: &DMatrix<Complex64>) -> Result<DMatrix<Complex64>, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err("matrix exponential requires a square matrix".into());
    }
    let n = matrix.nrows();
    let mut one_norm: f64 = 0.0;
    for column in 0..n {
        let column_sum = (0..n)
            .map(|row| matrix[(row, column)].norm())
            .sum::<f64>();
        one_norm = one_norm.max(column_sum);
    }
    if !one_norm.is_finite() {
        return Err("non-finite dense Hamiltonian norm".into());
    }

    let squarings = if one_norm <= 0.5 {
        0
    } else {
        (one_norm / 0.5).log2().ceil().max(0.0) as usize
    };
    if squarings > 1024 {
        return Err("dense oracle matrix exponential scaling is excessive".into());
    }
    let scaled = matrix / 2.0_f64.powi(squarings as i32);
    let identity = DMatrix::<Complex64>::identity(n, n);
    let mut sum = identity.clone();
    let mut term = identity;

    for order in 1..=120 {
        term = &term * &scaled;
        term *= Complex64::new(1.0 / order as f64, 0.0);
        sum += &term;
        if term.norm() <= 2.0e-16 * sum.norm().max(1.0) {
            break;
        }
        if order == 120 {
            return Err("dense matrix exponential Taylor series did not converge".into());
        }
    }
    for _ in 0..squarings {
        sum = &sum * &sum;
    }
    Ok(sum)
}

fn continuous_time_oracle(
    bits: &[bool],
    terms: &[U1TeBDHamiltonianTerm],
    total_time: f64,
) -> Result<DVector<Complex64>, String> {
    let hamiltonian = dense_hamiltonian(bits.len(), terms);
    let generator = hamiltonian * Complex64::new(0.0, -total_time);
    Ok(matrix_exponential(&generator)? * initial_dense(bits))
}

fn state_metrics(
    actual: &DVector<Complex64>,
    reference: &DVector<Complex64>,
) -> Value {
    let l2 = (actual - reference).norm();
    let actual_norm = actual.norm_squared();
    let reference_norm = reference.norm_squared();
    let overlap = reference.dotc(actual);
    let fidelity =
        overlap.norm_sqr() / (actual_norm * reference_norm).max(f64::MIN_POSITIVE);
    json!({
        "state_l2_error": l2,
        "fidelity": fidelity,
        "actual_norm_squared": actual_norm,
        "reference_norm_squared": reference_norm
    })
}

fn charge_metrics(state: &DVector<Complex64>, target_charge: usize) -> Value {
    let mut leakage = 0.0;
    let mut mean = 0.0;
    let mut variance = 0.0;
    for (basis, amplitude) in state.iter().enumerate() {
        let charge = basis.count_ones() as usize;
        let probability = amplitude.norm_sqr();
        mean += charge as f64 * probability;
        if charge != target_charge {
            leakage += probability;
        }
    }
    for (basis, amplitude) in state.iter().enumerate() {
        let charge = basis.count_ones() as f64;
        variance += (charge - mean).powi(2) * amplitude.norm_sqr();
    }
    json!({
        "total_charge_target": target_charge,
        "total_charge_expectation": mean,
        "total_charge_variance": variance,
        "charge_leakage_probability": leakage
    })
}

fn gate_quality_maxima(
    terms: &[U1TeBDHamiltonianTerm],
    time_step: f64,
    steps: usize,
    tolerance: f64,
) -> Result<(f64, f64), String> {
    let mut max_unitarity: f64 = 0.0;
    let mut max_charge: f64 = 0.0;
    for _ in 0..steps {
        for (parity, scale) in [(0_usize, 0.5_f64), (1, 1.0), (0, 0.5)] {
            for term in terms.iter().filter(|term| term.bond_index % 2 == parity) {
                let gate = term
                    .hamiltonian
                    .to_gate(time_step * scale, tolerance)
                    .map_err(|error| error.to_string())?;
                max_unitarity = max_unitarity.max(gate.unitarity_residual());
                max_charge = max_charge.max(gate.charge_conservation_residual());
            }
        }
    }
    Ok((max_unitarity, max_charge))
}

fn run_once(
    case: &Value,
    bits: &[bool],
    terms: &[U1TeBDHamiltonianTerm],
    stepper: &U1TeBDStrangStepper,
    policy: U1TwoSiteTruncationPolicy,
) -> Result<(Value, Duration), String> {
    let site_count = bits.len();
    let total_charge = integer(case, "total_charge")?;
    let time_step = number(case, "time_step")?;
    let steps = integer(case, "steps")?;
    let initial = ComplexU1Mps::product_state(bits)
        .map_err(|error| error.to_string())?;
    let input_norm_squared = initial
        .norm_squared(TOLERANCE)
        .map_err(|error| error.to_string())?;
    // Audit gate quality outside the timed region: runtime measures the
    // production Strang stepper, not a second copy of every local exponential.
    let (max_gate_unitarity_residual, max_gate_charge_residual) =
        gate_quality_maxima(terms, time_step, steps, policy.tolerance)?;

    let started = Instant::now();
    let mut state = initial;
    let mut gate_count = 0_usize;
    let mut local_l2_error_bound = 0.0;
    let mut global_l2_error_bound = Some(0.0_f64);
    let mut global_l2_bound_qualified = true;
    let mut max_gauge_residual: f64 = 0.0;
    let mut max_selection_rule_residual: f64 = 0.0;
    let mut discarded_weight = 0.0;
    let mut cutoff_discarded_weight = 0.0;
    let mut bond_cap_discarded_weight = 0.0;
    let mut stage_receipts = Vec::with_capacity(steps * 3);

    for _ in 0..steps {
        let result = stepper
            .step(&state, time_step, policy)
            .map_err(|error| error.to_string())?;

        for stage in &result.stages {
            gate_count += stage.bond_indices.len();
            for gauge in &stage.canonical_gauge_receipts {
                max_gauge_residual = max_gauge_residual
                    .max(gauge.maximum_left_isometry_residual)
                    .max(gauge.maximum_right_isometry_residual)
                    .max(gauge.factorization_max_residual)
                    .max(gauge.relative_norm_drift);
            }
            for truncation in &stage.gate_truncations {
                discarded_weight += truncation.discarded_weight;
                cutoff_discarded_weight += truncation.cutoff_discarded_weight;
                bond_cap_discarded_weight += truncation.bond_cap_discarded_weight;
            }
            local_l2_error_bound += stage.local_l2_error_bound;
            if stage.global_l2_bound_qualified {
                global_l2_error_bound = global_l2_error_bound
                    .map(|current| current + stage.global_l2_error_bound.unwrap_or(0.0));
            } else {
                global_l2_bound_qualified = false;
                global_l2_error_bound = None;
            }
            stage_receipts.push(json!({
                "parity": stage.parity,
                "time_step_scale": stage.time_step_scale,
                "bond_indices": stage.bond_indices,
                "local_l2_error_bound": stage.local_l2_error_bound,
                "global_l2_error_bound": stage.global_l2_error_bound,
                "global_l2_bound_qualified": stage.global_l2_bound_qualified
            }));
        }

        max_selection_rule_residual =
            max_selection_rule_residual.max(result.state.selection_rule_residual());
        state = result.state;
    }
    let runtime = started.elapsed();

    let output_norm_squared = state
        .norm_squared(TOLERANCE)
        .map_err(|error| error.to_string())?;
    let observed_max_bond = state.maximum_bond_dimension();
    let dense_actual = if site_count <= 12 {
        Some(DVector::from_vec(
            state.to_dense(12).map_err(|error| error.to_string())?,
        ))
    } else {
        None
    };
    let discrete_metrics = if let Some(actual) = &dense_actual {
        let reference = dense_discrete_oracle(bits, terms, time_step, steps)?;
        Some(state_metrics(actual, &reference))
    } else {
        None
    };
    let continuous_metrics = if site_count <= 10 {
        let actual = dense_actual.as_ref().expect("dense state exists");
        let reference =
            continuous_time_oracle(bits, terms, time_step * steps as f64)?;
        Some(state_metrics(actual, &reference))
    } else {
        None
    };
    let charge = dense_actual
        .as_ref()
        .map(|actual| charge_metrics(actual, total_charge));

    let result = json!({
        "implementation": "symthaea-frontier-physics::U1TeBDStrangStepper",
        "source_commit": option_env!("GITHUB_SHA").unwrap_or("local-unpinned"),
        "crate_version": env!("CARGO_PKG_VERSION"),
        "case_id": case["id"],
        "site_count": site_count,
        "total_charge": total_charge,
        "time_step": time_step,
        "steps": steps,
        "strang_order": ["even_dt_over_2", "odd_dt", "even_dt_over_2"],
        "gate_count": gate_count,
        "cutoff": policy.relative_singular_value_cutoff,
        "cutoff_mode": "relative_singular_value_global_rank",
        "max_bond": policy.max_bond_dimension,
        "renormalization": "disabled",
        "runtime_seconds": runtime.as_secs_f64(),
        "gate_throughput_per_second": if runtime.as_secs_f64() > 0.0 {
            gate_count as f64 / runtime.as_secs_f64()
        } else {
            0.0
        },
        "input_norm_squared": input_norm_squared,
        "output_norm_squared": output_norm_squared,
        "norm_drift": (output_norm_squared - input_norm_squared).abs()
            / input_norm_squared.abs().max(1.0),
        "observed_max_bond": observed_max_bond,
        "output_total_charge": state.total_charge(),
        "selection_rule_residual": max_selection_rule_residual,
        "canonical_gauge_max_residual": max_gauge_residual,
        "gate_unitarity_max_residual": max_gate_unitarity_residual,
        "gate_charge_conservation_max_residual": max_gate_charge_residual,
        "accumulated_discarded_weight": discarded_weight,
        "cutoff_discarded_weight": cutoff_discarded_weight,
        "bond_cap_discarded_weight": bond_cap_discarded_weight,
        "cumulative_local_l2_error_bound": local_l2_error_bound,
        "cumulative_global_l2_error_bound": global_l2_error_bound,
        "global_state_error_bound_qualified": global_l2_bound_qualified,
        "dense_discrete_oracle": discrete_metrics,
        "continuous_time_oracle": continuous_metrics,
        "charge_metrics": charge,
        "stage_receipts": stage_receipts
    });
    Ok((result, runtime))
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

fn parse_args() -> Result<(PathBuf, String, String, Option<f64>, Option<usize>, usize, usize, Option<PathBuf>), String> {
    let mut fixture = PathBuf::from("benchmarks/u1_tebd/fixtures/u1_tebd_cases_v1.json");
    let mut case_id = "u1-two-site-hopping-known-answer".to_string();
    let mut track = "fixed_bond".to_string();
    let mut cutoff = None;
    let mut max_bond = None;
    let mut warmups = 5_usize;
    let mut repeats = 20_usize;
    let mut output = None;
    let mut arguments = env::args().skip(1);

    while let Some(argument) = arguments.next() {
        let (key, inline) = argument.split_once('=').unwrap_or((&argument, ""));
        let value = if !inline.is_empty() {
            inline.to_string()
        } else if key == "--help" {
            println!("symthaea-u1-tebd-benchmark [--fixture PATH] [--case ID] [--track fixed_bond|matched_error] [--cutoff FLOAT] [--max-bond N] [--warmups N] [--repeats N] [--output PATH]");
            std::process::exit(0);
        } else {
            arguments.next().ok_or_else(|| format!("missing value for {key}"))?
        };
        match key {
            "--fixture" => fixture = PathBuf::from(value),
            "--case" => case_id = value,
            "--track" => track = value,
            "--cutoff" => cutoff = Some(value.parse().map_err(|error| format!("cutoff: {error}"))?),
            "--max-bond" => max_bond = Some(value.parse().map_err(|error| format!("max-bond: {error}"))?),
            "--warmups" => warmups = value.parse().map_err(|error| format!("warmups: {error}"))?,
            "--repeats" => repeats = value.parse().map_err(|error| format!("repeats: {error}"))?,
            "--output" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {key}")),
        }
    }
    Ok((fixture, case_id, track, cutoff, max_bond, warmups, repeats, output))
}

fn main() -> Result<(), String> {
    let (fixture_path, case_id, track, selected_cutoff, selected_bond, warmups, repeats, output_path) =
        parse_args()?;
    let (fixture, case) = load_case(&fixture_path, &case_id)?;
    let site_count = integer(&case, "site_count")?;
    let steps = integer(&case, "steps")?;
    let total_charge = integer(&case, "total_charge")?;
    let time_step = number(&case, "time_step")?;
    let bits = initial_bits(&case)?;
    if bits.len() != site_count {
        return Err("initial bit length differs from site_count".into());
    }
    let terms = make_terms(&case)?;

    let (cutoff, max_bond) = match track.as_str() {
        "fixed_bond" => (
            selected_cutoff.unwrap_or(0.0),
            selected_bond.unwrap_or_else(|| {
                integer(&case, "max_bond_dimension").unwrap_or(16)
            }),
        ),
        "matched_error" => (selected_cutoff.unwrap_or(1.0e-8), selected_bond.unwrap_or(256)),
        _ => return Err("track must be fixed_bond or matched_error".into()),
    };
    let policy = build_policy(max_bond, cutoff);
    let stepper = U1TeBDStrangStepper::new(site_count, terms.clone())
        .map_err(|error| error.to_string())?;

    for _ in 0..warmups {
        let _ = run_once(&case, &bits, &terms, &stepper, policy)?;
    }
    let records = (0..repeats.max(1))
        .map(|_| run_once(&case, &bits, &terms, &stepper, policy))
        .collect::<Result<Vec<_>, _>>()?;
    let times = records
        .iter()
        .map(|(_, runtime)| runtime.as_secs_f64())
        .collect::<Vec<_>>();
    let median = percentile(&times, 0.5);
    let p95 = percentile(&times, 0.95);
    let first = &records[0].0;
    let output = json!({
        "schema": "luminous.u1_tebd_benchmark_result.v1",
        "fixture_schema": fixture["schema"],
        "track": track,
        "case_id": case_id,
        "warmups": warmups,
        "repeats": records.len(),
        "runtime_median_seconds": median,
        "runtime_p95_seconds": p95,
        "host": {
            "os": env::consts::OS,
            "arch": env::consts::ARCH,
            "available_parallelism": std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1)
        },
        "records": records.iter().map(|(record, _)| record).collect::<Vec<_>>(),
        "qualification_summary": {
            "global_state_error_bound_qualified": first["global_state_error_bound_qualified"],
            "dense_discrete_state_l2_error": first["dense_discrete_oracle"]["state_l2_error"],
            "continuous_time_state_l2_error": first["continuous_time_oracle"]["state_l2_error"],
            "observed_max_bond": first["observed_max_bond"],
            "total_charge_target": total_charge,
            "site_count": site_count,
            "step_count": steps,
            "time_step": time_step
        }
    });

    let serialized = serde_json::to_string_pretty(&output)
        .map_err(|error| error.to_string())?;
    if let Some(output_path) = output_path {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        fs::write(output_path, format!("{serialized}\n"))
            .map_err(|error| error.to_string())?;
    }
    println!("{serialized}");
    Ok(())
}
