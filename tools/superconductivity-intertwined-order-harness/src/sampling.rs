use super::{Complex, IntertwinedModel};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{Value, json};

#[derive(Clone, Copy, Default)]
pub(super) struct SamplingOverrides {
    pub warmup_sweeps: Option<usize>,
    pub measurement_sweeps: Option<usize>,
    pub thinning: Option<usize>,
    pub proposal_scale: Option<f64>,
    pub independent_chains: Option<usize>,
    pub random_seed: Option<u64>,
}

#[derive(Clone, Copy)]
struct SamplingConfig {
    random_seed: u64,
    warmup_sweeps: usize,
    measurement_sweeps: usize,
    thinning: usize,
    proposal_scale: f64,
    independent_chains: usize,
}

fn required_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("sampling.{key} must be an unsigned integer"))
}

fn required_usize(value: &Value, key: &str) -> Result<usize, String> {
    usize::try_from(required_u64(value, key)?)
        .map_err(|_| format!("sampling.{key} exceeds usize"))
}

fn required_positive_f64(value: &Value, key: &str) -> Result<f64, String> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite() && *number > 0.0)
        .ok_or_else(|| format!("sampling.{key} must be positive and finite"))
}

impl SamplingConfig {
    fn from_fixture(
        fixture_bytes: &[u8],
        overrides: SamplingOverrides,
    ) -> Result<Self, String> {
        let fixture: Value = serde_json::from_slice(fixture_bytes)
            .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
        let sampling = fixture
            .get("sampling")
            .ok_or_else(|| "fixture missing sampling protocol".to_owned())?;
        if sampling.get("algorithm").and_then(Value::as_str)
            != Some("single_component_site_metropolis")
        {
            return Err("unsupported multicomponent sampling algorithm".to_owned());
        }

        let random_seed = overrides
            .random_seed
            .unwrap_or(required_u64(sampling, "random_seed")?);
        let warmup_sweeps = overrides
            .warmup_sweeps
            .unwrap_or(required_usize(sampling, "warmup_sweeps")?);
        let measurement_sweeps = overrides
            .measurement_sweeps
            .unwrap_or(required_usize(sampling, "measurement_sweeps")?);
        let thinning = overrides
            .thinning
            .unwrap_or(required_usize(sampling, "thinning")?);
        let proposal_scale = overrides
            .proposal_scale
            .unwrap_or(required_positive_f64(sampling, "proposal_scale")?);
        let independent_chains = overrides
            .independent_chains
            .unwrap_or(required_usize(sampling, "independent_chains")?);

        if measurement_sweeps < 2 {
            return Err("measurement_sweeps must be at least two".to_owned());
        }
        if thinning == 0 {
            return Err("thinning must be at least one".to_owned());
        }
        if !proposal_scale.is_finite() || proposal_scale <= 0.0 {
            return Err("proposal_scale must be positive and finite".to_owned());
        }
        if independent_chains == 0 || independent_chains > 64 {
            return Err("independent_chains must be between 1 and 64".to_owned());
        }

        Ok(Self {
            random_seed,
            warmup_sweeps,
            measurement_sweeps,
            thinning,
            proposal_scale,
            independent_chains,
        })
    }
}

fn sample_standard_deviation(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Some(variance.sqrt())
}

fn population_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

fn integrated_autocorrelation_time(values: &[f64]) -> Option<f64> {
    if values.len() < 8 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let centered: Vec<f64> = values.iter().map(|value| value - mean).collect();
    let variance = centered.iter().map(|value| value * value).sum::<f64>();
    if variance <= f64::EPSILON {
        return None;
    }
    let autocorrelation = |lag: usize| -> f64 {
        centered
            .iter()
            .zip(centered.iter().skip(lag))
            .map(|(left, right)| left * right)
            .sum::<f64>()
            / variance
    };

    let mut sum_pairs = 0.0;
    let mut lag = 1;
    while lag + 1 < values.len() / 2 {
        let pair = autocorrelation(lag) + autocorrelation(lag + 1);
        if !pair.is_finite() || pair <= 0.0 {
            break;
        }
        sum_pairs += pair;
        lag += 2;
    }
    Some((0.5 + sum_pairs).max(0.5))
}

fn series_summary(values: &[f64]) -> Value {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let standard_deviation = sample_standard_deviation(values);
    let integrated_tau = integrated_autocorrelation_time(values);
    let effective_sample_size =
        integrated_tau.map(|tau| (values.len() as f64 / (2.0 * tau)).min(values.len() as f64));
    let autocorrelation_corrected_se = match (standard_deviation, integrated_tau) {
        (Some(deviation), Some(tau)) => {
            Some(deviation * (2.0 * tau / values.len() as f64).sqrt())
        }
        _ => None,
    };
    json!({
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "integrated_autocorrelation_time_initial_positive_sequence": integrated_tau,
        "effective_sample_size_initial_positive_sequence": effective_sample_size,
        "autocorrelation_corrected_standard_error": autocorrelation_corrected_se,
        "sample_count": values.len(),
        "warning": "Short pilot chains can yield unstable autocorrelation estimates and do not establish equilibration."
    })
}

fn run_sweep(
    model: &IntertwinedModel,
    fields: &mut [Vec<Complex>],
    links: &[f64],
    current_energy: &mut f64,
    rng: &mut StdRng,
    proposal_scale: f64,
    temperature: f64,
) -> Result<(usize, usize), String> {
    let site_count = fields.len();
    let component_count = model.components.len();
    let mut accepted = 0;
    let mut invalid = 0;

    for _ in 0..(site_count * component_count) {
        let site = rng.gen_range(0..site_count);
        let component = rng.gen_range(0..component_count);
        let original = fields[site][component];
        let candidate = Complex {
            re: original.re + rng.gen_range(-proposal_scale..proposal_scale),
            im: original.im + rng.gen_range(-proposal_scale..proposal_scale),
        };
        let delta_energy = match model.site_component_energy_delta(
            fields,
            site,
            component,
            candidate,
            links,
        ) {
            Ok(delta) => delta,
            Err(_) => {
                invalid += 1;
                continue;
            }
        };
        let accept = delta_energy <= 0.0
            || rng.gen_range(0.0..1.0) < (-delta_energy / temperature).exp();
        if accept {
            fields[site][component] = candidate;
            *current_energy += delta_energy;
            accepted += 1;
        }
    }
    Ok((accepted, invalid))
}

fn relative_phase_lock_cosines(
    model: &IntertwinedModel,
    fields: &[Vec<Complex>],
) -> Vec<f64> {
    model
        .phase_locking_terms
        .iter()
        .map(|term| {
            let mut total = 0.0;
            let mut count = 0_usize;
            for site in fields {
                let first = site[term.first];
                let second = site[term.second];
                let denominator = (first.norm_sqr() * second.norm_sqr()).sqrt();
                if denominator > 1.0e-12 {
                    total += first.conj_product_real(second) / denominator;
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                total / count as f64
            }
        })
        .collect()
}

fn relative_phase_lock_sines(
    model: &IntertwinedModel,
    fields: &[Vec<Complex>],
) -> Vec<f64> {
    model
        .phase_locking_terms
        .iter()
        .map(|term| {
            let mut total = 0.0;
            let mut count = 0_usize;
            for site in fields {
                let first = site[term.first];
                let second = site[term.second];
                let denominator = (first.norm_sqr() * second.norm_sqr()).sqrt();
                if denominator > 1.0e-12 {
                    // Im(conj(psi_a) * psi_b) / (|psi_a||psi_b|), invariant
                    // under the single common electromagnetic U(1).
                    total += (first.re * second.im - first.im * second.re) / denominator;
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                total / count as f64
            }
        })
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn helicity_tensor_from_samples(
    current_series: &[Vec<f64>],
    diamagnetic_series: &[Vec<Vec<f64>>],
    temperature: f64,
    excluded_block: Option<(usize, usize)>,
) -> Result<Vec<Vec<f64>>, String> {
    let dimension = current_series.len();
    if dimension == 0 || diamagnetic_series.len() != dimension {
        return Err("helicity series dimensions are inconsistent".to_owned());
    }
    let sample_count = current_series[0].len();
    if sample_count < 2
        || current_series.iter().any(|series| series.len() != sample_count)
        || diamagnetic_series.iter().any(|row| {
            row.len() != dimension
                || row.iter().any(|series| series.len() != sample_count)
        })
    {
        return Err("helicity series have inconsistent or insufficient sample counts".to_owned());
    }
    let included: Vec<usize> = (0..sample_count)
        .filter(|index| {
            excluded_block.is_none_or(|(start, end)| *index < start || *index >= end)
        })
        .collect();
    if included.len() < 2 {
        return Err("helicity jackknife left fewer than two observations".to_owned());
    }

    let current_means: Vec<f64> = current_series
        .iter()
        .map(|series| {
            included.iter().map(|index| series[*index]).sum::<f64>()
                / included.len() as f64
        })
        .collect();
    let mut tensor = vec![vec![0.0; dimension]; dimension];
    for alpha in 0..dimension {
        for beta in 0..dimension {
            let diamagnetic_mean = included
                .iter()
                .map(|index| diamagnetic_series[alpha][beta][*index])
                .sum::<f64>()
                / included.len() as f64;
            let current_product_mean = included
                .iter()
                .map(|index| current_series[alpha][*index] * current_series[beta][*index])
                .sum::<f64>()
                / included.len() as f64;
            let covariance =
                current_product_mean - current_means[alpha] * current_means[beta];
            tensor[alpha][beta] = diamagnetic_mean - covariance / temperature;
        }
    }
    if tensor.iter().flatten().any(|value| !value.is_finite()) {
        return Err("helicity tensor contains non-finite entries".to_owned());
    }
    Ok(tensor)
}

fn block_jackknife_helicity(
    current_series: &[Vec<f64>],
    diamagnetic_series: &[Vec<Vec<f64>>],
    temperature: f64,
) -> Result<(Vec<Vec<f64>>, Option<Vec<Vec<f64>>>, Option<f64>), String> {
    let point_estimate =
        helicity_tensor_from_samples(current_series, diamagnetic_series, temperature, None)?;
    let sample_count = current_series[0].len();
    if sample_count < 20 {
        return Ok((point_estimate, None, None));
    }

    let block_count = 10_usize.min(sample_count / 4);
    if block_count < 4 {
        return Ok((point_estimate, None, None));
    }
    let mut leave_block_out = Vec::with_capacity(block_count);
    let mut leave_block_minimum_eigenvalues = Vec::with_capacity(block_count);
    for block in 0..block_count {
        let start = block * sample_count / block_count;
        let end = (block + 1) * sample_count / block_count;
        let estimate = helicity_tensor_from_samples(
            current_series,
            diamagnetic_series,
            temperature,
            Some((start, end)),
        )?;
        let minimum = super::minimum_symmetric_eigenvalue(&estimate)?;
        leave_block_minimum_eigenvalues.push(minimum);
        leave_block_out.push(estimate);
    }

    let dimension = point_estimate.len();
    let mut jackknife_mean = vec![vec![0.0; dimension]; dimension];
    for estimate in &leave_block_out {
        for alpha in 0..dimension {
            for beta in 0..dimension {
                jackknife_mean[alpha][beta] += estimate[alpha][beta] / block_count as f64;
            }
        }
    }
    let factor = (block_count - 1) as f64 / block_count as f64;
    let mut standard_error = vec![vec![0.0; dimension]; dimension];
    for alpha in 0..dimension {
        for beta in 0..dimension {
            let sum_squared = leave_block_out
                .iter()
                .map(|estimate| (estimate[alpha][beta] - jackknife_mean[alpha][beta]).powi(2))
                .sum::<f64>();
            standard_error[alpha][beta] = (factor * sum_squared).sqrt();
        }
    }

    let eigenvalue_mean =
        leave_block_minimum_eigenvalues.iter().sum::<f64>() / block_count as f64;
    let eigenvalue_variance = factor
        * leave_block_minimum_eigenvalues
            .iter()
            .map(|value| (value - eigenvalue_mean).powi(2))
            .sum::<f64>();
    let eigenvalue_standard_error = eigenvalue_variance.sqrt();
    Ok((
        point_estimate,
        Some(standard_error),
        Some(eigenvalue_standard_error),
    ))
}

fn run_chain(
    model: &IntertwinedModel,
    temperature: f64,
    config: SamplingConfig,
    seed: u64,
) -> Result<Value, String> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("multicomponent Metropolis sampling requires temperature > 0".to_owned());
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let links = model.initial_link_phases();
    let mut fields = model.initial_fields();
    let initial_energy = model.energy_for(&fields, &links)?.total;
    let mut current_energy = initial_energy;

    let mut accepted = 0_usize;
    let mut invalid = 0_usize;
    let mut attempted = 0_usize;

    for _ in 0..config.warmup_sweeps {
        let (sweep_accepted, sweep_invalid) = run_sweep(
            model,
            &mut fields,
            &links,
            &mut current_energy,
            &mut rng,
            config.proposal_scale,
            temperature,
        )?;
        accepted += sweep_accepted;
        invalid += sweep_invalid;
        attempted += fields.len() * model.components.len();
    }

    let component_count = model.components.len();
    let dimension = model.dimension;
    let mut energy_series = Vec::with_capacity(config.measurement_sweeps);
    let mut quadratic_series = Vec::with_capacity(config.measurement_sweeps);
    let mut quartic_series = Vec::with_capacity(config.measurement_sweeps);
    let mut gradient_series = Vec::with_capacity(config.measurement_sweeps);
    let mut amplitude_series: Vec<Vec<f64>> = (0..component_count)
        .map(|_| Vec::with_capacity(config.measurement_sweeps))
        .collect();
    let mut phase_lock_series: Vec<Vec<f64>> = (0..model.phase_locking_terms.len())
        .map(|_| Vec::with_capacity(config.measurement_sweeps))
        .collect();
    let mut phase_lock_sine_series: Vec<Vec<f64>> = (0..model.phase_locking_terms.len())
        .map(|_| Vec::with_capacity(config.measurement_sweeps))
        .collect();
    let mut current_series: Vec<Vec<f64>> = (0..dimension)
        .map(|_| Vec::with_capacity(config.measurement_sweeps))
        .collect();
    let mut diamagnetic_series: Vec<Vec<Vec<f64>>> = (0..dimension)
        .map(|_| {
            (0..dimension)
                .map(|_| Vec::with_capacity(config.measurement_sweeps))
                .collect()
        })
        .collect();
    let mut raw_measurements = Vec::with_capacity(config.measurement_sweeps);

    for measurement_index in 0..config.measurement_sweeps {
        for _ in 0..config.thinning {
            let (sweep_accepted, sweep_invalid) = run_sweep(
                model,
                &mut fields,
                &links,
                &mut current_energy,
                &mut rng,
                config.proposal_scale,
                temperature,
            )?;
            accepted += sweep_accepted;
            invalid += sweep_invalid;
            attempted += fields.len() * model.components.len();
        }

        let energy = model.energy_for(&fields, &links)?;
        current_energy = energy.total;
        let twist = model.bare_twist_response(&fields, &links)?;
        let amplitudes: Vec<f64> = (0..component_count)
            .map(|component| {
                fields
                    .iter()
                    .map(|site| site[component].norm_sqr().sqrt())
                    .sum::<f64>()
                    / fields.len() as f64
            })
            .collect();
        let squared_amplitudes: Vec<f64> = (0..component_count)
            .map(|component| {
                fields
                    .iter()
                    .map(|site| site[component].norm_sqr())
                    .sum::<f64>()
                    / fields.len() as f64
            })
            .collect();
        let relative_cosines = relative_phase_lock_cosines(model, &fields);
        let relative_sines = relative_phase_lock_sines(model, &fields);
        let energy_per_site = energy.total / fields.len() as f64;

        energy_series.push(energy_per_site);
        quadratic_series.push(energy.quadratic / fields.len() as f64);
        quartic_series.push(
            (energy.quartic_self + energy.quartic_density) / fields.len() as f64,
        );
        gradient_series.push(energy.gradient / fields.len() as f64);
        for component in 0..component_count {
            amplitude_series[component].push(amplitudes[component]);
        }
        for term in 0..relative_cosines.len() {
            phase_lock_series[term].push(relative_cosines[term]);
            phase_lock_sine_series[term].push(relative_sines[term]);
        }
        for axis in 0..dimension {
            current_series[axis].push(twist.current[axis]);
            for other_axis in 0..dimension {
                diamagnetic_series[axis][other_axis]
                    .push(twist.diamagnetic[axis][other_axis]);
            }
        }

        raw_measurements.push(json!({
            "measurement_index": measurement_index,
            "energy_per_site": energy_per_site,
            "quadratic_energy_per_site": energy.quadratic / fields.len() as f64,
            "quartic_energy_per_site": (energy.quartic_self + energy.quartic_density) / fields.len() as f64,
            "phase_locking_energy_per_site": energy.phase_locking / fields.len() as f64,
            "secondary_order_energy_per_site": energy.secondary_order / fields.len() as f64,
            "gradient_energy_per_site": energy.gradient / fields.len() as f64,
            "component_mean_amplitudes": amplitudes,
            "component_mean_squared_amplitudes": squared_amplitudes,
            "relative_phase_lock_cosines": relative_cosines,
            "relative_phase_lock_sines": relative_sines,
            "bare_twist_current": twist.current,
            "bare_twist_diamagnetic_tensor": twist.diamagnetic
        }));
    }

    let (helicity_modulus, helicity_jackknife_se, minimum_eigenvalue_jackknife_se) =
        block_jackknife_helicity(&current_series, &diamagnetic_series, temperature)?;
    let current_means: Vec<f64> = current_series.iter().map(|series| mean(series)).collect();
    let mut diamagnetic_mean = vec![vec![0.0; dimension]; dimension];
    let mut current_covariance = vec![vec![0.0; dimension]; dimension];
    for alpha in 0..dimension {
        for beta in 0..dimension {
            diamagnetic_mean[alpha][beta] = mean(&diamagnetic_series[alpha][beta]);
            let product_mean = current_series[alpha]
                .iter()
                .zip(&current_series[beta])
                .map(|(left, right)| left * right)
                .sum::<f64>()
                / config.measurement_sweeps as f64;
            current_covariance[alpha][beta] =
                product_mean - current_means[alpha] * current_means[beta];
        }
    }
    let min_stiffness_eigenvalue =
        super::minimum_symmetric_eigenvalue(&helicity_modulus)?;
    let stiffness_status = match minimum_eigenvalue_jackknife_se {
        None => "insufficient_samples_for_stiffness_jackknife",
        Some(standard_error)
            if min_stiffness_eigenvalue + 1.96 * standard_error
                < -model.stiffness_tolerance =>
        {
            "negative_stiffness_approx95_jackknife_interval_stable_claim_rejected"
        }
        Some(standard_error)
            if min_stiffness_eigenvalue - 1.96 * standard_error
                > model.stiffness_tolerance =>
        {
            "positive_stiffness_approx95_jackknife_interval_within_sample_only"
        }
        Some(_) => "stiffness_approx95_jackknife_interval_includes_zero_unresolved",
    };

    let energy_variance = population_variance(&energy_series);
    let heat_capacity_per_site =
        fields.len() as f64 * energy_variance / (temperature * temperature);
    if !heat_capacity_per_site.is_finite() || heat_capacity_per_site < 0.0 {
        return Err("multicomponent heat-capacity estimate is invalid".to_owned());
    }
    let acceptance_rate = if attempted > 0 {
        accepted as f64 / attempted as f64
    } else {
        0.0
    };

    let component_summaries: Vec<Value> = model
        .components
        .iter()
        .enumerate()
        .map(|(component, metadata)| json!({
            "component_id": metadata.id,
            "component_role": metadata.role,
            "mean_amplitude": series_summary(&amplitude_series[component]),
            "mean_squared_amplitude": series_summary(
                &raw_measurements.iter()
                    .map(|record| {
                        record["component_mean_squared_amplitudes"][component]
                            .as_f64().unwrap()
                    })
                    .collect::<Vec<_>>()
            )
        }))
        .collect();
    let phase_lock_summaries: Vec<Value> = model
        .phase_locking_terms
        .iter()
        .enumerate()
        .map(|(index, term)| json!({
            "component_a": model.components[term.first].id,
            "component_b": model.components[term.second].id,
            "coupling": term.coefficient,
            "mean_relative_phase_cosine": series_summary(&phase_lock_series[index]),
            "mean_relative_phase_sine": series_summary(&phase_lock_sine_series[index]),
            "time_reversal_classification": "not_inferred_from_relative_phase_sine_without_full_crystal_symmetry_action"
        }))
        .collect();

    Ok(json!({
        "status": "executed",
        "chain_seed": seed,
        "initial_energy": initial_energy,
        "final_energy": current_energy,
        "warmup_sweeps": config.warmup_sweeps,
        "measurement_sweeps": config.measurement_sweeps,
        "thinning": config.thinning,
        "proposal_scale": config.proposal_scale,
        "proposals_attempted": attempted,
        "proposals_accepted": accepted,
        "invalid_proposals": invalid,
        "acceptance_rate": acceptance_rate,
        "observables": {
            "energy_per_site": series_summary(&energy_series),
            "quadratic_energy_per_site": series_summary(&quadratic_series),
            "quartic_energy_per_site": series_summary(&quartic_series),
            "gradient_energy_per_site": series_summary(&gradient_series),
            "component_amplitudes": component_summaries,
            "relative_phase_locking": phase_lock_summaries,
            "heat_capacity_per_site_fluctuation_estimate": {
                "value": heat_capacity_per_site,
                "definition": "N * Var(E_per_site) / T^2",
                "warning": "Finite-chain point estimate only; blocked uncertainty and finite-size scaling are not yet implemented."
            }
        },
        "twist_response": {
            "convention": "common total boundary twist added to each positive unit edge as theta_axis / shape_axis",
            "mean_bare_current": current_means,
            "mean_diamagnetic_tensor": diamagnetic_mean,
            "current_covariance": current_covariance,
            "thermal_helicity_modulus_tensor": helicity_modulus,
            "thermal_helicity_modulus_jackknife_standard_error": helicity_jackknife_se,
            "minimum_stiffness_eigenvalue": min_stiffness_eigenvalue,
            "minimum_stiffness_eigenvalue_jackknife_standard_error": minimum_eigenvalue_jackknife_se,
            "approximate_95_percent_minimum_eigenvalue_interval": minimum_eigenvalue_jackknife_se.map(|se| [
                min_stiffness_eigenvalue - 1.96 * se,
                min_stiffness_eigenvalue + 1.96 * se
            ]),
            "stiffness_positivity_gate": stiffness_status,
            "stiffness_tolerance": model.stiffness_tolerance,
            "warning": "Block jackknife quantifies finite-chain resampling variability only; it does not establish thermalization, autocorrelation adequacy, finite-size convergence or material stability."
        },
        "raw_measurements": raw_measurements,
        "claim_ceiling": "Finite-size synthetic multicomponent GL Metropolis pilot only; no equilibration, thermodynamic-limit phase classification, material Tc, or experimental result is established."
    }))
}

pub(super) fn run(
    model: &IntertwinedModel,
    temperature: f64,
    fixture_bytes: &[u8],
    overrides: SamplingOverrides,
) -> Result<Value, String> {
    let config = SamplingConfig::from_fixture(fixture_bytes, overrides)?;
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("multicomponent GL sampler requires temperature > 0".to_owned());
    }
    let mut chains = Vec::with_capacity(config.independent_chains);
    for chain_index in 0..config.independent_chains {
        let seed = config
            .random_seed
            .checked_add(chain_index as u64)
            .ok_or_else(|| "independent chain seed range overflowed u64".to_owned())?;
        chains.push(run_chain(model, temperature, config, seed)?);
    }

    let min_stiffness = chains
        .iter()
        .map(|chain| {
            chain["twist_response"]["minimum_stiffness_eigenvalue"]
                .as_f64()
                .unwrap_or(f64::NAN)
        })
        .fold(f64::INFINITY, f64::min);
    let any_negative = chains.iter().any(|chain| {
        chain["twist_response"]["stiffness_positivity_gate"].as_str()
            == Some("negative_point_estimate_stable_claim_not_permitted_pending_uncertainty")
    });

    Ok(json!({
        "status": "executed",
        "contract": "SC-INTERTWINED-ORDER-METROPOLIS-PILOT-0.1",
        "instance_id": model.instance_id,
        "schema_sha256": model.schema_sha256,
        "fixture_sha256": model.fixture_sha256,
        "algorithm": "single_component_site_metropolis",
        "temperature": temperature,
        "temperature_unit": "dimensionless_GL_temperature",
        "warmup_sweeps_per_chain": config.warmup_sweeps,
        "measurement_sweeps_per_chain": config.measurement_sweeps,
        "thinning": config.thinning,
        "proposal_scale": config.proposal_scale,
        "independent_chains": config.independent_chains,
        "base_seed": config.random_seed,
        "minimum_chain_stiffness_eigenvalue": min_stiffness,
        "negative_stiffness_point_estimate_in_any_chain": any_negative,
        "chain_receipts": chains,
        "claim_ceiling": "Synthetic finite-size multicomponent GL Metropolis software pilot only; stiffness point estimates do not establish a thermodynamically stable superconducting phase."
    }))
}

#[cfg(test)]
mod tests {
    use super::{run, IntertwinedModel, SamplingOverrides};
    use serde_json::Value;
    use std::fs;
    use std::path::PathBuf;

    fn inputs() -> (IntertwinedModel, Vec<u8>, f64) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_intertwined_order_instance_v0_1.schema.json"),
        )
        .unwrap();
        let fixture = fs::read(
            root.join("data/research/superconductivity_intertwined_order_fixture_v0_1.json"),
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&fixture).unwrap();
        let temperature = value["thermal"]["temperature"].as_f64().unwrap();
        let model = IntertwinedModel::from_bytes(&schema, &fixture).unwrap();
        (model, fixture, temperature)
    }

    #[test]
    fn short_multicomponent_chain_is_deterministic_and_finite() {
        let (model, fixture, temperature) = inputs();
        let overrides = SamplingOverrides {
            warmup_sweeps: Some(2),
            measurement_sweeps: Some(4),
            thinning: Some(1),
            independent_chains: Some(2),
            random_seed: Some(23),
            ..SamplingOverrides::default()
        };
        let first = run(&model, temperature, &fixture, overrides).unwrap();
        let second = run(&model, temperature, &fixture, overrides).unwrap();
        assert_eq!(first, second);
        assert_eq!(first["chain_receipts"].as_array().unwrap().len(), 2);
        for chain in first["chain_receipts"].as_array().unwrap() {
            assert!((0.0..=1.0).contains(&chain["acceptance_rate"].as_f64().unwrap()));
            assert!(chain["observables"]["energy_per_site"]["mean"].as_f64().unwrap().is_finite());
            assert!(chain["twist_response"]["minimum_stiffness_eigenvalue"].as_f64().unwrap().is_finite());
            assert_eq!(chain["raw_measurements"].as_array().unwrap().len(), 4);
        }
    }

    #[test]
    fn short_pilot_reports_component_amplitudes_phase_locking_and_stiffness_gate() {
        let (model, fixture, temperature) = inputs();
        let output = run(
            &model,
            temperature,
            &fixture,
            SamplingOverrides {
                warmup_sweeps: Some(2),
                measurement_sweeps: Some(4),
                independent_chains: Some(2),
                random_seed: Some(77),
                ..SamplingOverrides::default()
            },
        )
        .unwrap();
        for chain in output["chain_receipts"].as_array().unwrap() {
            assert_eq!(
                chain["observables"]["component_amplitudes"]
                    .as_array()
                    .unwrap()
                    .len(),
                model.components.len()
            );
            assert_eq!(
                chain["observables"]["relative_phase_locking"]
                    .as_array()
                    .unwrap()
                    .len(),
                model.phase_locking_terms.len()
            );
            let status = chain["twist_response"]["stiffness_positivity_gate"]
                .as_str()
                .unwrap();
            assert!(status == "negative_point_estimate_stable_claim_not_permitted_pending_uncertainty"
                || status == "nonnegative_point_estimate_uncertainty_unresolved");
            assert!(chain["raw_measurements"]
                .as_array()
                .unwrap()
                .iter()
                .all(|row| row["component_mean_amplitudes"]
                    .as_array()
                    .is_some_and(|values| values.len() == model.components.len())));
        }
    }

    #[test]
    fn relative_phase_sine_is_common_gauge_invariant_and_time_reversal_odd() {
        let (model, _, _) = inputs();
        let fields = model.initial_fields();
        let before = relative_phase_lock_sines(&model, &fields);
        assert!(!before.is_empty());

        let angles: Vec<f64> = (0..model.sites.len())
            .map(|index| 0.37 * (index as f64 + 0.4).sin())
            .collect();
        let links = model.initial_link_phases();
        let (gauge_fields, _) =
            model.common_gauge_transform(&fields, &links, &angles).unwrap();
        let after_gauge = relative_phase_lock_sines(&model, &gauge_fields);
        for index in 0..before.len() {
            assert!((before[index] - after_gauge[index]).abs() < 1.0e-12);
        }

        let time_reversed: Vec<Vec<Complex>> = fields
            .iter()
            .map(|site| site.iter().map(|field| Complex {
                re: field.re,
                im: -field.im,
            }).collect())
            .collect();
        let after_time_reversal = relative_phase_lock_sines(&model, &time_reversed);
        for index in 0..before.len() {
            assert!((before[index] + after_time_reversal[index]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn distinct_chain_seeds_are_recorded() {
        let (model, fixture, temperature) = inputs();
        let output = run(
            &model,
            temperature,
            &fixture,
            SamplingOverrides {
                warmup_sweeps: Some(1),
                measurement_sweeps: Some(2),
                independent_chains: Some(3),
                random_seed: Some(1000),
                ..SamplingOverrides::default()
            },
        )
        .unwrap();
        let seeds: Vec<u64> = output["chain_receipts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|chain| chain["chain_seed"].as_u64().unwrap())
            .collect();
        assert_eq!(seeds, vec![1000, 1001, 1002]);
    }

    #[test]
    fn short_chains_do_not_emit_invented_stiffness_uncertainty() {
        let (model, fixture, temperature) = inputs();
        let output = run(
            &model,
            temperature,
            &fixture,
            SamplingOverrides {
                warmup_sweeps: Some(1),
                measurement_sweeps: Some(4),
                independent_chains: Some(1),
                ..SamplingOverrides::default()
            },
        )
        .unwrap();
        let chain = &output["chain_receipts"][0];
        assert_eq!(
            chain["twist_response"]["minimum_stiffness_eigenvalue_jackknife_standard_error"],
            Value::Null
        );
        assert_eq!(
            chain["twist_response"]["stiffness_positivity_gate"],
            "insufficient_samples_for_stiffness_jackknife"
        );
    }

    #[test]
    fn longer_chains_emit_finite_block_jackknife_stiffness_uncertainty() {
        let (model, fixture, temperature) = inputs();
        let output = run(
            &model,
            temperature,
            &fixture,
            SamplingOverrides {
                warmup_sweeps: Some(1),
                measurement_sweeps: Some(40),
                independent_chains: Some(1),
                random_seed: Some(90),
                ..SamplingOverrides::default()
            },
        )
        .unwrap();
        let chain = &output["chain_receipts"][0];
        let standard_error = chain["twist_response"]
            ["minimum_stiffness_eigenvalue_jackknife_standard_error"]
            .as_f64()
            .unwrap();
        assert!(standard_error.is_finite() && standard_error >= 0.0);
        let interval = chain["twist_response"]
            ["approximate_95_percent_minimum_eigenvalue_interval"]
            .as_array()
            .unwrap();
        assert_eq!(interval.len(), 2);
        assert!(interval.iter().all(|value| value.as_f64().is_some()));
    }

    #[test]
    fn sampler_rejects_zero_temperature() {
        let (model, fixture, _) = inputs();
        assert!(run(&model, 0.0, &fixture, SamplingOverrides::default()).is_err());
    }
}
