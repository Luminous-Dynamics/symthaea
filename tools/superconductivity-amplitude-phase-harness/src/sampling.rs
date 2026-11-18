use super::{Complex, GlModel};
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
    let value = required_u64(value, key)?;
    usize::try_from(value).map_err(|_| format!("sampling.{key} exceeds usize"))
}

fn required_positive_f64(value: &Value, key: &str) -> Result<f64, String> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite() && *number > 0.0)
        .ok_or_else(|| format!("sampling.{key} must be positive and finite"))?;
    Ok(number)
}

impl SamplingConfig {
    fn from_fixture(fixture_bytes: &[u8], overrides: SamplingOverrides) -> Result<Self, String> {
        let fixture: Value = serde_json::from_slice(fixture_bytes)
            .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
        let sampling = fixture
            .get("sampling")
            .ok_or_else(|| "fixture missing sampling protocol".to_owned())?;
        if sampling.get("algorithm").and_then(Value::as_str)
            != Some("single_site_complex_field_metropolis")
        {
            return Err("unsupported GL sampling algorithm".to_owned());
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
    let rho = |lag: usize| -> f64 {
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
        let pair_sum = rho(lag) + rho(lag + 1);
        if !pair_sum.is_finite() || pair_sum <= 0.0 {
            break;
        }
        sum_pairs += pair_sum;
        lag += 2;
    }
    Some((0.5 + sum_pairs).max(0.5))
}

fn population_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64
        - mean * mean)
        .max(0.0)
}

fn series_summary(values: &[f64]) -> Value {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let standard_deviation = sample_standard_deviation(values);
    let integrated_tau = integrated_autocorrelation_time(values);
    let effective_sample_size = integrated_tau
        .map(|tau| (values.len() as f64 / (2.0 * tau)).min(values.len() as f64));
    let autocorrelation_corrected_standard_error = match (standard_deviation, integrated_tau) {
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
        "autocorrelation_corrected_standard_error": autocorrelation_corrected_standard_error,
        "sample_count": values.len(),
        "warning": "Short pilot chains can yield unstable autocorrelation estimates; this statistic does not establish equilibration."
    })
}

fn run_sweep(
    model: &GlModel,
    fields: &mut [Complex],
    links: &[f64],
    current_energy: &mut f64,
    rng: &mut StdRng,
    proposal_scale: f64,
) -> Result<(usize, usize), String> {
    let mut accepted = 0;
    let mut invalid = 0;
    let proposal_count = fields.len();

    for _ in 0..proposal_count {
        let site = rng.gen_range(0..fields.len());
        let original = fields[site];
        let candidate = Complex {
            re: original.re + rng.gen_range(-proposal_scale..proposal_scale),
            im: original.im + rng.gen_range(-proposal_scale..proposal_scale),
        };
        let delta_energy =
            match model.site_energy_delta(fields, site, candidate, links) {
                Ok(delta) => delta,
                Err(_) => {
                    invalid += 1;
                    continue;
                }
            };

        let accept = delta_energy <= 0.0
            || rng.gen_range(0.0..1.0) < (-delta_energy / model.temperature).exp();
        if accept {
            fields[site] = candidate;
            *current_energy += delta_energy;
            accepted += 1;
        }
    }
    Ok((accepted, invalid))
}

fn run_chain(
    model: &GlModel,
    config: SamplingConfig,
    seed: u64,
) -> Result<Value, String> {
    if model.temperature <= 0.0 {
        return Err("Metropolis sampling requires temperature > 0".to_owned());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut fields = model.initial_fields();
    let links = model.initial_gauge_phases();
    let initial_energy = model.free_energy_for(&fields, &links)?.total;
    let mut current_energy = initial_energy;

    let mut accepted = 0_usize;
    let mut attempted = 0_usize;
    let mut invalid = 0_usize;
    for _ in 0..config.warmup_sweeps {
        let (sweep_accepted, sweep_invalid) = run_sweep(
            model,
            &mut fields,
            &links,
            &mut current_energy,
            &mut rng,
            config.proposal_scale,
        )?;
        accepted += sweep_accepted;
        invalid += sweep_invalid;
        attempted += fields.len();
    }

    let mut energy_series = Vec::with_capacity(config.measurement_sweeps);
    let mut local_energy_series = Vec::with_capacity(config.measurement_sweeps);
    let mut gradient_energy_series = Vec::with_capacity(config.measurement_sweeps);
    let mut amplitude_series = Vec::with_capacity(config.measurement_sweeps);
    let mut squared_amplitude_series = Vec::with_capacity(config.measurement_sweeps);
    let mut vortex_density_series = Vec::with_capacity(config.measurement_sweeps);
    let mut positive_vortex_series = Vec::with_capacity(config.measurement_sweeps);
    let mut negative_vortex_series = Vec::with_capacity(config.measurement_sweeps);
    let mut skipped_core_plaquette_series = Vec::with_capacity(config.measurement_sweeps);
    let mut raw_measurements = Vec::with_capacity(config.measurement_sweeps);
    let mut twist_current_series: Vec<Vec<f64>> = (0..model.dimension)
        .map(|_| Vec::with_capacity(config.measurement_sweeps))
        .collect();
    let mut diamagnetic_series: Vec<Vec<Vec<f64>>> = (0..model.dimension)
        .map(|_| {
            (0..model.dimension)
                .map(|_| Vec::with_capacity(config.measurement_sweeps))
                .collect()
        })
        .collect();

    for _ in 0..config.measurement_sweeps {
        for _ in 0..config.thinning {
            let (sweep_accepted, sweep_invalid) = run_sweep(
                model,
                &mut fields,
                &links,
                &mut current_energy,
                &mut rng,
                config.proposal_scale,
            )?;
            accepted += sweep_accepted;
            invalid += sweep_invalid;
            attempted += fields.len();
        }

        let energy = model.free_energy_for(&fields, &links)?;
        current_energy = energy.total;
        let amplitude_mean = fields
            .iter()
            .map(|field| field.norm_sqr().sqrt())
            .sum::<f64>()
            / fields.len() as f64;
        let squared_amplitude_mean = fields
            .iter()
            .map(|field| field.norm_sqr())
            .sum::<f64>()
            / fields.len() as f64;

        energy_series.push(energy.total / fields.len() as f64);
        local_energy_series.push(energy.local / fields.len() as f64);
        gradient_energy_series.push(energy.gradient / fields.len() as f64);
        amplitude_series.push(amplitude_mean);
        squared_amplitude_series.push(squared_amplitude_mean);

        let twist = model.twist_response(&fields, &links)?;
        for alpha in 0..model.dimension {
            twist_current_series[alpha].push(twist.current[alpha]);
            for beta in 0..model.dimension {
                diamagnetic_series[alpha][beta].push(twist.diamagnetic[alpha][beta]);
            }
        }

        let vortex_record = if let Some(vortices) = model.vortex_diagnostics(&fields, &links)? {
            let density = if vortices.valid_plaquettes > 0 {
                vortices.absolute_charge_sum as f64 / vortices.valid_plaquettes as f64
            } else {
                0.0
            };
            vortex_density_series.push(density);
            positive_vortex_series.push(vortices.positive_charge as f64);
            negative_vortex_series.push(vortices.negative_charge as f64);
            skipped_core_plaquette_series
                .push(vortices.skipped_low_amplitude_plaquettes as f64);
            json!({
                "positive_charge": vortices.positive_charge,
                "negative_charge": vortices.negative_charge,
                "absolute_charge_sum": vortices.absolute_charge_sum,
                "valid_plaquettes": vortices.valid_plaquettes,
                "skipped_low_amplitude_plaquettes": vortices.skipped_low_amplitude_plaquettes,
                "absolute_charge_density": density
            })
        } else {
            Value::Null
        };

        raw_measurements.push(json!({
            "measurement_index": raw_measurements.len(),
            "energy_per_node": energy.total / fields.len() as f64,
            "local_potential_energy_per_node": energy.local / fields.len() as f64,
            "gradient_energy_per_node": energy.gradient / fields.len() as f64,
            "mean_amplitude": amplitude_mean,
            "mean_squared_amplitude": squared_amplitude_mean,
            "twist_current": twist.current,
            "twist_diamagnetic": twist.diamagnetic,
            "vortex_diagnostics": vortex_record
        }));
    }

    if energy_series.iter().any(|value| !value.is_finite())
        || amplitude_series.iter().any(|value| !value.is_finite())
    {
        return Err("GL sampler produced a non-finite measurement".to_owned());
    }

    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let current_means: Vec<f64> = twist_current_series
        .iter()
        .map(|series| mean(series))
        .collect();
    let mut diamagnetic_means = vec![vec![0.0; model.dimension]; model.dimension];
    let mut current_covariance = vec![vec![0.0; model.dimension]; model.dimension];
    let mut helicity_modulus = vec![vec![0.0; model.dimension]; model.dimension];
    for alpha in 0..model.dimension {
        for beta in 0..model.dimension {
            diamagnetic_means[alpha][beta] = mean(&diamagnetic_series[alpha][beta]);
            let current_product_mean = twist_current_series[alpha]
                .iter()
                .zip(&twist_current_series[beta])
                .map(|(left, right)| left * right)
                .sum::<f64>()
                / config.measurement_sweeps as f64;
            current_covariance[alpha][beta] =
                current_product_mean - current_means[alpha] * current_means[beta];
            helicity_modulus[alpha][beta] = diamagnetic_means[alpha][beta]
                - current_covariance[alpha][beta] / model.temperature;
        }
    }

    if helicity_modulus
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("GL helicity-modulus estimator produced a non-finite tensor".to_owned());
    }
    let acceptance_rate = if attempted > 0 {
        accepted as f64 / attempted as f64
    } else {
        0.0
    };
    let energy_variance_per_node = population_variance(&energy_series);
    let heat_capacity_per_node = fields.len() as f64
        * energy_variance_per_node
        / (model.temperature * model.temperature);
    if !heat_capacity_per_node.is_finite() || heat_capacity_per_node < 0.0 {
        return Err("GL heat-capacity fluctuation estimator is invalid".to_owned());
    }

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
        "heat_capacity_per_node_fluctuation_estimate": heat_capacity_per_node,
        "energy_variance_per_node_squared": energy_variance_per_node,
        "twist_response": {
            "twist_convention": "A_ij(theta)=A_ij+sum_mu theta_mu*edge_displacement_mu/box_length_mu",
            "first_derivative_current_mean": current_means,
            "second_derivative_diamagnetic_mean": diamagnetic_means,
            "current_covariance": current_covariance,
            "thermal_helicity_modulus_tensor": helicity_modulus,
            "formula": "Y_mu_nu=<d2F/dtheta_mu dtheta_nu> - Cov(dF/dtheta_mu,dF/dtheta_nu)/T",
            "unit": "dimensionless_lattice_energy"
        },
        "observables": {
            "energy_per_node": series_summary(&energy_series),
            "heat_capacity_per_node_fluctuation_estimate": {
                "value": heat_capacity_per_node,
                "definition": "N * Var(E_per_node) / T^2",
                "warning": "Finite-chain fluctuation estimate only; autocorrelation-aware uncertainty and thermodynamic-limit scaling are not yet implemented."
            },
            "local_potential_energy_per_node": series_summary(&local_energy_series),
            "gradient_energy_per_node": series_summary(&gradient_energy_series),
            "mean_amplitude": series_summary(&amplitude_series),
            "mean_squared_amplitude": series_summary(&squared_amplitude_series),
            "vortex_charge_density": if vortex_density_series.is_empty() {
                Value::Null
            } else {
                series_summary(&vortex_density_series)
            },
            "positive_vortex_charge": if positive_vortex_series.is_empty() {
                Value::Null
            } else {
                series_summary(&positive_vortex_series)
            },
            "negative_vortex_charge": if negative_vortex_series.is_empty() {
                Value::Null
            } else {
                series_summary(&negative_vortex_series)
            },
            "skipped_low_amplitude_plaquettes": if skipped_core_plaquette_series.is_empty() {
                Value::Null
            } else {
                series_summary(&skipped_core_plaquette_series)
            }
        },
        "raw_measurements": raw_measurements,
        "claim_ceiling": "Finite synthetic lattice Metropolis pilot only; no equilibration guarantee, phase transition, thermodynamic-limit observable, material Tc, or experiment is established."
    }))
}

fn ensemble_summary(values: &[f64]) -> Value {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let deviation = sample_standard_deviation(values);
    let between_chain_standard_error =
        deviation.map(|value| value / (values.len() as f64).sqrt());
    json!({
        "mean_across_independent_chains": mean,
        "between_chain_sample_standard_deviation": deviation,
        "between_chain_standard_error": between_chain_standard_error,
        "chain_count": values.len(),
        "warning": "Between-chain spread does not demonstrate equilibration or replace within-chain autocorrelation analysis."
    })
}

pub(super) fn run(
    model: &GlModel,
    fixture_bytes: &[u8],
    overrides: SamplingOverrides,
) -> Result<Value, String> {
    let config = SamplingConfig::from_fixture(fixture_bytes, overrides)?;
    if model.temperature <= 0.0 {
        return Err("GL Metropolis sampler requires temperature > 0".to_owned());
    }
    let mut chains = Vec::with_capacity(config.independent_chains);
    for chain_index in 0..config.independent_chains {
        let seed = config
            .random_seed
            .checked_add(chain_index as u64)
            .ok_or_else(|| "independent-chain seed range overflowed u64".to_owned())?;
        chains.push(run_chain(model, config, seed)?);
    }

    let collect_means = |field: &str| -> Result<Vec<f64>, String> {
        chains
            .iter()
            .map(|chain| {
                chain
                    .get("observables")
                    .and_then(|observables| observables.get(field))
                    .and_then(|summary| summary.get("mean"))
                    .and_then(Value::as_f64)
                    .ok_or_else(|| format!("chain receipt missing mean for {field}"))
            })
            .collect()
    };

    let energy = collect_means("energy_per_node")?;
    let amplitude = collect_means("mean_amplitude")?;
    let squared_amplitude = collect_means("mean_squared_amplitude")?;
    let heat_capacity = chains
        .iter()
        .map(|chain| {
            chain
                .get("heat_capacity_per_node_fluctuation_estimate")
                .and_then(Value::as_f64)
                .ok_or_else(|| "chain receipt missing heat capacity estimate".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let vortex_density = chains
        .iter()
        .filter_map(|chain| {
            chain
                .get("observables")
                .and_then(|observables| observables.get("vortex_charge_density"))
                .and_then(|summary| summary.get("mean"))
                .and_then(Value::as_f64)
        })
        .collect::<Vec<_>>();
    let mut between_chain_helicity_modulus = vec![vec![Value::Null; model.dimension]; model.dimension];
    for alpha in 0..model.dimension {
        for beta in 0..model.dimension {
            let values: Vec<f64> = chains
                .iter()
                .map(|chain| {
                    chain
                        .get("twist_response")
                        .and_then(|response| response.get("thermal_helicity_modulus_tensor"))
                        .and_then(Value::as_array)
                        .and_then(|rows| rows.get(alpha))
                        .and_then(Value::as_array)
                        .and_then(|row| row.get(beta))
                        .and_then(Value::as_f64)
                        .ok_or_else(|| format!("chain missing helicity modulus component ({alpha},{beta})"))
                })
                .collect::<Result<_, _>>()?;
            between_chain_helicity_modulus[alpha][beta] =
                ensemble_summary(&values);
        }
    }

    Ok(json!({
        "status": "executed",
        "contract": "SC-AMPLITUDE-PHASE-METROPOLIS-PILOT-0.1",
        "instance_id": model.instance_id,
        "schema_sha256": model.schema_sha256,
        "fixture_sha256": model.fixture_sha256,
        "algorithm": "single_site_complex_field_metropolis",
        "temperature": model.temperature,
        "temperature_unit": "dimensionless_GL_temperature",
        "warmup_sweeps_per_chain": config.warmup_sweeps,
        "measurement_sweeps_per_chain": config.measurement_sweeps,
        "thinning": config.thinning,
        "proposal_scale": config.proposal_scale,
        "independent_chains": config.independent_chains,
        "base_seed": config.random_seed,
        "between_chain_summary": {
            "energy_per_node": ensemble_summary(&energy),
            "mean_amplitude": ensemble_summary(&amplitude),
            "mean_squared_amplitude": ensemble_summary(&squared_amplitude),
            "heat_capacity_per_node_fluctuation_estimate": ensemble_summary(&heat_capacity),
            "vortex_charge_density": if vortex_density.is_empty() {
                Value::Null
            } else {
                ensemble_summary(&vortex_density)
            },
            "thermal_helicity_modulus_tensor": between_chain_helicity_modulus
        },
        "chain_receipts": chains,
        "claim_ceiling": "Synthetic finite-size GL Metropolis software pilot only; autocorrelation diagnostics and independent chains do not establish equilibration, phase classification, material Tc, or experimental evidence."
    }))
}

#[cfg(test)]
mod tests {
    use super::{run, GlModel, SamplingOverrides};
    use serde_json::{json, Value};
    use std::fs;
    use std::path::PathBuf;

    fn inputs() -> (GlModel, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_amplitude_phase_instance_v0_1.schema.json"),
        )
        .unwrap();
        let fixture = fs::read(
            root.join("data/research/superconductivity_amplitude_phase_fixture_v0_1.json"),
        )
        .unwrap();
        let model = GlModel::from_bytes(&schema, &fixture).unwrap();
        (model, fixture)
    }

    #[test]
    fn short_metropolis_pilot_is_finite_and_reproducible() {
        let (model, fixture) = inputs();
        let overrides = SamplingOverrides {
            warmup_sweeps: Some(2),
            measurement_sweeps: Some(4),
            thinning: Some(1),
            independent_chains: Some(2),
            random_seed: Some(7),
            ..SamplingOverrides::default()
        };
        let first = run(&model, &fixture, overrides).unwrap();
        let second = run(&model, &fixture, overrides).unwrap();
        assert_eq!(first, second);
        assert_eq!(first["status"], "executed");
        assert_eq!(first["independent_chains"], 2);
        for chain in first["chain_receipts"].as_array().unwrap() {
            let acceptance = chain["acceptance_rate"].as_f64().unwrap();
            assert!((0.0..=1.0).contains(&acceptance));
            assert!(
                chain["observables"]["energy_per_node"]["mean"]
                    .as_f64()
                    .unwrap()
                    .is_finite()
            );
            let stiffness = chain["twist_response"]["thermal_helicity_modulus_tensor"]
                .as_array()
                .unwrap();
            assert_eq!(stiffness.len(), 2);
            assert!(stiffness.iter().all(|row| {
                row.as_array().is_some_and(|values| {
                    values.len() == 2 && values.iter().all(|value| value.is_number())
                })
            }));
        }
        let between_chain_stiffness = first["between_chain_summary"]
            ["thermal_helicity_modulus_tensor"]
            .as_array()
            .unwrap();
        assert_eq!(between_chain_stiffness.len(), 2);
    }

    #[test]
    fn chain_seeds_are_unique_and_deterministic() {
        let (model, fixture) = inputs();
        let output = run(
            &model,
            &fixture,
            SamplingOverrides {
                warmup_sweeps: Some(1),
                measurement_sweeps: Some(2),
                independent_chains: Some(3),
                random_seed: Some(100),
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
        assert_eq!(seeds, vec![100, 101, 102]);
    }

    #[test]
    fn proposal_scale_must_be_positive_and_finite() {
        let (model, fixture) = inputs();
        let result = run(
            &model,
            &fixture,
            SamplingOverrides {
                proposal_scale: Some(0.0),
                ..SamplingOverrides::default()
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn sampler_rejects_zero_temperature() {
        let (_, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["thermal"]["temperature"] = json!(0.0);
        let mutated = serde_json::to_vec(&value).unwrap();
        let schema = {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
            fs::read(
                root.join("data/schemas/superconductivity_amplitude_phase_instance_v0_1.schema.json"),
            )
            .unwrap()
        };
        let zero_temperature_model =
            GlModel::from_bytes(&schema, &mutated).unwrap();
        assert!(run(&zero_temperature_model, &mutated, SamplingOverrides::default()).is_err());
    }
}
