use jsonschema::validator_for;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::f64::consts::{PI, TAU};
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Copy)]
struct Bond {
    neighbour: usize,
    coupling: f64,
}

struct QxyModel {
    instance_id: String,
    schema_sha256: String,
    fixture_sha256: String,
    lattice_linear_size: usize,
    imaginary_time_slices: usize,
    t: f64,
    u: f64,
    temperature: f64,
    beta: f64,
    delta_tau: f64,
    k_spatial: f64,
    k_imaginary_time: f64,
    random_seed: u64,
    node_count: usize,
    adjacency: Vec<Vec<Bond>>,
    spatial_bonds: Vec<(usize, usize)>,
    temporal_bonds: Vec<(usize, usize)>,
}

struct Measurement {
    action_per_spacetime_site: f64,
    stiffness: f64,
    vortex_density: f64,
    current_correlation: Vec<f64>,
}

impl QxyModel {
    fn from_bytes(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Self, String> {
        let schema: Value = serde_json::from_slice(schema_bytes)
            .map_err(|error| format!("schema JSON parse failed: {error}"))?;
        let fixture: Value = serde_json::from_slice(fixture_bytes)
            .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
        const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
        if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
            return Err(format!("schema must declare {DIALECT}"));
        }
        let validator =
            validator_for(&schema).map_err(|error| format!("schema compilation failed: {error}"))?;
        if let Some(error) = validator.iter_errors(&fixture).next() {
            return Err(format!(
                "QXY input schema validation failed: {} at {}",
                error,
                error.instance_path()
            ));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("this initial QXY solver accepts synthetic fixtures only".to_owned());
        }

        let instance_id = fixture
            .get("instance_id")
            .and_then(Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| "QXY fixture requires instance_id".to_owned())?
            .to_owned();
        let schema_sha256 = format!("{:x}", Sha256::digest(schema_bytes));
        let fixture_sha256 = format!("{:x}", Sha256::digest(fixture_bytes));

        let model = fixture
            .get("model")
            .ok_or_else(|| "missing model declaration".to_owned())?;
        if model.get("kind").and_then(Value::as_str) != Some("source_reproduction")
            || model.get("source_model_id").and_then(Value::as_str)
                != Some("ponticelli-2026-qxy-2d-square-lattice-phase-representation")
            || model.get("hbar_convention").and_then(Value::as_str) != Some("hbar_equals_1")
            || model.get("kB_convention").and_then(Value::as_str) != Some("kB_equals_1")
            || model.get("pair_charge_convention").and_then(Value::as_str)
                != Some("2e_equals_1")
        {
            return Err(
                "QXY pilot requires the explicitly pinned Ponticelli source-model conventions"
                    .to_owned(),
            );
        }
        if model.get("energy_unit").and_then(Value::as_str) != Some("coupling_energy_units")
            || model.get("temperature_unit").and_then(Value::as_str)
                != Some("coupling_energy_units")
        {
            return Err("QXY model energy/temperature units are not supported".to_owned());
        }

        let graph = fixture
            .get("graph")
            .ok_or_else(|| "missing graph".to_owned())?;
        if graph.get("spatial_dimension").and_then(Value::as_u64) != Some(2)
            || graph.get("embedding").and_then(Value::as_str) != Some("physical_coordinates")
        {
            return Err(
                "QXY source pilot requires a two-dimensional physical square lattice".to_owned(),
            );
        }
        let lengths = graph
            .get("box_lengths")
            .and_then(Value::as_array)
            .ok_or_else(|| "graph.box_lengths must be an array".to_owned())?;
        if lengths.len() != 2 {
            return Err("QXY source pilot requires two box lengths".to_owned());
        }
        let lx = lengths[0]
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| "invalid x box length".to_owned())?;
        let ly = lengths[1]
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| "invalid y box length".to_owned())?;
        if (lx - ly).abs() > 1.0e-10 || (lx - lx.round()).abs() > 1.0e-10 || lx < 3.0 {
            return Err(
                "QXY source pilot requires an equal-integer periodic L x L square with L >= 3"
                    .to_owned(),
            );
        }
        let l = lx as usize;

        let axes = graph
            .get("periodic_axes")
            .and_then(Value::as_array)
            .ok_or_else(|| "periodic_axes must be an array".to_owned())?;
        if axes.len() != 2 || axes[0].as_u64() != Some(0) || axes[1].as_u64() != Some(1) {
            return Err(
                "QXY source pilot requires periodic boundaries on both spatial axes".to_owned(),
            );
        }

        let nodes = graph
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| "graph.nodes must be an array".to_owned())?;
        let spatial_sites = l * l;
        if nodes.len() != spatial_sites {
            return Err(format!(
                "square lattice requires L^2={spatial_sites} nodes; received {}",
                nodes.len()
            ));
        }
        let mut node_index = HashMap::with_capacity(nodes.len());
        let mut coordinates = vec![None; spatial_sites];
        for node in nodes {
            let id = node
                .get("node_id")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| "node missing non-empty node_id".to_owned())?;
            let position = node
                .get("position")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("node {id} missing position"))?;
            if position.len() != 2 {
                return Err(format!("node {id} position dimension mismatch"));
            }
            let x = position[0]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("node {id} invalid x coordinate"))?;
            let y = position[1]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("node {id} invalid y coordinate"))?;
            if (x - x.round()).abs() > 1.0e-10
                || (y - y.round()).abs() > 1.0e-10
                || x < 0.0
                || y < 0.0
                || x >= lx
                || y >= ly
            {
                return Err(format!("node {id} must occupy an integer lattice coordinate"));
            }
            let x = x as usize;
            let y = y as usize;
            let site = x * l + y;
            if coordinates[site].replace((x, y)).is_some() {
                return Err(format!("duplicate lattice coordinate ({x},{y})"));
            }
            if node_index.insert(id.to_owned(), site).is_some() {
                return Err(format!("duplicate node ID {id}"));
            }
        }
        if coordinates.iter().any(Option::is_none) {
            return Err("square lattice has missing coordinates".to_owned());
        }

        let edges = graph
            .get("edges")
            .and_then(Value::as_array)
            .ok_or_else(|| "graph.edges must be an array".to_owned())?;
        if edges.len() != 2 * spatial_sites {
            return Err(format!(
                "periodic square lattice requires {} positive-direction bonds; received {}",
                2 * spatial_sites,
                edges.len()
            ));
        }
        let mut edge_ids = HashSet::with_capacity(edges.len());
        let mut seen_bonds = HashSet::with_capacity(edges.len());
        let mut spatial_bonds = Vec::with_capacity(edges.len());
        let mut t_reference: Option<f64> = None;
        let mut any_flux = false;
        for edge in edges {
            let id = edge
                .get("edge_id")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| "edge missing non-empty edge_id".to_owned())?;
            if !edge_ids.insert(id.to_owned()) {
                return Err(format!("duplicate edge ID {id}"));
            }
            let source_id = edge
                .get("source")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("edge {id} missing source"))?;
            let target_id = edge
                .get("target")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("edge {id} missing target"))?;
            let source = *node_index
                .get(source_id)
                .ok_or_else(|| format!("edge {id} unresolved source"))?;
            let target = *node_index
                .get(target_id)
                .ok_or_else(|| format!("edge {id} unresolved target"))?;
            if source == target {
                return Err(format!("edge {id} self-loop is invalid"));
            }

            let energy = edge
                .get("josephson_energy")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or_else(|| format!("edge {id} josephson_energy must be positive"))?;
            if let Some(reference) = t_reference {
                if (energy - reference).abs() > 1.0e-10 {
                    return Err(
                        "source QXY pilot requires uniform nearest-neighbour Josephson energy t"
                            .to_owned(),
                    );
                }
            } else {
                t_reference = Some(energy);
            }

            let gauge = edge
                .get("gauge_link_phase_rad")
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("edge {id} missing gauge phase"))?;
            if !gauge.is_finite() {
                return Err(format!("edge {id} gauge phase must be finite"));
            }
            any_flux |= gauge.abs() > 1.0e-12;

            let source_position = coordinates[source].unwrap();
            let target_position = coordinates[target].unwrap();
            let image = edge
                .get("periodic_image_shift")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("edge {id} missing periodic_image_shift"))?;
            if image.len() != 2 {
                return Err(format!("edge {id} image-shift dimension mismatch"));
            }
            let shift_x = image[0]
                .as_i64()
                .ok_or_else(|| format!("edge {id} x image shift must be integer"))?;
            let shift_y = image[1]
                .as_i64()
                .ok_or_else(|| format!("edge {id} y image shift must be integer"))?;
            let dx = target_position.0 as i64 + shift_x * l as i64
                - source_position.0 as i64;
            let dy = target_position.1 as i64 + shift_y * l as i64
                - source_position.1 as i64;
            if !((dx == 1 && dy == 0) || (dx == 0 && dy == 1)) {
                return Err(format!(
                    "edge {id} is not a positive-direction nearest-neighbour square-lattice bond"
                ));
            }
            let declared_length = edge
                .get("physical_edge_length")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or_else(|| format!("edge {id} physical_edge_length must be positive"))?;
            if (declared_length - 1.0).abs() > 1.0e-10 {
                return Err(format!("edge {id} nearest-neighbour length must be one lattice unit"));
            }
            let bond_key = (source.min(target), source.max(target));
            if !seen_bonds.insert(bond_key) {
                return Err(format!("duplicate undirected spatial bond at edge {id}"));
            }
            spatial_bonds.push((source, target));
        }
        if any_flux
            || fixture["external_flux"]["mode"].as_str() != Some("zero")
            || fixture["external_flux"]["total_flux_quanta"].as_f64() != Some(0.0)
        {
            return Err(
                "QXY source pilot is restricted to zero external flux and zero gauge links"
                    .to_owned(),
            );
        }
        let t = t_reference.ok_or_else(|| "no Josephson bonds supplied".to_owned())?;

        let charging = fixture
            .get("charging")
            .ok_or_else(|| "missing charging parameters".to_owned())?;
        if charging.get("matrix_kind").and_then(Value::as_str) != Some("diagonal_scalar") {
            return Err(
                "QXY source model requires the compact on-site diagonal_scalar charging term"
                    .to_owned(),
            );
        }
        let u = charging
            .get("diagonal_value")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| "QXY on-site interaction U must be positive".to_owned())?;
        let offsets = charging
            .get("offset_charges")
            .and_then(Value::as_array)
            .ok_or_else(|| "offset_charges must be an array".to_owned())?;
        if offsets.len() != spatial_sites
            || offsets.iter().any(|value| {
                value
                    .as_f64()
                    .is_none_or(|charge| !charge.is_finite() || charge.abs() > 1.0e-12)
            })
        {
            return Err("source QXY pilot requires one zero offset charge per site".to_owned());
        }

        if fixture["bath"]["mode"].as_str() != Some("none")
            || fixture["bath"]["coupling_strength"].as_f64() != Some(0.0)
        {
            return Err(
                "Ponticelli source model has no dissipative bath in this reproduction track"
                    .to_owned(),
            );
        }

        let simulation = fixture
            .get("simulation")
            .ok_or_else(|| "missing simulation parameters".to_owned())?;
        if simulation.get("algorithm").and_then(Value::as_str)
            != Some("path_integral_quantum_monte_carlo")
            || simulation.get("representation").and_then(Value::as_str)
                != Some("phase_path_integral")
        {
            return Err(
                "QXY source pilot requires phase-representation path-integral QMC".to_owned(),
            );
        }
        if simulation.get("temperature_unit").and_then(Value::as_str)
            != Some("coupling_energy_units")
        {
            return Err("QXY temperature must use the t energy unit".to_owned());
        }
        let temperature = simulation["temperature"]
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| "QXY temperature must be positive".to_owned())?;
        let beta = 1.0 / temperature;
        let ntau = simulation["imaginary_time_slices"]
            .as_u64()
            .filter(|value| *value >= 4)
            .ok_or_else(|| "N_tau must be at least four".to_owned())? as usize;
        let delta_tau = simulation["imaginary_time_step"]
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| "Delta_tau must be positive".to_owned())?;
        if (ntau as f64 * delta_tau - beta).abs() > 1.0e-10 * beta.max(1.0) {
            return Err("N_tau * Delta_tau must equal beta = 1/T".to_owned());
        }
        let seed = simulation["random_seed"]
            .as_u64()
            .ok_or_else(|| "random_seed must be unsigned".to_owned())?;

        let k_spatial = t * delta_tau;
        let k_imaginary_time = 1.0 / (u * delta_tau);
        let node_count = spatial_sites * ntau;
        let mut adjacency = vec![Vec::new(); node_count];
        let mut spacetime_spatial_bonds = Vec::with_capacity(spatial_bonds.len() * ntau);
        let mut temporal_bonds = Vec::with_capacity(node_count);

        for tau in 0..ntau {
            let base = tau * spatial_sites;
            for (source, target) in &spatial_bonds {
                let i = base + source;
                let j = base + target;
                adjacency[i].push(Bond {
                    neighbour: j,
                    coupling: k_spatial,
                });
                adjacency[j].push(Bond {
                    neighbour: i,
                    coupling: k_spatial,
                });
                spacetime_spatial_bonds.push((i, j));
            }
        }
        for tau in 0..ntau {
            let next_tau = (tau + 1) % ntau;
            for site in 0..spatial_sites {
                let i = tau * spatial_sites + site;
                let j = next_tau * spatial_sites + site;
                adjacency[i].push(Bond {
                    neighbour: j,
                    coupling: k_imaginary_time,
                });
                adjacency[j].push(Bond {
                    neighbour: i,
                    coupling: k_imaginary_time,
                });
                temporal_bonds.push((i, j));
            }
        }

        Ok(Self {
            instance_id,
            schema_sha256,
            fixture_sha256,
            lattice_linear_size: l,
            imaginary_time_slices: ntau,
            t,
            u,
            temperature,
            beta,
            delta_tau,
            k_spatial,
            k_imaginary_time,
            random_seed: seed,
            node_count,
            adjacency,
            spatial_bonds: spacetime_spatial_bonds,
            temporal_bonds,
        })
    }
}

fn wrap_to_pi(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

fn wolff_cluster(
    model: &QxyModel,
    phases: &mut [f64],
    rng: &mut StdRng,
) -> usize {
    let axis = rng.gen_range(0.0..TAU);
    let seed = rng.gen_range(0..model.node_count);
    let mut in_cluster = vec![false; model.node_count];
    let mut queue = VecDeque::new();
    let mut cluster = Vec::new();

    in_cluster[seed] = true;
    queue.push_back(seed);
    cluster.push(seed);

    while let Some(site) = queue.pop_front() {
        let projection_site = (phases[site] - axis).cos();
        for bond in &model.adjacency[site] {
            if in_cluster[bond.neighbour] {
                continue;
            }
            let projection_neighbour = (phases[bond.neighbour] - axis).cos();
            let product = projection_site * projection_neighbour;
            if product <= 0.0 {
                continue;
            }
            let probability = 1.0 - (-2.0 * bond.coupling * product).exp();
            if rng.gen_range(0.0..1.0) < probability {
                in_cluster[bond.neighbour] = true;
                queue.push_back(bond.neighbour);
                cluster.push(bond.neighbour);
            }
        }
    }

    for site in &cluster {
        phases[*site] = (2.0 * axis + PI - phases[*site]).rem_euclid(TAU);
    }
    cluster.len()
}

fn metropolis_sweep(
    model: &QxyModel,
    phases: &mut [f64],
    rng: &mut StdRng,
) -> (usize, usize) {
    let mut accepted = 0;
    for _ in 0..model.node_count {
        let site = rng.gen_range(0..model.node_count);
        let old_phase = phases[site];
        let new_phase = rng.gen_range(0.0..TAU);
        let mut delta_action = 0.0;
        for bond in &model.adjacency[site] {
            let neighbour_phase = phases[bond.neighbour];
            delta_action += bond.coupling
                * ((old_phase - neighbour_phase).cos()
                    - (new_phase - neighbour_phase).cos());
        }
        if delta_action <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_action).exp() {
            phases[site] = new_phase;
            accepted += 1;
        }
    }
    (accepted, model.node_count)
}

fn sweep(
    model: &QxyModel,
    phases: &mut [f64],
    rng: &mut StdRng,
) -> (usize, usize, usize, usize) {
    let mut flipped_sites = 0;
    let mut clusters = 0;
    let mut cluster_size_sum = 0;
    while flipped_sites < model.node_count {
        let size = wolff_cluster(model, phases, rng);
        flipped_sites += size;
        cluster_size_sum += size;
        clusters += 1;
    }
    let (metropolis_accepted, metropolis_proposed) = metropolis_sweep(model, phases, rng);
    (
        clusters,
        cluster_size_sum,
        metropolis_accepted,
        metropolis_proposed,
    )
}

fn measure(model: &QxyModel, phases: &[f64]) -> Measurement {
    let mut spatial_cosine = 0.0;
    let mut spatial_sine = 0.0;
    for (source, target) in &model.spatial_bonds {
        let delta = phases[*source] - phases[*target];
        spatial_cosine += delta.cos();
        spatial_sine += delta.sin();
    }

    let l = model.lattice_linear_size;
    let bonds_per_time_slice = 2 * l * l;
    let mut current_by_imaginary_time = vec![0.0; model.imaginary_time_slices];
    for (tau, slice_current) in current_by_imaginary_time.iter_mut().enumerate() {
        let start = tau * bonds_per_time_slice;
        let end = start + bonds_per_time_slice;
        for (source, target) in &model.spatial_bonds[start..end] {
            *slice_current += model.t * (phases[*source] - phases[*target]).sin();
        }
    }
    let current_at_zero = current_by_imaginary_time[0];
    let current_correlation: Vec<f64> = current_by_imaginary_time
        .iter()
        .map(|current| current_at_zero * current)
        .collect();

    let mut spatial_action = 0.0;
    let mut total_action = 0.0;
    for (source, target) in &model.spatial_bonds {
        let cosine = (phases[*source] - phases[*target]).cos();
        spatial_action -= model.k_spatial * cosine;
    }
    for (source, target) in &model.temporal_bonds {
        let cosine = (phases[*source] - phases[*target]).cos();
        total_action -= model.k_imaginary_time * cosine;
    }
    total_action += spatial_action;

    let s_s = model.k_spatial * spatial_cosine / model.beta;
    let current = model.t * model.delta_tau * spatial_sine / model.beta;
    let stiffness = (s_s - model.beta * current * current)
        / (2.0 * model.t * (model.lattice_linear_size.pow(2) as f64));

    let mut total_vorticity = 0.0;
    for tau in 0..model.imaginary_time_slices {
        let base = tau * l * l;
        for x in 0..l {
            for y in 0..l {
                let p00 = base + x * l + y;
                let p10 = base + ((x + 1) % l) * l + y;
                let p11 = base + ((x + 1) % l) * l + (y + 1) % l;
                let p01 = base + x * l + (y + 1) % l;
                let circulation = wrap_to_pi(phases[p10] - phases[p00])
                    + wrap_to_pi(phases[p11] - phases[p10])
                    + wrap_to_pi(phases[p01] - phases[p11])
                    + wrap_to_pi(phases[p00] - phases[p01]);
                total_vorticity += circulation.abs() / TAU;
            }
        }
    }
    let vortex_density =
        total_vorticity / (model.imaginary_time_slices as f64 * (l * l) as f64);

    Measurement {
        action_per_spacetime_site: total_action / model.node_count as f64,
        stiffness,
        vortex_density,
        current_correlation,
    }
}

fn sample_standard_deviation(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Some(variance.sqrt())
}

fn lag_one_autocorrelation(values: &[f64]) -> Option<f64> {
    if values.len() < 3 {
        return None;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let total_variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>();
    if total_variance <= f64::EPSILON {
        return None;
    }
    let lag_covariance = values
        .windows(2)
        .map(|pair| (pair[0] - average) * (pair[1] - average))
        .sum::<f64>();
    Some(lag_covariance / total_variance)
}

fn integrated_autocorrelation_time(values: &[f64]) -> Option<f64> {
    if values.len() < 6 {
        return None;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let centered: Vec<f64> = values.iter().map(|value| value - average).collect();
    let variance = centered.iter().map(|value| value * value).sum::<f64>();
    if variance <= f64::EPSILON {
        return None;
    }

    let rho = |lag: usize| -> f64 {
        let covariance = centered
            .iter()
            .zip(centered.iter().skip(lag))
            .map(|(left, right)| left * right)
            .sum::<f64>();
        covariance / variance
    };

    // Initial-positive-sequence estimate: add autocorrelation pairs until
    // the first non-positive pair. Short chains remain only diagnostic.
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

fn sampling_summary(values: &[f64]) -> Value {
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let standard_deviation = sample_standard_deviation(values);
    let iid_standard_error = standard_deviation
        .map(|deviation| deviation / (values.len() as f64).sqrt());
    let integrated_tau = integrated_autocorrelation_time(values);
    let effective_sample_size =
        integrated_tau.map(|tau| (values.len() as f64 / (2.0 * tau)).min(values.len() as f64));
    json!({
        "mean": average,
        "sample_standard_deviation": standard_deviation,
        "iid_standard_error_not_autocorrelation_corrected": iid_standard_error,
        "lag_one_autocorrelation": lag_one_autocorrelation(values),
        "integrated_autocorrelation_time_initial_positive_sequence": integrated_tau,
        "effective_sample_size_initial_positive_sequence": effective_sample_size,
        "sample_count": values.len(),
        "warning": "These short-chain diagnostics do not establish equilibration or an autocorrelation-corrected uncertainty."
    })
}

fn simulate(
    model: &QxyModel,
    warmup_sweeps: usize,
    measurement_sweeps: usize,
    seed: u64,
) -> Result<Value, String> {
    if measurement_sweeps < 2 {
        return Err("measurement_sweeps must be at least two".to_owned());
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut phases: Vec<f64> = (0..model.node_count)
        .map(|_| rng.gen_range(0.0..TAU))
        .collect();

    let mut total_cluster_updates = 0_usize;
    let mut total_cluster_sites = 0_usize;
    let mut metropolis_accepted = 0_usize;
    let mut metropolis_proposed = 0_usize;

    for _ in 0..warmup_sweeps {
        let (clusters, cluster_sites, accepted, proposed) =
            sweep(model, &mut phases, &mut rng);
        total_cluster_updates += clusters;
        total_cluster_sites += cluster_sites;
        metropolis_accepted += accepted;
        metropolis_proposed += proposed;
    }

    let mut action_values = Vec::with_capacity(measurement_sweeps);
    let mut stiffness_values = Vec::with_capacity(measurement_sweeps);
    let mut vortex_values = Vec::with_capacity(measurement_sweeps);
    let mut current_correlation_samples: Vec<Vec<f64>> = (0..model.imaginary_time_slices)
        .map(|_| Vec::with_capacity(measurement_sweeps))
        .collect();
    for _ in 0..measurement_sweeps {
        let (clusters, cluster_sites, accepted, proposed) =
            sweep(model, &mut phases, &mut rng);
        total_cluster_updates += clusters;
        total_cluster_sites += cluster_sites;
        metropolis_accepted += accepted;
        metropolis_proposed += proposed;
        let measured = measure(model, &phases);
        action_values.push(measured.action_per_spacetime_site);
        stiffness_values.push(measured.stiffness);
        vortex_values.push(measured.vortex_density);
        for (tau_index, current_product) in measured.current_correlation.iter().enumerate() {
            current_correlation_samples[tau_index].push(*current_product);
        }
    }

    let current_correlation_function: Vec<Value> = current_correlation_samples
        .iter()
        .enumerate()
        .map(|(tau_index, samples)| {
            json!({
                "imaginary_time_index": tau_index,
                "imaginary_time": tau_index as f64 * model.delta_tau,
                "sampling": sampling_summary(samples)
            })
        })
        .collect();

    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let mean_action = mean(&action_values);
    let mean_stiffness = mean(&stiffness_values);
    let mean_vortex_density = mean(&vortex_values);
    let mean_cluster_size = if total_cluster_updates == 0 {
        0.0
    } else {
        total_cluster_sites as f64 / total_cluster_updates as f64
    };
    let acceptance_rate = if metropolis_proposed == 0 {
        0.0
    } else {
        metropolis_accepted as f64 / metropolis_proposed as f64
    };

    Ok(json!({
        "status": "pass",
        "contract": "SC-QXY-PIMC-SMOKE-0.1",
        "model": "Ponticelli-2026 QXY phase representation",
        "source_model_id": "ponticelli-2026-qxy-2d-square-lattice-phase-representation",
        "instance_id": model.instance_id,
        "schema_sha256": model.schema_sha256,
        "fixture_sha256": model.fixture_sha256,
        "algorithm": "anisotropic_3d_xy_Wolff_plus_Metropolis",
        "energy_reference_t": model.t,
        "interaction_U": model.u,
        "U_over_t": model.u / model.t,
        "temperature": model.temperature,
        "temperature_unit": "t",
        "beta": model.beta,
        "imaginary_time_step": model.delta_tau,
        "imaginary_time_slices": model.imaginary_time_slices,
        "K_spatial": model.k_spatial,
        "K_imaginary_time": model.k_imaginary_time,
        "L": model.lattice_linear_size,
        "spacetime_site_count": model.node_count,
        "warmup_sweeps": warmup_sweeps,
        "measurement_sweeps": measurement_sweeps,
        "random_seed": seed,
        "cluster_updates_total": total_cluster_updates,
        "mean_cluster_size": mean_cluster_size,
        "metropolis_acceptance_rate": acceptance_rate,
        "mean_action_per_spacetime_site": mean_action,
        "mean_finite_size_stiffness_source_normalization": mean_stiffness,
        "mean_total_vortex_density": mean_vortex_density,
        "current_correlation_function_F_tau": current_correlation_function,
        "current_correlation_definition": "F(tau)=<I(tau) I(0)>; I(tau)=t*sum_positive_spatial_bonds sin(phi_i(tau)-phi_j(tau))",
        "current_correlation_sampling_note": "Fixed imaginary-time origin; correlator is averaged over independent QMC measurement configurations, not averaged over the imaginary-time origin.",
        "sampling_diagnostics": {
            "action_per_spacetime_site": sampling_summary(&action_values),
            "finite_size_stiffness": sampling_summary(&stiffness_values),
            "total_vortex_density": sampling_summary(&vortex_values),
            "uncertainty_status": "iid_standard_errors_are_diagnostic_only_and_do_not_correct_for_markov_chain_autocorrelation"
        },
        "claim_ceiling": "Small synthetic QXY path-integral smoke result only; no source reproduction, thermodynamic-limit BKT/QCP estimate, critical exponent, conductivity spectrum, measured Tc or material result."
    }))
}

fn ensemble_summary(values: &[f64]) -> Value {
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let deviation = sample_standard_deviation(values);
    let standard_error = deviation.map(|value| value / (values.len() as f64).sqrt());
    json!({
        "mean_across_independent_seeds": average,
        "between_seed_sample_standard_deviation": deviation,
        "between_seed_standard_error": standard_error,
        "chain_count": values.len(),
        "warning": "Between-seed dispersion does not correct within-chain autocorrelation or inadequate equilibration."
    })
}

fn numeric_field_values(receipts: &[Value], field: &str) -> Result<Vec<f64>, String> {
    receipts
        .iter()
        .map(|receipt| {
            receipt
                .get(field)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("replicate receipt missing numeric field {field}"))
        })
        .collect()
}

fn simulate_replicates(
    model: &QxyModel,
    warmup_sweeps: usize,
    measurement_sweeps: usize,
    base_seed: u64,
    replicate_count: usize,
) -> Result<Value, String> {
    if replicate_count == 0 {
        return Err("replicate_count must be at least one".to_owned());
    }
    let mut receipts = Vec::with_capacity(replicate_count);
    for replicate in 0..replicate_count {
        let seed = base_seed
            .checked_add(replicate as u64)
            .ok_or_else(|| "replicate seed range overflowed u64".to_owned())?;
        receipts.push(simulate(
            model,
            warmup_sweeps,
            measurement_sweeps,
            seed,
        )?);
    }
    if replicate_count == 1 {
        return Ok(receipts.remove(0));
    }

    let action = numeric_field_values(&receipts, "mean_action_per_spacetime_site")?;
    let stiffness =
        numeric_field_values(&receipts, "mean_finite_size_stiffness_source_normalization")?;
    let vortices = numeric_field_values(&receipts, "mean_total_vortex_density")?;

    let mut current_correlation_between_chains = Vec::with_capacity(model.imaginary_time_slices);
    for tau_index in 0..model.imaginary_time_slices {
        let mut values = Vec::with_capacity(replicate_count);
        for receipt in &receipts {
            let value = receipt
                .get("current_correlation_function_F_tau")
                .and_then(Value::as_array)
                .and_then(|profile| profile.get(tau_index))
                .and_then(|entry| entry.get("sampling"))
                .and_then(|sampling| sampling.get("mean"))
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("replicate receipt missing F(tau) at index {tau_index}"))?;
            values.push(value);
        }
        current_correlation_between_chains.push(json!({
            "imaginary_time_index": tau_index,
            "imaginary_time": tau_index as f64 * model.delta_tau,
            "between_chain_summary": ensemble_summary(&values)
        }));
    }

    Ok(json!({
        "status": "pass",
        "contract": "SC-QXY-INDEPENDENT-SEED-ENSEMBLE-0.1",
        "source_model_id": "ponticelli-2026-qxy-2d-square-lattice-phase-representation",
        "instance_id": model.instance_id,
        "schema_sha256": model.schema_sha256,
        "fixture_sha256": model.fixture_sha256,
        "base_seed": base_seed,
        "replicate_count": replicate_count,
        "warmup_sweeps_per_chain": warmup_sweeps,
        "measurement_sweeps_per_chain": measurement_sweeps,
        "between_chain_summary": {
            "action_per_spacetime_site": ensemble_summary(&action),
            "finite_size_stiffness": ensemble_summary(&stiffness),
            "total_vortex_density": ensemble_summary(&vortices),
            "current_correlation_F_tau": current_correlation_between_chains
        },
        "per_chain_receipts": receipts,
        "claim_ceiling": "Independent-seed software ensemble only; between-seed dispersion is not an uncertainty-qualified thermodynamic result, source reproduction, or material observation."
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --fixture <source-qxy-instance.json> [--warmup-sweeps <n>] [--measurement-sweeps <n>] [--seed <n>] [--replicates <n>] [--out <receipt.json>]"
    );
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut fixture_path = None;
    let mut output_path = None;
    let mut warmup_sweeps = 2_usize;
    let mut measurement_sweeps = 3_usize;
    let mut seed = None;
    let mut replicates = 1_usize;
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--fixture" | "--out" | "--warmup-sweeps"
            | "--measurement-sweeps" | "--seed" | "--replicates" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let value = &args[index + 1];
                match args[index].as_str() {
                    "--schema" => schema_path = Some(PathBuf::from(value)),
                    "--fixture" => fixture_path = Some(PathBuf::from(value)),
                    "--out" => output_path = Some(PathBuf::from(value)),
                    "--warmup-sweeps" => {
                        warmup_sweeps = value.parse().unwrap_or_else(|_| {
                            eprintln!("warmup-sweeps must be integer");
                            process::exit(2);
                        })
                    }
                    "--measurement-sweeps" => {
                        measurement_sweeps = value.parse().unwrap_or_else(|_| {
                            eprintln!("measurement-sweeps must be integer");
                            process::exit(2);
                        })
                    }
                    "--seed" => {
                        seed = Some(value.parse().unwrap_or_else(|_| {
                            eprintln!("seed must be unsigned integer");
                            process::exit(2);
                        }))
                    }
                    "--replicates" => {
                        replicates = value.parse().unwrap_or_else(|_| {
                            eprintln!("replicates must be an integer");
                            process::exit(2);
                        })
                    }
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
            unknown => {
                eprintln!("unknown argument: {unknown}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }

    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let schema_bytes = read(&schema_path);
    let fixture_bytes = read(&fixture_path);
    let model = QxyModel::from_bytes(&schema_bytes, &fixture_bytes).unwrap_or_else(|error| {
        eprintln!("SC-QXY-PIMC-FAIL: {error}");
        process::exit(1);
    });
    let run_seed = seed.unwrap_or(model.random_seed);
    let receipt = simulate_replicates(
        &model,
        warmup_sweeps,
        measurement_sweeps,
        run_seed,
        replicates,
    )
    .unwrap_or_else(|error| {
        eprintln!("SC-QXY-PIMC-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing QXY smoke receipt should not fail");
    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {error}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_quantum_phase_instance_v0_1.schema.json"),
        )
        .expect("quantum phase schema must be readable");
        let fixture = fs::read(
            root.join("data/research/superconductivity_quantum_phase_QXY_smoke_fixture_v0_1.json"),
        )
        .expect("QXY smoke fixture must be readable");
        (schema, fixture)
    }

    #[test]
    fn source_discretization_derives_anisotropic_couplings() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_quantum_phase_instance_v0_1.schema.json"),
        )
        .unwrap();
        let fixture = fs::read(
            root.join("data/research/superconductivity_quantum_phase_ponticelli_2026_U4_T03_input_fixture_v0_1.json"),
        )
        .unwrap();
        let model = QxyModel::from_bytes(&schema, &fixture).unwrap();
        assert!((model.k_spatial - model.t * model.delta_tau).abs() < 1.0e-12);
        assert!((model.k_imaginary_time - 1.0 / (model.u * model.delta_tau)).abs() < 1.0e-10);
        assert!((model.u / model.t - 4.0).abs() < 1.0e-12);
        assert_eq!(model.imaginary_time_slices, 1000);
    }

    #[test]
    fn small_qxy_pimc_smoke_returns_finite_observables() {
        let (schema, fixture) = inputs();
        let model = QxyModel::from_bytes(&schema, &fixture).unwrap();
        let receipt = simulate(&model, 1, 2, 19).unwrap();
        assert_eq!(receipt["status"], "pass");
        for field in [
            "mean_action_per_spacetime_site",
            "mean_finite_size_stiffness_source_normalization",
            "mean_total_vortex_density",
            "metropolis_acceptance_rate",
        ] {
            assert!(receipt[field].as_f64().unwrap().is_finite(), "{field}");
        }
    }

    #[test]
    fn qxy_smoke_is_deterministic_for_identical_seed() {
        let (schema, fixture) = inputs();
        let model = QxyModel::from_bytes(&schema, &fixture).unwrap();
        let first = simulate(&model, 1, 2, 23).unwrap();
        let second = simulate(&model, 1, 2, 23).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn aligned_phase_configuration_has_unit_source_stiffness_and_zero_vortices() {
        let (schema, fixture) = inputs();
        let model = QxyModel::from_bytes(&schema, &fixture).unwrap();
        let phases = vec![0.37; model.node_count];
        let measured = measure(&model, &phases);
        assert!((measured.stiffness - 1.0).abs() < 1.0e-10);
        assert!(measured.vortex_density.abs() < 1.0e-12);
        assert_eq!(measured.current_correlation.len(), model.imaginary_time_slices);
        assert!(measured.current_correlation.iter().all(|value| value.abs() < 1.0e-12));
        let expected_action = -(2.0 * model.k_spatial + model.k_imaginary_time);
        assert!(
            (measured.action_per_spacetime_site - expected_action).abs() < 1.0e-10
        );
    }

    #[test]
    fn imaginary_time_slices_must_match_beta() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["simulation"]["imaginary_time_step"] = json!(0.2);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(QxyModel::from_bytes(&schema, &mutated).is_err());
    }

    #[test]
    fn receipt_is_bound_to_exact_schema_and_fixture_bytes() {
        let (schema, fixture) = inputs();
        let model = QxyModel::from_bytes(&schema, &fixture).unwrap();
        assert_eq!(
            model.fixture_sha256,
            format!("{:x}", Sha256::digest(&fixture))
        );
        assert_eq!(
            model.schema_sha256,
            format!("{:x}", Sha256::digest(&schema))
        );
        assert!(!model.instance_id.is_empty());
    }

    #[test]
    fn sampling_diagnostics_report_variance_and_lag_one_dependence() {
        let values = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let deviation = sample_standard_deviation(&values).unwrap();
        let rho = lag_one_autocorrelation(&values).unwrap();
        assert!(deviation > 0.0);
        assert!(rho < -0.5);
        let summary = sampling_summary(&values);
        assert_eq!(summary["sample_count"], 6);
        assert!(summary["iid_standard_error_not_autocorrelation_corrected"].is_number());
    }

    #[test]
    fn sampling_diagnostics_do_not_invent_standard_error_for_one_sample() {
        assert!(sample_standard_deviation(&[1.0]).is_none());
        assert!(lag_one_autocorrelation(&[1.0, 1.0]).is_none());
        assert!(integrated_autocorrelation_time(&[1.0, 2.0, 3.0, 4.0, 5.0]).is_none());
    }

    #[test]
    fn integrated_autocorrelation_estimator_detects_persistent_series() {
        let values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0];
        let tau = integrated_autocorrelation_time(&values).unwrap();
        assert!(tau >= 0.5);
        let summary = sampling_summary(&values);
        assert!(summary["effective_sample_size_initial_positive_sequence"].as_f64().unwrap() <= 10.0);
    }

    #[test]
    fn independent_seed_ensemble_is_reproducible() {
        let (schema, fixture) = inputs();
        let model = QxyModel::from_bytes(&schema, &fixture).unwrap();
        let first = simulate_replicates(&model, 1, 2, 200, 3).unwrap();
        let second = simulate_replicates(&model, 1, 2, 200, 3).unwrap();
        assert_eq!(first, second);
        assert_eq!(first["replicate_count"], 3);
        assert_eq!(first["per_chain_receipts"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn zero_replicate_ensemble_is_rejected() {
        let (schema, fixture) = inputs();
        let model = QxyModel::from_bytes(&schema, &fixture).unwrap();
        assert!(simulate_replicates(&model, 1, 2, 10, 0).is_err());
    }

    #[test]
    fn nonzero_gauge_field_is_rejected_by_source_solver() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["graph"]["edges"][0]["gauge_link_phase_rad"] = json!(0.1);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(QxyModel::from_bytes(&schema, &mutated).is_err());
    }

    #[test]
    fn nonzero_offset_charge_is_rejected_by_source_solver() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["charging"]["offset_charges"][0] = json!(0.25);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(QxyModel::from_bytes(&schema, &mutated).is_err());
    }

    #[test]
    fn malformed_parallel_bond_topology_is_rejected() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        let duplicate_source = value["graph"]["edges"][0]["source"].clone();
        let duplicate_target = value["graph"]["edges"][0]["target"].clone();
        value["graph"]["edges"][1]["source"] = duplicate_source;
        value["graph"]["edges"][1]["target"] = duplicate_target;
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(QxyModel::from_bytes(&schema, &mutated).is_err());
    }

    #[test]
    fn source_model_rejects_nonuniform_josephson_energy() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["graph"]["edges"][0]["josephson_energy"] = json!(1.1);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(QxyModel::from_bytes(&schema, &mutated).is_err());
    }

    #[test]
    fn model_seed_is_used_when_no_cli_override_is_supplied() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_quantum_phase_instance_v0_1.schema.json"),
        )
        .unwrap();
        let mut fixture: Value = serde_json::from_slice(
            &fs::read(
                root.join("data/research/superconductivity_quantum_phase_QXY_smoke_fixture_v0_1.json"),
            )
            .unwrap(),
        )
        .unwrap();
        fixture["simulation"]["random_seed"] = json!(1234);
        let model = QxyModel::from_bytes(&schema, &serde_json::to_vec(&fixture).unwrap()).unwrap();
        assert_eq!(model.random_seed, 1234);
    }
}
