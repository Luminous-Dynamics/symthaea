use jsonschema::validator_for;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::env;
use std::f64::consts::{PI, TAU};
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone)]
struct Edge {
    source: usize,
    target: usize,
    coupling: f64,
    displacement: Vec<f64>,
}

struct Network {
    dimension: usize,
    node_count: usize,
    edges: Vec<Edge>,
    adjacency: Vec<Vec<(usize, usize)>>,
    box_lengths: Vec<f64>,
}

struct Measurement {
    energy_per_node: f64,
    magnetization: f64,
    current: Vec<f64>,
    diamagnetic: Vec<Vec<f64>>,
}

impl Network {
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
                "phase-network schema validation failed: {} at {}",
                error,
                error.instance_path()
            ));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("this Monte Carlo pilot accepts synthetic fixtures only".to_owned());
        }

        let model = fixture
            .get("model")
            .ok_or_else(|| "missing model".to_owned())?;
        if model.get("pair_amplitude_mode").and_then(Value::as_str)
            != Some("fixed_not_simulated")
        {
            return Err("phase-only XY solver requires fixed_not_simulated pair amplitude".to_owned());
        }
        if model.get("temperature_unit").and_then(Value::as_str)
            != Some("coupling_energy_units")
        {
            return Err("temperature must remain in coupling-energy units".to_owned());
        }

        let graph = fixture
            .get("graph")
            .ok_or_else(|| "missing graph".to_owned())?;
        let dimension = graph
            .get("spatial_dimension")
            .and_then(Value::as_u64)
            .ok_or_else(|| "spatial_dimension must be an integer".to_owned())?
            as usize;
        if !(2..=3).contains(&dimension) {
            return Err("Wolff smoke solver supports two- and three-dimensional embeddings only".to_owned());
        }
        if graph.get("embedding").and_then(Value::as_str) != Some("physical_coordinates") {
            return Err("helicity calculation requires physical-coordinate embedding".to_owned());
        }

        let boundary = fixture
            .get("boundary_conditions")
            .ok_or_else(|| "missing boundary_conditions".to_owned())?;
        if boundary.get("kind").and_then(Value::as_str) != Some("periodic") {
            return Err("helicity calculation requires periodic boundaries on every axis".to_owned());
        }
        let axes = boundary
            .get("periodic_axes")
            .and_then(Value::as_array)
            .ok_or_else(|| "periodic_axes must be an array".to_owned())?;
        if axes.len() != dimension {
            return Err("periodic boundary must include every spatial axis".to_owned());
        }
        for (index, axis) in axes.iter().enumerate() {
            if axis.as_u64() != Some(index as u64) {
                return Err("periodic_axes must contain each spatial axis exactly once".to_owned());
            }
        }

        let box_values = boundary
            .get("box_lengths")
            .and_then(Value::as_array)
            .ok_or_else(|| "physical periodic model requires box_lengths".to_owned())?;
        if box_values.len() != dimension {
            return Err("box_lengths length must match spatial_dimension".to_owned());
        }
        let mut box_lengths = Vec::with_capacity(dimension);
        for value in box_values {
            let length = value
                .as_f64()
                .ok_or_else(|| "box_lengths must contain numbers".to_owned())?;
            if !length.is_finite() || length <= 0.0 {
                return Err("box_lengths must be finite and positive".to_owned());
            }
            box_lengths.push(length);
        }

        let twists = boundary
            .get("twist_angles_rad")
            .and_then(Value::as_array)
            .ok_or_else(|| "twist_angles_rad must be an array".to_owned())?;
        if twists.len() != dimension {
            return Err("twist vector length must match spatial_dimension".to_owned());
        }
        if twists.iter().any(|value| {
            value
                .as_f64()
                .is_none_or(|angle| !angle.is_finite() || angle.abs() > 1.0e-12)
        }) {
            return Err("Wolff pilot supports zero applied twist only".to_owned());
        }

        let nodes = graph
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| "nodes must be an array".to_owned())?;
        if nodes.len() < 2 {
            return Err("network requires at least two nodes".to_owned());
        }
        let mut node_ids = HashMap::with_capacity(nodes.len());
        let mut positions = Vec::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            let id = node
                .get("node_id")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| format!("node[{index}] has no node_id"))?;
            if node_ids.insert(id.to_owned(), index).is_some() {
                return Err(format!("duplicate node_id {id}"));
            }
            let coordinates = node
                .get("position")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("node {id} requires position coordinates"))?;
            if coordinates.len() != dimension {
                return Err(format!("node {id} coordinate dimension mismatch"));
            }
            let mut position = Vec::with_capacity(dimension);
            for coordinate in coordinates {
                let value = coordinate
                    .as_f64()
                    .ok_or_else(|| format!("node {id} position must be numeric"))?;
                if !value.is_finite() {
                    return Err(format!("node {id} position must be finite"));
                }
                position.push(value);
            }
            positions.push(position);
        }

        let edge_values = graph
            .get("edges")
            .and_then(Value::as_array)
            .ok_or_else(|| "edges must be an array".to_owned())?;
        if edge_values.is_empty() {
            return Err("network requires at least one edge".to_owned());
        }
        let mut edges = Vec::with_capacity(edge_values.len());
        let mut edge_ids = HashSet::with_capacity(edge_values.len());
        let mut total_coupling = 0.0_f64;
        let mut adjacency = vec![Vec::new(); nodes.len()];
        for edge in edge_values {
            let id = edge
                .get("edge_id")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| "edge missing non-empty edge_id".to_owned())?;
            if !edge_ids.insert(id.to_owned()) {
                return Err(format!("duplicate edge_id {id}"));
            }
            let source_id = edge
                .get("source")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("edge {id} missing source"))?;
            let target_id = edge
                .get("target")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("edge {id} missing target"))?;
            let source = *node_ids
                .get(source_id)
                .ok_or_else(|| format!("edge {id} source does not resolve"))?;
            let target = *node_ids
                .get(target_id)
                .ok_or_else(|| format!("edge {id} target does not resolve"))?;
            if source == target {
                return Err(format!("edge {id} is a self-loop"));
            }
            let coupling = edge
                .get("coupling_energy")
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("edge {id} coupling_energy missing"))?;
            if !coupling.is_finite() || coupling <= 0.0 {
                return Err(format!("edge {id} coupling_energy must be positive and finite"));
            }
            let gauge = edge
                .get("gauge_link_phase_rad")
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("edge {id} gauge phase missing"))?;
            if !gauge.is_finite() || gauge.abs() > 1.0e-12 {
                return Err(format!(
                    "edge {id} has nonzero gauge phase; use a frustration-capable update kernel"
                ));
            }

            let image_shift = edge
                .get("periodic_image_shift")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("edge {id} periodic_image_shift missing"))?;
            if image_shift.len() != dimension {
                return Err(format!("edge {id} image-shift dimension mismatch"));
            }
            let mut displacement = Vec::with_capacity(dimension);
            for axis in 0..dimension {
                let shift = image_shift[axis]
                    .as_i64()
                    .ok_or_else(|| format!("edge {id} image shifts must be integers"))?;
                let delta = positions[target][axis]
                    + (shift as f64) * box_lengths[axis]
                    - positions[source][axis];
                displacement.push(delta);
            }
            let physical_length = edge
                .get("physical_edge_length")
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("edge {id} physical_edge_length missing"))?;
            let derived_length = displacement.iter().map(|value| value * value).sum::<f64>().sqrt();
            if (derived_length - physical_length).abs() > 1.0e-10 {
                return Err(format!("edge {id} physical length disagrees with image-aware geometry"));
            }

            let edge_index = edges.len();
            total_coupling += coupling;
            edges.push(Edge {
                source,
                target,
                coupling,
                displacement,
            });
            adjacency[source].push((target, edge_index));
            adjacency[target].push((source, edge_index));
        }

        let budget = fixture
            .get("coupling_budget")
            .ok_or_else(|| "missing coupling_budget".to_owned())?;
        let declared_total = budget
            .get("total_coupling_energy")
            .and_then(Value::as_f64)
            .ok_or_else(|| "coupling_budget.total_coupling_energy must be numeric".to_owned())?;
        let declared_per_node = budget
            .get("per_node_coupling_budget")
            .and_then(Value::as_f64)
            .ok_or_else(|| "coupling_budget.per_node_coupling_budget must be numeric".to_owned())?;
        let calculated_per_node = total_coupling / nodes.len() as f64;
        if !total_coupling.is_finite()
            || !declared_total.is_finite()
            || !declared_per_node.is_finite()
            || (total_coupling - declared_total).abs() > 1.0e-10
            || (calculated_per_node - declared_per_node).abs() > 1.0e-10
        {
            return Err("declared coupling budget does not match the edge sum and node count".to_owned());
        }

        Ok(Self {
            dimension,
            node_count: nodes.len(),
            edges,
            adjacency,
            box_lengths,
        })
    }
}

fn wolff_cluster(
    network: &Network,
    phases: &mut [f64],
    beta: f64,
    rng: &mut StdRng,
) -> usize {
    let axis_angle = rng.gen_range(0.0..TAU);
    let seed = rng.gen_range(0..network.node_count);
    let mut in_cluster = vec![false; network.node_count];
    let mut queue = std::collections::VecDeque::new();
    let mut cluster = Vec::new();

    in_cluster[seed] = true;
    queue.push_back(seed);
    cluster.push(seed);

    while let Some(node) = queue.pop_front() {
        let projection_node = (phases[node] - axis_angle).cos();
        for (neighbor, edge_index) in &network.adjacency[node] {
            if in_cluster[*neighbor] {
                continue;
            }
            let edge = &network.edges[*edge_index];
            let projection_neighbor = (phases[*neighbor] - axis_angle).cos();
            let product = projection_node * projection_neighbor;
            if product <= 0.0 {
                continue;
            }
            let probability = 1.0 - (-2.0 * beta * edge.coupling * product).exp();
            if rng.gen_range(0.0..1.0) < probability {
                in_cluster[*neighbor] = true;
                queue.push_back(*neighbor);
                cluster.push(*neighbor);
            }
        }
    }

    for node in &cluster {
        phases[*node] = (2.0 * axis_angle + PI - phases[*node]).rem_euclid(TAU);
    }
    cluster.len()
}

fn sweep(network: &Network, phases: &mut [f64], beta: f64, rng: &mut StdRng) -> usize {
    let mut flipped = 0;
    let mut cluster_updates = 0;
    while flipped < network.node_count {
        flipped += wolff_cluster(network, phases, beta, rng);
        cluster_updates += 1;
    }
    cluster_updates
}

fn measure(network: &Network, phases: &[f64]) -> Measurement {
    let mut energy = 0.0;
    let mut magnetization_x = 0.0;
    let mut magnetization_y = 0.0;
    let mut current = vec![0.0; network.dimension];
    let mut diamagnetic = vec![vec![0.0; network.dimension]; network.dimension];

    for phase in phases {
        magnetization_x += phase.cos();
        magnetization_y += phase.sin();
    }
    for edge in &network.edges {
        let delta = phases[edge.source] - phases[edge.target];
        let cosine = delta.cos();
        let sine = delta.sin();
        energy -= edge.coupling * cosine;

        for alpha in 0..network.dimension {
            current[alpha] += edge.coupling
                * edge.displacement[alpha]
                / network.box_lengths[alpha]
                * sine;
            for beta in 0..network.dimension {
                diamagnetic[alpha][beta] += edge.coupling
                    * edge.displacement[alpha]
                    * edge.displacement[beta]
                    / (network.box_lengths[alpha] * network.box_lengths[beta])
                    * cosine;
            }
        }
    }

    Measurement {
        energy_per_node: energy / network.node_count as f64,
        magnetization: magnetization_x.hypot(magnetization_y) / network.node_count as f64,
        current,
        diamagnetic,
    }
}

fn lag_one_autocorrelation(values: &[f64]) -> Option<f64> {
    if values.len() < 3 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    if variance <= f64::EPSILON {
        return None;
    }
    let covariance = values
        .windows(2)
        .map(|pair| (pair[0] - mean) * (pair[1] - mean))
        .sum::<f64>();
    Some(covariance / variance)
}

fn simulate(
    network: &Network,
    temperature: f64,
    warmup_sweeps: usize,
    measurement_sweeps: usize,
    seed: u64,
) -> Result<Value, String> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("temperature must be finite and positive".to_owned());
    }
    if measurement_sweeps < 2 {
        return Err("measurement_sweeps must be at least two for diagnostics".to_owned());
    }
    let beta = 1.0 / temperature;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut phases: Vec<f64> = (0..network.node_count)
        .map(|_| rng.gen_range(0.0..TAU))
        .collect();

    let mut cluster_updates = 0_usize;
    for _ in 0..warmup_sweeps {
        cluster_updates += sweep(network, &mut phases, beta, &mut rng);
    }

    let mut measurements = Vec::with_capacity(measurement_sweeps);
    let mut energy_values = Vec::with_capacity(measurement_sweeps);
    for _ in 0..measurement_sweeps {
        cluster_updates += sweep(network, &mut phases, beta, &mut rng);
        let measurement = measure(network, &phases);
        energy_values.push(measurement.energy_per_node);
        measurements.push(measurement);
    }

    let sample_count = measurements.len() as f64;
    let mean_energy = measurements
        .iter()
        .map(|measurement| measurement.energy_per_node)
        .sum::<f64>()
        / sample_count;
    let mean_magnetization = measurements
        .iter()
        .map(|measurement| measurement.magnetization)
        .sum::<f64>()
        / sample_count;

    let volume = network.box_lengths.iter().product::<f64>();
    let mut mean_diamagnetic = vec![vec![0.0; network.dimension]; network.dimension];
    let mut mean_current = vec![0.0; network.dimension];
    let mut mean_current_products = vec![vec![0.0; network.dimension]; network.dimension];
    for measurement in &measurements {
        for alpha in 0..network.dimension {
            mean_current[alpha] += measurement.current[alpha] / sample_count;
            for beta in 0..network.dimension {
                mean_diamagnetic[alpha][beta] +=
                    measurement.diamagnetic[alpha][beta] / sample_count;
                mean_current_products[alpha][beta] +=
                    measurement.current[alpha] * measurement.current[beta] / sample_count;
            }
        }
    }

    let mut stiffness = vec![vec![0.0; network.dimension]; network.dimension];
    for alpha in 0..network.dimension {
        for beta in 0..network.dimension {
            let covariance = mean_current_products[alpha][beta]
                - mean_current[alpha] * mean_current[beta];
            let free_energy_curvature =
                mean_diamagnetic[alpha][beta] - beta * covariance;
            stiffness[alpha][beta] = network.box_lengths[alpha]
                * network.box_lengths[beta]
                / volume
                * free_energy_curvature;
        }
    }

    let energy_variance = measurements
        .iter()
        .map(|measurement| (measurement.energy_per_node - mean_energy).powi(2))
        .sum::<f64>()
        / sample_count;

    Ok(json!({
        "status": "pass",
        "contract": "SC-PHASE-NETWORK-MC-0.1",
        "model": "classical_xy_phase_only",
        "update_kernel": "wolff_embedded_cluster_zero_gauge_unfrustrated",
        "temperature": temperature,
        "temperature_unit": "coupling_energy_units",
        "kB_convention": "kB_equals_1",
        "random_seed": seed,
        "warmup_sweeps": warmup_sweeps,
        "measurement_sweeps": measurement_sweeps,
        "cluster_updates_total": cluster_updates,
        "node_count": network.node_count,
        "edge_count": network.edges.len(),
        "spatial_dimension": network.dimension,
        "mean_energy_per_node": mean_energy,
        "energy_variance_per_node_squared": energy_variance,
        "mean_finite_size_magnetization": mean_magnetization,
        "lag_one_energy_autocorrelation": lag_one_autocorrelation(&energy_values),
        "helicity_modulus_tensor": stiffness,
        "claim_ceiling": "Synthetic phase-only Monte Carlo smoke result; not a material prediction, thermodynamic-limit transition, measured Tc, or evidence of superconductivity."
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --fixture <instance.json> [--temperature <value>] [--warmup-sweeps <n>] [--measurement-sweeps <n>] [--seed <n>] [--out <receipt.json>]"
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
    let mut temperature = 1.0_f64;
    let mut warmup_sweeps = 20_usize;
    let mut measurement_sweeps = 40_usize;
    let mut seed = 7_u64;
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--fixture" | "--out" | "--temperature" | "--warmup-sweeps"
            | "--measurement-sweeps" | "--seed" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let value = &args[index + 1];
                match args[index].as_str() {
                    "--schema" => schema_path = Some(PathBuf::from(value)),
                    "--fixture" => fixture_path = Some(PathBuf::from(value)),
                    "--out" => output_path = Some(PathBuf::from(value)),
                    "--temperature" => {
                        temperature = value.parse().unwrap_or_else(|_| {
                            eprintln!("temperature must be numeric");
                            process::exit(2);
                        })
                    }
                    "--warmup-sweeps" => {
                        warmup_sweeps = value.parse().unwrap_or_else(|_| {
                            eprintln!("warmup-sweeps must be an integer");
                            process::exit(2);
                        })
                    }
                    "--measurement-sweeps" => {
                        measurement_sweeps = value.parse().unwrap_or_else(|_| {
                            eprintln!("measurement-sweeps must be an integer");
                            process::exit(2);
                        })
                    }
                    "--seed" => {
                        seed = value.parse().unwrap_or_else(|_| {
                            eprintln!("seed must be an unsigned integer");
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
    let network = Network::from_bytes(&schema_bytes, &fixture_bytes).unwrap_or_else(|error| {
        eprintln!("SC-PHASE-NETWORK-MC-FAIL: {error}");
        process::exit(1);
    });
    let receipt = simulate(
        &network,
        temperature,
        warmup_sweeps,
        measurement_sweeps,
        seed,
    )
    .unwrap_or_else(|error| {
        eprintln!("SC-PHASE-NETWORK-MC-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing the XY Monte Carlo receipt should not fail");

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
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_phase_network_instance_v0_1.schema.json"),
        )
        .expect("phase-network schema must be readable");
        let fixture = fs::read(
            root.join("data/research/superconductivity_phase_network_fixture_v0_1.json"),
        )
        .expect("phase-network fixture must be readable");
        (schema, fixture)
    }

    #[test]
    fn short_cluster_pilot_emits_finite_stiffness() {
        let (schema, fixture) = inputs();
        let network = Network::from_bytes(&schema, &fixture).unwrap();
        let receipt = simulate(&network, 1.0, 3, 5, 7).unwrap();
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["node_count"], 16);
        assert_eq!(receipt["measurement_sweeps"], 5);
        for row in receipt["helicity_modulus_tensor"].as_array().unwrap() {
            for value in row.as_array().unwrap() {
                assert!(value.as_f64().unwrap().is_finite());
            }
        }
    }

    #[test]
    fn identical_seed_and_inputs_are_deterministic() {
        let (schema, fixture) = inputs();
        let network = Network::from_bytes(&schema, &fixture).unwrap();
        let first = simulate(&network, 1.0, 3, 5, 42).unwrap();
        let second = simulate(&network, 1.0, 3, 5, 42).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn nonzero_gauge_link_is_not_silently_treated_as_ferromagnetic() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["graph"]["edges"][0]["gauge_link_phase_rad"] = json!(0.2);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(Network::from_bytes(&schema, &mutated).is_err());
    }

    #[test]
    fn nonzero_external_twist_is_not_silently_ignored() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["boundary_conditions"]["twist_angles_rad"] = json!([0.1, 0.0]);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(Network::from_bytes(&schema, &mutated).is_err());
    }

    #[test]
    fn zero_or_negative_temperature_fails_closed() {
        let (schema, fixture) = inputs();
        let network = Network::from_bytes(&schema, &fixture).unwrap();
        assert!(simulate(&network, 0.0, 1, 2, 1).is_err());
        assert!(simulate(&network, -1.0, 1, 2, 1).is_err());
    }

    #[test]
    fn duplicate_edge_ids_are_rejected_by_solver() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["graph"]["edges"][1]["edge_id"] = json!("e00r");
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(Network::from_bytes(&schema, &mutated).is_err());
    }

    #[test]
    fn solver_rechecks_coupling_budget() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["coupling_budget"]["total_coupling_energy"] = json!(99.0);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(Network::from_bytes(&schema, &mutated).is_err());
    }

    #[test]
    fn duplicate_node_ids_are_rejected_by_solver() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["graph"]["nodes"][1]["node_id"] = json!("n00");
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(Network::from_bytes(&schema, &mutated).is_err());
    }

    #[test]
    fn finite_size_fixture_ladder_resolves_8x8_and_16x16_graphs() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_phase_network_instance_v0_1.schema.json"),
        )
        .unwrap();
        for (filename, expected_nodes, expected_edges) in [
            ("superconductivity_phase_network_square_8x8_fixture_v0_1.json", 64, 128),
            ("superconductivity_phase_network_square_16x16_fixture_v0_1.json", 256, 512),
        ] {
            let fixture = fs::read(root.join("data/research").join(filename)).unwrap();
            let network = Network::from_bytes(&schema, &fixture).unwrap();
            assert_eq!(network.node_count, expected_nodes);
            assert_eq!(network.edges.len(), expected_edges);
        }
    }
}
