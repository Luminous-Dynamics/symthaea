use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn text_field<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {field}"))
}

fn number_field(value: &Value, field: &str) -> Result<f64, String> {
    let number = value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing or non-numeric field {field}"))?;
    if !number.is_finite() {
        return Err(format!("field {field} must be finite"));
    }
    Ok(number)
}

fn close_enough(actual: f64, expected: f64) -> bool {
    let scale = actual.abs().max(expected.abs()).max(1.0);
    (actual - expected).abs() <= 1.0e-10 * scale
}

fn validate_instance(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
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
            "phase-network instance schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }

    if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("this harness accepts synthetic fixture instances only".to_owned());
    }

    let model = fixture
        .get("model")
        .ok_or_else(|| "missing model".to_owned())?;
    if text_field(model, "pair_amplitude_mode")? != "fixed_not_simulated" {
        return Err("phase-only model must declare pair_amplitude_mode=fixed_not_simulated".to_owned());
    }
    if text_field(model, "temperature_unit")? != "coupling_energy_units" {
        return Err("phase-only temperature must remain in coupling-energy units".to_owned());
    }

    let graph = fixture
        .get("graph")
        .ok_or_else(|| "missing graph".to_owned())?;
    let dimension = graph
        .get("spatial_dimension")
        .and_then(Value::as_u64)
        .ok_or_else(|| "graph.spatial_dimension must be an unsigned integer".to_owned())?
        as usize;
    if !(1..=3).contains(&dimension) {
        return Err("graph.spatial_dimension must be between one and three".to_owned());
    }

    let embedding = text_field(graph, "embedding")?;
    if !matches!(embedding, "physical_coordinates" | "abstract_graph") {
        return Err(format!("unsupported graph embedding {embedding}"));
    }

    let nodes = graph
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| "graph.nodes must be an array".to_owned())?;
    if nodes.len() < 2 {
        return Err("phase network requires at least two nodes".to_owned());
    }
    let mut node_ids = HashSet::with_capacity(nodes.len());
    let mut node_positions = HashMap::with_capacity(nodes.len());
    for (index, node) in nodes.iter().enumerate() {
        let id = text_field(node, "node_id")
            .map_err(|error| format!("node[{index}]: {error}"))?;
        if !node_ids.insert(id.to_owned()) {
            return Err(format!("duplicate node_id: {id}"));
        }

        let position = node.get("position");
        match (embedding, position) {
            ("physical_coordinates", Some(Value::Array(coordinates))) => {
                if coordinates.len() != dimension {
                    return Err(format!(
                        "node {id}: coordinate dimension {} does not match graph dimension {dimension}",
                        coordinates.len()
                    ));
                }
                let mut values = Vec::with_capacity(dimension);
                for coordinate in coordinates {
                    let value = coordinate
                        .as_f64()
                        .ok_or_else(|| format!("node {id}: position must contain numeric values"))?;
                    if !value.is_finite() {
                        return Err(format!("node {id}: position values must be finite"));
                    }
                    values.push(value);
                }
                node_positions.insert(id.to_owned(), values);
            }
            ("physical_coordinates", _) => {
                return Err(format!("node {id}: physical embedding requires position coordinates"));
            }
            ("abstract_graph", Some(_)) => {
                return Err(format!(
                    "node {id}: abstract_graph must not silently carry physical coordinates"
                ));
            }
            ("abstract_graph", None) => {}
            _ => unreachable!(),
        }
    }

    let boundary = fixture
        .get("boundary_conditions")
        .ok_or_else(|| "missing boundary_conditions".to_owned())?;
    let boundary_kind = text_field(boundary, "kind")?;
    let periodic_axes = boundary
        .get("periodic_axes")
        .and_then(Value::as_array)
        .ok_or_else(|| "boundary_conditions.periodic_axes must be an array".to_owned())?;
    let mut axes = HashSet::with_capacity(periodic_axes.len());
    for axis in periodic_axes {
        let axis = axis
            .as_u64()
            .ok_or_else(|| "periodic_axes must contain non-negative integer axes".to_owned())?
            as usize;
        if axis >= dimension {
            return Err(format!(
                "periodic axis {axis} is outside declared dimension {dimension}"
            ));
        }
        if !axes.insert(axis) {
            return Err(format!("duplicate periodic axis {axis}"));
        }
    }
    match boundary_kind {
        "periodic" if axes.len() == dimension => {}
        "open" if axes.is_empty() => {}
        "mixed" if !axes.is_empty() && axes.len() < dimension => {}
        "periodic" => return Err("periodic boundary conditions must declare every spatial axis".to_owned()),
        "open" => return Err("open boundary conditions must not declare periodic axes".to_owned()),
        "mixed" => return Err("mixed boundary conditions require a non-empty proper subset of axes".to_owned()),
        other => return Err(format!("unsupported boundary condition kind {other}")),
    }

    let box_lengths: Vec<f64> = if embedding == "physical_coordinates" {
        let values = boundary
            .get("box_lengths")
            .and_then(Value::as_array)
            .ok_or_else(|| "physical embedding requires boundary_conditions.box_lengths".to_owned())?;
        if values.len() != dimension {
            return Err(format!(
                "box_lengths length {} does not match spatial dimension {dimension}",
                values.len()
            ));
        }
        let mut lengths = Vec::with_capacity(dimension);
        for value in values {
            let length = value
                .as_f64()
                .ok_or_else(|| "box_lengths must contain numeric values".to_owned())?;
            if !length.is_finite() || length <= 0.0 {
                return Err("box_lengths must contain finite positive values".to_owned());
            }
            lengths.push(length);
        }
        lengths
    } else {
        if boundary.get("box_lengths").is_some() {
            return Err("abstract_graph must not claim physical box lengths".to_owned());
        }
        Vec::new()
    };

    let edges = graph
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "graph.edges must be an array".to_owned())?;
    if edges.is_empty() {
        return Err("phase network requires at least one edge".to_owned());
    }
    let mut edge_ids = HashSet::with_capacity(edges.len());
    let mut total_coupling = 0.0_f64;
    for (index, edge) in edges.iter().enumerate() {
        let id = text_field(edge, "edge_id")
            .map_err(|error| format!("edge[{index}]: {error}"))?;
        if !edge_ids.insert(id.to_owned()) {
            return Err(format!("duplicate edge_id: {id}"));
        }

        let source = text_field(edge, "source")
            .map_err(|error| format!("edge {id}: {error}"))?;
        let target = text_field(edge, "target")
            .map_err(|error| format!("edge {id}: {error}"))?;
        if source == target {
            return Err(format!("edge {id}: self-loops are not admitted in this contract"));
        }
        if !node_ids.contains(source) || !node_ids.contains(target) {
            return Err(format!("edge {id}: source or target does not resolve to a declared node"));
        }

        let coupling = number_field(edge, "coupling_energy")
            .map_err(|error| format!("edge {id}: {error}"))?;
        if coupling <= 0.0 {
            return Err(format!("edge {id}: coupling_energy must be strictly positive"));
        }
        total_coupling += coupling;

        let gauge_phase = number_field(edge, "gauge_link_phase_rad")
            .map_err(|error| format!("edge {id}: {error}"))?;
        if !gauge_phase.is_finite() {
            return Err(format!("edge {id}: gauge-link phase must be finite"));
        }

        if embedding == "physical_coordinates" {
            let length = number_field(edge, "physical_edge_length")
                .map_err(|error| format!("edge {id}: physical embedding requires physical_edge_length: {error}"))?;
            if length <= 0.0 {
                return Err(format!("edge {id}: physical_edge_length must be positive"));
            }
            let image_shift = edge
                .get("periodic_image_shift")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("edge {id}: physical embedding requires periodic_image_shift"))?;
            if image_shift.len() != dimension {
                return Err(format!(
                    "edge {id}: periodic_image_shift length {} does not match spatial dimension {dimension}",
                    image_shift.len()
                ));
            }
            let source_position = node_positions
                .get(source)
                .ok_or_else(|| format!("edge {id}: source position does not resolve"))?;
            let target_position = node_positions
                .get(target)
                .ok_or_else(|| format!("edge {id}: target position does not resolve"))?;
            let mut squared_length = 0.0_f64;
            for axis in 0..dimension {
                let shift = image_shift[axis]
                    .as_i64()
                    .ok_or_else(|| format!("edge {id}: periodic_image_shift must contain integers"))?;
                if shift != 0 && !axes.contains(&axis) {
                    return Err(format!("edge {id}: image shift on non-periodic axis {axis}"));
                }
                let displacement =
                    target_position[axis] + (shift as f64) * box_lengths[axis] - source_position[axis];
                squared_length += displacement * displacement;
            }
            let derived_length = squared_length.sqrt();
            if !close_enough(derived_length, length) {
                return Err(format!(
                    "edge {id}: physical_edge_length {length} does not match image-aware coordinate distance {derived_length}"
                ));
            }
        } else if edge.get("periodic_image_shift").is_some() {
            return Err(format!(
                "edge {id}: abstract_graph must not declare physical periodic image shifts"
            ));
        }
    }
    if !total_coupling.is_finite() {
        return Err("sum of edge coupling energies overflowed".to_owned());
    }

    let twists = boundary
        .get("twist_angles_rad")
        .and_then(Value::as_array)
        .ok_or_else(|| "boundary_conditions.twist_angles_rad must be an array".to_owned())?;
    if twists.len() != dimension {
        return Err(format!(
            "twist_angles_rad length {} does not match spatial dimension {dimension}",
            twists.len()
        ));
    }
    for twist in twists {
        let value = twist
            .as_f64()
            .ok_or_else(|| "twist_angles_rad must contain numeric values".to_owned())?;
        if !value.is_finite() {
            return Err("twist_angles_rad values must be finite".to_owned());
        }
    }

    let budget = fixture
        .get("coupling_budget")
        .ok_or_else(|| "missing coupling_budget".to_owned())?;
    let declared_total = number_field(budget, "total_coupling_energy")?;
    let declared_per_node = number_field(budget, "per_node_coupling_budget")?;
    let calculated_per_node = total_coupling / nodes.len() as f64;
    if !close_enough(total_coupling, declared_total) {
        return Err(format!(
            "total coupling budget mismatch: edges sum to {total_coupling}, declared {declared_total}"
        ));
    }
    if !close_enough(calculated_per_node, declared_per_node) {
        return Err(format!(
            "per-node coupling budget mismatch: calculated {calculated_per_node}, declared {declared_per_node}"
        ));
    }

    let disorder = fixture
        .get("disorder")
        .ok_or_else(|| "missing disorder".to_owned())?;
    let disorder_strength = number_field(disorder, "strength")?;
    if disorder_strength < 0.0 {
        return Err("disorder strength must be non-negative".to_owned());
    }

    let digest = Sha256::digest(fixture_bytes);
    let input_sha256 = format!("{digest:x}");
    let instance_id = text_field(&fixture, "instance_id")?;

    Ok(json!({
        "status": "pass",
        "contract": "SC-PHASE-NETWORK-0.1",
        "instance_id": instance_id,
        "input_sha256": input_sha256,
        "spatial_dimension": dimension,
        "embedding": embedding,
        "topology_family": graph["topology_family"],
        "node_count": nodes.len(),
        "edge_count": edges.len(),
        "unique_node_count": node_ids.len(),
        "unique_edge_count": edge_ids.len(),
        "coupling_budget": {
            "edge_sum": total_coupling,
            "declared_total": declared_total,
            "calculated_per_node": calculated_per_node,
            "declared_per_node": declared_per_node
        },
        "boundary_conditions": {
            "kind": boundary_kind,
            "periodic_axis_count": axes.len(),
            "twist_axis_count": twists.len()
        },
        "checks": {
            "schema_valid": true,
            "fixture_only": true,
            "node_and_edge_ids_unique": true,
            "edge_endpoints_resolve": true,
            "self_loops_rejected": true,
            "embedding_dimensions_match": true,
            "boundary_axes_match_dimension": true,
            "twists_match_dimension": true,
            "coupling_budget_matches_edge_sum": true,
            "phase_only_pair_amplitude_is_explicit": true,
            "temperature_not_mapped_to_kelvin": true
        },
        "claim_ceiling": "Input-contract validation only; no XY/Josephson simulation, thermodynamic transition, Tc, or experimental material result is established."
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <phase-network-schema.json> --fixture <instance.json> [--out <receipt.json>]"
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
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--fixture" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--fixture" => fixture_path = Some(path),
                    "--out" => output_path = Some(path),
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
    let schema = read(&schema_path);
    let fixture = read(&fixture_path);
    let receipt = validate_instance(&schema, &fixture).unwrap_or_else(|error| {
        eprintln!("SC-PHASE-NETWORK-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing the phase-network receipt should not fail");

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

    fn inputs() -> (Vec<u8>, Value) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_phase_network_instance_v0_1.schema.json"),
        )
        .expect("phase-network schema must be readable");
        let fixture: Value = serde_json::from_slice(
            &fs::read(root.join("data/research/superconductivity_phase_network_fixture_v0_1.json"))
                .expect("phase-network fixture must be readable"),
        )
        .expect("phase-network fixture must parse");
        (schema, fixture)
    }

    fn audit_mutated(fixture: Value) -> Result<Value, String> {
        let (schema, _) = inputs();
        validate_instance(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn committed_square_lattice_fixture_passes() {
        let (schema, fixture) = inputs();
        let receipt = validate_instance(&schema, &serde_json::to_vec(&fixture).unwrap()).unwrap();
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["node_count"], 16);
        assert_eq!(receipt["edge_count"], 32);
        assert_eq!(receipt["coupling_budget"]["edge_sum"], 32.0);
        assert_eq!(receipt["coupling_budget"]["calculated_per_node"], 2.0);
    }

    #[test]
    fn duplicate_node_identity_fails_closed() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["nodes"][1]["node_id"] = json!("n00");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn duplicate_edge_identity_fails_closed() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["edges"][1]["edge_id"] = json!("e00r");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn unresolved_edge_endpoint_fails_closed() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["edges"][0]["target"] = json!("missing-node");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn self_loop_fails_closed() {
        let (_, mut fixture) = inputs();
        let source = fixture["graph"]["edges"][0]["source"].clone();
        fixture["graph"]["edges"][0]["target"] = source;
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn coupling_budget_must_match_edge_sum() {
        let (_, mut fixture) = inputs();
        fixture["coupling_budget"]["total_coupling_energy"] = json!(31.0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn per_node_budget_must_match_node_count() {
        let (_, mut fixture) = inputs();
        fixture["coupling_budget"]["per_node_coupling_budget"] = json!(3.0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn physical_coordinates_must_match_declared_dimension() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["nodes"][0]["position"] = json!([0.0]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn periodic_axes_must_resolve_within_dimension() {
        let (_, mut fixture) = inputs();
        fixture["boundary_conditions"]["periodic_axes"] = json!([0, 2]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn twist_vector_must_match_spatial_dimension() {
        let (_, mut fixture) = inputs();
        fixture["boundary_conditions"]["twist_angles_rad"] = json!([0.0]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn abstract_graph_cannot_claim_physical_positions() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["embedding"] = json!("abstract_graph");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn nonpositive_coupling_fails_closed() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["edges"][0]["coupling_energy"] = json!(0.0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn phase_only_model_cannot_claim_variable_pair_amplitude() {
        let (_, mut fixture) = inputs();
        fixture["model"]["pair_amplitude_mode"] = json!("dynamical");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn non_fixture_record_is_not_admitted_by_smoke_auditor() {
        let (_, mut fixture) = inputs();
        fixture["fixture_only"] = json!(false);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn periodic_image_shift_recovers_short_boundary_bond() {
        let (_, fixture) = inputs();
        let edge = fixture["graph"]["edges"]
            .as_array()
            .unwrap()
            .iter()
            .find(|edge| edge["edge_id"] == "e03r")
            .unwrap();
        assert_eq!(edge["physical_edge_length"], 1.0);
        assert_eq!(edge["periodic_image_shift"], json!([0, 1]));
        assert!(audit_mutated(fixture).is_ok());
    }

    #[test]
    fn incorrect_periodic_image_shift_fails_closed() {
        let (_, mut fixture) = inputs();
        let edge = fixture["graph"]["edges"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|edge| edge["edge_id"] == "e03r")
            .unwrap();
        edge["periodic_image_shift"] = json!([0, 0]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn physical_edge_length_must_match_image_aware_geometry() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["edges"][0]["physical_edge_length"] = json!(2.0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn abstract_graph_must_not_claim_physical_image_shifts() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["embedding"] = json!("abstract_graph");
        for node in fixture["graph"]["nodes"].as_array_mut().unwrap() {
            node.as_object_mut().unwrap().remove("position");
        }
        for edge in fixture["graph"]["edges"].as_array_mut().unwrap() {
            edge.as_object_mut().unwrap().remove("physical_edge_length");
            edge.as_object_mut().unwrap().remove("periodic_image_shift");
        }
        fixture["boundary_conditions"]
            .as_object_mut()
            .unwrap()
            .remove("box_lengths");
        assert!(audit_mutated(fixture).is_ok());
    }
}
