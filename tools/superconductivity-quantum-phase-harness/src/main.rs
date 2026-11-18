use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn required_text<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty field {field}"))
}

fn finite_number(value: &Value, context: &str) -> Result<f64, String> {
    let number = value
        .as_f64()
        .ok_or_else(|| format!("{context} must be numeric"))?;
    if !number.is_finite() {
        return Err(format!("{context} must be finite"));
    }
    Ok(number)
}

fn close_enough(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= 1.0e-10 * scale
}

fn validate_positive_definite_matrix(matrix: &[Vec<f64>]) -> Result<(), String> {
    let size = matrix.len();
    if size < 2 || matrix.iter().any(|row| row.len() != size) {
        return Err("charging matrix must be square and at least 2x2".to_owned());
    }
    for row in matrix {
        for value in row {
            if !value.is_finite() {
                return Err("charging matrix entries must be finite".to_owned());
            }
        }
    }

    for i in 0..size {
        for j in 0..i {
            if !close_enough(matrix[i][j], matrix[j][i]) {
                return Err(format!(
                    "charging matrix must be symmetric: U[{i},{j}] != U[{j},{i}]"
                ));
            }
        }
    }

    // Cholesky factorization: U = L L^T. Requiring strict positive
    // definiteness avoids silently admitting an unconstrained charge mode.
    let mut lower = vec![vec![0.0_f64; size]; size];
    for i in 0..size {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for k in 0..j {
                sum -= lower[i][k] * lower[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1.0e-12 {
                    return Err(format!(
                        "charging matrix must be positive definite; Cholesky pivot {i} is {sum}"
                    ));
                }
                lower[i][j] = sum.sqrt();
            } else {
                lower[i][j] = sum / lower[j][j];
            }
        }
    }
    Ok(())
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
            "quantum-phase schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }

    if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("this initial auditor admits synthetic fixture inputs only".to_owned());
    }

    let model = fixture
        .get("model")
        .ok_or_else(|| "missing model".to_owned())?;
    if required_text(model, "hbar_convention")? != "hbar_equals_1"
        || required_text(model, "kB_convention")? != "kB_equals_1"
        || required_text(model, "energy_unit")? != "coupling_energy_units"
        || required_text(model, "temperature_unit")? != "coupling_energy_units"
        || required_text(model, "charge_convention")? != "integer_cooper_pair_number"
        || required_text(model, "pair_charge_convention")? != "2e_equals_1"
        || required_text(model, "phase_compactification")? != "2pi"
    {
        return Err("unsupported quantum-phase units or operator conventions".to_owned());
    }
    let model_kind = required_text(model, "kind")?;
    if model_kind == "source_reproduction"
        && model.get("source_model_id").and_then(Value::as_str).is_none()
    {
        return Err("source_reproduction requires source_model_id".to_owned());
    }

    let graph = fixture
        .get("graph")
        .ok_or_else(|| "missing graph".to_owned())?;
    let dimension = graph
        .get("spatial_dimension")
        .and_then(Value::as_u64)
        .ok_or_else(|| "spatial_dimension must be an integer".to_owned())?
        as usize;
    if !(1..=3).contains(&dimension) {
        return Err("spatial_dimension must be between one and three".to_owned());
    }
    let box_lengths = graph
        .get("box_lengths")
        .and_then(Value::as_array)
        .ok_or_else(|| "graph.box_lengths must be an array".to_owned())?;
    if box_lengths.len() != dimension {
        return Err("box_lengths length must match spatial_dimension".to_owned());
    }
    let mut lengths = Vec::with_capacity(dimension);
    for value in box_lengths {
        let length = finite_number(value, "box length")?;
        if length <= 0.0 {
            return Err("box lengths must be positive".to_owned());
        }
        lengths.push(length);
    }

    let periodic_axes = graph
        .get("periodic_axes")
        .and_then(Value::as_array)
        .ok_or_else(|| "graph.periodic_axes must be an array".to_owned())?;
    let mut periodic = HashSet::with_capacity(periodic_axes.len());
    for value in periodic_axes {
        let axis = value
            .as_u64()
            .ok_or_else(|| "periodic axes must be non-negative integers".to_owned())?
            as usize;
        if axis >= dimension || !periodic.insert(axis) {
            return Err(format!("invalid or duplicate periodic axis {axis}"));
        }
    }

    let nodes = graph
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| "graph.nodes must be an array".to_owned())?;
    if nodes.len() < 2 {
        return Err("quantum phase graph requires at least two nodes".to_owned());
    }
    let mut node_indices = HashMap::with_capacity(nodes.len());
    let mut positions = Vec::with_capacity(nodes.len());
    for (index, node) in nodes.iter().enumerate() {
        let id = required_text(node, "node_id")
            .map_err(|error| format!("node[{index}]: {error}"))?;
        if node_indices.insert(id.to_owned(), index).is_some() {
            return Err(format!("duplicate node_id {id}"));
        }
        let coordinates = node
            .get("position")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("node {id} requires position"))?;
        if coordinates.len() != dimension {
            return Err(format!("node {id} coordinate dimension mismatch"));
        }
        let mut position = Vec::with_capacity(dimension);
        for value in coordinates {
            position.push(finite_number(value, &format!("node {id} coordinate"))?);
        }
        positions.push(position);
    }

    let edges = graph
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "graph.edges must be an array".to_owned())?;
    if edges.is_empty() {
        return Err("quantum phase graph requires at least one edge".to_owned());
    }
    let mut edge_ids = HashSet::with_capacity(edges.len());
    let mut total_josephson_energy = 0.0_f64;
    let mut any_nonzero_gauge_link = false;
    for edge in edges {
        let id = required_text(edge, "edge_id")?;
        if !edge_ids.insert(id.to_owned()) {
            return Err(format!("duplicate edge_id {id}"));
        }
        let source_id = required_text(edge, "source")?;
        let target_id = required_text(edge, "target")?;
        let source = *node_indices
            .get(source_id)
            .ok_or_else(|| format!("edge {id}: unresolved source {source_id}"))?;
        let target = *node_indices
            .get(target_id)
            .ok_or_else(|| format!("edge {id}: unresolved target {target_id}"))?;
        if source == target {
            return Err(format!("edge {id}: self-loops are not admitted"));
        }

        let josephson = finite_number(
            edge.get("josephson_energy")
                .ok_or_else(|| format!("edge {id}: missing josephson_energy"))?,
            "josephson_energy",
        )?;
        if josephson <= 0.0 {
            return Err(format!("edge {id}: josephson_energy must be positive"));
        }
        total_josephson_energy += josephson;

        let gauge = finite_number(
            edge.get("gauge_link_phase_rad")
                .ok_or_else(|| format!("edge {id}: missing gauge_link_phase_rad"))?,
            "gauge_link_phase_rad",
        )?;
        any_nonzero_gauge_link |= gauge.abs() > 1.0e-12;

        let image_shift = edge
            .get("periodic_image_shift")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("edge {id}: missing periodic_image_shift"))?;
        if image_shift.len() != dimension {
            return Err(format!("edge {id}: periodic image-shift dimension mismatch"));
        }
        let mut squared_distance = 0.0_f64;
        for axis in 0..dimension {
            let shift = image_shift[axis]
                .as_i64()
                .ok_or_else(|| format!("edge {id}: image shifts must be integers"))?;
            if shift != 0 && !periodic.contains(&axis) {
                return Err(format!("edge {id}: image shift on non-periodic axis {axis}"));
            }
            let displacement =
                positions[target][axis] + (shift as f64) * lengths[axis] - positions[source][axis];
            squared_distance += displacement * displacement;
        }
        let derived_length = squared_distance.sqrt();
        let declared_length = finite_number(
            edge.get("physical_edge_length")
                .ok_or_else(|| format!("edge {id}: missing physical_edge_length"))?,
            "physical_edge_length",
        )?;
        if declared_length <= 0.0 || !close_enough(derived_length, declared_length) {
            return Err(format!(
                "edge {id}: physical edge length disagrees with periodic-image geometry"
            ));
        }
    }
    if !total_josephson_energy.is_finite() {
        return Err("Josephson energy sum overflowed".to_owned());
    }

    let charging = fixture
        .get("charging")
        .ok_or_else(|| "missing charging data".to_owned())?;
    if required_text(charging, "energy_unit")? != "coupling_energy_units" {
        return Err("charging matrix energy unit mismatch".to_owned());
    }
    let matrix_kind = required_text(charging, "matrix_kind")?;
    let matrix_dimension = nodes.len();
    match matrix_kind {
        "diagonal_scalar" => {
            let diagonal_value = finite_number(
                charging
                    .get("diagonal_value")
                    .ok_or_else(|| "diagonal_scalar charging requires diagonal_value".to_owned())?,
                "diagonal_value",
            )?;
            if diagonal_value <= 0.0 {
                return Err("diagonal_scalar charging energy must be positive".to_owned());
            }
            if charging.get("matrix_u_ij").is_some() {
                return Err(
                    "diagonal_scalar charging must not expand a scalar into a dense matrix"
                        .to_owned(),
                );
            }
        }
        "diagonal_positive" | "dense_symmetric_positive_definite" => {
            let matrix_values = charging
                .get("matrix_u_ij")
                .and_then(Value::as_array)
                .ok_or_else(|| "charging.matrix_u_ij must be an array".to_owned())?;
            if matrix_values.len() != nodes.len() {
                return Err("charging matrix row count must match node count".to_owned());
            }
            let mut matrix = Vec::with_capacity(nodes.len());
            for (row_index, row) in matrix_values.iter().enumerate() {
                let values = row
                    .as_array()
                    .ok_or_else(|| format!("charging matrix row {row_index} must be an array"))?;
                if values.len() != nodes.len() {
                    return Err(format!(
                        "charging matrix row {row_index} length must match node count"
                    ));
                }
                let mut parsed = Vec::with_capacity(nodes.len());
                for value in values {
                    parsed.push(finite_number(value, "charging matrix entry")?);
                }
                matrix.push(parsed);
            }
            validate_positive_definite_matrix(&matrix)?;

            if matrix_kind == "diagonal_positive" {
                for i in 0..matrix.len() {
                    for j in 0..matrix.len() {
                        if i != j && matrix[i][j].abs() > 1.0e-12 {
                            return Err(
                                "diagonal_positive charging matrix contains an off-diagonal term"
                                    .to_owned(),
                            );
                        }
                    }
                }
            }
            if charging.get("diagonal_value").is_some() {
                return Err(
                    "matrix charging representations must not also declare diagonal_value"
                        .to_owned(),
                );
            }
        }
        other => return Err(format!("unsupported charging matrix kind {other}")),
    }

    let offset_charges = charging
        .get("offset_charges")
        .and_then(Value::as_array)
        .ok_or_else(|| "charging.offset_charges must be an array".to_owned())?;
    if offset_charges.len() != nodes.len() {
        return Err("offset charge count must match node count".to_owned());
    }
    for value in offset_charges {
        finite_number(value, "offset charge")?;
    }

    let simulation = fixture
        .get("simulation")
        .ok_or_else(|| "missing simulation".to_owned())?;
    if required_text(simulation, "temperature_unit")? != "coupling_energy_units" {
        return Err("simulation temperature unit mismatch".to_owned());
    }
    let temperature = finite_number(
        simulation
            .get("temperature")
            .ok_or_else(|| "missing simulation temperature".to_owned())?,
        "simulation temperature",
    )?;
    if temperature <= 0.0 {
        return Err("simulation temperature must be positive".to_owned());
    }
    let slices = simulation
        .get("imaginary_time_slices")
        .and_then(Value::as_u64)
        .ok_or_else(|| "imaginary_time_slices must be an integer".to_owned())?;
    if slices < 4 {
        return Err("imaginary_time_slices must be at least four".to_owned());
    }
    let imaginary_time_step = finite_number(
        simulation
            .get("imaginary_time_step")
            .ok_or_else(|| "missing imaginary_time_step".to_owned())?,
        "imaginary_time_step",
    )?;
    if imaginary_time_step <= 0.0
        || !close_enough(slices as f64 * imaginary_time_step, 1.0 / temperature)
    {
        return Err(
            "imaginary_time_slices * imaginary_time_step must equal beta = 1/temperature"
                .to_owned(),
        );
    }
    let representation = required_text(simulation, "representation")?;
    let cutoff = simulation
        .get("charge_basis_cutoff")
        .ok_or_else(|| "missing charge_basis_cutoff declaration".to_owned())?;
    let cutoff_mode = required_text(cutoff, "mode")?;
    let charge_cutoff = match representation {
        "phase_path_integral" => {
            if cutoff_mode != "not_applicable"
                || !cutoff
                    .get("max_abs_charge_state")
                    .is_some_and(Value::is_null)
            {
                return Err(
                    "phase_path_integral representation must declare no charge-basis cutoff"
                        .to_owned(),
                );
            }
            None
        }
        "integer_charge_basis" => {
            if cutoff_mode != "symmetric_integer" {
                return Err(
                    "integer_charge_basis requires symmetric_integer truncation".to_owned(),
                );
            }
            let value = cutoff
                .get("max_abs_charge_state")
                .and_then(Value::as_u64)
                .filter(|value| *value > 0)
                .ok_or_else(|| "integer charge basis requires a positive max_abs_charge_state".to_owned())?;
            Some(value)
        }
        other => return Err(format!("unsupported quantum representation {other}")),
    };
    simulation
        .get("random_seed")
        .and_then(Value::as_u64)
        .ok_or_else(|| "random_seed must be an unsigned integer".to_owned())?;

    let bath = fixture
        .get("bath")
        .ok_or_else(|| "missing bath declaration".to_owned())?;
    let bath_mode = required_text(bath, "mode")?;
    let bath_operator = required_text(bath, "coupling_operator")?;
    let bath_strength = finite_number(
        bath.get("coupling_strength")
            .ok_or_else(|| "missing bath coupling_strength".to_owned())?,
        "bath coupling_strength",
    )?;
    if bath_strength < 0.0 {
        return Err("bath coupling_strength must be non-negative".to_owned());
    }
    let spectral_density = bath
        .get("spectral_density")
        .and_then(Value::as_array)
        .ok_or_else(|| "bath.spectral_density must be an array".to_owned())?;
    match bath_mode {
        "none" => {
            if bath_operator != "none"
                || bath_strength != 0.0
                || !spectral_density.is_empty()
                || !bath.get("low_frequency_exponent").is_some_and(Value::is_null)
                || !bath.get("cutoff_energy").is_some_and(Value::is_null)
            {
                return Err("bath mode none requires zero coupling, null exponents/cutoff and an empty spectrum".to_owned());
            }
        }
        "analytic_power_law" => {
            if bath_operator == "none" || bath_strength <= 0.0 || !spectral_density.is_empty() {
                return Err("analytic_power_law bath requires an operator, positive coupling and no tabulated spectrum".to_owned());
            }
            let exponent = finite_number(
                bath.get("low_frequency_exponent")
                    .ok_or_else(|| "analytic bath missing low_frequency_exponent".to_owned())?,
                "bath low_frequency_exponent",
            )?;
            let cutoff = finite_number(
                bath.get("cutoff_energy")
                    .ok_or_else(|| "analytic bath missing cutoff_energy".to_owned())?,
                "bath cutoff_energy",
            )?;
            if exponent <= 0.0 || cutoff <= 0.0 {
                return Err("analytic bath exponent and cutoff must be positive".to_owned());
            }
        }
        "tabulated" => {
            if bath_operator == "none" || bath_strength <= 0.0 || spectral_density.len() < 2 {
                return Err("tabulated bath requires an operator, positive coupling and at least two spectral points".to_owned());
            }
            let mut previous_omega = 0.0_f64;
            for point in spectral_density {
                let omega = finite_number(
                    point.get("omega").ok_or_else(|| "spectral point missing omega".to_owned())?,
                    "bath omega",
                )?;
                let density = finite_number(
                    point.get("density").ok_or_else(|| "spectral point missing density".to_owned())?,
                    "bath density",
                )?;
                if omega <= previous_omega || density < 0.0 {
                    return Err("tabulated bath frequencies must be strictly increasing and densities non-negative".to_owned());
                }
                previous_omega = omega;
            }
            finite_number(
                bath.get("cutoff_energy")
                    .ok_or_else(|| "tabulated bath missing cutoff_energy".to_owned())?,
                "bath cutoff_energy",
            )?;
        }
        other => return Err(format!("unsupported bath mode {other}")),
    }

    let external_flux = fixture
        .get("external_flux")
        .ok_or_else(|| "missing external_flux declaration".to_owned())?;
    let flux_mode = required_text(external_flux, "mode")?;
    let total_flux = finite_number(
        external_flux
            .get("total_flux_quanta")
            .ok_or_else(|| "missing total_flux_quanta".to_owned())?,
        "total_flux_quanta",
    )?;
    if flux_mode == "zero" {
        if !close_enough(total_flux, 0.0) || any_nonzero_gauge_link {
            return Err("zero-flux declaration conflicts with nonzero gauge-link phase or flux".to_owned());
        }
    } else if flux_mode != "link_phases_declared" {
        return Err(format!("unsupported external flux mode {flux_mode}"));
    }

    let digest = Sha256::digest(fixture_bytes);
    let input_sha256 = format!("{digest:x}");
    let instance_id = required_text(&fixture, "instance_id")?;
    Ok(json!({
        "status": "pass",
        "contract": "SC-QUANTUM-PHASE-0.1",
        "instance_id": instance_id,
        "input_sha256": input_sha256,
        "node_count": nodes.len(),
        "edge_count": edges.len(),
        "spatial_dimension": dimension,
        "charging_matrix_dimension": matrix_dimension,
        "charging_matrix_representation": matrix_kind,
        "josephson_energy_sum": total_josephson_energy,
        "temperature": temperature,
        "temperature_unit": "coupling_energy_units",
        "imaginary_time_slices": slices,
        "imaginary_time_step": imaginary_time_step,
        "imaginary_time_beta": 1.0 / temperature,
        "representation": representation,
        "charge_state_cutoff": charge_cutoff,
        "bath_mode": bath_mode,
        "external_flux_mode": flux_mode,
        "checks": {
            "schema_valid": true,
            "synthetic_fixture_only": true,
            "node_and_edge_ids_unique": true,
            "edge_endpoints_resolve": true,
            "periodic_bond_geometry_matches": true,
            "charging_matrix_symmetric_positive_definite": true,
            "offset_charge_count_matches_nodes": true,
            "bath_convention_is_explicit": true,
            "gauge_and_flux_declaration_consistent": true,
            "quantum_units_are_explicit": true
        },
        "claim_ceiling": "Quantum-phase model-input audit only; no QMC state, critical point, conductivity, material Tc, or experimental result is established."
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <quantum-phase-schema.json> --fixture <instance.json> [--out <receipt.json>]"
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
    let receipt = validate_instance(&read(&schema_path), &read(&fixture_path))
        .unwrap_or_else(|error| {
            eprintln!("SC-QUANTUM-PHASE-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing the quantum-phase receipt should not fail");
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
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_quantum_phase_instance_v0_1.schema.json"),
        )
        .expect("quantum phase schema must be readable");
        let fixture: Value = serde_json::from_slice(
            &fs::read(root.join("data/research/superconductivity_quantum_phase_fixture_v0_1.json"))
                .expect("quantum phase fixture must be readable"),
        )
        .expect("quantum phase fixture must parse");
        (schema, fixture)
    }

    fn audit_mutated(fixture: Value) -> Result<Value, String> {
        let (schema, _) = inputs();
        validate_instance(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn committed_quantum_rotor_fixture_passes() {
        let (_, fixture) = inputs();
        let receipt = audit_mutated(fixture).unwrap();
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["node_count"], 4);
        assert_eq!(receipt["edge_count"], 4);
        assert_eq!(receipt["charging_matrix_dimension"], 4);
    }

    #[test]
    fn charging_matrix_must_be_symmetric() {
        let (_, mut fixture) = inputs();
        fixture["charging"]["matrix_u_ij"][0][1] = json!(0.2);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn charging_matrix_must_be_positive_definite() {
        let (_, mut fixture) = inputs();
        fixture["charging"]["matrix_u_ij"][0][0] = json!(-1.0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn matrix_dimension_must_match_node_count() {
        let (_, mut fixture) = inputs();
        fixture["charging"]["matrix_u_ij"].as_array_mut().unwrap().pop();
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn offset_charge_count_must_match_node_count() {
        let (_, mut fixture) = inputs();
        fixture["charging"]["offset_charges"].as_array_mut().unwrap().pop();
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn duplicate_edge_identity_fails_closed() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["edges"][1]["edge_id"] = json!("e0");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn unresolved_edge_endpoint_fails_closed() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["edges"][0]["target"] = json!("missing");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn periodic_image_geometry_must_match() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["edges"][3]["periodic_image_shift"] = json!([0]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn zero_bath_declaration_must_be_consistent() {
        let (_, mut fixture) = inputs();
        fixture["bath"]["coupling_strength"] = json!(0.5);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn zero_flux_must_not_hide_nonzero_gauge_links() {
        let (_, mut fixture) = inputs();
        fixture["graph"]["edges"][0]["gauge_link_phase_rad"] = json!(0.2);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn synthetic_only_auditor_rejects_non_fixture_inputs() {
        let (_, mut fixture) = inputs();
        fixture["fixture_only"] = json!(false);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn source_reproduction_requires_pinned_model_identity() {
        let (_, mut fixture) = inputs();
        fixture["model"]["kind"] = json!("source_reproduction");
        fixture["model"]["source_model_id"] = Value::Null;
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn imaginary_time_slices_and_step_must_recover_beta() {
        let (_, mut fixture) = inputs();
        fixture["simulation"]["imaginary_time_step"] = json!(0.1);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn pair_charge_normalization_must_be_explicit() {
        let (_, mut fixture) = inputs();
        fixture["model"]["pair_charge_convention"] = json!("electron_charge_equals_1");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn compact_scalar_on_site_charging_does_not_require_dense_matrix() {
        let (_, mut fixture) = inputs();
        fixture["charging"]["matrix_kind"] = json!("diagonal_scalar");
        fixture["charging"]["diagonal_value"] = json!(4.0);
        fixture["charging"].as_object_mut().unwrap().remove("matrix_u_ij");
        let receipt = audit_mutated(fixture).unwrap();
        assert_eq!(receipt["charging_matrix_dimension"], 4);
        assert_eq!(receipt["charging_matrix_representation"], "diagonal_scalar");
    }

    #[test]
    fn compact_scalar_on_site_charging_must_be_positive() {
        let (_, mut fixture) = inputs();
        fixture["charging"]["matrix_kind"] = json!("diagonal_scalar");
        fixture["charging"]["diagonal_value"] = json!(0.0);
        fixture["charging"].as_object_mut().unwrap().remove("matrix_u_ij");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn source_aligned_ponticelli_parameter_fixture_passes_as_input_only() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_quantum_phase_instance_v0_1.schema.json"),
        )
        .unwrap();
        let fixture: Value = serde_json::from_slice(
            &fs::read(
                root.join("data/research/superconductivity_quantum_phase_ponticelli_2026_U4_T03_input_fixture_v0_1.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let receipt = validate_instance(&schema, &serde_json::to_vec(&fixture).unwrap()).unwrap();
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["node_count"], 16);
        assert_eq!(receipt["edge_count"], 32);
        assert_eq!(receipt["charging_matrix_representation"], "diagonal_scalar");
        assert_eq!(receipt["imaginary_time_slices"], 1000);
        assert_eq!(receipt["temperature"], 0.3);
        assert!(fixture["claim_ceiling"].as_str().unwrap().contains("not a QMC result"));
    }

    #[test]
    fn phase_path_integral_must_not_claim_charge_basis_cutoff() {
        let (_, mut fixture) = inputs();
        fixture["simulation"]["charge_basis_cutoff"]["mode"] = json!("symmetric_integer");
        fixture["simulation"]["charge_basis_cutoff"]["max_abs_charge_state"] = json!(4);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn integer_charge_basis_requires_explicit_finite_cutoff() {
        let (_, mut fixture) = inputs();
        fixture["simulation"]["representation"] = json!("integer_charge_basis");
        fixture["simulation"]["charge_basis_cutoff"]["mode"] = json!("symmetric_integer");
        fixture["simulation"]["charge_basis_cutoff"]["max_abs_charge_state"] = json!(8);
        assert!(audit_mutated(fixture).is_ok());
        fixture["simulation"]["charge_basis_cutoff"]["max_abs_charge_state"] = Value::Null;
        assert!(audit_mutated(fixture).is_err());
    }
}
