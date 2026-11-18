use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::f64::consts::TAU;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn rotate(self, angle: f64) -> Self {
        let (sine, cosine) = angle.sin_cos();
        Self {
            re: cosine * self.re - sine * self.im,
            im: sine * self.re + cosine * self.im,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

#[derive(Clone, Debug)]
struct Node {
    id: String,
    position: Vec<f64>,
    psi: Complex,
    a: f64,
    b: f64,
}

#[derive(Clone, Debug)]
struct Edge {
    id: String,
    source: usize,
    target: usize,
    coupling: f64,
    gauge_phase: f64,
    length: f64,
    image_shift: Vec<i64>,
}

#[derive(Clone, Debug)]
struct GlModel {
    instance_id: String,
    schema_sha256: String,
    fixture_sha256: String,
    dimension: usize,
    box_lengths: Vec<f64>,
    periodic_axes: HashSet<usize>,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    temperature: f64,
    amplitude_threshold: f64,
    external_flux_quanta: f64,
    gauge_source: String,
}

#[derive(Clone, Copy, Debug)]
struct FreeEnergy {
    local: f64,
    gradient: f64,
    total: f64,
}

#[derive(Clone, Copy, Debug)]
struct VortexDiagnostics {
    positive_charge: usize,
    negative_charge: usize,
    absolute_charge_sum: usize,
    valid_plaquettes: usize,
    skipped_low_amplitude_plaquettes: usize,
}

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty text field {key}"))
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

impl GlModel {
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
                "amplitude-phase schema validation failed: {} at {}",
                error,
                error.instance_path()
            ));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("initial GL evaluator accepts synthetic fixture inputs only".to_owned());
        }
        if required_text(&fixture, "contract")? != "SC-AMPLITUDE-PHASE-0.1" {
            return Err("unsupported amplitude-phase contract".to_owned());
        }

        let instance_id = required_text(&fixture, "instance_id")?.to_owned();
        let schema_sha256 = format!("{:x}", Sha256::digest(schema_bytes));
        let fixture_sha256 = format!("{:x}", Sha256::digest(fixture_bytes));

        let model_value = fixture
            .get("model")
            .ok_or_else(|| "missing model".to_owned())?;
        if required_text(model_value, "kind")? != "static_classical_lattice_gl"
            || required_text(model_value, "free_energy_unit")?
                != "dimensionless_lattice_energy"
            || required_text(model_value, "order_parameter_unit")? != "dimensionless"
            || required_text(model_value, "gauge_convention")?
                != "psi_i_minus_exp_iAij_psi_j"
        {
            return Err("unsupported GL model or unit convention".to_owned());
        }
        let amplitude_threshold = finite_number(
            model_value
                .get("vortex_phase_amplitude_threshold")
                .ok_or_else(|| "missing vortex phase threshold".to_owned())?,
            "vortex phase amplitude threshold",
        )?;
        if amplitude_threshold < 0.0 {
            return Err("vortex phase threshold must be non-negative".to_owned());
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
            return Err("GL graph spatial_dimension must be 2 or 3".to_owned());
        }
        if required_text(graph, "embedding")? != "physical_coordinates" {
            return Err("GL evaluator requires physical-coordinate graph embedding".to_owned());
        }

        let lengths_json = graph
            .get("box_lengths")
            .and_then(Value::as_array)
            .ok_or_else(|| "box_lengths must be an array".to_owned())?;
        if lengths_json.len() != dimension {
            return Err("box_lengths length must match spatial_dimension".to_owned());
        }
        let mut box_lengths = Vec::with_capacity(dimension);
        for value in lengths_json {
            let length = finite_number(value, "box length")?;
            if length <= 0.0 {
                return Err("box lengths must be positive".to_owned());
            }
            box_lengths.push(length);
        }

        let axes_json = graph
            .get("periodic_axes")
            .and_then(Value::as_array)
            .ok_or_else(|| "periodic_axes must be an array".to_owned())?;
        let mut periodic_axes = HashSet::with_capacity(axes_json.len());
        for value in axes_json {
            let axis = value
                .as_u64()
                .ok_or_else(|| "periodic axes must be non-negative integers".to_owned())?
                as usize;
            if axis >= dimension || !periodic_axes.insert(axis) {
                return Err(format!("invalid or duplicate periodic axis {axis}"));
            }
        }

        let nodes_json = graph
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| "graph.nodes must be an array".to_owned())?;
        if nodes_json.len() < 4 {
            return Err("GL graph requires at least four nodes".to_owned());
        }
        let mut id_to_index = HashMap::with_capacity(nodes_json.len());
        let mut nodes = Vec::with_capacity(nodes_json.len());
        for node_json in nodes_json {
            let id = required_text(node_json, "node_id")?.to_owned();
            if id_to_index.insert(id.clone(), nodes.len()).is_some() {
                return Err(format!("duplicate GL node ID {id}"));
            }

            let position_json = node_json
                .get("position")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("node {id} requires position"))?;
            if position_json.len() != dimension {
                return Err(format!("node {id} position dimension mismatch"));
            }
            let mut position = Vec::with_capacity(dimension);
            for axis in 0..dimension {
                let coordinate = finite_number(
                    &position_json[axis],
                    &format!("node {id} coordinate {axis}"),
                )?;
                if coordinate < 0.0 || coordinate >= box_lengths[axis] {
                    return Err(format!("node {id} lies outside the declared box"));
                }
                position.push(coordinate);
            }

            let psi = Complex {
                re: finite_number(
                    node_json
                        .get("psi_real")
                        .ok_or_else(|| format!("node {id} missing psi_real"))?,
                    "psi_real",
                )?,
                im: finite_number(
                    node_json
                        .get("psi_imag")
                        .ok_or_else(|| format!("node {id} missing psi_imag"))?,
                    "psi_imag",
                )?,
            };
            let a = finite_number(
                node_json
                    .get("a_coefficient")
                    .ok_or_else(|| format!("node {id} missing a_coefficient"))?,
                "a_coefficient",
            )?;
            let b = finite_number(
                node_json
                    .get("b_coefficient")
                    .ok_or_else(|| format!("node {id} missing b_coefficient"))?,
                "b_coefficient",
            )?;
            if b <= 0.0 {
                return Err(format!("node {id} quartic coefficient b must be positive"));
            }
            nodes.push(Node {
                id,
                position,
                psi,
                a,
                b,
            });
        }

        let edges_json = graph
            .get("edges")
            .and_then(Value::as_array)
            .ok_or_else(|| "graph.edges must be an array".to_owned())?;
        if edges_json.is_empty() {
            return Err("GL graph requires at least one edge".to_owned());
        }
        let mut edge_ids = HashSet::with_capacity(edges_json.len());
        let mut endpoint_pairs = HashSet::with_capacity(edges_json.len());
        let mut edges = Vec::with_capacity(edges_json.len());
        let mut any_nonzero_gauge_link = false;

        for edge_json in edges_json {
            let id = required_text(edge_json, "edge_id")?.to_owned();
            if !edge_ids.insert(id.clone()) {
                return Err(format!("duplicate GL edge ID {id}"));
            }
            let source_id = required_text(edge_json, "source")?;
            let target_id = required_text(edge_json, "target")?;
            let source = *id_to_index
                .get(source_id)
                .ok_or_else(|| format!("edge {id} unresolved source {source_id}"))?;
            let target = *id_to_index
                .get(target_id)
                .ok_or_else(|| format!("edge {id} unresolved target {target_id}"))?;
            if source == target {
                return Err(format!("edge {id} self-loop is not admitted"));
            }
            if !endpoint_pairs.insert((source.min(target), source.max(target))) {
                return Err(format!("duplicate undirected GL bond at edge {id}"));
            }

            let coupling = finite_number(
                edge_json
                    .get("gradient_coupling")
                    .ok_or_else(|| format!("edge {id} missing gradient_coupling"))?,
                "gradient_coupling",
            )?;
            if coupling <= 0.0 {
                return Err(format!("edge {id} gradient coupling must be positive"));
            }
            let gauge_phase = finite_number(
                edge_json
                    .get("gauge_link_phase_rad")
                    .ok_or_else(|| format!("edge {id} missing gauge_link_phase_rad"))?,
                "gauge_link_phase_rad",
            )?;
            any_nonzero_gauge_link |= gauge_phase.abs() > 1.0e-12;

            let image_json = edge_json
                .get("periodic_image_shift")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("edge {id} missing periodic_image_shift"))?;
            if image_json.len() != dimension {
                return Err(format!("edge {id} image-shift dimension mismatch"));
            }
            let mut image_shift = Vec::with_capacity(dimension);
            let mut squared_distance = 0.0;
            for axis in 0..dimension {
                let shift = image_json[axis]
                    .as_i64()
                    .ok_or_else(|| format!("edge {id} image shift must be integer"))?;
                if shift != 0 && !periodic_axes.contains(&axis) {
                    return Err(format!("edge {id} image shift on non-periodic axis {axis}"));
                }
                image_shift.push(shift);
                let displacement = nodes[target].position[axis]
                    + shift as f64 * box_lengths[axis]
                    - nodes[source].position[axis];
                squared_distance += displacement * displacement;
            }
            let derived_length = squared_distance.sqrt();
            let declared_length = finite_number(
                edge_json
                    .get("physical_edge_length")
                    .ok_or_else(|| format!("edge {id} missing physical_edge_length"))?,
                "physical_edge_length",
            )?;
            if declared_length <= 0.0 || !close_enough(derived_length, declared_length) {
                return Err(format!(
                    "edge {id} physical length disagrees with periodic-image geometry"
                ));
            }
            edges.push(Edge {
                id,
                source,
                target,
                coupling,
                gauge_phase,
                length: declared_length,
                image_shift,
            });
        }

        let electromagnetism = fixture
            .get("electromagnetism")
            .ok_or_else(|| "missing electromagnetism declaration".to_owned())?;
        if required_text(electromagnetism, "mode")? != "fixed_link_field" {
            return Err("initial GL evaluator supports fixed link fields only".to_owned());
        }
        let gauge_source = required_text(electromagnetism, "link_field_source")?.to_owned();
        let external_flux_quanta = finite_number(
            electromagnetism
                .get("external_flux_quanta")
                .ok_or_else(|| "missing external_flux_quanta".to_owned())?,
            "external_flux_quanta",
        )?;
        match gauge_source.as_str() {
            "zero" => {
                if any_nonzero_gauge_link || !close_enough(external_flux_quanta, 0.0) {
                    return Err(
                        "zero link-field declaration conflicts with nonzero link phase/flux"
                            .to_owned(),
                    );
                }
            }
            "explicit_link_phases" => {}
            other => return Err(format!("unsupported link field source {other}")),
        }

        let thermal = fixture
            .get("thermal")
            .ok_or_else(|| "missing thermal declaration".to_owned())?;
        if required_text(thermal, "temperature_unit")?
            != "dimensionless_GL_temperature"
        {
            return Err("unsupported GL temperature unit".to_owned());
        }
        let temperature = finite_number(
            thermal
                .get("temperature")
                .ok_or_else(|| "missing temperature".to_owned())?,
            "temperature",
        )?;
        if temperature < 0.0 {
            return Err("temperature must be non-negative".to_owned());
        }

        Ok(Self {
            instance_id,
            schema_sha256,
            fixture_sha256,
            dimension,
            box_lengths,
            periodic_axes,
            nodes,
            edges,
            temperature,
            amplitude_threshold,
            external_flux_quanta,
            gauge_source,
        })
    }

    fn free_energy_for(&self, fields: &[Complex], gauge_phases: &[f64]) -> Result<FreeEnergy, String> {
        if fields.len() != self.nodes.len() {
            return Err("complex-field count must equal node count".to_owned());
        }
        if gauge_phases.len() != self.edges.len() {
            return Err("gauge-link phase count must equal edge count".to_owned());
        }
        let mut local = 0.0;
        for (node, psi) in self.nodes.iter().zip(fields) {
            if !psi.re.is_finite() || !psi.im.is_finite() {
                return Err(format!("node {} has non-finite complex field", node.id));
            }
            let rho_squared = psi.norm_sqr();
            local += node.a * rho_squared + 0.5 * node.b * rho_squared * rho_squared;
        }

        let mut gradient = 0.0;
        for (edge, gauge_phase) in self.edges.iter().zip(gauge_phases) {
            if !gauge_phase.is_finite() {
                return Err(format!("edge {} has non-finite gauge phase", edge.id));
            }
            let transported_target = fields[edge.target].rotate(*gauge_phase);
            let difference = fields[edge.source].sub(transported_target);
            gradient += edge.coupling * difference.norm_sqr();
        }
        let total = local + gradient;
        if !local.is_finite() || !gradient.is_finite() || !total.is_finite() {
            return Err("GL free energy overflowed or became non-finite".to_owned());
        }
        Ok(FreeEnergy {
            local,
            gradient,
            total,
        })
    }

    fn initial_fields(&self) -> Vec<Complex> {
        self.nodes.iter().map(|node| node.psi).collect()
    }

    fn initial_gauge_phases(&self) -> Vec<f64> {
        self.edges.iter().map(|edge| edge.gauge_phase).collect()
    }

    fn gauge_transform(
        &self,
        fields: &[Complex],
        gauge_phases: &[f64],
        node_gauge_angles: &[f64],
    ) -> Result<(Vec<Complex>, Vec<f64>), String> {
        if node_gauge_angles.len() != self.nodes.len() {
            return Err("gauge transformation requires one angle per node".to_owned());
        }
        if gauge_phases.len() != self.edges.len() {
            return Err("gauge transformation requires one phase per edge".to_owned());
        }
        let transformed_fields = fields
            .iter()
            .zip(node_gauge_angles)
            .map(|(field, angle)| field.rotate(*angle))
            .collect();
        let transformed_links = self
            .edges
            .iter()
            .zip(gauge_phases)
            .map(|(edge, phase)| {
                phase + node_gauge_angles[edge.source] - node_gauge_angles[edge.target]
            })
            .collect();
        Ok((transformed_fields, transformed_links))
    }

    fn vortex_diagnostics(
        &self,
        fields: &[Complex],
        gauge_phases: &[f64],
    ) -> Result<Option<VortexDiagnostics>, String> {
        if self.dimension != 2
            || gauge_phases.len() != self.edges.len()
            || fields.len() != self.nodes.len()
        {
            return Ok(None);
        }

        let mut grid_lengths = Vec::with_capacity(2);
        for length in &self.box_lengths {
            let rounded = length.round();
            if (*length - rounded).abs() > 1.0e-10 || rounded < 2.0 {
                return Ok(None);
            }
            grid_lengths.push(rounded as usize);
        }
        let nx = grid_lengths[0];
        let ny = grid_lengths[1];
        let mut grid = vec![None; nx * ny];
        for (index, node) in self.nodes.iter().enumerate() {
            let x = node.position[0].round();
            let y = node.position[1].round();
            if (node.position[0] - x).abs() > 1.0e-10
                || (node.position[1] - y).abs() > 1.0e-10
                || x < 0.0
                || y < 0.0
                || x >= nx as f64
                || y >= ny as f64
            {
                return Ok(None);
            }
            let slot = x as usize * ny + y as usize;
            if grid[slot].replace(index).is_some() {
                return Err("duplicate integer grid coordinate in GL vortex diagnostic".to_owned());
            }
        }
        if grid.iter().any(Option::is_none) {
            return Ok(None);
        }

        let mut edge_lookup = HashMap::with_capacity(self.edges.len());
        for (edge_index, edge) in self.edges.iter().enumerate() {
            edge_lookup.insert(
                (edge.source.min(edge.target), edge.source.max(edge.target)),
                edge_index,
            );
        }

        let phase_of = |field: Complex| field.im.atan2(field.re);
        let mut diagnostics = VortexDiagnostics {
            positive_charge: 0,
            negative_charge: 0,
            absolute_charge_sum: 0,
            valid_plaquettes: 0,
            skipped_low_amplitude_plaquettes: 0,
        };
        let x_plaquettes = if self.periodic_axes.contains(&0) {
            nx
        } else {
            nx.saturating_sub(1)
        };
        let y_plaquettes = if self.periodic_axes.contains(&1) {
            ny
        } else {
            ny.saturating_sub(1)
        };

        for x in 0..x_plaquettes {
            for y in 0..y_plaquettes {
                let x_next = if x + 1 == nx { 0 } else { x + 1 };
                let y_next = if y + 1 == ny { 0 } else { y + 1 };
                let corners = [
                    grid[x * ny + y].unwrap(),
                    grid[x_next * ny + y].unwrap(),
                    grid[x_next * ny + y_next].unwrap(),
                    grid[x * ny + y_next].unwrap(),
                ];
                let low_amplitude = corners.iter().any(|index| {
                    fields[*index].norm_sqr().sqrt()
                        <= self.amplitude_threshold.max(f64::EPSILON)
                });
                if low_amplitude {
                    diagnostics.skipped_low_amplitude_plaquettes += 1;
                    continue;
                }

                let mut covariant_phase_sum = 0.0;
                let mut oriented_gauge_flux = 0.0;
                for side in 0..4 {
                    let from = corners[side];
                    let to = corners[(side + 1) % 4];
                    let key = (from.min(to), from.max(to));
                    let Some(edge_index) = edge_lookup.get(&key).copied() else {
                        return Err(format!(
                            "missing GL edge for plaquette link {} -> {}",
                            self.nodes[from].id, self.nodes[to].id
                        ));
                    };
                    let edge = &self.edges[edge_index];
                    let signed_gauge = if edge.source == from && edge.target == to {
                        edge.gauge_phase
                    } else if edge.source == to && edge.target == from {
                        -edge.gauge_phase
                    } else {
                        return Err("GL edge lookup endpoints are inconsistent".to_owned());
                    };
                    let phase_difference =
                        phase_of(fields[to]) - phase_of(fields[from]) + signed_gauge;
                    covariant_phase_sum += wrap_to_pi(phase_difference);
                    oriented_gauge_flux += signed_gauge;
                }

                // The covariant phase circulation contains both integer winding
                // and plaquette link flux. Subtract the link flux to recover the
                // integer winding. Near-zero-amplitude plaquettes were skipped.
                let charge_value =
                    (covariant_phase_sum - oriented_gauge_flux) / std::f64::consts::TAU;
                let integer_charge = charge_value.round();
                if !charge_value.is_finite() || (charge_value - integer_charge).abs() > 1.0e-6 {
                    return Err(format!(
                        "non-integral gauge-covariant plaquette winding {charge_value}"
                    ));
                }
                let charge = integer_charge as i64;
                if charge > 0 {
                    diagnostics.positive_charge += charge as usize;
                } else if charge < 0 {
                    diagnostics.negative_charge += charge.unsigned_abs() as usize;
                }
                diagnostics.absolute_charge_sum += charge.unsigned_abs() as usize;
                diagnostics.valid_plaquettes += 1;
            }
        }
        Ok(Some(diagnostics))
    }

    fn receipt(
        &self,
        energy: FreeEnergy,
        gauge_residual: f64,
        vortices: Option<VortexDiagnostics>,
    ) -> Value {
        let mean_amplitude = self
            .nodes
            .iter()
            .map(|node| node.psi.norm_sqr().sqrt())
            .sum::<f64>()
            / self.nodes.len() as f64;
        let mean_squared_amplitude = self
            .nodes
            .iter()
            .map(|node| node.psi.norm_sqr())
            .sum::<f64>()
            / self.nodes.len() as f64;
        json!({
            "status": "pass",
            "contract": "SC-AMPLITUDE-PHASE-AUDIT-0.1",
            "instance_id": self.instance_id,
            "schema_sha256": self.schema_sha256,
            "fixture_sha256": self.fixture_sha256,
            "spatial_dimension": self.dimension,
            "node_count": self.nodes.len(),
            "edge_count": self.edges.len(),
            "box_lengths": self.box_lengths,
            "periodic_axis_count": self.periodic_axes.len(),
            "edge_length_sum": self.edges.iter().map(|edge| edge.length).sum::<f64>(),
            "periodic_image_edge_count": self
                .edges
                .iter()
                .filter(|edge| edge.image_shift.iter().any(|shift| *shift != 0))
                .count(),
            "temperature": self.temperature,
            "temperature_unit": "dimensionless_GL_temperature",
            "external_flux_quanta": self.external_flux_quanta,
            "link_field_source": self.gauge_source,
            "mean_amplitude": mean_amplitude,
            "mean_squared_amplitude": mean_squared_amplitude,
            "free_energy": {
                "local_potential": energy.local,
                "gradient": energy.gradient,
                "total": energy.total,
                "unit": "dimensionless_lattice_energy"
            },
            "gauge_invariance_test": {
                "absolute_free_energy_residual": gauge_residual,
                "tolerance": 1.0e-9,
                "passed": gauge_residual <= 1.0e-9
            },
            "vortex_phase_amplitude_threshold": self.amplitude_threshold,
            "vortex_diagnostics": vortices.map(|values| json!({
                "positive_vortex_charge": values.positive_charge,
                "negative_vortex_charge": values.negative_charge,
                "absolute_vortex_charge_sum": values.absolute_charge_sum,
                "valid_plaquette_count": values.valid_plaquettes,
                "skipped_low_amplitude_plaquette_count": values.skipped_low_amplitude_plaquettes,
                "absolute_vortex_charge_density_per_valid_plaquette": if values.valid_plaquettes > 0 {
                    values.absolute_charge_sum as f64 / values.valid_plaquettes as f64
                } else {
                    0.0
                },
                "definition": "integer gauge-invariant winding from sum(wrap(delta_phase + oriented_link_A)) minus oriented plaquette link flux, divided by 2pi"
            })),
            "claim_ceiling": "Static synthetic lattice GL input and free-energy evaluation only; no Monte Carlo, equilibrium, phase transition, Tc, material property, or experimental result is established."
        })
    }
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --fixture <gl-instance.json> [--out <receipt.json>]"
    );
}

fn audit_and_evaluate(schema: &[u8], fixture: &[u8]) -> Result<Value, String> {
    let model = GlModel::from_bytes(schema, fixture)?;
    let fields = model.initial_fields();
    let links = model.initial_gauge_phases();
    let energy = model.free_energy_for(&fields, &links)?;

    let gauge_angles: Vec<f64> = (0..model.nodes.len())
        .map(|index| {
            let x = index as f64 + 1.0;
            0.173 * x + 0.071 * x * x
        })
        .collect();
    let (transformed_fields, transformed_links) =
        model.gauge_transform(&fields, &links, &gauge_angles)?;
    let transformed_energy =
        model.free_energy_for(&transformed_fields, &transformed_links)?;
    let gauge_residual = (energy.total - transformed_energy.total).abs();
    if gauge_residual > 1.0e-9 {
        return Err(format!(
            "local gauge transformation changed the GL free energy by {gauge_residual}"
        ));
    }
    let vortices = model.vortex_diagnostics(&fields, &links)?;
    Ok(model.receipt(energy, gauge_residual, vortices))
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
    let receipt = audit_and_evaluate(&read(&schema_path), &read(&fixture_path))
        .unwrap_or_else(|error| {
            eprintln!("SC-AMPLITUDE-PHASE-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing the GL audit receipt should not fail");
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
            root.join("data/schemas/superconductivity_amplitude_phase_instance_v0_1.schema.json"),
        )
        .expect("GL schema must be readable");
        let fixture = fs::read(
            root.join("data/research/superconductivity_amplitude_phase_fixture_v0_1.json"),
        )
        .expect("GL fixture must be readable");
        (schema, fixture)
    }

    fn mutate_fixture(mutation: impl FnOnce(&mut Value)) -> Result<Value, String> {
        let (schema, bytes) = inputs();
        let mut fixture: Value = serde_json::from_slice(&bytes).unwrap();
        mutation(&mut fixture);
        GlModel::from_bytes(&schema, &serde_json::to_vec(&fixture).unwrap())
            .map(|model| {
                let energy = model
                    .free_energy_for(&model.initial_fields(), &model.initial_gauge_phases())
                    .unwrap();
                model.receipt(energy, 0.0, model.vortex_diagnostics(
                    &model.initial_fields(),
                    &model.initial_gauge_phases(),
                ).unwrap())
            })
    }

    #[test]
    fn uniform_fixture_has_analytic_local_and_total_energy() {
        let (schema, fixture) = inputs();
        let model = GlModel::from_bytes(&schema, &fixture).unwrap();
        let energy = model
            .free_energy_for(&model.initial_fields(), &model.initial_gauge_phases())
            .unwrap();
        assert!((energy.local - (-0.5 * 16.0)).abs() < 1.0e-12);
        assert!(energy.gradient.abs() < 1.0e-12);
        assert!((energy.total - (-8.0)).abs() < 1.0e-12);
    }

    #[test]
    fn local_gauge_transformation_preserves_free_energy() {
        let (schema, fixture) = inputs();
        let model = GlModel::from_bytes(&schema, &fixture).unwrap();
        let fields = model.initial_fields();
        let links = model.initial_gauge_phases();
        let initial = model.free_energy_for(&fields, &links).unwrap();
        let angles: Vec<f64> = (0..model.nodes.len())
            .map(|index| 0.19 * (index as f64 + 1.0).powi(2) + 0.03)
            .collect();
        let (transformed_fields, transformed_links) =
            model.gauge_transform(&fields, &links, &angles).unwrap();
        let transformed = model
            .free_energy_for(&transformed_fields, &transformed_links)
            .unwrap();
        assert!((initial.total - transformed.total).abs() < 1.0e-10);
        assert!((initial.local - transformed.local).abs() < 1.0e-10);
        assert!((initial.gradient - transformed.gradient).abs() < 1.0e-10);
    }

    #[test]
    fn gauge_invariance_holds_for_nonuniform_complex_fields_and_nonzero_links() {
        let (schema, fixture) = inputs();
        let model = GlModel::from_bytes(&schema, &fixture).unwrap();
        let fields: Vec<Complex> = (0..model.nodes.len())
            .map(|index| {
                let x = index as f64;
                Complex {
                    re: 0.7 + 0.11 * (0.4 * x).cos(),
                    im: 0.13 * (0.7 * x).sin(),
                }
            })
            .collect();
        let links: Vec<f64> = (0..model.edges.len())
            .map(|index| 0.21 * (index as f64 * 0.37).sin())
            .collect();
        let angles: Vec<f64> = (0..model.nodes.len())
            .map(|index| 0.31 * (index as f64 + 1.0).cos())
            .collect();
        let initial = model.free_energy_for(&fields, &links).unwrap();
        let (transformed_fields, transformed_links) =
            model.gauge_transform(&fields, &links, &angles).unwrap();
        let transformed = model
            .free_energy_for(&transformed_fields, &transformed_links)
            .unwrap();
        assert!((initial.local - transformed.local).abs() < 1.0e-10);
        assert!((initial.gradient - transformed.gradient).abs() < 1.0e-10);
        assert!((initial.total - transformed.total).abs() < 1.0e-10);
    }

    #[test]
    fn zero_field_uniform_state_has_zero_gradient_energy() {
        let (schema, fixture) = inputs();
        let model = GlModel::from_bytes(&schema, &fixture).unwrap();
        let energy = model
            .free_energy_for(&model.initial_fields(), &model.initial_gauge_phases())
            .unwrap();
        assert!(energy.gradient.abs() < 1.0e-12);
    }

    #[test]
    fn negative_quartic_coefficient_is_rejected() {
        assert!(mutate_fixture(|fixture| {
            fixture["graph"]["nodes"][0]["b_coefficient"] = json!(-1.0);
        })
        .is_err());
    }

    #[test]
    fn unresolved_edge_endpoint_is_rejected() {
        assert!(mutate_fixture(|fixture| {
            fixture["graph"]["edges"][0]["target"] = json!("missing-node");
        })
        .is_err());
    }

    #[test]
    fn periodic_bond_geometry_is_checked() {
        assert!(mutate_fixture(|fixture| {
            fixture["graph"]["edges"][6]["periodic_image_shift"] = json!([0, 0]);
        })
        .is_err());
    }

    #[test]
    fn duplicate_node_identity_is_rejected() {
        assert!(mutate_fixture(|fixture| {
            fixture["graph"]["nodes"][1]["node_id"] = json!("n0_0");
        })
        .is_err());
    }

    #[test]
    fn zero_link_field_declaration_rejects_hidden_flux() {
        assert!(mutate_fixture(|fixture| {
            fixture["graph"]["edges"][0]["gauge_link_phase_rad"] = json!(0.3);
        })
        .is_err());
    }

    #[test]
    fn nonuniform_complex_field_has_exact_gradient_energy_for_one_suppressed_site() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["graph"]["nodes"][0]["psi_real"] = json!(0.0);
        value["graph"]["nodes"][0]["psi_imag"] = json!(0.0);
        let model =
            GlModel::from_bytes(&schema, &serde_json::to_vec(&value).unwrap()).unwrap();
        let energy = model
            .free_energy_for(&model.initial_fields(), &model.initial_gauge_phases())
            .unwrap();
        // One zero-amplitude site has four unit-coupling neighbours on the 4x4 torus.
        assert!((energy.local - (-7.5)).abs() < 1.0e-12);
        assert!((energy.gradient - 4.0).abs() < 1.0e-12);
        assert!((energy.total - (-3.5)).abs() < 1.0e-12);
    }

    #[test]
    fn evaluator_receipt_is_bound_to_exact_inputs() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_evaluate(&schema, &fixture).unwrap();
        assert_eq!(
            receipt["fixture_sha256"],
            format!("{:x}", Sha256::digest(&fixture))
        );
        assert_eq!(
            receipt["schema_sha256"],
            format!("{:x}", Sha256::digest(&schema))
        );
        assert_eq!(receipt["gauge_invariance_test"]["passed"], true);
    }

    #[test]
    fn uniform_complex_field_has_zero_gauge_invariant_vorticity() {
        let (schema, fixture) = inputs();
        let model = GlModel::from_bytes(&schema, &fixture).unwrap();
        let vortices = model
            .vortex_diagnostics(&model.initial_fields(), &model.initial_gauge_phases())
            .unwrap()
            .unwrap();
        assert_eq!(vortices.positive_charge, 0);
        assert_eq!(vortices.negative_charge, 0);
        assert_eq!(vortices.absolute_charge_sum, 0);
        assert_eq!(vortices.valid_plaquettes, 16);
    }

    #[test]
    fn vortex_winding_is_invariant_under_a_local_gauge_transform() {
        let (schema, fixture) = inputs();
        let model = GlModel::from_bytes(&schema, &fixture).unwrap();
        let fields: Vec<Complex> = model
            .nodes
            .iter()
            .map(|node| {
                let x = node.position[0] - 1.5;
                let y = node.position[1] - 1.5;
                let angle = y.atan2(x);
                Complex { re: angle.cos(), im: angle.sin() }
            })
            .collect();
        let links = model.initial_gauge_phases();
        let angles: Vec<f64> = (0..model.nodes.len())
            .map(|index| 0.23 * (index as f64 + 1.0).sin())
            .collect();
        let before = model.vortex_diagnostics(&fields, &links).unwrap().unwrap();
        let (transformed_fields, transformed_links) =
            model.gauge_transform(&fields, &links, &angles).unwrap();
        let after = model
            .vortex_diagnostics(&transformed_fields, &transformed_links)
            .unwrap()
            .unwrap();
        assert_eq!(before.positive_charge, after.positive_charge);
        assert_eq!(before.negative_charge, after.negative_charge);
        assert_eq!(before.absolute_charge_sum, after.absolute_charge_sum);
    }

    #[test]
    fn vortex_diagnostic_skips_plaquettes_with_suppressed_amplitude() {
        let (schema, fixture) = inputs();
        let mut value: Value = serde_json::from_slice(&fixture).unwrap();
        value["graph"]["nodes"][0]["psi_real"] = json!(0.0);
        value["graph"]["nodes"][0]["psi_imag"] = json!(0.0);
        let model =
            GlModel::from_bytes(&schema, &serde_json::to_vec(&value).unwrap()).unwrap();
        let vortices = model
            .vortex_diagnostics(&model.initial_fields(), &model.initial_gauge_phases())
            .unwrap()
            .unwrap();
        assert_eq!(vortices.skipped_low_amplitude_plaquettes, 4);
        assert_eq!(vortices.valid_plaquettes, 12);
    }

    #[test]
    fn wrong_gauge_transform_sign_fails_the_negative_control() {
        let (schema, fixture) = inputs();
        let model = GlModel::from_bytes(&schema, &fixture).unwrap();
        let fields = model.initial_fields();
        let links = model.initial_gauge_phases();
        let angles: Vec<f64> = (0..model.nodes.len())
            .map(|index| 0.17 * (index as f64 + 1.0).powi(2))
            .collect();
        let transformed_fields: Vec<Complex> = fields
            .iter()
            .zip(&angles)
            .map(|(field, angle)| field.rotate(*angle))
            .collect();
        let wrong_links: Vec<f64> = model
            .edges
            .iter()
            .zip(&links)
            .map(|(edge, phase)| phase - angles[edge.source] + angles[edge.target])
            .collect();
        let before = model.free_energy_for(&fields, &links).unwrap();
        let after = model
            .free_energy_for(&transformed_fields, &wrong_links)
            .unwrap();
        assert!((before.total - after.total).abs() > 1.0e-3);
    }
}
