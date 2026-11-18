use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn rotate(self, angle: f64) -> Self {
        let (sine, cosine) = angle.sin_cos();
        Self {
            re: cosine * self.re - sine * self.im,
            im: sine * self.re + cosine * self.im,
        }
    }

    fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn conj_product_real(self, other: Self) -> f64 {
        self.re * other.re + self.im * other.im
    }

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

#[derive(Clone, Debug)]
struct Component {
    id: String,
    role: String,
    band_orbital_id: String,
    symmetry_label: String,
    momentum: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Site {
    id: String,
    position: Vec<f64>,
    fields: Vec<Complex>,
}

#[derive(Clone, Debug)]
struct Edge {
    id: String,
    source: usize,
    target: usize,
    axis: usize,
    image_shift: Vec<i64>,
    gauge_phase: f64,
}

#[derive(Clone, Debug)]
struct DensityTerm {
    first: usize,
    second: usize,
    coefficient: f64,
}

#[derive(Clone, Debug)]
struct PhaseLockTerm {
    first: usize,
    second: usize,
    coefficient: f64,
}

#[derive(Clone, Debug)]
struct SecondaryOrder {
    id: String,
    kind: String,
    site_values: Vec<f64>,
    component_coefficients: Vec<(usize, f64)>,
}

#[derive(Clone, Debug)]
struct EnergyBreakdown {
    quadratic: f64,
    quartic_self: f64,
    quartic_density: f64,
    phase_locking: f64,
    secondary_order: f64,
    gradient: f64,
    total: f64,
}

#[derive(Clone, Debug)]
struct IntertwinedModel {
    instance_id: String,
    schema_sha256: String,
    fixture_sha256: String,
    dimension: usize,
    shape: Vec<usize>,
    boundary_conditions: Vec<String>,
    components: Vec<Component>,
    sites: Vec<Site>,
    edges: Vec<Edge>,
    quadratic: Vec<Vec<f64>>,
    gradient_by_axis: Vec<Vec<Vec<f64>>>,
    quartic_self: Vec<f64>,
    density_terms: Vec<DensityTerm>,
    phase_locking_terms: Vec<PhaseLockTerm>,
    secondary_orders: Vec<SecondaryOrder>,
    stiffness_tolerance: f64,
    gradient_min_eigenvalues: Vec<f64>,
    quartic_min_eigenvalue: f64,
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

fn close_enough(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance * left.abs().max(right.abs()).max(1.0)
}

fn parse_symmetric_matrix(
    value: &Value,
    size: usize,
    context: &str,
) -> Result<Vec<Vec<f64>>, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a matrix"))?;
    if rows.len() != size {
        return Err(format!("{context} must have {size} rows"));
    }
    let mut matrix = Vec::with_capacity(size);
    for (row_index, row_value) in rows.iter().enumerate() {
        let row = row_value
            .as_array()
            .ok_or_else(|| format!("{context} row {row_index} must be an array"))?;
        if row.len() != size {
            return Err(format!("{context} row {row_index} must have {size} entries"));
        }
        matrix.push(
            row.iter()
                .enumerate()
                .map(|(column, value)| {
                    finite_number(value, &format!("{context}[{row_index}][{column}]"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    for i in 0..size {
        for j in 0..size {
            if !close_enough(matrix[i][j], matrix[j][i], 1.0e-10) {
                return Err(format!("{context} must be symmetric"));
            }
        }
    }
    Ok(matrix)
}

fn minimum_symmetric_eigenvalue(matrix: &[Vec<f64>]) -> Result<f64, String> {
    let size = matrix.len();
    if size == 0 || matrix.iter().any(|row| row.len() != size) {
        return Err("eigenvalue calculation requires a non-empty square matrix".to_owned());
    }
    let mut work = matrix.to_vec();
    let iterations = 100 * size * size;
    for _ in 0..iterations {
        let mut p = 0;
        let mut q = 0;
        let mut largest = 0.0_f64;
        for i in 0..size {
            for j in (i + 1)..size {
                if work[i][j].abs() > largest {
                    largest = work[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest < 1.0e-13 {
            break;
        }

        let app = work[p][p];
        let aqq = work[q][q];
        let apq = work[p][q];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let (sine, cosine) = angle.sin_cos();

        for k in 0..size {
            if k == p || k == q {
                continue;
            }
            let akp = work[k][p];
            let akq = work[k][q];
            let new_kp = cosine * akp - sine * akq;
            let new_kq = sine * akp + cosine * akq;
            work[k][p] = new_kp;
            work[p][k] = new_kp;
            work[k][q] = new_kq;
            work[q][k] = new_kq;
        }

        work[p][p] = cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq;
        work[q][q] = sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq;
        work[p][q] = 0.0;
        work[q][p] = 0.0;
    }
    let minimum = (0..size)
        .map(|index| work[index][index])
        .fold(f64::INFINITY, f64::min);
    if minimum.is_finite() {
        Ok(minimum)
    } else {
        Err("symmetric eigenvalue calculation became non-finite".to_owned())
    }
}

fn parse_coordinate(value: &Value, dimension: usize, context: &str) -> Result<Vec<f64>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    if values.len() != dimension {
        return Err(format!("{context} must have {dimension} coordinates"));
    }
    values
        .iter()
        .enumerate()
        .map(|(axis, value)| finite_number(value, &format!("{context}[{axis}]")))
        .collect()
}

fn parse_component_index(
    id: &str,
    component_index: &HashMap<String, usize>,
    context: &str,
) -> Result<usize, String> {
    component_index
        .get(id)
        .copied()
        .ok_or_else(|| format!("{context} references unknown component {id}"))
}

impl IntertwinedModel {
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
                "intertwined-order schema validation failed: {} at {}",
                error,
                error.instance_path()
            ));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("initial multicomponent evaluator accepts synthetic fixtures only".to_owned());
        }
        if required_text(&fixture, "contract")? != "SC-INTERTWINED-ORDER-0.1" {
            return Err("unsupported intertwined-order contract".to_owned());
        }
        let instance_id = required_text(&fixture, "instance_id")?.to_owned();

        let lattice = fixture
            .get("lattice")
            .ok_or_else(|| "missing lattice declaration".to_owned())?;
        let dimension = lattice
            .get("dimension")
            .and_then(Value::as_u64)
            .ok_or_else(|| "lattice.dimension must be an integer".to_owned())?
            as usize;
        if !(1..=3).contains(&dimension) {
            return Err("initial evaluator supports one to three dimensions".to_owned());
        }
        let shape_values = lattice
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| "lattice.shape must be an array".to_owned())?;
        if shape_values.len() != dimension {
            return Err("lattice.shape length must match dimension".to_owned());
        }
        let shape: Vec<usize> = shape_values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .filter(|length| *length >= 2)
                    .map(|length| length as usize)
                    .ok_or_else(|| "lattice shape dimensions must be integers >= 2".to_owned())
            })
            .collect::<Result<_, _>>()?;
        let expected_site_count = shape.iter().try_fold(1_usize, |product, length| {
            product
                .checked_mul(*length)
                .ok_or_else(|| "lattice site count overflowed usize".to_owned())
        })?;
        let boundary_values = lattice
            .get("boundary_conditions")
            .and_then(Value::as_array)
            .ok_or_else(|| "boundary_conditions must be an array".to_owned())?;
        if boundary_values.len() != dimension {
            return Err("boundary_conditions length must match dimension".to_owned());
        }
        let boundary_conditions: Vec<String> = boundary_values
            .iter()
            .map(|value| {
                let text = value
                    .as_str()
                    .ok_or_else(|| "boundary condition must be a string".to_owned())?;
                if text != "periodic" && text != "open" {
                    return Err(format!("unsupported boundary condition {text}"));
                }
                Ok(text.to_owned())
            })
            .collect::<Result<_, _>>()?;

        let component_values = fixture
            .get("components")
            .and_then(Value::as_array)
            .ok_or_else(|| "components must be an array".to_owned())?;
        if component_values.len() < 2 {
            return Err("at least two order-parameter components are required".to_owned());
        }
        let mut component_ids = HashMap::with_capacity(component_values.len());
        let mut components = Vec::with_capacity(component_values.len());
        for component_value in component_values {
            let id = required_text(component_value, "component_id")?.to_owned();
            if component_ids.insert(id.clone(), components.len()).is_some() {
                return Err(format!("duplicate component ID {id}"));
            }
            let role = required_text(component_value, "component_role")?.to_owned();
            let band_orbital_id = required_text(component_value, "band_orbital_id")?.to_owned();
            let symmetry_label = required_text(component_value, "symmetry_label")?.to_owned();
            if required_text(component_value, "gauge_charge")? != "Cooper_pair_charge_2e" {
                return Err(format!(
                    "component {id} must carry the common Cooper-pair gauge charge"
                ));
            }
            if component_value.get("complex_field").and_then(Value::as_bool) != Some(true) {
                return Err(format!("component {id} must be a complex field"));
            }
            let momentum = parse_coordinate(
                component_value
                    .get("momentum_vector")
                    .ok_or_else(|| format!("component {id} missing momentum_vector"))?,
                dimension,
                &format!("component {id} momentum"),
            )?;
            if momentum.iter().any(|value| value.abs() > 1.0e-12) {
                return Err(format!(
                    "component {id} has nonzero momentum; finite-Q PDW evaluation is not implemented in this version"
                ));
            }
            components.push(Component {
                id,
                role,
                band_orbital_id,
                symmetry_label,
                momentum,
            });
        }
        let component_count = components.len();

        let field_configuration = fixture
            .get("field_configuration")
            .ok_or_else(|| "missing field_configuration".to_owned())?;
        let site_values = field_configuration
            .get("sites")
            .and_then(Value::as_array)
            .ok_or_else(|| "field_configuration.sites must be an array".to_owned())?;
        if site_values.len() != expected_site_count {
            return Err(format!(
                "lattice requires {expected_site_count} sites but {} were supplied",
                site_values.len()
            ));
        }
        let mut site_index = HashMap::with_capacity(site_values.len());
        let mut sites = Vec::with_capacity(site_values.len());
        let mut coordinate_set = HashSet::with_capacity(site_values.len());
        for site_value in site_values {
            let id = required_text(site_value, "site_id")?.to_owned();
            if site_index.insert(id.clone(), sites.len()).is_some() {
                return Err(format!("duplicate site ID {id}"));
            }
            let position = parse_coordinate(
                site_value
                    .get("position")
                    .ok_or_else(|| format!("site {id} missing position"))?,
                dimension,
                &format!("site {id} position"),
            )?;
            let mut integer_coordinates = Vec::with_capacity(dimension);
            for axis in 0..dimension {
                let rounded = position[axis].round();
                if (position[axis] - rounded).abs() > 1.0e-10
                    || rounded < 0.0
                    || rounded >= shape[axis] as f64
                {
                    return Err(format!("site {id} coordinate is outside the integer lattice"));
                }
                integer_coordinates.push(rounded as usize);
            }
            if !coordinate_set.insert(integer_coordinates) {
                return Err(format!("duplicate lattice coordinate at site {id}"));
            }

            let field_values = site_value
                .get("component_fields")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("site {id} component_fields must be an array"))?;
            if field_values.len() != component_count {
                return Err(format!("site {id} must have one field per component"));
            }
            let mut fields = vec![Complex::default(); component_count];
            let mut seen_components = HashSet::with_capacity(component_count);
            for field_value in field_values {
                let component_id = required_text(field_value, "component_id")?;
                let component = parse_component_index(
                    component_id,
                    &component_ids,
                    &format!("site {id}"),
                )?;
                if !seen_components.insert(component) {
                    return Err(format!("site {id} repeats component {component_id}"));
                }
                fields[component] = Complex {
                    re: finite_number(
                        field_value
                            .get("psi_real")
                            .ok_or_else(|| format!("site {id} missing psi_real"))?,
                        "psi_real",
                    )?,
                    im: finite_number(
                        field_value
                            .get("psi_imag")
                            .ok_or_else(|| format!("site {id} missing psi_imag"))?,
                        "psi_imag",
                    )?,
                };
            }
            sites.push(Site {
                id,
                position,
                fields,
            });
        }
        if coordinate_set.len() != expected_site_count {
            return Err("site coordinates do not cover the declared lattice exactly".to_owned());
        }

        let edge_values = lattice
            .get("site_edges")
            .and_then(Value::as_array)
            .ok_or_else(|| "lattice.site_edges must be an array".to_owned())?;
        let mut edge_ids = HashSet::with_capacity(edge_values.len());
        let mut edges = Vec::with_capacity(edge_values.len());
        for edge_value in edge_values {
            let id = required_text(edge_value, "edge_id")?.to_owned();
            if !edge_ids.insert(id.clone()) {
                return Err(format!("duplicate lattice edge ID {id}"));
            }
            let source_id = required_text(edge_value, "source_site_id")?;
            let target_id = required_text(edge_value, "target_site_id")?;
            let source = site_index
                .get(source_id)
                .copied()
                .ok_or_else(|| format!("edge {id} has unknown source site {source_id}"))?;
            let target = site_index
                .get(target_id)
                .copied()
                .ok_or_else(|| format!("edge {id} has unknown target site {target_id}"))?;
            if source == target {
                return Err(format!("edge {id} must not be a self-loop"));
            }
            let axis = edge_value
                .get("axis_index")
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("edge {id} axis_index must be an integer"))?
                as usize;
            if axis >= dimension {
                return Err(format!("edge {id} axis index is out of range"));
            }
            let shift_values = edge_value
                .get("periodic_image_shift")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("edge {id} periodic_image_shift must be an array"))?;
            if shift_values.len() != dimension {
                return Err(format!("edge {id} image-shift dimension mismatch"));
            }
            let mut image_shift = Vec::with_capacity(dimension);
            for shift_value in shift_values {
                image_shift.push(
                    shift_value
                        .as_i64()
                        .ok_or_else(|| format!("edge {id} image shifts must be integers"))?,
                );
            }
            for direction in 0..dimension {
                if image_shift[direction] != 0
                    && boundary_conditions[direction] != "periodic"
                {
                    return Err(format!(
                        "edge {id} has periodic image shift on an open boundary"
                    ));
                }
            }
            let gauge_phase = finite_number(
                edge_value
                    .get("gauge_link_phase_rad")
                    .ok_or_else(|| format!("edge {id} missing gauge_link_phase_rad"))?,
                "gauge_link_phase_rad",
            )?;

            let mut displacement = Vec::with_capacity(dimension);
            for direction in 0..dimension {
                displacement.push(
                    sites[target].position[direction]
                        + image_shift[direction] as f64 * shape[direction] as f64
                        - sites[source].position[direction],
                );
            }
            for direction in 0..dimension {
                let expected = if direction == axis { 1.0 } else { 0.0 };
                if !close_enough(displacement[direction], expected, 1.0e-10) {
                    return Err(format!(
                        "edge {id} must be an oriented positive unit bond on axis {axis}"
                    ));
                }
            }
            edges.push(Edge {
                id,
                source,
                target,
                axis,
                image_shift,
                gauge_phase,
            });
        }
        if edges.is_empty() {
            return Err("at least one lattice edge is required".to_owned());
        }

        let couplings = fixture
            .get("couplings")
            .ok_or_else(|| "missing coupling declaration".to_owned())?;
        let quadratic = parse_symmetric_matrix(
            couplings
                .get("quadratic_matrix")
                .ok_or_else(|| "missing quadratic_matrix".to_owned())?,
            component_count,
            "quadratic_matrix",
        )?;
        let gradient_values = couplings
            .get("gradient_matrices_by_axis")
            .and_then(Value::as_array)
            .ok_or_else(|| "gradient_matrices_by_axis must be an array".to_owned())?;
        if gradient_values.len() != dimension {
            return Err("gradient matrix count must match lattice dimension".to_owned());
        }
        let mut gradient_by_axis = Vec::with_capacity(dimension);
        let mut gradient_min_eigenvalues = Vec::with_capacity(dimension);
        for (axis, matrix_value) in gradient_values.iter().enumerate() {
            let matrix = parse_symmetric_matrix(
                matrix_value,
                component_count,
                &format!("gradient matrix axis {axis}"),
            )?;
            let minimum = minimum_symmetric_eigenvalue(&matrix)?;
            if minimum < -1.0e-10 {
                return Err(format!(
                    "gradient matrix on axis {axis} is not positive semidefinite (min eigenvalue {minimum})"
                ));
            }
            gradient_min_eigenvalues.push(minimum);
            gradient_by_axis.push(matrix);
        }

        let beta_values = couplings
            .get("quartic_self_coefficients")
            .and_then(Value::as_array)
            .ok_or_else(|| "quartic_self_coefficients must be an array".to_owned())?;
        if beta_values.len() != component_count {
            return Err("quartic self-coefficient count must match components".to_owned());
        }
        let quartic_self: Vec<f64> = beta_values
            .iter()
            .enumerate()
            .map(|(index, value)| finite_number(value, &format!("quartic_self_coefficients[{index}]")))
            .collect::<Result<_, _>>()?;
        if quartic_self.iter().any(|value| *value <= 0.0) {
            return Err("all quartic self coefficients must be positive".to_owned());
        }
        let mut quartic_matrix = vec![vec![0.0; component_count]; component_count];
        for component in 0..component_count {
            quartic_matrix[component][component] = 0.5 * quartic_self[component];
        }
        let mut density_terms = Vec::new();
        let mut density_pairs = HashSet::new();
        let density_values = couplings
            .get("quartic_density_density_terms")
            .and_then(Value::as_array)
            .ok_or_else(|| "quartic_density_density_terms must be an array".to_owned())?;
        for term in density_values {
            let first_id = required_text(term, "component_a")?;
            let second_id = required_text(term, "component_b")?;
            let first = parse_component_index(first_id, &component_ids, "density term")?;
            let second = parse_component_index(second_id, &component_ids, "density term")?;
            if first == second {
                return Err("density-density terms must couple distinct components".to_owned());
            }
            let pair = (first.min(second), first.max(second));
            if !density_pairs.insert(pair) {
                return Err(format!("duplicate density-density coupling {first_id}/{second_id}"));
            }
            let coefficient = finite_number(
                term.get("coefficient")
                    .ok_or_else(|| "density term missing coefficient".to_owned())?,
                "density-density coefficient",
            )?;
            quartic_matrix[first][second] = 0.5 * coefficient;
            quartic_matrix[second][first] = 0.5 * coefficient;
            density_terms.push(DensityTerm {
                first,
                second,
                coefficient,
            });
        }
        let quartic_min_eigenvalue = minimum_symmetric_eigenvalue(&quartic_matrix)?;
        if quartic_min_eigenvalue < -1.0e-10 {
            return Err(format!(
                "quartic amplitude form is not positive semidefinite (min eigenvalue {quartic_min_eigenvalue}); initial evaluator requires this sufficient boundedness condition"
            ));
        }

        let mut phase_locking_terms = Vec::new();
        let mut phase_pairs = HashSet::new();
        let phase_values = couplings
            .get("phase_locking_terms")
            .and_then(Value::as_array)
            .ok_or_else(|| "phase_locking_terms must be an array".to_owned())?;
        for term in phase_values {
            let first_id = required_text(term, "component_a")?;
            let second_id = required_text(term, "component_b")?;
            let first = parse_component_index(first_id, &component_ids, "phase-locking term")?;
            let second = parse_component_index(second_id, &component_ids, "phase-locking term")?;
            if first == second {
                return Err("phase-locking terms must couple distinct components".to_owned());
            }
            let pair = (first.min(second), first.max(second));
            if !phase_pairs.insert(pair) {
                return Err(format!("duplicate phase-locking term {first_id}/{second_id}"));
            }
            let coefficient = finite_number(
                term.get("coefficient")
                    .ok_or_else(|| "phase-locking term missing coefficient".to_owned())?,
                "phase-locking coefficient",
            )?;
            if required_text(term, "convention")?
                != "minus_two_J_real_psi_a_conjugate_psi_b"
            {
                return Err("unsupported phase-locking convention".to_owned());
            }
            phase_locking_terms.push(PhaseLockTerm {
                first,
                second,
                coefficient,
            });
        }
        if required_text(couplings, "gauge_convention")?
            != "single_common_U1_gauge_transform_for_all_charge_2e_components"
        {
            return Err("all pair components must share one common electromagnetic U(1)".to_owned());
        }

        let secondary_values = fixture
            .get("secondary_order_fields")
            .and_then(Value::as_array)
            .ok_or_else(|| "secondary_order_fields must be an array".to_owned())?;
        let mut secondary_orders = Vec::with_capacity(secondary_values.len());
        let mut secondary_ids = HashSet::new();
        for secondary in secondary_values {
            let id = required_text(secondary, "field_id")?.to_owned();
            if !secondary_ids.insert(id.clone()) {
                return Err(format!("duplicate secondary order field ID {id}"));
            }
            let kind = required_text(secondary, "field_kind")?.to_owned();
            let values = secondary
                .get("site_values")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("secondary field {id} site_values must be an array"))?;
            if values.len() != sites.len() {
                return Err(format!("secondary field {id} must have one value per site"));
            }
            let mut site_value_map = HashMap::with_capacity(values.len());
            for item in values {
                let site_id = required_text(item, "site_id")?;
                let index = site_index
                    .get(site_id)
                    .copied()
                    .ok_or_else(|| format!("secondary field {id} has unknown site {site_id}"))?;
                let value = finite_number(
                    item.get("value")
                        .ok_or_else(|| format!("secondary field {id} missing value"))?,
                    "secondary order site value",
                )?;
                if site_value_map.insert(index, value).is_some() {
                    return Err(format!("secondary field {id} repeats site {site_id}"));
                }
            }
            if site_value_map.len() != sites.len() {
                return Err(format!("secondary field {id} does not cover every site"));
            }
            let site_values = (0..sites.len())
                .map(|index| site_value_map[&index])
                .collect();

            let component_values = secondary
                .get("component_couplings")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("secondary field {id} component_couplings must be an array"))?;
            let mut component_couplings = Vec::with_capacity(component_values.len());
            let mut seen_components = HashSet::new();
            for item in component_values {
                let component_id = required_text(item, "component_id")?;
                let component = parse_component_index(
                    component_id,
                    &component_ids,
                    &format!("secondary field {id}"),
                )?;
                if !seen_components.insert(component) {
                    return Err(format!("secondary field {id} repeats component {component_id}"));
                }
                if required_text(item, "term")?
                    != "secondary_field_times_component_amplitude_squared"
                {
                    return Err(format!("secondary field {id} has unsupported coupling term"));
                }
                let coefficient = finite_number(
                    item.get("coefficient")
                        .ok_or_else(|| format!("secondary field {id} coupling missing coefficient"))?,
                    "secondary field coefficient",
                )?;
                component_couplings.push((component, coefficient));
            }
            secondary_orders.push(SecondaryOrder {
                id,
                kind,
                site_values,
                component_coefficients: component_couplings,
            });
        }

        let stability_gates = fixture
            .get("stability_gates")
            .ok_or_else(|| "missing stability_gates".to_owned())?;
        if stability_gates
            .get("superfluid_stiffness_must_be_positive_semidefinite")
            .and_then(Value::as_bool)
            != Some(true)
            || stability_gates
                .get("negative_stiffness_is_hard_failure")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err("stiffness positivity must be declared as a hard stability gate".to_owned());
        }
        let stiffness_tolerance = finite_number(
            stability_gates
                .get("stiffness_eigenvalue_tolerance")
                .ok_or_else(|| "missing stiffness_eigenvalue_tolerance".to_owned())?,
            "stiffness eigenvalue tolerance",
        )?;
        if stiffness_tolerance < 0.0 {
            return Err("stiffness eigenvalue tolerance must be non-negative".to_owned());
        }

        Ok(Self {
            instance_id,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            fixture_sha256: format!("{:x}", Sha256::digest(fixture_bytes)),
            dimension,
            shape,
            boundary_conditions,
            components,
            sites,
            edges,
            quadratic,
            gradient_by_axis,
            quartic_self,
            density_terms,
            phase_locking_terms,
            secondary_orders,
            stiffness_tolerance,
            gradient_min_eigenvalues,
            quartic_min_eigenvalue,
        })
    }

    fn initial_fields(&self) -> Vec<Vec<Complex>> {
        self.sites.iter().map(|site| site.fields.clone()).collect()
    }

    fn initial_link_phases(&self) -> Vec<f64> {
        self.edges.iter().map(|edge| edge.gauge_phase).collect()
    }

    fn energy_for(
        &self,
        fields: &[Vec<Complex>],
        link_phases: &[f64],
    ) -> Result<EnergyBreakdown, String> {
        if fields.len() != self.sites.len()
            || fields.iter().any(|site_fields| site_fields.len() != self.components.len())
            || link_phases.len() != self.edges.len()
        {
            return Err("multicomponent field or link dimensions do not match the model".to_owned());
        }
        if fields.iter().flatten().any(|field| {
            !field.re.is_finite() || !field.im.is_finite()
        }) || link_phases.iter().any(|phase| !phase.is_finite()) {
            return Err("multicomponent field or link phase is non-finite".to_owned());
        }

        let component_count = self.components.len();
        let mut quadratic = 0.0;
        let mut quartic_self = 0.0;
        let mut quartic_density = 0.0;
        let mut phase_locking = 0.0;
        let mut secondary_order = 0.0;
        let mut gradient = 0.0;

        for site_index in 0..self.sites.len() {
            let site_fields = &fields[site_index];
            for first in 0..component_count {
                for second in 0..component_count {
                    quadratic += self.quadratic[first][second]
                        * site_fields[first].conj_product_real(site_fields[second]);
                }
                let magnitude_squared = site_fields[first].norm_sqr();
                quartic_self += 0.5
                    * self.quartic_self[first]
                    * magnitude_squared
                    * magnitude_squared;
            }
            for term in &self.density_terms {
                quartic_density += term.coefficient
                    * site_fields[term.first].norm_sqr()
                    * site_fields[term.second].norm_sqr();
            }
            for term in &self.phase_locking_terms {
                phase_locking -= 2.0
                    * term.coefficient
                    * site_fields[term.first].conj_product_real(site_fields[term.second]);
            }
            for secondary in &self.secondary_orders {
                let order_value = secondary.site_values[site_index];
                for (component, coefficient) in &secondary.component_coefficients {
                    secondary_order += coefficient
                        * order_value
                        * site_fields[*component].norm_sqr();
                }
            }
        }

        for (edge_index, edge) in self.edges.iter().enumerate() {
            let gauge_phase = link_phases[edge_index];
            let mut differences = Vec::with_capacity(component_count);
            for component in 0..component_count {
                differences.push(
                    fields[edge.source][component]
                        .sub(fields[edge.target][component].rotate(gauge_phase)),
                );
            }
            let stiffness_matrix = &self.gradient_by_axis[edge.axis];
            for first in 0..component_count {
                for second in 0..component_count {
                    gradient += stiffness_matrix[first][second]
                        * differences[first].conj_product_real(differences[second]);
                }
            }
        }

        let total = quadratic
            + quartic_self
            + quartic_density
            + phase_locking
            + secondary_order
            + gradient;
        if [
            quadratic,
            quartic_self,
            quartic_density,
            phase_locking,
            secondary_order,
            gradient,
            total,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err("multicomponent GL energy became non-finite".to_owned());
        }
        Ok(EnergyBreakdown {
            quadratic,
            quartic_self,
            quartic_density,
            phase_locking,
            secondary_order,
            gradient,
            total,
        })
    }

    fn common_gauge_transform(
        &self,
        fields: &[Vec<Complex>],
        link_phases: &[f64],
        site_angles: &[f64],
    ) -> Result<(Vec<Vec<Complex>>, Vec<f64>), String> {
        if site_angles.len() != self.sites.len()
            || link_phases.len() != self.edges.len()
            || fields.len() != self.sites.len()
        {
            return Err("gauge transformation dimensions do not match the model".to_owned());
        }
        if site_angles.iter().any(|angle| !angle.is_finite()) {
            return Err("gauge transformation contains non-finite angles".to_owned());
        }
        let transformed_fields = fields
            .iter()
            .zip(site_angles)
            .map(|(site_fields, angle)| {
                site_fields.iter().map(|field| field.rotate(*angle)).collect()
            })
            .collect();
        let transformed_links = self
            .edges
            .iter()
            .zip(link_phases)
            .map(|(edge, phase)| {
                phase + site_angles[edge.source] - site_angles[edge.target]
            })
            .collect();
        Ok((transformed_fields, transformed_links))
    }

    fn component_mean_amplitudes(&self, fields: &[Vec<Complex>]) -> Vec<f64> {
        (0..self.components.len())
            .map(|component| {
                fields
                    .iter()
                    .map(|site| site[component].norm_sqr().sqrt())
                    .sum::<f64>()
                    / fields.len() as f64
            })
            .collect()
    }

    fn receipt(&self, energy: &EnergyBreakdown, gauge_residual: f64) -> Value {
        let component_mean_amplitudes = self.component_mean_amplitudes(&self.initial_fields());
        let component_metadata: Vec<Value> = self
            .components
            .iter()
            .enumerate()
            .map(|(index, component)| {
                json!({
                    "component_id": component.id,
                    "component_role": component.role,
                    "band_orbital_id": component.band_orbital_id,
                    "symmetry_label": component.symmetry_label,
                    "momentum_vector": component.momentum,
                    "mean_amplitude": component_mean_amplitudes[index],
                    "basis_gauge_charge": "Cooper_pair_charge_2e"
                })
            })
            .collect();
        json!({
            "status": "pass",
            "contract": "SC-INTERTWINED-ORDER-AUDIT-0.1",
            "instance_id": self.instance_id,
            "schema_sha256": self.schema_sha256,
            "fixture_sha256": self.fixture_sha256,
            "lattice": {
                "dimension": self.dimension,
                "shape": self.shape,
                "boundary_conditions": self.boundary_conditions,
                "site_count": self.sites.len(),
                "edge_count": self.edges.len(),
                "sites": self.sites.iter().map(|site| json!({
                    "site_id": site.id,
                    "position": site.position
                })).collect::<Vec<_>>(),
                "edges": self.edges.iter().map(|edge| json!({
                    "edge_id": edge.id,
                    "source_site_id": self.sites[edge.source].id,
                    "target_site_id": self.sites[edge.target].id,
                    "axis_index": edge.axis,
                    "periodic_image_shift": edge.image_shift,
                    "gauge_link_phase_rad": edge.gauge_phase
                })).collect::<Vec<_>>()
            },
            "components": component_metadata,
            "model_scope": {
                "evaluated_component_momenta": "zero_only",
                "finite_q_pdw_evaluated": false,
                "common_electromagnetic_u1": true,
                "secondary_order_field_count": self.secondary_orders.len()
            },
            "free_energy": {
                "quadratic": energy.quadratic,
                "quartic_self": energy.quartic_self,
                "quartic_density": energy.quartic_density,
                "phase_locking": energy.phase_locking,
                "secondary_order": energy.secondary_order,
                "gradient": energy.gradient,
                "total": energy.total,
                "unit": "dimensionless_lattice_energy"
            },
            "model_well_posedness": {
                "gradient_matrix_min_eigenvalues": self.gradient_min_eigenvalues,
                "quartic_amplitude_form_min_eigenvalue": self.quartic_min_eigenvalue,
                "gradient_matrices_positive_semidefinite": true,
                "quartic_amplitude_form_positive_semidefinite": true,
                "quartic_psd_note": "Positive semidefiniteness is a sufficient boundedness condition for this quartic amplitude form, not a necessary condition for all GL functionals."
            },
            "common_gauge_invariance": {
                "absolute_energy_residual": gauge_residual,
                "tolerance": 1.0e-9,
                "passed": gauge_residual <= 1.0e-9
            },
            "superfluid_stiffness_gate": {
                "required": true,
                "negative_eigenvalue_is_hard_failure": true,
                "tolerance": self.stiffness_tolerance,
                "status": "not_evaluated_by_static_configuration_auditor"
            },
            "claim_ceiling": "Synthetic zero-momentum multicomponent GL configuration and static energy only; no thermal equilibrium, stiffness, PDW identification, time-reversal symmetry breaking, Tc, or experimental result is established."
        })
    }
}

fn audit_and_evaluate(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let model = IntertwinedModel::from_bytes(schema_bytes, fixture_bytes)?;
    let fields = model.initial_fields();
    let links = model.initial_link_phases();
    let energy = model.energy_for(&fields, &links)?;

    let gauge_angles: Vec<f64> = (0..model.sites.len())
        .map(|index| {
            let value = index as f64 + 1.0;
            0.19 * value + 0.037 * value * value
        })
        .collect();
    let (transformed_fields, transformed_links) =
        model.common_gauge_transform(&fields, &links, &gauge_angles)?;
    let transformed_energy = model.energy_for(&transformed_fields, &transformed_links)?;
    let gauge_residual = (energy.total - transformed_energy.total).abs();
    if gauge_residual > 1.0e-9 {
        return Err(format!(
            "common electromagnetic gauge transformation changed energy by {gauge_residual}"
        ));
    }
    Ok(model.receipt(&energy, gauge_residual))
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --fixture <intertwined-order-instance.json> [--out <receipt.json>]"
    );
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
            eprintln!("SC-INTERTWINED-ORDER-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing intertwined-order receipt should not fail");
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
    use std::fs;

    fn inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_intertwined_order_instance_v0_1.schema.json"),
        )
        .unwrap();
        let fixture = fs::read(
            root.join("data/research/superconductivity_intertwined_order_fixture_v0_1.json"),
        )
        .unwrap();
        (schema, fixture)
    }

    fn mutate_and_parse(mutation: impl FnOnce(&mut Value)) -> Result<IntertwinedModel, String> {
        let (schema, bytes) = inputs();
        let mut fixture: Value = serde_json::from_slice(&bytes).unwrap();
        mutation(&mut fixture);
        IntertwinedModel::from_bytes(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn synthetic_three_component_fixture_has_finite_energy() {
        let (schema, fixture) = inputs();
        let receipt = audit_and_evaluate(&schema, &fixture).unwrap();
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["components"].as_array().unwrap().len(), 3);
        assert!(receipt["free_energy"]["total"].as_f64().unwrap().is_finite());
        assert_eq!(receipt["common_gauge_invariance"]["passed"], true);
    }

    #[test]
    fn common_local_u1_transformation_preserves_all_component_energy() {
        let (schema, fixture) = inputs();
        let model = IntertwinedModel::from_bytes(&schema, &fixture).unwrap();
        let fields = model.initial_fields();
        let links: Vec<f64> = (0..model.edges.len())
            .map(|index| 0.17 * (index as f64 * 0.31).sin())
            .collect();
        let angles: Vec<f64> = (0..model.sites.len())
            .map(|index| 0.23 * (index as f64 + 0.5).cos())
            .collect();
        let before = model.energy_for(&fields, &links).unwrap();
        let (transformed_fields, transformed_links) =
            model.common_gauge_transform(&fields, &links, &angles).unwrap();
        let after = model
            .energy_for(&transformed_fields, &transformed_links)
            .unwrap();
        assert!((before.total - after.total).abs() < 1.0e-9);
        assert!((before.gradient - after.gradient).abs() < 1.0e-9);
        assert!((before.quadratic - after.quadratic).abs() < 1.0e-9);
        assert!((before.phase_locking - after.phase_locking).abs() < 1.0e-9);
    }

    #[test]
    fn independent_component_phase_rotation_is_not_a_gauge_symmetry() {
        let (schema, fixture) = inputs();
        let model = IntertwinedModel::from_bytes(&schema, &fixture).unwrap();
        let fields = model.initial_fields();
        let links = model.initial_link_phases();
        let before = model.energy_for(&fields, &links).unwrap();
        let mut independently_rotated = fields.clone();
        for site in &mut independently_rotated {
            site[0] = site[0].rotate(0.71);
        }
        let after = model.energy_for(&independently_rotated, &links).unwrap();
        assert!((before.total - after.total).abs() > 1.0e-6);
    }

    #[test]
    fn uniform_component_fields_have_zero_gradient_energy_at_zero_link_phase() {
        let (schema, fixture) = inputs();
        let model = IntertwinedModel::from_bytes(&schema, &fixture).unwrap();
        let fields = model.initial_fields();
        let energy = model
            .energy_for(&fields, &vec![0.0; model.edges.len()])
            .unwrap();
        assert!(energy.gradient.abs() < 1.0e-12);
    }

    #[test]
    fn finite_momentum_pdw_component_is_rejected_by_zero_q_evaluator() {
        assert!(mutate_and_parse(|fixture| {
            fixture["components"][0]["momentum_vector"] = json!([0.25, 0.0]);
        })
        .is_err());
    }

    #[test]
    fn non_positive_semidefinite_gradient_matrix_is_rejected() {
        assert!(mutate_and_parse(|fixture| {
            fixture["couplings"]["gradient_matrices_by_axis"][0] =
                json!([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        })
        .is_err());
    }

    #[test]
    fn unbounded_quartic_amplitude_form_is_rejected() {
        assert!(mutate_and_parse(|fixture| {
            fixture["couplings"]["quartic_density_density_terms"][0]["coefficient"] =
                json!(-4.0);
        })
        .is_err());
    }

    #[test]
    fn unknown_component_field_identity_is_rejected() {
        assert!(mutate_and_parse(|fixture| {
            fixture["field_configuration"]["sites"][0]["component_fields"][0]["component_id"] =
                json!("missing-component");
        })
        .is_err());
    }

    #[test]
    fn basis_permutation_preserves_physical_energy_when_all_tensors_are_permuted() {
        let (schema, bytes) = inputs();
        let original: Value = serde_json::from_slice(&bytes).unwrap();
        let original_model = IntertwinedModel::from_bytes(&schema, &bytes).unwrap();
        let original_energy = original_model
            .energy_for(
                &original_model.initial_fields(),
                &original_model.initial_link_phases(),
            )
            .unwrap()
            .total;

        let permutation = [2_usize, 0, 1];
        let mut permuted = original;
        let old_components = permuted["components"].as_array().unwrap().clone();
        permuted["components"] =
            json!(permutation.iter().map(|index| old_components[*index].clone()).collect::<Vec<_>>());

        for site in permuted["field_configuration"]["sites"].as_array_mut().unwrap() {
            let old_fields = site["component_fields"].as_array().unwrap().clone();
            site["component_fields"] =
                json!(permutation.iter().map(|index| old_fields[*index].clone()).collect::<Vec<_>>());
        }
        let reorder_matrix = |matrix: &Value| -> Value {
            let rows = matrix.as_array().unwrap();
            json!(permutation.iter().map(|i| {
                let row = rows[*i].as_array().unwrap();
                permutation.iter().map(|j| row[*j].clone()).collect::<Vec<_>>()
            }).collect::<Vec<_>>())
        };
        permuted["couplings"]["quadratic_matrix"] =
            reorder_matrix(&permuted["couplings"]["quadratic_matrix"]);
        for axis in permuted["couplings"]["gradient_matrices_by_axis"]
            .as_array_mut()
            .unwrap()
        {
            *axis = reorder_matrix(axis);
        }
        let old_beta = permuted["couplings"]["quartic_self_coefficients"]
            .as_array()
            .unwrap()
            .clone();
        permuted["couplings"]["quartic_self_coefficients"] =
            json!(permutation.iter().map(|i| old_beta[*i].clone()).collect::<Vec<_>>());

        let permuted_bytes = serde_json::to_vec(&permuted).unwrap();
        let permuted_model = IntertwinedModel::from_bytes(&schema, &permuted_bytes).unwrap();
        let permuted_energy = permuted_model
            .energy_for(
                &permuted_model.initial_fields(),
                &permuted_model.initial_link_phases(),
            )
            .unwrap()
            .total;
        assert!((original_energy - permuted_energy).abs() < 1.0e-10);
    }
}
