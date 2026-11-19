use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::f64::consts::TAU;
use std::fs;
use std::process;

#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }

    fn sub(self, other: Self) -> Self {
        Self { re: self.re - other.re, im: self.im - other.im }
    }

    fn scale(self, factor: f64) -> Self {
        Self { re: self.re * factor, im: self.im * factor }
    }

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
}

#[derive(Clone, Debug)]
struct Mode {
    id: String,
    symmetry_label: String,
    q: usize,
    envelope: Vec<Complex>,
}

#[derive(Clone, Debug)]
struct Link {
    id: String,
    source: usize,
    target: usize,
    gauge_phase: f64,
}

#[derive(Clone, Debug)]
struct Model {
    instance_id: String,
    schema_sha256: String,
    fixture_sha256: String,
    length: usize,
    alpha: f64,
    beta: f64,
    gradient_coupling: f64,
    site_ids: Vec<String>,
    modes: Vec<Mode>,
    links: Vec<Link>,
}

#[derive(Clone, Copy, Debug)]
struct Energy {
    local: f64,
    gradient: f64,
    total: f64,
}

fn required_text(value: &Value, key: &str) -> Result<String, String> {
    value.get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("missing or empty text field {key}"))
}

fn finite_number(value: &Value, context: &str) -> Result<f64, String> {
    let number = value.as_f64().ok_or_else(|| format!("{context} must be numeric"))?;
    if !number.is_finite() {
        return Err(format!("{context} must be finite"));
    }
    Ok(number)
}

fn close_enough(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance * left.abs().max(right.abs()).max(1.0)
}

impl Model {
    fn from_bytes(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Self, String> {
        let schema: Value = serde_json::from_slice(schema_bytes)
            .map_err(|error| format!("schema JSON parse failed: {error}"))?;
        let fixture: Value = serde_json::from_slice(fixture_bytes)
            .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
        const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
        if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
            return Err(format!("schema must declare {DIALECT}"));
        }
        let validator = validator_for(&schema)
            .map_err(|error| format!("schema compilation failed: {error}"))?;
        if let Some(error) = validator.iter_errors(&fixture).next() {
            return Err(format!("finite-Q schema validation failed: {error} at {}", error.instance_path()));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("finite-Q v0.1 accepts synthetic fixtures only".to_owned());
        }
        if required_text(&fixture, "contract")? != "SC-FINITE-Q-LATTICE-0.1" {
            return Err("unsupported finite-Q contract".to_owned());
        }
        let instance_id = required_text(&fixture, "instance_id")?;
        let model = fixture.get("model").ok_or_else(|| "missing model".to_owned())?;
        if required_text(model, "kind")? != "static_commensurate_finite_q_lattice_gl"
            || required_text(model, "energy_unit")? != "dimensionless_lattice_energy"
            || required_text(model, "order_parameter_unit")? != "dimensionless"
            || required_text(model, "link_convention")? != "delta_source_minus_exp_iA_delta_target"
            || required_text(model, "momentum_convention")? != "q_index_integer_mod_ring_length"
        {
            return Err("unsupported finite-Q model or unit convention".to_owned());
        }
        let alpha = finite_number(model.get("alpha").ok_or_else(|| "missing alpha".to_owned())?, "alpha")?;
        let beta = finite_number(model.get("beta").ok_or_else(|| "missing beta".to_owned())?, "beta")?;
        let gradient_coupling = finite_number(
            model.get("gradient_coupling").ok_or_else(|| "missing gradient_coupling".to_owned())?,
            "gradient_coupling",
        )?;
        if beta <= 0.0 || gradient_coupling < 0.0 {
            return Err("beta must be positive and gradient_coupling non-negative".to_owned());
        }

        let lattice = fixture.get("lattice").ok_or_else(|| "missing lattice".to_owned())?;
        if lattice.get("dimension").and_then(Value::as_u64) != Some(1)
            || required_text(lattice, "boundary_condition")? != "periodic"
        {
            return Err("v0.1 supports only one-dimensional periodic rings".to_owned());
        }
        let length = lattice.get("length").and_then(Value::as_u64)
            .ok_or_else(|| "lattice.length must be an integer".to_owned())? as usize;
        if !(4..=256).contains(&length) {
            return Err("ring length must be in 4..=256".to_owned());
        }
        let site_values = lattice.get("site_ids").and_then(Value::as_array)
            .ok_or_else(|| "site_ids must be an array".to_owned())?;
        if site_values.len() != length {
            return Err("site_ids must enumerate every ring site exactly once".to_owned());
        }
        let mut site_ids = Vec::with_capacity(length);
        let mut seen_sites = HashSet::with_capacity(length);
        for item in site_values {
            let id = item.as_str().filter(|s| !s.trim().is_empty())
                .ok_or_else(|| "site ID must be a non-empty string".to_owned())?.to_owned();
            if !seen_sites.insert(id.clone()) {
                return Err(format!("duplicate site ID {id}"));
            }
            site_ids.push(id);
        }

        let link_values = lattice.get("links").and_then(Value::as_array)
            .ok_or_else(|| "links must be an array".to_owned())?;
        if link_values.len() != length {
            return Err("periodic ring requires exactly one directed link per site".to_owned());
        }
        let mut links = Vec::with_capacity(length);
        let mut link_ids = HashSet::with_capacity(length);
        let mut link_pairs = HashSet::with_capacity(length);
        for item in link_values {
            let id = required_text(item, "link_id")?;
            if !link_ids.insert(id.clone()) {
                return Err(format!("duplicate link ID {id}"));
            }
            let source = item.get("source_index").and_then(Value::as_u64)
                .ok_or_else(|| format!("link {id} source_index must be an integer"))? as usize;
            let target = item.get("target_index").and_then(Value::as_u64)
                .ok_or_else(|| format!("link {id} target_index must be an integer"))? as usize;
            if source >= length || target >= length || target != (source + 1) % length {
                return Err(format!("link {id} violates oriented periodic-ring geometry"));
            }
            if !link_pairs.insert((source, target)) {
                return Err(format!("duplicate directed ring edge in {id}"));
            }
            let gauge_phase = finite_number(
                item.get("gauge_phase_rad").ok_or_else(|| format!("link {id} missing gauge_phase_rad"))?,
                "gauge_phase_rad",
            )?;
            links.push(Link { id, source, target, gauge_phase });
        }
        links.sort_by_key(|edge| edge.source);
        for (site, edge) in links.iter().enumerate() {
            if edge.source != site {
                return Err("ring links must cover every directed source exactly once".to_owned());
            }
        }

        let mode_values = fixture.get("modes").and_then(Value::as_array)
            .ok_or_else(|| "modes must be an array".to_owned())?;
        if mode_values.is_empty() {
            return Err("at least one Fourier mode is required".to_owned());
        }
        let mut modes = Vec::with_capacity(mode_values.len());
        let mut mode_ids = HashSet::with_capacity(mode_values.len());
        let mut q_indices = HashSet::with_capacity(mode_values.len());
        for item in mode_values {
            let id = required_text(item, "mode_id")?;
            if !mode_ids.insert(id.clone()) {
                return Err(format!("duplicate mode ID {id}"));
            }
            let symmetry_label = required_text(item, "symmetry_label")?;
            let raw_q = item.get("q_index").and_then(Value::as_u64)
                .ok_or_else(|| format!("mode {id} q_index must be a non-negative integer"))? as usize;
            if raw_q >= length {
                return Err(format!("mode {id} q_index must be canonical in 0..{length}"));
            }
            if !q_indices.insert(raw_q) {
                return Err(format!("duplicate physical momentum q_index {raw_q}"));
            }
            let entries = item.get("envelope").and_then(Value::as_array)
                .ok_or_else(|| format!("mode {id} envelope must be an array"))?;
            if entries.len() != length {
                return Err(format!("mode {id} envelope must cover all {length} sites"));
            }
            let mut envelope = vec![None; length];
            for entry in entries {
                let site = entry.get("site_index").and_then(Value::as_u64)
                    .ok_or_else(|| format!("mode {id} site_index must be an integer"))? as usize;
                if site >= length || envelope[site].is_some() {
                    return Err(format!("mode {id} has duplicate or out-of-range site index {site}"));
                }
                envelope[site] = Some(Complex {
                    re: finite_number(entry.get("real").ok_or_else(|| "missing envelope real".to_owned())?, "envelope real")?,
                    im: finite_number(entry.get("imag").ok_or_else(|| "missing envelope imag".to_owned())?, "envelope imag")?,
                });
            }
            let envelope = envelope.into_iter().enumerate().map(|(site, value)| {
                value.ok_or_else(|| format!("mode {id} is missing site index {site}"))
            }).collect::<Result<Vec<_>, _>>()?;
            modes.push(Mode { id, symmetry_label, q: raw_q, envelope });
        }
        modes.sort_by_key(|mode| mode.q);

        Ok(Self {
            instance_id,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            fixture_sha256: format!("{:x}", Sha256::digest(fixture_bytes)),
            length,
            alpha,
            beta,
            gradient_coupling,
            site_ids,
            modes,
            links,
        })
    }

    fn physical_field(&self) -> Vec<Complex> {
        (0..self.length).map(|site| {
            self.modes.iter().fold(Complex { re: 0.0, im: 0.0 }, |sum, mode| {
                let angle = TAU * mode.q as f64 * site as f64 / self.length as f64;
                sum.add(mode.envelope[site].rotate(angle))
            })
        }).collect()
    }

    fn energy_for_field(&self, field: &[Complex], links: &[Link]) -> Result<Energy, String> {
        if field.len() != self.length {
            return Err("physical field length mismatch".to_owned());
        }
        let local = field.iter().map(|value| {
            let rho = value.norm_sqr();
            self.alpha * rho + 0.5 * self.beta * rho * rho
        }).sum::<f64>();
        let mut gradient = 0.0;
        for edge in links {
            let transported = field[edge.target].rotate(edge.gauge_phase);
            gradient += self.gradient_coupling * field[edge.source].sub(transported).norm_sqr();
        }
        let total = local + gradient;
        if !local.is_finite() || !gradient.is_finite() || !total.is_finite() {
            return Err("free-energy evaluation produced a non-finite value".to_owned());
        }
        Ok(Energy { local, gradient, total })
    }

    fn energy(&self) -> Result<Energy, String> {
        self.energy_for_field(&self.physical_field(), &self.links)
    }

    fn gauge_transformed(&self, chi: &[f64], wrong_link_sign: bool) -> Result<Self, String> {
        if chi.len() != self.length || chi.iter().any(|x| !x.is_finite()) {
            return Err("gauge transformation requires one finite phase per site".to_owned());
        }
        let mut transformed = self.clone();
        for mode in &mut transformed.modes {
            for (site, value) in mode.envelope.iter_mut().enumerate() {
                *value = value.rotate(chi[site]);
            }
        }
        for edge in &mut transformed.links {
            let delta = if wrong_link_sign {
                chi[edge.target] - chi[edge.source]
            } else {
                chi[edge.source] - chi[edge.target]
            };
            edge.gauge_phase += delta;
        }
        Ok(transformed)
    }

    #[cfg(test)]
    fn independently_rotated_mode(&self, mode_index: usize, chi: &[f64]) -> Result<Self, String> {
        if chi.len() != self.length || mode_index >= self.modes.len() {
            return Err("invalid independent component rotation".to_owned());
        }
        let mut transformed = self.clone();
        for (site, value) in transformed.modes[mode_index].envelope.iter_mut().enumerate() {
            *value = value.rotate(chi[site]);
        }
        for edge in &mut transformed.links {
            edge.gauge_phase += chi[edge.source] - chi[edge.target];
        }
        Ok(transformed)
    }

    fn fourier_spectrum(&self, field: &[Complex]) -> (Vec<Complex>, f64) {
        let mut coefficients = Vec::with_capacity(self.length);
        for q in 0..self.length {
            let sum = field.iter().enumerate().fold(
                Complex { re: 0.0, im: 0.0 },
                |acc, (site, value)| {
                    let angle = -TAU * q as f64 * site as f64 / self.length as f64;
                    acc.add(value.rotate(angle))
                },
            );
            coefficients.push(sum.scale(1.0 / self.length as f64));
        }
        let residual = self.parseval_residual(field, &coefficients);
        (coefficients, residual)
    }

    fn parseval_residual(&self, field: &[Complex], spectrum: &[Complex]) -> f64 {
        let real_space_mean = field.iter().map(|value| value.norm_sqr()).sum::<f64>()
            / self.length as f64;
        let reciprocal_sum = spectrum.iter().map(|value| value.norm_sqr()).sum::<f64>();
        reciprocal_sum - real_space_mean
    }

    fn momentum_quartet_allowed(&self, a: usize, b: usize, c: usize, d: usize) -> bool {
        let sum = self.modes[a].q as isize - self.modes[b].q as isize
            + self.modes[c].q as isize - self.modes[d].q as isize;
        sum.rem_euclid(self.length as isize) == 0
    }

    fn momentum_quartet_counts(&self) -> (usize, usize) {
        let count = self.modes.len();
        let mut allowed = 0;
        let mut total = 0;
        for a in 0..count {
            for b in 0..count {
                for c in 0..count {
                    for d in 0..count {
                        total += 1;
                        if self.momentum_quartet_allowed(a, b, c, d) {
                            allowed += 1;
                        }
                    }
                }
            }
        }
        (allowed, total)
    }

    fn receipt(&self) -> Result<Value, String> {
        let field = self.physical_field();
        let energy = self.energy_for_field(&field, &self.links)?;
        let (spectrum, parseval_residual) = self.fourier_spectrum(&field);
        let total_spectral_weight = spectrum.iter().map(|v| v.norm_sqr()).sum::<f64>();
        let finite_q_spectral_weight = spectrum.iter().enumerate()
            .filter(|(q, _)| *q != 0)
            .map(|(_, value)| value.norm_sqr()).sum::<f64>();
        let chi: Vec<f64> = (0..self.length)
            .map(|site| 0.37 * (site as f64 + 0.5).sin() + 0.11 * site as f64)
            .collect();
        let gauged = self.gauge_transformed(&chi, false)?;
        let gauge_energy = gauged.energy()?.total;
        let (allowed_quartets, total_quartets) = self.momentum_quartet_counts();
        Ok(json!({
            "contract": "SC-FINITE-Q-LATTICE-0.1",
            "instance_id": self.instance_id,
            "authority": "synthetic_fixture_only",
            "input_sha256": {
                "schema": self.schema_sha256,
                "fixture": self.fixture_sha256
            },
            "lattice": {
                "dimension": 1,
                "boundary_condition": "periodic",
                "length": self.length,
                "sites": self.site_ids.len(),
                "directed_links": self.links.len(),
                "link_ids": self.links.iter().map(|edge| edge.id.clone()).collect::<Vec<_>>()
            },
            "modes": self.modes.iter().map(|mode| json!({
                "mode_id": mode.id,
                "symmetry_label": mode.symmetry_label,
                "q_index": mode.q,
                "q_over_2pi": mode.q as f64 / self.length as f64
            })).collect::<Vec<_>>(),
            "physical_field": field.iter().map(|v| json!({"real":v.re,"imag":v.im})).collect::<Vec<_>>(),
            "free_energy": {
                "local_potential": energy.local,
                "gradient": energy.gradient,
                "total": energy.total,
                "unit": "dimensionless_lattice_energy"
            },
            "fourier": {
                "coefficients": spectrum.iter().enumerate().map(|(q,v)| json!({
                    "q_index":q,"real":v.re,"imag":v.im,"weight":v.norm_sqr()
                })).collect::<Vec<_>>(),
                "total_spectral_weight": total_spectral_weight,
                "finite_q_spectral_weight": finite_q_spectral_weight,
                "finite_q_spectral_weight_fraction": if total_spectral_weight > 0.0 {
                    finite_q_spectral_weight / total_spectral_weight
                } else { 0.0 },
                "parseval_residual": parseval_residual
            },
            "symmetry_audits": {
                "common_local_u1_gauge_energy_residual": (energy.total - gauge_energy).abs(),
                "quartic_momentum_conserving_ordered_tuples": allowed_quartets,
                "quartic_momentum_tuple_count": total_quartets
            },
            "stability_status": "not_evaluated",
            "thermal_helicity_modulus_status": "not_evaluated",
            "material_inference_status": "not_evaluated"
        }))
    }
}

fn read(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("SC-FINITE-Q-FAIL: cannot read {path}: {error}");
        process::exit(2);
    })
}

fn usage(program: &str) {
    eprintln!("usage: {program} --schema PATH --fixture PATH [--out PATH]");
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
                let value = args[index + 1].clone();
                match args[index].as_str() {
                    "--schema" => schema_path = Some(value),
                    "--fixture" => fixture_path = Some(value),
                    "--out" => output_path = Some(value),
                    _ => unreachable!(),
                }
                index += 2;
            }
            _ => {
                usage(&args[0]);
                process::exit(2);
            }
        }
    }
    let Some(schema_path) = schema_path else {
        usage(&args[0]);
        process::exit(2);
    };
    let Some(fixture_path) = fixture_path else {
        usage(&args[0]);
        process::exit(2);
    };
    let schema_bytes = read(&schema_path);
    let fixture_bytes = read(&fixture_path);
    let model = Model::from_bytes(&schema_bytes, &fixture_bytes).unwrap_or_else(|error| {
        eprintln!("SC-FINITE-Q-FAIL: {error}");
        process::exit(1);
    });
    let receipt = model.receipt().unwrap_or_else(|error| {
        eprintln!("SC-FINITE-Q-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt).expect("JSON receipt must serialize");
    if let Some(path) = output_path {
        fs::write(&path, format!("{rendered}\n")).unwrap_or_else(|error| {
            eprintln!("SC-FINITE-Q-FAIL: cannot write {path}: {error}");
            process::exit(2);
        });
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::{Complex, Model, finite_number};
    use serde_json::{Value, json};
    use std::fs;

    fn inputs() -> (Vec<u8>, Value) {
        let schema = fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/schemas/superconductivity_finite_q_lattice_v0_1.schema.json"
        )).expect("schema must be present");
        let fixture: Value = serde_json::from_slice(&fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/research/superconductivity_finite_q_lattice_fixture_v0_1.json"
        )).expect("fixture must be present")).expect("fixture must parse");
        (schema, fixture)
    }

    fn model_from(fixture: &Value) -> Model {
        let (schema, _) = inputs();
        let bytes = serde_json::to_vec(fixture).unwrap();
        Model::from_bytes(&schema, &bytes).unwrap()
    }

    fn expected_model_error(fixture: &Value) -> bool {
        let (schema, _) = inputs();
        let bytes = serde_json::to_vec(fixture).unwrap();
        Model::from_bytes(&schema, &bytes).is_err()
    }

    #[test]
    fn fixture_has_nonzero_finite_q_spectral_weight_and_parseval_holds() {
        let (schema, fixture) = inputs();
        let model = Model::from_bytes(&schema, &serde_json::to_vec(&fixture).unwrap()).unwrap();
        let field = model.physical_field();
        let (spectrum, residual) = model.fourier_spectrum(&field);
        assert!(residual.abs() < 1.0e-12);
        assert!(spectrum.iter().enumerate().any(|(q, v)| q != 0 && v.norm_sqr() > 1.0e-12));
        assert!(model.momentum_quartet_counts().0 > 0);
    }

    #[test]
    fn hand_calculated_plus_minus_q_fixture_energy_is_reproduced() {
        let model = model_from(&inputs().1);
        let energy = model.energy().unwrap();
        assert!((energy.local - (-0.1472)).abs() < 1.0e-12);
        assert!((energy.gradient - 0.32).abs() < 1.0e-12);
        assert!((energy.total - 0.1728).abs() < 1.0e-12);
    }

    #[test]
    fn common_local_gauge_transform_preserves_free_energy() {
        let model = model_from(&inputs().1);
        let chi: Vec<f64> = (0..model.length)
            .map(|site| 0.4 * (site as f64).cos() - 0.13 * site as f64)
            .collect();
        let transformed = model.gauge_transformed(&chi, false).unwrap();
        assert!((model.energy().unwrap().total - transformed.energy().unwrap().total).abs() < 1.0e-10);
    }

    #[test]
    fn wrong_link_sign_is_rejected_by_gauge_invariance_negative_control() {
        let model = model_from(&inputs().1);
        let chi: Vec<f64> = (0..model.length)
            .map(|site| 0.6 * (site as f64 * 0.7).sin())
            .collect();
        let transformed = model.gauge_transformed(&chi, true).unwrap();
        assert!((model.energy().unwrap().total - transformed.energy().unwrap().total).abs() > 1.0e-8);
    }

    #[test]
    fn independent_component_local_rotation_is_not_a_gauge_symmetry() {
        let model = model_from(&inputs().1);
        let chi: Vec<f64> = (0..model.length)
            .map(|site| 0.5 * (site as f64 * 0.8).cos())
            .collect();
        let transformed = model.independently_rotated_mode(1, &chi).unwrap();
        assert!((model.energy().unwrap().total - transformed.energy().unwrap().total).abs() > 1.0e-8);
    }

    #[test]
    fn duplicate_q_index_and_reciprocal_alias_are_rejected() {
        let mut fixture = inputs().1;
        fixture["modes"][2]["q_index"] = json!(2);
        assert!(expected_model_error(&fixture));
        let mut fixture = inputs().1;
        fixture["modes"][2]["q_index"] = json!(10);
        assert!(expected_model_error(&fixture));
    }

    #[test]
    fn incomplete_envelope_and_bad_periodic_link_are_rejected() {
        let mut fixture = inputs().1;
        fixture["modes"][0]["envelope"].as_array_mut().unwrap().pop();
        assert!(expected_model_error(&fixture));
        let mut fixture = inputs().1;
        fixture["lattice"]["links"][0]["target_index"] = json!(3);
        assert!(expected_model_error(&fixture));
    }

    #[test]
    fn quartic_momentum_selection_is_modulo_reciprocal_lattice() {
        let model = model_from(&inputs().1);
        let zero_mode = model.modes.iter().position(|mode| mode.q == 0).unwrap();
        let plus_mode = model.modes.iter().position(|mode| mode.q == 2).unwrap();
        let minus_mode = model.modes.iter().position(|mode| mode.q == 6).unwrap();
        assert!(model.momentum_quartet_allowed(plus_mode, zero_mode, minus_mode, zero_mode));
        assert!(!model.momentum_quartet_allowed(plus_mode, zero_mode, zero_mode, zero_mode));
    }

    #[test]
    fn zero_momentum_limit_matches_scalar_lattice_gl_energy() {
        let (schema, mut fixture) = inputs();
        fixture["modes"] = json!([{
            "mode_id":"uniform",
            "symmetry_label":"q-zero-scalar",
            "q_index":0,
            "envelope":(0..8).map(|site| json!({"site_index":site,"real":0.7,"imag":-0.1})).collect::<Vec<_>>()
        }]);
        let model = Model::from_bytes(&schema, &serde_json::to_vec(&fixture).unwrap()).unwrap();
        let energy = model.energy().unwrap();
        let amplitude_squared = Complex { re:0.7, im:-0.1 }.norm_sqr();
        let expected_local = 8.0 * (model.alpha * amplitude_squared
            + 0.5 * model.beta * amplitude_squared * amplitude_squared);
        assert!((energy.local - expected_local).abs() < 1.0e-12);
        assert!(energy.gradient.abs() < 1.0e-12);
    }

    #[test]
    fn deliberately_corrupted_fourier_spectrum_fails_parseval_negative_control() {
        let model = model_from(&inputs().1);
        let field = model.physical_field();
        let (mut spectrum, _) = model.fourier_spectrum(&field);
        spectrum[0] = spectrum[0].add(Complex { re: 0.1, im: -0.03 });
        assert!(model.parseval_residual(&field, &spectrum).abs() > 1.0e-4);
    }

    #[test]
    fn mode_permutation_does_not_change_the_physical_field_or_energy() {
        let mut fixture = inputs().1;
        let original = model_from(&fixture);
        fixture["modes"].as_array_mut().unwrap().reverse();
        let permuted = model_from(&fixture);
        assert!((original.energy().unwrap().total - permuted.energy().unwrap().total).abs() < 1.0e-12);
        for (left, right) in original.physical_field().iter().zip(permuted.physical_field()) {
            assert!((left.re - right.re).abs() < 1.0e-12);
            assert!((left.im - right.im).abs() < 1.0e-12);
        }
    }

    #[test]
    fn one_site_translation_preserves_periodic_ring_energy() {
        let mut fixture = inputs().1;
        let original = model_from(&fixture);
        let length = original.length;
        for mode in fixture["modes"].as_array_mut().unwrap() {
            let q = mode["q_index"].as_u64().unwrap() as usize;
            let old: Vec<Complex> = mode["envelope"].as_array().unwrap().iter().map(|entry| Complex {
                re: entry["real"].as_f64().unwrap(),
                im: entry["imag"].as_f64().unwrap(),
            }).collect();
            let phase = std::f64::consts::TAU * q as f64 / length as f64;
            let shifted: Vec<Value> = (0..length).map(|site| {
                let value = old[(site + 1) % length].rotate(phase);
                json!({"site_index":site,"real":value.re,"imag":value.im})
            }).collect();
            mode["envelope"] = json!(shifted);
        }
        let translated = model_from(&fixture);
        assert!((original.energy().unwrap().total - translated.energy().unwrap().total).abs() < 1.0e-10);
    }

    #[test]
    fn reconstructed_field_energy_matches_independent_real_space_sum() {
        let model = model_from(&inputs().1);
        let field = model.physical_field();
        let independent_local: f64 = field.iter().map(|value| {
            let rho = value.re * value.re + value.im * value.im;
            model.alpha * rho + 0.5 * model.beta * rho * rho
        }).sum();
        let independent_gradient: f64 = model.links.iter().map(|edge| {
            let angle = edge.gauge_phase;
            let (sine, cosine) = angle.sin_cos();
            let target = field[edge.target];
            let transported = Complex {
                re: cosine * target.re - sine * target.im,
                im: sine * target.re + cosine * target.im,
            };
            let difference = Complex {
                re: field[edge.source].re - transported.re,
                im: field[edge.source].im - transported.im,
            };
            model.gradient_coupling * difference.norm_sqr()
        }).sum();
        let energy = model.energy().unwrap();
        assert!((energy.local - independent_local).abs() < 1.0e-12);
        assert!((energy.gradient - independent_gradient).abs() < 1.0e-12);
    }

    #[test]
    fn numeric_input_rejects_nonfinite_values() {
        assert!(finite_number(&json!(1.0), "finite test").is_ok());
        assert!(finite_number(&Value::Null, "finite test").is_err());
    }
}
