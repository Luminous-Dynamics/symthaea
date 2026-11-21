use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Copy, Debug)]
struct Interval {
    lower: f64,
    upper: f64,
}
#[derive(Clone, Debug)]
struct Species {
    id: String,
    formula: String,
    composition: Vec<(String, f64)>,
    charge: f64,
}
#[derive(Clone, Debug)]
struct ReactionTerm {
    species_index: usize,
    coefficient: f64,
}
#[derive(Clone, Debug)]
struct Reaction {
    id: String,
    role: String,
    standard_state_id: String,
    terms: Vec<ReactionTerm>,
    delta_g: Interval,
    activation_barrier: Interval,
    source_class: String,
}
#[derive(Clone, Debug)]
struct Path {
    id: String,
    role: String,
    input_species: Vec<usize>,
    target_species: usize,
    reactions: Vec<usize>,
}
#[derive(Clone, Debug)]
struct CycleTerm {
    reaction_index: usize,
    orientation: i8,
}
#[derive(Clone, Debug)]
struct Cycle {
    id: String,
    terms: Vec<CycleTerm>,
}
#[derive(Clone, Debug)]
struct Balance {
    element_residuals: Vec<(String, f64)>,
    charge_residual: f64,
    balanced: bool,
}
#[derive(Clone, Debug)]
struct Network {
    id: String,
    schema_sha256: String,
    fixture_sha256: String,
    conditions_id: String,
    temperature_k: f64,
    pressure_gpa: f64,
    atmosphere: String,
    chemical_potential_context: String,
    extent_basis_id: String,
    extent_unit: String,
    standard_state_id: String,
    balance_tolerance: f64,
    cycle_tolerance: f64,
    species: Vec<Species>,
    reactions: Vec<Reaction>,
    paths: Vec<Path>,
    cycles: Vec<Cycle>,
    balances: Vec<Balance>,
}

fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|x| !x.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}
fn number_field(value: &Value, key: &str) -> Result<f64, String> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric field {key}"))?;
    if !number.is_finite() {
        return Err(format!("{key} must be finite"));
    }
    Ok(number)
}
fn interval(value: &Value, label: &str) -> Result<Interval, String> {
    let lower = number_field(value, "lower")?;
    let upper = number_field(value, "upper")?;
    if lower > upper {
        return Err(format!("{label} interval lower bound exceeds upper bound"));
    }
    Ok(Interval { lower, upper })
}
fn add_intervals(left: Interval, right: Interval) -> Interval {
    Interval {
        lower: left.lower + right.lower,
        upper: left.upper + right.upper,
    }
}
fn negate_interval(value: Interval) -> Interval {
    Interval {
        lower: -value.upper,
        upper: -value.lower,
    }
}
fn element_balance(species: &[Species], reaction: &Reaction, tolerance: f64) -> Balance {
    let mut residuals: HashMap<String, f64> = HashMap::new();
    let mut charge = 0.0;
    for term in &reaction.terms {
        let entry = &species[term.species_index];
        charge += term.coefficient * entry.charge;
        for (element, count) in &entry.composition {
            *residuals.entry(element.clone()).or_default() += term.coefficient * count;
        }
    }
    let mut elements: Vec<(String, f64)> = residuals.into_iter().collect();
    elements.sort_by(|a, b| a.0.cmp(&b.0));
    let balanced =
        elements.iter().all(|(_, value)| value.abs() <= tolerance) && charge.abs() <= tolerance;
    Balance {
        element_residuals: elements,
        charge_residual: charge,
        balanced,
    }
}
fn parse_species(rows: &[Value]) -> Result<(Vec<Species>, HashMap<String, usize>), String> {
    let mut species = Vec::with_capacity(rows.len());
    let mut index = HashMap::new();
    for row in rows {
        let id = text_field(row, "species_id")?.to_owned();
        if index.insert(id.clone(), species.len()).is_some() {
            return Err(format!("duplicate species ID {id}"));
        }
        let composition_value = row
            .get("elemental_composition")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("species {id} elemental_composition must be an object"))?;
        if composition_value.is_empty() {
            return Err(format!(
                "species {id} elemental_composition cannot be empty"
            ));
        }
        let mut composition = Vec::with_capacity(composition_value.len());
        for (element, value) in composition_value {
            if element.trim().is_empty() {
                return Err(format!("species {id} contains empty element symbol"));
            }
            let count = value
                .as_f64()
                .ok_or_else(|| format!("species {id} element {element} count must be numeric"))?;
            if !count.is_finite() || count <= 0.0 {
                return Err(format!(
                    "species {id} element {element} count must be positive and finite"
                ));
            }
            composition.push((element.clone(), count));
        }
        composition.sort_by(|a, b| a.0.cmp(&b.0));
        let charge = number_field(row, "charge")?;
        if charge.fract().abs() > 1.0e-12 {
            return Err(format!("species {id} charge must be an integer"));
        }
        species.push(Species {
            id,
            formula: text_field(row, "formula_as_declared")?.to_owned(),
            composition,
            charge,
        });
    }
    Ok((species, index))
}
fn parse_reactions(
    rows: &[Value],
    species: &[Species],
    species_index: &HashMap<String, usize>,
    conditions_id: &str,
    basis_id: &str,
    basis_unit: &str,
    standard_state_id: &str,
    balance_tolerance: f64,
) -> Result<(Vec<Reaction>, HashMap<String, usize>, Vec<Balance>), String> {
    let mut reactions = Vec::with_capacity(rows.len());
    let mut reaction_index = HashMap::new();
    let mut balances = Vec::with_capacity(rows.len());
    for row in rows {
        let id = text_field(row, "reaction_id")?.to_owned();
        if reaction_index.insert(id.clone(), reactions.len()).is_some() {
            return Err(format!("duplicate reaction ID {id}"));
        }
        if text_field(row, "conditions_id")? != conditions_id {
            return Err(format!(
                "reaction {id} condition ID does not match the network"
            ));
        }
        if text_field(row, "extent_basis_id")? != basis_id
            || text_field(row, "extent_unit")? != basis_unit
        {
            return Err(format!(
                "reaction {id} extent basis/unit does not match the network"
            ));
        }
        if text_field(row, "standard_state_id")? != standard_state_id {
            return Err(format!(
                "reaction {id} standard-state ID does not match the network"
            ));
        }
        let delta_g = interval(
            row.get("delta_g_interval_eV")
                .ok_or("reaction missing delta_g_interval_eV")?,
            &format!("reaction {id} ΔG"),
        )?;
        let activation_barrier = interval(
            row.get("activation_barrier_interval_eV")
                .ok_or("reaction missing activation_barrier_interval_eV")?,
            &format!("reaction {id} activation barrier"),
        )?;
        if activation_barrier.lower < 0.0 {
            return Err(format!(
                "reaction {id} activation barrier must be non-negative"
            ));
        }
        let mut terms = Vec::new();
        let mut seen = HashSet::new();
        let mut has_reactant = false;
        let mut has_product = false;
        for term in row
            .get("stoichiometric_terms")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("reaction {id} stoichiometric_terms must be an array"))?
        {
            let species_id = text_field(term, "species_id")?;
            let Some(species_position) = species_index.get(species_id) else {
                return Err(format!(
                    "reaction {id} references unknown species {species_id}"
                ));
            };
            if !seen.insert(*species_position) {
                return Err(format!("reaction {id} repeats species term {species_id}"));
            }
            let coefficient = number_field(term, "coefficient")?;
            if coefficient.abs() <= 1.0e-15 {
                return Err(format!(
                    "reaction {id} has a zero stoichiometric coefficient"
                ));
            }
            has_reactant |= coefficient < 0.0;
            has_product |= coefficient > 0.0;
            terms.push(ReactionTerm {
                species_index: *species_position,
                coefficient,
            });
        }
        if !has_reactant || !has_product {
            return Err(format!(
                "reaction {id} must contain both reactant and product terms"
            ));
        }
        let reaction = Reaction {
            id,
            role: text_field(row, "role")?.to_owned(),
            standard_state_id: text_field(row, "standard_state_id")?.to_owned(),
            terms,
            delta_g,
            activation_barrier,
            source_class: text_field(row, "source_class")?.to_owned(),
        };
        let balance = element_balance(species, &reaction, balance_tolerance);
        if !balance.balanced {
            let offending = balance
                .element_residuals
                .iter()
                .filter(|(_, value)| value.abs() > balance_tolerance)
                .map(|(element, value)| format!("{element} residual={value}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "reaction {} fails computed conservation: {}; charge residual={}",
                reaction.id, offending, balance.charge_residual
            ));
        }
        reactions.push(reaction);
        balances.push(balance);
    }
    Ok((reactions, reaction_index, balances))
}
fn parse_paths(
    rows: &[Value],
    reactions: &[Reaction],
    reaction_index: &HashMap<String, usize>,
    species: &[Species],
    species_index: &HashMap<String, usize>,
) -> Result<Vec<Path>, String> {
    let mut paths = Vec::with_capacity(rows.len());
    let mut path_ids = HashSet::new();
    for row in rows {
        let id = text_field(row, "path_id")?.to_owned();
        if !path_ids.insert(id.clone()) {
            return Err(format!("duplicate path ID {id}"));
        }
        let mut inputs = Vec::new();
        for species_id in row
            .get("input_species_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("path {id} input_species_ids must be an array"))?
        {
            let species_id = species_id
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| format!("path {id} has invalid input species ID"))?;
            inputs.push(*species_index.get(species_id).ok_or_else(|| {
                format!("path {id} references unknown input species {species_id}")
            })?);
        }
        let target_id = text_field(row, "target_species_id")?;
        let target_species = *species_index
            .get(target_id)
            .ok_or_else(|| format!("path {id} references unknown target species {target_id}"))?;
        let reaction_rows = row
            .get("reaction_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("path {id} reaction_ids must be an array"))?;
        if reaction_rows.is_empty() {
            return Err(format!("path {id} needs at least one reaction"));
        }
        let mut path_reactions = Vec::new();
        let mut seen_reactions = HashSet::new();
        for reaction_id in reaction_rows {
            let reaction_id = reaction_id
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| format!("path {id} contains an invalid reaction ID"))?;
            let Some(position) = reaction_index.get(reaction_id) else {
                return Err(format!(
                    "path {id} references unknown reaction {reaction_id}"
                ));
            };
            if !seen_reactions.insert(*position) {
                return Err(format!("path {id} repeats reaction {reaction_id}"));
            }
            path_reactions.push(*position);
        }
        let first = &reactions[path_reactions[0]];
        for input in &inputs {
            if !first
                .terms
                .iter()
                .any(|term| term.species_index == *input && term.coefficient < 0.0)
            {
                return Err(format!(
                    "path {id} first reaction does not consume input species {}",
                    species[*input].id
                ));
            }
        }
        for pair in path_reactions.windows(2) {
            let previous = &reactions[pair[0]];
            let next = &reactions[pair[1]];
            let connected = previous.terms.iter().any(|produced| {
                produced.coefficient > 0.0
                    && next.terms.iter().any(|consumed| {
                        consumed.species_index == produced.species_index
                            && consumed.coefficient < 0.0
                    })
            });
            if !connected {
                return Err(format!(
                    "path {id} is disconnected between reactions {} and {}",
                    previous.id, next.id
                ));
            }
        }
        let last = &reactions[*path_reactions.last().expect("non-empty path")];
        if !last
            .terms
            .iter()
            .any(|term| term.species_index == target_species && term.coefficient > 0.0)
        {
            return Err(format!(
                "path {id} final reaction does not produce target species {target_id}"
            ));
        }
        paths.push(Path {
            id,
            role: text_field(row, "path_role")?.to_owned(),
            input_species: inputs,
            target_species,
            reactions: path_reactions,
        });
    }
    Ok(paths)
}
fn parse_cycles(
    rows: &[Value],
    reaction_index: &HashMap<String, usize>,
) -> Result<Vec<Cycle>, String> {
    let mut cycles = Vec::with_capacity(rows.len());
    let mut cycle_ids = HashSet::new();
    for row in rows {
        let id = text_field(row, "cycle_id")?.to_owned();
        if !cycle_ids.insert(id.clone()) {
            return Err(format!("duplicate thermodynamic cycle ID {id}"));
        }
        let mut terms = Vec::new();
        let mut seen = HashSet::new();
        for term in row
            .get("terms")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("cycle {id} terms must be an array"))?
        {
            let reaction_id = text_field(term, "reaction_id")?;
            let position = *reaction_index
                .get(reaction_id)
                .ok_or_else(|| format!("cycle {id} references unknown reaction {reaction_id}"))?;
            if !seen.insert(position) {
                return Err(format!("cycle {id} repeats reaction {reaction_id}"));
            }
            let orientation = term
                .get("orientation")
                .and_then(Value::as_i64)
                .ok_or_else(|| format!("cycle {id} orientation must be integer +1 or -1"))?;
            if orientation != 1 && orientation != -1 {
                return Err(format!("cycle {id} orientation must be +1 or -1"));
            }
            terms.push(CycleTerm {
                reaction_index: position,
                orientation: orientation as i8,
            });
        }
        cycles.push(Cycle { id, terms });
    }
    Ok(cycles)
}
impl Network {
    fn from_bytes(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Self, String> {
        let schema: Value = serde_json::from_slice(schema_bytes)
            .map_err(|e| format!("schema JSON parse failed: {e}"))?;
        let fixture: Value = serde_json::from_slice(fixture_bytes)
            .map_err(|e| format!("fixture JSON parse failed: {e}"))?;
        const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
        if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
            return Err(format!("schema must declare {DIALECT}"));
        }
        let validator =
            validator_for(&schema).map_err(|e| format!("schema compilation failed: {e}"))?;
        if let Some(error) = validator.iter_errors(&fixture).next() {
            return Err(format!(
                "thermochemical network schema validation failed: {error} at {}",
                error.instance_path()
            ));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err(
                "thermochemical network auditor accepts synthetic fixtures only".to_owned(),
            );
        }
        if text_field(&fixture, "contract")? != "SC-THERMOCHEMICAL-NETWORK-0.1" {
            return Err("unsupported thermochemical network contract".to_owned());
        }
        let conditions = fixture
            .get("conditions")
            .ok_or("network missing conditions")?;
        let conditions_id = text_field(conditions, "conditions_id")?.to_owned();
        let temperature_k = number_field(conditions, "temperature_k")?;
        let pressure_gpa = number_field(conditions, "pressure_gpa")?;
        if temperature_k <= 0.0 || pressure_gpa < 0.0 {
            return Err(
                "network temperature must be positive and pressure non-negative".to_owned(),
            );
        }
        let extent_basis_id = text_field(&fixture, "reference_extent_basis_id")?.to_owned();
        let extent_unit = text_field(&fixture, "reference_extent_unit")?.to_owned();
        let standard_state_id = text_field(&fixture, "standard_state_id")?.to_owned();
        let balance_tolerance = number_field(&fixture, "balance_tolerance")?;
        let cycle_tolerance = number_field(&fixture, "cycle_closure_tolerance_eV")?;
        if balance_tolerance < 0.0 || cycle_tolerance < 0.0 {
            return Err("network tolerances must be non-negative".to_owned());
        }
        let species_rows = fixture
            .get("species")
            .and_then(Value::as_array)
            .ok_or("network species must be an array")?;
        let (species, species_index) = parse_species(species_rows)?;
        let reaction_rows = fixture
            .get("reactions")
            .and_then(Value::as_array)
            .ok_or("network reactions must be an array")?;
        let (reactions, reaction_index, balances) = parse_reactions(
            reaction_rows,
            &species,
            &species_index,
            &conditions_id,
            &extent_basis_id,
            &extent_unit,
            &standard_state_id,
            balance_tolerance,
        )?;
        let paths = parse_paths(
            fixture
                .get("paths")
                .and_then(Value::as_array)
                .ok_or("network paths must be an array")?,
            &reactions,
            &reaction_index,
            &species,
            &species_index,
        )?;
        let cycles = parse_cycles(
            fixture
                .get("cycles")
                .and_then(Value::as_array)
                .ok_or("network cycles must be an array")?,
            &reaction_index,
        )?;
        Ok(Self {
            id: text_field(&fixture, "network_id")?.to_owned(),
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            fixture_sha256: format!("{:x}", Sha256::digest(fixture_bytes)),
            conditions_id,
            temperature_k,
            pressure_gpa,
            atmosphere: text_field(conditions, "atmosphere")?.to_owned(),
            chemical_potential_context: text_field(conditions, "chemical_potential_context")?
                .to_owned(),
            extent_basis_id,
            extent_unit,
            standard_state_id,
            balance_tolerance,
            cycle_tolerance,
            species,
            reactions,
            paths,
            cycles,
            balances,
        })
    }

    fn path_receipt(&self, path: &Path) -> Value {
        let mut delta_g = Interval {
            lower: 0.0,
            upper: 0.0,
        };
        let mut barrier_lower: f64 = 0.0;
        let mut barrier_upper: f64 = 0.0;
        let steps: Vec<Value> = path
            .reactions
            .iter()
            .map(|index| {
                let reaction = &self.reactions[*index];
                delta_g = add_intervals(delta_g, reaction.delta_g);
                barrier_lower = barrier_lower.max(reaction.activation_barrier.lower);
                barrier_upper = barrier_upper.max(reaction.activation_barrier.upper);
                json!({
                    "reaction_id": reaction.id,
                    "role": reaction.role,
                    "delta_g_interval_eV": {
                        "lower": reaction.delta_g.lower,
                        "upper": reaction.delta_g.upper
                    },
                    "activation_barrier_interval_eV": {
                        "lower": reaction.activation_barrier.lower,
                        "upper": reaction.activation_barrier.upper
                    },
                    "balance": self.balances[*index].balanced
                })
            })
            .collect();
        json!({
            "path_id": path.id,
            "path_role": path.role,
            "input_species_ids": path.input_species.iter().map(|i| self.species[*i].id.clone()).collect::<Vec<_>>(),
            "target_species_id": self.species[path.target_species].id,
            "ordered_reactions": steps,
            "route_delta_g_interval_eV_per_reference_extent": {
                "lower": delta_g.lower,
                "upper": delta_g.upper
            },
            "maximum_activation_barrier_interval_eV": {
                "lower": barrier_lower,
                "upper": barrier_upper
            },
            "barrier_interpretation": "Maximum per-step barrier interval envelope; not an Arrhenius rate, reaction-time or synthesis-probability estimate."
        })
    }

    fn cycle_receipt(&self, cycle: &Cycle) -> Value {
        let mut total = Interval {
            lower: 0.0,
            upper: 0.0,
        };
        let mut stoichiometric_residuals = vec![0.0; self.species.len()];
        let mut terms = Vec::new();
        for term in &cycle.terms {
            let reaction = &self.reactions[term.reaction_index];
            let orientation = term.orientation as f64;
            let signed = if term.orientation == 1 {
                reaction.delta_g
            } else {
                negate_interval(reaction.delta_g)
            };
            total = add_intervals(total, signed);
            for species_term in &reaction.terms {
                stoichiometric_residuals[species_term.species_index] +=
                    orientation * species_term.coefficient;
            }
            terms.push(json!({
                "reaction_id": reaction.id,
                "orientation": term.orientation,
                "signed_delta_g_interval_eV": {
                    "lower": signed.lower,
                    "upper": signed.upper
                }
            }));
        }
        let stoichiometry_closed = stoichiometric_residuals
            .iter()
            .all(|residual| residual.abs() <= self.balance_tolerance);
        let residuals: Vec<Value> = self
            .species
            .iter()
            .enumerate()
            .filter_map(|(index, species)| {
                let residual = stoichiometric_residuals[index];
                (residual.abs() > self.balance_tolerance).then(|| {
                    json!({
                        "species_id": species.id,
                        "net_coefficient": residual
                    })
                })
            })
            .collect();
        let energy_consistent =
            total.lower <= self.cycle_tolerance && total.upper >= -self.cycle_tolerance;
        let closure_status = if !stoichiometry_closed {
            "cycle_stoichiometry_does_not_close"
        } else if energy_consistent {
            "interval_intersects_zero_closure_tolerance"
        } else {
            "interval_excludes_zero_thermodynamic_inconsistency"
        };
        json!({
            "cycle_id": cycle.id,
            "terms": terms,
            "stoichiometric_closure_status": if stoichiometry_closed {
                "net_species_coefficients_close_within_tolerance"
            } else {
                "net_species_coefficients_do_not_close"
            },
            "stoichiometric_residuals": residuals,
            "cycle_delta_g_interval_eV": {
                "lower": total.lower,
                "upper": total.upper
            },
            "zero_closure_tolerance_eV": self.cycle_tolerance,
            "closure_status": closure_status,
            "interpretation": "Stoichiometric closure is computed from the species ledger. Energy closure is interval-based and assumes the declared reaction extent; closure within uncertainty is a consistency check only, not proof that source energies are correct or that the system reaches equilibrium."
        })
    }

    fn receipt(&self) -> Value {
        let reactions: Vec<Value> = self.reactions.iter().enumerate().map(|(i, reaction)| {
            let balance = &self.balances[i];
            json!({
                "reaction_id": reaction.id,
                "role": reaction.role,
                "standard_state_id": reaction.standard_state_id,
                "source_class": reaction.source_class,
                "computed_element_residuals": balance.element_residuals.iter().map(|(element, residual)| json!({
                    "element": element,
                    "residual": residual
                })).collect::<Vec<_>>(),
                "computed_charge_residual": balance.charge_residual,
                "conservation_status": if balance.balanced { "balanced_within_tolerance" } else { "conservation_failed" },
                "delta_g_interval_eV": {
                    "lower": reaction.delta_g.lower,
                    "upper": reaction.delta_g.upper
                },
                "activation_barrier_interval_eV": {
                    "lower": reaction.activation_barrier.lower,
                    "upper": reaction.activation_barrier.upper
                }
            })
        }).collect();
        json!({
            "status": "synthetic_thermochemical_network_audit_pass",
            "contract": "SC-THERMOCHEMICAL-NETWORK-AUDIT-0.1",
            "network_id": self.id,
            "schema_sha256": self.schema_sha256,
            "fixture_sha256": self.fixture_sha256,
            "conditions": {
                "conditions_id": self.conditions_id,
                "temperature_k": self.temperature_k,
                "pressure_gpa": self.pressure_gpa,
                "atmosphere": self.atmosphere,
                "chemical_potential_context": self.chemical_potential_context
            },
            "reference_extent_basis_id": self.extent_basis_id,
            "reference_extent_unit": self.extent_unit,
            "standard_state_id": self.standard_state_id,
            "balance_tolerance": self.balance_tolerance,
            "species": self.species.iter().map(|s| json!({
                "species_id": s.id,
                "formula_as_declared": s.formula,
                "elemental_composition": s.composition.iter().map(|(e,n)| json!({"element":e,"atom_count":n})).collect::<Vec<_>>(),
                "charge": s.charge,
                "composition_interpretation": "Declared machine-readable composition used for conservation arithmetic; formula string is not independently parsed."
            })).collect::<Vec<_>>(),
            "reactions": reactions,
            "paths": self.paths.iter().map(|path| self.path_receipt(path)).collect::<Vec<_>>(),
            "thermodynamic_cycles": self.cycles.iter().map(|cycle| self.cycle_receipt(cycle)).collect::<Vec<_>>(),
            "claim_ceiling": "Synthetic stoichiometric and thermodynamic consistency audit only; no DFT calculation, material synthesis, superconducting phase or laboratory authorization is established."
        })
    }
}

fn audit(schema: &[u8], fixture: &[u8]) -> Result<Value, String> {
    let network = Network::from_bytes(schema, fixture)?;
    Ok(network.receipt())
}
fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| {
        eprintln!("could not read {}: {e}", path.display());
        process::exit(1);
    })
}
fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --fixture <fixture.json> [--out <receipt.json>]"
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
            other => {
                eprintln!("unknown argument {other}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }
    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let output = audit(&read(&schema_path), &read(&fixture_path)).unwrap_or_else(|error| {
        eprintln!("SC-THERMOCHEMICAL-NETWORK-FAIL: {error}");
        process::exit(1);
    });
    let rendered =
        serde_json::to_string_pretty(&output).expect("JSON receipt serialization cannot fail");
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

    fn inputs_for(file: &str) -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(
                root.join("data/schemas/superconductivity_thermochemical_network_v0_1.schema.json"),
            )
            .unwrap(),
            fs::read(
                root.join("tools/superconductivity-discovery-harness/fixtures")
                    .join(file),
            )
            .unwrap(),
        )
    }
    fn inputs() -> (Vec<u8>, Vec<u8>) {
        inputs_for("thermochemical_network_smoke_v0_1.json")
    }

    #[test]
    fn independently_computes_element_and_charge_balance_for_every_reaction() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        let reactions = receipt["reactions"].as_array().unwrap();
        assert_eq!(reactions.len(), 4);
        for reaction in reactions {
            assert_eq!(reaction["conservation_status"], "balanced_within_tolerance");
            assert!(
                reaction["computed_element_residuals"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|x| x["residual"].as_f64().unwrap().abs() <= 1.0e-9)
            );
            assert!(reaction["computed_charge_residual"].as_f64().unwrap().abs() <= 1.0e-9);
        }
    }

    #[test]
    fn route_delta_g_is_summed_on_the_declared_common_extent_basis() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        let target = receipt["paths"]
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["path_id"] == "target_two_step")
            .unwrap();
        assert!(
            (target["route_delta_g_interval_eV_per_reference_extent"]["lower"]
                .as_f64()
                .unwrap()
                + 0.90)
                .abs()
                < 1.0e-12
        );
        assert!(
            (target["route_delta_g_interval_eV_per_reference_extent"]["upper"]
                .as_f64()
                .unwrap()
                + 0.70)
                .abs()
                < 1.0e-12
        );
        assert!(
            (target["maximum_activation_barrier_interval_eV"]["lower"]
                .as_f64()
                .unwrap()
                - 0.9)
                .abs()
                < 1.0e-12
        );
        assert!(
            (target["maximum_activation_barrier_interval_eV"]["upper"]
                .as_f64()
                .unwrap()
                - 1.2)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn cycle_interval_closes_within_uncertainty_when_it_intersects_zero() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        let cycle = &receipt["thermodynamic_cycles"][0];
        assert_eq!(
            cycle["closure_status"],
            "interval_intersects_zero_closure_tolerance"
        );
        assert!(
            cycle["cycle_delta_g_interval_eV"]["lower"]
                .as_f64()
                .unwrap()
                < 0.0
        );
        assert!(
            cycle["cycle_delta_g_interval_eV"]["upper"]
                .as_f64()
                .unwrap()
                > 0.0
        );
    }

    #[test]
    fn deliberately_unbalanced_reaction_is_rejected_by_computed_conservation() {
        let (schema, fixture) = inputs_for("thermochemical_network_unbalanced_v0_1.json");
        let error = Network::from_bytes(&schema, &fixture).unwrap_err();
        assert!(error.contains("fails computed conservation"));
        assert!(error.contains("C residual"));
    }

    #[test]
    fn charge_imbalance_is_rejected_even_when_element_counts_close() {
        let (schema, fixture) = inputs_for("thermochemical_network_unbalanced_charge_v0_1.json");
        let error = Network::from_bytes(&schema, &fixture).unwrap_err();
        assert!(error.contains("charge residual=-1"));
    }

    #[test]
    fn thermodynamic_cycle_excluding_zero_is_reported_as_inconsistent() {
        let (schema, fixture) = inputs_for("thermochemical_network_cycle_inconsistent_v0_1.json");
        let receipt = audit(&schema, &fixture).unwrap();
        assert_eq!(
            receipt["thermodynamic_cycles"][0]["closure_status"],
            "interval_excludes_zero_thermodynamic_inconsistency"
        );
    }

    #[test]
    fn disconnected_route_is_rejected() {
        let (schema, fixture) = inputs_for("thermochemical_network_disconnected_v0_1.json");
        let error = Network::from_bytes(&schema, &fixture).unwrap_err();
        assert!(error.contains("is disconnected"));
    }

    #[test]
    fn cycle_stoichiometry_must_close_before_energy_closure_is_admitted() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cycles"][0]["terms"][2]["orientation"] = json!(-1);
        let network = Network::from_bytes(&schema, &serde_json::to_vec(&data).unwrap()).unwrap();
        let receipt = network.receipt();
        assert_eq!(
            receipt["thermodynamic_cycles"][0]["stoichiometric_closure_status"],
            "net_species_coefficients_do_not_close"
        );
        assert_eq!(
            receipt["thermodynamic_cycles"][0]["closure_status"],
            "cycle_stoichiometry_does_not_close"
        );
    }

    #[test]
    fn reaction_extent_basis_mismatch_fails_closed() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["reactions"][1]["extent_basis_id"] = json!("different_basis");
        assert!(Network::from_bytes(&schema, &serde_json::to_vec(&data).unwrap()).is_err());
    }

    #[test]
    fn standard_state_mismatch_fails_closed() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["reactions"][1]["standard_state_id"] = json!("different_standard_state");
        let error = Network::from_bytes(&schema, &serde_json::to_vec(&data).unwrap()).unwrap_err();
        assert!(error.contains("standard-state ID does not match"));
    }

    #[test]
    fn path_must_end_with_the_declared_target_species() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["paths"][0]["target_species_id"] = json!("competitor_abc2");
        assert!(Network::from_bytes(&schema, &serde_json::to_vec(&data).unwrap()).is_err());
    }
}
