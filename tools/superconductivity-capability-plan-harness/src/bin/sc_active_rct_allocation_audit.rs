use jsonschema::validator_for;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Debug)]
struct EffectState { id: String, prior: f64, ate: f64 }
#[derive(Clone, Debug)]
struct Family { id: String, weight: f64, states: Vec<EffectState> }
#[derive(Clone, Debug)]
struct World { id: String, prior: f64, family_states: Vec<usize>, ate: f64 }
#[derive(Clone, Debug)]
struct Decision { id: String, losses: Vec<f64> }
#[derive(Clone, Debug)]
struct Resource { id: String, unit: String, capacity: u64, semantics: String }
#[derive(Clone, Debug)]
struct Candidate {
    id: String,
    family: usize,
    batch: String,
    specimens: Vec<String>,
    matched: bool,
    treatment_available: bool,
    control_available: bool,
    eligible: bool,
    exclusion_reason: String,
    randomization_probability: f64,
    historical_propensity: f64,
    domain_discrepancy: f64,
    representation_weight: f64,
    consumption: Vec<u64>,
    likelihoods: Vec<[f64; 3]>,
}
#[derive(Clone, Debug)]
struct Constraints {
    max_selected: usize,
    min_selected: usize,
    min_per_family: usize,
    max_per_family: usize,
    max_per_batch: usize,
    min_randomization_probability: f64,
    max_randomization_probability: f64,
}
#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    joint_cate_epig: f64,
    focused_ate_epig: f64,
    ate_variance_reduction: f64,
    decision_evsi: f64,
    generic_predictive_entropy: f64,
}
#[derive(Clone, Debug)]
struct Plan {
    selected: Vec<usize>,
    metrics: Metrics,
    consumption: Vec<u64>,
}
#[derive(Clone, Debug)]
struct Instance {
    id: String,
    schema_sha256: String,
    fixture_sha256: String,
    outcome_labels: [String; 3],
    families: Vec<Family>,
    worlds: Vec<World>,
    ate_bin_keys: Vec<i64>,
    ate_bin_by_world: Vec<usize>,
    decisions: Vec<Decision>,
    resources: Vec<Resource>,
    constraints: Constraints,
    candidates: Vec<Candidate>,
    prior_joint_entropy: f64,
    prior_ate_entropy: f64,
    prior_ate_variance: f64,
    prior_decision_risk: f64,
}

fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}
fn number_field(value: &Value, key: &str) -> Result<f64, String> {
    let x = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric field {key}"))?;
    if !x.is_finite() { return Err(format!("{key} must be finite")); }
    Ok(x)
}
fn bool_field(value: &Value, key: &str) -> Result<bool, String> {
    value.get(key).and_then(Value::as_bool).ok_or_else(|| format!("{key} must be boolean"))
}
fn array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))
}
fn entropy(probabilities: &[f64]) -> f64 {
    -probabilities.iter().filter(|p| **p > 0.0).map(|p| p * p.ln()).sum::<f64>()
}
fn ate_bin(value: f64) -> i64 { (value * 1_000_000_000.0).round() as i64 }
fn checked_add(left: &[u64], right: &[u64]) -> Option<Vec<u64>> {
    if left.len() != right.len() { return None; }
    left.iter().zip(right.iter()).map(|(a,b)|a.checked_add(*b)).collect()
}
fn fits(used: &[u64], resources: &[Resource]) -> bool {
    used.len() == resources.len() && used.iter().zip(resources.iter()).all(|(u,r)| *u <= r.capacity)
}
fn sum_cost(selected: &[usize], candidates: &[Candidate], resource_count: usize) -> Result<Vec<u64>, String> {
    let mut total = vec![0_u64; resource_count];
    for index in selected {
        total = checked_add(&total, &candidates[*index].consumption)
            .ok_or_else(|| "resource integer overflow while summing allocation".to_owned())?;
    }
    Ok(total)
}
fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() { None } else { Some(values.iter().sum::<f64>() / values.len() as f64) }
}

impl Instance {
    fn from_bytes(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Self, String> {
        let schema: Value = serde_json::from_slice(schema_bytes).map_err(|e| format!("schema JSON parse failed: {e}"))?;
        let fixture: Value = serde_json::from_slice(fixture_bytes).map_err(|e| format!("fixture JSON parse failed: {e}"))?;
        const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
        if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
            return Err(format!("schema must declare {DIALECT}"));
        }
        let validator = validator_for(&schema).map_err(|e| format!("schema compilation failed: {e}"))?;
        if let Some(error) = validator.iter_errors(&fixture).next() {
            return Err(format!("active RCT schema validation failed: {error} at {}", error.instance_path()));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("active RCT auditor accepts synthetic fixtures only".to_owned());
        }
        if text_field(&fixture, "contract")? != "SC-ACTIVE-RCT-ALLOCATION-0.1" {
            return Err("unsupported active RCT allocation contract".to_owned());
        }
        let id = text_field(&fixture, "instance_id")?.to_owned();

        let outcome_rows = array(&fixture, "outcome_labels")?;
        if outcome_rows.len() != 3 { return Err("v0.1 requires exactly three outcome labels".to_owned()); }
        let mut outcome_labels = [String::new(), String::new(), String::new()];
        for (i, row) in outcome_rows.iter().enumerate() {
            outcome_labels[i] = row.as_str().filter(|s| !s.trim().is_empty())
                .ok_or_else(|| "outcome labels must be non-empty strings".to_owned())?.to_owned();
        }
        if outcome_labels.iter().collect::<HashSet<_>>().len() != 3 {
            return Err("outcome labels must be unique".to_owned());
        }

        let mut families = Vec::new();
        let mut family_index = HashMap::new();
        let mut family_weight_sum = 0.0;
        for row in array(&fixture, "families")? {
            let family_id = text_field(row, "family_id")?.to_owned();
            if family_index.insert(family_id.clone(), families.len()).is_some() {
                return Err(format!("duplicate material family {family_id}"));
            }
            let weight = number_field(row, "application_weight")?;
            if weight <= 0.0 || weight > 1.0 { return Err(format!("family {family_id} application weight must be in (0,1]")); }
            family_weight_sum += weight;
            let mut states = Vec::new();
            let mut state_ids = HashSet::new();
            let mut prior_sum = 0.0;
            for state in array(row, "effect_states")? {
                let state_id = text_field(state, "state_id")?.to_owned();
                if !state_ids.insert(state_id.clone()) { return Err(format!("family {family_id} duplicates state {state_id}")); }
                let prior = number_field(state, "prior_probability")?;
                let ate = number_field(state, "ate")?;
                if prior <= 0.0 || prior > 1.0 { return Err(format!("family {family_id} state prior must be in (0,1]")); }
                prior_sum += prior;
                states.push(EffectState { id: state_id, prior, ate });
            }
            if states.len() < 2 || (prior_sum - 1.0).abs() > 1.0e-9 {
                return Err(format!("family {family_id} effect-state priors must sum to one"));
            }
            families.push(Family { id: family_id, weight, states });
        }
        if families.len() < 2 || families.len() > 5 || (family_weight_sum-1.0).abs() > 1.0e-9 {
            return Err("family count must be in [2,5] and application weights must sum to one".to_owned());
        }

        let resource_rows = array(&fixture, "resources")?;
        let mut resources = Vec::new();
        let mut resource_index = HashMap::new();
        for row in resource_rows {
            let resource_id = text_field(row, "resource_id")?.to_owned();
            if resource_index.insert(resource_id.clone(), resources.len()).is_some() {
                return Err(format!("duplicate resource {resource_id}"));
            }
            let capacity = row.get("capacity").and_then(Value::as_u64)
                .filter(|n| *n > 0).ok_or_else(|| format!("resource {resource_id} capacity must be positive integer"))?;
            resources.push(Resource {
                id: resource_id,
                unit: text_field(row, "unit")?.to_owned(),
                capacity,
                semantics: text_field(row, "semantics")?.to_owned(),
            });
        }
        if resources.len() < 2 || resources.len() > 8 { return Err("resource count must be in [2,8]".to_owned()); }

        let constraints_row = fixture.get("allocation_constraints").ok_or("missing allocation_constraints")?;
        let constraints = Constraints {
            max_selected: constraints_row.get("max_selected_blocks").and_then(Value::as_u64).ok_or("max_selected_blocks must be integer")? as usize,
            min_selected: constraints_row.get("min_selected_blocks").and_then(Value::as_u64).ok_or("min_selected_blocks must be integer")? as usize,
            min_per_family: constraints_row.get("min_blocks_per_family").and_then(Value::as_u64).ok_or("min_blocks_per_family must be integer")? as usize,
            max_per_family: constraints_row.get("max_blocks_per_family").and_then(Value::as_u64).ok_or("max_blocks_per_family must be integer")? as usize,
            max_per_batch: constraints_row.get("max_blocks_per_batch").and_then(Value::as_u64).ok_or("max_blocks_per_batch must be integer")? as usize,
            min_randomization_probability: number_field(constraints_row, "min_randomization_probability")?,
            max_randomization_probability: number_field(constraints_row, "max_randomization_probability")?,
        };
        if constraints.min_selected == 0 || constraints.min_selected > constraints.max_selected
            || constraints.max_selected > 5 || constraints.min_per_family == 0
            || constraints.max_per_family < constraints.min_per_family || constraints.max_per_batch == 0
            || constraints.min_randomization_probability <= 0.0
            || constraints.max_randomization_probability >= 1.0
            || constraints.min_randomization_probability >= constraints.max_randomization_probability {
            return Err("invalid bounded allocation constraints".to_owned());
        }
        if constraints.min_per_family * families.len() > constraints.max_selected {
            return Err("family minimum representation exceeds maximum selected blocks".to_owned());
        }

        let mut worlds = Vec::new();
        let mut world_index = HashMap::new();
        let mut world_prior_sum = 0.0;
        for row in array(&fixture, "world_states")? {
            let world_id = text_field(row, "world_state_id")?.to_owned();
            if world_index.insert(world_id.clone(), worlds.len()).is_some() {
                return Err(format!("duplicate joint world state {world_id}"));
            }
            let prior = number_field(row, "prior_probability")?;
            if prior <= 0.0 || prior > 1.0 { return Err(format!("world {world_id} prior must be in (0,1]")); }
            world_prior_sum += prior;
            let mut family_states = vec![None; families.len()];
            for entry in array(row, "family_state_ids")? {
                let fid = text_field(entry, "family_id")?;
                let fi = *family_index.get(fid).ok_or_else(|| format!("world {world_id} references unknown family {fid}"))?;
                if family_states[fi].is_some() { return Err(format!("world {world_id} repeats family {fid}")); }
                let state_id = text_field(entry, "family_state_id")?;
                let si = families[fi].states.iter().position(|s| s.id == state_id)
                    .ok_or_else(|| format!("world {world_id} references unknown state {state_id} in {fid}"))?;
                family_states[fi] = Some(si);
            }
            if family_states.iter().any(Option::is_none) {
                return Err(format!("world {world_id} must specify every family exactly once"));
            }
            let family_states: Vec<usize> = family_states.into_iter().map(Option::unwrap).collect();
            let expected_ate = families.iter().enumerate()
                .map(|(fi, family)| family.weight * family.states[family_states[fi]].ate).sum::<f64>();
            let supplied_ate = number_field(row, "target_population_ate")?;
            if (expected_ate-supplied_ate).abs() > 1.0e-8 {
                return Err(format!("world {world_id} target_population_ate does not match family CATE values and application weights"));
            }
            worlds.push(World { id: world_id, prior, family_states, ate: supplied_ate });
        }
        if worlds.len() < 4 || worlds.len() > 125 || (world_prior_sum-1.0).abs() > 1.0e-9 {
            return Err("joint worlds must be in [4,125] and priors must sum to one".to_owned());
        }
        // Validate each family-state marginal. This allows correlated joint priors
        // while preventing a mismatch between the family priors and world table.
        for (fi, family) in families.iter().enumerate() {
            for (si, state) in family.states.iter().enumerate() {
                let marginal = worlds.iter().filter(|w|w.family_states[fi]==si).map(|w|w.prior).sum::<f64>();
                if (marginal-state.prior).abs() > 1.0e-8 {
                    return Err(format!("family {} state {} marginal prior does not match joint world table", family.id, state.id));
                }
            }
        }
        let mut ate_bin_keys: Vec<i64> = worlds.iter().map(|w|ate_bin(w.ate)).collect();
        ate_bin_keys.sort_unstable();
        ate_bin_keys.dedup();
        let ate_bin_by_world: Vec<usize> = worlds.iter().map(|w| {
            ate_bin_keys.binary_search(&ate_bin(w.ate)).expect("ATE bin was collected")
        }).collect();

        let decision_rows = array(&fixture, "decisions")?;
        let mut decisions = Vec::new();
        let mut decision_ids = HashSet::new();
        for row in decision_rows {
            let decision_id = text_field(row, "decision_id")?.to_owned();
            if !decision_ids.insert(decision_id.clone()) { return Err(format!("duplicate decision {decision_id}")); }
            let mut losses = vec![None; worlds.len()];
            for loss_row in array(row, "losses")? {
                let world_id = text_field(loss_row, "world_state_id")?;
                let wi = *world_index.get(world_id).ok_or_else(|| format!("decision {decision_id} references unknown world {world_id}"))?;
                if losses[wi].is_some() { return Err(format!("decision {decision_id} repeats world {world_id}")); }
                let loss = number_field(loss_row, "loss")?;
                if loss < 0.0 { return Err("decision losses must be non-negative".to_owned()); }
                losses[wi] = Some(loss);
            }
            if losses.iter().any(Option::is_none) { return Err(format!("decision {decision_id} must define a loss for every world")); }
            decisions.push(Decision { id: decision_id, losses: losses.into_iter().map(Option::unwrap).collect() });
        }
        if decisions.len() < 2 { return Err("at least two downstream decisions are required".to_owned()); }
        decisions.sort_by(|a,b|a.id.cmp(&b.id));

        let candidate_rows = array(&fixture, "candidate_blocks")?;
        if candidate_rows.len() < 2 || candidate_rows.len() > 12 {
            return Err("candidate block count must be in [2,12] for bounded exact enumeration".to_owned());
        }
        let mut candidates = Vec::new();
        let mut candidate_ids = HashSet::new();
        let mut specimen_ids = HashSet::new();
        for row in candidate_rows {
            let block_id = text_field(row, "block_id")?.to_owned();
            if !candidate_ids.insert(block_id.clone()) { return Err(format!("duplicate candidate block {block_id}")); }
            let family_id = text_field(row, "family_id")?;
            let family = *family_index.get(family_id).ok_or_else(|| format!("candidate {block_id} references unknown family {family_id}"))?;
            let batch = text_field(row, "batch_id")?.to_owned();
            let specimens = array(row, "specimen_ids")?.iter().map(|x| {
                x.as_str().filter(|s|!s.trim().is_empty()).map(str::to_owned)
                    .ok_or_else(||format!("candidate {block_id} specimen ids must be non-empty strings"))
            }).collect::<Result<Vec<_>,_>>()?;
            if specimens.is_empty() || specimens.len()>2 { return Err(format!("candidate {block_id} must identify one or two specimens")); }
            for specimen in &specimens {
                if !specimen_ids.insert(specimen.clone()) { return Err(format!("specimen {specimen} appears in more than one candidate block")); }
            }
            let matched = bool_field(row,"matched_pair_qualified")?;
            let treatment_available = bool_field(row,"treatment_available")?;
            let control_available = bool_field(row,"control_available")?;
            let eligible = bool_field(row,"eligible")?;
            let exclusion_reason = row.get("exclusion_reason").and_then(Value::as_str)
                .ok_or("exclusion_reason must be text")?.to_owned();
            if !eligible && exclusion_reason.trim().is_empty() { return Err(format!("ineligible candidate {block_id} needs an exclusion reason")); }
            if eligible && (!matched || !treatment_available || !control_available || specimens.len()!=2) {
                return Err(format!("eligible candidate {block_id} lacks a qualified matched treatment/control pair"));
            }
            let randomization_probability = number_field(row,"randomization_probability")?;
            let historical_propensity = number_field(row,"historical_treatment_propensity")?;
            let domain_discrepancy = number_field(row,"domain_discrepancy")?;
            let representation_weight = number_field(row,"family_representation_weight")?;
            if randomization_probability<=0.0 || randomization_probability>=1.0
                || historical_propensity<=0.0 || historical_propensity>=1.0
                || !(0.0..=1.0).contains(&domain_discrepancy)
                || (representation_weight-families[family].weight).abs()>1.0e-9 {
                return Err(format!("candidate {block_id} has invalid propensity/discrepancy/representation metadata"));
            }
            let mut consumption = vec![0_u64; resources.len()];
            for item in array(row,"resource_consumption")? {
                let resource_id = text_field(item,"resource_id")?;
                let ri = resources.iter().position(|r|r.id==resource_id)
                    .ok_or_else(||format!("candidate {block_id} references unknown resource {resource_id}"))?;
                if consumption[ri]!=0 { return Err(format!("candidate {block_id} repeats resource {resource_id}")); }
                if text_field(item,"unit")? != resources[ri].unit { return Err(format!("candidate {block_id} unit mismatch for {resource_id}")); }
                consumption[ri] = item.get("quantity").and_then(Value::as_u64).filter(|q|*q>0)
                    .ok_or_else(||format!("candidate {block_id} quantity must be a positive integer"))?;
            }
            if eligible && consumption.iter().all(|x|*x==0) { return Err(format!("eligible candidate {block_id} has no resource cost")); }
            let mut likelihoods = vec![None; families[family].states.len()];
            for likelihood in array(row,"observation_likelihoods")? {
                let state_id = text_field(likelihood,"family_state_id")?;
                let si = families[family].states.iter().position(|s|s.id==state_id)
                    .ok_or_else(||format!("candidate {block_id} likelihood references unknown family state {state_id}"))?;
                if likelihoods[si].is_some() { return Err(format!("candidate {block_id} duplicates likelihood for {state_id}")); }
                let probs = array(likelihood,"outcome_probabilities")?;
                if probs.len()!=3 { return Err(format!("candidate {block_id} likelihood must have three outcome probabilities")); }
                let values = [
                    probs[0].as_f64().ok_or("likelihood must be numeric")?,
                    probs[1].as_f64().ok_or("likelihood must be numeric")?,
                    probs[2].as_f64().ok_or("likelihood must be numeric")?,
                ];
                if values.iter().any(|p|!p.is_finite()||*p<0.0||*p>1.0)
                    || (values.iter().sum::<f64>()-1.0).abs()>1.0e-9 {
                    return Err(format!("candidate {block_id} has invalid outcome likelihoods"));
                }
                likelihoods[si]=Some(values);
            }
            if likelihoods.iter().any(Option::is_none) { return Err(format!("candidate {block_id} needs one likelihood row per family effect state")); }
            candidates.push(Candidate {
                id:block_id,family,batch,specimens,matched,treatment_available,control_available,eligible,
                exclusion_reason,randomization_probability,historical_propensity,domain_discrepancy,
                representation_weight,consumption,likelihoods:likelihoods.into_iter().map(Option::unwrap).collect(),
            });
        }
        candidates.sort_by(|a,b|a.id.cmp(&b.id));

        let held_out_world_state_id = text_field(&fixture, "held_out_world_state_id")?;
        if !world_index.contains_key(held_out_world_state_id) {
            return Err(format!("held_out_world_state_id {held_out_world_state_id} is not in the world-state registry"));
        }
        // The held-out identifier is validated for benchmark integrity only.
        // It is intentionally not copied into the instance or used in acquisition.

        let prior: Vec<f64> = worlds.iter().map(|w|w.prior).collect();
        let prior_joint_entropy=entropy(&prior);
        let mut prior_ate_probs=vec![0.0;ate_bin_keys.len()];
        for (wi,world) in worlds.iter().enumerate() { prior_ate_probs[ate_bin_by_world[wi]]+=world.prior; }
        let prior_ate_entropy=entropy(&prior_ate_probs);
        let prior_ate_mean=worlds.iter().map(|w|w.prior*w.ate).sum::<f64>();
        let prior_ate_variance=worlds.iter().map(|w|w.prior*(w.ate-prior_ate_mean).powi(2)).sum::<f64>();
        let prior_decision_risk=decisions.iter().map(|d|worlds.iter().zip(d.losses.iter()).map(|(w,l)|w.prior*l).sum::<f64>())
            .fold(f64::INFINITY,f64::min);
        let instance=Self {
            id,
            schema_sha256:format!("{:x}",Sha256::digest(schema_bytes)),
            fixture_sha256:format!("{:x}",Sha256::digest(fixture_bytes)),
            outcome_labels,families,worlds,ate_bin_keys,ate_bin_by_world,decisions,resources,constraints,
            candidates,prior_joint_entropy,prior_ate_entropy,prior_ate_variance,prior_decision_risk,
        };
        instance.validate_global_constraints()?;
        Ok(instance)
    }

    fn validate_global_constraints(&self) -> Result<(),String> {
        for candidate in &self.candidates {
            if candidate.eligible && (candidate.randomization_probability<self.constraints.min_randomization_probability
                || candidate.randomization_probability>self.constraints.max_randomization_probability) {
                return Err(format!("eligible candidate {} violates randomized assignment positivity gate",candidate.id));
            }
        }
        Ok(())
    }

    fn resource_use(&self, selected:&[usize]) -> Result<Vec<u64>,String> {
        sum_cost(selected,&self.candidates,self.resources.len())
    }

    fn prefix_feasible(&self, selected:&[usize]) -> bool {
        if selected.len()>self.constraints.max_selected { return false; }
        let mut family_counts=vec![0usize;self.families.len()];
        let mut batch_counts:HashMap<&str,usize>=HashMap::new();
        let mut used=vec![0u64;self.resources.len()];
        let mut seen=HashSet::new();
        for index in selected {
            if *index>=self.candidates.len() || !seen.insert(*index) { return false; }
            let c=&self.candidates[*index];
            if !c.eligible || !c.matched || !c.treatment_available || !c.control_available
                || c.randomization_probability<self.constraints.min_randomization_probability
                || c.randomization_probability>self.constraints.max_randomization_probability {
                return false;
            }
            family_counts[c.family]+=1;
            *batch_counts.entry(&c.batch).or_default()+=1;
            for (slot,quantity) in used.iter_mut().zip(c.consumption.iter()) {
                let Some(next)=slot.checked_add(*quantity) else { return false; };
                *slot=next;
            }
        }
        if family_counts.iter().any(|n|*n>self.constraints.max_per_family)
            || batch_counts.values().any(|n|*n>self.constraints.max_per_batch)
            || !fits(&used,&self.resources) { return false; }
        let missing:usize=family_counts.iter().map(|n|self.constraints.min_per_family.saturating_sub(*n)).sum();
        if missing>self.constraints.max_selected-selected.len() { return false; }
        for fi in 0..self.families.len() {
            let remaining=self.candidates.iter().enumerate().filter(|(i,c)|!seen.contains(i)&&c.eligible&&c.family==fi
                && c.randomization_probability>=self.constraints.min_randomization_probability
                && c.randomization_probability<=self.constraints.max_randomization_probability).count();
            if family_counts[fi]+remaining<self.constraints.min_per_family { return false; }
        }
        true
    }

    fn plan_feasible(&self, selected:&[usize]) -> bool {
        if selected.len()<self.constraints.min_selected || selected.len()>self.constraints.max_selected { return false; }
        if !self.prefix_feasible(selected) { return false; }
        let mut counts=vec![0usize;self.families.len()];
        for index in selected { counts[self.candidates[*index].family]+=1; }
        counts.iter().all(|n|*n>=self.constraints.min_per_family)
    }

    fn terminal_risk(&self, posterior:&[f64]) -> f64 {
        self.decisions.iter().map(|d|posterior.iter().zip(d.losses.iter()).map(|(p,l)|p*l).sum::<f64>())
            .fold(f64::INFINITY,f64::min)
    }

    fn candidate_predictive_entropy(&self, candidate_index:usize) -> f64 {
        let candidate=&self.candidates[candidate_index];
        let mut outcome_probs=[0.0;3];
        for world in &self.worlds {
            let state_index=world.family_states[candidate.family];
            for y in 0..3 {
                outcome_probs[y]+=world.prior*candidate.likelihoods[state_index][y];
            }
        }
        entropy(&outcome_probs)
    }

    fn metrics(&self, selected:&[usize]) -> Metrics {
        let outcome_count=3usize.pow(selected.len() as u32);
        let prior:Vec<f64>=self.worlds.iter().map(|w|w.prior).collect();
        let mut expected_joint_entropy=0.0;
        let mut expected_ate_entropy=0.0;
        let mut expected_ate_variance=0.0;
        let mut expected_decision_risk=0.0;
        for code in 0..outcome_count {
            let mut digits=code;
            let mut unnormalized=prior.clone();
            for candidate_index in selected {
                let outcome=digits%3;
                digits/=3;
                let candidate=&self.candidates[*candidate_index];
                for (wi,world) in self.worlds.iter().enumerate() {
                    unnormalized[wi]*=candidate.likelihoods[world.family_states[candidate.family]][outcome];
                }
            }
            let predictive_probability: f64=unnormalized.iter().sum();
            if predictive_probability<=0.0 { continue; }
            let posterior:Vec<f64>=unnormalized.iter().map(|mass|mass/predictive_probability).collect();
            expected_joint_entropy+=predictive_probability*entropy(&posterior);
            let mut ate_posterior=vec![0.0;self.ate_bin_keys.len()];
            for (wi,p) in posterior.iter().enumerate() { ate_posterior[self.ate_bin_by_world[wi]]+=p; }
            expected_ate_entropy+=predictive_probability*entropy(&ate_posterior);
            let posterior_mean=self.worlds.iter().zip(posterior.iter()).map(|(w,p)|w.ate*p).sum::<f64>();
            let posterior_variance=self.worlds.iter().zip(posterior.iter()).map(|(w,p)|p*(w.ate-posterior_mean).powi(2)).sum::<f64>();
            expected_ate_variance+=predictive_probability*posterior_variance;
            expected_decision_risk+=predictive_probability*self.terminal_risk(&posterior);
        }
        let generic_predictive_entropy=selected.iter().map(|i|self.candidate_predictive_entropy(*i)).sum::<f64>();
        Metrics {
            joint_cate_epig:(self.prior_joint_entropy-expected_joint_entropy).max(0.0),
            focused_ate_epig:(self.prior_ate_entropy-expected_ate_entropy).max(0.0),
            ate_variance_reduction:(self.prior_ate_variance-expected_ate_variance).max(0.0),
            decision_evsi:(self.prior_decision_risk-expected_decision_risk).max(0.0),
            generic_predictive_entropy,
        }
    }

    fn enumerate_feasible_plans(&self) -> Vec<Plan> {
        let candidate_count=self.candidates.len();
        let subset_count=1u64<<candidate_count;
        let mut plans=Vec::new();
        for mask in 0..subset_count {
            let selected:Vec<usize>=(0..candidate_count).filter(|i|(mask&(1u64<<i))!=0).collect();
            if !self.plan_feasible(&selected) { continue; }
            let Ok(consumption)=self.resource_use(&selected) else { continue; };
            plans.push(Plan { metrics:self.metrics(&selected),selected,consumption });
        }
        plans
    }

    fn metric_value(metrics:Metrics,objective:&str)->f64 {
        match objective {
            "joint_cate_epig"=>metrics.joint_cate_epig,
            "focused_ate_epig"=>metrics.focused_ate_epig,
            "ate_variance_reduction"=>metrics.ate_variance_reduction,
            "decision_evsi"=>metrics.decision_evsi,
            "generic_predictive_entropy"=>metrics.generic_predictive_entropy,
            _=>f64::NEG_INFINITY,
        }
    }

    fn plan_signature(&self,selected:&[usize])->String {
        selected.iter().map(|i|self.candidates[*i].id.as_str()).collect::<Vec<_>>().join("|")
    }

    fn better_plan(&self,candidate:&Plan,incumbent:&Plan,objective:&str)->bool {
        let a=Self::metric_value(candidate.metrics,objective);
        let b=Self::metric_value(incumbent.metrics,objective);
        if a>b+1.0e-12 { return true; }
        if (a-b).abs()>1.0e-12 { return false; }
        let cost_a=candidate.consumption.first().copied().unwrap_or(0);
        let cost_b=incumbent.consumption.first().copied().unwrap_or(0);
        if cost_a!=cost_b { return cost_a<cost_b; }
        self.plan_signature(&candidate.selected)<self.plan_signature(&incumbent.selected)
    }

    fn exact_oracle(&self,plans:&[Plan],objective:&str)->Option<Plan> {
        let mut best:Option<Plan>=None;
        for plan in plans {
            if best.as_ref().is_none_or(|current|self.better_plan(plan,current,objective)) {
                best=Some(plan.clone());
            }
        }
        best
    }

    fn overlap_deficit(candidate:&Candidate)->f64 {
        1.0-2.0*candidate.historical_propensity.min(1.0-candidate.historical_propensity)
    }

    fn greedy(&self,objective:&str)->Option<Plan> {
        let mut selected=Vec::new();
        loop {
            if selected.len()>=self.constraints.max_selected { break; }
            let current_metrics=self.metrics(&selected);
            let current_score=if objective=="budgeted_active_surrogate" {
                current_metrics.focused_ate_epig
                    + selected.iter().map(|i|0.20*Self::overlap_deficit(&self.candidates[*i])+0.20*self.candidates[*i].domain_discrepancy).sum::<f64>()
            } else { Self::metric_value(current_metrics,objective) };
            let mut best:Option<(usize,f64,f64,String)>=None;
            for index in 0..self.candidates.len() {
                if selected.contains(&index) { continue; }
                let mut next=selected.clone(); next.push(index);
                if !self.prefix_feasible(&next) { continue; }
                let metrics=self.metrics(&next);
                let candidate_score=if objective=="budgeted_active_surrogate" {
                    metrics.focused_ate_epig
                        + next.iter().map(|i|0.20*Self::overlap_deficit(&self.candidates[*i])+0.20*self.candidates[*i].domain_discrepancy).sum::<f64>()
                } else { Self::metric_value(metrics,objective) };
                let gain=(candidate_score-current_score).max(0.0);
                let cost=self.candidates[index].consumption.first().copied().unwrap_or(1).max(1) as f64;
                let ratio=gain/cost;
                let signature=self.candidates[index].id.clone();
                let replace=best.as_ref().is_none_or(|(_,best_ratio,best_gain,best_id)| {
                    ratio>*best_ratio+1.0e-15
                    || ((ratio-*best_ratio).abs()<=1.0e-15 && (gain>*best_gain+1.0e-12
                        || ((gain-*best_gain).abs()<=1.0e-12 && signature<*best_id)))
                });
                if replace { best=Some((index,ratio,gain,signature)); }
            }
            let Some((index,_,_,_))=best else { break; };
            selected.push(index);
        }
        if !self.plan_feasible(&selected) { return None; }
        let consumption=self.resource_use(&selected).ok()?;
        Some(Plan { metrics:self.metrics(&selected),selected,consumption })
    }

    fn random_baseline(&self,plans:&[Plan],seeds:usize)->Value {
        if plans.is_empty() { return json!({"status":"no_feasible_plans"}); }
        let mut values:HashMap<&str,Vec<f64>>=HashMap::new();
        let metrics_keys=["joint_cate_epig","focused_ate_epig","ate_variance_reduction","decision_evsi","generic_predictive_entropy"];
        let mut state=1u64;
        for _ in 0..seeds {
            let draw=splitmix64(&mut state) as usize;
            let plan=&plans[draw%plans.len()];
            for key in metrics_keys {
                values.entry(key).or_default().push(Self::metric_value(plan.metrics,key));
            }
        }
        let summary:Vec<Value>=metrics_keys.iter().map(|key|{
            let xs=&values[key];
            let avg=xs.iter().sum::<f64>()/xs.len() as f64;
            let variance=xs.iter().map(|x|(x-avg).powi(2)).sum::<f64>()/xs.len() as f64;
            json!({"objective":key,"mean":avg,"population_stddev":variance.sqrt()})
        }).collect();
        json!({"status":"deterministic_random_feasible_baseline","seed_count":seeds,"seed_generator":"SplitMix64","metrics":summary})
    }

    fn plan_receipt(&self,plan:&Plan)->Value {
        let selected:Vec<Value>=plan.selected.iter().map(|i|{
            let c=&self.candidates[*i];
            json!({
                "block_id":c.id,
                "family_id":self.families[c.family].id,
                "batch_id":c.batch,
                "specimen_ids":c.specimens,
                "randomization_design":{
                    "type":"within_pair_one_treated_one_control",
                    "treated_probability":c.randomization_probability,
                    "control_probability":1.0-c.randomization_probability,
                    "status":"design_only_not_executed"
                },
                "historical_treatment_propensity":c.historical_propensity,
                "historical_overlap_deficit":Self::overlap_deficit(c),
                "domain_discrepancy":c.domain_discrepancy,
                "resource_consumption":self.resources.iter().enumerate().filter_map(|(ri,r)|{
                    let q=c.consumption[ri];
                    (q>0).then(||json!({"resource_id":r.id,"unit":r.unit,"quantity":q}))
                }).collect::<Vec<_>>()
            })
        }).collect();
        let resource_usage:Vec<Value>=self.resources.iter().enumerate().map(|(i,r)|json!({
            "resource_id":r.id,"unit":r.unit,"capacity":r.capacity,"used":plan.consumption[i],
            "remaining":r.capacity-plan.consumption[i],"binding":r.capacity==plan.consumption[i],"semantics":r.semantics
        })).collect();
        json!({
            "selected_blocks":selected,
            "selected_block_count":plan.selected.len(),
            "metrics":{
                "joint_cate_epig_nats":plan.metrics.joint_cate_epig,
                "focused_ate_epig_nats":plan.metrics.focused_ate_epig,
                "expected_ate_variance_reduction":plan.metrics.ate_variance_reduction,
                "decision_evsi_loss_units":plan.metrics.decision_evsi,
                "generic_predictive_entropy_sum_nats":plan.metrics.generic_predictive_entropy
            },
            "resource_usage":resource_usage
        })
    }

    fn receipt(&self)->Result<Value,String>{
        let plans=self.enumerate_feasible_plans();
        if plans.is_empty(){return Err("no feasible RCT allocation satisfies family, batch, positivity and vector-resource constraints".to_owned());}
        let objectives=[
            ("joint_cate_epig","joint_family_cate_epig_oracle"),
            ("focused_ate_epig","focused_ate_epig_oracle"),
            ("ate_variance_reduction","ate_precision_oracle"),
            ("decision_evsi","decision_value_oracle"),
            ("generic_predictive_entropy","generic_predictive_entropy_oracle"),
        ];
        let oracles:Vec<Value>=objectives.iter().filter_map(|(objective,name)|{
            self.exact_oracle(&plans,objective).map(|plan|json!({
                "oracle":name,
                "objective":objective,
                "plan":self.plan_receipt(&plan)
            }))
        }).collect();
        let greedy_objectives=[
            ("joint_cate_epig","greedy_joint_cate_epig"),
            ("focused_ate_epig","greedy_focused_ate_epig"),
            ("ate_variance_reduction","greedy_ate_precision"),
            ("decision_evsi","greedy_decision_value"),
            ("generic_predictive_entropy","greedy_generic_predictive_entropy"),
            ("budgeted_active_surrogate","greedy_budgeted_active_overlap_discrepancy"),
        ];
        let greedy:Vec<Value>=greedy_objectives.iter().map(|(objective,name)|{
            json!({
                "baseline":name,
                "objective":objective,
                "plan":self.greedy(objective).map(|p|self.plan_receipt(&p))
            })
        }).collect();
        let excluded:Vec<Value>=self.candidates.iter().filter(|c|!c.eligible).map(|c|json!({
            "block_id":c.id,"family_id":self.families[c.family].id,
            "matched_pair_qualified":c.matched,
            "treatment_available":c.treatment_available,
            "control_available":c.control_available,
            "exclusion_reason":c.exclusion_reason
        })).collect();
        let family_coverage:Vec<Value>=self.families.iter().map(|f|{
            let eligible=self.candidates.iter().filter(|c|c.eligible&&self.families[c.family].id==f.id).count();
            json!({"family_id":f.id,"application_weight":f.weight,"eligible_candidate_blocks":eligible,"required_minimum":self.constraints.min_per_family,"maximum":self.constraints.max_per_family})
        }).collect();
        Ok(json!({
            "status":"synthetic_active_rct_allocation_pass",
            "contract":"SC-ACTIVE-RCT-ALLOCATION-AUDIT-0.1",
            "instance_id":self.id,
            "schema_sha256":self.schema_sha256,
            "fixture_sha256":self.fixture_sha256,
            "family_count":self.families.len(),
            "joint_world_state_count":self.worlds.len(),
            "candidate_block_count":self.candidates.len(),
            "eligible_candidate_block_count":self.candidates.iter().filter(|c|c.eligible).count(),
            "feasible_allocation_count":plans.len(),
            "prior":{
                "joint_family_cate_entropy_nats":self.prior_joint_entropy,
                "focused_application_ate_entropy_nats":self.prior_ate_entropy,
                "application_weighted_ate_variance":self.prior_ate_variance,
                "minimum_expected_decision_loss":self.prior_decision_risk
            },
            "family_coverage_constraints":family_coverage,
            "excluded_candidates":excluded,
            "exact_oracles":oracles,
            "greedy_baselines":greedy,
            "random_stratified_baseline":self.random_baseline(&plans,64),
            "claim_ceiling":"Synthetic active randomized-specimen allocation benchmark only. The held-out world state is not used by acquisition or planning. Allocation outputs are design-only and do not execute or authorize any real specimen intervention."
        }))
    }
}

fn splitmix64(state:&mut u64)->u64{
    *state=state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z=*state;
    z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
    z^(z>>31)
}

fn audit(schema:&[u8],fixture:&[u8])->Result<Value,String>{
    let instance=Instance::from_bytes(schema,fixture)?;
    instance.receipt()
}
fn read(path:&PathBuf)->Vec<u8>{
    fs::read(path).unwrap_or_else(|e|{eprintln!("could not read {}: {e}",path.display());process::exit(1);})
}
fn usage(program:&str){
    eprintln!("Usage: {program} --schema <schema.json> --fixture <fixture.json> [--out <receipt.json>]");
}
fn main(){
    let args:Vec<String>=env::args().collect();
    let mut schema_path=None;let mut fixture_path=None;let mut output_path=None;let mut i=1;
    while i<args.len(){
        match args[i].as_str(){
            "--schema"|"--fixture"|"--out"=>{
                if i+1>=args.len(){usage(&args[0]);process::exit(2);}
                let path=PathBuf::from(&args[i+1]);
                match args[i].as_str(){"--schema"=>schema_path=Some(path),"--fixture"=>fixture_path=Some(path),"--out"=>output_path=Some(path),_=>unreachable!()}
                i+=2;
            }
            "--help"|"-h"=>{usage(&args[0]);return;}
            other=>{eprintln!("unknown argument {other}");usage(&args[0]);process::exit(2);}
        }
    }
    let(Some(schema_path),Some(fixture_path))=(schema_path,fixture_path)else{usage(&args[0]);process::exit(2);};
    let receipt=audit(&read(&schema_path),&read(&fixture_path)).unwrap_or_else(|e|{
        eprintln!("SC-ACTIVE-RCT-ALLOCATION-FAIL: {e}");process::exit(1);
    });
    let rendered=serde_json::to_string_pretty(&receipt).expect("JSON receipt serialization cannot fail");
    if let Some(path)=output_path{
        if let Err(e)=fs::write(&path,format!("{rendered}\n")){eprintln!("could not write {}: {e}",path.display());process::exit(1);}
    }else{println!("{rendered}");}
}

#[cfg(test)]
mod tests{
    use super::*;
    use std::path::PathBuf;
    fn inputs_for(filename:&str)->(Vec<u8>,Vec<u8>){
        let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/superconductivity_active_rct_allocation_v0_1.schema.json")).unwrap(),
            fs::read(root.join("data/research").join(filename)).unwrap()
        )
    }
    fn inputs()->(Vec<u8>,Vec<u8>){inputs_for("superconductivity_active_rct_allocation_fixture_v0_1.json")}

    #[test]
    fn exact_oracles_obey_family_batch_positivity_and_vector_resource_constraints(){
        let(schema,fixture)=inputs();
        let instance=Instance::from_bytes(&schema,&fixture).unwrap();
        let plans=instance.enumerate_feasible_plans();
        assert_eq!(plans.len(),87);
        for plan in plans{
            assert!(instance.plan_feasible(&plan.selected));
            assert!(fits(&plan.consumption,&instance.resources));
            let mut family_counts=vec![0usize;instance.families.len()];
            for index in &plan.selected{
                let candidate=&instance.candidates[*index];
                assert!(candidate.randomization_probability>=instance.constraints.min_randomization_probability);
                assert!(candidate.randomization_probability<=instance.constraints.max_randomization_probability);
                family_counts[candidate.family]+=1;
            }
            assert!(family_counts.iter().all(|n|*n>=1&&*n<=2));
        }
    }

    #[test]
    fn decision_evsi_and_ate_precision_can_select_a_different_plan_than_cate_epig(){
        let(schema,fixture)=inputs();
        let receipt=audit(&schema,&fixture).unwrap();
        let oracles=receipt["exact_oracles"].as_array().unwrap();
        let find=|name:&str|oracles.iter().find(|x|x["oracle"]==name).unwrap();
        let cate=find("joint_family_cate_epig_oracle");
        let variance=find("ate_precision_oracle");
        let decision=find("decision_value_oracle");
        let cate_ids:HashSet<String>=cate["plan"]["selected_blocks"].as_array().unwrap().iter().map(|x|x["block_id"].as_str().unwrap().to_owned()).collect();
        let variance_ids:HashSet<String>=variance["plan"]["selected_blocks"].as_array().unwrap().iter().map(|x|x["block_id"].as_str().unwrap().to_owned()).collect();
        let decision_ids:HashSet<String>=decision["plan"]["selected_blocks"].as_array().unwrap().iter().map(|x|x["block_id"].as_str().unwrap().to_owned()).collect();
        assert_ne!(cate_ids,variance_ids);
        assert_ne!(cate_ids,decision_ids);
        assert!(decision["plan"]["metrics"]["decision_evsi_loss_units"].as_f64().unwrap()
            >cate["plan"]["metrics"]["decision_evsi_loss_units"].as_f64().unwrap());
    }

    #[test]
    fn no_control_candidate_is_excluded_and_never_selected(){
        let(schema,fixture)=inputs();
        let receipt=audit(&schema,&fixture).unwrap();
        let excluded=&receipt["excluded_candidates"];
        assert_eq!(excluded.as_array().unwrap().len(),1);
        assert_eq!(excluded[0]["block_id"],"family_beta_unmatched_control");
        for oracle in receipt["exact_oracles"].as_array().unwrap(){
            for selected in oracle["plan"]["selected_blocks"].as_array().unwrap(){
                assert_ne!(selected["block_id"],"family_beta_unmatched_control");
                assert_eq!(selected["randomization_design"]["status"],"design_only_not_executed");
            }
        }
    }

    #[test]
    fn changing_held_out_truth_does_not_change_any_acquisition_plan(){
        let(schema,fixture)=inputs();
        let before=audit(&schema,&fixture).unwrap();
        let mut value:Value=serde_json::from_slice(&fixture).unwrap();
        value["held_out_world_state_id"]=json!("world_alpha_high_beta_low_gamma_high");
        let after=audit(&schema,&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(before["exact_oracles"],after["exact_oracles"]);
        assert_eq!(before["greedy_baselines"],after["greedy_baselines"]);
    }

    #[test]
    fn random_stratified_baseline_is_reproducible(){
        let(schema,fixture)=inputs();
        let a=audit(&schema,&fixture).unwrap();
        let b=audit(&schema,&fixture).unwrap();
        assert_eq!(a["random_stratified_baseline"],b["random_stratified_baseline"]);
        assert_eq!(a["random_stratified_baseline"]["seed_count"],64);
    }

    #[test]
    fn held_out_world_must_exist_but_is_not_used_in_planning(){
        let(schema,fixture)=inputs();
        let mut value:Value=serde_json::from_slice(&fixture).unwrap();
        value["held_out_world_state_id"]=json!("not_a_world");
        assert!(Instance::from_bytes(&schema,&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
