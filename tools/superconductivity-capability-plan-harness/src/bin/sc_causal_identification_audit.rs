use jsonschema::validator_for;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Debug)]
struct Sample {
    id: String,
    batch: String,
    family: String,
    baseline_pairing: f64,
    baseline_coherence: f64,
}
#[derive(Clone, Debug)]
struct Record {
    sample_id: String,
    batch: String,
    family: String,
    treatment: bool,
    propensity: f64,
    delivered_dose: f64,
    outcome_observed: bool,
    outcome: f64,
}
#[derive(Clone, Debug)]
struct PotentialOutcome {
    sample_id: String,
    control: f64,
    treated: f64,
}
#[derive(Clone, Debug)]
struct Cohort {
    id: String,
    assignment_mechanism: String,
    records: Vec<Record>,
}
#[derive(Clone, Debug)]
struct Estimate {
    value: Option<f64>,
    n_treated: usize,
    n_control: usize,
    n_observed: usize,
    reason: Option<String>,
}
#[derive(Clone, Debug)]
struct RandomizationResult {
    observed_statistic: f64,
    p_value_two_sided: f64,
    assignment_count: u64,
    treated_per_block: Vec<(String, usize, usize)>,
}
#[derive(Clone, Debug)]
struct Instance {
    id: String,
    schema_sha256: String,
    fixture_sha256: String,
    target_outcome: String,
    treatment_receipt_semantics: String,
    samples: Vec<Sample>,
    cohorts: Vec<Cohort>,
    truth: Vec<PotentialOutcome>,
}

fn field_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}
fn field_number(value: &Value, key: &str) -> Result<f64, String> {
    let n = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric field {key}"))?;
    if !n.is_finite() { return Err(format!("field {key} must be finite")); }
    Ok(n)
}
fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() { None } else { Some(values.iter().sum::<f64>() / values.len() as f64) }
}

impl Instance {
    fn from_bytes(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Self, String> {
        let schema: Value = serde_json::from_slice(schema_bytes)
            .map_err(|e| format!("schema JSON parse failed: {e}"))?;
        let fixture: Value = serde_json::from_slice(fixture_bytes)
            .map_err(|e| format!("fixture JSON parse failed: {e}"))?;
        const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
        if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
            return Err(format!("schema must declare {DIALECT}"));
        }
        let validator = validator_for(&schema).map_err(|e| format!("schema compilation failed: {e}"))?;
        if let Some(error) = validator.iter_errors(&fixture).next() {
            return Err(format!("causal identification schema validation failed: {error} at {}", error.instance_path()));
        }
        if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err("causal identification auditor accepts synthetic fixtures only".to_owned());
        }
        if field_text(&fixture, "contract")? != "SC-CAUSAL-IDENTIFICATION-0.1" {
            return Err("unsupported causal identification contract".to_owned());
        }
        let id = field_text(&fixture, "instance_id")?.to_owned();
        let target_outcome = field_text(&fixture, "target_outcome")?.to_owned();
        let treatment_receipt_semantics = field_text(&fixture, "treatment_receipt_semantics")?.to_owned();

        let sample_rows = fixture.get("samples").and_then(Value::as_array)
            .ok_or_else(|| "samples must be an array".to_owned())?;
        let mut samples = Vec::with_capacity(sample_rows.len());
        let mut sample_index = HashMap::new();
        for row in sample_rows {
            let sample_id = field_text(row, "sample_id")?.to_owned();
            if sample_index.insert(sample_id.clone(), samples.len()).is_some() {
                return Err(format!("duplicate sample_id {sample_id}"));
            }
            let baseline_pairing = field_number(row, "baseline_pairing_proxy")?;
            let baseline_coherence = field_number(row, "baseline_coherence_proxy")?;
            if !(0.0..=1.0).contains(&baseline_pairing) || !(0.0..=1.0).contains(&baseline_coherence) {
                return Err(format!("sample {sample_id} baseline proxies must be in [0,1]"));
            }
            samples.push(Sample {
                id: sample_id,
                batch: field_text(row, "batch_id")?.to_owned(),
                family: field_text(row, "material_family")?.to_owned(),
                baseline_pairing,
                baseline_coherence,
            });
        }
        if samples.len() < 4 { return Err("at least four experimental units are required".to_owned()); }

        let cohort_rows = fixture.get("cohorts").and_then(Value::as_array)
            .ok_or_else(|| "cohorts must be an array".to_owned())?;
        let mut cohorts = Vec::with_capacity(cohort_rows.len());
        let mut cohort_ids = HashSet::new();
        for row in cohort_rows {
            let cohort_id = field_text(row, "cohort_id")?.to_owned();
            if !cohort_ids.insert(cohort_id.clone()) { return Err(format!("duplicate cohort {cohort_id}")); }
            let assignment_mechanism = field_text(row, "assignment_mechanism")?.to_owned();
            let mut records = Vec::new();
            let mut observed_ids = HashSet::new();
            for record_row in row.get("records").and_then(Value::as_array)
                .ok_or_else(|| format!("cohort {cohort_id} records must be an array"))? {
                let sample_id = field_text(record_row, "sample_id")?.to_owned();
                if !observed_ids.insert(sample_id.clone()) {
                    return Err(format!("cohort {cohort_id} duplicates sample {sample_id}"));
                }
                let Some(&si) = sample_index.get(&sample_id) else {
                    return Err(format!("cohort {cohort_id} references unknown sample {sample_id}"));
                };
                let batch = field_text(record_row, "batch_id")?.to_owned();
                let family = field_text(record_row, "material_family")?.to_owned();
                if batch != samples[si].batch || family != samples[si].family {
                    return Err(format!("cohort {cohort_id} sample {sample_id} has inconsistent batch/family lineage"));
                }
                let treatment = record_row.get("treatment_assigned").and_then(Value::as_bool)
                    .ok_or_else(|| format!("sample {sample_id} treatment_assigned must be boolean"))?;
                let propensity = field_number(record_row, "assignment_probability")?;
                if !(0.0..1.0).contains(&propensity) {
                    return Err(format!("sample {sample_id} assignment probability must be strictly between 0 and 1"));
                }
                let delivered_dose = field_number(record_row, "delivered_dose_fraction")?;
                if !(0.0..=1.0).contains(&delivered_dose) {
                    return Err(format!("sample {sample_id} delivered dose must be in [0,1]"));
                }
                let outcome_observed = record_row.get("outcome_observed").and_then(Value::as_bool)
                    .ok_or_else(|| format!("sample {sample_id} outcome_observed must be boolean"))?;
                let outcome = field_number(record_row, "observed_outcome")?;
                records.push(Record {
                    sample_id, batch, family, treatment, propensity, delivered_dose,
                    outcome_observed, outcome,
                });
            }
            if records.len() != samples.len() || observed_ids.len() != samples.len() {
                return Err(format!("cohort {cohort_id} must contain exactly one record per specimen"));
            }
            cohorts.push(Cohort { id: cohort_id, assignment_mechanism, records });
        }

        let truth_rows = fixture.get("ground_truth").and_then(Value::as_array)
            .ok_or_else(|| "ground_truth must be an array".to_owned())?;
        let mut truth = Vec::with_capacity(truth_rows.len());
        let mut truth_ids = HashSet::new();
        for row in truth_rows {
            let sample_id = field_text(row, "sample_id")?.to_owned();
            if !sample_index.contains_key(&sample_id) || !truth_ids.insert(sample_id.clone()) {
                return Err(format!("ground_truth has unknown or duplicate sample {sample_id}"));
            }
            let control = field_number(row, "outcome_under_control")?;
            let treated = field_number(row, "outcome_under_treatment")?;
            truth.push(PotentialOutcome { sample_id, control, treated });
        }
        if truth.len() != samples.len() {
            return Err("ground_truth must define both potential outcomes for every sample".to_owned());
        }
        cohorts.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(Self {
            id,
            schema_sha256: format!("{:x}", Sha256::digest(schema_bytes)),
            fixture_sha256: format!("{:x}", Sha256::digest(fixture_bytes)),
            target_outcome,
            treatment_receipt_semantics,
            samples,
            cohorts,
            truth,
        })
    }

    fn finite_population_ate(&self) -> f64 {
        self.truth.iter().map(|x| x.treated - x.control).sum::<f64>() / self.truth.len() as f64
    }

    fn pooled_difference(&self, cohort: &Cohort) -> Estimate {
        let observed: Vec<&Record> = cohort.records.iter().filter(|r| r.outcome_observed).collect();
        let treated: Vec<f64> = observed.iter().filter(|r| r.treatment).map(|r| r.outcome).collect();
        let control: Vec<f64> = observed.iter().filter(|r| !r.treatment).map(|r| r.outcome).collect();
        let value = match (mean(&treated), mean(&control)) {
            (Some(t), Some(c)) => Some(t-c),
            _ => None,
        };
        Estimate {
            value,
            n_treated: treated.len(),
            n_control: control.len(),
            n_observed: observed.len(),
            reason: value.is_none().then(|| "pooled comparison lacks observed outcomes in one arm".to_owned()),
        }
    }

    fn blocked_difference(&self, cohort: &Cohort) -> Estimate {
        let mut blocks: HashMap<&str, Vec<&Record>> = HashMap::new();
        for record in cohort.records.iter().filter(|r| r.outcome_observed) {
            blocks.entry(&record.batch).or_default().push(record);
        }
        let mut weighted_sum = 0.0;
        let mut observed_count = 0usize;
        let mut treated_count = 0usize;
        let mut control_count = 0usize;
        for (batch, records) in blocks {
            let treated: Vec<f64> = records.iter().filter(|r| r.treatment).map(|r| r.outcome).collect();
            let control: Vec<f64> = records.iter().filter(|r| !r.treatment).map(|r| r.outcome).collect();
            let (Some(t), Some(c)) = (mean(&treated), mean(&control)) else {
                return Estimate {
                    value: None,
                    n_treated: treated_count + treated.len(),
                    n_control: control_count + control.len(),
                    n_observed: observed_count + records.len(),
                    reason: Some(format!("batch {batch} has no overlap between treatment arms")),
                };
            };
            weighted_sum += records.len() as f64 * (t-c);
            observed_count += records.len();
            treated_count += treated.len();
            control_count += control.len();
        }
        let value = (observed_count > 0).then_some(weighted_sum / observed_count as f64);
        Estimate {
            value,
            n_treated: treated_count,
            n_control: control_count,
            n_observed: observed_count,
            reason: None,
        }
    }

    // Horvitz-Thompson estimate. Its causal interpretation requires a correct
    // assignment propensity, conditional exchangeability, consistency and positivity.
    fn inverse_propensity_estimate(&self, cohort: &Cohort) -> Estimate {
        if cohort.records.iter().any(|r| !r.outcome_observed) {
            return Estimate {
                value: None,
                n_treated: cohort.records.iter().filter(|r| r.treatment).count(),
                n_control: cohort.records.iter().filter(|r| !r.treatment).count(),
                n_observed: cohort.records.iter().filter(|r| r.outcome_observed).count(),
                reason: Some("missing outcomes: v0.1 inverse-propensity estimators abstain without a declared missingness model".to_owned()),
            };
        }
        let observed: Vec<&Record> = cohort.records.iter().filter(|r| r.outcome_observed).collect();
        let bad_overlap = observed.iter().any(|r| r.propensity < 0.05 || r.propensity > 0.95);
        if bad_overlap {
            return Estimate {
                value: None,
                n_treated: observed.iter().filter(|r| r.treatment).count(),
                n_control: observed.iter().filter(|r| !r.treatment).count(),
                n_observed: observed.len(),
                reason: Some("assignment propensity violates the preregistered [0.05,0.95] overlap gate".to_owned()),
            };
        }
        let total: f64 = observed.iter().map(|r| {
            if r.treatment {
                r.outcome / r.propensity
            } else {
                -r.outcome / (1.0-r.propensity)
            }
        }).sum();
        let n = observed.len();
        Estimate {
            value: (n > 0).then_some(total / n as f64),
            n_treated: observed.iter().filter(|r| r.treatment).count(),
            n_control: observed.iter().filter(|r| !r.treatment).count(),
            n_observed: n,
            reason: None,
        }
    }

    // Hájek normalized inverse-probability estimator. It is a distinct,
    // generally finite-sample biased comparator to Horvitz-Thompson.
    fn hajek_estimate(&self, cohort: &Cohort) -> Estimate {
        if cohort.records.iter().any(|r| !r.outcome_observed) {
            return Estimate {
                value: None,
                n_treated: cohort.records.iter().filter(|r| r.treatment).count(),
                n_control: cohort.records.iter().filter(|r| !r.treatment).count(),
                n_observed: cohort.records.iter().filter(|r| r.outcome_observed).count(),
                reason: Some("missing outcomes: v0.1 inverse-propensity estimators abstain without a declared missingness model".to_owned()),
            };
        }
        let observed: Vec<&Record> = cohort.records.iter().filter(|r| r.outcome_observed).collect();
        if observed.iter().any(|r| r.propensity < 0.05 || r.propensity > 0.95) {
            return Estimate {
                value: None,
                n_treated: observed.iter().filter(|r| r.treatment).count(),
                n_control: observed.iter().filter(|r| !r.treatment).count(),
                n_observed: observed.len(),
                reason: Some("assignment propensity violates the preregistered [0.05,0.95] overlap gate".to_owned()),
            };
        }
        let treated: Vec<&Record> = observed.iter().copied().filter(|r| r.treatment).collect();
        let control: Vec<&Record> = observed.iter().copied().filter(|r| !r.treatment).collect();
        let treated_weight: f64 = treated.iter().map(|r| 1.0 / r.propensity).sum();
        let control_weight: f64 = control.iter().map(|r| 1.0 / (1.0-r.propensity)).sum();
        if treated_weight <= 0.0 || control_weight <= 0.0 {
            return Estimate {
                value: None,
                n_treated: treated.len(),
                n_control: control.len(),
                n_observed: observed.len(),
                reason: Some("Hájek estimator lacks positive weight in one assignment arm".to_owned()),
            };
        }
        let treated_mean = treated.iter().map(|r| r.outcome / r.propensity).sum::<f64>() / treated_weight;
        let control_mean = control.iter().map(|r| r.outcome / (1.0-r.propensity)).sum::<f64>() / control_weight;
        Estimate {
            value: Some(treated_mean-control_mean),
            n_treated: treated.len(),
            n_control: control.len(),
            n_observed: observed.len(),
            reason: None,
        }
    }

    // Family-stratified AIPW comparator. This is a deliberately simple
    // baseline for the tiny synthetic fixture, not a production nuisance model.
    fn family_aipw_estimate(&self, cohort: &Cohort) -> Estimate {
        let n_treated = cohort.records.iter().filter(|r| r.treatment).count();
        let n_control = cohort.records.len() - n_treated;
        if cohort.records.iter().any(|r| !r.outcome_observed) {
            return Estimate {
                value: None, n_treated, n_control,
                n_observed: cohort.records.iter().filter(|r| r.outcome_observed).count(),
                reason: Some("missing outcomes: v0.1 AIPW comparator abstains without a declared missingness model".to_owned()),
            };
        }
        if cohort.records.iter().any(|r| r.propensity < 0.05 || r.propensity > 0.95) {
            return Estimate {
                value: None, n_treated, n_control, n_observed: cohort.records.len(),
                reason: Some("assignment propensity violates the preregistered [0.05,0.95] overlap gate".to_owned()),
            };
        }
        let mut groups: HashMap<&str, (Vec<f64>, Vec<f64>)> = HashMap::new();
        for record in &cohort.records {
            let entry = groups.entry(&record.family).or_default();
            if record.treatment { entry.0.push(record.outcome); }
            else { entry.1.push(record.outcome); }
        }
        let mut means: HashMap<&str, (f64, f64)> = HashMap::new();
        for (family, (treated, control)) in groups {
            let (Some(m1), Some(m0)) = (mean(&treated), mean(&control)) else {
                return Estimate {
                    value: None, n_treated, n_control, n_observed: cohort.records.len(),
                    reason: Some(format!("family {family} lacks both assignment arms for the AIPW outcome model")),
                };
            };
            means.insert(family, (m1, m0));
        }
        let total: f64 = cohort.records.iter().map(|record| {
            let (m1, m0) = means[record.family.as_str()];
            if record.treatment {
                m1 - m0 + (record.outcome - m1) / record.propensity
            } else {
                m1 - m0 - (record.outcome - m0) / (1.0-record.propensity)
            }
        }).sum();
        Estimate {
            value: Some(total / cohort.records.len() as f64),
            n_treated, n_control, n_observed: cohort.records.len(), reason: None,
        }
    }

    fn propensity_diagnostics(&self, cohort: &Cohort) -> Value {
        let mut treated_weights = Vec::new();
        let mut control_weights = Vec::new();
        for record in cohort.records.iter().filter(|r| r.outcome_observed) {
            if record.treatment { treated_weights.push(1.0 / record.propensity); }
            else { control_weights.push(1.0 / (1.0-record.propensity)); }
        }
        let effective_sample_size = |weights: &[f64]| -> Option<f64> {
            if weights.is_empty() { return None; }
            let sum: f64 = weights.iter().sum();
            let sum_sq: f64 = weights.iter().map(|w| w*w).sum();
            (sum_sq > 0.0).then_some(sum*sum/sum_sq)
        };
        let propensities: Vec<f64> = cohort.records.iter().map(|r| r.propensity).collect();
        let min = propensities.iter().copied().fold(f64::INFINITY, f64::min);
        let max = propensities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        json!({
            "minimum_assignment_probability":min,
            "maximum_assignment_probability":max,
            "preregistered_overlap_gate":[0.05,0.95],
            "outside_overlap_gate_count":propensities.iter().filter(|p| **p<0.05 || **p>0.95).count(),
            "treated_inverse_probability_effective_sample_size":effective_sample_size(&treated_weights),
            "control_inverse_probability_effective_sample_size":effective_sample_size(&control_weights),
            "interpretation":"Diagnostics only; empirical overlap does not prove absence of unmeasured confounding."
        })
    }

    fn family_specific_descriptive_effects(&self, cohort: &Cohort) -> Value {
        let mut groups: HashMap<&str, (Vec<f64>, Vec<f64>)> = HashMap::new();
        for record in cohort.records.iter().filter(|r| r.outcome_observed) {
            let entry=groups.entry(&record.family).or_default();
            if record.treatment { entry.0.push(record.outcome); }
            else { entry.1.push(record.outcome); }
        }
        let mut rows:Vec<_>=groups.into_iter().collect();
        rows.sort_by(|a,b|a.0.cmp(b.0));
        json!(rows.into_iter().map(|(family,(treated,control))|json!({
            "material_family":family,
            "treated_count":treated.len(),
            "control_count":control.len(),
            "complete_case_difference_in_means":match (mean(&treated),mean(&control)) {
                (Some(t),Some(c))=>Some(t-c),
                _=>None
            },
            "interpretation":"Descriptive within-family contrast; causal only under the relevant assignment and identification assumptions."
        })).collect::<Vec<_>>())
    }

    fn assignment_combinations(indices: &[usize], treated_count: usize) -> Vec<Vec<usize>> {
        fn walk(
            indices: &[usize],
            treated_count: usize,
            start: usize,
            current: &mut Vec<usize>,
            output: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == treated_count {
                output.push(current.clone());
                return;
            }
            for i in start..indices.len() {
                if current.len() + (indices.len()-i) < treated_count { break; }
                current.push(indices[i]);
                walk(indices, treated_count, i+1, current, output);
                current.pop();
            }
        }
        let mut output = Vec::new();
        walk(indices, treated_count, 0, &mut Vec::new(), &mut output);
        output
    }

    fn blocked_statistic(records: &[Record], treated_ids: &HashSet<String>) -> Option<f64> {
        let mut blocks: HashMap<&str, Vec<&Record>> = HashMap::new();
        for record in records.iter().filter(|r| r.outcome_observed) {
            blocks.entry(&record.batch).or_default().push(record);
        }
        let mut weighted = 0.0;
        let mut n_total = 0usize;
        for (_batch, block) in blocks {
            let mut treated = Vec::new();
            let mut control = Vec::new();
            for record in &block {
                if treated_ids.contains(&record.sample_id) {
                    treated.push(record.outcome);
                } else {
                    control.push(record.outcome);
                }
            }
            let (Some(t), Some(c)) = (mean(&treated), mean(&control)) else { return None; };
            weighted += block.len() as f64 * (t-c);
            n_total += block.len();
        }
        (n_total > 0).then_some(weighted / n_total as f64)
    }

    fn exact_blocked_randomization(&self, cohort: &Cohort) -> Result<RandomizationResult, String> {
        if cohort.assignment_mechanism != "blocked_randomized" {
            return Err("exact randomization inference is reserved for blocked_randomized cohorts".to_owned());
        }
        if cohort.records.iter().any(|r| (r.propensity-0.5).abs() > 1.0e-9) {
            return Err("blocked randomization fixture must declare equal 0.5 assignment probabilities".to_owned());
        }
        let mut blocks: HashMap<String, Vec<usize>> = HashMap::new();
        for (index, record) in cohort.records.iter().enumerate() {
            if !record.outcome_observed {
                return Err("exact blocked randomization inference abstains when outcomes are missing; missingness cannot be silently imputed".to_owned());
            }
            blocks.entry(record.batch.clone()).or_default().push(index);
        }
        let mut ordered_blocks: Vec<(String, Vec<usize>, usize, Vec<Vec<usize>>)> = Vec::new();
        for (batch, indices) in blocks {
            let treated_count = indices.iter().filter(|i| cohort.records[**i].treatment).count();
            if treated_count == 0 || treated_count == indices.len() {
                return Err(format!("randomized block {batch} lacks both treatment arms"));
            }
            let assignments = Self::assignment_combinations(&indices, treated_count);
            ordered_blocks.push((batch, indices, treated_count, assignments));
        }
        ordered_blocks.sort_by(|a,b| a.0.cmp(&b.0));
        let observed_ids: HashSet<String> = cohort.records.iter().filter(|r| r.treatment)
            .map(|r| r.sample_id.clone()).collect();
        let observed_statistic = Self::blocked_statistic(&cohort.records, &observed_ids)
            .ok_or_else(|| "observed blocked statistic is not estimable".to_owned())?;
        let mut extreme = 0_u64;
        let mut total = 0_u64;
        fn enumerate(
            depth: usize,
            blocks: &[(String, Vec<usize>, usize, Vec<Vec<usize>>)],
            records: &[Record],
            selected: &mut HashSet<String>,
            observed_abs: f64,
            extreme: &mut u64,
            total: &mut u64,
        ) {
            if depth == blocks.len() {
                if let Some(statistic) = Instance::blocked_statistic(records, selected) {
                    *total += 1;
                    if statistic.abs() + 1.0e-12 >= observed_abs { *extreme += 1; }
                }
                return;
            }
            for assignment in &blocks[depth].3 {
                for index in assignment { selected.insert(records[*index].sample_id.clone()); }
                enumerate(depth+1, blocks, records, selected, observed_abs, extreme, total);
                for index in assignment { selected.remove(&records[*index].sample_id); }
            }
        }
        enumerate(
            0,&ordered_blocks,&cohort.records,&mut HashSet::new(),
            observed_statistic.abs(),&mut extreme,&mut total
        );
        if total == 0 { return Err("exact randomization enumeration produced no assignments".to_owned()); }
        let treated_per_block = ordered_blocks.iter().map(|(id,indices,k,_)| {
            (id.clone(),*k,indices.len()-*k)
        }).collect();
        Ok(RandomizationResult {
            observed_statistic,
            p_value_two_sided: extreme as f64 / total as f64,
            assignment_count: total,
            treated_per_block,
        })
    }

    fn truth_for(&self, sample_id: &str) -> Option<&PotentialOutcome> {
        self.truth.iter().find(|row| row.sample_id == sample_id)
    }

    fn estimate_receipt(&self, estimate: &Estimate, truth: f64) -> Value {
        json!({
            "estimate":estimate.value,
            "absolute_error_vs_finite_population_ate":estimate.value.map(|v|(v-truth).abs()),
            "n_treated":estimate.n_treated,
            "n_control":estimate.n_control,
            "n_observed":estimate.n_observed,
            "abstention_reason":estimate.reason
        })
    }

    fn baseline_balance(&self, cohort: &Cohort) -> Value {
        let observed: Vec<&Record> = cohort.records.iter().filter(|r| r.outcome_observed).collect();
        let pairing_mean = |treated: bool| -> Option<f64> {
            let ids: HashSet<&str> = observed.iter().filter(|r| r.treatment==treated)
                .map(|r| r.sample_id.as_str()).collect();
            mean(&self.samples.iter().filter(|s|ids.contains(s.id.as_str()))
                .map(|s|s.baseline_pairing).collect::<Vec<_>>())
        };
        let coherence_mean = |treated: bool| -> Option<f64> {
            let ids: HashSet<&str> = observed.iter().filter(|r| r.treatment==treated)
                .map(|r| r.sample_id.as_str()).collect();
            mean(&self.samples.iter().filter(|s|ids.contains(s.id.as_str()))
                .map(|s|s.baseline_coherence).collect::<Vec<_>>())
        };
        let family_counts: Vec<Value> = {
            let mut grouped: HashMap<&str,(usize,usize)> = HashMap::new();
            for record in &observed {
                let entry=grouped.entry(&record.family).or_default();
                if record.treatment { entry.0+=1; } else { entry.1+=1; }
            }
            let mut rows: Vec<_>=grouped.into_iter().collect();
            rows.sort_by(|a,b|a.0.cmp(b.0));
            rows.into_iter().map(|(family,(treated,control))|json!({
                "material_family":family,"treated":treated,"control":control
            })).collect()
        };
        let dose_by_assignment = |treated: bool| -> Option<f64> {
            mean(&cohort.records.iter().filter(|r|r.treatment==treated)
                .map(|r|r.delivered_dose).collect::<Vec<_>>())
        };
        let batch_counts: Vec<Value> = {
            let mut grouped: HashMap<&str,(usize,usize)> = HashMap::new();
            for record in &observed {
                let entry=grouped.entry(&record.batch).or_default();
                if record.treatment { entry.0+=1; } else { entry.1+=1; }
            }
            let mut rows: Vec<_>=grouped.into_iter().collect();
            rows.sort_by(|a,b|a.0.cmp(b.0));
            rows.into_iter().map(|(batch,(treated,control))|json!({
                "batch_id":batch,"treated":treated,"control":control
            })).collect()
        };
        json!({
            "baseline_pairing_proxy_mean_treated":pairing_mean(true),
            "baseline_pairing_proxy_mean_control":pairing_mean(false),
            "baseline_coherence_proxy_mean_treated":coherence_mean(true),
            "baseline_coherence_proxy_mean_control":coherence_mean(false),
            "mean_delivered_dose_fraction_treated":dose_by_assignment(true),
            "mean_delivered_dose_fraction_control":dose_by_assignment(false),
            "assignment_counts_by_material_family":family_counts,
            "assignment_counts_by_batch":batch_counts
        })
    }
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let instance=Instance::from_bytes(schema_bytes,fixture_bytes)?;
    let truth=instance.finite_population_ate();
    let mut cohort_receipts=Vec::new();
    for cohort in &instance.cohorts {
        let pooled=instance.pooled_difference(cohort);
        let blocked=instance.blocked_difference(cohort);
        let ht=instance.inverse_propensity_estimate(cohort);
        let hajek=instance.hajek_estimate(cohort);
        let aipw=instance.family_aipw_estimate(cohort);
        let randomization=if cohort.assignment_mechanism=="blocked_randomized" {
            match instance.exact_blocked_randomization(cohort) {
                Ok(result) => Some(json!({
                    "status":"exact_randomization_inference_available",
                    "observed_blocked_statistic":result.observed_statistic,
                    "two_sided_sharp_null_p_value":result.p_value_two_sided,
                    "enumerated_assignments":result.assignment_count,
                    "treated_control_counts_by_block":result.treated_per_block
                })),
                Err(reason) => Some(json!({"status":"abstained","reason":reason}))
            }
        } else { None };
        cohort_receipts.push(json!({
            "cohort_id":cohort.id,
            "assignment_mechanism":cohort.assignment_mechanism,
            "record_count":cohort.records.len(),
            "assignment_counts":{
                "treated":cohort.records.iter().filter(|r|r.treatment).count(),
                "control":cohort.records.iter().filter(|r|!r.treatment).count(),
                "missing_outcomes":cohort.records.iter().filter(|r|!r.outcome_observed).count()
            },
            "pooled_difference_in_means":instance.estimate_receipt(&pooled,truth),
            "batch_blocked_difference_in_means":instance.estimate_receipt(&blocked,truth),
            "horvitz_thompson_known_propensity":{
                "interpretation":"Causal only under correct assignment propensities, conditional exchangeability, consistency, positivity and appropriate handling of missing outcomes; observational weighting alone does not establish these assumptions.",
                "estimate":instance.estimate_receipt(&ht,truth)
            },
            "hajek_normalized_inverse_propensity":{
                "interpretation":"Normalized weighting comparator; generally finite-sample biased. Requires the same assignment and missingness assumptions as inverse-probability estimation.",
                "estimate":instance.estimate_receipt(&hajek,truth)
            },
            "family_stratified_aipw_comparator":{
                "interpretation":"Simple family-stratified outcome regression plus known assignment propensities. Requires conditional exchangeability given the adjustment set, correct propensity or outcome nuisance model, consistency, positivity and appropriate missingness handling. This tiny-fixture comparator is not cross-fitted.",
                "estimate":instance.estimate_receipt(&aipw,truth)
            },
            "propensity_overlap_and_effective_sample_size":instance.propensity_diagnostics(cohort),
            "family_specific_descriptive_contrasts":instance.family_specific_descriptive_effects(cohort),
            "baseline_balance_and_batch_counts":instance.baseline_balance(cohort),
            "exact_blocked_randomization_inference":randomization
        }));
    }
    let truth_by_family: Value = {
        let mut grouped:HashMap<&str,Vec<f64>>=HashMap::new();
        for sample in &instance.samples {
            if let Some(truth_row)=instance.truth_for(&sample.id) {
                grouped.entry(&sample.family).or_default().push(truth_row.treated-truth_row.control);
            }
        }
        let mut rows:Vec<_>=grouped.into_iter().collect();
        rows.sort_by(|a,b|a.0.cmp(b.0));
        json!(rows.into_iter().map(|(family,effects)|json!({
            "material_family":family,
            "finite_population_ate":mean(&effects),
            "sample_count":effects.len()
        })).collect::<Vec<_>>())
    };
    Ok(json!({
        "status":"synthetic_causal_identification_pass",
        "contract":"SC-CAUSAL-IDENTIFICATION-AUDIT-0.1",
        "instance_id":instance.id,
        "target_outcome":instance.target_outcome,
        "estimand":"finite_population_average_treatment_effect",
        "treatment_receipt_semantics":instance.treatment_receipt_semantics,
        "sample_count":instance.samples.len(),
        "cohort_count":instance.cohorts.len(),
        "finite_population_ground_truth_ate":truth,
        "finite_population_effect_by_material_family":truth_by_family,
        "cohorts":cohort_receipts,
        "schema_sha256":instance.schema_sha256,
        "fixture_sha256":instance.fixture_sha256,
        "claim_ceiling":"Synthetic assignment and finite-sample causal-estimation benchmark only. The ground truth is used only for benchmark scoring; no empirical superconductivity treatment effect or laboratory causal claim is established."
    }))
}

fn read(path:&PathBuf)->Vec<u8>{
    fs::read(path).unwrap_or_else(|e|{
        eprintln!("could not read {}: {e}",path.display());
        process::exit(1);
    })
}
fn usage(program:&str){
    eprintln!("Usage: {program} --schema <schema.json> --fixture <fixture.json> [--out <receipt.json>]");
}
fn main(){
    let args:Vec<String>=env::args().collect();
    let mut schema_path=None;let mut fixture_path=None;let mut output_path=None;let mut index=1;
    while index<args.len(){
        match args[index].as_str(){
            "--schema"|"--fixture"|"--out"=>{
                if index+1>=args.len(){usage(&args[0]);process::exit(2);}
                let path=PathBuf::from(&args[index+1]);
                match args[index].as_str(){
                    "--schema"=>schema_path=Some(path),
                    "--fixture"=>fixture_path=Some(path),
                    "--out"=>output_path=Some(path),
                    _=>unreachable!(),
                }
                index+=2;
            }
            "--help"|"-h"=>{usage(&args[0]);return;}
            unknown=>{eprintln!("unknown argument: {unknown}");usage(&args[0]);process::exit(2);}
        }
    }
    let (Some(schema_path),Some(fixture_path))=(schema_path,fixture_path) else{
        usage(&args[0]);process::exit(2);
    };
    let receipt=audit(&read(&schema_path),&read(&fixture_path)).unwrap_or_else(|e|{
        eprintln!("SC-CAUSAL-IDENTIFICATION-FAIL: {e}");
        process::exit(1);
    });
    let rendered=serde_json::to_string_pretty(&receipt).expect("JSON receipt serialization cannot fail");
    if let Some(path)=output_path{
        if let Err(e)=fs::write(&path,format!("{rendered}\n")){
            eprintln!("could not write {}: {e}",path.display());
            process::exit(1);
        }
    }else{println!("{rendered}");}
}

#[cfg(test)]
mod tests{
    use super::*;
    fn inputs_for(filename:&str)->(Vec<u8>,Vec<u8>){
        let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/superconductivity_causal_identification_v0_1.schema.json")).unwrap(),
            fs::read(root.join("data/research").join(filename)).unwrap()
        )
    }
    fn inputs()->(Vec<u8>,Vec<u8>){
        inputs_for("superconductivity_causal_identification_fixture_v0_1.json")
    }

    #[test]
    fn randomized_blocked_assignment_runs_exact_randomization_oracle(){
        let(schema,fixture)=inputs();
        let receipt=audit(&schema,&fixture).unwrap();
        let randomized=receipt["cohorts"].as_array().unwrap().iter()
            .find(|x|x["cohort_id"]=="blocked_randomized").unwrap();
        let exact=&randomized["exact_blocked_randomization_inference"];
        assert_eq!(exact["status"],"exact_randomization_inference_available");
        assert_eq!(exact["enumerated_assignments"],216);
        assert!((exact["two_sided_sharp_null_p_value"].as_f64().unwrap()-0.018518518518518517).abs()<1.0e-12);
    }

    #[test]
    fn historical_targeting_exposes_pooled_batch_bias_and_propensity_assumptions(){
        let(schema,fixture)=inputs();
        let receipt=audit(&schema,&fixture).unwrap();
        let historical=receipt["cohorts"].as_array().unwrap().iter()
            .find(|x|x["cohort_id"]=="historical_targeting").unwrap();
        let pooled=historical["pooled_difference_in_means"]["estimate"].as_f64().unwrap();
        let weighted=historical["horvitz_thompson_known_propensity"]["estimate"]["estimate"].as_f64().unwrap();
        let truth=receipt["finite_population_ground_truth_ate"].as_f64().unwrap();
        assert!((pooled-truth).abs()>(weighted-truth).abs());
    }

    #[test]
    fn potential_outcomes_are_used_for_scoring_not_estimators(){
        let(schema,fixture)=inputs();
        let mut value:Value=serde_json::from_slice(&fixture).unwrap();
        let baseline=audit(&schema,&fixture).unwrap();
        // Perturbing only held-out truth changes the scoring target, not observed-data estimators.
        value["ground_truth"][0]["outcome_under_treatment"]=json!(
            value["ground_truth"][0]["outcome_under_treatment"].as_f64().unwrap()+100.0
        );
        let changed=audit(&schema,&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(
            baseline["cohorts"][0]["pooled_difference_in_means"]["estimate"],
            changed["cohorts"][0]["pooled_difference_in_means"]["estimate"]
        );
        assert_ne!(
            baseline["finite_population_ground_truth_ate"],
            changed["finite_population_ground_truth_ate"]
        );
    }
    #[test]
    fn missing_randomized_outcome_is_reported_and_exact_test_abstains() {
        let (schema,fixture)=inputs();
        let mut value:Value=serde_json::from_slice(&fixture).unwrap();
        value["cohorts"][0]["records"][0]["outcome_observed"]=json!(false);
        let receipt=audit(&schema,&serde_json::to_vec(&value).unwrap()).unwrap();
        let randomized=receipt["cohorts"].as_array().unwrap().iter()
            .find(|x|x["cohort_id"]=="blocked_randomized").unwrap();
        assert_eq!(randomized["assignment_counts"]["missing_outcomes"],1);
        assert_eq!(randomized["exact_blocked_randomization_inference"]["status"],"abstained");
        assert!(randomized["exact_blocked_randomization_inference"]["reason"].as_str().unwrap().contains("missing"));
    }

    #[test]
    fn dose_receipt_preserves_assignment_separately_from_delivered_treatment() {
        let (schema,fixture)=inputs();
        let receipt=audit(&schema,&fixture).unwrap();
        let randomized=receipt["cohorts"].as_array().unwrap().iter()
            .find(|x|x["cohort_id"]=="blocked_randomized").unwrap();
        assert!((randomized["baseline_balance_and_batch_counts"]["mean_delivered_dose_fraction_treated"].as_f64().unwrap()-0.9333333333333332).abs()<1.0e-12);
        assert_eq!(randomized["baseline_balance_and_batch_counts"]["mean_delivered_dose_fraction_control"],0.0);
    }

    #[test]
    fn assigned_treatment_with_partial_delivery_remains_in_the_itt_arm() {
        let (schema,fixture)=inputs();
        let receipt=audit(&schema,&fixture).unwrap();
        let randomized=receipt["cohorts"].as_array().unwrap().iter()
            .find(|x|x["cohort_id"]=="blocked_randomized").unwrap();
        assert_eq!(
            randomized["baseline_balance_and_batch_counts"]["mean_delivered_dose_fraction_treated"],
            0.9333333333333332
        );
        assert_eq!(
            receipt["treatment_receipt_semantics"],
            "Primary estimand is intention-to-treat under assignment policy. Delivered dose is separately recorded for compliance auditing; v0.1 does not estimate a dose-response effect."
        );
        // The assigned-treated unit with a 0.6 dose is still counted as treated.
        assert_eq!(randomized["assignment_counts"]["treated"],6);
    }

    #[test]
    fn poor_propensity_overlap_causes_weighted_estimators_to_abstain() {
        let (schema,fixture)=inputs();
        let mut value:Value=serde_json::from_slice(&fixture).unwrap();
        value["cohorts"][1]["records"][0]["assignment_probability"]=json!(0.99);
        let receipt=audit(&schema,&serde_json::to_vec(&value).unwrap()).unwrap();
        let historical=receipt["cohorts"].as_array().unwrap().iter()
            .find(|x|x["cohort_id"]=="historical_targeting").unwrap();
        assert!(historical["horvitz_thompson_known_propensity"]["estimate"]["estimate"].is_null());
        assert!(historical["hajek_normalized_inverse_propensity"]["estimate"]["estimate"].is_null());
        assert!(historical["horvitz_thompson_known_propensity"]["estimate"]["abstention_reason"].as_str().is_some());
    }
    #[test]
    fn aipw_comparator_and_overlap_diagnostics_are_reported() {
        let (schema,fixture)=inputs();
        let receipt=audit(&schema,&fixture).unwrap();
        let historical=receipt["cohorts"].as_array().unwrap().iter()
            .find(|x|x["cohort_id"]=="historical_targeting").unwrap();
        assert!(historical["family_stratified_aipw_comparator"]["estimate"]["estimate"].as_f64().is_some());
        assert!(historical["propensity_overlap_and_effective_sample_size"]["treated_inverse_probability_effective_sample_size"].as_f64().is_some());
        assert_eq!(historical["family_specific_descriptive_contrasts"].as_array().unwrap().len(),2);
    }

    #[test]
    fn missing_randomized_outcome_abstains_all_estimators_that_need_outcome_model() {
        let (schema,fixture)=inputs();
        let mut value:Value=serde_json::from_slice(&fixture).unwrap();
        value["cohorts"][0]["records"][0]["outcome_observed"]=json!(false);
        let receipt=audit(&schema,&serde_json::to_vec(&value).unwrap()).unwrap();
        let randomized=receipt["cohorts"].as_array().unwrap().iter()
            .find(|x|x["cohort_id"]=="blocked_randomized").unwrap();
        assert!(randomized["horvitz_thompson_known_propensity"]["estimate"]["estimate"].is_null());
        assert!(randomized["hajek_normalized_inverse_propensity"]["estimate"]["estimate"].is_null());
        assert!(randomized["family_stratified_aipw_comparator"]["estimate"]["estimate"].is_null());
        assert_eq!(randomized["exact_blocked_randomization_inference"]["status"],"abstained");
    }


    #[test]
    fn frozen_missing_outcome_fixture_abstains_without_dropping_specimens(){
        let(schema,fixture)=inputs_for("superconductivity_causal_identification_fixture_missing_v0_1.json");
        let receipt=audit(&schema,&fixture).unwrap();
        let randomized=receipt["cohorts"].as_array().unwrap().iter()
            .find(|x|x["cohort_id"]=="blocked_randomized").unwrap();
        assert_eq!(randomized["assignment_counts"]["missing_outcomes"],1);
        assert!(randomized["horvitz_thompson_known_propensity"]["estimate"]["estimate"].is_null());
        assert!(randomized["hajek_normalized_inverse_propensity"]["estimate"]["estimate"].is_null());
        assert!(randomized["family_stratified_aipw_comparator"]["estimate"]["estimate"].is_null());
        assert_eq!(randomized["exact_blocked_randomization_inference"]["status"],"abstained");
    }

    #[test]
    fn frozen_positivity_fixture_abstains_all_weighted_estimators(){
        let(schema,fixture)=inputs_for("superconductivity_causal_identification_fixture_positivity_v0_1.json");
        let receipt=audit(&schema,&fixture).unwrap();
        let historical=receipt["cohorts"].as_array().unwrap().iter()
            .find(|x|x["cohort_id"]=="historical_targeting").unwrap();
        assert!(historical["horvitz_thompson_known_propensity"]["estimate"]["estimate"].is_null());
        assert!(historical["hajek_normalized_inverse_propensity"]["estimate"]["estimate"].is_null());
        assert!(historical["family_stratified_aipw_comparator"]["estimate"]["estimate"].is_null());
        assert_eq!(historical["propensity_overlap_and_effective_sample_size"]["outside_overlap_gate_count"],1);
    }

}
