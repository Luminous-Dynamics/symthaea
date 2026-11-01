//! HARDSEC-023 Predictive Rank Martingale (PRM) auditor.
//!
//! Implements the Kuang-Xia 2026 fixed-reference PRM using the exact
//! Pólya-urn predictive null law for reused calibration ranks. The monitor
//! contains Order and Dispersion feature PRMs with ONS-adapted predictable
//! betting coefficients and an equal-weight martingale portfolio.
//!
//! The finite-sample anytime type-I guarantee requires iid calibration and
//! online observations within each context, plus independent uniform tie
//! randomization for atoms. The receipt cannot authenticate these assumptions.
//! In particular, the method does not establish validity for serially
//! dependent PMU windows.
//!
//! Usage:
//!   hardsec_predictive_rank_martingale --plan plan.json \
//!     --data data.json --context-receipt context.json --receipt receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PLAN_SCHEMA: &str = "hardsec-predictive-rank-martingale-plan-v1";
const DATA_SCHEMA: &str = "hardsec-predictive-rank-martingale-data-v1";
const RECEIPT_SCHEMA: &str = "hardsec-predictive-rank-martingale-receipt-v1";
const QUALIFIED_PMU: &str = "qualified_for_named_acquisition_conditions";
const MIN_FACTOR: f64 = 0.25;
const ONS_STEP: f64 = 4.5;
const EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
struct Sample {
    unit_id: String,
    score: f64,
    tie_uniform: f64,
}

#[derive(Debug, Clone)]
struct OnlineRecord {
    sequence_index: usize,
    window_id: String,
    context_id: String,
    unit_id: String,
    score: f64,
    tie_uniform: f64,
    sensor_state: String,
    pmu_qualification_result: String,
}

#[derive(Debug, Clone)]
struct FeatureState {
    feature_id: String,
    feature_type: String,
    portfolio_weight: f64,
    feature_values: Vec<f64>,
    lambda: f64,
    accumulator: f64,
    log_wealth: f64,
    first_crossing_index: Option<usize>,
}

#[derive(Debug, Clone)]
struct FeatureStep {
    observed_value: f64,
    predictive_mean: f64,
    centered_payoff: f64,
    max_abs_payoff: f64,
    b_t: f64,
    factor: f64,
    gradient: f64,
    accumulator_after: f64,
    lambda_unprojected: f64,
    log_wealth: f64,
    crossing_index: Option<usize>,
}

#[derive(Debug, Clone)]
struct Plan {
    monitor_id: String,
    evidence_domain: String,
    stream_id: String,
    evaluation_group_id: String,
    context_id: String,
    calibration_execution_group_id: String,
    evaluation_execution_group_id: String,
    calibration_device_boot_group_id: String,
    evaluation_device_boot_group_id: String,
    scoring_artifact_sha256: String,
    score_transform_id: String,
    tie_break_randomization_artifact_sha256: String,
    target_alpha: f64,
    context_assignment_policy_id: String,
    context_assignment_artifact_sha256: String,
    features: Vec<(String, String, f64)>,
}

#[derive(Debug, Clone)]
struct Data {
    calibration_samples: Vec<Sample>,
    online_records: Vec<OnlineRecord>,
}

#[derive(Debug, Clone)]
struct ContextAssignment {
    sequence_index: usize,
    window_id: String,
    selected_context_id: String,
}

#[derive(Debug, Clone)]
struct Args {
    plan: PathBuf,
    data: PathBuf,
    context_receipt: PathBuf,
    receipt: PathBuf,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn nonempty(value: &str, context: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(format!("{context} must be non-empty and contain no control characters"));
    }
    Ok(())
}

fn string_field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    let result = value.get(key).and_then(Value::as_str)
        .ok_or_else(|| format!("{context}.{key} must be a string"))?;
    nonempty(result, &format!("{context}.{key}"))?;
    Ok(result)
}

fn f64_field(value: &Value, key: &str, context: &str) -> Result<f64, String> {
    let result = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| format!("{context}.{key} must be numeric"))?;
    if !result.is_finite() {
        return Err(format!("{context}.{key} must be finite"));
    }
    Ok(result)
}

fn usize_field(value: &Value, key: &str, context: &str) -> Result<usize, String> {
    let raw = value.get(key).and_then(Value::as_u64)
        .ok_or_else(|| format!("{context}.{key} must be a non-negative integer"))?;
    usize::try_from(raw).map_err(|_| format!("{context}.{key} is outside the supported range"))
}

fn valid_sha(value: &str, context: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(format!("{context} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn validate_ons_contract(value: &Value) -> Result<(), String> {
    let context = "plan.ons";
    if string_field(value, "update", context)?
        != "lambda_next=project(lambda+(9/2)*g/(1+sum_g_squared),[-b_next,b_next])" {
        return Err("plan.ons.update does not match the implemented ONS update".to_owned());
    }
    if string_field(value, "projection_radius", context)?
        != "min(1,3/(4*max_abs_centered_feature))" {
        return Err("plan.ons.projection_radius does not match the implemented factor bound".to_owned());
    }
    for (field, expected) in [
        ("initial_lambda", 0.0),
        ("initial_accumulator", 1.0),
        ("minimum_betting_factor", MIN_FACTOR),
    ] {
        let actual = f64_field(value, field, context)?;
        if (actual - expected).abs() > EPS {
            return Err(format!("plan.ons.{field} does not match the implemented PRM contract"));
        }
    }
    Ok(())
}

fn parse_plan(value: &Value) -> Result<Plan, String> {
    if string_field(value, "schema_version", "plan")? != PLAN_SCHEMA {
        return Err("unsupported PRM plan schema_version".to_owned());
    }
    let monitor_id = string_field(value, "monitor_id", "plan")?.to_owned();
    let evidence_domain = string_field(value, "evidence_domain", "plan")?.to_owned();
    if !matches!(evidence_domain.as_str(), "synthetic_fixture" | "simulator" | "physical_hardware") {
        return Err("unsupported PRM evidence_domain".to_owned());
    }
    let stream_id = string_field(value, "stream_id", "plan")?.to_owned();
    let evaluation_group_id = string_field(value, "evaluation_group_id", "plan")?.to_owned();
    let context_id = string_field(value, "context_id", "plan")?.to_owned();
    let calibration_execution_group_id = string_field(value, "calibration_execution_group_id", "plan")?.to_owned();
    let evaluation_execution_group_id = string_field(value, "evaluation_execution_group_id", "plan")?.to_owned();
    let calibration_device_boot_group_id = string_field(value, "calibration_device_boot_group_id", "plan")?.to_owned();
    let evaluation_device_boot_group_id = string_field(value, "evaluation_device_boot_group_id", "plan")?.to_owned();
    if calibration_execution_group_id == evaluation_execution_group_id {
        return Err("calibration and evaluation execution groups must be distinct".to_owned());
    }
    if calibration_device_boot_group_id == evaluation_device_boot_group_id {
        return Err("calibration and evaluation device boot groups must be distinct".to_owned());
    }
    let scoring_artifact_sha256 = string_field(value, "scoring_artifact_sha256", "plan")?.to_owned();
    valid_sha(&scoring_artifact_sha256, "plan.scoring_artifact_sha256")?;
    if string_field(value, "score_direction", "plan")? != "higher_is_more_anomalous" {
        return Err("unsupported PRM score_direction".to_owned());
    }
    let score_transform_id = string_field(value, "score_transform_id", "plan")?.to_owned();
    if string_field(value, "tie_break_policy", "plan")? != "independent_uniform_lexicographic" {
        return Err("PRM requires independent_uniform_lexicographic tie breaking".to_owned());
    }
    let tie_break_randomization_artifact_sha256 =
        string_field(value, "tie_break_randomization_artifact_sha256", "plan")?.to_owned();
    valid_sha(&tie_break_randomization_artifact_sha256, "plan.tie_break_randomization_artifact_sha256")?;
    if string_field(value, "tie_break_independence_status", "plan")? != "assumption_not_verified" {
        return Err("tie-break independence must remain assumption_not_verified".to_owned());
    }
    let target_alpha = f64_field(value, "target_alpha", "plan")?;
    if !(0.0..1.0).contains(&target_alpha) {
        return Err("target_alpha must be in (0,1)".to_owned());
    }
    let context_assignment_policy_id = string_field(value, "context_assignment_policy_id", "plan")?.to_owned();
    let context_assignment_artifact_sha256 =
        string_field(value, "context_assignment_artifact_sha256", "plan")?.to_owned();
    valid_sha(&context_assignment_artifact_sha256, "plan.context_assignment_artifact_sha256")?;
    if scoring_artifact_sha256 == context_assignment_artifact_sha256 {
        return Err("score and context assignment artifacts must be separately pinned".to_owned());
    }
    for key in [
        "calibration_iid_status", "online_iid_status",
        "calibration_online_same_null_status", "score_transform_fixed_status",
        "context_exogeneity_status", "observation_admission_exogeneity_status",
    ] {
        if string_field(value, key, "plan")? != "assumption_not_verified" {
            return Err(format!("{key} must remain assumption_not_verified"));
        }
    }
    if string_field(value, "pmu_qualification_required", "plan")? != QUALIFIED_PMU {
        return Err("unsupported PMU qualification requirement".to_owned());
    }

    let feature_values = value.get("features").and_then(Value::as_array)
        .ok_or("plan.features must be an array")?;
    if feature_values.len() < 2 {
        return Err("PRM needs at least two pre-specified features".to_owned());
    }
    let mut feature_ids = BTreeSet::new();
    let mut features = Vec::with_capacity(feature_values.len());
    let mut weight_sum = 0.0;
    for (index, feature) in feature_values.iter().enumerate() {
        let context = format!("plan.features[{index}]");
        let feature_id = string_field(feature, "feature_id", &context)?.to_owned();
        let feature_type = string_field(feature, "feature_type", &context)?.to_owned();
        if !matches!(feature_type.as_str(), "order" | "dispersion") {
            return Err(format!("{context}.feature_type is unsupported"));
        }
        if !feature_ids.insert(feature_id.clone()) {
            return Err(format!("duplicate feature_id {feature_id}"));
        }
        let portfolio_weight = f64_field(feature, "portfolio_weight", &context)?;
        if portfolio_weight <= 0.0 || portfolio_weight > 1.0 {
            return Err(format!("{context}.portfolio_weight must be in (0,1]"));
        }
        weight_sum += portfolio_weight;
        features.push((feature_id, feature_type, portfolio_weight));
    }
    if (weight_sum - 1.0).abs() > EPS {
        return Err(format!("PRM portfolio weights must sum to one; observed {weight_sum}"));
    }
    let ons = value.get("ons").ok_or("plan.ons missing")?;
    validate_ons_contract(ons)?;
    Ok(Plan {
        monitor_id, evidence_domain, stream_id, evaluation_group_id, context_id,
        calibration_execution_group_id, evaluation_execution_group_id,
        calibration_device_boot_group_id, evaluation_device_boot_group_id,
        scoring_artifact_sha256, score_transform_id,
        tie_break_randomization_artifact_sha256, target_alpha,
        context_assignment_policy_id, context_assignment_artifact_sha256, features
    })
}

fn parse_data(value: &Value, plan: &Plan) -> Result<Data, String> {
    if string_field(value, "schema_version", "data")? != DATA_SCHEMA {
        return Err("unsupported PRM data schema_version".to_owned());
    }
    let matches = [
        ("evidence_domain", plan.evidence_domain.as_str()),
        ("stream_id", plan.stream_id.as_str()),
        ("evaluation_group_id", plan.evaluation_group_id.as_str()),
        ("context_id", plan.context_id.as_str()),
        ("scoring_artifact_sha256", plan.scoring_artifact_sha256.as_str()),
        ("tie_break_randomization_artifact_sha256", plan.tie_break_randomization_artifact_sha256.as_str()),
        ("calibration_execution_group_id", plan.calibration_execution_group_id.as_str()),
        ("evaluation_execution_group_id", plan.evaluation_execution_group_id.as_str()),
        ("calibration_device_boot_group_id", plan.calibration_device_boot_group_id.as_str()),
        ("evaluation_device_boot_group_id", plan.evaluation_device_boot_group_id.as_str())
    ];
    for (field, expected) in matches {
        if string_field(value, field, "data")? != expected {
            return Err(format!("data.{field} does not match PRM plan"));
        }
    }

    let calibration_values = value.get("calibration_samples").and_then(Value::as_array)
        .ok_or("data.calibration_samples must be an array")?;
    if calibration_values.len() < 2 {
        return Err("at least two calibration samples are required".to_owned());
    }
    let mut calibration_units = BTreeSet::new();
    let mut calibration = Vec::with_capacity(calibration_values.len());
    for (index, item) in calibration_values.iter().enumerate() {
        let context = format!("data.calibration_samples[{index}]");
        let unit_id = string_field(item, "sample_unit_id", &context)?.to_owned();
        if !calibration_units.insert(unit_id.clone()) {
            return Err(format!("duplicate calibration sample_unit_id {unit_id}"));
        }
        let score = f64_field(item, "score", &context)?;
        let tie_uniform = f64_field(item, "tie_break_uniform", &context)?;
        if !(0.0..1.0).contains(&tie_uniform) {
            return Err(format!("{context}.tie_break_uniform must be in [0,1)"));
        }
        calibration.push(Sample { unit_id, score, tie_uniform });
    }
    calibration.sort_by(|a,b| a.score.total_cmp(&b.score).then(a.tie_uniform.total_cmp(&b.tie_uniform)));
    for pair in calibration.windows(2) {
        if pair[0].score == pair[1].score && pair[0].tie_uniform == pair[1].tie_uniform {
            return Err("duplicate calibration score/tie-break pair".to_owned());
        }
    }

    let online_values = value.get("online_records").and_then(Value::as_array)
        .ok_or("data.online_records must be an array")?;
    if online_values.is_empty() {
        return Err("data.online_records must not be empty".to_owned());
    }
    let mut online_units = BTreeSet::new();
    let mut online_windows = BTreeSet::new();
    let mut online = Vec::with_capacity(online_values.len());
    for (index, item) in online_values.iter().enumerate() {
        let context = format!("data.online_records[{index}]");
        let sequence_index = usize_field(item, "sequence_index", &context)?;
        if sequence_index != index + 1 {
            return Err(format!("{context}.sequence_index must be contiguous from one"));
        }
        let window_id = string_field(item, "window_id", &context)?.to_owned();
        if !online_windows.insert(window_id.clone()) {
            return Err(format!("duplicate PRM window_id {window_id}"));
        }
        let context_id = string_field(item, "context_id", &context)?.to_owned();
        if context_id != plan.context_id {
            return Err(format!("{context}.context_id does not match plan context"));
        }
        let unit_id = string_field(item, "sample_unit_id", &context)?.to_owned();
        if !online_units.insert(unit_id.clone()) || calibration_units.contains(&unit_id) {
            return Err(format!("duplicate or cross-split sample_unit_id {unit_id}"));
        }
        let score = f64_field(item, "anomaly_score", &context)?;
        let tie_uniform = f64_field(item, "tie_break_uniform", &context)?;
        if !(0.0..1.0).contains(&tie_uniform) {
            return Err(format!("{context}.tie_break_uniform must be in [0,1)"));
        }
        let sensor_state = string_field(item, "sensor_state", &context)?.to_owned();
        if !matches!(sensor_state.as_str(), "valid" | "missing" | "stale" | "misaligned" | "multiplexed" | "untrusted") {
            return Err(format!("{context}.sensor_state is unsupported"));
        }
        let pmu_qualification_result = string_field(item, "pmu_qualification_result", &context)?.to_owned();
        online.push(OnlineRecord { sequence_index, window_id, context_id, unit_id, score, tie_uniform, sensor_state, pmu_qualification_result });
    }
    Ok(Data { calibration_samples:calibration, online_records:online })
}

fn parse_context_receipt(value: &Value, plan: &Plan, data: &Data) -> Result<Vec<ContextAssignment>, String> {
    if string_field(value, "schema_version", "context_receipt")? != "hardsec-context-assignment-receipt-v1"
        || string_field(value, "result", "context_receipt")? != "audited" {
        return Err("context assignment receipt schema/result invalid".to_owned());
    }
    for (field, expected) in [
        ("policy_id", plan.context_assignment_policy_id.as_str()),
        ("stream_id", plan.stream_id.as_str()),
        ("evaluation_group_id", plan.evaluation_group_id.as_str()),
        ("evidence_domain", plan.evidence_domain.as_str()),
        ("assignment_artifact_sha256", plan.context_assignment_artifact_sha256.as_str()),
        ("anomaly_score_artifact_sha256", plan.scoring_artifact_sha256.as_str())
    ] {
        if string_field(value, field, "context_receipt")? != expected {
            return Err(format!("context receipt {field} does not match PRM plan"));
        }
    }
    let mapping = value.get("runtime_mapping").ok_or("context receipt runtime_mapping missing")?;
    if mapping.get("contains_truth_labels").and_then(Value::as_bool) != Some(false)
        || mapping.get("mapping_digest_bound_to_assignment_input").and_then(Value::as_bool) != Some(true) {
        return Err("context runtime mapping must be truth-free and digest-bound".to_owned());
    }
    let rows = mapping.get("assignments").and_then(Value::as_array)
        .ok_or("context runtime mapping assignments missing")?;
    if rows.len() != data.online_records.len() {
        return Err("context receipt must map every PRM online record".to_owned());
    }
    let integrity = value.get("assignment_integrity").ok_or("context receipt assignment_integrity missing")?;
    let accepted = rows.iter().filter(|row| row["decision_status"] == "accepted").count();
    let abstained = rows.len()-accepted;
    if usize_field(integrity,"assignment_record_count","context_receipt.assignment_integrity")? != rows.len()
        || usize_field(integrity,"accepted_count","context_receipt.assignment_integrity")? != accepted
        || usize_field(integrity,"abstained_count","context_receipt.assignment_integrity")? != abstained {
        return Err("context receipt aggregate counts do not match runtime mapping".to_owned());
    }
    if accepted != rows.len() || abstained != 0 {
        return Err("PRM v1 requires every observation in the tested iid sequence to have an accepted context assignment; selective admission is not covered by the iid rank theorem".to_owned());
    }

    let mut output = Vec::with_capacity(rows.len());
    for (i,(row,record)) in rows.iter().zip(data.online_records.iter()).enumerate() {
        let c=format!("context_receipt.runtime_mapping.assignments[{i}]");
        if usize_field(row,"sequence_index",&c)? != record.sequence_index
            || string_field(row,"window_id",&c)? != record.window_id {
            return Err(format!("{c} sequence/window identity mismatch"));
        }
        if string_field(row,"decision_status",&c)? != "accepted"
            || string_field(row,"decision_reason",&c)? != "accepted"
            || string_field(row,"selected_context_id",&c)? != plan.context_id {
            return Err(format!("{c} must accept the planned PRM context"));
        }
        output.push(ContextAssignment {sequence_index:record.sequence_index,window_id:record.window_id.clone(),selected_context_id:plan.context_id.clone()});
    }
    Ok(output)
}

fn feature_value(feature_type:&str,u:f64)->f64 {
    match feature_type {
        "order" => u-0.5,
        "dispersion" => (2.0*u-1.0).abs()-0.5,
        _ => unreachable!("feature type validated in plan")
    }
}

fn feature_values(feature_type:&str,n:usize)->Vec<f64> {
    (0..=n).map(|j|feature_value(feature_type,j as f64/n as f64)).collect()
}

fn predictive_law(counts:&[u64],n:usize,t:usize)->Vec<f64> {
    let denominator=(n+t) as f64;
    counts.iter().map(|count|(1.0+*count as f64)/denominator).collect()
}

fn max_abs_centered(values:&[f64],pi:&[f64])->f64 {
    let mean=values.iter().zip(pi.iter()).map(|(h,p)|h*p).sum::<f64>();
    values.iter().map(|h|(h-mean).abs()).fold(0.0,f64::max)
}

fn projection_radius(max_abs_z:f64)->f64 {
    if max_abs_z<=EPS {1.0} else {1.0_f64.min(0.75/max_abs_z)}
}

fn lex_le(left_score:f64,left_tie:f64,right_score:f64,right_tie:f64)->bool {
    left_score<right_score || (left_score==right_score && left_tie<=right_tie)
}

fn rank_against_reference(calibration:&[Sample],score:f64,tie:f64)->usize {
    let count=calibration.partition_point(|sample|lex_le(sample.score,sample.tie_uniform,score,tie));
    count+1
}

fn log_sum_exp(values:&[f64])->Result<f64,String>{
    let max=values.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    if !max.is_finite(){return Err("PRM portfolio log-sum-exp has no finite terms".to_owned());}
    let sum=values.iter().map(|x|(*x-max).exp()).sum::<f64>();
    let result=max+sum.ln();
    if !result.is_finite(){return Err("PRM portfolio log wealth is non-finite".to_owned());}
    Ok(result)
}

fn read_hex_sha(value:&Value,field:&str,context:&str)->Result<(),String>{
    valid_sha(string_field(value,field,context)?,&format!("{context}.{field}"))
}

fn audit_values(
    plan_value:&Value,data_value:&Value,context_value:&Value,
    plan_bytes:&[u8],data_bytes:&[u8],context_bytes:&[u8]
)->Result<Value,String>{
    let plan=parse_plan(plan_value)?;
    let data=parse_data(data_value,&plan)?;
    let assignments=parse_context_receipt(context_value,&plan,&data)?;
    let n=data.calibration_samples.len();
    let mut counts=vec![0_u64;n+1];
    let mut features:Vec<FeatureState>=plan.features.iter().map(|(id,kind,weight)|FeatureState{
        feature_id:id.clone(),feature_type:kind.clone(),portfolio_weight:*weight,
        feature_values:feature_values(kind,n),lambda:0.0,accumulator:1.0,log_wealth:0.0,first_crossing_index:None
    }).collect();
    let mut output=Vec::with_capacity(data.online_records.len());
    let threshold=(-plan.target_alpha.ln());
    let mut portfolio_first_crossing=None;
    let mut last_portfolio_log=0.0;
    let mut min_component_factor=f64::INFINITY;

    for (record,assignment) in data.online_records.iter().zip(assignments.iter()) {
        if record.sensor_state!="valid" || record.pmu_qualification_result!=QUALIFIED_PMU {
            return Err(format!("PRM v1 requires all observations valid/qualified; rejected row {}",record.sequence_index));
        }
        if assignment.selected_context_id!=plan.context_id {
            return Err("PRM context assignment does not match plan context".to_owned());
        }
        let t=record.sequence_index;
        let pi=predictive_law(&counts,n,t);
        let rank=rank_against_reference(&data.calibration_samples,record.score,record.tie_uniform);
        let rank_index=rank-1;
        let observed_pi=pi[rank_index];
        if observed_pi<=0.0 || !observed_pi.is_finite() {
            return Err("PRM predictive rank probability must be positive".to_owned());
        }
        let counts_before=counts.clone();
        let mut pending=Vec::with_capacity(features.len());
        for state in features.iter_mut() {
            let h_observed=state.feature_values[rank_index];
            let conditional_mean=state.feature_values.iter().zip(pi.iter()).map(|(h,p)|h*p).sum::<f64>();
            let z=h_observed-conditional_mean;
            let max_abs_z=max_abs_centered(&state.feature_values,&pi);
            let b_t=projection_radius(max_abs_z);
            if state.lambda < -b_t-EPS || state.lambda > b_t+EPS {
                return Err(format!("feature {} ONS lambda escaped predictable domain",state.feature_id));
            }
            let factor=1.0+state.lambda*z;
            if factor<MIN_FACTOR-EPS || !factor.is_finite() {
                return Err(format!("feature {} betting factor violates minimum",state.feature_id));
            }
            min_component_factor=min_component_factor.min(factor);
            state.log_wealth+=factor.ln();
            let gradient=z/factor;
            let accumulator_after=state.accumulator+gradient*gradient;
            let lambda_unprojected=state.lambda+ONS_STEP*gradient/accumulator_after;
            let crossing_now=state.first_crossing_index.is_none() && state.log_wealth>=threshold;
            if crossing_now {state.first_crossing_index=Some(t);}
            pending.push(FeatureStep{
                observed_value:h_observed,predictive_mean:conditional_mean,centered_payoff:z,
                max_abs_payoff:max_abs_z,b_t,factor,gradient,accumulator_after,
                lambda_unprojected,log_wealth:state.log_wealth,crossing_index:state.first_crossing_index
            });
        }
        // There is one new rank observation, regardless of portfolio width.
        // Advance the shared Pólya-urn history exactly once.
        counts[rank_index]+=1;
        let pi_next=predictive_law(&counts,n,t+1);
        let mut feature_rows=Vec::with_capacity(features.len());
        for (state,step) in features.iter_mut().zip(pending.iter()) {
            let max_abs_next=max_abs_centered(&state.feature_values,&pi_next);
            let b_next=projection_radius(max_abs_next);
            let lambda_next=step.lambda_unprojected.clamp(-b_next,b_next);
            feature_rows.push(json!({
                "feature_id":state.feature_id,
                "feature_type":state.feature_type,
                "portfolio_weight":state.portfolio_weight,
                "observed_feature_value":step.observed_value,
                "predictive_feature_mean":step.predictive_mean,
                "centered_payoff_z":step.centered_payoff,
                "predictive_max_abs_payoff_B":step.max_abs_payoff,
                "ons_projection_radius_b_t":step.b_t,
                "lambda_t_predictable":state.lambda,
                "betting_factor":step.factor,
                "minimum_factor_required":MIN_FACTOR,
                "log_wealth":step.log_wealth,
                "accumulator_before":state.accumulator,
                "ons_gradient":step.gradient,
                "accumulator_after":step.accumulator_after,
                "lambda_unprojected_next":step.lambda_unprojected,
                "ons_projection_radius_b_next":b_next,
                "lambda_next":lambda_next,
                "first_crossing_sequence_index":step.crossing_index
            }));
            state.accumulator=step.accumulator_after;
            state.lambda=lambda_next;
        }
        let log_terms:Vec<f64>=features.iter().map(|f|f.portfolio_weight.ln()+f.log_wealth).collect();
        let portfolio_log=log_sum_exp(&log_terms)?;
        if portfolio_first_crossing.is_none() && portfolio_log>=threshold {
            portfolio_first_crossing=Some(t);
        }
        last_portfolio_log=portfolio_log;
        output.push(json!({
            "sequence_index":record.sequence_index,
            "window_id":record.window_id,
            "context_id":record.context_id,
            "context_assignment_status":"accepted",
            "context_assignment_reason":"accepted",
            "sample_unit_id":record.unit_id,
            "score":record.score,
            "tie_break_uniform":record.tie_uniform,
            "rank":rank,
            "predictive_rank_law":{
                "formula":"pi[t,j]=(1+N[t-1,j])/(n+t)",
                "calibration_size_n":n,
                "online_time_t":t,
                "denominator_n_plus_t":n+t,
                "observed_rank_probability_numerator":1+counts_before[rank_index],
                "observed_rank_probability":observed_pi,
                "rank_history_counts_before":counts_before,
                "rank_history_counts_after":counts
            },
            "features":feature_rows,
            "portfolio":{
                "weights_sum":features.iter().map(|f|f.portfolio_weight).sum::<f64>(),
                "log_e_value":portfolio_log,
                "e_value":if portfolio_log<700.0 {Some(portfolio_log.exp())} else {None},
                "threshold_log_e_value":threshold,
                "crossing_now":portfolio_first_crossing==Some(t),
                "first_crossing_sequence_index":portfolio_first_crossing
            }
        }));
    }

    let component_reports:Vec<Value>=features.iter().map(|f|json!({
        "feature_id":f.feature_id,"feature_type":f.feature_type,"portfolio_weight":f.portfolio_weight,
        "final_log_wealth":f.log_wealth,
        "final_e_value":if f.log_wealth<700.0 {Some(f.log_wealth.exp())} else {None},
        "first_crossing_sequence_index":f.first_crossing_index,
        "final_lambda":f.lambda,"final_ons_accumulator":f.accumulator
    })).collect();

    let result=if portfolio_first_crossing.is_some() {"alarm_threshold_crossed"} else {"no_alarm_threshold_crossed"};
    Ok(json!({
        "schema_version":RECEIPT_SCHEMA,
        "result":result,
        "claim_scope":"kuang_xia_predictive_rank_martingale_mechanics_under_unverified_iid_assumptions",
        "monitor_id":plan.monitor_id,
        "evidence_domain":plan.evidence_domain,
        "stream_id":plan.stream_id,
        "evaluation_group_id":plan.evaluation_group_id,
        "context_id":plan.context_id,
        "scoring_artifact_sha256":plan.scoring_artifact_sha256,
        "score_transform_id":plan.score_transform_id,
        "tie_break_randomization_artifact_sha256":plan.tie_break_randomization_artifact_sha256,
        "inputs":{
            "plan_sha256":sha256(plan_bytes),
            "data_sha256":sha256(data_bytes),
            "context_assignment_receipt_sha256":sha256(context_bytes)
        },
        "groups":{
            "calibration_execution_group_id":plan.calibration_execution_group_id,
            "evaluation_execution_group_id":plan.evaluation_execution_group_id,
            "calibration_device_boot_group_id":plan.calibration_device_boot_group_id,
            "evaluation_device_boot_group_id":plan.evaluation_device_boot_group_id,
            "execution_groups_distinct_by_declaration":true,
            "device_boot_groups_distinct_by_declaration":true,
            "sample_unit_ids_unique":true,
            "independence_authenticated":false
        },
        "calibration":{
            "sample_count":n,
            "fixed_reference_updated":false,
            "sample_digest":sha256(&serde_json::to_vec(&data.calibration_samples.iter().map(|x|json!({"unit":x.unit_id,"score":x.score,"tie":x.tie_uniform})).collect::<Vec<_>>()).unwrap())
        },
        "sequential_monitor":{
            "target_alpha":plan.target_alpha,
            "threshold_log_e_value":threshold,
            "online_observation_count":data.online_records.len(),
            "first_alarm_sequence_index":portfolio_first_crossing,
            "final_portfolio_log_e_value":last_portfolio_log,
            "first_crossing_is_latched":portfolio_first_crossing.is_some(),
            "component_reports":component_reports,
            "minimum_component_betting_factor_observed":min_component_factor,
            "portfolio_weights_sum":features.iter().map(|f|f.portfolio_weight).sum::<f64>(),
            "context_assignment_receipt_sha256":sha256(context_bytes),
            "window_decisions":output
        },
        "predictive_rank_law":{
            "name":"polya_urn_predictive_rank_law",
            "formula":"P(R_t=j | R_1,...,R_{t-1})=(1+N_{t-1,j})/(n+t)",
            "calibration_bins":n+1,
            "fixed_calibration_rank_reuse_modelled":true,
            "ranks_are_assumed_iid":false
        },
        "statistical_guarantees":{
            "method":"predictive_rank_martingale_order_dispersion_portfolio",
            "finite_sample_anytime_type_i_error_upper_bound_if_all_prm_assumptions_hold":plan.target_alpha,
            "calibration_iid_status":"assumption_not_verified",
            "online_iid_status":"assumption_not_verified",
            "calibration_online_same_null_status":"assumption_not_verified",
            "tie_break_randomization_independence_status":"assumption_not_verified",
            "score_transform_fixed_status":"assumption_not_verified",
            "context_exogeneity_status":"assumption_not_verified",
            "observation_admission_exogeneity_status":"assumption_not_verified",
            "theorem_guarantee_status":"not_established_from_input_receipts",
            "marginal_not_calibration_conditional":true,
            "effective_sample_size_estimate":null,
            "sequential_pfa_valid_for_arbitrary_serial_dependence":false
        },
        "limitations":[
            "The PRM exact predictive rank law holds when the fixed calibration sample and online observations are iid from the same unknown distribution under the null; it does not cover arbitrary autocorrelated PMU telemetry.",
            "The theorem gives marginal finite-sample anytime type-I control jointly over calibration and stream; it does not give calibration-conditional validity for every realized fixed reference.",
            "Independent uniform tie randomization is required for atomic/discrete score distributions; the randomization artifact digest is caller-declared and its independence is not authenticated.",
            "Distinct sample_unit, execution and boot identifiers are caller declarations; they do not prove independence.",
            "The scalar scoring transform must be fixed before calibration and monitoring; the receipt binds its digest but does not authenticate its execution.",
            "Selective admission/abstention may create a selected sequence; this v1 implementation rejects abstained records and does not prove selection exogeneity.",
            "The Order feature targets directional/stochastic-order changes; Dispersion targets center-versus-tail changes. Other departures can have zero contrast for both features.",
            "The PRM fixed-calibration power ceiling means an arbitrarily long stream does not create more independent calibration information."
        ]
    }))
}

fn audit(
    plan:&Value,data:&Value,context:&Value
)->Result<Value,String>{
    let plan_bytes=serde_json::to_vec_pretty(plan).map_err(|e|e.to_string())?;
    let data_bytes=serde_json::to_vec_pretty(data).map_err(|e|e.to_string())?;
    let context_bytes=serde_json::to_vec_pretty(context).map_err(|e|e.to_string())?;
    audit_values(plan,data,context,&plan_bytes,&data_bytes,&context_bytes)
}

fn read_input(path:&Path,label:&str)->Result<(Vec<u8>,Value),String>{
    let metadata=fs::symlink_metadata(path).map_err(|e|format!("cannot stat {label}: {e}"))?;
    if metadata.file_type().is_symlink()||!metadata.is_file(){return Err(format!("{label} must be a regular non-symlink file"));}
    let bytes=fs::read(path).map_err(|e|format!("cannot read {label}: {e}"))?;
    let value=serde_json::from_slice(&bytes).map_err(|e|format!("invalid {label} JSON: {e}"))?;
    Ok((bytes,value))
}

fn parse_args()->Result<Args,String>{
    let mut plan=None;let mut data=None;let mut context_receipt=None;let mut receipt=None;
    let mut args=env::args().skip(1);
    while let Some(flag)=args.next(){
        let target=match flag.as_str(){
            "--plan"=>&mut plan,"--data"=>&mut data,"--context-receipt"=>&mut context_receipt,"--receipt"=>&mut receipt,
            _=>return Err(format!("unknown argument {flag}"))
        };
        let path=args.next().ok_or_else(||format!("{flag} requires a path"))?;
        if target.replace(PathBuf::from(path)).is_some(){return Err(format!("duplicate argument {flag}"));}
    }
    Ok(Args{
        plan:plan.ok_or("missing --plan")?,data:data.ok_or("missing --data")?,
        context_receipt:context_receipt.ok_or("missing --context-receipt")?,receipt:receipt.ok_or("missing --receipt")?
    })
}

fn validate_receipt_location(receipt:&Path,inputs:&[&Path])->Result<(),String>{
    if receipt.exists(){return Err(format!("refusing to overwrite receipt {}",receipt.display()));}
    let parent=receipt.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or_else(||Path::new("."));
    let canonical_parent=fs::canonicalize(parent).map_err(|e|format!("receipt parent must exist: {e}"))?;
    for input in inputs{
        let canonical=fs::canonicalize(input).map_err(|e|format!("cannot canonicalize input {}: {e}",input.display()))?;
        let input_parent=canonical.parent().ok_or("input has no parent")?;
        if canonical_parent.starts_with(input_parent){return Err("receipt must be outside every input directory".to_owned());}
    }
    Ok(())
}

fn run()->Result<(),String>{
    let args=parse_args()?;
    let (plan_bytes,plan)=read_input(&args.plan,"plan")?;
    let (data_bytes,data)=read_input(&args.data,"data")?;
    let (context_bytes,context)=read_input(&args.context_receipt,"context receipt")?;
    validate_receipt_location(&args.receipt,&[args.plan.as_path(),args.data.as_path(),args.context_receipt.as_path()])?;
    let result=audit_values(&plan,&data,&context,&plan_bytes,&data_bytes,&context_bytes)?;
    let encoded=serde_json::to_vec_pretty(&result).map_err(|e|format!("cannot encode receipt: {e}"))?;
    let mut file=OpenOptions::new().write(true).create_new(true).open(&args.receipt).map_err(|e|format!("cannot create receipt without overwrite: {e}"))?;
    file.write_all(&encoded).map_err(|e|format!("cannot write receipt: {e}"))?;
    file.write_all(b"\n").map_err(|e|format!("cannot finish receipt: {e}"))?;
    Ok(())
}

fn main()->ExitCode{
    match run(){Ok(())=>ExitCode::SUCCESS,Err(error)=>{eprintln!("HARDSEC-023 PRM audit failed: {error}");ExitCode::FAILURE}}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture()->(Value,Value,Value){
        let p=serde_json::from_slice(include_bytes!("../tests/fixtures/hardsec/prm-v1/plan.json")).unwrap();
        let d=serde_json::from_slice(include_bytes!("../tests/fixtures/hardsec/prm-v1/data.json")).unwrap();
        let c=serde_json::from_slice(include_bytes!("../tests/fixtures/hardsec/prm-v1/context-receipt.json")).unwrap();
        (p,d,c)
    }

    #[test]
    fn known_answer_reports_anytime_prm_portfolio_mechanics(){
        let (p,d,c)=fixture();
        let r=audit(&p,&d,&c).unwrap();
        assert_eq!(r["result"],"alarm_threshold_crossed");
        assert_eq!(r["sequential_monitor"]["online_observation_count"],60);
        assert_eq!(r["sequential_monitor"]["portfolio_weights_sum"],1.0);
        assert_eq!(r["predictive_rank_law"]["fixed_calibration_rank_reuse_modelled"],true);
        assert_eq!(r["statistical_guarantees"]["theorem_guarantee_status"],"not_established_from_input_receipts");
        assert_eq!(r["statistical_guarantees"]["calibration_iid_status"],"assumption_not_verified");
        assert_eq!(r["statistical_guarantees"]["sequential_pfa_valid_for_arbitrary_serial_dependence"],false);
    }

    #[test]
    fn first_predictive_rank_law_is_uniform_over_n_plus_one_bins(){
        let (p,d,c)=fixture();
        let r=audit(&p,&d,&c).unwrap();
        let first=&r["sequential_monitor"]["window_decisions"][0]["predictive_rank_law"];
        assert_eq!(first["denominator_n_plus_t"],601);
        assert_eq!(first["observed_rank_probability_numerator"],1);
        assert!((first["observed_rank_probability"].as_f64().unwrap()-1.0/601.0).abs()<1e-14);
        assert!(first["rank_history_counts_before"].as_array().unwrap().iter().all(|x|x==0));
    }

    #[test]
    fn predictive_law_updates_repeated_rank_counts_polya_urn_style(){
        let (p,mut d,c)=fixture();
        let rank=rank_against_reference(&{
            let mut x=Vec::new();for i in 0..600{x.push(Sample{unit_id:i.to_string(),score:(i as f64+.5)/600.0,tie_uniform:.5});}x
        },0.4,.5);
        let score=0.4;
        d["online_records"]=json!([
            {"sequence_index":1,"window_id":"a","context_id":"compute","sample_unit_id":"a","anomaly_score":score,"tie_break_uniform":0.3,"sensor_state":"valid","pmu_qualification_result":QUALIFIED_PMU},
            {"sequence_index":2,"window_id":"b","context_id":"compute","sample_unit_id":"b","anomaly_score":score,"tie_break_uniform":0.3,"sensor_state":"valid","pmu_qualification_result":QUALIFIED_PMU}
        ]);
        let mut cc=c.clone();
        cc["runtime_mapping"]["assignments"]=json!([
            {"sequence_index":1,"window_id":"a","decision_status":"accepted","decision_reason":"accepted","selected_context_id":"compute"},
            {"sequence_index":2,"window_id":"b","decision_status":"accepted","decision_reason":"accepted","selected_context_id":"compute"}
        ]);
        cc["assignment_integrity"]["assignment_record_count"]=json!(2);
        cc["assignment_integrity"]["accepted_count"]=json!(2);
        cc["assignment_integrity"]["abstained_count"]=json!(0);
        let r=audit(&p,&d,&cc).unwrap();
        let rows=r["sequential_monitor"]["window_decisions"].as_array().unwrap();
        assert_eq!(rows[0]["rank"],rank);
        assert_eq!(rows[1]["rank"],rank);
        assert_eq!(rows[1]["predictive_rank_law"]["observed_rank_probability_numerator"],2);
        assert_eq!(rows[1]["predictive_rank_law"]["denominator_n_plus_t"],602);
        assert!((rows[1]["predictive_rank_law"]["observed_rank_probability"].as_f64().unwrap()-2.0/602.0).abs()<1e-14);
    }

    #[test]
    fn predictive_feature_payoffs_are_conditionally_centered(){
        let (p,d,c)=fixture();
        let plan=parse_plan(&p).unwrap();
        let data=parse_data(&d,&plan).unwrap();
        let n=data.calibration_samples.len();
        let counts=vec![3_u64;n+1];
        let pi=predictive_law(&counts,n,4);
        for (_,kind,_) in plan.features{
            let h=feature_values(&kind,n);
            let mean=h.iter().zip(pi.iter()).map(|(a,b)|a*b).sum::<f64>();
            let expectation=h.iter().zip(pi.iter()).map(|(value,prob)|prob*(value-mean)).sum::<f64>();
            assert!(expectation.abs()<1e-12);
        }
        let _=c;
    }

    #[test]
    fn every_predictable_prm_bet_has_exact_unit_null_expectation(){
        let (p,d,c)=fixture();
        let plan=parse_plan(&p).unwrap();
        let data=parse_data(&d,&plan).unwrap();
        let n=data.calibration_samples.len();
        let mut counts=vec![0_u64;n+1];
        let pi=predictive_law(&counts,n,1);
        for (_,kind,_) in &plan.features{
            let h=feature_values(kind,n);
            let mu=h.iter().zip(pi.iter()).map(|(v,p)|v*p).sum::<f64>();
            let radius=projection_radius(max_abs_centered(&h,&pi));
            let lambda=0.37_f64.clamp(-radius,radius);
            let expectation=h.iter().zip(pi.iter()).map(|(v,p)|p*(1.0+lambda*(v-mu))).sum::<f64>();
            assert!((expectation-1.0).abs()<1e-12);
        }
        let _=c;
    }

    #[test]
    fn betting_factors_are_bounded_away_from_zero(){
        let (p,d,c)=fixture();
        let r=audit(&p,&d,&c).unwrap();
        for row in r["sequential_monitor"]["window_decisions"].as_array().unwrap(){
            for feature in row["features"].as_array().unwrap(){
                assert!(feature["betting_factor"].as_f64().unwrap()>=MIN_FACTOR-EPS);
                let radius=feature["ons_projection_radius_b_t"].as_f64().unwrap();
                let lambda=feature["lambda_t_predictable"].as_f64().unwrap();
                assert!(lambda.abs()<=radius+EPS);
            }
        }
    }

    #[test]
    fn portfolio_wealth_is_the_weighted_sum_not_sum_of_logs(){
        let (p,d,c)=fixture();
        let r=audit(&p,&d,&c).unwrap();
        let features=r["sequential_monitor"]["component_reports"].as_array().unwrap();
        let expected=features.iter().map(|f|f["portfolio_weight"].as_f64().unwrap()*f["final_e_value"].as_f64().unwrap()).sum::<f64>();
        assert!((expected-r["sequential_monitor"]["final_portfolio_log_e_value"].as_f64().unwrap().exp()).abs()<1e-8);
    }


    #[test]
    fn shared_rank_history_advances_once_per_observation_not_per_feature(){
        let (p,d,c)=fixture();
        let r=audit(&p,&d,&c).unwrap();
        for (i,row) in r["sequential_monitor"]["window_decisions"].as_array().unwrap().iter().enumerate(){
            let law=&row["predictive_rank_law"];
            let before=law["rank_history_counts_before"].as_array().unwrap().iter().map(|x|x.as_u64().unwrap()).sum::<u64>();
            let after=law["rank_history_counts_after"].as_array().unwrap().iter().map(|x|x.as_u64().unwrap()).sum::<u64>();
            assert_eq!(before,i as u64);
            assert_eq!(after,(i+1) as u64);
            assert_eq!(after-before,1);
        }
    }

    #[test]
    fn randomized_tie_rank_is_lexicographic_and_deterministic(){
        let calibration=vec![
            Sample{unit_id:"a".into(),score:1.0,tie_uniform:0.1},
            Sample{unit_id:"b".into(),score:1.0,tie_uniform:0.4},
            Sample{unit_id:"c".into(),score:1.0,tie_uniform:0.9},
        ];
        assert_eq!(rank_against_reference(&calibration,1.0,0.05),1);
        assert_eq!(rank_against_reference(&calibration,1.0,0.5),3);
        assert_eq!(rank_against_reference(&calibration,1.0,0.95),4);
    }

    #[test]
    fn rejects_calibration_and_online_execution_overlap(){
        let (mut p,d,c)=fixture();
        p["evaluation_execution_group_id"]=p["calibration_execution_group_id"].clone();
        assert!(audit(&p,&d,&c).unwrap_err().contains("execution groups must be distinct"));
    }

    #[test]
    fn rejects_score_artifact_mismatch(){
        let (p,mut d,c)=fixture();
        d["scoring_artifact_sha256"]=json!("f".repeat(64));
        assert!(audit(&p,&d,&c).unwrap_err().contains("does not match PRM plan"));
    }

    #[test]
    fn rejects_randomization_artifact_mismatch(){
        let (p,mut d,c)=fixture();
        d["tie_break_randomization_artifact_sha256"]=json!("f".repeat(64));
        assert!(audit(&p,&d,&c).unwrap_err().contains("does not match PRM plan"));
    }

    #[test]
    fn rejects_tie_break_uniform_outside_unit_interval(){
        let (p,mut d,c)=fixture();
        d["online_records"][0]["tie_break_uniform"]=json!(1.0);
        assert!(audit(&p,&d,&c).unwrap_err().contains("must be in [0,1)"));
    }

    #[test]
    fn rejects_duplicate_calibration_unit_ids(){
        let (p,mut d,c)=fixture();
        d["calibration_samples"][1]["sample_unit_id"]=d["calibration_samples"][0]["sample_unit_id"].clone();
        assert!(audit(&p,&d,&c).unwrap_err().contains("duplicate calibration sample_unit_id"));
    }

    #[test]
    fn rejects_context_assignment_receipt_that_does_not_cover_every_row(){
        let (p,d,mut c)=fixture();
        c["runtime_mapping"]["assignments"].as_array_mut().unwrap().pop();
        assert!(audit(&p,&d,&c).unwrap_err().contains("must map every PRM online record"));
    }

    #[test]
    fn rejects_context_receipt_with_wrong_score_artifact(){
        let (p,d,mut c)=fixture();
        c["anomaly_score_artifact_sha256"]=json!("f".repeat(64));
        assert!(audit(&p,&d,&c).unwrap_err().contains("does not match PRM plan"));
    }

    #[test]
    fn rejects_nonaccepted_context_row_in_prm_iid_sequence(){
        let (p,d,mut c)=fixture();
        c["runtime_mapping"]["assignments"][0]["decision_status"]=json!("abstained");
        c["runtime_mapping"]["assignments"][0]["decision_reason"]=json!("ambiguous_assignment");
        c["runtime_mapping"]["assignments"][0]["selected_context_id"]=Value::Null;
        c["assignment_integrity"]["accepted_count"]=json!(59);
        c["assignment_integrity"]["abstained_count"]=json!(1);
        assert!(audit(&p,&d,&c).unwrap_err().contains("requires every observation"));
    }

    #[test]
    fn rejects_fixed_assumption_claims_marked_verified(){
        let (mut p,d,c)=fixture();
        p["online_iid_status"]=json!("verified");
        assert!(audit(&p,&d,&c).unwrap_err().contains("must remain assumption_not_verified"));
    }

    #[test]
    fn feature_order_and_dispersion_definitions_are_exact(){
        assert!((feature_value("order",0.0)+0.5).abs()<EPS);
        assert!((feature_value("order",1.0)-0.5).abs()<EPS);
        assert!((feature_value("dispersion",0.0)-0.5).abs()<EPS);
        assert!((feature_value("dispersion",0.5)+0.5).abs()<EPS);
        assert!((feature_value("dispersion",1.0)-0.5).abs()<EPS);
    }

    #[test]
    fn rejects_portfolio_weights_that_do_not_sum_to_one(){
        let (mut p,d,c)=fixture();
        p["features"][0]["portfolio_weight"]=json!(0.2);
        assert!(audit(&p,&d,&c).unwrap_err().contains("weights must sum to one"));
    }

    #[test]
    fn rejects_ons_update_formula_mismatch(){
        let (mut p,_,_)=fixture();
        p["ons"]["update"]=json!("lambda_next=lambda+g");
        assert!(parse_plan(&p).unwrap_err().contains("ONS update"));
    }

    #[test]
    fn rejects_ons_projection_radius_mismatch(){
        let (mut p,_,_)=fixture();
        p["ons"]["projection_radius"]=json!("unbounded");
        assert!(parse_plan(&p).unwrap_err().contains("projection_radius"));
    }

    #[test]
    fn rejects_ons_initialization_and_factor_mismatches(){
        for (field,value) in [
            ("initial_lambda",json!(0.1)),
            ("initial_accumulator",json!(2.0)),
            ("minimum_betting_factor",json!(0.0)),
        ] {
            let (mut p,_,_)=fixture();
            p["ons"][field]=value;
            assert!(parse_plan(&p).unwrap_err().contains(field));
        }
    }

    #[test]
    fn exact_polya_tree_anytime_false_alarm_mass_is_bounded(){
        fn walk(
            n: usize, horizon: usize, alpha: f64, time: usize, counts: &[u64],
            values: &[Vec<f64>], lambdas: &[f64], accumulators: &[f64],
            wealths: &[f64], path_probability: f64, total_mass: &mut f64,
            crossing_mass: &mut f64,
        ) {
            if time > horizon {
                *total_mass += path_probability;
                return;
            }
            let pi = predictive_law(counts, n, time);
            for rank_index in 0..=n {
                let transition_probability = pi[rank_index];
                let mut next_lambdas = Vec::with_capacity(values.len());
                let mut next_accumulators = Vec::with_capacity(values.len());
                let mut next_wealths = Vec::with_capacity(values.len());
                for feature_index in 0..values.len() {
                    let h = &values[feature_index];
                    let mean = h.iter().zip(pi.iter())
                        .map(|(value, probability)| value * probability).sum::<f64>();
                    let z = h[rank_index] - mean;
                    let radius = projection_radius(max_abs_centered(h, &pi));
                    let factor = 1.0 + lambdas[feature_index] * z;
                    assert!(factor >= MIN_FACTOR - EPS);
                    let gradient = z / factor;
                    let accumulator = accumulators[feature_index] + gradient * gradient;
                    let lambda = lambdas[feature_index] + ONS_STEP * gradient / accumulator;
                    next_lambdas.push(lambda);
                    next_accumulators.push(accumulator);
                    next_wealths.push(wealths[feature_index] * factor);
                    assert!(radius > 0.0);
                }
                let mut next_counts = counts.to_vec();
                next_counts[rank_index] += 1;
                let next_pi = predictive_law(&next_counts, n, time + 1);
                for feature_index in 0..values.len() {
                    let radius = projection_radius(max_abs_centered(&values[feature_index], &next_pi));
                    next_lambdas[feature_index] =
                        next_lambdas[feature_index].clamp(-radius, radius);
                }
                let portfolio_wealth = 0.5 * next_wealths[0] + 0.5 * next_wealths[1];
                let child_probability = path_probability * transition_probability;
                if portfolio_wealth >= 1.0 / alpha {
                    *crossing_mass += child_probability;
                    *total_mass += child_probability;
                } else {
                    walk(
                        n, horizon, alpha, time + 1, &next_counts, values,
                        &next_lambdas, &next_accumulators, &next_wealths,
                        child_probability, total_mass, crossing_mass,
                    );
                }
            }
        }

        let n = 2;
        let alpha = 0.5;
        let values = vec![feature_values("order", n), feature_values("dispersion", n)];
        let mut total_mass = 0.0;
        let mut crossing_mass = 0.0;
        walk(
            n, 8, alpha, 1, &vec![0; n + 1], &values, &[0.0, 0.0],
            &[1.0, 1.0], &[1.0, 1.0], 1.0, &mut total_mass, &mut crossing_mass,
        );
        assert!((total_mass - 1.0).abs() < 1e-10);
        assert!(crossing_mass > 0.0);
        assert!(crossing_mass <= alpha + 1e-12);
    }

    #[test]
    fn calibration_reference_is_immutable(){
        let (p,d,c)=fixture();
        let r=audit(&p,&d,&c).unwrap();
        assert_eq!(r["calibration"]["fixed_reference_updated"],false);
        assert_eq!(r["predictive_rank_law"]["ranks_are_assumed_iid"],false);
    }

    #[test]
    fn refuses_to_infer_validity_for_serially_dependent_pmu(){
        let (p,d,c)=fixture();
        let r=audit(&p,&d,&c).unwrap();
        assert_eq!(r["statistical_guarantees"]["sequential_pfa_valid_for_arbitrary_serial_dependence"],false);
        assert_eq!(r["statistical_guarantees"]["theorem_guarantee_status"],"not_established_from_input_receipts");
    }
}
