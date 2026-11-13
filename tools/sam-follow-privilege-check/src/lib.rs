use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde_json::Value;

pub fn validate_files(schema_path: &Path, report_path: &Path) -> Result<(), Vec<String>> {
    let schema_text = fs::read_to_string(schema_path)
        .map_err(|error| vec![format!("read schema {}: {error}", schema_path.display())])?;
    let report_text = fs::read_to_string(report_path)
        .map_err(|error| vec![format!("read report {}: {error}", report_path.display())])?;
    let schema: Value = serde_json::from_str(&schema_text)
        .map_err(|error| vec![format!("parse schema {}: {error}", schema_path.display())])?;
    let report: Value = serde_json::from_str(&report_text)
        .map_err(|error| vec![format!("parse report {}: {error}", report_path.display())])?;
    validate_values(&schema, &report)
}

pub fn validate_values(schema: &Value, report: &Value) -> Result<(), Vec<String>> {
    if !jsonschema::meta::is_valid(schema) {
        return Err(vec![
            "schema is invalid under its declared JSON Schema meta-schema".into(),
        ]);
    }
    let validator = jsonschema::options()
        .offline()
        .build(schema)
        .map_err(|error| vec![format!("compile schema: {error}")])?;
    let mut errors: Vec<String> = validator
        .iter_errors(report)
        .map(|error| format!("schema {}: {error}", error.instance_path()))
        .collect();
    errors.extend(validate_semantics(report));
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_semantics(report: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let Some(benchmark) = report.get("benchmark") else {
        return errors;
    };

    let mut arms = BTreeMap::new();
    if let Some(values) = benchmark.get("arms").and_then(Value::as_array) {
        for (index, arm) in values.iter().enumerate() {
            let id = arm.get("arm_id").and_then(Value::as_str);
            if let Some(id) = id
                && arms.insert(id.to_owned(), arm).is_some()
            {
                errors.push(format!("benchmark.arms[{index}]: duplicate arm_id {id}"));
            }
            if id == Some("C") {
                let disabled = [
                    ("teacher_blending", "disabled"),
                    ("teacher_supervision", "disabled"),
                    ("learning_updates", "disabled"),
                    ("evaluator_truth", "post_commit_only"),
                    (
                        "observation_profile",
                        "simulation_20d_with_sim_peer_channels",
                    ),
                ];
                for (field, expected) in disabled {
                    if arm.get(field).and_then(Value::as_str) != Some(expected) {
                        errors.push(format!("arm C: {field} must be {expected}"));
                    }
                }
            }
            if id == Some("D")
                && arm.get("eligibility").and_then(Value::as_str)
                    != Some("blocked_until_scene_admission")
            {
                errors.push("arm D must remain blocked until scene admission".into());
            }
            if id == Some("E")
                && arm.get("eligibility").and_then(Value::as_str) != Some("non_authoritative_only")
            {
                errors.push("arm E must remain non-authoritative".into());
            }
        }
    }

    let trials = report.get("trials").and_then(Value::as_array);
    if report.get("benchmark_status").and_then(Value::as_str) == Some("specification_only")
        && trials.is_some_and(|rows| !rows.is_empty())
    {
        errors.push("specification_only report must not contain experiment trials".into());
    }

    let mut trial_ids = BTreeSet::new();
    let mut paired_trials: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    if let Some(trials) = trials {
        for (index, trial) in trials.iter().enumerate() {
            let prefix = format!("trials[{index}]");
            let trial_id = trial.get("trial_id").and_then(Value::as_str);
            if let Some(id) = trial_id
                && !trial_ids.insert(id.to_owned())
            {
                errors.push(format!("{prefix}: duplicate trial_id {id}"));
            }
            let pair_id = trial.get("paired_case_id").and_then(Value::as_str);
            let pair_role = trial.get("paired_role").and_then(Value::as_str);
            let intervention = trial.get("intervention_kind").and_then(Value::as_str);
            match pair_role {
                Some("standalone") if pair_id.is_some() || intervention != Some("none") => {
                    errors.push(format!(
                        "{prefix}: standalone trial cannot declare a pair or intervention"
                    ));
                }
                Some("baseline") if pair_id.is_none() || intervention != Some("none") => {
                    errors.push(format!("{prefix}: baseline trial requires paired_case_id and intervention_kind=none"));
                }
                Some("intervention") if pair_id.is_none() || intervention == Some("none") => {
                    errors.push(format!("{prefix}: intervention trial requires paired_case_id and a non-none intervention"));
                }
                _ => {}
            }
            if let Some(pair_id) = pair_id {
                paired_trials
                    .entry(pair_id.to_owned())
                    .or_default()
                    .push(trial);
            }
            let arm_id = trial.get("arm_id").and_then(Value::as_str);
            let Some(arm) = arm_id.and_then(|id| arms.get(id).copied()) else {
                errors.push(format!(
                    "{prefix}: arm_id does not resolve in benchmark arms"
                ));
                continue;
            };
            validate_trial(&mut errors, &prefix, trial, arm);
        }
    }
    validate_paired_trials(&mut errors, &paired_trials);
    errors
}

fn validate_paired_trials(errors: &mut Vec<String>, groups: &BTreeMap<String, Vec<&Value>>) {
    for (pair_id, trials) in groups {
        let prefix = format!("paired_case[{pair_id}]");
        if trials.len() != 2 {
            errors.push(format!(
                "{prefix}: paired case must contain exactly two trials"
            ));
            continue;
        }
        let baseline = trials
            .iter()
            .copied()
            .find(|trial| trial.get("paired_role").and_then(Value::as_str) == Some("baseline"));
        let intervention = trials
            .iter()
            .copied()
            .find(|trial| trial.get("paired_role").and_then(Value::as_str) == Some("intervention"));
        let (Some(baseline), Some(intervention)) = (baseline, intervention) else {
            errors.push(format!(
                "{prefix}: requires exactly one baseline and one intervention"
            ));
            continue;
        };
        for field in [
            "episode_id",
            "scenario_id",
            "actor_policy_id",
            "arm_id",
            "controller_digest",
            "action_observation_tick",
            "action_commit_tick",
        ] {
            if baseline.get(field) != intervention.get(field) {
                errors.push(format!("{prefix}: paired trials differ in {field}"));
            }
        }
        let kind = intervention
            .get("intervention_kind")
            .and_then(Value::as_str)
            .unwrap_or("none");
        match kind {
            "oracle_sidecar_mutation" => {
                if intervention.get("arm_id").and_then(Value::as_str) != Some("B") {
                    errors.push(format!(
                        "{prefix}: oracle-sidecar mutation requires masked-observation Arm B"
                    ));
                }
                let before = baseline
                    .get("observation")
                    .and_then(|x| x.get("policy_input_digest"));
                let after = intervention
                    .get("observation")
                    .and_then(|x| x.get("policy_input_digest"));
                let raw_before = baseline
                    .get("actions")
                    .and_then(|x| x.get("student_raw_action_digest"));
                let raw_after = intervention
                    .get("actions")
                    .and_then(|x| x.get("student_raw_action_digest"));
                if before != after || raw_before != raw_after {
                    errors.push(format!("{prefix}: masked oracle-sidecar mutation changed policy input or raw student action"));
                }
                if role_evidence_map(baseline, "simulator_oracle")
                    == role_evidence_map(intervention, "simulator_oracle")
                {
                    errors.push(format!("{prefix}: oracle-sidecar intervention did not change simulator-oracle evidence"));
                }
            }
            "raw_v2x_mutation" => {
                if intervention.get("arm_id").and_then(Value::as_str) != Some("E") {
                    errors.push(format!(
                        "{prefix}: V2X mutation requires cooperative-advisory Arm E"
                    ));
                }
                if baseline.get("follow_target") != intervention.get("follow_target") {
                    errors.push(format!(
                        "{prefix}: V2X mutation changed the local Follow target"
                    ));
                }
                if role_evidence_map(baseline, "remote_cooperative_claim")
                    == role_evidence_map(intervention, "remote_cooperative_claim")
                {
                    errors.push(format!(
                        "{prefix}: V2X intervention did not change remote-claim evidence"
                    ));
                }
            }
            "post_commit_shadow_teacher" => {
                if intervention.get("arm_id").and_then(Value::as_str) != Some("C") {
                    errors.push(format!(
                        "{prefix}: post-commit shadow teacher requires frozen Arm C"
                    ));
                }
                if baseline
                    .get("actions")
                    .and_then(|x| x.get("executed_action_digest"))
                    != intervention
                        .get("actions")
                        .and_then(|x| x.get("executed_action_digest"))
                {
                    errors.push(format!(
                        "{prefix}: post-commit shadow teacher changed executed action"
                    ));
                }
                let teacher = intervention.get("teacher");
                if teacher
                    .and_then(|x| x.get("evaluation_only"))
                    .and_then(Value::as_bool)
                    != Some(true)
                    || teacher
                        .and_then(|x| x.get("reference_only"))
                        .and_then(Value::as_bool)
                        != Some(true)
                    || teacher
                        .and_then(|x| x.get("blended"))
                        .and_then(Value::as_bool)
                        != Some(false)
                    || teacher
                        .and_then(|x| x.get("supervised"))
                        .and_then(Value::as_bool)
                        != Some(false)
                {
                    errors.push(format!("{prefix}: intervention must be an evaluator-only, reference-only, unblended and unsupervised teacher"));
                }
            }
            other => errors.push(format!("{prefix}: unsupported paired intervention {other}")),
        }
    }
}

fn role_evidence_map(trial: &Value, role: &str) -> BTreeMap<String, String> {
    trial
        .get("observation")
        .and_then(|x| x.get("sources"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|source| source.get("source_role").and_then(Value::as_str) == Some(role))
        .filter_map(|source| {
            Some((
                source.get("source_id")?.as_str()?.to_owned(),
                source.get("evidence_ref")?.as_str()?.to_owned(),
            ))
        })
        .collect()
}

fn validate_trial(errors: &mut Vec<String>, prefix: &str, trial: &Value, arm: &Value) {
    let observation = trial.get("observation");
    let profile = observation
        .and_then(|x| x.get("profile_id"))
        .and_then(Value::as_str);
    if arm.get("arm_id").and_then(Value::as_str) == Some("D") {
        errors.push(format!(
            "{prefix}: Arm D is blocked until scene-admitted local perception is qualified"
        ));
    }
    let sources = observation
        .and_then(|x| x.get("sources"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if profile != arm.get("observation_profile").and_then(Value::as_str) {
        errors.push(format!(
            "{prefix}.observation: profile does not match declared benchmark arm"
        ));
    }
    let observation_tick = trial.get("action_observation_tick").and_then(Value::as_u64);
    let commit_tick = trial.get("action_commit_tick").and_then(Value::as_u64);
    if observation_tick
        .zip(commit_tick)
        .is_none_or(|(observed, committed)| observed > committed)
    {
        errors.push(format!(
            "{prefix}: action observation tick must not exceed action commit tick"
        ));
    }
    let mut source_roles = BTreeMap::new();
    for (index, source) in sources.iter().enumerate() {
        let source_id = source.get("source_id").and_then(Value::as_str);
        let role = source.get("source_role").and_then(Value::as_str);
        let visible = source.get("policy_visible").and_then(Value::as_bool) == Some(true);
        if let (Some(source_id), Some(role)) = (source_id, role) {
            if source_roles
                .insert(source_id.to_owned(), role.to_owned())
                .is_some()
            {
                errors.push(format!(
                    "{prefix}.observation.sources[{index}]: duplicate source_id {source_id}"
                ));
            }
            if matches!(
                profile,
                Some("local_sensor_only" | "simulation_20d_peer_channels_masked")
            ) && visible
                && matches!(role, "simulator_oracle" | "remote_cooperative_claim")
            {
                errors.push(format!(
                    "{prefix}.observation.sources[{index}]: {role} cannot be policy-visible in this masked observation profile"
                ));
            }
            if visible && role == "unknown" {
                errors.push(format!("{prefix}.observation.sources[{index}]: unknown-role source cannot be policy-visible"));
            }
            if visible
                && !observation_tick.is_some_and(|tick| {
                    source
                        .get("received_tick")
                        .and_then(Value::as_u64)
                        .is_some_and(|received| received <= tick)
                })
            {
                errors.push(format!("{prefix}.observation.sources[{index}]: policy-visible source was received after the action observation tick"));
            }
        }
    }

    let teacher = trial.get("teacher");
    let computed = teacher
        .and_then(|x| x.get("action_computed"))
        .and_then(Value::as_bool)
        == Some(true);
    let blended = teacher
        .and_then(|x| x.get("blended"))
        .and_then(Value::as_bool)
        == Some(true);
    let coefficient = teacher
        .and_then(|x| x.get("blend_coefficient"))
        .and_then(Value::as_f64)
        .unwrap_or(-1.0);
    let supervised = teacher
        .and_then(|x| x.get("supervised"))
        .and_then(Value::as_bool)
        == Some(true);
    let evaluation_only = teacher
        .and_then(|x| x.get("evaluation_only"))
        .and_then(Value::as_bool)
        == Some(true);
    let reference_only = teacher
        .and_then(|x| x.get("reference_only"))
        .and_then(Value::as_bool)
        == Some(true);
    let teacher_digest = teacher.and_then(|x| x.get("action_digest"));

    if blended && (!computed || coefficient <= 0.0) {
        errors.push(format!(
            "{prefix}.teacher: blending requires a computed action and positive coefficient"
        ));
    }
    if !blended && coefficient != 0.0 {
        errors.push(format!(
            "{prefix}.teacher: blend coefficient must be zero when not blended"
        ));
    }
    if supervised && !computed {
        errors.push(format!(
            "{prefix}.teacher: supervision requires a computed teacher action"
        ));
    }
    if computed && teacher_digest.and_then(Value::as_str).is_none() {
        errors.push(format!(
            "{prefix}.teacher: computed teacher action requires an action digest"
        ));
    }
    if !computed {
        let no_use = !blended && !supervised && !evaluation_only && !reference_only;
        if !no_use || teacher_digest.is_some_and(|x| !x.is_null()) {
            errors.push(format!(
                "{prefix}.teacher: absent teacher action cannot have use flags or a digest"
            ));
        }
    }
    if reference_only && (!computed || blended || supervised) {
        errors.push(format!(
            "{prefix}.teacher: reference_only requires computed, unblended and unsupervised action"
        ));
    }
    if evaluation_only && (!reference_only || blended || supervised) {
        errors.push(format!(
            "{prefix}.teacher: evaluation_only must be reference_only, unblended and unsupervised"
        ));
    }

    let learning = trial.get("learning");
    let optimizer = learning
        .and_then(|x| x.get("optimizer_update"))
        .and_then(Value::as_bool)
        == Some(true);
    let replay_insert = learning
        .and_then(|x| x.get("replay_inserted"))
        .and_then(Value::as_bool)
        == Some(true);
    let replay_sample = learning
        .and_then(|x| x.get("replay_sampled"))
        .and_then(Value::as_bool)
        == Some(true);
    let exploration = learning
        .and_then(|x| x.get("exploration_applied"))
        .and_then(Value::as_bool)
        == Some(true);
    let before = learning
        .and_then(|x| x.get("parameter_digest_before"))
        .and_then(Value::as_str);
    let after = learning
        .and_then(|x| x.get("parameter_digest_after"))
        .and_then(Value::as_str);
    let weights_changed = learning
        .and_then(|x| x.get("weights_changed"))
        .and_then(Value::as_bool)
        == Some(true);
    let digest_changed = before.zip(after).is_some_and(|(a, b)| a != b);
    if weights_changed != digest_changed {
        errors.push(format!(
            "{prefix}.learning: weights_changed flag disagrees with parameter digests"
        ));
    }

    if arm.get("arm_id").and_then(Value::as_str) == Some("C") {
        if blended || supervised || optimizer || replay_insert || replay_sample || exploration {
            errors.push(format!("{prefix}: arm C must disable teacher blend/supervision, optimizer, replay and exploration"));
        }
        if weights_changed || before != after {
            errors.push(format!(
                "{prefix}: arm C must preserve exact parameter digest"
            ));
        }
        if trial
            .get("evaluator_truth")
            .and_then(|x| x.get("available"))
            .and_then(Value::as_bool)
            == Some(true)
        {
            let commit = trial.get("action_commit_tick").and_then(Value::as_u64);
            let delivered = trial
                .get("evaluator_truth")
                .and_then(|x| x.get("delivered_tick"))
                .and_then(Value::as_u64);
            if commit.zip(delivered).is_none_or(|(c, d)| d <= c) {
                errors.push(format!(
                    "{prefix}: arm C evaluator truth must be delivered strictly after action commit"
                ));
            }
            if trial
                .get("evaluator_truth")
                .and_then(|x| x.get("policy_visible"))
                .and_then(Value::as_bool)
                == Some(true)
            {
                errors.push(format!(
                    "{prefix}: arm C evaluator truth cannot be policy-visible"
                ));
            }
        }
    }

    let inference = trial.get("inference_state");
    let recurrent_changed = inference
        .and_then(|x| x.get("recurrent_digest_before"))
        .and_then(Value::as_str)
        .zip(
            inference
                .and_then(|x| x.get("recurrent_digest_after"))
                .and_then(Value::as_str),
        )
        .is_some_and(|(a, b)| a != b);
    let recurrent_evolved = inference
        .and_then(|x| x.get("recurrent_evolved"))
        .and_then(Value::as_bool)
        == Some(true);
    if recurrent_changed != recurrent_evolved {
        errors.push(format!(
            "{prefix}.inference_state: recurrent_evolved flag disagrees with recurrent digests"
        ));
    }

    let truth = trial.get("evaluator_truth");
    let truth_available = truth
        .and_then(|x| x.get("available"))
        .and_then(Value::as_bool)
        == Some(true);
    if !truth_available
        && (truth
            .and_then(|x| x.get("delivered_tick"))
            .is_some_and(|tick| !tick.is_null())
            || truth
                .and_then(|x| x.get("source_ids"))
                .and_then(Value::as_array)
                .is_some_and(|ids| !ids.is_empty()))
    {
        errors.push(format!("{prefix}.evaluator_truth: unavailable truth must not carry a delivery tick or source IDs"));
    }
    if truth_available {
        let commit = trial.get("action_commit_tick").and_then(Value::as_u64);
        let delivered = truth
            .and_then(|x| x.get("delivered_tick"))
            .and_then(Value::as_u64);
        if commit.zip(delivered).is_none_or(|(c, d)| d <= c)
            && arm.get("evaluator_truth").and_then(Value::as_str) == Some("post_commit_only")
        {
            errors.push(format!(
                "{prefix}.evaluator_truth: post_commit_only truth must arrive after action commit"
            ));
        }
        if truth
            .and_then(|x| x.get("policy_visible"))
            .and_then(Value::as_bool)
            == Some(true)
        {
            errors.push(format!("{prefix}.evaluator_truth cannot be policy-visible"));
        }
        if let Some(ids) = truth
            .and_then(|x| x.get("source_ids"))
            .and_then(Value::as_array)
        {
            for id in ids {
                let Some(id) = id.as_str() else { continue };
                if sources.iter().any(|source| {
                    source.get("source_id").and_then(Value::as_str) == Some(id)
                        && source.get("policy_visible").and_then(Value::as_bool) == Some(true)
                }) {
                    errors.push(format!(
                        "{prefix}.evaluator_truth source {id} is also policy-visible"
                    ));
                }
            }
        }
    }

    let target = trial.get("follow_target");
    let available = target
        .and_then(|x| x.get("available"))
        .and_then(Value::as_bool)
        == Some(true);
    let target_role = target
        .and_then(|x| x.get("source_role"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let target_status = target
        .and_then(|x| x.get("status"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if available {
        if !matches!(target_status, "current") {
            errors.push(format!(
                "{prefix}.follow_target: available target must have current status"
            ));
        }
        let id = target
            .and_then(|x| x.get("target_id"))
            .and_then(Value::as_str);
        let age = target.and_then(|x| x.get("age_ms")).and_then(Value::as_f64);
        if id.is_none() || age.is_none() {
            errors.push(format!(
                "{prefix}.follow_target: available target requires ID and age"
            ));
        }
        let allowed_role = match arm.get("arm_id").and_then(Value::as_str) {
            Some("A") | Some("B") | Some("C") => target_role == "simulator_oracle",
            Some("D") | Some("E") => target_role == "local_sensor_observation",
            _ => false,
        };
        if !allowed_role {
            errors.push(format!(
                "{prefix}.follow_target: source role {target_role} is not admissible for this arm"
            ));
        }
        let source_exists = sources.iter().any(|source| {
            source.get("source_role").and_then(Value::as_str) == Some(target_role)
                && source.get("source_id").and_then(Value::as_str)
                    == target
                        .and_then(|x| x.get("target_id"))
                        .and_then(Value::as_str)
        });
        if !source_exists {
            errors.push(format!(
                "{prefix}.follow_target: target source must resolve in observation sources"
            ));
        }
    } else {
        if !matches!(target_status, "missing" | "stale" | "rejected" | "unknown") {
            errors.push(format!("{prefix}.follow_target: unavailable target must state missing/stale/rejected/unknown"));
        }
        if target
            .and_then(|x| x.get("target_id"))
            .is_some_and(|id| !id.is_null())
            || target
                .and_then(|x| x.get("age_ms"))
                .is_some_and(|age| !age.is_null())
            || target
                .and_then(|x| x.get("frame_id"))
                .is_some_and(|frame| !frame.is_null())
        {
            errors.push(format!(
                "{prefix}.follow_target: unavailable target must not carry target ID, age or frame"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Value {
        serde_json::from_str(include_str!(
            "../../../docs/autonomy/SAM_FOLLOW_PRIVILEGE_BENCHMARK_SCHEMA_V0_1.json"
        ))
        .expect("schema JSON")
    }

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../docs/autonomy/fixtures/SAM_FOLLOW_PRIVILEGE_BENCHMARK_V0_1.json"
        ))
        .expect("fixture JSON")
    }

    fn digest(ch: char) -> String {
        format!("sha256:{}", std::iter::repeat_n(ch, 64).collect::<String>())
    }

    fn valid_arm_c_trial() -> Value {
        let t = serde_json::json!({
            "trial_id":"trial-c-001",
            "episode_id":"episode-001",
            "scenario_id":"scenario-crossing-001",
            "actor_policy_id":"actor-policy-synthetic-01",
            "arm_id":"C",
            "controller_digest":digest('8'),
            "paired_case_id":null,
            "paired_role":"standalone",
            "intervention_kind":"none",
            "action_observation_tick":100,
            "action_commit_tick":110,
            "observation":{
                "profile_id":"simulation_20d_with_sim_peer_channels",
                "policy_input_digest":digest('a'),
                "sources":[
                    {"source_id":"sim-peer-1","source_role":"simulator_oracle","policy_visible":true,"observed_tick":99,"received_tick":100,"evidence_ref":"sim-truth-tick-99"},
                    {"source_id":"evaluator-truth-1","source_role":"simulator_oracle","policy_visible":false,"observed_tick":99,"received_tick":120,"evidence_ref":"post-commit-evaluator-truth"}
                ]
            },
            "teacher":{
                "action_computed":true,"action_digest":digest('b'),"blended":false,"blend_coefficient":0.0,
                "supervised":false,"evaluation_only":true,"reference_only":true
            },
            "learning":{
                "optimizer_update":false,"replay_inserted":false,"replay_sampled":false,"exploration_applied":false,
                "parameter_digest_before":digest('c'),"parameter_digest_after":digest('c'),"weights_changed":false
            },
            "inference_state":{
                "recurrent_evolved":true,"recurrent_digest_before":digest('d'),"recurrent_digest_after":digest('e'),
                "normalization_evolved":true,"fep_tau_evolved":true
            },
            "actions":{"student_raw_action_digest":digest('f'),"post_blend_action_digest":digest('f'),"executed_action_digest":digest('f')},
            "evaluator_truth":{"available":true,"delivered_tick":120,"source_ids":["evaluator-truth-1"],"policy_visible":false},
            "follow_target":{"available":true,"source_role":"simulator_oracle","target_id":"sim-peer-1","age_ms":1.0,"frame_id":"road-local-01","status":"current"}
        });
        // Keep the fixture explicit: simulator input is allowed in this arm; only evaluator truth is post-commit.
        t
    }

    fn campaign_report(trials: Vec<Value>) -> Value {
        serde_json::json!({
            "schema_id":"luminous.sam.follow-privilege-benchmark",
            "schema_version":"0.1.0",
            "benchmark_status":"campaign_incomplete",
            "claim_ceiling":"synthetic_only",
            "benchmark":fixture()["benchmark"].clone(),
            "trials":trials
        })
    }

    fn set_pair_identity(trial: &mut Value, pair_id: &str, role: &str, intervention: &str) {
        trial["paired_case_id"] = Value::String(pair_id.to_owned());
        trial["paired_role"] = Value::String(role.to_owned());
        trial["intervention_kind"] = Value::String(intervention.to_owned());
    }

    fn make_arm_b_trial() -> Value {
        let mut trial = valid_arm_c_trial();
        trial["arm_id"] = Value::String("B".into());
        trial["observation"]["profile_id"] =
            Value::String("simulation_20d_peer_channels_masked".into());
        trial["observation"]["sources"][0]["policy_visible"] = Value::Bool(false);
        trial
    }

    fn make_arm_e_trial() -> Value {
        let mut trial = valid_arm_c_trial();
        trial["arm_id"] = Value::String("E".into());
        trial["observation"]["profile_id"] =
            Value::String("local_sensor_plus_remote_advisory".into());
        trial["observation"]["sources"] = serde_json::json!([
            {
                "source_id":"local-track-1","source_role":"local_sensor_observation","policy_visible":true,
                "observed_tick":99,"received_tick":100,"evidence_ref":"local-track-evidence-01"
            },
            {
                "source_id":"v2x-claim-1","source_role":"remote_cooperative_claim","policy_visible":true,
                "observed_tick":98,"received_tick":100,"evidence_ref":"remote-claim-version-01"
            },
            {
                "source_id":"evaluator-truth-1","source_role":"simulator_oracle","policy_visible":false,
                "observed_tick":99,"received_tick":120,"evidence_ref":"post-commit-evaluator-truth"
            }
        ]);
        trial["follow_target"] = serde_json::json!({
            "available":true,"source_role":"local_sensor_observation","target_id":"local-track-1",
            "age_ms":1.0,"frame_id":"road-local-01","status":"current"
        });
        trial
    }

    #[test]
    fn specification_fixture_is_valid_and_contains_no_trials() {
        let f = fixture();
        assert_eq!(f["benchmark_status"], "specification_only");
        assert_eq!(f["trials"].as_array().unwrap().len(), 0);
        validate_values(&schema(), &f).expect("specification fixture must validate");
    }

    #[test]
    fn accepts_arm_c_with_simulated_inputs_and_post_commit_truth() {
        validate_values(&schema(), &campaign_report(vec![valid_arm_c_trial()]))
            .expect("valid Arm C trial");
    }

    #[test]
    fn rejects_evaluator_truth_before_action_commit() {
        let mut trial = valid_arm_c_trial();
        trial["evaluator_truth"]["delivered_tick"] = Value::from(109);
        let report = serde_json::json!({"schema_id":"luminous.sam.follow-privilege-benchmark","schema_version":"0.1.0","benchmark_status":"campaign_incomplete","claim_ceiling":"synthetic_only","benchmark":fixture()["benchmark"].clone(),"trials":[trial]});
        assert!(validate_values(&schema(), &report).is_err());
    }

    #[test]
    fn rejects_evaluator_truth_in_policy_input() {
        let mut trial = valid_arm_c_trial();
        trial["observation"]["sources"][1]["policy_visible"] = Value::Bool(true);
        let report = serde_json::json!({"schema_id":"luminous.sam.follow-privilege-benchmark","schema_version":"0.1.0","benchmark_status":"campaign_incomplete","claim_ceiling":"synthetic_only","benchmark":fixture()["benchmark"].clone(),"trials":[trial]});
        assert!(validate_values(&schema(), &report).is_err());
    }

    #[test]
    fn rejects_arm_c_teacher_blending() {
        let mut trial = valid_arm_c_trial();
        trial["teacher"]["blended"] = Value::Bool(true);
        trial["teacher"]["blend_coefficient"] = Value::from(0.1);
        let report = serde_json::json!({"schema_id":"luminous.sam.follow-privilege-benchmark","schema_version":"0.1.0","benchmark_status":"campaign_incomplete","claim_ceiling":"synthetic_only","benchmark":fixture()["benchmark"].clone(),"trials":[trial]});
        assert!(validate_values(&schema(), &report).is_err());
    }

    #[test]
    fn rejects_arm_c_teacher_supervision_even_when_blend_is_zero() {
        let mut trial = valid_arm_c_trial();
        trial["teacher"]["supervised"] = Value::Bool(true);
        let report = serde_json::json!({"schema_id":"luminous.sam.follow-privilege-benchmark","schema_version":"0.1.0","benchmark_status":"campaign_incomplete","claim_ceiling":"synthetic_only","benchmark":fixture()["benchmark"].clone(),"trials":[trial]});
        assert!(validate_values(&schema(), &report).is_err());
    }

    #[test]
    fn permits_recurrent_state_evolution_without_weight_changes() {
        let mut trial = valid_arm_c_trial();
        trial["inference_state"]["recurrent_evolved"] = Value::Bool(true);
        assert!(validate_values(&schema(), &serde_json::json!({"schema_id":"luminous.sam.follow-privilege-benchmark","schema_version":"0.1.0","benchmark_status":"campaign_incomplete","claim_ceiling":"synthetic_only","benchmark":fixture()["benchmark"].clone(),"trials":[trial]})).is_ok());
    }

    #[test]
    fn rejects_parameter_change_in_frozen_arm() {
        let mut trial = valid_arm_c_trial();
        trial["learning"]["parameter_digest_after"] = Value::String(digest('g'));
        trial["learning"]["weights_changed"] = Value::Bool(true);
        let report = serde_json::json!({"schema_id":"luminous.sam.follow-privilege-benchmark","schema_version":"0.1.0","benchmark_status":"campaign_incomplete","claim_ceiling":"synthetic_only","benchmark":fixture()["benchmark"].clone(),"trials":[trial]});
        assert!(validate_values(&schema(), &report).is_err());
    }

    #[test]
    fn rejects_v2x_claim_as_local_follow_target() {
        let mut trial = valid_arm_c_trial();
        trial["follow_target"]["source_role"] = Value::String("remote_cooperative_claim".into());
        let report = serde_json::json!({"schema_id":"luminous.sam.follow-privilege-benchmark","schema_version":"0.1.0","benchmark_status":"campaign_incomplete","claim_ceiling":"synthetic_only","benchmark":fixture()["benchmark"].clone(),"trials":[trial]});
        assert!(validate_values(&schema(), &report).is_err());
    }

    #[test]
    fn rejects_local_perception_profile_with_visible_simulator_oracle() {
        let mut trial = valid_arm_c_trial();
        trial["observation"]["profile_id"] = Value::String("local_sensor_only".into());
        let report = serde_json::json!({"schema_id":"luminous.sam.follow-privilege-benchmark","schema_version":"0.1.0","benchmark_status":"campaign_incomplete","claim_ceiling":"synthetic_only","benchmark":fixture()["benchmark"].clone(),"trials":[trial]});
        assert!(validate_values(&schema(), &report).is_err());
    }

    #[test]
    fn accepts_masked_oracle_sidecar_pair_without_student_input_or_action_change() {
        let mut baseline = make_arm_b_trial();
        set_pair_identity(&mut baseline, "pair-oracle-01", "baseline", "none");
        let mut intervention = baseline.clone();
        intervention["trial_id"] = Value::String("trial-b-intervention".into());
        set_pair_identity(
            &mut intervention,
            "pair-oracle-01",
            "intervention",
            "oracle_sidecar_mutation",
        );
        intervention["observation"]["sources"][0]["evidence_ref"] =
            Value::String("sim-truth-tick-99-mutated".into());
        validate_values(&schema(), &campaign_report(vec![baseline, intervention]))
            .expect("masked oracle sidecar must not change student input or action");
    }

    #[test]
    fn rejects_masked_oracle_sidecar_pair_when_student_action_changes() {
        let mut baseline = make_arm_b_trial();
        set_pair_identity(&mut baseline, "pair-oracle-02", "baseline", "none");
        let mut intervention = baseline.clone();
        intervention["trial_id"] = Value::String("trial-b-intervention-02".into());
        set_pair_identity(
            &mut intervention,
            "pair-oracle-02",
            "intervention",
            "oracle_sidecar_mutation",
        );
        intervention["observation"]["sources"][0]["evidence_ref"] =
            Value::String("sim-truth-tick-99-mutated".into());
        intervention["actions"]["student_raw_action_digest"] = Value::String(digest('g'));
        let report = campaign_report(vec![baseline, intervention]);
        assert!(validate_values(&schema(), &report).is_err());
    }

    #[test]
    fn accepts_v2x_claim_mutation_when_local_follow_target_is_unchanged() {
        let mut baseline = make_arm_e_trial();
        set_pair_identity(&mut baseline, "pair-v2x-01", "baseline", "none");
        let mut intervention = baseline.clone();
        intervention["trial_id"] = Value::String("trial-e-intervention".into());
        set_pair_identity(
            &mut intervention,
            "pair-v2x-01",
            "intervention",
            "raw_v2x_mutation",
        );
        intervention["observation"]["sources"][1]["evidence_ref"] =
            Value::String("remote-claim-version-02".into());
        intervention["observation"]["policy_input_digest"] = Value::String(digest('9'));
        validate_values(&schema(), &campaign_report(vec![baseline, intervention]))
            .expect("V2X mutation must preserve local track identity/evidence");
    }

    #[test]
    fn rejects_v2x_claim_mutation_when_local_follow_target_changes() {
        let mut baseline = make_arm_e_trial();
        set_pair_identity(&mut baseline, "pair-v2x-02", "baseline", "none");
        let mut intervention = baseline.clone();
        intervention["trial_id"] = Value::String("trial-e-intervention-02".into());
        set_pair_identity(
            &mut intervention,
            "pair-v2x-02",
            "intervention",
            "raw_v2x_mutation",
        );
        intervention["observation"]["sources"][1]["evidence_ref"] =
            Value::String("remote-claim-version-02".into());
        intervention["follow_target"]["target_id"] = Value::String("v2x-claim-1".into());
        let report = campaign_report(vec![baseline, intervention]);
        assert!(validate_values(&schema(), &report).is_err());
    }

    #[test]
    fn accepts_post_commit_shadow_teacher_when_executed_action_is_unchanged() {
        let mut baseline = valid_arm_c_trial();
        set_pair_identity(&mut baseline, "pair-shadow-01", "baseline", "none");
        baseline["teacher"] = serde_json::json!({
            "action_computed":false,"action_digest":null,"blended":false,"blend_coefficient":0.0,
            "supervised":false,"evaluation_only":false,"reference_only":false
        });
        let mut intervention = baseline.clone();
        intervention["trial_id"] = Value::String("trial-shadow-intervention".into());
        set_pair_identity(
            &mut intervention,
            "pair-shadow-01",
            "intervention",
            "post_commit_shadow_teacher",
        );
        intervention["teacher"] = serde_json::json!({
            "action_computed":true,"action_digest":digest('b'),"blended":false,"blend_coefficient":0.0,
            "supervised":false,"evaluation_only":true,"reference_only":true
        });
        validate_values(&schema(), &campaign_report(vec![baseline, intervention]))
            .expect("post-commit shadow teacher must not affect executed action");
    }

    #[test]
    fn rejects_post_commit_shadow_teacher_that_changes_executed_action() {
        let mut baseline = valid_arm_c_trial();
        set_pair_identity(&mut baseline, "pair-shadow-02", "baseline", "none");
        baseline["teacher"] = serde_json::json!({
            "action_computed":false,"action_digest":null,"blended":false,"blend_coefficient":0.0,
            "supervised":false,"evaluation_only":false,"reference_only":false
        });
        let mut intervention = baseline.clone();
        intervention["trial_id"] = Value::String("trial-shadow-intervention-02".into());
        set_pair_identity(
            &mut intervention,
            "pair-shadow-02",
            "intervention",
            "post_commit_shadow_teacher",
        );
        intervention["teacher"] = serde_json::json!({
            "action_computed":true,"action_digest":digest('b'),"blended":false,"blend_coefficient":0.0,
            "supervised":false,"evaluation_only":true,"reference_only":true
        });
        intervention["actions"]["executed_action_digest"] = Value::String(digest('7'));
        let report = campaign_report(vec![baseline, intervention]);
        assert!(validate_values(&schema(), &report).is_err());
    }

    #[test]
    fn rejects_reference_only_teacher_that_was_blended() {
        let mut trial = valid_arm_c_trial();
        trial["teacher"]["blended"] = Value::Bool(true);
        trial["teacher"]["blend_coefficient"] = Value::from(0.2);
        let report = serde_json::json!({"schema_id":"luminous.sam.follow-privilege-benchmark","schema_version":"0.1.0","benchmark_status":"campaign_incomplete","claim_ceiling":"synthetic_only","benchmark":fixture()["benchmark"].clone(),"trials":[trial]});
        assert!(validate_values(&schema(), &report).is_err());
    }
}
