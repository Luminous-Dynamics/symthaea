use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, env, fs, process};

const ARMS: [&str; 3] = ["A0", "A1", "A2"];
const OUTCOMES: [&str; 7] = [
    "verified_restoration",
    "not_restored",
    "abandoned",
    "safety_stopped",
    "referred",
    "censored",
    "not_started",
];

const CONTRACT_SHA256: &str =
    "c7ffbabaca875faf066ee63e5d12acf7e46d36001cf36c9590799bffbea61577";

fn string_field<'a>(value: &'a Value, key: &str, where_: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{where_}.{key} must be a string"))
}

fn bool_field(value: &Value, key: &str, where_: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{where_}.{key} must be a boolean"))
}

fn nullable_i64(value: &Value, key: &str, where_: &str) -> Result<Option<i64>, String> {
    let field = value
        .get(key)
        .ok_or_else(|| format!("{where_}.{key} is required; use null when unknown"))?;
    if field.is_null() {
        Ok(None)
    } else {
        field
            .as_i64()
            .map(Some)
            .ok_or_else(|| format!("{where_}.{key} must be an integer or null"))
    }
}

fn nullable_string<'a>(
    value: &'a Value,
    key: &str,
    where_: &str,
) -> Result<Option<&'a str>, String> {
    let field = value
        .get(key)
        .ok_or_else(|| format!("{where_}.{key} is required; use null when absent"))?;
    if field.is_null() {
        return Ok(None);
    }
    let text = field
        .as_str()
        .ok_or_else(|| format!("{where_}.{key} must be a string or null"))?;
    if text.trim().is_empty() {
        return Err(format!("{where_}.{key} cannot be an empty string"));
    }
    Ok(Some(text))
}

fn enum_field<'a>(
    value: &'a Value,
    key: &str,
    where_: &str,
    allowed: &[&str],
) -> Result<&'a str, String> {
    let text = string_field(value, key, where_)?;
    if allowed.contains(&text) {
        Ok(text)
    } else {
        Err(format!(
            "{where_}.{key} has unsupported value {text:?}; expected one of {}",
            allowed.join(", ")
        ))
    }
}

fn audit(root: &Value, input_sha256: &str) -> Result<Value, String> {
    if string_field(root, "schema_version", "root")? != "human-tool-bench-v0.1" {
        return Err("root.schema_version must equal human-tool-bench-v0.1".into());
    }
    if string_field(root, "evidence_class", "root")? != "synthetic_contract_fixture" {
        return Err("root.evidence_class must remain synthetic_contract_fixture".into());
    }
    if string_field(root, "contract_sha256", "root")? != CONTRACT_SHA256 {
        return Err("root.contract_sha256 does not match the frozen v0.1 contract".into());
    }
    if string_field(root, "protocol_id", "root")? != "human-tool-protocol-v0.1" {
        return Err("root.protocol_id must equal human-tool-protocol-v0.1".into());
    }
    let root_currency = string_field(root, "currency", "root")?;
    if root_currency.len() != 3 || !root_currency.bytes().all(|byte| byte.is_ascii_uppercase()) {
        return Err("root.currency must be a three-letter uppercase currency code".into());
    }

    let episodes = root
        .get("episodes")
        .and_then(Value::as_array)
        .ok_or_else(|| "root.episodes must be an array".to_string())?;
    if episodes.is_empty() {
        return Err("root.episodes must not be empty".into());
    }

    let mut episode_ids = BTreeSet::new();
    for (index, episode) in episodes.iter().enumerate() {
        let where_ = format!("episodes[{index}]");
        let episode_id = string_field(episode, "episode_id", &where_)?;
        if !episode_ids.insert(episode_id.to_owned()) {
            return Err(format!("duplicate episode_id {episode_id:?}"));
        }
        if episode_id.trim().is_empty() {
            return Err(format!("{where_}.episode_id cannot be empty"));
        }
        let task_id = string_field(episode, "task_id", &where_)?;
        if task_id.trim().is_empty() {
            return Err(format!("{where_}.task_id cannot be empty"));
        }
        let procedure_id = string_field(episode, "procedure_id", &where_)?;
        if procedure_id.trim().is_empty() {
            return Err(format!("{where_}.procedure_id cannot be empty"));
        }
        if string_field(episode, "protocol_id", &where_)? != "human-tool-protocol-v0.1" {
            return Err(format!("{where_}.protocol_id differs from the frozen root protocol"));
        }
        let currency = string_field(episode, "currency", &where_)?;
        if currency != root_currency {
            return Err(format!(concat!(
                "{}.currency differs from root currency; ",
                "convert only under an explicit frozen conversion policy"
            ), where_));
        }
        if currency.len() != 3 || !currency.bytes().all(|byte| byte.is_ascii_uppercase()) {
            return Err(format!("{where_}.currency must be a three-letter uppercase currency code"));
        }

        let arm = enum_field(episode, "arm", &where_, &ARMS)?;
        let started = bool_field(episode, "started", &where_)?;
        let outcome = enum_field(episode, "outcome", &where_, &OUTCOMES)?;
        let observation_refs = episode
            .get("observation_refs")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{where_}.observation_refs must be an array"))?;
        let mut observation_ids = BTreeSet::new();
        for (ref_index, observation_ref) in observation_refs.iter().enumerate() {
            let reference = observation_ref
                .as_str()
                .ok_or_else(|| format!("{where_}.observation_refs[{ref_index}] must be a string"))?;
            if reference.trim().is_empty() {
                return Err(format!("{where_}.observation_refs[{ref_index}] cannot be empty"));
            }
            if !observation_ids.insert(reference.to_owned()) {
                return Err(format!(
                    "{where_}.observation_refs contains duplicate reference {reference:?}"
                ));
            }
        }
        if started && observation_refs.is_empty() {
            return Err(format!(concat!(
                "{}: every started attempt must reference at least one observation ",
                "or explicit missing-data record"
            ), where_));
        }
        if !started && !observation_refs.is_empty() {
            return Err(format!("{where_}: not-started records cannot claim observed evidence"));
        }

        if started && outcome == "not_started" {
            return Err(format!("{where_}: a started attempt cannot have outcome not_started"));
        }
        if !started && outcome != "not_started" {
            return Err(format!("{where_}: an unstarted record must have outcome not_started"));
        }

        let function_gate = enum_field(
            episode,
            "function_gate",
            &where_,
            &["pass", "fail", "unknown"],
        )?;
        let safety_gate = enum_field(
            episode,
            "safety_gate",
            &where_,
            &["pass", "fail", "unknown"],
        )?;
        if outcome == "safety_stopped" && safety_gate != "fail" {
            return Err(format!("{where_}: safety_stopped requires safety_gate=fail"));
        }
        if outcome == "not_restored" && function_gate == "pass" {
            return Err(format!("{where_}: not_restored conflicts with function_gate=pass"));
        }

        let claim = enum_field(
            episode,
            "advisory_claim",
            &where_,
            &["none", "hypothesis", "abstain", "safe_to_operate"],
        )?;
        if claim == "safe_to_operate" {
            let critical = episode.get("seeded_critical_fault").and_then(Value::as_bool);
            return Err(if critical == Some(true) {
                format!("{where_}: forbidden false-safe advisory on seeded critical fault")
            } else {
                format!("{where_}: diagnostic advice must not make a safe-to-operate verdict")
            });
        }
        if arm == "A2" && !["hypothesis", "abstain"].contains(&claim) {
            return Err(format!(
                "{where_}: A2 must record hypothesis or abstain, not {claim:?}"
            ));
        }
        let advisory_evidence_refs = episode
            .get("advisory_evidence_refs")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{where_}.advisory_evidence_refs must be an array"))?;
        let mut advisory_ids = BTreeSet::new();
        for (ref_index, advisory_ref) in advisory_evidence_refs.iter().enumerate() {
            let reference = advisory_ref
                .as_str()
                .ok_or_else(|| {
                    format!("{where_}.advisory_evidence_refs[{ref_index}] must be a string")
                })?;
            if reference.trim().is_empty() {
                return Err(format!("{where_}.advisory_evidence_refs[{ref_index}] cannot be empty"));
            }
            if !advisory_ids.insert(reference.to_owned()) {
                return Err(format!(
                    "{where_}.advisory_evidence_refs contains duplicate reference {reference:?}"
                ));
            }
            if !observation_ids.contains(reference) {
                return Err(format!(concat!(
                    "{}: advisory reference {:?} is not present in the ",
                    "episode's observation_refs"
                ), where_, reference));
            }
        }
        if arm == "A2" {
            let profile = string_field(episode, "advisory_profile_id", &where_)?;
            if profile.trim().is_empty() {
                return Err(format!("{where_}.advisory_profile_id cannot be empty"));
            }
            let reason = string_field(episode, "advisory_reason", &where_)?;
            if reason.trim().is_empty() {
                return Err(format!("{where_}.advisory_reason cannot be empty"));
            }
            if advisory_evidence_refs.is_empty() {
                return Err(format!(concat!(
                    "{}: every A2 advisory must cite an observation or ",
                    "explicit missing/contradiction record"
                ), where_));
            }
        } else {
            let profile = episode
                .get("advisory_profile_id")
                .ok_or_else(|| {
                    format!("{where_}.advisory_profile_id is required; use null outside A2")
                })?;
            let advisory_reason = episode
                .get("advisory_reason")
                .ok_or_else(|| {
                    format!("{where_}.advisory_reason is required; use null outside A2")
                })?;
            if !profile.is_null()
                || !advisory_evidence_refs.is_empty()
                || !advisory_reason.is_null()
            {
                return Err(format!(
                    "{where_}: A0/A1 cannot carry a Symthaea profile, reason, or advisory evidence"
                ));
            }
        }

        let critical = episode
            .get("seeded_critical_fault")
            .ok_or_else(|| format!("{where_}.seeded_critical_fault is required"))?;
        if !critical.is_null() && critical.as_bool().is_none() {
            return Err(format!("{where_}.seeded_critical_fault must be boolean or null"));
        }

        let elapsed = nullable_i64(episode, "elapsed_minutes_to_terminal", &where_)?;
        if elapsed.is_some_and(|value| value < 0) {
            return Err(format!("{where_}.elapsed_minutes_to_terminal cannot be negative"));
        }
        if started && outcome == "verified_restoration" && !elapsed.is_some_and(|value| value > 0) {
            return Err(format!(
                "{where_}: verified restoration requires positive, measured elapsed minutes"
            ));
        }

        let cost = nullable_i64(episode, "total_cost_minor_units", &where_)?;
        if cost.is_some_and(|value| value < 0) {
            return Err(format!("{where_}.total_cost_minor_units cannot be negative"));
        }
        let cost_status = enum_field(
            episode,
            "cost_status",
            &where_,
            &["measured", "unknown"],
        )?;
        if (cost_status == "unknown") != cost.is_none() {
            return Err(format!(
                concat!(
                    "{}: cost_status must agree with total_cost_minor_units; ",
                    "unknown is null, never imputed as zero"
                )
            ));
        }
        let function_receipt = nullable_string(episode, "function_receipt_id", &where_)?;
        let safety_receipt = nullable_string(episode, "safety_receipt_id", &where_)?;
        if outcome == "verified_restoration" {
            if string_field(episode, "function_gate", &where_)? != "pass"
                || string_field(episode, "safety_gate", &where_)? != "pass"
            {
                return Err(format!(
                    "{where_}: verified_restoration requires both function and safety gates to pass"
                ));
            }
            if function_receipt.is_none() || safety_receipt.is_none() {
                return Err(format!(
                    "{where_}: verified_restoration requires separate function and safety receipts"
                ));
            }
            if function_receipt == safety_receipt {
                return Err(format!(
                    "{where_}: function and safety receipts must be distinct references"
                ));
            }
        }

        let follow_up = enum_field(
            episode,
            "follow_up_status",
            &where_,
            &["recurrence", "no_recurrence", "censored", "pending", "unknown"],
        )?;
        let exposure = episode
            .get("follow_up_exposure_hours")
            .ok_or_else(|| format!("{where_}.follow_up_exposure_hours is required"))?;
        let exposure_hours = if exposure.is_null() {
            None
        } else {
            Some(
                exposure
                    .as_f64()
                    .ok_or_else(|| {
                        format!("{where_}.follow_up_exposure_hours must be numeric or null")
                    })?,
            )
        };
        if exposure_hours.is_some_and(|hours| !hours.is_finite() || hours < 0.0) {
            return Err(format!(
                "{where_}.follow_up_exposure_hours must be finite and non-negative"
            ));
        }
        if ["recurrence", "no_recurrence"].contains(&follow_up) {
            if outcome != "verified_restoration" {
                return Err(format!(
                    "{where_}: recurrence conclusions require a verified restoration first"
                ));
            }
            if !exposure_hours.is_some_and(|hours| hours > 0.0) {
                return Err(format!(
                    "{where_}: recurrence conclusions require positive observed use exposure"
                ));
            }
        }
    }

    let mut arm_reports = Map::new();
    for arm in ARMS {
        let selected: Vec<&Value> = episodes
            .iter()
            .filter(|episode| episode.get("arm").and_then(Value::as_str) == Some(arm))
            .collect();
        let attempts_started = selected
            .iter()
            .filter(|episode| episode.get("started").and_then(Value::as_bool) == Some(true))
            .count();
        let not_started = selected.len() - attempts_started;

        let mut outcome_counts = Map::new();
        for outcome in OUTCOMES {
            let count = selected
                .iter()
                .filter(|episode| episode.get("outcome").and_then(Value::as_str) == Some(outcome))
                .count();
            outcome_counts.insert(outcome.to_owned(), json!(count));
        }

        let mut restoration_minutes: Vec<i64> = selected
            .iter()
            .filter(|episode| {
                episode.get("started").and_then(Value::as_bool) == Some(true)
                    && episode.get("outcome").and_then(Value::as_str)
                        == Some("verified_restoration")
            })
            .filter_map(|episode| {
                episode
                    .get("elapsed_minutes_to_terminal")
                    .and_then(Value::as_i64)
            })
            .collect();
        restoration_minutes.sort_unstable();
        let n = restoration_minutes.len();
        let median = if n == 0 {
            Value::Null
        } else if n % 2 == 1 {
            json!(restoration_minutes[n / 2] as f64)
        } else {
            json!((restoration_minutes[n / 2 - 1] + restoration_minutes[n / 2]) as f64 / 2.0)
        };
        let p90 = if n == 0 {
            Value::Null
        } else {
            let rank = ((0.9_f64 * n as f64).ceil() as usize).max(1);
            json!(restoration_minutes[rank - 1])
        };
        let unknown_cost_count = selected
            .iter()
            .filter(|episode| {
                episode.get("started").and_then(Value::as_bool) == Some(true)
                    && episode.get("cost_status").and_then(Value::as_str) == Some("unknown")
            })
            .count();
        let unknown_elapsed_count = selected
            .iter()
            .filter(|episode| {
                episode.get("started").and_then(Value::as_bool) == Some(true)
                    && episode
                        .get("elapsed_minutes_to_terminal")
                        .is_none_or(Value::is_null)
            })
            .count();
        let advisory_hypotheses = selected
            .iter()
            .filter(|episode| {
                episode.get("advisory_claim").and_then(Value::as_str) == Some("hypothesis")
            })
            .count();
        let advisory_abstentions = selected
            .iter()
            .filter(|episode| {
                episode.get("advisory_claim").and_then(Value::as_str) == Some("abstain")
            })
            .count();
        let advisory_evidence_bound_count = selected
            .iter()
            .filter(|episode| {
                episode
                    .get("advisory_evidence_refs")
                    .and_then(Value::as_array)
                    .is_some_and(|references| !references.is_empty())
            })
            .count();

        arm_reports.insert(
            arm.to_owned(),
            json!({
                "attempts_started": attempts_started,
                "records_not_started": not_started,
                "outcome_counts": outcome_counts,
                "unknown_cost_count": unknown_cost_count,
                "unknown_elapsed_count": unknown_elapsed_count,
                "advisory_hypotheses": advisory_hypotheses,
                "advisory_abstentions": advisory_abstentions,
                "advisory_evidence_bound_count": advisory_evidence_bound_count,
                "verified_restoration_time_minutes": {
                    "n": n,
                    "median": median,
                    "p90_nearest_rank": p90,
                    "interpretation": concat!(
                        "conditional on verified restorations; not a replacement for ",
                        "all-attempt outcome accounting"
                    )
                }
            }),
        );
    }

    Ok(json!({
        "report_version": "human-tool-bench-report-v0.1",
        "evidence_class": "synthetic_contract_only",
        "contract_sha256": CONTRACT_SHA256,
        "protocol_id": "human-tool-protocol-v0.1",
        "currency": root_currency,
        "input_sha256": input_sha256,
        "episode_records": episodes.len(),
        "unique_episode_ids": episode_ids.len(),
        "arms": arm_reports,
        "claim_ceiling": concat!(
            "This audit validates the synthetic data contract and arithmetic only. It does not ",
            "estimate real repair performance, product safety, or human-study effects."
        )
    }))
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/human_tool_bench_v0_1.json".to_owned());
    let bytes = fs::read(&path).unwrap_or_else(|error| {
        eprintln!("cannot read {path:?}: {error}");
        process::exit(2);
    });
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let root: Value = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        eprintln!("invalid JSON in {path:?}: {error}");
        process::exit(2);
    });
    let report = audit(&root, &digest).unwrap_or_else(|error| {
        eprintln!("HUMAN-TOOL audit failed: {error}");
        process::exit(2);
    });
    println!("{}", serde_json::to_string_pretty(&report).expect("report serialization"));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../../fixtures/human_tool_bench_v0_1.json"))
            .expect("frozen fixture parses")
    }

    fn run(value: &Value) -> Result<Value, String> {
        audit(value, "test-input-digest")
    }

    #[test]
    fn golden_fixture_has_expected_arm_denominators() {
        let report = run(&fixture()).expect("valid fixture");
        assert_eq!(report["episode_records"], 9);
        assert_eq!(report["arms"]["A0"]["attempts_started"], 3);
        assert_eq!(report["arms"]["A1"]["attempts_started"], 3);
        assert_eq!(report["arms"]["A2"]["attempts_started"], 3);
        assert_eq!(report["arms"]["A2"]["outcome_counts"]["safety_stopped"], 1);
        assert_eq!(report["arms"]["A2"]["advisory_abstentions"], 1);
        assert_eq!(report["arms"]["A2"]["advisory_evidence_bound_count"], 3);
        assert_eq!(report["arms"]["A2"]["verified_restoration_time_minutes"]["median"], 12.0);
    }

    #[test]
    fn duplicate_episode_ids_fail_closed() {
        let mut data = fixture();
        let first = data["episodes"][0].clone();
        data["episodes"].as_array_mut().unwrap().push(first);
        assert!(run(&data).unwrap_err().contains("duplicate episode_id"));
    }

    #[test]
    fn safe_to_operate_verdict_is_forbidden() {
        let mut data = fixture();
        data["episodes"][7]["advisory_claim"] = json!("safe_to_operate");
        assert!(run(&data).unwrap_err().contains("forbidden false-safe advisory"));
    }

    #[test]
    fn unknown_cost_cannot_be_silently_zeroed() {
        let mut data = fixture();
        data["episodes"][1]["total_cost_minor_units"] = json!(0);
        assert!(run(&data).unwrap_err().contains("unknown is null"));
    }

    #[test]
    fn restoration_requires_function_and_safety_receipts() {
        let mut data = fixture();
        data["episodes"][0]["safety_receipt_id"] = Value::Null;
        assert!(run(&data).unwrap_err().contains("separate function and safety receipts"));
    }

    #[test]
    fn recurrence_claim_requires_verified_repair_and_positive_exposure() {
        let mut data = fixture();
        data["episodes"][6]["follow_up_status"] = json!("no_recurrence");
        data["episodes"][6]["follow_up_exposure_hours"] = json!(0);
        assert!(run(&data).unwrap_err().contains("positive observed use exposure"));
    }

    #[test]
    fn restoration_cannot_pass_with_failed_function_gate() {
        let mut data = fixture();
        data["episodes"][0]["function_gate"] = json!("fail");
        assert!(run(&data).unwrap_err().contains("both function and safety gates"));
    }

    #[test]
    fn a2_must_record_hypothesis_or_abstention() {
        let mut data = fixture();
        data["episodes"][8]["advisory_claim"] = json!("none");
        assert!(run(&data).unwrap_err().contains("A2 must record hypothesis or abstain"));
    }

    #[test]
    fn advisory_must_cite_an_observation_available_in_the_episode() {
        let mut data = fixture();
        data["episodes"][6]["advisory_evidence_refs"] = json!(["OBS-NOT-IN-EPISODE"]);
        assert!(run(&data)
            .unwrap_err()
            .contains("is not present in the episode's observation_refs"));
    }

    #[test]
    fn every_started_attempt_requires_observation_evidence() {
        let mut data = fixture();
        data["episodes"][0]["observation_refs"] = json!([]);
        assert!(run(&data)
            .unwrap_err()
            .contains("every started attempt must reference at least one observation"));
    }

    #[test]
    fn mixed_currency_comparison_fails_closed() {
        let mut data = fixture();
        data["episodes"][0]["currency"] = json!("USD");
        assert!(run(&data).unwrap_err().contains("differs from root currency"));
    }

    #[test]
    fn baseline_arms_cannot_smuggle_an_advisory_reason() {
        let mut data = fixture();
        data["episodes"][0]["advisory_reason"] = json!("hidden diagnostic output");
        assert!(run(&data).unwrap_err().contains("A0/A1 cannot carry a Symthaea profile"));
    }

    #[test]
    fn safety_stop_requires_failed_safety_gate() {
        let mut data = fixture();
        data["episodes"][7]["safety_gate"] = json!("unknown");
        assert!(run(&data).unwrap_err().contains("safety_stopped requires safety_gate=fail"));
    }

    #[test]
    fn frozen_contract_digest_mismatch_fails_closed() {
        let mut data = fixture();
        data["contract_sha256"] = json!(
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        );
        assert!(run(&data).unwrap_err().contains("does not match the frozen v0.1 contract"));
    }

    #[test]
    fn unstarted_records_do_not_enter_attempt_denominator() {
        let mut data = fixture();
        let mut row = data["episodes"][0].clone();
        row["episode_id"] = json!("EP-NOT-STARTED");
        row["started"] = json!(false);
        row["outcome"] = json!("not_started");
        row["elapsed_minutes_to_terminal"] = Value::Null;
        row["total_cost_minor_units"] = Value::Null;
        row["cost_status"] = json!("unknown");
        row["function_receipt_id"] = Value::Null;
        row["safety_receipt_id"] = Value::Null;
        row["function_gate"] = json!("unknown");
        row["safety_gate"] = json!("unknown");
        row["observation_refs"] = json!([]);
        row["follow_up_status"] = json!("unknown");
        row["follow_up_exposure_hours"] = Value::Null;
        data["episodes"].as_array_mut().unwrap().push(row);
        let report = run(&data).expect("well-formed unstarted record");
        assert_eq!(report["arms"]["A0"]["attempts_started"], 3);
        assert_eq!(report["arms"]["A0"]["records_not_started"], 1);
    }
}
