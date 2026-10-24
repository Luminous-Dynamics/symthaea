//! Independent consistency auditor for coding_gcs_ablation_benchmark receipts.
//!
//! This checks internal arithmetic, task/run coverage, oracle controls and pairing.
//! It does not independently execute the compiler or authenticate the runner. Use
//! externally pinned source and manifest hashes when auditing results for publication.

use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fs;

fn is_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn str_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn u64_field(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn bool_field(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn nearly_equal(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1.0e-8_f64.max(a.abs().max(b.abs()) * 1.0e-8)
}

fn push_check(errors: &mut Vec<String>, condition: bool, message: impl Into<String>) {
    if !condition {
        errors.push(message.into());
    }
}

fn read_f64(value: &Value, key: &str) -> Option<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
}

fn manifest_digest(manifest: &[Value]) -> Option<String> {
    // Match the producer's ordered, length-prefixed (task ID, task digest) encoding.
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(manifest.len() as u64).to_be_bytes());
    for task in manifest {
        let id = str_field(task, "task_id")?;
        let digest = str_field(task, "task_digest_blake3")?;
        if !is_hex(digest, 64) {
            return None;
        }
        hasher.update(&(id.len() as u64).to_be_bytes());
        hasher.update(id.as_bytes());
        hasher.update(&(digest.len() as u64).to_be_bytes());
        hasher.update(digest.as_bytes());
    }
    Some(hasher.finalize().to_hex().to_string())
}

fn task_digest_from_manifest(task: &Value) -> Option<String> {
    let id = str_field(task, "task_id")?;
    let description = str_field(task, "description")?;
    let expected_symbol = str_field(task, "expected_symbol")?;
    let reference_source = str_field(task, "reference_source")?;
    let test_source = str_field(task, "test_source")?;
    let expected_test_functions = u64_field(task, "expected_test_functions")?;
    let mutants = task.get("mutant_sources")?.as_array()?;

    // Mirror the producer's length-prefixed content encoding. This avoids
    // ambiguous fixture hashes when any source field contains newlines.
    let mut hasher = blake3::Hasher::new();
    for field in [
        id,
        description,
        expected_symbol,
        reference_source,
        test_source,
    ] {
        hasher.update(&(field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    hasher.update(&expected_test_functions.to_be_bytes());
    hasher.update(&(mutants.len() as u64).to_be_bytes());
    for mutant in mutants {
        let source = mutant.as_str()?;
        hasher.update(&(source.len() as u64).to_be_bytes());
        hasher.update(source.as_bytes());
    }
    Some(hasher.finalize().to_hex().to_string())
}

fn score_success(run: &Value, expected: u64) -> bool {
    expected > 0
        && bool_field(run, "compiled") == Some(true)
        && bool_field(run, "simulated") == Some(false)
        && run.get("runtime_error").is_some_and(Value::is_null)
        && u64_field(run, "tests_passed") == Some(expected)
        && u64_field(run, "tests_failed") == Some(0)
        && u64_field(run, "tests_passed")
            .unwrap_or(0)
            .saturating_add(u64_field(run, "tests_failed").unwrap_or(0))
            == expected
}

fn oracle_complete(run: &Value, expected: u64) -> bool {
    // A nonzero test-process exit is expected when assertions fail. If all
    // declared tests are accounted for, the oracle is complete; runtime_error
    // still prevents the run from being credited as a behavioral success.
    expected > 0
        && bool_field(run, "compiled") == Some(true)
        && bool_field(run, "simulated") == Some(false)
        && u64_field(run, "tests_passed")
            .unwrap_or(0)
            .saturating_add(u64_field(run, "tests_failed").unwrap_or(0))
            == expected
}

fn verify_mode_summary(
    summary: &Value,
    selected_runs: &[&Value],
    enabled: bool,
    errors: &mut Vec<String>,
) {
    let label = if enabled { "gcs_on" } else { "gcs_off" };
    let passed = selected_runs
        .iter()
        .filter(|run| bool_field(run, "behavioral_success") == Some(true))
        .count() as u64;
    let compiles = selected_runs
        .iter()
        .filter(|run| {
            bool_field(run, "compiled") == Some(true) && bool_field(run, "simulated") == Some(false)
        })
        .count() as u64;
    let complete = selected_runs
        .iter()
        .filter(|run| bool_field(run, "oracle_complete") == Some(true))
        .count() as u64;
    let time_total: u64 = selected_runs
        .iter()
        .map(|run| u64_field(run, "agent_elapsed_ms").unwrap_or(0))
        .fold(0_u64, u64::saturating_add);
    let energy_total: f64 = selected_runs
        .iter()
        .map(|run| read_f64(run, "energy").unwrap_or(0.0))
        .sum();
    let count = selected_runs.len() as u64;
    let checks = [
        ("runs", count),
        ("behavioral_successes", passed),
        ("real_compiles", compiles),
        ("complete_oracles", complete),
        ("agent_elapsed_ms_total", time_total),
    ];
    for (key, expected) in checks {
        push_check(
            errors,
            u64_field(summary, key) == Some(expected),
            format!("summary.{label}.{key} must equal {expected}"),
        );
    }

    for (key, expected) in [
        (
            "pass_rate",
            if count == 0 {
                0.0
            } else {
                passed as f64 / count as f64
            },
        ),
        (
            "complete_oracle_rate",
            if count == 0 {
                0.0
            } else {
                complete as f64 / count as f64
            },
        ),
        (
            "agent_elapsed_ms_mean",
            if count == 0 {
                0.0
            } else {
                time_total as f64 / count as f64
            },
        ),
        ("energy_total", energy_total),
        (
            "energy_mean",
            if count == 0 {
                0.0
            } else {
                energy_total / count as f64
            },
        ),
    ] {
        push_check(
            errors,
            read_f64(summary, key).is_some_and(|actual| nearly_equal(actual, expected)),
            format!("summary.{label}.{key} is inconsistent with individual runs"),
        );
    }
}

fn audit(
    report: &Value,
    expected_source_sha: Option<&str>,
    expected_manifest_sha: Option<&str>,
) -> Vec<String> {
    let mut errors = Vec::new();

    push_check(
        &mut errors,
        u64_field(report, "schema_version") == Some(3),
        "schema_version must be 3",
    );
    push_check(
        &mut errors,
        str_field(report, "benchmark") == Some("symthaea_gcs_runtime_ablation"),
        "unexpected benchmark identifier",
    );
    let model_mode = str_field(report, "model_mode").unwrap_or("");
    let model_label = str_field(report, "model_label").unwrap_or("");
    push_check(
        &mut errors,
        matches!(model_mode, "native" | "ollama" | "cloud"),
        "model_mode must be native, ollama, or cloud",
    );
    push_check(
        &mut errors,
        !model_label.trim().is_empty(),
        "model_label must be non-empty",
    );
    let runtime_configuration = report.get("runtime_configuration");
    push_check(
        &mut errors,
        runtime_configuration.and_then(|v| str_field(v, "requested_backend")) == Some(model_mode),
        "runtime_configuration.requested_backend must match model_mode",
    );
    push_check(
        &mut errors,
        runtime_configuration.and_then(|v| str_field(v, "requested_model")) == Some(model_label),
        "runtime_configuration.requested_model must match model_label",
    );
    let source_sha = str_field(report, "source_revision").unwrap_or("");
    push_check(
        &mut errors,
        is_hex(source_sha, 40),
        "source_revision must be a 40-character commit SHA; rerun with GIT_COMMIT_SHA set",
    );
    if let Some(expected) = expected_source_sha {
        push_check(
            &mut errors,
            source_sha.eq_ignore_ascii_case(expected),
            "source_revision does not match --expected-source-sha",
        );
    }

    let Some(manifest) = report.get("task_manifest").and_then(Value::as_array) else {
        return vec!["task_manifest must be an array".to_string()];
    };
    if manifest.is_empty() || manifest.len() > 1_000 {
        return vec!["task_manifest length must be in 1..=1000".to_string()];
    }
    push_check(
        &mut errors,
        !manifest.is_empty(),
        "task_manifest must not be empty",
    );
    let computed_manifest_sha = manifest_digest(manifest);
    push_check(
        &mut errors,
        computed_manifest_sha.as_deref() == str_field(report, "task_manifest_blake3"),
        "task_manifest_blake3 does not match the manifest's ordered task IDs and task digests",
    );
    if let Some(expected) = expected_manifest_sha {
        push_check(
            &mut errors,
            str_field(report, "task_manifest_blake3")
                .is_some_and(|actual| actual.eq_ignore_ascii_case(expected)),
            "task_manifest_blake3 does not match --expected-manifest-blake3",
        );
    }

    let mut expected_counts: HashMap<String, u64> = HashMap::new();
    let mut reference_hashes: HashMap<String, String> = HashMap::new();
    let mut mutant_hashes: HashMap<String, Vec<String>> = HashMap::new();
    let mut task_order: HashMap<String, usize> = HashMap::new();
    for (index, task) in manifest.iter().enumerate() {
        let id = str_field(task, "task_id").unwrap_or("");
        let digest = str_field(task, "task_digest_blake3").unwrap_or("");
        let reference_digest = str_field(task, "reference_source_blake3").unwrap_or("");
        let test_digest = str_field(task, "test_source_blake3").unwrap_or("");
        let reference_source = str_field(task, "reference_source").unwrap_or("");
        let test_source = str_field(task, "test_source").unwrap_or("");
        let mutant_sources = task.get("mutant_sources").and_then(Value::as_array);
        let mut current_mutant_hashes = Vec::new();
        if let Some(mutants) = mutant_sources {
            for mutant in mutants {
                if let Some(source) = mutant.as_str() {
                    current_mutant_hashes
                        .push(blake3::hash(source.as_bytes()).to_hex().to_string());
                } else {
                    errors.push(format!(
                        "task_manifest[{index}] contains a non-string mutant source"
                    ));
                }
            }
        }
        let expected = u64_field(task, "expected_test_functions").unwrap_or(0);
        let recomputed_task_digest = task_digest_from_manifest(task);
        push_check(
            &mut errors,
            !id.is_empty(),
            format!("task_manifest[{index}] has empty task_id"),
        );
        push_check(
            &mut errors,
            is_hex(digest, 64) && is_hex(reference_digest, 64) && is_hex(test_digest, 64),
            format!("task_manifest[{index}] has malformed BLAKE3 digest"),
        );
        push_check(
            &mut errors,
            recomputed_task_digest.as_deref() == Some(digest),
            format!("task_manifest[{index}] task digest does not match its declared content"),
        );
        push_check(
            &mut errors,
            is_hex(reference_digest, 64)
                && reference_digest == blake3::hash(reference_source.as_bytes()).to_hex().as_str(),
            format!(
                "task_manifest[{index}] reference-source digest does not match the included source"
            ),
        );
        push_check(
            &mut errors,
            is_hex(test_digest, 64)
                && test_digest == blake3::hash(test_source.as_bytes()).to_hex().as_str(),
            format!(
                "task_manifest[{index}] test-source digest does not match the included fixture"
            ),
        );
        push_check(
            &mut errors,
            expected > 0,
            format!("task_manifest[{index}] must declare at least one expected test"),
        );
        push_check(
            &mut errors,
            !current_mutant_hashes.is_empty(),
            format!("task_manifest[{index}] must include at least one semantic mutant"),
        );
        push_check(
            &mut errors,
            !expected_counts.contains_key(id),
            format!("duplicate task_id {id} in task_manifest"),
        );
        expected_counts.insert(id.to_string(), expected);
        reference_hashes.insert(id.to_string(), reference_digest.to_string());
        mutant_hashes.insert(id.to_string(), current_mutant_hashes);
        task_order.insert(id.to_string(), index);
    }

    let task_count = expected_counts.len() as u64;
    let repeats = u64_field(report, "repeats").unwrap_or(0);
    push_check(
        &mut errors,
        u64_field(report, "task_count") == Some(task_count),
        "task_count does not match the manifest",
    );
    push_check(
        &mut errors,
        repeats > 0 && repeats <= 20,
        "repeats must be in 1..=20",
    );
    if repeats == 0 || repeats > 20 {
        return errors;
    }

    let runtime = report.get("runtime_configuration").unwrap_or(report);
    push_check(
        &mut errors,
        bool_field(runtime, "external_oracle_real_exec") == Some(true),
        "receipt does not assert real external-oracle execution",
    );
    push_check(
        &mut errors,
        bool_field(runtime, "compiled_with_geodesic_synthesis") == Some(true),
        "receipt was not compiled with geodesic_synthesis",
    );
    push_check(
        &mut errors,
        bool_field(runtime, "agent_internal_real_exec") == Some(false),
        "unexpected internal real-execution configuration for this benchmark",
    );
    push_check(
        &mut errors,
        str_field(report, "model_label").is_some_and(|value| !value.trim().is_empty()),
        "model_label must be present",
    );
    push_check(
        &mut errors,
        matches!(
            str_field(report, "model_mode"),
            Some("native" | "ollama" | "cloud")
        ),
        "model_mode must be native, ollama, or cloud",
    );

    // Verify the preflight controls are present and align with the frozen task bank.
    let empty_source_digest = blake3::hash(b"").to_hex().to_string();
    let controls = report.get("oracle_controls").unwrap_or(report);
    let references = controls.get("reference_controls").and_then(Value::as_array);
    let empty_patches = controls
        .get("empty_patch_controls")
        .and_then(Value::as_array);
    push_check(
        &mut errors,
        bool_field(controls, "all_references_pass") == Some(true),
        "oracle_controls.all_references_pass must be true",
    );
    push_check(
        &mut errors,
        bool_field(controls, "all_empty_patches_fail") == Some(true),
        "oracle_controls.all_empty_patches_fail must be true",
    );
    push_check(
        &mut errors,
        references.is_some_and(|items| items.len() == expected_counts.len()),
        "reference control count does not match task manifest",
    );
    push_check(
        &mut errors,
        empty_patches.is_some_and(|items| items.len() == expected_counts.len()),
        "empty-patch control count does not match task manifest",
    );

    if let Some(items) = references {
        let mut seen = HashMap::new();
        for control in items {
            let id = str_field(control, "task_id").unwrap_or("");
            let expected = expected_counts.get(id).copied().unwrap_or(0);
            push_check(
                &mut errors,
                expected > 0,
                format!("unknown reference-control task {id}"),
            );
            push_check(
                &mut errors,
                seen.insert(id.to_string(), ()).is_none(),
                format!("duplicate reference-control task {id}"),
            );
            push_check(
                &mut errors,
                bool_field(control, "reference_passes") == Some(true)
                    && bool_field(control, "compiled") == Some(true)
                    && bool_field(control, "simulated") == Some(false)
                    && control.get("runtime_error").is_some_and(Value::is_null)
                    && bool_field(control, "oracle_complete") == Some(true)
                    && u64_field(control, "tests_expected") == Some(expected)
                    && u64_field(control, "tests_passed") == Some(expected)
                    && u64_field(control, "tests_failed") == Some(0),
                format!("reference control failed or has incomplete counts for task {id}"),
            );
            push_check(
                &mut errors,
                str_field(control, "reference_blake3").is_some_and(|s| {
                    is_hex(s, 64)
                        && reference_hashes
                            .get(id)
                            .is_some_and(|expected_hash| expected_hash.as_str() == s)
                }),
                format!("reference digest is malformed or differs from manifest for task {id}"),
            );
        }
        for id in expected_counts.keys() {
            push_check(
                &mut errors,
                seen.contains_key(id),
                format!("missing reference control for task {id}"),
            );
        }
    }
    if let Some(items) = empty_patches {
        let mut seen = HashMap::new();
        for control in items {
            let id = str_field(control, "task_id").unwrap_or("");
            push_check(
                &mut errors,
                expected_counts.contains_key(id),
                format!("unknown empty-patch control task {id}"),
            );
            push_check(
                &mut errors,
                seen.insert(id.to_string(), ()).is_none(),
                format!("duplicate empty-patch control task {id}"),
            );
            let compile_errors = control.get("compile_errors").and_then(Value::as_array);
            let has_real_rustc_diagnostic = compile_errors.is_some_and(|errors| {
                errors.iter().any(|error| {
                    error
                        .as_str()
                        .is_some_and(|message| message.trim_start().starts_with("error"))
                })
            });
            push_check(
                &mut errors,
                bool_field(control, "empty_patch_fails") == Some(true)
                    && bool_field(control, "compiled") == Some(false)
                    && bool_field(control, "simulated") == Some(false)
                    && control.get("runtime_error").is_some_and(Value::is_null)
                    && has_real_rustc_diagnostic,
                format!(
                    "empty patch was not rejected by a real, non-simulated rustc compile for task {id}"
                ),
            );
            push_check(
                &mut errors,
                str_field(control, "empty_patch_blake3")
                    .is_some_and(|s| is_hex(s, 64) && s == empty_source_digest.as_str()),
                format!(
                    "empty-patch digest is malformed or differs from the empty source for task {id}"
                ),
            );
        }
        for id in expected_counts.keys() {
            push_check(
                &mut errors,
                seen.contains_key(id),
                format!("missing empty-patch control for task {id}"),
            );
        }
    }

    let expected_mutation_count: usize = mutant_hashes.values().map(Vec::len).sum();
    let mutation_controls = controls.get("mutation_controls").and_then(Value::as_array);
    push_check(
        &mut errors,
        bool_field(controls, "all_mutants_killed") == Some(true),
        "oracle_controls.all_mutants_killed must be true",
    );
    push_check(
        &mut errors,
        mutation_controls.is_some_and(|items| items.len() == expected_mutation_count),
        "mutation control count does not match manifest mutant count",
    );

    if let Some(items) = mutation_controls {
        let mut seen: HashMap<(String, usize), ()> = HashMap::new();
        for (index, control) in items.iter().enumerate() {
            let id = str_field(control, "task_id").unwrap_or("");
            let mutation_index =
                u64_field(control, "mutation_index").and_then(|value| usize::try_from(value).ok());
            let expected_count = expected_counts.get(id).copied().unwrap_or(0);
            let expected_hash = mutation_index.and_then(|mutation_index| {
                mutant_hashes
                    .get(id)
                    .and_then(|hashes| hashes.get(mutation_index))
            });
            push_check(
                &mut errors,
                expected_hash.is_some(),
                format!("mutation_controls[{index}] references an unknown task/mutant"),
            );
            if let Some(mutation_index) = mutation_index {
                push_check(
                    &mut errors,
                    seen.insert((id.to_string(), mutation_index), ()).is_none(),
                    format!("duplicate mutation control for task {id} mutant {mutation_index}"),
                );
            }
            let source_hash = str_field(control, "mutation_source_blake3").unwrap_or("");
            push_check(
                &mut errors,
                expected_hash.is_some_and(|digest| digest.as_str() == source_hash)
                    && is_hex(source_hash, 64),
                format!("mutation control source digest mismatch for task {id}"),
            );
            let passed = u64_field(control, "tests_passed").unwrap_or(0);
            let failed = u64_field(control, "tests_failed").unwrap_or(0);
            // Full test counts plus a non-simulated compile establish that the
            // oracle ran. An assertion failure can set runtime_error because the
            // test binary exits nonzero; count it as killed only if tests failed.
            let complete = expected_count > 0
                && bool_field(control, "compiled") == Some(true)
                && bool_field(control, "simulated") == Some(false)
                && passed.saturating_add(failed) == expected_count;
            let killed = complete && failed > 0;
            push_check(
                &mut errors,
                u64_field(control, "tests_expected") == Some(expected_count),
                format!("mutation control expected test count mismatch for task {id}"),
            );
            push_check(
                &mut errors,
                bool_field(control, "oracle_complete") == Some(complete),
                format!("mutation control oracle completeness mismatch for task {id}"),
            );
            push_check(
                &mut errors,
                bool_field(control, "mutant_killed") == Some(killed) && killed,
                format!(
                    "semantic mutant was not behaviorally killed for task {id} mutant {:?}",
                    mutation_index
                ),
            );
        }
        for (id, hashes) in &mutant_hashes {
            for mutation_index in 0..hashes.len() {
                push_check(
                    &mut errors,
                    seen.contains_key(&(id.clone(), mutation_index)),
                    format!("missing mutation control for task {id} mutant {mutation_index}"),
                );
            }
        }
    }

    let Some(runs) = report.get("runs").and_then(Value::as_array) else {
        return errors
            .into_iter()
            .chain(std::iter::once("runs must be an array".to_string()))
            .collect();
    };
    let expected_runs = task_count.saturating_mul(repeats).saturating_mul(2);
    push_check(
        &mut errors,
        runs.len() as u64 == expected_runs,
        format!("runs has {} entries; expected {expected_runs}", runs.len()),
    );

    let mut pair_slots: HashMap<(String, u64), [Option<&Value>; 2]> = HashMap::new();
    for (index, run) in runs.iter().enumerate() {
        let id = str_field(run, "task_id").unwrap_or("");
        let repeat = u64_field(run, "repeat").unwrap_or(0);
        let expected = expected_counts.get(id).copied().unwrap_or(0);
        let enabled = bool_field(run, "gcs_advisories_enabled");
        push_check(
            &mut errors,
            expected > 0,
            format!("runs[{index}] references unknown task {id}"),
        );
        push_check(
            &mut errors,
            repeat > 0 && repeat <= repeats,
            format!("runs[{index}] has out-of-range repeat"),
        );
        push_check(
            &mut errors,
            u64_field(run, "expected_test_functions") == Some(expected),
            format!("runs[{index}] expected test count differs from manifest"),
        );
        let valid_generation_tiers = run
            .get("generation_tiers")
            .and_then(Value::as_array)
            .is_some_and(|tiers| {
                tiers.iter().all(|tier| {
                    tier.as_str().is_some_and(|name| {
                        matches!(
                            name,
                            "Native" | "Hardware" | "LocalLLM" | "CloudLLM" | "Simulated"
                        )
                    })
                }) && (!tiers.is_empty() || bool_field(run, "behavioral_success") == Some(false))
            });
        push_check(
            &mut errors,
            valid_generation_tiers,
            format!("runs[{index}] has missing or invalid observed generation_tiers"),
        );

        push_check(
            &mut errors,
            str_field(run, "generated_source_blake3").is_some_and(|s| is_hex(s, 64)),
            format!("runs[{index}] has malformed generated-source digest"),
        );

        push_check(
            &mut errors,
            bool_field(run, "compiled").is_some()
                && bool_field(run, "simulated").is_some()
                && u64_field(run, "tests_passed").is_some()
                && u64_field(run, "tests_failed").is_some()
                && run.get("runtime_error").is_some(),
            format!("runs[{index}] is missing compile/oracle status fields"),
        );
        let complete = oracle_complete(run, expected);
        push_check(
            &mut errors,
            bool_field(run, "oracle_complete") == Some(complete),
            format!("runs[{index}] oracle_complete is inconsistent with execution counts"),
        );
        push_check(
            &mut errors,
            bool_field(run, "behavioral_success") == Some(score_success(run, expected)),
            format!("runs[{index}] behavioral_success is inconsistent with fail-closed scoring"),
        );
        push_check(
            &mut errors,
            read_f64(run, "energy").is_some_and(|v| v >= 0.0),
            format!("runs[{index}] energy must be finite and nonnegative"),
        );
        push_check(
            &mut errors,
            u64_field(run, "agent_elapsed_ms").is_some(),
            format!("runs[{index}] missing agent_elapsed_ms"),
        );
        push_check(
            &mut errors,
            u64_field(run, "verification_elapsed_ms").is_some(),
            format!("runs[{index}] missing verification_elapsed_ms"),
        );

        if let Some(enabled) = enabled {
            let slot = if enabled { 1 } else { 0 };
            let entry = pair_slots
                .entry((id.to_string(), repeat))
                .or_insert([None, None]);
            push_check(
                &mut errors,
                entry[slot].is_none(),
                format!(
                    "duplicate {} run for task {id} repeat {repeat}",
                    if enabled { "GCS-on" } else { "GCS-off" }
                ),
            );
            if entry[slot].is_none() {
                entry[slot] = Some(run);
            }
        } else {
            errors.push(format!("runs[{index}] missing gcs_advisories_enabled"));
        }
    }

    let mut on_wins = 0_u64;
    let mut off_wins = 0_u64;
    let mut ties = 0_u64;
    let mut unresolved_pairs = 0_u64;
    for (id, expected) in &expected_counts {
        for repeat in 1..=repeats {
            let Some(slots) = pair_slots.get(&(id.to_string(), repeat)) else {
                errors.push(format!("missing paired runs for task {id} repeat {repeat}"));
                continue;
            };
            let (Some(off), Some(on)) = (slots[0], slots[1]) else {
                errors.push(format!("pair incomplete for task {id} repeat {repeat}"));
                continue;
            };
            let order = str_field(off, "execution_order").unwrap_or("");
            push_check(
                &mut errors,
                order == str_field(on, "execution_order").unwrap_or(""),
                format!("off/on execution order differs for task {id} repeat {repeat}"),
            );
            let idx = task_order.get(id).copied().unwrap_or(0) as u64;
            let expected_order = if (idx + repeat - 1) % 2 == 0 {
                "off_then_on"
            } else {
                "on_then_off"
            };
            push_check(
                &mut errors,
                order == expected_order,
                format!("unexpected pair order for task {id} repeat {repeat}"),
            );
            push_check(
                &mut errors,
                u64_field(off, "expected_test_functions") == Some(*expected)
                    && u64_field(on, "expected_test_functions") == Some(*expected),
                format!(
                    "paired oracle counts disagree with manifest for task {id} repeat {repeat}"
                ),
            );

            // An incomplete oracle makes the pair unresolved, not tied.
            if !oracle_complete(off, *expected) || !oracle_complete(on, *expected) {
                unresolved_pairs += 1;
                continue;
            }

            match (
                bool_field(off, "behavioral_success") == Some(true),
                bool_field(on, "behavioral_success") == Some(true),
            ) {
                (false, true) => on_wins += 1,
                (true, false) => off_wins += 1,
                _ => ties += 1,
            }
        }
    }

    let summary = report.get("summary").unwrap_or(report);
    let off_runs: Vec<&Value> = runs
        .iter()
        .filter(|r| bool_field(r, "gcs_advisories_enabled") == Some(false))
        .collect();
    let on_runs: Vec<&Value> = runs
        .iter()
        .filter(|r| bool_field(r, "gcs_advisories_enabled") == Some(true))
        .collect();
    verify_mode_summary(
        summary.get("gcs_off").unwrap_or(summary),
        &off_runs,
        false,
        &mut errors,
    );
    verify_mode_summary(
        summary.get("gcs_on").unwrap_or(summary),
        &on_runs,
        true,
        &mut errors,
    );

    let paired = summary.get("paired").unwrap_or(summary);
    let pair_count = task_count.saturating_mul(repeats);
    let comparable_pairs = pair_count.saturating_sub(unresolved_pairs);
    let expected_delta = if comparable_pairs == 0 {
        0.0
    } else {
        (on_wins as f64 - off_wins as f64) / comparable_pairs as f64 * 100.0
    };
    push_check(
        &mut errors,
        u64_field(paired, "paired_task_trials") == Some(pair_count),
        "summary.paired.paired_task_trials is inconsistent",
    );
    push_check(
        &mut errors,
        u64_field(paired, "comparable_pairs") == Some(comparable_pairs),
        "summary.paired.comparable_pairs is inconsistent with complete oracle pairs",
    );
    push_check(
        &mut errors,
        u64_field(paired, "unresolved_pairs") == Some(unresolved_pairs),
        "summary.paired.unresolved_pairs is inconsistent with incomplete oracle pairs",
    );
    push_check(
        &mut errors,
        u64_field(paired, "gcs_on_wins") == Some(on_wins),
        "summary.paired.gcs_on_wins is inconsistent with individual pairs",
    );
    push_check(
        &mut errors,
        u64_field(paired, "gcs_off_wins") == Some(off_wins),
        "summary.paired.gcs_off_wins is inconsistent with individual pairs",
    );
    push_check(
        &mut errors,
        u64_field(paired, "ties") == Some(ties),
        "summary.paired.ties is inconsistent with individual pairs",
    );
    push_check(
        &mut errors,
        on_wins + off_wins + ties == comparable_pairs,
        "paired wins/losses/ties do not sum to complete comparable pairs",
    );
    push_check(
        &mut errors,
        read_f64(paired, "net_pass_delta_percentage_points")
            .is_some_and(|actual| nearly_equal(actual, expected_delta)),
        "summary.paired.net_pass_delta_percentage_points is inconsistent",
    );

    errors
}

fn next_bootstrap_u64(state: &mut u64) -> u64 {
    // Fixed, explicitly seeded xorshift64* gives deterministic resampling without
    // pulling a statistical runtime dependency into the audit binary.
    let mut value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    value.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn percentile_sorted(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() || !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let position = probability * (sorted.len().saturating_sub(1) as f64);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Some(sorted[lower] + (sorted[upper] - sorted[lower]) * fraction)
}

fn bootstrap_task_cluster_interval(
    task_deltas: &[f64],
    seed: u64,
    resamples: usize,
) -> Option<(f64, f64, f64)> {
    if task_deltas.is_empty()
        || resamples == 0
        || task_deltas.iter().any(|delta| !delta.is_finite())
    {
        return None;
    }

    // Equal weight per task: repeat-level paired differences are averaged inside
    // each task cluster, then entire task clusters are resampled with replacement.
    let point_estimate = task_deltas.iter().sum::<f64>() / task_deltas.len() as f64;
    let mut state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    let mut samples = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut sum = 0.0;
        for _ in 0..task_deltas.len() {
            let index = (next_bootstrap_u64(&mut state) % task_deltas.len() as u64) as usize;
            sum += task_deltas[index];
        }
        samples.push(sum / task_deltas.len() as f64);
    }
    samples.sort_by(|left, right| left.total_cmp(right));
    Some((
        point_estimate,
        percentile_sorted(&samples, 0.025)?,
        percentile_sorted(&samples, 0.975)?,
    ))
}

fn clustered_bootstrap_summary(report: &Value) -> Option<Value> {
    const BOOTSTRAP_RESAMPLES: usize = 10_000;

    let manifest = report.get("task_manifest")?.as_array()?;
    let runs = report.get("runs")?.as_array()?;
    let repeats = u64_field(report, "repeats")?;
    if repeats == 0 || repeats > 20 {
        return None;
    }

    let manifest_digest = str_field(report, "task_manifest_blake3")?;
    if !is_hex(manifest_digest, 64) {
        return None;
    }
    let seed = u64::from_str_radix(&manifest_digest[..16], 16).ok()?;
    let mut expected_counts = HashMap::new();
    for task in manifest {
        expected_counts.insert(
            str_field(task, "task_id")?.to_string(),
            u64_field(task, "expected_test_functions")?,
        );
    }

    let mut slots: HashMap<(String, u64), [Option<&Value>; 2]> = HashMap::new();
    for run in runs {
        let id = str_field(run, "task_id")?.to_string();
        let repeat = u64_field(run, "repeat")?;
        let enabled = bool_field(run, "gcs_advisories_enabled")?;
        let slot = if enabled { 1 } else { 0 };
        slots.entry((id, repeat)).or_insert([None, None])[slot] = Some(run);
    }

    let mut per_task_deltas = Vec::new();
    let mut comparable_pairs = 0_u64;
    let mut unresolved_pairs = 0_u64;
    let mut tasks_without_comparable_pairs = 0_u64;
    // Iterate the receipt manifest, not HashMap iteration order, so the same
    // frozen receipt yields byte-for-byte stable bootstrap samples across runs.
    for task in manifest {
        let task_id = str_field(task, "task_id")?;
        let expected = u64_field(task, "expected_test_functions")?;
        let mut paired_deltas = Vec::new();
        for repeat in 1..=repeats {
            let Some(pair) = slots.get(&(task_id.to_string(), repeat)) else {
                unresolved_pairs += 1;
                continue;
            };
            let (Some(off), Some(on)) = (pair[0], pair[1]) else {
                unresolved_pairs += 1;
                continue;
            };
            if !oracle_complete(off, expected) || !oracle_complete(on, expected) {
                unresolved_pairs += 1;
                continue;
            }
            let off_success = score_success(off, expected) as u8 as f64;
            let on_success = score_success(on, expected) as u8 as f64;
            paired_deltas.push(on_success - off_success);
            comparable_pairs += 1;
        }
        if paired_deltas.is_empty() {
            tasks_without_comparable_pairs += 1;
        } else {
            per_task_deltas.push(paired_deltas.iter().sum::<f64>() / paired_deltas.len() as f64);
        }
    }

    if per_task_deltas.is_empty() {
        return Some(serde_json::json!({
            "method": "deterministic percentile bootstrap over task-identity clusters",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "task_cluster_count": 0,
            "tasks_without_comparable_pairs": tasks_without_comparable_pairs,
            "comparable_repeat_pairs": comparable_pairs,
            "unresolved_repeat_pairs": unresolved_pairs,
            "status": "not_estimable: no task has a complete off/on pair",
            "warning": "No confidence interval is reported without comparable outcomes."
        }));
    }

    let (point, lower, upper) =
        bootstrap_task_cluster_interval(&per_task_deltas, seed, BOOTSTRAP_RESAMPLES)?;
    Some(serde_json::json!({
        "method": "deterministic percentile bootstrap over task-identity clusters",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "rng_seed_hex": format!("{seed:016x}"),
        "task_cluster_count": per_task_deltas.len(),
        "tasks_without_comparable_pairs": tasks_without_comparable_pairs,
        "comparable_repeat_pairs": comparable_pairs,
        "unresolved_repeat_pairs": unresolved_pairs,
        "mean_task_level_delta_percentage_points": point * 100.0,
        "ci95_lower_percentage_points": lower * 100.0,
        "ci95_upper_percentage_points": upper * 100.0,
        "weighting": "Equal weight per task ID; repeat-level paired outcomes are averaged within task before bootstrapping.",
        "warning": if per_task_deltas.len() < 30 {
            "Small number of task clusters; interval is descriptive and does not justify a field-wide or SOTA claim. Cluster by repository or task family in a larger evaluation."
        } else {
            "Task-identity clusters are assumed independent. For repository-level generalization, cluster by repository or task family where the design supports it."
        }
    }))
}

fn parse_pin_args(args: &[String]) -> Result<(Option<String>, Option<String>), String> {
    let mut expected_source_sha: Option<String> = None;
    let mut expected_manifest_sha: Option<String> = None;
    let mut index = 0;
    while index < args.len() {
        let option = args[index].as_str();
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {option}"))?;
        if value.trim().is_empty() || value.starts_with("--") {
            return Err(format!("Missing or invalid value for {option}"));
        }
        match option {
            "--expected-source-sha" => {
                if expected_source_sha.is_some() {
                    return Err("--expected-source-sha may only be specified once".to_string());
                }
                expected_source_sha = Some(value.clone());
            }
            "--expected-manifest-blake3" => {
                if expected_manifest_sha.is_some() {
                    return Err("--expected-manifest-blake3 may only be specified once".to_string());
                }
                expected_manifest_sha = Some(value.clone());
            }
            _ => return Err(format!("Unknown option: {option}")),
        }
        index += 2;
    }
    Ok((expected_source_sha, expected_manifest_sha))
}

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} RECEIPT.json [--expected-source-sha <40-hex>] [--expected-manifest-blake3 <64-hex>]\n\
         Without external pins, this checks receipt consistency only; use both pins for a stronger audit."
    );
}

fn main() {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "coding_gcs_ablation_verify".into());
    let Some(path) = args.next() else {
        print_usage(&program);
        std::process::exit(2);
    };
    if path == "--help" || path == "-h" {
        print_usage(&program);
        return;
    }

    let rest: Vec<String> = args.collect();
    let (expected_source_sha, expected_manifest_sha) = match parse_pin_args(&rest) {
        Ok(pins) => pins,
        Err(error) => {
            eprintln!("{error}");
            print_usage(&program);
            std::process::exit(2);
        }
    };

    if expected_source_sha
        .as_deref()
        .is_some_and(|s| !is_hex(s, 40))
    {
        eprintln!("--expected-source-sha must be 40 hexadecimal characters");
        std::process::exit(2);
    }
    if expected_manifest_sha
        .as_deref()
        .is_some_and(|s| !is_hex(s, 64))
    {
        eprintln!("--expected-manifest-blake3 must be 64 hexadecimal characters");
        std::process::exit(2);
    }

    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("Cannot read receipt {}: {error}", path);
            std::process::exit(2);
        }
    };
    let report: Value = match serde_json::from_str(&input) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("Invalid JSON receipt: {error}");
            std::process::exit(2);
        }
    };

    let errors = audit(
        &report,
        expected_source_sha.as_deref(),
        expected_manifest_sha.as_deref(),
    );
    if errors.is_empty() {
        println!("RECEIPT CONSISTENT");
        println!("Tasks: {}", u64_field(&report, "task_count").unwrap_or(0));
        println!("Repeats: {}", u64_field(&report, "repeats").unwrap_or(0));
        println!(
            "Paired trials: {}",
            report
                .pointer("/summary/paired/paired_task_trials")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        );
        println!(
            "Task manifest BLAKE3: {}",
            str_field(&report, "task_manifest_blake3").unwrap_or("<missing>")
        );
        println!(
            "External source pin: {}",
            if expected_source_sha.is_some() {
                "matched"
            } else {
                "not supplied"
            }
        );
        println!(
            "External manifest pin: {}",
            if expected_manifest_sha.is_some() {
                "matched"
            } else {
                "not supplied"
            }
        );
        if let Some(statistics) = clustered_bootstrap_summary(&report) {
            let clusters = u64_field(&statistics, "task_cluster_count").unwrap_or(0);
            let comparable = u64_field(&statistics, "comparable_repeat_pairs").unwrap_or(0);
            let unresolved = u64_field(&statistics, "unresolved_repeat_pairs").unwrap_or(0);
            println!(
                "Task-cluster bootstrap: {clusters} task clusters; {comparable} comparable pairs; {unresolved} unresolved pairs."
            );
            if let (Some(estimate), Some(lower), Some(upper)) = (
                read_f64(&statistics, "mean_task_level_delta_percentage_points"),
                read_f64(&statistics, "ci95_lower_percentage_points"),
                read_f64(&statistics, "ci95_upper_percentage_points"),
            ) {
                println!(
                    "Equal-task GCS effect: {estimate:.2} percentage points (descriptive 95% percentile bootstrap CI [{lower:.2}, {upper:.2}])."
                );
            }
            if let Some(warning) = str_field(&statistics, "warning") {
                println!("Statistical caveat: {warning}");
            }
        }
        println!(
            "Caveat: consistency checks are not independent execution or authenticated attestation."
        );
    } else {
        eprintln!("RECEIPT REJECTED: {} problem(s)", errors.len());
        for (index, error) in errors.iter().enumerate() {
            eprintln!("{}. {error}", index + 1);
        }
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_run(enabled: bool, success: bool, elapsed: u64, energy: f64, digest: &str) -> Value {
        serde_json::json!({
            "task_id": "task_a",
            "repeat": 1,
            "gcs_advisories_enabled": enabled,
            "execution_order": "off_then_on",
            "code_written": true,
            "expected_symbol_found": true,
            "generated_source_blake3": digest,
            "compiled": true,
            "simulated": false,
            "expected_test_functions": 1,
            "tests_passed": if success { 1 } else { 0 },
            "tests_failed": if success { 0 } else { 1 },
            "oracle_complete": true,
            "behavioral_success": success,
            "runtime_error": null,
            "compile_errors": vec![],
            "agent_elapsed_ms": elapsed,
            "verification_elapsed_ms": 20,
            "iterations": 1,
            "energy": energy,
            "final_phase": "Testing",
            "generation_tiers": vec!["Native"]
        })
    }

    fn valid_fixture() -> Value {
        let task_id = "task_a";
        let description = "Implement task_a as a true-returning predicate.";
        let expected_symbol = "task_a";
        let reference_source = "pub fn task_a() -> bool { true }";
        let test_source = "#[test] fn task_a_is_true() { assert!(task_a()); }";
        let mutant_sources = vec!["pub fn task_a() -> bool { false }"];
        let mut task_hasher = blake3::Hasher::new();
        for field in [
            task_id,
            description,
            expected_symbol,
            reference_source,
            test_source,
        ] {
            task_hasher.update(&(field.len() as u64).to_be_bytes());
            task_hasher.update(field.as_bytes());
        }
        task_hasher.update(&1_u64.to_be_bytes());
        task_hasher.update(&(mutant_sources.len() as u64).to_be_bytes());
        for mutant in &mutant_sources {
            task_hasher.update(&(mutant.len() as u64).to_be_bytes());
            task_hasher.update(mutant.as_bytes());
        }
        let task_digest = task_hasher.finalize().to_hex().to_string();
        let reference_digest = blake3::hash(reference_source.as_bytes())
            .to_hex()
            .to_string();
        let test_digest = blake3::hash(test_source.as_bytes()).to_hex().to_string();
        let mutant_digest = blake3::hash(mutant_sources[0].as_bytes())
            .to_hex()
            .to_string();
        let manifest = vec![serde_json::json!({
            "task_id": task_id,
            "description": description,
            "expected_symbol": expected_symbol,
            "reference_source": reference_source,
            "test_source": test_source,
            "mutant_sources": mutant_sources,
            "task_digest_blake3": task_digest,
            "expected_test_functions": 1,
            "reference_source_blake3": reference_digest,
            "test_source_blake3": test_digest
        })];
        let manifest_sha = manifest_digest(&manifest).unwrap();
        serde_json::json!({
            "schema_version": 3,
            "benchmark": "symthaea_gcs_runtime_ablation",
            "source_revision": "1".repeat(40),
            "model_mode": "native",
            "model_label": "unit-test model",
            "runtime_configuration": {
                "compiled_with_geodesic_synthesis": true,
                "gcs_toggle": "CodingAgentConfig.enable_gcs_advisories",
                "agent_internal_real_exec": false,
                "external_oracle_real_exec": true,
                "requested_backend": "native",
                "requested_model": "unit-test model"
            },
            "task_count": 1,
            "repeats": 1,
            "task_manifest_blake3": manifest_sha,
            "task_manifest": manifest,
            "oracle_controls": {
                "reference_controls": [{
                    "task_id": "task_a",
                    "reference_blake3": reference_digest,
                    "reference_passes": true,
                    "compiled": true,
                    "simulated": false,
                    "runtime_error": null,
                    "tests_expected": 1,
                    "tests_passed": 1,
                    "tests_failed": 0,
                    "oracle_complete": true
                }],
                "empty_patch_controls": [{
                    "task_id": "task_a",
                    "empty_patch_blake3": blake3::hash(b"").to_hex().to_string(),
                    "empty_patch_fails": true,
                    "compiled": false,
                    "simulated": false,
                    "runtime_error": null,
                    "compile_errors": ["error[E0425]: cannot find function task_a"]
                }],
                "mutation_controls": [{
                    "task_id": "task_a",
                    "mutation_index": 0,
                    "mutation_source_blake3": mutant_digest,
                    "compiled": true,
                    "simulated": false,
                    "tests_expected": 1,
                    "tests_passed": 0,
                    "tests_failed": 1,
                    "runtime_error": null,
                    "oracle_complete": true,
                    "mutant_killed": true,
                    "compile_errors": []
                }],
                "all_references_pass": true,
                "all_empty_patches_fail": true,
                "all_mutants_killed": true
            },
            "summary": {
                "gcs_off": {
                    "gcs_advisories_enabled": false,
                    "runs": 1,
                    "behavioral_successes": 1,
                    "pass_rate": 1.0,
                    "real_compiles": 1,
                    "complete_oracles": 1,
                    "complete_oracle_rate": 1.0,
                    "agent_elapsed_ms_total": 100,
                    "agent_elapsed_ms_mean": 100.0,
                    "energy_total": 1.0,
                    "energy_mean": 1.0
                },
                "gcs_on": {
                    "gcs_advisories_enabled": true,
                    "runs": 1,
                    "behavioral_successes": 0,
                    "pass_rate": 0.0,
                    "real_compiles": 1,
                    "complete_oracles": 1,
                    "complete_oracle_rate": 1.0,
                    "agent_elapsed_ms_total": 120,
                    "agent_elapsed_ms_mean": 120.0,
                    "energy_total": 1.5,
                    "energy_mean": 1.5
                },
                "paired": {
                    "paired_task_trials": 1,
                    "comparable_pairs": 1,
                    "unresolved_pairs": 0,
                    "gcs_on_wins": 0,
                    "gcs_off_wins": 1,
                    "ties": 0,
                    "net_pass_delta_percentage_points": -100.0,
                    "interpretation": "Unit fixture only."
                }
            },
            "runs": [
                fixture_run(false, true, 100, 1.0, &"e".repeat(64)),
                fixture_run(true, false, 120, 1.5, &"f".repeat(64))
            ]
        })
    }

    #[test]
    fn external_pin_cli_rejects_duplicate_or_ambiguous_flags() {
        let source = "a".repeat(40);
        let manifest = "b".repeat(64);
        let valid = vec![
            "--expected-source-sha".to_string(),
            source.clone(),
            "--expected-manifest-blake3".to_string(),
            manifest.clone(),
        ];
        assert_eq!(
            parse_pin_args(&valid).unwrap(),
            (Some(source.clone()), Some(manifest.clone()))
        );
        assert!(
            parse_pin_args(&vec![
                "--expected-source-sha".to_string(),
                source.clone(),
                "--expected-source-sha".to_string(),
                source.clone(),
            ])
            .is_err()
        );
        assert!(
            parse_pin_args(&vec![
                "--expected-manifest-blake3".to_string(),
                manifest.clone(),
                "--expected-manifest-blake3".to_string(),
                manifest.clone(),
            ])
            .is_err()
        );
        assert!(parse_pin_args(&vec!["--expected-source-sha".to_string()]).is_err());
        assert!(
            parse_pin_args(&vec![
                "--expected-source-sha".to_string(),
                "--expected-manifest-blake3".to_string(),
            ])
            .is_err()
        );
        assert!(parse_pin_args(&vec!["--unknown".to_string(), "value".to_string()]).is_err());
    }

    #[test]
    fn digest_format_is_exact() {
        assert!(is_hex(&"a".repeat(64), 64));
        assert!(!is_hex(&"g".repeat(64), 64));
        assert!(!is_hex(&"a".repeat(63), 64));
        assert!(!is_hex(&"a".repeat(65), 64));
    }

    #[test]
    fn manifest_digest_binds_order_and_identifiers() {
        let task = |id: &str, digest: &str| serde_json::json!({"task_id": id, "task_digest_blake3": digest});
        let a = task("a", &"a".repeat(64));
        let b = task("b", &"b".repeat(64));
        let forward = manifest_digest(&[a.clone(), b.clone()]).unwrap();
        let reverse = manifest_digest(&[b.clone(), a.clone()]).unwrap();
        let renamed = manifest_digest(&[task("x", &"a".repeat(64)), b]).unwrap();
        assert_eq!(forward.len(), 64);
        assert_ne!(forward, reverse);
        assert_ne!(forward, renamed);
    }

    #[test]
    fn score_requires_complete_real_behavioral_evidence() {
        let base = serde_json::json!({
            "compiled": true,
            "simulated": false,
            "runtime_error": null,
            "tests_passed": 3,
            "tests_failed": 0
        });
        assert!(score_success(&base, 3));
        assert!(!score_success(&base, 4));
        assert!(!score_success(&base, 0));

        let mut no_compile = base.clone();
        no_compile["compiled"] = Value::Bool(false);
        assert!(!score_success(&no_compile, 3));

        let mut simulated = base.clone();
        simulated["simulated"] = Value::Bool(true);
        assert!(!score_success(&simulated, 3));

        let mut runtime_error = base.clone();
        runtime_error["runtime_error"] = Value::String("runner failed".to_string());
        assert!(!score_success(&runtime_error, 3));

        let mut failed_test = base.clone();
        failed_test["tests_failed"] = Value::from(1);
        failed_test["tests_passed"] = Value::from(2);
        assert!(!score_success(&failed_test, 3));
    }

    #[test]
    fn oracle_completeness_is_not_the_same_as_success() {
        let run = serde_json::json!({
            "compiled": true,
            "simulated": false,
            "runtime_error": "test binary exited nonzero after one assertion failed",
            "tests_passed": 2,
            "tests_failed": 1
        });
        assert!(oracle_complete(&run, 3));
        assert!(!score_success(&run, 3));
    }

    #[test]
    fn valid_receipt_passes_with_external_pins() {
        let report = valid_fixture();
        let manifest_sha = str_field(&report, "task_manifest_blake3").unwrap();
        let expected_source_sha = "1".repeat(40);
        assert!(
            audit(
                &report,
                Some(expected_source_sha.as_str()),
                Some(manifest_sha)
            )
            .is_empty()
        );
    }

    #[test]
    fn task_cluster_bootstrap_is_deterministic_and_bounded() {
        let deltas = [-1.0, 0.0, 1.0];
        let first = bootstrap_task_cluster_interval(&deltas, 0x1234_5678, 2_000).unwrap();
        let second = bootstrap_task_cluster_interval(&deltas, 0x1234_5678, 2_000).unwrap();
        assert_eq!(first, second);
        assert!((first.0 - 0.0).abs() < 1.0e-12);
        assert!(first.1 >= -1.0 && first.1 <= 1.0);
        assert!(first.2 >= -1.0 && first.2 <= 1.0);
        assert!(first.1 <= first.2);
    }

    #[test]
    fn task_cluster_bootstrap_rejects_empty_or_nonfinite_deltas() {
        assert!(bootstrap_task_cluster_interval(&[], 1, 1_000).is_none());
        assert!(bootstrap_task_cluster_interval(&[f64::NAN], 1, 1_000).is_none());
        assert!(bootstrap_task_cluster_interval(&[0.0], 1, 0).is_none());
    }

    #[test]
    fn valid_receipt_gets_explicit_descriptive_cluster_interval() {
        let report = valid_fixture();
        let summary = clustered_bootstrap_summary(&report).unwrap();
        assert_eq!(u64_field(&summary, "task_cluster_count"), Some(1));
        assert_eq!(
            read_f64(&summary, "mean_task_level_delta_percentage_points"),
            Some(-100.0)
        );
        assert_eq!(
            read_f64(&summary, "ci95_lower_percentage_points"),
            Some(-100.0)
        );
        assert_eq!(
            read_f64(&summary, "ci95_upper_percentage_points"),
            Some(-100.0)
        );
        assert!(
            str_field(&summary, "warning")
                .unwrap()
                .contains("Small number of task clusters")
        );
    }

    #[test]
    fn inconsistent_model_identity_metadata_is_rejected() {
        let mut report = valid_fixture();
        report["runtime_configuration"]["requested_model"] =
            Value::String("different-model".to_string());
        let errors = audit(&report, None, None);
        assert!(errors.iter().any(|error| error.contains("requested_model")));
    }

    #[test]
    fn unknown_model_mode_is_rejected() {
        let mut report = valid_fixture();
        report["model_mode"] = Value::String("mystery".to_string());
        let errors = audit(&report, None, None);
        assert!(errors.iter().any(|error| error.contains("model_mode")));
    }

    #[test]
    fn missing_generation_tiers_are_rejected() {
        let mut report = valid_fixture();
        report["runs"][0]
            .as_object_mut()
            .unwrap()
            .remove("generation_tiers");
        let errors = audit(&report, None, None);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("generation_tiers"))
        );
    }

    #[test]
    fn unknown_generation_tier_is_rejected() {
        let mut report = valid_fixture();
        report["runs"][0]["generation_tiers"] = serde_json::json!(["ImaginaryLLM"]);
        let errors = audit(&report, None, None);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("generation_tiers"))
        );
    }

    #[test]
    fn empty_generation_tiers_are_rejected_when_run_claims_success() {
        let mut report = valid_fixture();
        report["runs"][0]["generation_tiers"] = serde_json::json!([]);
        report["runs"][0]["behavioral_success"] = Value::Bool(true);
        let errors = audit(&report, None, None);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("generation_tiers"))
        );
    }

    #[test]
    fn empty_generation_tiers_can_explain_a_failed_no_generation_run() {
        let mut report = valid_fixture();
        report["runs"][0]["generation_tiers"] = serde_json::json!([]);
        report["runs"][0]["behavioral_success"] = Value::Bool(false);
        let errors = audit(&report, None, None);
        assert!(
            !errors
                .iter()
                .any(|error| error.contains("generation_tiers"))
        );
    }

    #[test]
    fn tampered_paired_summary_is_rejected() {
        let mut report = valid_fixture();
        report["summary"]["paired"]["gcs_on_wins"] = Value::from(1);
        let errors = audit(&report, None, None);
        assert!(errors.iter().any(|error| error.contains("gcs_on_wins")));
    }

    #[test]
    fn omitted_run_is_rejected() {
        let mut report = valid_fixture();
        report["runs"].as_array_mut().unwrap().remove(0);
        let errors = audit(&report, None, None);
        assert!(errors.iter().any(|error| error.contains("expected 2")));
        assert!(errors.iter().any(|error| error.contains("pair incomplete")));
    }

    #[test]
    fn killed_mutant_can_have_nonzero_test_process_exit() {
        let mut report = valid_fixture();
        report["oracle_controls"]["mutation_controls"][0]["runtime_error"] =
            Value::String("test binary exited nonzero after assertion failure".to_string());
        assert!(audit(&report, None, None).is_empty());
    }

    #[test]
    fn surviving_mutant_is_rejected() {
        let mut report = valid_fixture();
        report["oracle_controls"]["mutation_controls"][0]["tests_failed"] = Value::from(0);
        report["oracle_controls"]["mutation_controls"][0]["tests_passed"] = Value::from(1);
        report["oracle_controls"]["mutation_controls"][0]["mutant_killed"] = Value::Bool(false);
        let errors = audit(&report, None, None);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("not behaviorally killed"))
        );
    }

    #[test]
    fn incomplete_pair_is_unresolved_not_a_tie() {
        let mut report = valid_fixture();
        report["runs"][1]["compiled"] = Value::Bool(false);
        report["runs"][1]["tests_passed"] = Value::from(0);
        report["runs"][1]["tests_failed"] = Value::from(0);
        report["runs"][1]["oracle_complete"] = Value::Bool(false);
        report["summary"]["gcs_on"]["real_compiles"] = Value::from(0);
        report["summary"]["gcs_on"]["complete_oracles"] = Value::from(0);
        report["summary"]["gcs_on"]["complete_oracle_rate"] = Value::from(0.0);
        report["summary"]["paired"]["comparable_pairs"] = Value::from(0);
        report["summary"]["paired"]["unresolved_pairs"] = Value::from(1);
        report["summary"]["paired"]["gcs_off_wins"] = Value::from(0);
        report["summary"]["paired"]["ties"] = Value::from(0);
        report["summary"]["paired"]["net_pass_delta_percentage_points"] = Value::from(0.0);
        assert!(audit(&report, None, None).is_empty());
    }

    #[test]
    fn unexpected_external_source_pin_is_rejected() {
        let report = valid_fixture();
        assert!(
            audit(&report, Some(&"2".repeat(40)), None)
                .iter()
                .any(|error| error.contains("--expected-source-sha"))
        );
    }

    #[test]
    fn tampered_test_fixture_is_rejected_even_before_external_pin_check() {
        let mut report = valid_fixture();
        report["task_manifest"][0]["test_source"] =
            Value::String("#[test] fn task_a_is_true() { assert!(!task_a()); }".to_string());
        let errors = audit(&report, None, None);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("test-source digest"))
        );
        assert!(errors.iter().any(|error| error.contains("task digest")));
    }

    #[test]
    fn declared_oracle_count_is_digest_bound_and_must_match_observed_runs() {
        let mut report = valid_fixture();
        report["task_manifest"][0]["expected_test_functions"] = Value::from(2);
        let errors = audit(&report, None, None);
        assert!(errors.iter().any(|error| error.contains("task digest")));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("expected test count"))
        );
    }

    #[test]
    fn tampered_task_digest_is_rejected_even_if_outer_digest_is_recomputed() {
        let mut report = valid_fixture();
        report["task_manifest"][0]["task_digest_blake3"] = Value::String("d".repeat(64));
        let new_outer = manifest_digest(report["task_manifest"].as_array().unwrap()).unwrap();
        report["task_manifest_blake3"] = Value::String(new_outer);
        let errors = audit(&report, None, None);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("task digest does not match"))
        );
    }
}
