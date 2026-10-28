//! Deterministic task-set generation and post-run reproduction.
//!
//! This module provides a small, versioned synthetic constructor family for
//! protocol qualification. It is not a qualified scientific benchmark: the
//! built-in worlds are synthetic fixtures and the manifest's pinned generator
//! revision still has to identify the exact source snapshot used for a run.

use crate::{
    GenerationManifestV1, compute_artifact_commitment, sha256_hex, validate_manifest_bytes,
    verify_reveal,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const GENERATOR_CONFIG_SCHEMA: &str =
    "symthaea.science_bench_refresh_generator_config.v1";
const TASK_GENERATOR_DOMAIN: &[u8] = b"symthaea.science_bench_refresh.taskgen.v1\0";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorConfigV1 {
    pub schema: String,
    pub config_version: String,
    pub recipes: Vec<FamilyRecipeV1>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyRecipeV1 {
    pub family_id: String,
    pub kind: String,
    pub instruction: String,
    pub samples_per_task: u64,
    pub x_start: f64,
    pub x_step: f64,
    pub noise_amplitude: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RegenerationReportV1 {
    pub schema: String,
    pub generation_id: String,
    pub manifest_sha256: String,
    pub generator_config_sha256: String,
    pub seed_commitment_matches: bool,
    pub task_set_commitment_matches: bool,
    pub regenerated_task_set_sha256: String,
    pub committed_task_set_sha256: String,
    pub regenerated_bytes: u64,
    pub committed_bytes: u64,
    pub byte_exact_match: bool,
    pub task_set_contract_valid: bool,
    pub verified: bool,
    pub issues: Vec<String>,
}

fn decode_manifest(bytes: &[u8]) -> Result<GenerationManifestV1, String> {
    serde_json::from_slice(bytes)
        .map_err(|error| format!("manifest does not match the strict v1 shape: {error}"))
}

fn decode_config(bytes: &[u8]) -> Result<GeneratorConfigV1, String> {
    let config: GeneratorConfigV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("generator config does not match the strict v1 shape: {error}"))?;
    if config.schema != GENERATOR_CONFIG_SCHEMA {
        return Err(format!(
            "generator config schema must be {GENERATOR_CONFIG_SCHEMA}"
        ));
    }
    if config.config_version.trim().is_empty() {
        return Err("generator config_version must not be empty".to_owned());
    }
    if config.recipes.is_empty() {
        return Err("generator config must declare at least one recipe".to_owned());
    }

    let mut seen = BTreeSet::new();
    for recipe in &config.recipes {
        if recipe.family_id.trim().is_empty() {
            return Err("generator recipe family_id must not be empty".to_owned());
        }
        if !seen.insert(recipe.family_id.as_str()) {
            return Err(format!(
                "duplicate family recipe in generator config: {}",
                recipe.family_id
            ));
        }
        if recipe.instruction.trim().is_empty() {
            return Err(format!(
                "recipe {} must provide a visible instruction",
                recipe.family_id
            ));
        }
        if recipe.samples_per_task < 3 || recipe.samples_per_task > 10_000 {
            return Err(format!(
                "recipe {} samples_per_task must be in 3..=10000",
                recipe.family_id
            ));
        }
        if !recipe.x_start.is_finite()
            || recipe.x_start.abs() > 1.0e9
            || !recipe.x_step.is_finite()
            || recipe.x_step <= 0.0
            || recipe.x_step > 1.0e9
        {
            return Err(format!(
                "recipe {} requires finite x_start and positive finite x_step",
                recipe.family_id
            ));
        }
        if !recipe.noise_amplitude.is_finite()
            || recipe.noise_amplitude < 0.0
            || recipe.noise_amplitude > 1.0e9
        {
            return Err(format!(
                "recipe {} noise_amplitude must be finite and non-negative",
                recipe.family_id
            ));
        }
        let allowed = match recipe.kind.as_str() {
            "semantic_invariance" => recipe.noise_amplitude == 0.0,
            "sampling_noise_shift" => recipe.noise_amplitude > 0.0,
            "confounding_null" => recipe.noise_amplitude > 0.0,
            "identifiability_abstention" => recipe.noise_amplitude > 0.0,
            _ => false,
        };
        if !allowed {
            return Err(format!(
                "recipe {} has unsupported kind/noise combination",
                recipe.family_id
            ));
        }
    }
    Ok(config)
}

fn seed_word(
    seed: &[u8],
    generation_id: &str,
    family_id: &str,
    local_index: u64,
    field: &str,
) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(TASK_GENERATOR_DOMAIN);
    let index_bytes = local_index.to_be_bytes();
    let parts: [&[u8]; 5] = [
        seed,
        generation_id.as_bytes(),
        family_id.as_bytes(),
        &index_bytes,
        field.as_bytes(),
    ];
    for part in parts {
        hasher.update(u64::try_from(part.len()).unwrap_or(u64::MAX).to_be_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    u64::from_be_bytes([
        digest[0], digest[1], digest[2], digest[3],
        digest[4], digest[5], digest[6], digest[7],
    ])
}

fn unit_fraction(seed: &[u8], generation_id: &str, family_id: &str, index: u64, field: &str) -> f64 {
    let raw = seed_word(seed, generation_id, family_id, index, field) % 1_000_001;
    raw as f64 / 1_000_000.0
}

fn coefficient(seed: &[u8], generation_id: &str, family_id: &str, index: u64, field: &str) -> f64 {
    let raw = seed_word(seed, generation_id, family_id, index, field) % 7;
    f64::from(i32::try_from(raw).unwrap_or(0) - 3)
}

fn noise(
    seed: &[u8],
    generation_id: &str,
    family_id: &str,
    task_index: u64,
    sample_index: u64,
    amplitude: f64,
) -> f64 {
    if amplitude == 0.0 {
        return 0.0;
    }
    let raw = seed_word(
        seed,
        generation_id,
        family_id,
        task_index
            .wrapping_mul(1_000_003)
            .wrapping_add(sample_index),
        "observation_noise",
    ) % 2_000_001;
    ((raw as f64 / 1_000_000.0) - 1.0) * amplitude
}

fn task_id(generation_id: &str, family_id: &str, index: u64) -> String {
    format!("{generation_id}::{family_id}::{index:06}")
}

fn generate_one_task(
    generation_id: &str,
    seed: &[u8],
    recipe: &FamilyRecipeV1,
    index: u64,
) -> Result<Value, String> {
    let slope =
        1.0 + unit_fraction(seed, generation_id, &recipe.family_id, index, "slope") * 4.0;
    let intercept = coefficient(seed, generation_id, &recipe.family_id, index, "intercept");
    let instruction = recipe.instruction.clone();
    let count = usize::try_from(recipe.samples_per_task)
        .map_err(|_| "samples_per_task does not fit memory index".to_owned())?;
    let id = task_id(generation_id, &recipe.family_id, index);

    let (policy_input, evaluator_only) = match recipe.kind.as_str() {
        "semantic_invariance" => {
            let scales = [0.25_f64, 0.5, 2.0, 4.0];
            let scale_index = usize::try_from(seed_word(
                seed,
                generation_id,
                &recipe.family_id,
                index,
                "scale",
            ) % 4)
            .unwrap_or(0);
            let scale = scales[scale_index];
            let offset = coefficient(seed, generation_id, &recipe.family_id, index, "coordinate_offset");
            let observations: Vec<Value> = (0..count)
                .map(|sample| {
                    let latent_x = recipe.x_start
                        + f64::from(u32::try_from(sample).unwrap_or(0)) * recipe.x_step;
                    let visible_u = scale * latent_x + offset;
                    let y = slope * latent_x + intercept;
                    json!({"u": visible_u, "y": y})
                })
                .collect();
            let visible_slope = slope / scale;
            let visible_intercept = intercept - slope * offset / scale;
            (
                json!({
                    "instruction": instruction,
                    "coordinate_semantics": format!("visible coordinate u = {scale} * x + ({offset})"),
                    "observations": observations,
                    "declared_measurement_noise": 0.0
                }),
                json!({
                    "target_kind": "unique_law_under_declared_reparameterization",
                    "latent_law": {"slope": slope, "intercept": intercept},
                    "visible_coordinate_law": {"slope": visible_slope, "intercept": visible_intercept},
                    "inverse_coordinate_transform": {"scale": scale, "offset": offset}
                }),
            )
        }
        "sampling_noise_shift" => {
            let observations: Vec<Value> = (0..count)
                .map(|sample| {
                    let x = recipe.x_start
                        + f64::from(u32::try_from(sample).unwrap_or(0)) * recipe.x_step;
                    let err = noise(
                        seed, generation_id, &recipe.family_id, index,
                        u64::try_from(sample).unwrap_or(0), recipe.noise_amplitude,
                    );
                    json!({"x": x, "y": slope * x + intercept + err})
                })
                .collect();
            (
                json!({
                    "instruction": instruction,
                    "observations": observations,
                    "declared_measurement_noise_amplitude": recipe.noise_amplitude,
                    "sample_budget": recipe.samples_per_task
                }),
                json!({
                    "target_kind": "noisy_linear_law",
                    "law": {"slope": slope, "intercept": intercept},
                    "noise_amplitude": recipe.noise_amplitude
                }),
            )
        }
        "confounding_null" => {
            let observations: Vec<Value> = (0..count)
                .map(|sample| {
                    let z = recipe.x_start + f64::from(u32::try_from(sample).unwrap_or(0)) * recipe.x_step;
                    let x_err = noise(
                        seed, generation_id, &recipe.family_id, index,
                        u64::try_from(sample).unwrap_or(0).wrapping_mul(2), recipe.noise_amplitude,
                    );
                    let y_err = noise(
                        seed, generation_id, &recipe.family_id, index,
                        u64::try_from(sample).unwrap_or(0).wrapping_mul(2).wrapping_add(1),
                        recipe.noise_amplitude,
                    );
                    json!({"z": z, "x": z + x_err, "y": z + y_err})
                })
                .collect();
            (
                json!({
                    "instruction": instruction,
                    "observations": observations,
                    "causal_warning": "an observed association may be explained by a common cause",
                    "sample_budget": recipe.samples_per_task
                }),
                json!({
                    "target_kind": "no_direct_effect",
                    "common_cause": "z",
                    "direct_effect": 0.0,
                    "noise_amplitude": recipe.noise_amplitude
                }),
            )
        }
        "identifiability_abstention" => {
            let narrow_step = recipe.x_step / 100.0;
            let observations: Vec<Value> = (0..count)
                .map(|sample| {
                    let x = recipe.x_start
                        + f64::from(u32::try_from(sample).unwrap_or(0)) * narrow_step;
                    let err = noise(
                        seed, generation_id, &recipe.family_id, index,
                        u64::try_from(sample).unwrap_or(0), recipe.noise_amplitude,
                    );
                    json!({"x": x, "y": slope * x + intercept + err})
                })
                .collect();
            (
                json!({
                    "instruction": instruction,
                    "observations": observations,
                    "observed_x_span": (count.saturating_sub(1) as f64) * narrow_step,
                    "declared_measurement_noise_amplitude": recipe.noise_amplitude,
                    "required_response_mode": "report_identifiability_or_abstain"
                }),
                json!({
                    "target_kind": "not_identifiable_from_declared_sample",
                    "one_compatible_law": {"kind": "linear", "slope": slope, "intercept": intercept},
                    "claim_limit": "the observations do not uniquely establish a global law"
                }),
            )
        }
        other => return Err(format!("unsupported recipe kind: {other}")),
    };

    Ok(json!({
        "task_id": id,
        "family_id": recipe.family_id,
        "policy_input": policy_input,
        "evaluator_only": evaluator_only
    }))
}

/// Generate the exact canonical task-set bytes from a valid sealed manifest,
/// a revealed seed, and the exact frozen generator-config bytes.
///
/// This checks the raw config digest and frozen protocol/seed commitments.
/// It does not by itself prove that this module's code matches the Git commit
/// named in the manifest; the caller must bind the binary/source revision.
pub fn generate_task_set_bytes(
    manifest_bytes: &[u8],
    seed_bytes: &[u8],
    generator_config_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let validation = validate_manifest_bytes(manifest_bytes)?;
    if !validation.valid {
        return Err(format!("manifest is invalid: {}", validation.issues.join("; ")));
    }
    let manifest = decode_manifest(manifest_bytes)?;
    if manifest.purpose != "qualified_run" || manifest.lane != "sealed_refresh" {
        return Err("task regeneration requires a qualified sealed_refresh manifest".to_owned());
    }
    if seed_bytes.len() < 32 {
        return Err("seed must contain at least 32 bytes".to_owned());
    }
    let actual_config_sha = sha256_hex(generator_config_bytes);
    if actual_config_sha != manifest.generator.configuration_sha256 {
        return Err("generator config bytes do not match generator.configuration_sha256".to_owned());
    }
    let computed_seed_commitment = compute_artifact_commitment(manifest_bytes, "seed", seed_bytes)?;
    if computed_seed_commitment != manifest.seed_commitment_sha256 {
        return Err("seed bytes do not match seed_commitment_sha256".to_owned());
    }

    let config = decode_config(generator_config_bytes)?;
    let mut recipe_by_family = BTreeMap::new();
    for recipe in config.recipes {
        recipe_by_family.insert(recipe.family_id.clone(), recipe);
    }
    let manifest_families: BTreeSet<&str> = manifest
        .task_family_mixture
        .iter()
        .map(|family| family.family_id.as_str())
        .collect();
    let config_families: BTreeSet<&str> = recipe_by_family.keys().map(String::as_str).collect();
    if manifest_families != config_families {
        return Err("generator config recipe families must equal the manifest family set".to_owned());
    }

    let mut tasks = Vec::new();
    for family in &manifest.task_family_mixture {
        let recipe = recipe_by_family
            .get(&family.family_id)
            .ok_or_else(|| format!("missing recipe for family {}", family.family_id))?;
        for index in 0..family.planned_task_count {
            tasks.push(generate_one_task(
                &manifest.generation_id,
                seed_bytes,
                recipe,
                index,
            )?);
        }
    }
    let task_set = json!({
        "schema": crate::TASK_SET_SCHEMA,
        "generation_id": manifest.generation_id,
        "tasks": tasks
    });
    let mut bytes = serde_json::to_vec_pretty(&task_set)
        .map_err(|error| format!("cannot serialize generated task set: {error}"))?;
    bytes.push(b'\n');
    let task_set_issues = crate::validate_task_set_contract(&manifest, &bytes);
    if !task_set_issues.is_empty() {
        return Err(format!(
            "generator produced a task set that violates its contract: {}",
            task_set_issues.join("; ")
        ));
    }
    Ok(bytes)
}

/// Regenerate and compare the exact revealed task-set bytes. A successful
/// report proves deterministic byte reproduction under the pinned inputs,
/// not secrecy, execution isolation, or physical scientific validity.
pub fn verify_regeneration(
    manifest_bytes: &[u8],
    seed_bytes: &[u8],
    generator_config_bytes: &[u8],
    committed_task_set_bytes: &[u8],
) -> Result<RegenerationReportV1, String> {
    let regenerated = generate_task_set_bytes(manifest_bytes, seed_bytes, generator_config_bytes)?;
    let manifest = decode_manifest(manifest_bytes)?;
    let reveal_report = verify_reveal(manifest_bytes, seed_bytes, committed_task_set_bytes)?;
    let generated_hash = sha256_hex(&regenerated);
    let committed_hash = sha256_hex(committed_task_set_bytes);
    let byte_exact_match = regenerated == committed_task_set_bytes;
    let task_set_commitment_matches = reveal_report.task_set_commitment_matches;
    let task_set_contract_valid = reveal_report.task_set_contract_valid;
    let seed_commitment_matches = reveal_report.seed_commitment_matches;
    let mut issues = reveal_report.issues;
    if !byte_exact_match {
        issues.push(
            "regenerated task-set bytes differ from the revealed committed bytes".to_owned(),
        );
    }
    let verified = seed_commitment_matches
        && task_set_commitment_matches
        && task_set_contract_valid
        && byte_exact_match;

    Ok(RegenerationReportV1 {
        schema: "symthaea.science_bench_refresh_regeneration_report.v1".to_owned(),
        generation_id: manifest.generation_id,
        manifest_sha256: sha256_hex(manifest_bytes),
        generator_config_sha256: sha256_hex(generator_config_bytes),
        seed_commitment_matches,
        task_set_commitment_matches,
        regenerated_task_set_sha256: generated_hash,
        committed_task_set_sha256: committed_hash,
        regenerated_bytes: u64::try_from(regenerated.len())
            .map_err(|_| "regenerated bytes exceed u64".to_owned())?,
        committed_bytes: u64::try_from(committed_task_set_bytes.len())
            .map_err(|_| "committed bytes exceed u64".to_owned())?,
        byte_exact_match,
        task_set_contract_valid,
        verified,
        issues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compute_protocol_commitment, sha256_hex};
    use serde_json::json;

    fn demo_config() -> Vec<u8> {
        serde_json::to_vec_pretty(&json!({
            "schema": GENERATOR_CONFIG_SCHEMA,
            "config_version": "fixture-v1",
            "recipes": [
                {"family_id":"semantic_invariance","kind":"semantic_invariance","instruction":"Recover the relationship invariant under a known coordinate transform.","samples_per_task":5,"x_start":-2.0,"x_step":1.0,"noise_amplitude":0.0},
                {"family_id":"sampling_noise_shift","kind":"sampling_noise_shift","instruction":"Recover the law despite declared measurement noise.","samples_per_task":6,"x_start":-1.0,"x_step":0.5,"noise_amplitude":0.05},
                {"family_id":"confounding_null","kind":"confounding_null","instruction":"Decide whether x has a direct effect on y or a common cause explains the association.","samples_per_task":8,"x_start":-2.0,"x_step":0.5,"noise_amplitude":0.02},
                {"family_id":"identifiability_abstention","kind":"identifiability_abstention","instruction":"State the identifiable equivalence class or abstain.","samples_per_task":5,"x_start":0.0,"x_step":0.25,"noise_amplitude":0.03}
            ]
        }))
        .expect("config serializes")
    }

    fn manifest_for_test(config_bytes: &[u8], seed: &[u8]) -> Vec<u8> {
        let mut manifest = json!({
            "schema": crate::MANIFEST_SCHEMA,
            "generation_id": "generator-test-0001",
            "purpose": "qualified_run",
            "lane": "sealed_refresh",
            "generator": {
                "repository": "https://example.invalid/generator",
                "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "configuration_sha256": sha256_hex(config_bytes)
            },
            "evaluator": {
                "version": "synthetic-evaluator-v1",
                "source_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "oracle_separation": "independent_evaluator_boundary_verified"
            },
            "protocol_commitment_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "seed_commitment_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "task_set_commitment_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "expected_task_count": 8,
            "task_family_mixture": [
                {"family_id":"semantic_invariance","weight":0.25,"planned_task_count":2},
                {"family_id":"sampling_noise_shift","weight":0.25,"planned_task_count":2},
                {"family_id":"confounding_null","weight":0.25,"planned_task_count":2},
                {"family_id":"identifiability_abstention","weight":0.25,"planned_task_count":2}
            ],
            "policy_freeze": {
                "subject_id":"synthetic-frozen-policy",
                "source_or_policy_sha256":"1111111111111111111111111111111111111111111111111111111111111111",
                "prompt_sha256":"2222222222222222222222222222222222222222222222222222222222222222",
                "configuration_sha256":"3333333333333333333333333333333333333333333333333333333333333333",
                "tool_manifest_sha256":"4444444444444444444444444444444444444444444444444444444444444444"
            },
            "budget": {"max_observations":1000,"max_experiments":100,"max_tool_calls":20,"max_total_tokens":50000,"max_wall_clock_seconds":3600},
            "analysis_plan_sha256":"5555555555555555555555555555555555555555555555555555555555555555",
            "post_run_reveal_policy":"reveal_seed_and_task_bytes_after_submission"
        });
        let first = serde_json::to_vec(&manifest).unwrap();
        let protocol = compute_protocol_commitment(&first).unwrap();
        manifest["protocol_commitment_sha256"] = json!(protocol);
        let second = serde_json::to_vec(&manifest).unwrap();
        manifest["seed_commitment_sha256"] = json!(
            compute_artifact_commitment(&second, "seed", seed).unwrap()
        );
        serde_json::to_vec_pretty(&manifest).unwrap()
    }

    fn complete_manifest_and_tasks() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let config = demo_config();
        let seed = vec![0x6d; 32];
        let manifest = manifest_for_test(&config, &seed);
        let generated = generate_task_set_bytes(&manifest, &seed, &config).unwrap();
        let provisional = serde_json::from_slice::<Value>(&manifest).unwrap();
        let _ = provisional;
        (manifest, seed, config, generated)
    }

    #[test]
    fn verify_regeneration_accepts_exact_replay_and_task_commitment() {
        let (manifest, seed, config, generated) = complete_manifest_and_tasks();
        let final_manifest =
            manifest_with_task_commitment(&manifest, &seed, &config, &generated);
        let report = verify_regeneration(&final_manifest, &seed, &config, &generated).unwrap();
        assert!(report.verified, "{:?}", report.issues);
        assert!(report.seed_commitment_matches);
        assert!(report.task_set_commitment_matches);
        assert!(report.task_set_contract_valid);
        assert!(report.byte_exact_match);
        assert_eq!(report.regenerated_task_set_sha256, report.committed_task_set_sha256);
    }

    #[test]
    fn different_seed_cannot_reproduce_committed_task_set() {
        let (manifest, seed, config, generated) = complete_manifest_and_tasks();
        let final_manifest =
            manifest_with_task_commitment(&manifest, &seed, &config, &generated);
        let mut different_seed = seed.clone();
        different_seed[0] ^= 1;
        let error = verify_regeneration(&final_manifest, &different_seed, &config, &generated)
            .expect_err("wrong seed must fail before task comparison");
        assert!(error.contains("seed bytes do not match"));
    }

    #[test]
    fn generator_schema_fixture_is_valid_json_and_matches_version() {
        let schema: Value = serde_json::from_str(include_str!(
            "../schema/generator-config.schema.json"
        ))
        .expect("generator schema JSON");
        assert_eq!(
            schema["properties"]["schema"]["const"].as_str(),
            Some(GENERATOR_CONFIG_SCHEMA)
        );
        assert_eq!(sha256_hex(&demo_config()).len(), 64);
    }

    #[test]
    fn generated_task_set_is_byte_deterministic() {
        let (manifest, seed, config, generated) = complete_manifest_and_tasks();
        let repeated = generate_task_set_bytes(&manifest, &seed, &config).unwrap();
        assert_eq!(generated, repeated);
    }

    #[test]
    fn generated_task_count_and_family_counts_match_manifest() {
        let (manifest, seed, config, generated) = complete_manifest_and_tasks();
        let report = crate::verify_reveal(
            &manifest_with_task_commitment(&manifest, &seed, &config, &generated),
            &seed,
            &generated,
        );
        assert!(report.is_ok());
        let task_set: Value = serde_json::from_slice(&generated).unwrap();
        assert_eq!(task_set["tasks"].as_array().unwrap().len(), 8);
    }

    fn manifest_with_task_commitment(
        manifest: &[u8],
        seed: &[u8],
        config: &[u8],
        task_set: &[u8],
    ) -> Vec<u8> {
        let mut value: Value = serde_json::from_slice(manifest).unwrap();
        value["task_set_commitment_sha256"] = json!(
            compute_artifact_commitment(manifest, "task-set", task_set).unwrap()
        );
        let _ = seed;
        let _ = config;
        serde_json::to_vec_pretty(&value).unwrap()
    }

    #[test]
    fn changed_seed_changes_task_set() {
        let (manifest, seed, config, generated) = complete_manifest_and_tasks();
        let mut changed = seed.clone();
        changed[0] ^= 0x01;
        let error = generate_task_set_bytes(&manifest, &changed, &config)
            .expect_err("a seed different from the committed seed must be rejected");
        assert!(error.contains("seed bytes do not match"));
        assert!(!generated.is_empty());
    }

    #[test]
    fn altered_config_bytes_are_rejected() {
        let (manifest, seed, config, _) = complete_manifest_and_tasks();
        let mut changed = config.clone();
        changed.push(b' ');
        let error = generate_task_set_bytes(&manifest, &seed, &changed)
            .expect_err("any config-byte mutation must break the pinned digest");
        assert!(error.contains("generator config bytes do not match"));
    }

    #[test]
    fn unknown_recipe_family_is_rejected() {
        let config = serde_json::to_vec(&json!({
            "schema": GENERATOR_CONFIG_SCHEMA,
            "config_version": "bad-v1",
            "recipes": [{"family_id":"unknown","kind":"semantic_invariance","instruction":"x","samples_per_task":5,"x_start":0.0,"x_step":1.0,"noise_amplitude":0.0}]
        })).unwrap();
        let seed = vec![0x6d; 32];
        let manifest = manifest_for_test(&demo_config(), &seed);
        let mut value: Value = serde_json::from_slice(&manifest).unwrap();
        value["generator"]["configuration_sha256"] = json!(sha256_hex(&config));
        let raw = serde_json::to_vec(&value).unwrap();
        let protocol = compute_protocol_commitment(&raw).unwrap();
        value["protocol_commitment_sha256"] = json!(protocol);
        let final_bytes = serde_json::to_vec(&value).unwrap();
        value["seed_commitment_sha256"] = json!(
            compute_artifact_commitment(&final_bytes, "seed", &seed).unwrap()
        );
        let raw = serde_json::to_vec_pretty(&value).unwrap();
        let error = generate_task_set_bytes(&raw, &seed, &config)
            .expect_err("the manifest and config family sets differ");
        assert!(error.contains("families must equal"));
    }
}
