// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic factorized scenario expansion for the synthetic mobility benchmark.
//!
//! This creates synthetic benchmark cases from the reviewed v2 control corpus. It is not a
//! simulator of human physiology, a preference model, or evidence about real wearers.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::preference_drift_benchmark::{
    validate_preference_drift_fixture, PreferenceDriftFixtureFindingV1,
};

/// Schema version for generator configuration and manifest receipts.
pub const PREFERENCE_DRIFT_GENERATOR_SCHEMA_VERSION_V1: u64 = 1;
/// Generator identity recorded in generated fixture provenance.
pub const PREFERENCE_DRIFT_GENERATOR_ID_V1: &str = "factorized_seeded_generator_v1";
const MAX_GENERATED_SCENARIOS_V1: usize = 50_000;
const MAX_WEARER_SESSIONS_V1: usize = 2_000;
const MAX_REPETITIONS_PER_FAMILY_V1: usize = 100;

/// Configuration for deterministic expansion of reviewed scenario templates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceDriftGeneratorConfigV1 {
    /// Root seed for reproducible synthetic axis variation.
    pub seed: u64,
    /// Number of synthetic wearer/session groups generated.
    pub wearer_session_count: usize,
    /// Disjoint wearer/session groups reserved for development.
    pub development_wearer_session_count: usize,
    /// Repetitions for each held-out template/wearer pair; development cases are emitted once.
    pub held_out_repetitions_per_family: usize,
}

impl Default for PreferenceDriftGeneratorConfigV1 {
    fn default() -> Self {
        Self {
            seed: 20_261_018,
            wearer_session_count: 50,
            development_wearer_session_count: 10,
            held_out_repetitions_per_family: 3,
        }
    }
}

/// Auditable manifest for one generated suite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceDriftGenerationManifestV1 {
    /// Manifest schema version.
    pub schema_version: u64,
    /// Version label stored in the generated fixture.
    pub fixture_version: String,
    /// Version of the scenario generator.
    pub generator_version: String,
    /// Root seed for all generated choices.
    pub seed: u64,
    /// Number of distinct synthetic wearer/session groups.
    pub wearer_session_count: usize,
    /// Number of development-only wearer/session groups.
    pub development_wearer_session_count: usize,
    /// Number of held-out-only wearer/session groups.
    pub held_out_wearer_session_count: usize,
    /// Number of source scenario-family groups held out from development.
    pub held_out_family_count: usize,
    /// Number of generated development cases.
    pub development_scenario_count: usize,
    /// Number of generated held-out cases.
    pub held_out_scenario_count: usize,
    /// Total generated scenario count.
    pub scenario_count: usize,
    /// Number of candidates in the shared synthetic catalog.
    pub candidate_count: usize,
    /// Scenario counts keyed by scenario-family ID, in canonical key order.
    pub scenario_counts_by_family: BTreeMap<String, usize>,
    /// Split assignment keyed by scenario-family ID, in canonical key order.
    pub scenario_family_splits: BTreeMap<String, String>,
    /// Counts for each categorical generator axis, keyed by axis then value.
    pub variation_counts_by_axis: BTreeMap<String, BTreeMap<String, usize>>,
    /// Episodes with an applied synthetic preference reversal and reversed evaluation order.
    pub preference_reversal_scenario_count: usize,
    /// BLAKE3 digest of the exact serialized fixture bytes, including trailing newline.
    pub fixture_blake3_hex: String,
    /// Explicit limitation on interpretation.
    pub claim_ceiling: String,
}

/// Generated synthetic suite plus its detached content-digest manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedPreferenceDriftSuiteV1 {
    /// Pretty-printed generated fixture, with one trailing newline.
    pub fixture_json: String,
    /// Counts, seed, split provenance, and content digest.
    pub manifest: PreferenceDriftGenerationManifestV1,
}

#[derive(Debug, Clone)]
struct SplitMix64V1(u64);

impl SplitMix64V1 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn bounded(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        (self.next_u64() % upper_exclusive as u64) as usize
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let other = self.bounded(index + 1);
            values.swap(index, other);
        }
    }
}

fn finding(path: impl Into<String>, reason: impl Into<String>) -> Vec<PreferenceDriftFixtureFindingV1> {
    vec![PreferenceDriftFixtureFindingV1 {
        path: path.into(),
        reason: reason.into(),
    }]
}

fn checked_scenario_count(
    development_wearers: usize,
    development_template_count: usize,
    held_out_wearers: usize,
    held_out_template_count: usize,
    held_out_repetitions: usize,
) -> Result<(usize, usize, usize), Vec<PreferenceDriftFixtureFindingV1>> {
    let development_count = development_wearers
        .checked_mul(development_template_count)
        .ok_or_else(|| finding("$.config", "development scenario count overflow"))?;
    let held_out_count = held_out_wearers
        .checked_mul(held_out_template_count)
        .and_then(|count| count.checked_mul(held_out_repetitions))
        .ok_or_else(|| finding("$.config", "held-out scenario count overflow"))?;
    let total = development_count
        .checked_add(held_out_count)
        .ok_or_else(|| finding("$.config", "total scenario count overflow"))?;
    if total > MAX_GENERATED_SCENARIOS_V1 {
        return Err(finding(
            "$.config",
            format!(
                "generated scenario count {total} exceeds safety limit {}",
                MAX_GENERATED_SCENARIOS_V1
            ),
        ));
    }
    Ok((development_count, held_out_count, total))
}

fn apply_axis_variation(
    mut scenario: Value,
    template_id: &str,
    wearer_session_index: usize,
    repetition: usize,
    rng: &mut SplitMix64V1,
) -> Result<Value, Vec<PreferenceDriftFixtureFindingV1>> {
    let scenario_id = format!("GEN-{template_id}-W{wearer_session_index:04}-R{repetition:03}");
    let source_context_id = scenario
        .get("current_context_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| finding("$.scenarios.current_context_id", "template lacks current_context_id"))?;
    let task_id = scenario
        .get("task_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| finding("$.scenarios.task_id", "template lacks task_id"))?;

    scenario["scenario_id"] = json!(scenario_id);
    scenario["synthetic_wearer_id"] = json!(format!("synthetic-session-{wearer_session_index:04}"));

    let base_step = scenario
        .get("current_step")
        .and_then(Value::as_u64)
        .ok_or_else(|| finding("$.scenarios.current_step", "template lacks current_step"))?;
    let time_offset = rng.bounded(6) as u64;
    let current_step = base_step.checked_add(time_offset).ok_or_else(|| {
        finding("$.scenarios.current_step", "current-step variation overflow")
    })?;
    scenario["current_step"] = json!(current_step);
    scenario["preference_max_age_steps"] = json!(rng.bounded(4) as u64);

    let query_budget = rng.bounded(3) as u64;
    let queries_used = if query_budget == 0 {
        0
    } else {
        rng.bounded(query_budget as usize + 1) as u64
    };
    scenario["query_budget_total"] = json!(query_budget);
    scenario["queries_already_used"] = json!(queries_used);

    const TERRAINS: [&str; 4] = ["level", "ramp", "uneven", "stairs"];
    const FATIGUE_STATES: [&str; 3] = ["rested", "mild_fatigue", "fatigued"];
    const PAYLOAD_CLASSES: [&str; 3] = ["light", "standard", "loaded"];
    const TIME_WINDOWS: [&str; 3] = ["early", "middle", "late"];
    let terrain = TERRAINS[rng.bounded(TERRAINS.len())];
    let fatigue = FATIGUE_STATES[rng.bounded(FATIGUE_STATES.len())];
    let payload_class = PAYLOAD_CLASSES[rng.bounded(PAYLOAD_CLASSES.len())];
    let time_window = TIME_WINDOWS[rng.bounded(TIME_WINDOWS.len())];
    let current_context_id = format!(
        "{}__gen_{}_{}_{}_{}",
        source_context_id, terrain, fatigue, payload_class, time_window
    );
    {
        let context = scenario
            .get_mut("current_context")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| finding("$.scenarios.current_context", "template lacks context object"))?;
        context.insert("terrain".to_owned(), json!(terrain));
        context.insert("fatigue".to_owned(), json!(fatigue));
        context.insert("payload_class".to_owned(), json!(payload_class));
        context.insert("time_window".to_owned(), json!(time_window));
    }
    scenario["current_context_id"] = json!(current_context_id);

    let has_current_context_observation = scenario
        .get("preference_observations")
        .and_then(Value::as_array)
        .is_some_and(|observations| {
            observations.iter().any(|observation| {
                observation.get("task_id").and_then(Value::as_str) == Some(task_id.as_str())
                    && observation.get("context_id").and_then(Value::as_str)
                        == Some(source_context_id.as_str())
            })
        });
    let has_current_pairwise_feedback = scenario
        .get("preference_observations")
        .and_then(Value::as_array)
        .is_some_and(|observations| {
            observations.iter().any(|observation| {
                observation.get("task_id").and_then(Value::as_str) == Some(task_id.as_str())
                    && observation.get("context_id").and_then(Value::as_str)
                        == Some(source_context_id.as_str())
                    && observation.get("preferred_candidate_id").and_then(Value::as_str).is_some()
                    && observation.get("rejected_candidate_id").and_then(Value::as_str).is_some()
            })
        });

    // Context-derived synthetic feedback is never treated as real wearer data: its provenance
    // is rewritten explicitly, and a historical-only episode keeps the source context IDs.
    let feedback_mode = if !has_current_context_observation {
        "historical_only"
    } else if has_current_pairwise_feedback {
        match rng.bounded(3) {
            0 => "historical_only",
            1 => "fresh_synthetic_context",
            _ => "reversed_synthetic_context",
        }
    } else if rng.bounded(2) == 0 {
        "historical_only"
    } else {
        "fresh_synthetic_context"
    };

    let mut reversal_applied = false;
    let observations = scenario
        .get_mut("preference_observations")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| {
            finding(
                "$.scenarios.preference_observations",
                "template lacks observation array",
            )
        })?;
    for (index, observation) in observations.iter_mut().enumerate() {
        let source_observation_id = observation
            .get("observation_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("observation-{index}"));
        observation["observation_id"] = json!(format!("{scenario_id}-O{index:02}"));
        let current_task_and_context =
            observation.get("task_id").and_then(Value::as_str) == Some(task_id.as_str())
                && observation.get("context_id").and_then(Value::as_str)
                    == Some(source_context_id.as_str());
        if current_task_and_context && feedback_mode != "historical_only" {
            let source_observation_source = observation
                .get("source")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| "unspecified_synthetic_source".to_owned());
            observation["source_observation_id"] = json!(source_observation_id);
            observation["source_observation_source"] = json!(source_observation_source);
            observation["context_id"] = json!(current_context_id);
            observation["source"] = json!("explicit_synthetic_generated_ordinal_feedback");

            if feedback_mode == "reversed_synthetic_context" {
                let preferred = observation.get("preferred_candidate_id").cloned();
                let rejected = observation.get("rejected_candidate_id").cloned();
                if let (Some(preferred), Some(rejected)) = (preferred, rejected) {
                    observation["preferred_candidate_id"] = rejected;
                    observation["rejected_candidate_id"] = preferred;
                    reversal_applied = true;
                }
            }
        }
    }
    rng.shuffle(observations);

    // Evaluation labels are generated only from the declared synthetic template oracle. For a
    // scripted preference reversal, reverse that template order before any policy output is scored.
    scenario["evaluation_oracle"]["context_id"] = json!(current_context_id);
    if reversal_applied
        && scenario.pointer("/evaluation_oracle/status").and_then(Value::as_str)
            == Some("ranking")
    {
        let mut preferred_order = scenario
            .pointer("/evaluation_oracle/preferred_order")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| finding("$.evaluation_oracle.preferred_order", "ranking lacks order"))?;
        let acceptable_count = scenario
            .pointer("/evaluation_oracle/acceptable_candidate_ids")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or_default();
        preferred_order.reverse();
        let acceptable = preferred_order
            .iter()
            .take(acceptable_count)
            .cloned()
            .collect::<Vec<_>>();
        scenario["evaluation_oracle"]["preferred_order"] = Value::Array(preferred_order);
        scenario["evaluation_oracle"]["acceptable_candidate_ids"] = Value::Array(acceptable);
    }

    scenario["generator_variation"] = json!({
        "terrain": terrain,
        "fatigue": fatigue,
        "payload_class": payload_class,
        "time_window": time_window,
        "feedback_mode": feedback_mode,
        "preference_reversed": reversal_applied,
        "source_scenario_template_id": template_id
    });

    let eligible = scenario
        .get_mut("eligible_candidate_ids")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| {
            finding(
                "$.scenarios.eligible_candidate_ids",
                "template lacks eligible-candidate array",
            )
        })?;
    rng.shuffle(eligible);

    Ok(scenario)
}

/// Generate a reproducible suite from a validated v2 hand-authored control fixture.
///
/// Development and held-out wearer/session IDs are disjoint. Whole scenario families retain
/// their declared split. The generator varies feedback age, freshness horizon, query budget,
/// candidate ordering, observation ordering, wearer/session identity, and repetition. The
/// generator's pseudo-random choices are deterministic and are not cryptographic.
pub fn generate_preference_drift_suite(
    source_fixture_json: &str,
    config: &PreferenceDriftGeneratorConfigV1,
) -> Result<GeneratedPreferenceDriftSuiteV1, Vec<PreferenceDriftFixtureFindingV1>> {
    validate_preference_drift_fixture(source_fixture_json)?;

    if config.wearer_session_count < 2
        || config.wearer_session_count > MAX_WEARER_SESSIONS_V1
        || config.development_wearer_session_count == 0
        || config.development_wearer_session_count >= config.wearer_session_count
        || config.held_out_repetitions_per_family == 0
        || config.held_out_repetitions_per_family > MAX_REPETITIONS_PER_FAMILY_V1
    {
        return Err(finding(
            "$.config",
            format!(
                "require 2..={MAX_WEARER_SESSIONS_V1} wearer/session groups, at least one but fewer than all groups reserved for development, and 1..={MAX_REPETITIONS_PER_FAMILY_V1} held-out repetitions"
            ),
        ));
    }

    let mut root: Value = serde_json::from_str(source_fixture_json)
        .map_err(|error| finding("$", format!("invalid source fixture JSON: {error}")))?;
    let source_scenarios = root
        .get("scenarios")
        .and_then(Value::as_array)
        .ok_or_else(|| finding("$.scenarios", "source scenario array is missing"))?
        .clone();
    let development_templates: Vec<Value> = source_scenarios
        .iter()
        .filter(|scenario| scenario.get("split").and_then(Value::as_str) == Some("development"))
        .cloned()
        .collect();
    let held_out_templates: Vec<Value> = source_scenarios
        .iter()
        .filter(|scenario| scenario.get("split").and_then(Value::as_str) == Some("held_out"))
        .cloned()
        .collect();
    if development_templates.is_empty() || held_out_templates.is_empty() {
        return Err(finding(
            "$.scenarios",
            "source fixture needs both development and held-out templates",
        ));
    }

    let held_out_families: BTreeSet<String> = held_out_templates
        .iter()
        .filter_map(|scenario| scenario.get("scenario_family").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    if held_out_families.is_empty() {
        return Err(finding(
            "$.scenarios",
            "held-out templates need at least one named scenario family",
        ));
    }

    let held_out_wearer_count =
        config.wearer_session_count - config.development_wearer_session_count;
    let (development_count, held_out_count, _) = checked_scenario_count(
        config.development_wearer_session_count,
        development_templates.len(),
        held_out_wearer_count,
        held_out_templates.len(),
        config.held_out_repetitions_per_family,
    )?;

    let mut rng = SplitMix64V1::new(config.seed);
    let mut generated_scenarios = Vec::with_capacity(development_count + held_out_count);

    // Development sessions see only development templates.
    for wearer_index in 0..config.development_wearer_session_count {
        for template in &development_templates {
            let template_id = template
                .get("scenario_id")
                .and_then(Value::as_str)
                .unwrap_or("development-template");
            generated_scenarios.push(apply_axis_variation(
                template.clone(),
                template_id,
                wearer_index,
                0,
                &mut rng,
            )?);
        }
    }

    // Held-out sessions see only held-out families. No wearer/session identity crosses splits.
    for wearer_index in config.development_wearer_session_count..config.wearer_session_count {
        for template in &held_out_templates {
            let template_id = template
                .get("scenario_id")
                .and_then(Value::as_str)
                .unwrap_or("heldout-template");
            for repetition in 0..config.held_out_repetitions_per_family {
                generated_scenarios.push(apply_axis_variation(
                    template.clone(),
                    template_id,
                    wearer_index,
                    repetition,
                    &mut rng,
                )?);
            }
        }
    }

    let mut scenario_counts_by_family = BTreeMap::<String, usize>::new();
    let mut scenario_family_splits = BTreeMap::<String, String>::new();
    let mut variation_counts_by_axis =
        BTreeMap::<String, BTreeMap<String, usize>>::new();
    let mut preference_reversal_scenario_count = 0usize;
    for scenario in &generated_scenarios {
        let family = scenario
            .get("scenario_family")
            .and_then(Value::as_str)
            .ok_or_else(|| finding("$.scenarios.scenario_family", "generated scenario lacks family ID"))?
            .to_owned();
        let split = scenario
            .get("split")
            .and_then(Value::as_str)
            .ok_or_else(|| finding("$.scenarios.split", "generated scenario lacks split"))?
            .to_owned();
        *scenario_counts_by_family.entry(family.clone()).or_default() += 1;
        let variation = scenario
            .get("generator_variation")
            .and_then(Value::as_object)
            .ok_or_else(|| finding("$.scenarios.generator_variation", "generated scenario lacks axis receipt"))?;
        for axis in ["terrain", "fatigue", "payload_class", "time_window", "feedback_mode"] {
            let value = variation
                .get(axis)
                .and_then(Value::as_str)
                .ok_or_else(|| finding("$.scenarios.generator_variation", format!("missing axis {axis}")))?;
            *variation_counts_by_axis
                .entry(axis.to_owned())
                .or_default()
                .entry(value.to_owned())
                .or_default() += 1;
        }
        if variation
            .get("preference_reversed")
            .and_then(Value::as_bool)
            == Some(true)
        {
            preference_reversal_scenario_count += 1;
        }
        if let Some(previous) = scenario_family_splits.insert(family.clone(), split.clone()) {
            if previous != split {
                return Err(finding(
                    "$.scenarios.scenario_family",
                    format!("scenario family {family} was assigned to both {previous} and {split}"),
                ));
            }
        }
    }

    root["fixture_version"] = json!("0.3.0");
    root["generation"] = json!({
        "method": PREFERENCE_DRIFT_GENERATOR_ID_V1,
        "generator_version": "0.1.0",
        "random_generator_used": true,
        "empirical_data_used": false,
        "seed": config.seed,
        "config": config,
        "note": "Deterministic seeded synthetic factorization of hand-authored control templates. Wearer/session splits are disjoint and scenario families remain within one split. No empirical or human-subject data is used."
    });
    root["scenarios"] = Value::Array(generated_scenarios);

    let candidate_count = root
        .get("candidate_catalog")
        .and_then(Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| finding("$.candidate_catalog", "candidate catalog is missing"))?;
    let generated_root_json = serde_json::to_string_pretty(&root)
        .map_err(|error| finding("$", format!("cannot serialize generated fixture: {error}")))?;
    let fixture_json = format!("{generated_root_json}\n");

    // Validate the generated suite with the same contract used by the replay and baseline tools.
    let validated = validate_preference_drift_fixture(&fixture_json)?;
    if validated.development_scenario_count != development_count
        || validated.held_out_scenario_count != held_out_count
        || validated.scenario_count != development_count + held_out_count
    {
        return Err(finding(
            "$.scenarios",
            "generated split counts disagree with the manifest calculation",
        ));
    }

    let fixture_blake3_hex = blake3::hash(fixture_json.as_bytes()).to_hex().to_string();
    let manifest = PreferenceDriftGenerationManifestV1 {
        schema_version: PREFERENCE_DRIFT_GENERATOR_SCHEMA_VERSION_V1,
        fixture_version: "0.3.0".to_owned(),
        generator_version: "0.1.0".to_owned(),
        seed: config.seed,
        wearer_session_count: config.wearer_session_count,
        development_wearer_session_count: config.development_wearer_session_count,
        held_out_wearer_session_count: held_out_wearer_count,
        held_out_family_count: held_out_families.len(),
        development_scenario_count: development_count,
        held_out_scenario_count: held_out_count,
        scenario_count: validated.scenario_count,
        candidate_count,
        scenario_counts_by_family,
        scenario_family_splits,
        variation_counts_by_axis,
        preference_reversal_scenario_count,
        fixture_blake3_hex,
        claim_ceiling: "Synthetic scenario-generator determinism, structural validity, split isolation, and policy-contract stress only. No real-wearer preference, safety, clinical, or deployment claim.".to_owned(),
    };

    Ok(GeneratedPreferenceDriftSuiteV1 {
        fixture_json,
        manifest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_FIXTURE: &str =
        include_str!("../../../../docs/research/open-mobility/fixtures/preference-drift-scenarios-v3.json");

    #[test]
    fn generated_suite_is_reproducible_and_digest_matches_exact_bytes() {
        let config = PreferenceDriftGeneratorConfigV1::default();
        let first = generate_preference_drift_suite(SOURCE_FIXTURE, &config).unwrap();
        let second = generate_preference_drift_suite(SOURCE_FIXTURE, &config).unwrap();
        assert_eq!(first.fixture_json, second.fixture_json);
        assert_eq!(first.manifest, second.manifest);
        assert_eq!(
            blake3::hash(first.fixture_json.as_bytes()).to_hex().to_string(),
            first.manifest.fixture_blake3_hex
        );
    }

    #[test]
    fn default_suite_has_disjoint_wearers_and_grouped_family_splits() {
        let suite = generate_preference_drift_suite(
            SOURCE_FIXTURE,
            &PreferenceDriftGeneratorConfigV1::default(),
        )
        .unwrap();
        let fixture: Value = serde_json::from_str(&suite.fixture_json).unwrap();
        let scenarios = fixture["scenarios"].as_array().unwrap();
        assert_eq!(suite.manifest.wearer_session_count, 50);
        assert_eq!(suite.manifest.development_wearer_session_count, 10);
        assert_eq!(suite.manifest.held_out_wearer_session_count, 40);
        assert_eq!(suite.manifest.held_out_family_count, 12);
        assert_eq!(suite.manifest.development_scenario_count, 20);
        assert_eq!(suite.manifest.held_out_scenario_count, 1440);
        assert_eq!(suite.manifest.scenario_count, 1460);
        assert_eq!(scenarios.len(), 1460);
        assert_eq!(
            suite.manifest.scenario_counts_by_family.get("steady_context_explicit_preference"),
            Some(&20)
        );
        assert_eq!(suite.manifest.scenario_family_splits.len(), 11);
        assert_eq!(
            suite.manifest.scenario_counts_by_family.values().sum::<usize>(),
            1460
        );
        assert_eq!(
            suite.manifest.variation_counts_by_axis["terrain"].values().sum::<usize>(),
            1460
        );
        assert_eq!(
            suite.manifest.variation_counts_by_axis["fatigue"].len(),
            3
        );
        assert_eq!(
            suite.manifest.variation_counts_by_axis["payload_class"].len(),
            3
        );
        assert_eq!(
            suite.manifest.variation_counts_by_axis["time_window"].len(),
            3
        );
        assert!(suite.manifest.preference_reversal_scenario_count > 0);

        let mut split_by_family = BTreeMap::<String, String>::new();
        let mut split_by_wearer = BTreeMap::<String, String>::new();
        for scenario in scenarios {
            let split = scenario["split"].as_str().unwrap().to_owned();
            let family = scenario["scenario_family"].as_str().unwrap().to_owned();
            let wearer = scenario["synthetic_wearer_id"].as_str().unwrap().to_owned();
            if let Some(previous) = split_by_family.insert(family.clone(), split.clone()) {
                assert_eq!(previous, split, "scenario family crosses splits: {family}");
            }
            if let Some(previous) = split_by_wearer.insert(wearer.clone(), split.clone()) {
                assert_eq!(previous, split, "wearer/session crosses splits: {wearer}");
            }
        }
        assert_eq!(split_by_wearer.len(), 50);
    }

    #[test]
    fn generated_context_and_feedback_provenance_are_explicit() {
        let suite = generate_preference_drift_suite(
            SOURCE_FIXTURE,
            &PreferenceDriftGeneratorConfigV1::default(),
        )
        .unwrap();
        let fixture: Value = serde_json::from_str(&suite.fixture_json).unwrap();
        let scenarios = fixture["scenarios"].as_array().unwrap();
        let mut applied_reversals = 0usize;

        for scenario in scenarios {
            assert_eq!(
                scenario["current_context_id"],
                scenario["evaluation_oracle"]["context_id"],
                "evaluation context should identify the generated current context"
            );
            let variation = scenario
                .get("generator_variation")
                .and_then(Value::as_object)
                .expect("factorized axis receipt");
            assert!(variation.get("terrain").and_then(Value::as_str).is_some());
            assert!(variation.get("fatigue").and_then(Value::as_str).is_some());
            assert!(variation.get("payload_class").and_then(Value::as_str).is_some());
            assert!(variation.get("time_window").and_then(Value::as_str).is_some());

            if variation.get("preference_reversed").and_then(Value::as_bool) == Some(true) {
                applied_reversals += 1;
                assert_eq!(
                    variation.get("feedback_mode").and_then(Value::as_str),
                    Some("reversed_synthetic_context")
                );
            }

            for observation in scenario["preference_observations"].as_array().unwrap() {
                if observation.get("source_observation_id").is_some() {
                    assert!(
                        observation
                            .get("source")
                            .and_then(Value::as_str)
                            .is_some_and(|source| source.starts_with("explicit_synthetic_"))
                    );
                    assert!(
                        observation
                            .get("source_observation_source")
                            .and_then(Value::as_str)
                            .is_some_and(|source| source.starts_with("explicit_synthetic_"))
                    );
                }
            }
        }

        assert_eq!(
            applied_reversals,
            suite.manifest.preference_reversal_scenario_count
        );
    }

    #[test]
    fn context_no_reversal_and_inclusive_freshness_boundary_are_explicit_controls() {
        let fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        let scenarios = fixture["scenarios"].as_array().unwrap();

        let no_reversal = scenarios
            .iter()
            .find(|scenario| {
                scenario["scenario_id"] == "HOLD-CONTEXT-NO-REVERSAL-001"
            })
            .expect("context-change-without-reversal control");
        let no_reversal_assessment =
            crate::preference_drift_benchmark::replay_preference_episode(no_reversal).unwrap();
        assert_eq!(
            no_reversal_assessment.action,
            crate::preference_drift_benchmark::PreferenceReplayActionV1::Recommend
        );
        assert_eq!(
            no_reversal_assessment.recommended_candidate_id.as_deref(),
            Some("flex")
        );
        assert_eq!(no_reversal_assessment.stale_or_mismatched_observation_count, 1);

        let freshness_boundary = scenarios
            .iter()
            .find(|scenario| {
                scenario["scenario_id"] == "HOLD-FRESHNESS-BOUNDARY-001"
            })
            .expect("inclusive freshness-boundary control");
        let boundary_assessment =
            crate::preference_drift_benchmark::replay_preference_episode(freshness_boundary)
                .unwrap();
        assert_eq!(
            boundary_assessment.action,
            crate::preference_drift_benchmark::PreferenceReplayActionV1::Recommend
        );
        assert_eq!(
            boundary_assessment.recommended_candidate_id.as_deref(),
            Some("stable")
        );
        assert_eq!(boundary_assessment.used_observation_count, 2);
        assert_eq!(boundary_assessment.context_matched_observation_count, 3);
        assert_eq!(boundary_assessment.expired_preference_observation_count, 1);
    }

    #[test]
    fn generator_rejects_invalid_split_and_runaway_size_configuration() {
        let invalid_split = PreferenceDriftGeneratorConfigV1 {
            development_wearer_session_count: 50,
            ..PreferenceDriftGeneratorConfigV1::default()
        };
        assert!(generate_preference_drift_suite(SOURCE_FIXTURE, &invalid_split).is_err());

        let oversized = PreferenceDriftGeneratorConfigV1 {
            wearer_session_count: MAX_WEARER_SESSIONS_V1,
            development_wearer_session_count: MAX_WEARER_SESSIONS_V1 / 2,
            held_out_repetitions_per_family: MAX_REPETITIONS_PER_FAMILY_V1,
            ..PreferenceDriftGeneratorConfigV1::default()
        };
        assert!(generate_preference_drift_suite(SOURCE_FIXTURE, &oversized).is_err());
    }
}
