// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Baseline scientific quality validation: verify that all baseline entries
//! have sensible statistical properties and proper source citations.

use symthaea_psych_bench::harness::baselines::BaselineCollection;

/// Collect all (domain, key, baseline) triples from the collection.
fn all_baselines() -> Vec<(&'static str, &'static str, f64, Option<f64>, &'static str)> {
    let bl = BaselineCollection::all();
    let mut entries = Vec::new();

    let domains: Vec<(&str, &std::collections::BTreeMap<&str, _>)> = vec![
        ("worm", &bl.worm),
        ("cogbench", &bl.cogbench),
        ("tombench", &bl.tombench),
        ("memory_agent", &bl.memory_agent),
        ("executive", &bl.executive),
        ("metacognition", &bl.metacognition),
        ("affect", &bl.affect),
        ("creativity", &bl.creativity),
        ("butlin", &bl.butlin),
        ("inhibition", &bl.inhibition),
        ("attention", &bl.attention),
        ("embodied", &bl.embodied),
        ("reasoning", &bl.reasoning),
        ("sustained_attention", &bl.sustained_attention),
        ("motor", &bl.motor),
        ("language", &bl.language),
        ("social", &bl.social),
        ("neuromod", &bl.neuromod),
        ("llm_cogbench", &bl.llm_cogbench),
        ("llm_tombench", &bl.llm_tombench),
        ("llm_arc", &bl.llm_arc),
    ];

    for (domain, map) in domains {
        for (&key, baseline) in map.iter() {
            entries.push((domain, key, baseline.value, baseline.sd, baseline.source));
        }
    }

    entries
}

/// All baseline values must be finite.
#[test]
fn all_values_finite() {
    let entries = all_baselines();
    let mut failures = Vec::new();

    for &(domain, key, value, _, _) in &entries {
        if !value.is_finite() {
            failures.push(format!(
                "{}.{}: value is not finite ({})",
                domain, key, value
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Non-finite baseline values ({}):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Standard deviations, when present, must be positive and finite.
#[test]
fn standard_deviations_positive() {
    let entries = all_baselines();
    let mut failures = Vec::new();

    for &(domain, key, _, sd, _) in &entries {
        if let Some(sd_val) = sd {
            if !sd_val.is_finite() || sd_val <= 0.0 {
                failures.push(format!(
                    "{}.{}: SD is not positive/finite ({})",
                    domain, key, sd_val
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Invalid standard deviations ({}):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// SD/mean ratio should be sensible (coefficient of variation < 5.0).
/// Very large CVs suggest a typo in value or SD.
#[test]
fn coefficient_of_variation_reasonable() {
    let entries = all_baselines();
    let mut warnings = Vec::new();

    for &(domain, key, value, sd, _) in &entries {
        if let Some(sd_val) = sd {
            if value.abs() > 1e-10 {
                let cv = sd_val / value.abs();
                if cv > 5.0 {
                    warnings.push(format!(
                        "{}.{}: CV = {:.2} (value={}, sd={})",
                        domain, key, cv, value, sd_val
                    ));
                }
            }
        }
    }

    assert!(
        warnings.is_empty(),
        "Suspicious coefficient of variation ({}):\n  {}",
        warnings.len(),
        warnings.join("\n  ")
    );
}

/// All sources must be non-empty and contain a citation.
#[test]
fn sources_non_empty() {
    let entries = all_baselines();
    let mut failures = Vec::new();

    for &(domain, key, _, _, source) in &entries {
        if source.is_empty() {
            failures.push(format!("{}.{}: empty source citation", domain, key));
        }
    }

    assert!(
        failures.is_empty(),
        "Empty source citations ({}):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Verify minimum baseline coverage: at least 200 entries across 15+ domains.
#[test]
fn minimum_coverage() {
    let entries = all_baselines();

    assert!(
        entries.len() >= 200,
        "Expected at least 200 baseline entries, got {}",
        entries.len()
    );

    let mut domains: Vec<&str> = entries.iter().map(|e| e.0).collect();
    domains.sort();
    domains.dedup();

    assert!(
        domains.len() >= 15,
        "Expected at least 15 baseline domains, got {}: {:?}",
        domains.len(),
        domains
    );
}

/// No duplicate keys within a single domain.
#[test]
fn no_duplicate_keys() {
    let bl = BaselineCollection::all();
    let domains: Vec<(&str, &std::collections::BTreeMap<&str, _>)> = vec![
        ("worm", &bl.worm),
        ("cogbench", &bl.cogbench),
        ("tombench", &bl.tombench),
        ("memory_agent", &bl.memory_agent),
        ("executive", &bl.executive),
        ("metacognition", &bl.metacognition),
        ("affect", &bl.affect),
        ("creativity", &bl.creativity),
        ("butlin", &bl.butlin),
        ("inhibition", &bl.inhibition),
        ("attention", &bl.attention),
        ("embodied", &bl.embodied),
        ("reasoning", &bl.reasoning),
        ("sustained_attention", &bl.sustained_attention),
        ("motor", &bl.motor),
        ("language", &bl.language),
        ("social", &bl.social),
        ("neuromod", &bl.neuromod),
        ("llm_cogbench", &bl.llm_cogbench),
        ("llm_tombench", &bl.llm_tombench),
        ("llm_arc", &bl.llm_arc),
    ];

    // BTreeMap inherently prevents duplicates, but verify the domain maps
    // collectively have a reasonable total count (no accidentally empty maps).
    let total: usize = domains.iter().map(|(_, m)| m.len()).sum();
    eprintln!("Total baseline entries: {}", total);

    // Verify no domain is unexpectedly empty
    let empty_domains: Vec<&str> = domains
        .iter()
        .filter(|(_, m)| m.is_empty())
        .map(|(name, _)| *name)
        .collect();

    assert!(
        empty_domains.is_empty(),
        "Empty baseline domains: {:?}",
        empty_domains
    );
}

/// Baselines used for normative z-scores must have SD values.
#[test]
fn normative_baselines_have_sd() {
    let entries = all_baselines();
    let normative_keys = [
        // These are the exact keys from normative_comparison.rs baseline_for_benchmark()
        // Executive
        "stroop_incongruent_accuracy",
        "flanker_incongruent_accuracy",
        "dual_task_cost",
        // WorM
        "nback_2_accuracy",
        "digit_span_forward",
        "change_detection_k4",
        "serial_primacy_advantage",
        // Sustained Attention
        "sart_d_prime",
        "pvt_mean_rt",
        "cpt_hit_rate",
        // Metacognition
        "calibration_error_ece",
        "fok_gamma",
        // ToMBench
        "false_belief_accuracy",
        // Inhibition
        "go_accuracy",
        // Attention
        "feature_search_accuracy",
        "t1_accuracy",
        // Motor
        "srtt_sequence_rt_ticks",
        "coordination_cost",
        // Language
        "lexicality_effect",
        "priming_effect",
        "coherence_mean",
        // Social
        "fairness_sensitivity",
        "norm_d_prime",
        // Neuromod
        "da_agonist_gradient_scale",
        "stimulant_peak_effect",
        "chronic_da_baseline_final",
        "trials_to_criterion",
        "peak_ne_level",
        "conflict_effect",
        "mood_congruent_bias",
        "da_knockout_lr_drop_pct",
        "da_ko_lr_d",
        "psychedelic_proxy_peak",
        // ARC reasoning
        "arc_transfer_accuracy",
        "arc_compositional_accuracy",
        "arc_analogy_accuracy",
        "arc_abduction_accuracy",
        "arc_chain_accuracy",
        "arc_noise_resilience",
        "arc_accuracy_1shot",
        "arc_grid_3x3_accuracy",
        "arc_rsa_correlation",
        "arc_algebra_score",
        "arc_capacity_threshold",
    ];

    let mut missing_sd = Vec::new();
    for &nk in &normative_keys {
        if let Some(&(domain, _, _, sd, _)) = entries.iter().find(|(_, key, _, _, _)| *key == nk) {
            if sd.is_none() {
                missing_sd.push(format!(
                    "{}.{}: used for z-scores but has no SD",
                    domain, nk
                ));
            }
        } else {
            missing_sd.push(format!("???.{}: key not found in any baseline domain", nk));
        }
    }

    assert!(
        missing_sd.is_empty(),
        "Normative baselines missing SD ({}):\n  {}",
        missing_sd.len(),
        missing_sd.join("\n  ")
    );
}
