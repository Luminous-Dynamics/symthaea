// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build a hash-addressed Analyst workbench from an existing paired pack.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use symthaea_muse_protocol::AudioIntegrityEvidence;

const VERSION: &str = "muse-analyst-paired-pack-v1";

#[derive(Deserialize)]
struct Artifact {
    filename: String,
    wav_sha256: String,
    score_sha256: String,
    recipe_sha256: String,
    seed: u64,
    grammar_profile: serde_json::Value,
    performance_dialect: serde_json::Value,
    block_id: Option<String>,
    premise_id: Option<String>,
    motif_id: Option<String>,
    observed_controls: serde_json::Value,
}

#[derive(Deserialize)]
struct StructuralTruth {
    family: String,
    structural_plan: serde_json::Value,
    phrase_start_positions: Vec<f64>,
    cadence_positions: Vec<f64>,
    climax_position: Option<f64>,
    recurrence_intervals: Vec<f64>,
    literal_or_transposed_motif_occurrences: Vec<f64>,
    density_arc: Vec<f64>,
    harmonic_pitch_class_trajectory: Vec<f64>,
    declared_development_operations: Vec<String>,
    #[serde(default)]
    composer_assertion_trace: serde_json::Value,
}

#[derive(Deserialize)]
struct CloneWarning {
    family: String,
    clip_a: String,
    clip_b: String,
    reasons: Vec<String>,
}

#[derive(Deserialize)]
struct NuisanceBaseline {
    accuracy: f64,
    pairwise_accuracy: BTreeMap<String, f64>,
    #[serde(default)]
    evaluation_unit: String,
    #[serde(default)]
    group_count: usize,
    #[serde(default)]
    cluster_bootstrap_accuracy_95: [f64; 2],
    #[serde(default)]
    exact_within_group_permutation_p_value: f64,
    #[serde(default)]
    feature_group_accuracy: BTreeMap<String, f64>,
    #[serde(default)]
    unavailable_feature_groups: Vec<String>,
}

#[derive(Serialize)]
struct ClipEvidence {
    filename: String,
    audio_sha256: String,
    score_sha256: String,
    recipe_sha256: String,
    seed: u64,
    family: String,
    grammar_profile: serde_json::Value,
    performance_dialect: serde_json::Value,
    observed_controls: serde_json::Value,
    phrase_start_positions: Vec<f64>,
    cadence_positions: Vec<f64>,
    climax_position: Option<f64>,
    recurrence_intervals: Vec<f64>,
    motif_occurrence_positions: Vec<f64>,
    motif_occurrence_count: usize,
    motif_ending_return: bool,
    declared_development_operations: Vec<String>,
    plan_obligations: Vec<PlanObligation>,
    composer_assertion_trace: serde_json::Value,
    audio_integrity: Option<AudioIntegrityEvidence>,
    human_review_reasons: Vec<String>,
}

#[derive(Serialize)]
struct PlanObligation {
    code: String,
    fulfilled: bool,
    evidence: String,
}

#[derive(Serialize)]
struct PairwiseDifference {
    family_a: String,
    family_b: String,
    phrase_boundary_distance: f64,
    cadence_distance: f64,
    density_trajectory_distance: f64,
    harmonic_trajectory_distance: f64,
}

#[derive(Serialize)]
struct BlockEvidence {
    block_id: String,
    premise_id: String,
    motif_id: String,
    audio_hashes: Vec<String>,
    families: Vec<String>,
    pairwise_differences: Vec<PairwiseDifference>,
}

#[derive(Serialize)]
struct ReviewItem {
    severity: String,
    code: String,
    target: String,
    reason: String,
    required_reviewer: String,
}

#[derive(Serialize)]
struct Workbench {
    analyzer_version: &'static str,
    source_pack: String,
    nuisance_only_accuracy: f64,
    nuisance_pairwise_accuracy: BTreeMap<String, f64>,
    nuisance_evaluation_unit: String,
    nuisance_group_count: usize,
    nuisance_cluster_bootstrap_accuracy_95: [f64; 2],
    nuisance_exact_permutation_p_value: f64,
    nuisance_feature_group_accuracy: BTreeMap<String, f64>,
    nuisance_unavailable_feature_groups: Vec<String>,
    clips: Vec<ClipEvidence>,
    blocks: Vec<BlockEvidence>,
    review_queue: Vec<ReviewItem>,
    limitations: Vec<&'static str>,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn curve_distance(left: &[f64], right: &[f64]) -> f64 {
    let count = left.len().min(right.len());
    if count == 0 {
        return if left.is_empty() && right.is_empty() {
            0.0
        } else {
            1.0
        };
    }
    left.iter()
        .zip(right)
        .take(count)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / count as f64
}

fn plan_obligations(value: &serde_json::Value) -> Vec<PlanObligation> {
    value
        .pointer("/plan/obligations")
        .or_else(|| value.get("obligations"))
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| PlanObligation {
                    code: item
                        .get("code")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unnamed")
                        .to_string(),
                    fulfilled: item
                        .get("fulfilled")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false),
                    evidence: item
                        .get("evidence")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("not emitted")
                        .to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn paired_difference(
    left: (&Artifact, &StructuralTruth),
    right: (&Artifact, &StructuralTruth),
) -> PairwiseDifference {
    PairwiseDifference {
        family_a: left.1.family.clone(),
        family_b: right.1.family.clone(),
        phrase_boundary_distance: curve_distance(
            &left.1.phrase_start_positions,
            &right.1.phrase_start_positions,
        ),
        cadence_distance: curve_distance(&left.1.cadence_positions, &right.1.cadence_positions),
        density_trajectory_distance: curve_distance(&left.1.density_arc, &right.1.density_arc),
        harmonic_trajectory_distance: curve_distance(
            &left.1.harmonic_pitch_class_trajectory,
            &right.1.harmonic_pitch_class_trajectory,
        ),
    }
}

fn build(root: &Path) -> Workbench {
    let sealed = root.join("sealed");
    let artifacts: Vec<Artifact> = read_json(&sealed.join("artifacts.json"));
    let traced_truth = sealed.join("structural_truth_with_composer_trace_v2.json");
    let legacy_truth = sealed.join("structural_truth_by_sha256.json");
    let truths: BTreeMap<String, StructuralTruth> = read_json(if traced_truth.exists() {
        traced_truth.as_path()
    } else {
        legacy_truth.as_path()
    });
    let clones: Vec<CloneWarning> = read_json(&sealed.join("clone_warnings.json"));
    let integrity_path = sealed.join("audio_integrity_by_sha256.json");
    let integrity: BTreeMap<String, AudioIntegrityEvidence> = if integrity_path.exists() {
        read_json(&integrity_path)
    } else {
        BTreeMap::new()
    };
    let block_safe = sealed.join("nuisance_baseline_block_safe_v2.json");
    let nuisance: NuisanceBaseline = read_json(if block_safe.exists() {
        &block_safe
    } else {
        let legacy = sealed.join("nuisance_baseline.json");
        return build_with_nuisance(
            root,
            artifacts,
            truths,
            clones,
            integrity,
            read_json(&legacy),
        );
    });
    build_with_nuisance(root, artifacts, truths, clones, integrity, nuisance)
}

fn build_with_nuisance(
    root: &Path,
    artifacts: Vec<Artifact>,
    truths: BTreeMap<String, StructuralTruth>,
    clones: Vec<CloneWarning>,
    integrity: BTreeMap<String, AudioIntegrityEvidence>,
    nuisance: NuisanceBaseline,
) -> Workbench {
    let mut review_queue = Vec::new();
    let mut clips = Vec::new();

    for artifact in &artifacts {
        let truth = &truths[&artifact.wav_sha256];
        let obligations = plan_obligations(&truth.structural_plan);
        let mut reasons: Vec<String> = Vec::new();
        if truth.literal_or_transposed_motif_occurrences.is_empty() {
            reasons.push("No literal/transposed motif occurrence met the pack threshold.".into());
            review_queue.push(ReviewItem {
                severity: "review".into(),
                code: "motif-not-detected".into(),
                target: artifact.wav_sha256.clone(),
                reason: reasons.last().unwrap().clone(),
                required_reviewer: "motif reviewer".into(),
            });
        }
        if obligations.iter().any(|item| !item.fulfilled) {
            reasons.push("One or more grammar-plan obligations failed.".into());
        }
        if truth.family == "ModalArcInformed" {
            reasons.push(
                "Culturally qualified output requires expert review for authenticity claims."
                    .into(),
            );
            review_queue.push(ReviewItem {
                severity: "required".into(),
                code: "cultural-review-required".into(),
                target: artifact.wav_sha256.clone(),
                reason: reasons.last().unwrap().clone(),
                required_reviewer: "qualified tradition bearer or musician".into(),
            });
        }
        if let Some(audio) = integrity.get(&artifact.wav_sha256)
            && !audio.issues.is_empty()
        {
            reasons.push(format!(
                "Audio integrity checks triggered: {}.",
                audio.issues.join(", ")
            ));
            review_queue.push(ReviewItem {
                severity: "required".into(),
                code: "audio-integrity".into(),
                target: artifact.wav_sha256.clone(),
                reason: reasons.last().unwrap().clone(),
                required_reviewer: "audio/rendering engineer".into(),
            });
        }
        clips.push(ClipEvidence {
            filename: artifact.filename.clone(),
            audio_sha256: artifact.wav_sha256.clone(),
            score_sha256: artifact.score_sha256.clone(),
            recipe_sha256: artifact.recipe_sha256.clone(),
            seed: artifact.seed,
            family: truth.family.clone(),
            grammar_profile: artifact.grammar_profile.clone(),
            performance_dialect: artifact.performance_dialect.clone(),
            observed_controls: artifact.observed_controls.clone(),
            phrase_start_positions: truth.phrase_start_positions.clone(),
            cadence_positions: truth.cadence_positions.clone(),
            climax_position: truth.climax_position,
            recurrence_intervals: truth.recurrence_intervals.clone(),
            motif_occurrence_count: truth.literal_or_transposed_motif_occurrences.len(),
            motif_ending_return: truth
                .literal_or_transposed_motif_occurrences
                .iter()
                .any(|position| *position >= 0.8),
            motif_occurrence_positions: truth.literal_or_transposed_motif_occurrences.clone(),
            declared_development_operations: truth.declared_development_operations.clone(),
            plan_obligations: obligations,
            composer_assertion_trace: truth.composer_assertion_trace.clone(),
            audio_integrity: integrity.get(&artifact.wav_sha256).cloned(),
            human_review_reasons: reasons,
        });
    }

    // Deterministic random-audit lane: select at least one otherwise-unflagged
    // clip per grammar using its content hash as the sampling key. This makes
    // Analyst false negatives observable instead of reviewing only alerts.
    let already_flagged: std::collections::BTreeSet<String> = review_queue
        .iter()
        .map(|item| item.target.clone())
        .collect();
    let families: std::collections::BTreeSet<_> =
        clips.iter().map(|clip| clip.family.as_str()).collect();
    for family in families {
        if let Some(sample) = clips
            .iter()
            .filter(|clip| clip.family == family)
            .filter(|clip| !already_flagged.contains(clip.audio_sha256.as_str()))
            .min_by_key(|clip| &clip.audio_sha256)
        {
            review_queue.push(ReviewItem {
                severity: "audit".into(),
                code: "random-accepted-piece-audit".into(),
                target: sample.audio_sha256.clone(),
                reason: format!(
                    "Deterministic content-hash sample from otherwise accepted {} output.",
                    sample.family
                ),
                required_reviewer: "general listener".into(),
            });
        }
    }

    for warning in clones {
        review_queue.push(ReviewItem {
            severity: "review".into(),
            code: "within-family-template-collision".into(),
            target: format!("{} / {}", warning.clip_a, warning.clip_b),
            reason: format!("{}: {}", warning.family, warning.reasons.join(", ")),
            required_reviewer: "composer engineer".into(),
        });
    }
    if nuisance.accuracy >= 0.75 {
        review_queue.push(ReviewItem {
            severity: "review".into(),
            code: "nuisance-cue-leakage".into(),
            target: root.display().to_string(),
            reason: format!(
                "Nuisance-only accuracy is {:.1}%.",
                nuisance.accuracy * 100.0
            ),
            required_reviewer: "study methodologist".into(),
        });
    }

    let mut grouped: BTreeMap<String, Vec<(&Artifact, &StructuralTruth)>> = BTreeMap::new();
    for artifact in &artifacts {
        if let Some(block) = &artifact.block_id {
            grouped
                .entry(block.clone())
                .or_default()
                .push((artifact, &truths[&artifact.wav_sha256]));
        }
    }
    let blocks = grouped
        .into_iter()
        .map(|(block_id, mut members)| {
            members.sort_by(|left, right| left.1.family.cmp(&right.1.family));
            let mut differences = Vec::new();
            for left in 0..members.len() {
                for right in left + 1..members.len() {
                    differences.push(paired_difference(members[left], members[right]));
                }
            }
            BlockEvidence {
                block_id,
                premise_id: members[0].0.premise_id.clone().unwrap_or_default(),
                motif_id: members[0].0.motif_id.clone().unwrap_or_default(),
                audio_hashes: members
                    .iter()
                    .map(|member| member.0.wav_sha256.clone())
                    .collect(),
                families: members
                    .iter()
                    .map(|member| member.1.family.clone())
                    .collect(),
                pairwise_differences: differences,
            }
        })
        .collect();

    Workbench {
        analyzer_version: VERSION,
        source_pack: root.display().to_string(),
        nuisance_only_accuracy: nuisance.accuracy,
        nuisance_pairwise_accuracy: nuisance.pairwise_accuracy,
        nuisance_evaluation_unit: nuisance.evaluation_unit,
        nuisance_group_count: nuisance.group_count,
        nuisance_cluster_bootstrap_accuracy_95: nuisance.cluster_bootstrap_accuracy_95,
        nuisance_exact_permutation_p_value: nuisance.exact_within_group_permutation_p_value,
        nuisance_feature_group_accuracy: nuisance.feature_group_accuracy,
        nuisance_unavailable_feature_groups: nuisance.unavailable_feature_groups,
        clips,
        blocks,
        review_queue,
        limitations: vec![
            "This report analyzes sealed symbolic evidence and does not infer beauty.",
            "The pack occurrence scan is narrower than the Studio transformation-aware scan.",
            "Audio embeddings and human observations remain separate evidence layers.",
        ],
    }
}

fn write_html(root: &Path) {
    let html = r#"<!doctype html><meta charset="utf-8"><title>Muse Analyst</title>
<style>body{font:15px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#101216;color:#e7e9ef}pre{white-space:pre-wrap;background:#191d25;border:1px solid #303747;border-radius:12px;padding:16px}</style>
<h1>Muse Analyst · paired pack</h1><p>Machine-readable causal, motif, control, and escalation evidence.</p><pre id="report">Loading…</pre>
<script>fetch("analyst_bundle.json").then(function(r){return r.json()}).then(function(d){document.getElementById("report").textContent=JSON.stringify(d,null,2)});</script>"#;
    std::fs::write(root.join("analyst.html"), html).expect("write analyst page");
}

fn main() {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("usage: muse_analyst_pack PACK_DIR");
            std::process::exit(2);
        });
    let report = build(&root);
    std::fs::write(
        root.join("analyst_bundle.json"),
        serde_json::to_vec_pretty(&report).expect("serialize report"),
    )
    .expect("write report");
    write_html(&root);
    println!(
        "{}: {} clips, {} blocks, {} review items",
        root.display(),
        report.clips.len(),
        report.blocks.len(),
        report.review_queue.len()
    );
}
