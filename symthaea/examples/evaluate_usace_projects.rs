// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evaluate USACE dam projects using real Symthaea AssetEvaluator.
//!
//! Loads dam data from Terra Atlas, runs each through the Eight Harmonies
//! evaluation pipeline, and outputs genuine trust scores.
//!
//! Usage:
//!   cargo run --example evaluate_usace_projects --features mycelix
//!
//! Output: JSON array of scored projects to stdout, summary to stderr.

use std::collections::HashMap;
use std::fs;

use symthaea::consciousness::asset_evaluator::{AssetEvaluator, AssetMetadata};
use symthaea::consciousness::mycelix_bridge::ConsciousnessSnapshot;

/// USACE project from the JSON data file.
#[derive(serde::Deserialize)]
struct UsaceProject {
    id: String,
    name: String,
    #[serde(rename = "type")]
    project_type: String,
    location: Location,
    #[serde(rename = "capacityMw")]
    capacity_mw: f64,
    developer: String,
    status: String,
    environmental: Option<Environmental>,
    metadata: Option<ProjectMetadata>,
}

#[derive(serde::Deserialize)]
struct Location {
    latitude: f64,
    longitude: f64,
    state: String,
    #[serde(default)]
    county: String,
    #[serde(default)]
    terrain: String,
}

#[derive(serde::Deserialize)]
struct Environmental {
    #[serde(rename = "waterFlow", default)]
    water_flow: f64,
}

#[derive(serde::Deserialize)]
struct ProjectMetadata {
    #[serde(rename = "isExistingDam", default)]
    is_existing_dam: bool,
    #[serde(rename = "hasHydropower", default)]
    has_hydropower: bool,
    #[serde(rename = "damHeight", default)]
    dam_height: u32,
    #[serde(rename = "yearBuilt", default)]
    year_built: u32,
    #[serde(rename = "retrofitPotential", default)]
    retrofit_potential: String,
    #[serde(default)]
    storage: u64,
    #[serde(rename = "hazardClass", default)]
    hazard_class: String,
}

#[derive(serde::Serialize)]
struct ScoredProject {
    id: String,
    name: String,
    state: String,
    capacity_mw: f64,
    retrofit_potential: String,
    phi_score: f64,
    harmony_alignment: f64,
    per_harmony: HashMap<String, f64>,
    care_activation: f64,
    meta_awareness: f64,
    recommendation: String,
    violations: Vec<String>,
    authenticity: f64,
}

#[derive(serde::Deserialize)]
struct UsaceData {
    projects: Vec<UsaceProject>,
}

fn main() {
    // Load USACE data
    let data_path = std::env::var("USACE_DATA_PATH")
        .unwrap_or_else(|_| "../terra-atlas-mvp/data/usace-dams.json".to_string());

    eprintln!("Loading USACE data from {}...", data_path);
    let raw = fs::read_to_string(&data_path).expect("Failed to read USACE data");
    let data: UsaceData = serde_json::from_str(&raw).expect("Failed to parse USACE data");
    eprintln!("Loaded {} projects", data.projects.len());

    // Limit to top projects by capacity (evaluating 100K+ would take too long)
    let max_projects: usize = std::env::var("MAX_PROJECTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let mut projects = data.projects;
    projects.sort_by(|a, b| {
        b.capacity_mw
            .partial_cmp(&a.capacity_mw)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_projects: Vec<_> = projects.into_iter().take(max_projects).collect();
    eprintln!("Evaluating top {} by capacity...", top_projects.len());

    // Create evaluator and a representative consciousness snapshot
    // This represents the evaluating Symthaea instance's state
    let mut evaluator = AssetEvaluator::new();
    let snapshot = ConsciousnessSnapshot::new(
        0.72, // phi — moderate-high integration
        0.70, // meta_awareness
        0.80, // self_model_accuracy
        0.85, // coherence
        0.30, // affective_valence (neutral-positive)
        0.65, // care_activation
    );

    let mut scored: Vec<ScoredProject> = Vec::new();
    let mut total_phi = 0.0;
    let mut total_harmony = 0.0;

    for (i, project) in top_projects.iter().enumerate() {
        // Build impact claims from real data
        let mut impact_claims = vec![
            format!(
                "Hydroelectric retrofit of existing {} dam",
                if project
                    .metadata
                    .as_ref()
                    .map_or(false, |m| m.is_existing_dam)
                {
                    "active"
                } else {
                    "inactive"
                }
            ),
            format!("Capacity: {:.1} MW of clean energy", project.capacity_mw),
        ];

        if let Some(ref env) = project.environmental {
            if env.water_flow > 0.0 {
                impact_claims.push(format!("Water flow: {:.1} units", env.water_flow));
            }
        }

        if let Some(ref meta) = project.metadata {
            if meta.dam_height > 0 {
                impact_claims.push(format!(
                    "Dam height: {} ft, storage: {} acre-ft",
                    meta.dam_height, meta.storage
                ));
            }
            if meta.year_built > 0 {
                impact_claims.push(format!(
                    "Built {}, {} infrastructure",
                    meta.year_built,
                    if meta.year_built < 1960 {
                        "heritage"
                    } else if meta.year_built < 1990 {
                        "mature"
                    } else {
                        "modern"
                    }
                ));
            }
            if !meta.hazard_class.is_empty() {
                impact_claims.push(format!("Hazard classification: {}", meta.hazard_class));
            }
        }

        let description = format!(
            "{} — {} hydroelectric retrofit in {}, {}. Developer: {}. Status: {}.",
            project.name,
            project.capacity_mw,
            project.location.state,
            project.location.terrain,
            project.developer,
            project.status,
        );

        let metadata = AssetMetadata {
            description,
            project_type: "Hydro".to_string(),
            capacity_mw: project.capacity_mw,
            community_did: None,
            impact_claims,
        };

        let score = evaluator.evaluate(&metadata, &snapshot);

        total_phi += score.phi_score;
        total_harmony += score.harmony_alignment;

        scored.push(ScoredProject {
            id: project.id.clone(),
            name: project.name.clone(),
            state: project.location.state.clone(),
            capacity_mw: project.capacity_mw,
            retrofit_potential: project
                .metadata
                .as_ref()
                .map_or("unknown".into(), |m| m.retrofit_potential.clone()),
            phi_score: score.phi_score,
            harmony_alignment: score.harmony_alignment,
            per_harmony: score.per_harmony,
            care_activation: score.care_activation,
            meta_awareness: score.meta_awareness,
            recommendation: format!("{:?}", score.recommendation),
            violations: score.violations,
            authenticity: score.authenticity,
        });

        if (i + 1) % 25 == 0 {
            eprintln!("  Evaluated {}/{}", i + 1, top_projects.len());
        }
    }

    let count = scored.len() as f64;
    eprintln!("\n=== Results ===");
    eprintln!("Projects evaluated: {}", scored.len());
    eprintln!("Mean Phi:     {:.4}", total_phi / count);
    eprintln!("Mean Harmony: {:.4}", total_harmony / count);

    // Score distribution
    let high = scored.iter().filter(|s| s.phi_score >= 0.7).count();
    let mid = scored
        .iter()
        .filter(|s| s.phi_score >= 0.4 && s.phi_score < 0.7)
        .count();
    let low = scored.iter().filter(|s| s.phi_score < 0.4).count();
    eprintln!(
        "Distribution: {} Aligned, {} Emerging, {} Nascent",
        high, mid, low
    );

    // Top 10
    scored.sort_by(|a, b| {
        b.harmony_alignment
            .partial_cmp(&a.harmony_alignment)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("\nTop 10 by Harmony Alignment:");
    for (i, s) in scored.iter().take(10).enumerate() {
        eprintln!(
            "  {}. {} ({}, {:.1} MW) — Phi: {:.3}, Harmony: {:.3}, {:?}",
            i + 1,
            s.name,
            s.state,
            s.capacity_mw,
            s.phi_score,
            s.harmony_alignment,
            s.recommendation
        );
    }

    // Output full JSON to stdout
    println!("{}", serde_json::to_string_pretty(&scored).unwrap());
}