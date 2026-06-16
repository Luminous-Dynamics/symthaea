// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Polymath Drive - cross-domain collision synthesis during sleep.

use anyhow::Result;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

use crate::databases::{ConsciousnessDatabase, MemoryRecord, MemoryType};
use crate::language::llm_organ::{LLMOrgan, LLMQuery, QueryType};
use crate::school::curriculum::Curriculum;
use crate::school::objective::LearningObjective;

use symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16;
use symthaea_dream::CausalLink;

const DEFAULT_COLLISIONS: usize = 1;
const DEFAULT_MAX_ATTEMPTS: usize = 8;
const DEFAULT_MIN_SIMILARITY: f32 = 0.25;
const DEFAULT_MAX_SIMILARITY: f32 = 0.85;

#[derive(Debug, Clone)]
pub struct PolymathDriveConfig {
    pub collisions: usize,
    pub max_attempts: usize,
    pub min_similarity: f32,
    pub max_similarity: f32,
}

impl PolymathDriveConfig {
    pub fn from_env() -> Self {
        Self {
            collisions: env_usize("SYMTHAEA_POLYMATH_COLLISIONS", DEFAULT_COLLISIONS),
            max_attempts: env_usize("SYMTHAEA_POLYMATH_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
            min_similarity: env_f32("SYMTHAEA_POLYMATH_MIN_SIMILARITY", DEFAULT_MIN_SIMILARITY),
            max_similarity: env_f32("SYMTHAEA_POLYMATH_MAX_SIMILARITY", DEFAULT_MAX_SIMILARITY),
        }
    }

    pub fn enabled(&self) -> bool {
        self.collisions > 0
    }
}

#[derive(Debug, Default, Clone)]
pub struct PolymathReport {
    pub collisions_requested: usize,
    pub collisions_attempted: usize,
    pub collisions_created: usize,
    pub links_stored: usize,
    pub memories_stored: usize,
}

pub async fn run_polymath_collisions(
    llm: &mut LLMOrgan,
    curriculum: &Curriculum,
    database: Option<Arc<dyn ConsciousnessDatabase>>,
) -> Result<PolymathReport> {
    let config = PolymathDriveConfig::from_env();
    let mut report = PolymathReport {
        collisions_requested: config.collisions,
        ..PolymathReport::default()
    };

    if !config.enabled() {
        return Ok(report);
    }

    let Some(db) = database else {
        warn!("Polymath Drive disabled: no database attached");
        return Ok(report);
    };

    let mut domains: HashMap<String, Vec<&LearningObjective>> = HashMap::new();
    for obj in &curriculum.objectives {
        domains
            .entry(obj.domain.name().to_string())
            .or_default()
            .push(obj);
    }

    if domains.len() < 2 {
        return Ok(report);
    }

    let mut domain_keys: Vec<String> = domains.keys().cloned().collect();
    let mut rng = rand::rngs::StdRng::from_rng(rand::thread_rng()).expect("seeding StdRng");

    for _ in 0..config.collisions {
        let mut chosen: Option<(&LearningObjective, &LearningObjective, f32)> = None;

        for _ in 0..config.max_attempts {
            domain_keys.shuffle(&mut rng);
            let domain_a = &domain_keys[0];
            let domain_b = &domain_keys[1];

            // SAFETY: domains are built from non-empty objective groups
            let obj_a = domains[domain_a]
                .choose(&mut rng)
                .copied()
                .expect("domain non-empty");
            let obj_b = domains[domain_b]
                .choose(&mut rng)
                .copied()
                .expect("domain non-empty");

            let similarity = obj_a.encoding.similarity(&obj_b.encoding);
            if similarity >= config.min_similarity && similarity <= config.max_similarity {
                chosen = Some((obj_a, obj_b, similarity));
                break;
            }
        }

        let Some((obj_a, obj_b, similarity)) = chosen else {
            continue;
        };

        report.collisions_attempted += 1;

        let bridge = generate_bridge(llm, obj_a, obj_b, similarity).await;
        let collision_hv = obj_a.encoding.bind(&obj_b.encoding);
        let collision_bin = real_hv_to_hv16(&collision_hv);
        let timestamp_ms = now_ms();
        let record_id = format!("polymath:{}:{}:{}", obj_a.id, obj_b.id, timestamp_ms);
        let content = format!(
            "{} ({} <-> {}): {}",
            bridge.title,
            obj_a.domain.name(),
            obj_b.domain.name(),
            bridge.body
        );

        let topics = vec![
            "polymath".to_string(),
            slug_topic(obj_a.domain.name()),
            slug_topic(obj_b.domain.name()),
        ];

        let metadata = json!({
            "objective_a_id": obj_a.id,
            "objective_b_id": obj_b.id,
            "similarity": similarity,
            "domain_a": obj_a.domain.name(),
            "domain_b": obj_b.domain.name(),
            "bridge_title": bridge.title,
        })
        .to_string();

        let record = MemoryRecord {
            id: record_id,
            memory_type: MemoryType::Semantic,
            encoding: collision_bin,
            content,
            timestamp_ms,
            valence: 0.4,
            arousal: 0.5,
            psi: 0.6,
            topics,
            metadata,
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };

        if let Err(e) = db.store(record).await {
            warn!(error = %e, "Polymath Drive failed to store bridge memory");
        } else {
            report.memories_stored += 1;
        }

        let link = CausalLink {
            action_fingerprint: polymath_fingerprint(&obj_a.id, &obj_b.id),
            state_context: obj_a.encoding.values.clone(),
            outcome: obj_b.encoding.values.clone(),
            weight: similarity,
        };

        if let Err(e) = db.store_causal_links(&[link]).await {
            warn!(error = %e, "Polymath Drive failed to store causal link");
        } else {
            report.links_stored += 1;
        }

        report.collisions_created += 1;
        info!(
            target: "symthaea::polymath",
            objective_a = %obj_a.id,
            objective_b = %obj_b.id,
            similarity = %similarity,
            "Polymath collision synthesized"
        );
    }

    Ok(report)
}

struct BridgeText {
    title: String,
    body: String,
}

async fn generate_bridge(
    llm: &mut LLMOrgan,
    obj_a: &LearningObjective,
    obj_b: &LearningObjective,
    similarity: f32,
) -> BridgeText {
    let prompt = format!(
        "Create a cross-domain bridge between two learning objectives.\n\
Objective A: {name_a}\nDomain A: {domain_a}\nDescription A: {desc_a}\n\n\
Objective B: {name_b}\nDomain B: {domain_b}\nDescription B: {desc_b}\n\n\
Similarity: {similarity:.3}\n\n\
Output format:\n\
Title: <short bridge title>\n\
Bridge: <2-4 sentences describing how A maps to B and one concrete application>\n",
        name_a = obj_a.name,
        domain_a = obj_a.domain.name(),
        desc_a = obj_a.description,
        name_b = obj_b.name,
        domain_b = obj_b.domain.name(),
        desc_b = obj_b.description,
        similarity = similarity
    );

    let query = LLMQuery {
        query_type: QueryType::Analysis,
        content: prompt,
        context: Vec::new(),
        system_prompt: Some(
            "You are the Polymath Drive. Produce concise, specific cross-domain bridges. \
Output only the requested Title and Bridge lines."
                .to_string(),
        ),
        params: None,
    };

    let generation = llm.query_async(query).await;
    let raw = generation.text.trim();

    if raw.is_empty() {
        return fallback_bridge(obj_a, obj_b);
    }

    let mut title = String::new();
    let mut bridge = String::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.to_lowercase().starts_with("title:") {
            title = line.trim_start_matches("Title:").trim().to_string();
        } else if line.to_lowercase().starts_with("bridge:") {
            bridge = line.trim_start_matches("Bridge:").trim().to_string();
        }
    }

    if title.is_empty() || bridge.is_empty() {
        return fallback_bridge(obj_a, obj_b);
    }

    BridgeText {
        title,
        body: bridge,
    }
}

fn fallback_bridge(obj_a: &LearningObjective, obj_b: &LearningObjective) -> BridgeText {
    BridgeText {
        title: format!("Bridge: {} to {}", obj_a.domain.name(), obj_b.domain.name()),
        body: format!(
            "Map \"{}\" into \"{}\" by treating its core mechanism as a reusable pattern. \
Apply the pattern from {} to improve {} by reusing its constraints and feedback loops.",
            obj_a.name,
            obj_b.name,
            obj_a.domain.name(),
            obj_b.domain.name()
        ),
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn slug_topic(input: &str) -> String {
    let mut out = String::new();
    for ch in input.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

fn polymath_fingerprint(objective_a: &str, objective_b: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    ("polymath", objective_a, objective_b).hash(&mut hasher);
    hasher.finish()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
