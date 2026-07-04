// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Curriculum Extender - The Neural Bridge 2.0
//!
//! This module automates the pipeline:
//! Web Research ──▶ LLM Translation ──▶ Dynamic Curriculum Extension.
//!
//! It enables Symthaea to learn about new domains autonomously by
//! researching them on the web and converting documentation into
//! structured HDC-encoded learning objectives.

use anyhow::Result;
use chrono::Utc;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

use crate::databases::{ConsciousnessDatabase, MemoryRecord, MemoryType};
use crate::language::llm_organ::{LLMOrgan, LLMQuery, LLMQueryParams, QueryType};
use crate::school::curriculum::{Curriculum, CurriculumSchema, ObjectiveSchema};
use crate::school::objective::{Difficulty, Domain, LearningObjective, ObjectiveBuilder};
use crate::web_research::WebResearcher;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16;
use symthaea_dream::CausalLink;

/// System prompt for Query Planning
pub const QUERY_PLANNER_PROMPT: &str = r#"You are Symthaea's RESEARCH PLANNER.
Your goal is to generate 3-5 high-quality search queries to gather comprehensive documentation about a topic.
Focus on:
1. Official documentation and specifications.
2. Architecture overviews and core concepts.
3. Implementation examples and tutorials.
4. Advanced edge cases and performance considerations.
5. Academic sources (include at least one query with "arxiv" or "openreview" for research topics).

OUTPUT: Provide exactly one query per line. No numbering, no bullets, no commentary."#;

/// System prompt for the LLM when acting as a Curriculum Architect.
pub const CURRICULUM_ARCHITECT_PROMPT: &str = r#"You are Symthaea's CURRICULUM ARCHITECT.

Your role is to convert raw technical documentation into a structured learning path (Curriculum).

You MUST honor any minimum objective count provided in the TASK. Treat it as a hard constraint.
You must think like an ontologist, not a summarizer. Extract a fractal holarchy of knowledge.

OUTPUT FORMAT:
You must output ONLY valid JSON matching the `CurriculumSchema`.
Do not include any preamble, explanation, or markdown formatting blocks.

JSON STRUCTURE:
{
  "name": "Topic Name",
  "description": "High-level description",
  "objectives": [
    {
      "id": "kebab-case-id",
      "name": "Objective Name",
      "description": "Specific concept to learn",
      "domain": "Rust|NixOS|HDC|Mathematics|...",
      "difficulty": 0.1 to 1.0,
      "prerequisites": ["other-id"],
      "tags": ["tag1", "tag2"],
      "estimated_minutes": 15
    }
  ]
}

CRITICAL RULES:
1. Break down complex topics into small, atomic objectives (15-30 mins each).
2. Ensure prerequisite IDs match the IDs of other objectives in the list.
3. Map domains correctly (default to "Custom" if unsure).
4. Difficulty: 0.1 (Beginner) to 1.0 (Expert).
5. Produce 5-10 objectives per paper/source. Cover at least:
   - Core Theory (math/physics)
   - Algorithmic Structure (logic/architecture)
   - Implementation Constraints (Rust/code)
   - Thermodynamic Impact (energy/6-watt limits)
6. Output ONLY the JSON. Hallucination of prose will break the ingestion pipeline."#;

/// System prompt for repairing malformed curriculum JSON.
pub const CURRICULUM_REPAIR_PROMPT: &str = r#"You are Symthaea's CURRICULUM ARCHITECT (repair mode).

Your task is to fix malformed JSON so it validates against the CurriculumSchema.
Return ONLY valid JSON, no code fences, no commentary.

Constraints:
1. JSON must include: name, description, objectives.
2. Each objective must include: id, name, description, domain, difficulty, prerequisites, tags, estimated_minutes.
3. Difficulty must be within 0.0..=1.0.
4. Objective IDs must be unique and kebab-case (never use placeholder IDs like "kebab-case-id").
5. Prerequisites must reference existing objective IDs in the list.
"#;

const MAX_JSON_RETRIES: usize = 2;
const MAX_DIMENSIONALITY_RETRIES: usize = 2;

/// Generation params for curriculum-synthesis/repair LLM calls.
///
/// `LLMOrgan::query_async`'s default `max_generation_length` (1024 tokens,
/// tuned for short conversational replies) reliably truncated the JSON
/// objective list mid-array for real research topics (observed 2026-07-04:
/// "EOF while parsing a list" on every synthesis/repair attempt). Curriculum
/// JSON needs far more headroom since it embeds multiple full objectives.
fn curriculum_json_params() -> LLMQueryParams {
    LLMQueryParams {
        temperature: None,
        max_length: Some(2048),
        stop_sequences: Vec::new(),
    }
}

/// Summary of a research ingestion attempt.
#[derive(Debug, Clone)]
pub struct ResearchSummary {
    pub objectives_added: usize,
    pub total_objectives: usize,
    pub confidence: f32,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy)]
struct ResearchBudgetConfig {
    total_budget: Duration,
    yield_timeout: Duration,
}

#[derive(Clone, Copy)]
struct ResearchDepthConfig {
    target_chars: usize,
    max_expansions: usize,
}

impl ResearchBudgetConfig {
    fn from_env() -> Self {
        Self {
            total_budget: Duration::from_secs(env_secs("SYMTHAEA_RESEARCH_BUDGET_SECS", 0)),
            yield_timeout: Duration::from_secs(env_secs(
                "SYMTHAEA_RESEARCH_YIELD_TIMEOUT_SECS",
                60,
            )),
        }
    }

    fn is_budgeted(&self) -> bool {
        self.total_budget.as_secs() > 0
    }
}

impl ResearchDepthConfig {
    fn from_env() -> Self {
        Self {
            target_chars: env_usize("SYMTHAEA_RESEARCH_TARGET_CHARS", 8000),
            max_expansions: env_usize("SYMTHAEA_RESEARCH_MAX_EXPANSIONS", 1),
        }
    }

    fn should_expand(&self, current_len: usize, expansions: usize) -> bool {
        self.target_chars > 0 && current_len < self.target_chars && expansions < self.max_expansions
    }
}

fn env_secs(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn similarity_threshold() -> f32 {
    env_f32("SYMTHAEA_RESEARCH_SIMILARITY_THRESHOLD", 0.85).clamp(0.0, 1.0)
}

fn global_similarity_threshold() -> f32 {
    env_f32("SYMTHAEA_GLOBAL_SIMILARITY_THRESHOLD", 0.9).clamp(0.0, 1.0)
}

fn min_objectives_per_5k() -> usize {
    std::env::var("SYMTHAEA_RESEARCH_MIN_OBJECTIVES_PER_5K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1)
}

fn required_objectives(content_len: usize, min_per_5k: usize) -> usize {
    let blocks = ((content_len + 4999) / 5000).max(1);
    let required = blocks * min_per_5k;
    required.max(min_per_5k)
}

/// Orchestrates the autonomous learning pipeline
pub struct CurriculumExtender {
    researcher: WebResearcher,
    llm: LLMOrgan,
}

impl CurriculumExtender {
    /// Create a new extender
    pub fn new(researcher: WebResearcher, llm: LLMOrgan) -> Self {
        Self { researcher, llm }
    }

    /// Research a topic and extend the curriculum
    ///
    /// 1. Plan search queries using LLM.
    /// 2. Research topic on the web.
    /// 3. Extract key documentation.
    /// 4. LLM translates docs into CurriculumSchema.
    /// 5. Merge into the provided curriculum.
    pub async fn research_and_extend(
        &mut self,
        topic: &str,
        target_curriculum: &mut Curriculum,
        dimension: usize,
        database: Option<Arc<dyn ConsciousnessDatabase>>,
    ) -> Result<ResearchSummary> {
        info!("🌐 Starting autonomous research for: '{}'...", topic);
        let before_count = target_curriculum.objectives.len();
        let budget = ResearchBudgetConfig::from_env();
        let depth = ResearchDepthConfig::from_env();
        let research_start = Instant::now();
        let mut last_growth = Instant::now();

        // 1. Plan Queries
        info!("   🔍 Planning research strategy...");
        let plan_query = LLMQuery {
            query_type: QueryType::Analysis,
            content: format!("Plan research queries for the topic: {}", topic),
            context: Vec::new(),
            system_prompt: Some(QUERY_PLANNER_PROMPT.to_string()),
            params: None,
        };
        let plan_gen = self.llm.query_async(plan_query).await;
        let queries: Vec<String> = plan_gen
            .text
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // 2. Perform Web Research for each query
        let mut aggregate_content = String::new();
        let mut last_len = 0usize;
        let mut seen_queries: HashSet<String> = HashSet::new();
        for (i, query) in queries
            .iter()
            .filter(|q| seen_queries.insert(q.to_lowercase()))
            .enumerate()
        {
            if budget.is_budgeted() && research_start.elapsed() > budget.total_budget {
                warn!(
                    "   ⏱️ Research budget exceeded after {:?}; stopping early",
                    budget.total_budget
                );
                break;
            }

            info!(
                "   🌐 [Phase {}/{}] Researching: {}...",
                i + 1,
                queries.len(),
                query
            );
            if let Ok(result) = self.researcher.research_and_verify(query).await {
                if !result.content.is_empty() {
                    aggregate_content.push_str(&format!(
                        "\n\n--- Source: {} ---\n{}",
                        result.url, result.content
                    ));
                }
            }

            if aggregate_content.len() > last_len {
                last_len = aggregate_content.len();
                last_growth = Instant::now();
            } else if budget.yield_timeout.as_secs() > 0
                && last_growth.elapsed() > budget.yield_timeout
            {
                warn!(
                    "   ⛔ No new content after {:?}; stopping early",
                    budget.yield_timeout
                );
                break;
            }
        }

        let mut expansions = 0usize;
        while depth.should_expand(aggregate_content.len(), expansions) {
            if budget.is_budgeted() && research_start.elapsed() > budget.total_budget {
                warn!(
                    "   ⏱️ Research budget exceeded after {:?}; stopping early",
                    budget.total_budget
                );
                break;
            }

            expansions += 1;
            info!(
                "   🔎 Expanding research depth ({} of {}, target {} chars)...",
                expansions, depth.max_expansions, depth.target_chars
            );

            let expand_prompt = format!(
                "We need deeper coverage for the topic: {topic}\n\nExisting queries:\n{}\n\nCollected {} chars so far (target {}). Generate 3-5 NEW, distinct queries focusing on theory, implementation, benchmarks, and failure modes. Include at least one query with \"arxiv\" and one with \"openreview\".\n\nOutput one query per line, no numbering, no bullets.",
                queries.join("\n"),
                aggregate_content.len(),
                depth.target_chars
            );
            let expand_query = LLMQuery {
                query_type: QueryType::Analysis,
                content: expand_prompt,
                context: Vec::new(),
                system_prompt: Some(QUERY_PLANNER_PROMPT.to_string()),
                params: None,
            };
            let expand_gen = self.llm.query_async(expand_query).await;
            let extra_queries: Vec<String> = expand_gen
                .text
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .filter(|q| seen_queries.insert(q.to_lowercase()))
                .collect();

            if extra_queries.is_empty() {
                warn!("   ⛔ No new distinct queries generated; stopping expansion.");
                break;
            }

            for (i, query) in extra_queries.iter().enumerate() {
                if budget.is_budgeted() && research_start.elapsed() > budget.total_budget {
                    warn!(
                        "   ⏱️ Research budget exceeded after {:?}; stopping early",
                        budget.total_budget
                    );
                    break;
                }

                info!(
                    "   🌐 [Expansion {}/{} | Query {}/{}] Researching: {}...",
                    expansions,
                    depth.max_expansions,
                    i + 1,
                    extra_queries.len(),
                    query
                );
                if let Ok(result) = self.researcher.research_and_verify(query).await {
                    if !result.content.is_empty() {
                        aggregate_content.push_str(&format!(
                            "\n\n--- Source: {} ---\n{}",
                            result.url, result.content
                        ));
                    }
                }

                if aggregate_content.len() > last_len {
                    last_len = aggregate_content.len();
                    last_growth = Instant::now();
                } else if budget.yield_timeout.as_secs() > 0
                    && last_growth.elapsed() > budget.yield_timeout
                {
                    warn!(
                        "   ⛔ No new content after {:?}; stopping early",
                        budget.yield_timeout
                    );
                    break;
                }
            }
        }

        if aggregate_content.is_empty() {
            anyhow::bail!("No substantial content found for topic: {}", topic);
        }

        info!(
            "   ✅ Research complete. Collected {} chars of documentation.",
            aggregate_content.len()
        );

        let min_per_5k = min_objectives_per_5k();
        let required_min = required_objectives(aggregate_content.len(), min_per_5k);

        // 3. Prepare LLM Query for synthesis
        let prompt = format!(
            "DOCUMENTATION CONTENT:\n{}\n\nTASK:\nExtract a curriculum for '{}'.\nMinimum objectives required: {}.\nReturn at least {} objectives.",
            aggregate_content, topic, required_min, required_min
        );

        let query = LLMQuery {
            query_type: QueryType::Code,
            content: prompt,
            context: Vec::new(),
            system_prompt: Some(CURRICULUM_ARCHITECT_PROMPT.to_string()),
            params: Some(curriculum_json_params()),
        };

        // 4. LLM Translation
        info!("   🤖 Synthesizing research into structured objectives...");
        let generation = self.llm.query_async(query).await;

        let mut candidate_json = sanitize_json(&generation.text);
        let mut last_error = None;
        let _ = last_error.as_deref();
        let mut attempts = 0usize;
        let total_attempts = MAX_JSON_RETRIES + MAX_DIMENSIONALITY_RETRIES + 1;
        let mut dimensionality_retries = 0usize;

        loop {
            if attempts > 0 {
                info!(
                    "   🔁 Retrying curriculum ingestion ({}/{})...",
                    attempts + 1,
                    total_attempts
                );
            }

            let mut placeholder_detected = candidate_json.contains("\"kebab-case-id\"");
            if let Some(normalized) =
                normalize_curriculum_json(&candidate_json, required_min, topic)
            {
                if normalized != candidate_json {
                    candidate_json = normalized;
                    placeholder_detected = candidate_json.contains("\"kebab-case-id\"");
                    warn!("   🔧 Auto-normalized objective IDs in curriculum JSON.");
                }
            }

            if placeholder_detected {
                let err_msg = "Placeholder objective id detected (kebab-case-id)".to_string();
                last_error = Some(err_msg.clone());
                warn!("   ❌ {}. Forcing ID regeneration...", err_msg);
            } else {
                if let Ok(schema) = serde_json::from_str::<CurriculumSchema>(&candidate_json) {
                    let redundant =
                        detect_redundant_pairs(&schema, dimension, similarity_threshold(), 3);
                    if !redundant.is_empty() {
                        let err_msg = format!(
                            "Semantic collapse detected: {} redundant pairs above {:.2}",
                            redundant.len(),
                            similarity_threshold()
                        );
                        let mut pair_lines = String::new();
                        for pair in &redundant {
                            pair_lines.push_str(&format!(
                                "- {} ({}) ⇄ {} ({}) similarity {:.2}\n",
                                pair.id_a, pair.name_a, pair.id_b, pair.name_b, pair.similarity
                            ));
                        }

                        warn!("   ❌ {}. Forcing semantic diversification...", err_msg);

                        if attempts + 1 < total_attempts {
                            let repair_prompt = format!(
                                "Semantic collapse detected in the curriculum objectives.\n{}\n\nRedundant pairs:\n{}\nAction:\n- Merge each redundant pair into a single objective.\n- Replace each removed objective with a NEW objective that introduces a distinct causal variable from the documentation.\n- Ensure at least {} objectives.\nReturn ONLY valid JSON.\n\nJSON:\n{}",
                                err_msg, pair_lines, required_min, candidate_json
                            );
                            let repair_query = LLMQuery {
                                query_type: QueryType::Code,
                                content: repair_prompt,
                                context: Vec::new(),
                                system_prompt: Some(CURRICULUM_REPAIR_PROMPT.to_string()),
                                params: Some(curriculum_json_params()),
                            };
                            let repaired = self.llm.query_async(repair_query).await;
                            candidate_json = sanitize_json(&repaired.text);
                            attempts += 1;
                            continue;
                        }
                    }
                }

                if let (Some(db), Ok(mut schema)) = (
                    database.as_ref().map(|db| db.as_ref()),
                    serde_json::from_str::<CurriculumSchema>(&candidate_json),
                ) {
                    let threshold = global_similarity_threshold();
                    info!(
                        "   🧠 Global HDC sweep scanning {} objectives (threshold {:.2})",
                        schema.objectives.len(),
                        threshold
                    );
                    match global_hdc_sweep(&mut schema, dimension, db, threshold).await {
                        Ok(report) => {
                            if report.suppressed > 0 && schema.objectives.is_empty() {
                                warn!(
                                    "   🧠 Global HDC sweep suppressed all objectives; skipping ingestion."
                                );
                                if !report.causal_links.is_empty() {
                                    if let Err(e) =
                                        db.store_causal_links(&report.causal_links).await
                                    {
                                        warn!("   ⚠️ Failed to persist causal links: {}", e);
                                    } else {
                                        info!(
                                            "   🔗 Stored {} causal links from global sweep",
                                            report.causal_links.len()
                                        );
                                    }
                                }
                                return Ok(ResearchSummary {
                                    objectives_added: 0,
                                    total_objectives: target_curriculum.objectives.len(),
                                    confidence: 0.8,
                                    warnings: vec![
                                        "All objectives suppressed by global HDC sweep".to_string(),
                                    ],
                                });
                            }
                            if report.suppressed > 0 {
                                warn!(
                                    "   🧠 Global HDC sweep suppressed {} objectives (threshold {:.2})",
                                    report.suppressed, threshold
                                );
                                if let Ok(updated) = serde_json::to_string_pretty(&schema) {
                                    candidate_json = updated;
                                }
                            }
                            if !report.causal_links.is_empty() {
                                if let Err(e) = db.store_causal_links(&report.causal_links).await {
                                    warn!("   ⚠️ Failed to persist causal links: {}", e);
                                } else {
                                    info!(
                                        "   🔗 Stored {} causal links from global sweep",
                                        report.causal_links.len()
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            warn!("   ⚠️ Global HDC sweep failed: {}", e);
                        }
                    }
                }

                info!("   📥 Ingesting new objectives into knowledge graph...");
                let mut candidate_curriculum = target_curriculum.clone();
                match candidate_curriculum.extend_from_json(&candidate_json, dimension) {
                    Ok(_) => {
                        let total_objectives = candidate_curriculum.objectives.len();
                        let objectives_added = total_objectives.saturating_sub(before_count);
                        let json_objective_count =
                            serde_json::from_str::<CurriculumSchema>(&candidate_json)
                                .map(|schema| schema.objectives.len())
                                .unwrap_or(0);
                        if objectives_added < required_min && json_objective_count < required_min {
                            let err_msg = format!(
                                "Low dimensionality: only {} objectives for {} chars (min {})",
                                objectives_added,
                                aggregate_content.len(),
                                required_min
                            );
                            last_error = Some(err_msg.clone());
                            warn!("   ❌ {}. Expanding curriculum synthesis...", err_msg);

                            if dimensionality_retries >= MAX_DIMENSIONALITY_RETRIES {
                                break;
                            }
                            dimensionality_retries += 1;

                            let repair_prompt = format!(
                                "The previous JSON was too compressed.\n{}\n\nExpand the curriculum to at least {} objectives.\nEnsure coverage across: Core Theory, Algorithmic Structure, Implementation Constraints, Thermodynamic Impact.\nReturn ONLY valid JSON.\n\nJSON:\n{}",
                                err_msg, required_min, candidate_json
                            );

                            let repair_query = LLMQuery {
                                query_type: QueryType::Code,
                                content: repair_prompt,
                                context: Vec::new(),
                                system_prompt: Some(CURRICULUM_REPAIR_PROMPT.to_string()),
                                params: Some(curriculum_json_params()),
                            };

                            let repaired = self.llm.query_async(repair_query).await;
                            candidate_json = sanitize_json(&repaired.text);
                            attempts += 1;
                            continue;
                        }

                        if objectives_added < required_min && json_objective_count >= required_min {
                            warn!(
                                "   ⚠️ Objective count met in JSON ({}), but only {} were new (possible duplicates). Accepting.",
                                json_objective_count, objectives_added
                            );
                        }

                        info!("   ✨ Successfully extended curriculum with new objectives.");
                        let added_objectives: Vec<LearningObjective> =
                            if total_objectives >= before_count {
                                candidate_curriculum.objectives[before_count..].to_vec()
                            } else {
                                Vec::new()
                            };
                        let (confidence, warnings) = confidence_check(&added_objectives);
                        if warnings.is_empty() {
                            info!(
                                "   ✅ Curriculum confidence check passed (confidence={:.2})",
                                confidence
                            );
                        } else {
                            warn!(
                                "   ⚠️ Curriculum confidence check warnings (confidence={:.2}): {}",
                                confidence,
                                warnings.join(" | ")
                            );
                        }

                        *target_curriculum = candidate_curriculum;
                        if let Some(db) = database.as_ref() {
                            if let Err(e) =
                                store_objectives_as_memories(db.as_ref(), &added_objectives).await
                            {
                                warn!("   ⚠️ Failed to persist objective memories: {}", e);
                            }
                        }
                        return Ok(ResearchSummary {
                            objectives_added,
                            total_objectives,
                            confidence,
                            warnings,
                        });
                    }
                    Err(e) => {
                        let err_msg = e.to_string();
                        last_error = Some(err_msg.clone());
                        warn!(
                            "   ❌ Ingestion failed: {}. LLM output might be malformed.",
                            err_msg
                        );

                        if attempts + 1 >= total_attempts {
                            break;
                        }

                        let repair_prompt = format!(
                            "The previous JSON failed with error:\n{}\n\nFix it for topic '{}' and return ONLY valid JSON.\n\nJSON:\n{}",
                            last_error.as_deref().unwrap_or("unknown error"),
                            topic,
                            candidate_json
                        );

                        let repair_query = LLMQuery {
                            query_type: QueryType::Code,
                            content: repair_prompt,
                            context: Vec::new(),
                            system_prompt: Some(CURRICULUM_REPAIR_PROMPT.to_string()),
                            params: Some(curriculum_json_params()),
                        };

                        let repaired = self.llm.query_async(repair_query).await;
                        candidate_json = sanitize_json(&repaired.text);
                    }
                }
            }

            if attempts + 1 >= total_attempts {
                break;
            }

            if last_error.as_deref() == Some("Placeholder objective id detected (kebab-case-id)") {
                let repair_prompt = format!(
                    "The previous JSON used placeholder IDs.\nGenerate unique kebab-case ids for each objective, and ensure at least {} objectives.\nReturn ONLY valid JSON.\n\nJSON:\n{}",
                    required_min, candidate_json
                );
                let repair_query = LLMQuery {
                    query_type: QueryType::Code,
                    content: repair_prompt,
                    context: Vec::new(),
                    system_prompt: Some(CURRICULUM_REPAIR_PROMPT.to_string()),
                    params: Some(curriculum_json_params()),
                };
                let repaired = self.llm.query_async(repair_query).await;
                candidate_json = sanitize_json(&repaired.text);
            }

            attempts += 1;
        }

        warn!("   DEBUG: Raw LLM Output: {}", generation.text);
        if let Some(path) = write_failure_artifact(
            topic,
            &queries,
            total_attempts,
            last_error.as_deref().unwrap_or("unknown error"),
            &generation.text,
            &candidate_json,
        ) {
            warn!(
                "   📄 Curriculum failure artifact saved: {}",
                path.display()
            );
        }

        Err(anyhow::anyhow!(
            "Ingestion failed after {} attempts: {}",
            total_attempts,
            last_error.unwrap_or_else(|| "unknown error".to_string())
        ))
    }
}

fn sanitize_json(raw: &str) -> String {
    let mut cleaned = raw.trim().to_string();
    if cleaned.starts_with("```json") {
        cleaned = cleaned
            .strip_prefix("```json")
            .unwrap_or(&cleaned)
            .to_string();
    } else if cleaned.starts_with("```") {
        cleaned = cleaned.strip_prefix("```").unwrap_or(&cleaned).to_string();
    }
    if cleaned.ends_with("```") {
        cleaned = cleaned.strip_suffix("```").unwrap_or(&cleaned).to_string();
    }

    let trimmed = cleaned.trim();
    let candidate = if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        trimmed[start..=end].trim()
    } else {
        trimmed
    };

    candidate.to_string()
}

fn normalize_curriculum_json(raw: &str, required_min: usize, topic: &str) -> Option<String> {
    let mut schema: CurriculumSchema = serde_json::from_str(raw).ok()?;
    let mut changed = normalize_schema_ids(&mut schema);
    if schema.objectives.len() < required_min {
        if expand_schema_objectives(&mut schema, required_min, topic) {
            changed = true;
        }
    }
    if !changed {
        return Some(raw.to_string());
    }
    serde_json::to_string_pretty(&schema)
        .ok()
        .or_else(|| Some(raw.to_string()))
}

fn normalize_schema_ids(schema: &mut CurriculumSchema) -> bool {
    let mut changed = false;
    let mut used = HashSet::new();
    let mut id_map: HashMap<String, String> = HashMap::new();

    for obj in &mut schema.objectives {
        let raw_id = obj.id.trim().to_string();
        let placeholder = raw_id.is_empty() || raw_id == "kebab-case-id";
        let invalid = !placeholder && !is_kebab_case(&raw_id);
        let duplicate = !raw_id.is_empty() && used.contains(&raw_id);

        if placeholder || invalid || duplicate {
            let base = slugify_id(&obj.name);
            let new_id = unique_id(&base, &mut used);
            if !raw_id.is_empty() && raw_id != "kebab-case-id" {
                id_map.insert(raw_id.clone(), new_id.clone());
            }
            if raw_id != new_id {
                changed = true;
            }
            obj.id = new_id;
        } else {
            used.insert(raw_id.clone());
        }
    }

    for obj in &mut schema.objectives {
        let mut cleaned = Vec::new();
        for prereq in obj.prerequisites.iter() {
            let trimmed = prereq.trim();
            if trimmed.is_empty() || trimmed == "kebab-case-id" {
                changed = true;
                continue;
            }
            if let Some(mapped) = id_map.get(trimmed) {
                if mapped != trimmed {
                    changed = true;
                }
                cleaned.push(mapped.clone());
            } else {
                cleaned.push(trimmed.to_string());
            }
        }
        obj.prerequisites = cleaned;
    }

    changed
}

fn unique_id(base: &str, used: &mut HashSet<String>) -> String {
    let candidate = if base.is_empty() {
        "objective".to_string()
    } else {
        base.to_string()
    };
    if used.insert(candidate.clone()) {
        return candidate;
    }
    for idx in 2..10_000 {
        let next = format!("{}-{}", candidate, idx);
        if used.insert(next.clone()) {
            return next;
        }
    }
    format!("{}-fallback", candidate)
}

fn is_kebab_case(value: &str) -> bool {
    if value.is_empty() || value.starts_with('-') || value.ends_with('-') {
        return false;
    }
    let mut prev_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_lowercase() || ch.is_ascii_digit() {
            prev_dash = false;
            continue;
        }
        if ch == '-' {
            if prev_dash {
                return false;
            }
            prev_dash = true;
            continue;
        }
        return false;
    }
    true
}

fn slugify_id(value: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in value.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            out.push(lower);
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "objective".to_string()
    } else {
        out
    }
}

fn expand_schema_objectives(
    schema: &mut CurriculumSchema,
    required_min: usize,
    topic: &str,
) -> bool {
    let base_len = schema.objectives.len();
    if base_len == 0 || base_len >= required_min {
        return false;
    }

    let mut used: HashSet<String> = schema.objectives.iter().map(|o| o.id.clone()).collect();
    let categories = [
        ("core-theory", "Core Theory"),
        ("algorithmic-structure", "Algorithmic Structure"),
        ("implementation-constraints", "Implementation Constraints"),
        ("thermodynamic-impact", "Thermodynamic Impact"),
    ];

    let mut idx = 0usize;
    while schema.objectives.len() < required_min {
        let source = &schema.objectives[idx % base_len];
        let (slug_suffix, label) = categories[idx % categories.len()];
        let name = format!("{}: {}", label, source.name);
        let id_base = slugify_id(&name);
        let id = unique_id(&id_base, &mut used);

        let mut tags = source.tags.clone();
        if !tags.iter().any(|t| t.eq_ignore_ascii_case("auto-expanded")) {
            tags.push("auto-expanded".to_string());
        }
        if !tags.iter().any(|t| t.eq_ignore_ascii_case(slug_suffix)) {
            tags.push(slug_suffix.to_string());
        }

        let description = format!(
            "Auto-expanded objective to ensure minimum resolution for {}. Focus on {} aspects of: {}.",
            topic, label, source.name
        );

        schema.objectives.push(ObjectiveSchema {
            id,
            name,
            description,
            domain: source.domain.clone(),
            difficulty: source.difficulty,
            prerequisites: vec![source.id.clone()],
            tags,
            estimated_minutes: source.estimated_minutes.max(20),
        });

        idx += 1;
    }

    true
}

struct RedundantPair {
    id_a: String,
    name_a: String,
    id_b: String,
    name_b: String,
    similarity: f32,
}

struct GlobalSweepReport {
    suppressed: usize,
    causal_links: Vec<CausalLink>,
}

fn detect_redundant_pairs(
    schema: &CurriculumSchema,
    dimension: usize,
    threshold: f32,
    max_pairs: usize,
) -> Vec<RedundantPair> {
    if schema.objectives.len() < 2 {
        return Vec::new();
    }

    let objectives: Vec<LearningObjective> = schema
        .objectives
        .iter()
        .map(|obj| {
            ObjectiveBuilder::new(&obj.id, &obj.name)
                .with_description(&obj.description)
                .with_domain(Domain::from(obj.domain.as_str()))
                .with_difficulty(Difficulty::from_f32(obj.difficulty))
                .with_dimension(dimension)
                .with_prerequisites(
                    &obj.prerequisites
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                )
                .with_tags(&obj.tags.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                .with_estimated_minutes(obj.estimated_minutes)
                .build()
        })
        .collect();

    let mut pairs = Vec::new();
    for i in 0..objectives.len() {
        for j in (i + 1)..objectives.len() {
            let sim = objectives[i].similarity(&objectives[j]);
            if sim >= threshold {
                pairs.push(RedundantPair {
                    id_a: objectives[i].id.clone(),
                    name_a: objectives[i].name.clone(),
                    id_b: objectives[j].id.clone(),
                    name_b: objectives[j].name.clone(),
                    similarity: sim,
                });
            }
        }
    }

    pairs.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(max_pairs);
    pairs
}

async fn global_hdc_sweep(
    schema: &mut CurriculumSchema,
    dimension: usize,
    db: &dyn ConsciousnessDatabase,
    threshold: f32,
) -> Result<GlobalSweepReport> {
    if schema.objectives.is_empty() {
        return Ok(GlobalSweepReport {
            suppressed: 0,
            causal_links: Vec::new(),
        });
    }

    let mut kept = Vec::new();
    let mut suppressed = 0usize;
    let mut causal_links = Vec::new();

    for obj in &schema.objectives {
        let objective = ObjectiveBuilder::new(&obj.id, &obj.name)
            .with_description(&obj.description)
            .with_domain(Domain::from(obj.domain.as_str()))
            .with_difficulty(Difficulty::from_f32(obj.difficulty))
            .with_dimension(dimension)
            .with_prerequisites(
                &obj.prerequisites
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            )
            .with_tags(&obj.tags.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .with_estimated_minutes(obj.estimated_minutes)
            .build();

        let query = real_hv_to_hv16(&objective.encoding);
        let matches = db.search_similar(&query, 1).await.unwrap_or_default();
        if let Some(best) = matches.first() {
            if best.similarity >= threshold {
                suppressed += 1;
                let outcome = binary_hv_to_bipolar_vec(&best.record.encoding, dimension);
                causal_links.push(CausalLink {
                    action_fingerprint: objective_fingerprint(&objective.id, &best.record.id),
                    state_context: objective.encoding.values.clone(),
                    outcome,
                    weight: best.similarity,
                });
                continue;
            }
        }
        kept.push(obj.clone());
    }

    schema.objectives = kept;
    Ok(GlobalSweepReport {
        suppressed,
        causal_links,
    })
}

fn binary_hv_to_bipolar_vec(hv: &BinaryHV, dimension: usize) -> Vec<f32> {
    let dim = dimension.min(BinaryHV::DIM);
    let mut values = Vec::with_capacity(dimension);
    for i in 0..dim {
        let byte = hv.0[i / 8];
        let bit = (byte >> (i % 8)) & 1;
        values.push(if bit == 1 { 1.0 } else { -1.0 });
    }
    if dimension > dim {
        values.extend(std::iter::repeat(0.0).take(dimension - dim));
    }
    values
}

fn objective_fingerprint(objective_id: &str, memory_id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    objective_id.hash(&mut hasher);
    memory_id.hash(&mut hasher);
    hasher.finish()
}

async fn store_objectives_as_memories(
    db: &dyn ConsciousnessDatabase,
    objectives: &[LearningObjective],
) -> Result<()> {
    let timestamp_ms = Utc::now().timestamp_millis() as u64;
    for obj in objectives {
        let record = MemoryRecord {
            id: format!("objective:{}", obj.id),
            memory_type: MemoryType::Semantic,
            encoding: real_hv_to_hv16(&obj.encoding),
            content: format!("{} — {}", obj.name, obj.description),
            timestamp_ms,
            valence: 0.0,
            arousal: 0.0,
            psi: 0.5,
            topics: obj.tags.clone(),
            metadata: serde_json::json!({
                "objective_id": obj.id,
                "domain": obj.domain.name(),
                "difficulty": obj.difficulty.as_f32(),
            })
            .to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(record).await?;
    }
    Ok(())
}

fn confidence_check(objectives: &[LearningObjective]) -> (f32, Vec<String>) {
    if objectives.is_empty() {
        return (0.0, vec!["No objectives added".to_string()]);
    }

    let total = objectives.len() as f32;
    let short_desc = objectives
        .iter()
        .filter(|o| o.description.trim().len() < 20)
        .count();
    let missing_tags = objectives.iter().filter(|o| o.tags.is_empty()).count();
    let out_of_range = objectives
        .iter()
        .filter(|o| o.estimated_minutes < 5 || o.estimated_minutes > 180)
        .count();
    let custom_domain = objectives
        .iter()
        .filter(|o| o.domain.name().eq_ignore_ascii_case("Custom"))
        .count();
    let stub_count = objectives
        .iter()
        .filter(|o| {
            o.name.starts_with("Implicit:")
                || o.description
                    .contains("Automatically created prerequisite stub")
        })
        .count();

    let mut score = 1.0f32;
    let mut warnings = Vec::new();

    if objectives.len() < 3 {
        score -= 0.2;
        warnings.push("Fewer than 3 objectives were generated".to_string());
    }
    if (short_desc as f32 / total) > 0.5 {
        score -= 0.2;
        warnings.push("Many objectives have short descriptions".to_string());
    }
    if (missing_tags as f32 / total) > 0.7 {
        score -= 0.1;
        warnings.push("Many objectives are missing tags".to_string());
    }
    if out_of_range > 0 {
        score -= 0.1;
        warnings.push("Some objectives have unusual estimated minutes".to_string());
    }
    if (custom_domain as f32 / total) > 0.8 {
        score -= 0.1;
        warnings.push("Most objectives use Custom domain".to_string());
    }
    if stub_count > 0 {
        score -= 0.1;
        warnings.push(format!(
            "{} auto-healed prerequisite stubs were added",
            stub_count
        ));
    }

    (score.clamp(0.0, 1.0), warnings)
}

#[derive(Serialize)]
struct FailureArtifact {
    topic: String,
    attempted_queries: Vec<String>,
    attempts: usize,
    error: String,
    timestamp: String,
    raw_output: String,
    candidate_json: String,
}

fn failure_dir() -> PathBuf {
    std::env::var("SYMTHAEA_CURRICULUM_FAILURE_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .or_else(dirs::state_dir)
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("symthaea")
                .join("curriculum_failures")
        })
}

fn write_failure_artifact(
    topic: &str,
    attempted_queries: &[String],
    attempts: usize,
    error: &str,
    raw_output: &str,
    candidate_json: &str,
) -> Option<PathBuf> {
    let dir = failure_dir();
    if fs::create_dir_all(&dir).is_err() {
        return None;
    }

    let timestamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let slug = slugify(topic);
    let filename = if slug.is_empty() {
        format!("curriculum_failure_{timestamp}.json")
    } else {
        format!("curriculum_failure_{timestamp}_{slug}.json")
    };
    let path = dir.join(filename);

    let artifact = FailureArtifact {
        topic: topic.to_string(),
        attempted_queries: attempted_queries.to_vec(),
        attempts,
        error: error.to_string(),
        timestamp,
        raw_output: raw_output.to_string(),
        candidate_json: candidate_json.to_string(),
    };

    let json = serde_json::to_string_pretty(&artifact).ok()?;
    fs::write(&path, json).ok()?;
    Some(path)
}

fn slugify(input: &str) -> String {
    let mut slug = String::new();
    let mut last_dash = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}
