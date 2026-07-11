// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-curriculum-sync: turn new curriculum objectives into Broca training pairs.
//!
//! Reads the persisted curriculum (`SYMTHAEA_CURRICULUM_PATH`, same store
//! `CurriculumExtender` writes to), finds objectives not yet consumed
//! (tracked externally in `BROCA_BRIDGE_STATE_PATH`, never written onto the
//! curriculum schema itself), asks the configured LLM to synthesize a few
//! explanatory demonstration texts per objective, encodes them into Broca's
//! canonical `TrainingPair` format, and writes the accepted pairs plus a
//! provenance manifest atomically. This is one stage of a bridge that never
//! promotes a checkpoint automatically — see
//! `/home/tstoltz/.claude/plans/fuzzy-beaming-brook.md` for the full design.
//!
//! Usage:
//!   broca-curriculum-sync              # synthesize pairs for all new objectives
//!   broca-curriculum-sync --count-only # print how many objectives are new, no LLM calls

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use symthaea::language::llm_backend::LLMBackend;
use symthaea::language::{LLMOrgan, LLMOrganConfig, LLMQuery, OllamaBackend, QueryType};
use symthaea_broca::BpeTokenizer;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::generator::BrocaGenerator;
use symthaea_broca::training::{TrainingDataset, TrainingPair};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

// Local mirror of `symthaea::school::curriculum_loader`'s on-disk JSON shape
// (`CurriculumStore { meta, curriculum: CurriculumSpec }`, `ObjectiveSpec`).
// Deliberately does NOT depend on `symthaea::school` — that module requires
// the `school_learning` feature, which does not currently compile on this
// branch (unrelated pre-existing breakage: duplicated definitions in
// engineering_curriculum.rs, a missing symthaea_sim_bridge dependency, a
// stale symthaea_runtime crate reference, and a couple of struct-shape
// mismatches in code_learning.rs — see task notes). The on-disk field names
// are a stable interchange format (also consumed by CurriculumExtender),
// so parsing them directly here is a reasonable, low-coupling workaround
// rather than a hack: it reads the same file, just without dragging in a
// currently-broken compile closure to do it.
#[derive(Debug, Clone, Deserialize)]
struct ObjectiveSpec {
    id: String,
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_domain")]
    domain: String,
    #[serde(default = "default_difficulty")]
    difficulty: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "default_minutes")]
    estimated_minutes: u32,
}

fn default_domain() -> String {
    "NixOS".to_string()
}
fn default_difficulty() -> String {
    "Intermediate".to_string()
}
fn default_minutes() -> u32 {
    15
}

#[derive(Debug, Clone, Deserialize)]
struct CurriculumSpec {
    #[serde(default)]
    objectives: Vec<ObjectiveSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct CurriculumStore {
    curriculum: CurriculumSpec,
}

/// Mirrors `Difficulty::as_f32()` in `symthaea::school::objective` — kept in
/// sync by hand since this bin intentionally avoids depending on that module.
fn difficulty_as_f32(difficulty: &str) -> f32 {
    match difficulty {
        "Beginner" => 0.1,
        "Elementary" => 0.3,
        "Intermediate" => 0.5,
        "Advanced" => 0.7,
        "Expert" => 0.9,
        other => other.parse::<f32>().unwrap_or(0.5),
    }
}

/// Number of distinct demonstration texts to request per objective.
const PAIRS_PER_OBJECTIVE: usize = 4;
/// Rejection thresholds for the mechanical content filter.
const MIN_TEXT_LEN: usize = 8;
const MAX_TEXT_LEN: usize = 2000;
const REFUSAL_PHRASES: &[&str] = &[
    "i cannot",
    "i can't",
    "as an ai",
    "as a language model",
    "i'm not able to",
    "i am not able to",
];

const SYNTHESIS_SYSTEM_PROMPT: &str = r#"You are Symthaea's TRAINING TEXT SYNTHESIZER.

Given a learning objective (name, domain, difficulty, description, tags), write
several short, distinct, self-contained explanations of the concept — as if
articulating understanding of it in conversation. Each explanation should be
2-4 sentences, factual and concrete, with no meta-commentary about the task,
no headers, no numbering, no disclaimers.

Separate each explanation with a line containing only: ---

Do not repeat the objective's own description verbatim; explain the concept
in your own words, from a few different angles."#;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BridgeState {
    trained_objective_ids: HashSet<String>,
    last_run_utc: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ObjectiveManifestEntry {
    objective_id: String,
    objective_hash: String,
    llm_backend: String,
    model: String,
    prompt_hash: String,
    generated_at: String,
    accepted_pair_hashes: Vec<String>,
    rejected_pair_hashes: Vec<String>,
    filter_version: String,
}

/// One synthesized text per objective is held out of training entirely and
/// saved here instead — never seen during fine-tuning, so later generating
/// from `channels` on a candidate checkpoint and comparing against
/// `reference_text` is genuine evidence of whether the objective's content
/// actually landed, not just a regression-absence check. See
/// broca_topic_coverage.rs (symthaea-broca), which consumes this file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HoldoutEntry {
    objective_id: String,
    channels: Vec<f32>,
    reference_text: String,
    /// Same content-conditioning signal stored on this objective's
    /// `TrainingPair`s (see `TrainingPair::semantic_hv`) — held-out
    /// generation should be conditioned identically to training, or a
    /// missing signal here would silently reproduce the collapse it's
    /// meant to catch. See `broca_topic_coverage.rs`.
    #[serde(default)]
    semantic_hv: Option<Vec<f32>>,
}

fn state_path() -> PathBuf {
    std::env::var("BROCA_BRIDGE_STATE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/broca_curriculum_bridge_state.json"))
}

/// The checkpoint whose tokenizer + token-embedding table is used to encode
/// each objective's `semantic_hv` (read-only — never trained or saved here).
/// Defaults to the production checkpoint so the encoding is consistent with
/// what `--resume` will fine-tune from this cycle.
fn synthesis_checkpoint_path() -> PathBuf {
    std::env::var("BROCA_SYNTHESIS_CHECKPOINT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("crates/domains/symthaea-broca/data/models/broca-checkpoint-latest.bin")
        })
}

/// Encode `text` as a bag-of-token-embeddings HV, mirroring
/// `BrocaBridge::encode_knowledge_context`'s pattern exactly (tokenize with
/// Broca's own tokenizer, look up each token's embedding in Broca's own
/// embedding table, bundle into one HV) — reused here rather than inventing
/// new text-encoding infrastructure. Returns `None` if `text` tokenizes to
/// nothing (e.g. empty input).
fn encode_semantic_hv(generator: &BrocaGenerator, text: &str) -> Option<Vec<f32>> {
    let ids = generator.tokenizer().encode(text);
    let all_embs = generator.controller().token_embeddings();
    let embs: Vec<&ContinuousHV> = ids
        .iter()
        .filter_map(|&id| all_embs.get(id as usize))
        .collect();
    if embs.is_empty() {
        None
    } else {
        Some(ContinuousHV::bundle(&embs).values)
    }
}

fn curriculum_path() -> PathBuf {
    std::env::var("SYMTHAEA_CURRICULUM_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::data_local_dir()
                .or_else(dirs::state_dir)
                .or_else(dirs::home_dir)
                .unwrap_or_else(|| PathBuf::from("."))
                .join("symthaea")
                .join("curriculum.json")
        })
}

fn curriculum_derived_dir() -> PathBuf {
    std::env::var("BROCA_CURRICULUM_DERIVED_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/curriculum-derived"))
}

fn load_state(path: &Path) -> BridgeState {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Write `contents` to `path` atomically: write to a sibling temp file, then rename.
/// Renaming is atomic on the same filesystem, so a crash between these two calls
/// never leaves a partially-written file at `path`.
fn atomic_write(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    let tmp = path.with_extension(format!(
        "{}.tmp{}",
        path.extension().and_then(|e| e.to_str()).unwrap_or(""),
        std::process::id()
    ));
    std::fs::write(&tmp, contents)
        .with_context(|| format!("writing temp file {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Cheap, non-cryptographic hash for provenance/dedup purposes — not a security
/// boundary, just an audit trail, so std's DefaultHasher is sufficient and avoids
/// pulling in a new required dependency.
fn hash_str(s: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn objective_hash(obj: &ObjectiveSpec) -> String {
    hash_str(&format!(
        "{}|{}|{}|{}|{}|{}|{}",
        obj.id,
        obj.name,
        obj.description,
        obj.domain,
        obj.difficulty,
        obj.estimated_minutes,
        obj.tags.join(",")
    ))
}

/// Mechanical content filter: rejects empty/degenerate/refusal-shaped/duplicate
/// text before it can become permanent training signal. Not fact-checking —
/// that residual risk is mitigated by the human promotion gate downstream, not here.
fn rejection_reason(text: &str, seen_hashes: &HashSet<String>) -> Option<&'static str> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Some("empty");
    }
    if trimmed.len() < MIN_TEXT_LEN {
        return Some("too_short");
    }
    if trimmed.len() > MAX_TEXT_LEN {
        return Some("too_long");
    }
    let lower = trimmed.to_lowercase();
    if REFUSAL_PHRASES.iter().any(|p| lower.contains(p)) {
        return Some("refusal_phrase");
    }
    if seen_hashes.contains(&hash_str(&lower)) {
        return Some("duplicate");
    }
    None
}

/// Map an objective's difficulty into the epistemic-cube channel setters,
/// following the same pattern as `gen_epistemic_training.rs::build_channels`:
/// harder objectives get a lower certainty tier and a higher required psi.
fn build_channels(obj: &ObjectiveSpec) -> ThoughtChannels {
    let mut ch = ThoughtChannels::default();

    // Intent: fixed "explain/inform" slot in the 8-wide one-hot intent block.
    for i in 0..8 {
        ch.channels[i] = if i == 1 { 1.0 } else { 0.0 };
    }

    let difficulty = difficulty_as_f32(&obj.difficulty).clamp(0.0, 1.0);
    // Easier objectives -> higher certainty tier (4 = most certain); harder -> lower.
    let e_tier = ((1.0 - difficulty) * 4.0).round().clamp(0.0, 4.0) as u8;
    let epistemic_ordinal = match e_tier {
        4 => 0.0,
        3 => 0.5,
        2 => 2.0,
        1 => 3.0,
        _ => 3.5,
    };
    ch.set_epistemic(epistemic_ordinal);

    let psi = (0.3 + difficulty * 0.6).clamp(0.0, 1.0);
    ch.set_consciousness(psi, psi * 0.8, difficulty);
    ch.set_emotion(0.15, (0.3 + difficulty * 0.3).clamp(0.0, 1.0), 0.6);

    let quality = (1.0 - difficulty * 0.4).clamp(0.0, 1.0);
    // n/m tiers have no strong signal from a curriculum objective alone; hold
    // them at a neutral mid-value rather than inventing a mapping with no basis.
    ch.set_epistemic_cube(e_tier, 2, 2, difficulty, quality);

    ch
}

fn build_llm_organ() -> LLMOrgan {
    let backend: Arc<dyn LLMBackend> = Arc::new(OllamaBackend::new());
    LLMOrgan::with_backend(LLMOrganConfig::default(), backend)
}

async fn synthesize_for_objective(
    llm: &mut LLMOrgan,
    obj: &ObjectiveSpec,
    tokenizer: &BpeTokenizer,
    corpus_hashes: &HashSet<String>,
    model_name: &str,
    semantic_generator: &BrocaGenerator,
) -> Result<(
    Vec<TrainingPair>,
    Option<HoldoutEntry>,
    ObjectiveManifestEntry,
)> {
    // Computed once per objective from its actual identity (name + description
    // + domain + tags), not from difficulty — this is the content-conditioning
    // signal `build_channels()` structurally cannot carry (see Part 3 of the
    // plan file: `build_channels()` collapses to ~2 distinct vectors across a
    // real curriculum batch because it derives everything from `difficulty`).
    let semantic_text = format!(
        "{} {} {} {}",
        obj.name,
        obj.description,
        obj.domain,
        obj.tags.join(" ")
    );
    let semantic_hv = encode_semantic_hv(semantic_generator, &semantic_text);
    let prompt = format!(
        "Objective: {}\nDomain: {}\nDifficulty: {} ({:.2})\nDescription: {}\nTags: {}\n\nWrite {} distinct explanations as instructed.",
        obj.name,
        obj.domain,
        obj.difficulty,
        difficulty_as_f32(&obj.difficulty),
        obj.description,
        obj.tags.join(", "),
        PAIRS_PER_OBJECTIVE
    );
    let prompt_hash = hash_str(&format!("{SYNTHESIS_SYSTEM_PROMPT}\n{prompt}"));

    let query = LLMQuery {
        query_type: QueryType::Analysis,
        content: prompt,
        context: Vec::new(),
        system_prompt: Some(SYNTHESIS_SYSTEM_PROMPT.to_string()),
        params: None,
    };
    let result = llm.query_async(query).await;

    // Collect (text, channels) first; the LAST accepted one is held out of
    // training entirely (see HoldoutEntry) rather than turned into a
    // TrainingPair, so there's at least one genuinely-unseen example per
    // objective to check learning against later.
    let mut accepted_texts = Vec::new();
    let mut accepted_hashes = Vec::new();
    let mut rejected_hashes = Vec::new();
    let mut seen_this_objective: HashSet<String> = HashSet::new();

    for chunk in result.text.split("---") {
        let text = chunk.trim().to_string();
        if text.is_empty() {
            continue;
        }
        let h = hash_str(&text);
        let lower_hash = hash_str(&text.to_lowercase());

        let reason = rejection_reason(&text, corpus_hashes)
            .or_else(|| rejection_reason(&text, &seen_this_objective));

        if let Some(reason) = reason {
            rejected_hashes.push(format!("{h}:{reason}"));
            continue;
        }

        seen_this_objective.insert(lower_hash);
        accepted_texts.push(text);
        accepted_hashes.push(h);
    }

    // Hold out the last accepted text when there are enough to spare at
    // least one for training; with only one accepted text, keep it for
    // training rather than holding out the objective's only example.
    let holdout = if accepted_texts.len() >= 2 {
        let holdout_text = accepted_texts.pop().unwrap();
        let channels = build_channels(obj);
        Some(HoldoutEntry {
            objective_id: obj.id.clone(),
            channels: channels.channels.to_vec(),
            reference_text: holdout_text,
            semantic_hv: semantic_hv.clone(),
        })
    } else {
        None
    };

    let accepted: Vec<TrainingPair> = accepted_texts
        .into_iter()
        .map(|text| {
            let mut pair = TrainingPair::new(build_channels(obj), text, tokenizer);
            pair.semantic_hv = semantic_hv.clone();
            pair
        })
        .collect();

    let manifest_entry = ObjectiveManifestEntry {
        objective_id: obj.id.clone(),
        objective_hash: objective_hash(obj),
        llm_backend: "ollama".to_string(),
        model: model_name.to_string(),
        prompt_hash,
        generated_at: chrono::Utc::now().to_rfc3339(),
        accepted_pair_hashes: accepted_hashes,
        rejected_pair_hashes: rejected_hashes,
        filter_version: "v1".to_string(),
    };

    Ok((accepted, holdout, manifest_entry))
}

/// Existing corpus text hashes, so synthesized pairs that just restate what's
/// already in the base corpus or a prior curriculum-derived batch are rejected
/// as duplicates rather than silently added again.
fn load_existing_corpus_hashes() -> HashSet<String> {
    let mut hashes = HashSet::new();
    let base_corpus = std::env::var("BROCA_BASE_CORPUS_PATH").unwrap_or_else(|_| {
        "crates/domains/symthaea-broca/data/train-combined-v8.jsonl".to_string()
    });
    if let Ok(dataset) = TrainingDataset::from_jsonl(&base_corpus) {
        for pair in &dataset.pairs {
            hashes.insert(hash_str(&pair.target_text.to_lowercase()));
        }
    }
    if let Ok(entries) = std::fs::read_dir(curriculum_derived_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                if let Ok(dataset) = TrainingDataset::from_jsonl(&path.to_string_lossy()) {
                    for pair in &dataset.pairs {
                        hashes.insert(hash_str(&pair.target_text.to_lowercase()));
                    }
                }
            }
        }
    }
    hashes
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".parse().unwrap()),
        )
        .init();

    let count_only = std::env::args().any(|a| a == "--count-only");

    let curriculum_file = curriculum_path();
    if !curriculum_file.exists() {
        println!(
            "[broca-curriculum-sync] no curriculum found at {}; nothing to do",
            curriculum_file.display()
        );
        return Ok(());
    }
    let curriculum_json = std::fs::read_to_string(&curriculum_file)
        .with_context(|| format!("reading curriculum from {}", curriculum_file.display()))?;
    let store: CurriculumStore = serde_json::from_str(&curriculum_json)
        .with_context(|| format!("parsing curriculum JSON from {}", curriculum_file.display()))?;

    let state_file = state_path();
    let mut state = load_state(&state_file);

    let new_objectives: Vec<&ObjectiveSpec> = store
        .curriculum
        .objectives
        .iter()
        .filter(|obj| !state.trained_objective_ids.contains(&obj.id))
        .collect();

    if count_only {
        println!("{}", new_objectives.len());
        return Ok(());
    }

    if new_objectives.is_empty() {
        println!("[broca-curriculum-sync] no new objectives since last run");
        return Ok(());
    }

    println!(
        "[broca-curriculum-sync] synthesizing training pairs for {} new objective(s)",
        new_objectives.len()
    );

    let model_name =
        std::env::var("SYMTHAEA_LLM_MODEL").unwrap_or_else(|_| "gemma4:e2b".to_string());
    let mut llm = build_llm_organ();
    let tokenizer = BpeTokenizer::default_minimal();
    let corpus_hashes = load_existing_corpus_hashes();

    // Loaded once, read-only, purely for its tokenizer + token-embedding
    // table (see `encode_semantic_hv`) — never trained or saved by this
    // binary. Using the current production checkpoint keeps each cycle's
    // semantic encoding consistent with the weights `--resume` fine-tunes.
    let synth_checkpoint = synthesis_checkpoint_path();
    let genesis = GenesisSeed::from_phrase("symthaea luminous dynamics");
    let (semantic_generator, ..) = BrocaGenerator::from_checkpoint(&synth_checkpoint, &genesis)
        .with_context(|| {
            format!(
                "loading checkpoint {} for semantic_hv encoding (override with BROCA_SYNTHESIS_CHECKPOINT_PATH)",
                synth_checkpoint.display()
            )
        })?;

    let mut dataset = TrainingDataset::default();
    let mut manifest_entries = Vec::new();
    let mut consumed_ids = Vec::new();
    let mut holdout_entries = Vec::new();

    for obj in &new_objectives {
        let (pairs, holdout, entry) = synthesize_for_objective(
            &mut llm,
            obj,
            &tokenizer,
            &corpus_hashes,
            &model_name,
            &semantic_generator,
        )
        .await?;

        println!(
            "[broca-curriculum-sync]   {} -> {} accepted, {} rejected, holdout: {}",
            obj.id,
            entry.accepted_pair_hashes.len(),
            entry.rejected_pair_hashes.len(),
            holdout.is_some()
        );

        if !pairs.is_empty() {
            consumed_ids.push(obj.id.clone());
            dataset.pairs.extend(pairs);
        }
        if let Some(h) = holdout {
            holdout_entries.push(h);
        }
        manifest_entries.push(entry);
    }

    if dataset.pairs.is_empty() {
        println!("[broca-curriculum-sync] no pairs survived the content filter; state unchanged");
        return Ok(());
    }

    let derived_dir = curriculum_derived_dir();
    let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let jsonl_path = derived_dir.join(format!("{timestamp}.jsonl"));
    let manifest_path = derived_dir.join(format!("{timestamp}.manifest.json"));
    // Deliberately in a separate holdout/ subdirectory, not just a distinct
    // suffix: the cycle script globs "$CURRICULUM_DERIVED_DIR"/*.jsonl to
    // build the training merge, and *.holdout.jsonl would still match that
    // glob (it ends in .jsonl too) — putting it in its own subdirectory
    // means the non-recursive glob can never accidentally pull held-out
    // text into training, with no filter logic to keep in sync.
    let holdout_path = derived_dir
        .join("holdout")
        .join(format!("{timestamp}.holdout.jsonl"));

    // Write the JSONL via TrainingDataset::to_jsonl into a temp path, then
    // atomically rename both it and the manifest into place. Only after both
    // renames succeed do we update the state file — an interrupted run must
    // never mark an objective consumed while losing its generated pairs.
    let jsonl_contents = {
        let mut out = String::new();
        for pair in &dataset.pairs {
            out.push_str(&serde_json::to_string(pair)?);
            out.push('\n');
        }
        out
    };
    atomic_write(&jsonl_path, &jsonl_contents)?;
    atomic_write(
        &manifest_path,
        &serde_json::to_string_pretty(&manifest_entries)?,
    )?;
    if !holdout_entries.is_empty() {
        let holdout_contents = {
            let mut out = String::new();
            for entry in &holdout_entries {
                out.push_str(&serde_json::to_string(entry)?);
                out.push('\n');
            }
            out
        };
        atomic_write(&holdout_path, &holdout_contents)?;
    }

    state.trained_objective_ids.extend(consumed_ids);
    state.last_run_utc = Some(chrono::Utc::now().to_rfc3339());
    atomic_write(&state_file, &serde_json::to_string_pretty(&state)?)?;

    println!(
        "[broca-curriculum-sync] wrote {} pairs to {} (manifest: {}, holdout: {} entries)",
        dataset.pairs.len(),
        jsonl_path.display(),
        manifest_path.display(),
        holdout_entries.len()
    );

    Ok(())
}
