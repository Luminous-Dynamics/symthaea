// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-poetry-eval: batch generation + honest evaluation of Broca poetry.
//!
//! Implements Phase 2.4 (LLM-judge rubric) and Phase 2.5 (form-compliance
//! eval) of `ART_CULTURE_REVIEW_AND_PLAN_2026-07-06.md`.
//!
//! Generates real poems from a trained Broca checkpoint across the three
//! `PoeticForm` presets (haiku/tanka/free-verse), validates each one
//! mechanically via `creative_mode::validate_poem` (Phase 2.5 — no LLM
//! needed), and optionally sends each poem to a local Ollama model for a
//! 1-5 rubric judgement (Phase 2.4 — novelty/coherence/emotional
//! resonance/craft).
//!
//! Honesty discipline (matches `broca-topic-coverage` and the grounded
//! psych-bench creativity work): never fabricate a score. If the judge
//! model is unreachable, the report says so explicitly rather than
//! emitting zeros that look like real (bad) scores. If a single poem's
//! judge response can't be parsed, that poem's `judge_scores` is `None`,
//! not a default value.
//!
//! KNOWN LIMITATION: the LLM judge is itself an unvalidated proxy for
//! human aesthetic judgement — no correlation with human raters has been
//! measured for this rubric. Treat its scores as a rough, cheap signal,
//! not a stand-in for human evaluation. This is different from Phase 2.2
//! (RAT/DAT), which is grounded against published human norms.
//!
//! Usage:
//!   broca-poetry-eval --checkpoint <path> [--n-per-form 10] [--skip-judge]
//!       [--judge-model gemma4:e2b] [--json-out report.json]

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process;
use std::time::Duration;

use symthaea_broca::creative_mode::{CreativeGating, PoeticForm, ValidationError, validate_poem};
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::generator::BrocaGenerator;
use symthaea_core::genesis::GenesisSeed;

/// Known real checkpoint path (see `creative_bridge.rs::POETRY_CHECKPOINT_PATHS`).
/// Used as the default `--checkpoint` value for convenience; always overridable.
const DEFAULT_CHECKPOINT_PATH: &str =
    "crates/domains/symthaea-broca/data/models/broca-checkpoint-latest.bin";

/// Genesis phrase the shipped checkpoint was trained under. MUST match, or
/// checkpoint restore silently misaligns weights with inputs (see
/// `creative_bridge.rs::try_load_poetry_generator` for the full explanation).
const DEFAULT_GENESIS_PHRASE: &str = "symthaea luminous dynamics";

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Options {
    checkpoint_path: String,
    genesis_phrase: String,
    n_per_form: usize,
    seed: u64,
    max_gen_tokens: Option<usize>,
    skip_judge: bool,
    judge_model: String,
    judge_base_url: String,
    judge_max_tokens: usize,
    judge_timeout_secs: u64,
    json_out: Option<String>,
    allow_checkpoint_recovery: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            checkpoint_path: DEFAULT_CHECKPOINT_PATH.to_string(),
            genesis_phrase: DEFAULT_GENESIS_PHRASE.to_string(),
            n_per_form: 10,
            seed: 42,
            max_gen_tokens: None,
            skip_judge: false,
            judge_model: "gemma4:e2b".to_string(),
            judge_base_url: "http://127.0.0.1:11434".to_string(),
            judge_max_tokens: 400,
            judge_timeout_secs: 60,
            json_out: None,
            allow_checkpoint_recovery: false,
        }
    }
}

fn parse_args(args: &[String]) -> Result<Options> {
    let mut opts = Options::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                i += 1;
                opts.checkpoint_path = value(args, i, "--checkpoint")?.to_string();
            }
            "--genesis" => {
                i += 1;
                opts.genesis_phrase = value(args, i, "--genesis")?.to_string();
            }
            "--n-per-form" => {
                i += 1;
                opts.n_per_form = value(args, i, "--n-per-form")?.parse()?;
            }
            "--seed" => {
                i += 1;
                opts.seed = value(args, i, "--seed")?.parse()?;
            }
            "--max-gen-tokens" => {
                i += 1;
                opts.max_gen_tokens = Some(value(args, i, "--max-gen-tokens")?.parse()?);
            }
            "--skip-judge" => {
                opts.skip_judge = true;
            }
            "--judge-model" => {
                i += 1;
                opts.judge_model = value(args, i, "--judge-model")?.to_string();
            }
            "--judge-url" => {
                i += 1;
                opts.judge_base_url = value(args, i, "--judge-url")?.to_string();
            }
            "--judge-max-tokens" => {
                i += 1;
                opts.judge_max_tokens = value(args, i, "--judge-max-tokens")?.parse()?;
            }
            "--judge-timeout-secs" => {
                i += 1;
                opts.judge_timeout_secs = value(args, i, "--judge-timeout-secs")?.parse()?;
            }
            "--json-out" => {
                i += 1;
                opts.json_out = Some(value(args, i, "--json-out")?.to_string());
            }
            "--allow-checkpoint-recovery" => {
                opts.allow_checkpoint_recovery = true;
            }
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
        i += 1;
    }
    Ok(opts)
}

fn value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index)
        .map(|s| s.as_str())
        .with_context(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "Usage: broca-poetry-eval [--checkpoint PATH] [--genesis PHRASE] \
         [--n-per-form N] [--seed N] [--max-gen-tokens N] [--skip-judge] \
         [--judge-model NAME] [--judge-url URL] [--judge-max-tokens N] \
         [--judge-timeout-secs N] [--json-out PATH] [--allow-checkpoint-recovery]"
    );
}

// ─── Report schema ──────────────────────────────────────────────────────────

/// A single rubric dimension's judged score, 1-5, with a short justification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct RubricScore {
    score: f32,
    #[serde(default)]
    justification: Option<String>,
}

/// LLM-judge scores for one poem. Every dimension is independently optional:
/// a poem that only partially parses (e.g. the model reported craft but not
/// novelty) keeps the dimensions it has rather than being discarded whole.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
struct JudgeScores {
    #[serde(default)]
    novelty: Option<RubricScore>,
    #[serde(default)]
    coherence: Option<RubricScore>,
    #[serde(default)]
    emotional_resonance: Option<RubricScore>,
    #[serde(default)]
    craft: Option<RubricScore>,
}

impl JudgeScores {
    fn is_empty(&self) -> bool {
        self.novelty.is_none()
            && self.coherence.is_none()
            && self.emotional_resonance.is_none()
            && self.craft.is_none()
    }
}

/// One generated poem plus its mechanical validation and (optional) judge scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PoemRecord {
    form: String,
    index: usize,
    text: String,
    /// Real syllable-count validation from `creative_mode::validate_poem` —
    /// never reimplemented here, per the plan's explicit instruction.
    line_syllable_counts: Vec<usize>,
    target_syllable_counts: Vec<u8>,
    valid: bool,
    errors: Vec<ValidationError>,
    /// `None` when the judge was skipped, unavailable, or this poem's
    /// response failed to parse into any dimension — never a fabricated 0.
    judge_scores: Option<JudgeScores>,
}

/// Per-form form-compliance rollup (Phase 2.5 — no LLM required).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ErrorTypeCounts {
    too_few_lines: usize,
    too_many_lines: usize,
    wrong_syllable_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct FormComplianceStats {
    form: String,
    n: usize,
    compliant_count: usize,
    /// Fraction of poems with zero `ValidationError`s.
    compliance_rate: f32,
    /// Mean |actual - target| syllable count across all lines that have a
    /// target (i.e. excludes free verse, which has no target counts).
    /// `None` when no line in this form had a target to compare against.
    mean_syllable_deviation: Option<f32>,
    error_counts: ErrorTypeCounts,
}

/// Aggregate LLM-judge rollup (Phase 2.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JudgeAggregate {
    /// Whether the judge model was reachable at all. If `false`, every
    /// poem's `judge_scores` is `None` and the means below are all `None` —
    /// this must read as "judge unavailable", not "poems scored 0/5".
    available: bool,
    model: String,
    url: String,
    unavailable_reason: Option<String>,
    n_poems: usize,
    /// HTTP/network request failures (judge was available at startup but
    /// individual calls still failed transiently).
    n_request_failures: usize,
    /// Requests that succeeded but whose response text parsed into zero
    /// rubric dimensions.
    n_parse_failures: usize,
    /// Poems with at least one parsed dimension.
    n_scored: usize,
    mean_novelty: Option<f32>,
    mean_coherence: Option<f32>,
    mean_emotional_resonance: Option<f32>,
    mean_craft: Option<f32>,
    /// Standing caveat, always present regardless of outcome.
    limitation_note: String,
}

const JUDGE_LIMITATION_NOTE: &str = "An LLM judge is an imperfect, unvalidated proxy for human \
aesthetic judgement — no correlation against human raters has been measured for this rubric. \
Treat these scores as a cheap directional signal only, not equivalent to human evaluation. \
Contrast with Phase 2.2 (RAT/DAT), which is grounded against published human norms.";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PoetryEvalReport {
    schema_version: u32,
    checkpoint_path: String,
    genesis_phrase: String,
    n_per_form: usize,
    compliance: Vec<FormComplianceStats>,
    judge: JudgeAggregate,
    poems: Vec<PoemRecord>,
}

// ─── Form-compliance aggregation (Phase 2.5, pure/testable) ────────────────

/// Compute compliance stats for one form from validation-shaped inputs.
///
/// Takes the raw fields rather than a `PoemRecord` so it's directly testable
/// against synthetic data without needing a checkpoint.
fn compliance_stats(
    form_label: &str,
    line_syllable_counts: &[Vec<usize>],
    target_syllable_counts: &[Vec<u8>],
    valids: &[bool],
    errors: &[Vec<ValidationError>],
) -> FormComplianceStats {
    let n = valids.len();
    let compliant_count = valids.iter().filter(|v| **v).count();
    let compliance_rate = if n > 0 {
        compliant_count as f32 / n as f32
    } else {
        0.0
    };

    let mut total_dev = 0.0f32;
    let mut total_lines_with_target = 0usize;
    for (actual_lines, target_lines) in line_syllable_counts.iter().zip(target_syllable_counts) {
        for (actual, target) in actual_lines.iter().zip(target_lines.iter()) {
            total_dev += (*actual as f32 - *target as f32).abs();
            total_lines_with_target += 1;
        }
    }
    let mean_syllable_deviation = if total_lines_with_target > 0 {
        Some(total_dev / total_lines_with_target as f32)
    } else {
        None
    };

    let mut too_few_lines = 0usize;
    let mut too_many_lines = 0usize;
    let mut wrong_syllable_count = 0usize;
    for poem_errors in errors {
        for e in poem_errors {
            match e {
                ValidationError::TooFewLines { .. } => too_few_lines += 1,
                ValidationError::TooManyLines { .. } => too_many_lines += 1,
                ValidationError::WrongSyllableCount { .. } => wrong_syllable_count += 1,
            }
        }
    }

    FormComplianceStats {
        form: form_label.to_string(),
        n,
        compliant_count,
        compliance_rate,
        mean_syllable_deviation,
        error_counts: ErrorTypeCounts {
            too_few_lines,
            too_many_lines,
            wrong_syllable_count,
        },
    }
}

fn compliance_stats_for_records(form_label: &str, records: &[&PoemRecord]) -> FormComplianceStats {
    let line_syllable_counts: Vec<Vec<usize>> = records
        .iter()
        .map(|r| r.line_syllable_counts.clone())
        .collect();
    let target_syllable_counts: Vec<Vec<u8>> = records
        .iter()
        .map(|r| r.target_syllable_counts.clone())
        .collect();
    let valids: Vec<bool> = records.iter().map(|r| r.valid).collect();
    let errors: Vec<Vec<ValidationError>> = records.iter().map(|r| r.errors.clone()).collect();
    compliance_stats(
        form_label,
        &line_syllable_counts,
        &target_syllable_counts,
        &valids,
        &errors,
    )
}

// ─── Judge aggregation (Phase 2.4, pure/testable) ──────────────────────────

fn aggregate_judge_scores(
    poems: &[PoemRecord],
) -> (Option<f32>, Option<f32>, Option<f32>, Option<f32>, usize) {
    let mut n_scored = 0usize;
    let (mut nov_sum, mut nov_n) = (0.0f32, 0usize);
    let (mut coh_sum, mut coh_n) = (0.0f32, 0usize);
    let (mut emo_sum, mut emo_n) = (0.0f32, 0usize);
    let (mut craft_sum, mut craft_n) = (0.0f32, 0usize);

    for poem in poems {
        let Some(scores) = &poem.judge_scores else {
            continue;
        };
        if scores.is_empty() {
            continue;
        }
        n_scored += 1;
        if let Some(s) = &scores.novelty {
            nov_sum += s.score;
            nov_n += 1;
        }
        if let Some(s) = &scores.coherence {
            coh_sum += s.score;
            coh_n += 1;
        }
        if let Some(s) = &scores.emotional_resonance {
            emo_sum += s.score;
            emo_n += 1;
        }
        if let Some(s) = &scores.craft {
            craft_sum += s.score;
            craft_n += 1;
        }
    }

    let mean = |sum: f32, n: usize| if n > 0 { Some(sum / n as f32) } else { None };
    (
        mean(nov_sum, nov_n),
        mean(coh_sum, coh_n),
        mean(emo_sum, emo_n),
        mean(craft_sum, craft_n),
        n_scored,
    )
}

// ─── Judge response parsing (Phase 2.4, fault-tolerant) ────────────────────

/// Parse a raw LLM judge response into `JudgeScores`, never fabricating a
/// value on failure. Tries, in order: (1) the whole response as clean JSON,
/// (2) the first balanced `{...}` substring (handles markdown fences/prose
/// wrapping the JSON), (3) a regex line-scan for "dimension: N" patterns as
/// a last-resort fault-tolerant extractor. Returns `None` if nothing at all
/// could be recovered (including an empty response).
fn parse_judge_response(raw: &str) -> Option<JudgeScores> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(scores) = serde_json::from_str::<JudgeScores>(trimmed) {
        if !scores.is_empty() {
            return Some(scores);
        }
    }

    if let Some(json_slice) = extract_json_object(trimmed) {
        if let Ok(scores) = serde_json::from_str::<JudgeScores>(&json_slice) {
            if !scores.is_empty() {
                return Some(scores);
            }
        }
    }

    let partial = regex_scan_scores(trimmed);
    if !partial.is_empty() {
        return Some(partial);
    }

    None
}

/// Extract the first balanced-brace `{...}` substring from `text`, if any.
fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn regex_scan_scores(text: &str) -> JudgeScores {
    JudgeScores {
        novelty: regex_scan_dimension(text, &["novelty"]),
        coherence: regex_scan_dimension(text, &["coherence"]),
        emotional_resonance: regex_scan_dimension(
            text,
            &["emotional resonance", "emotional_resonance", "emotion"],
        ),
        craft: regex_scan_dimension(text, &["craft"]),
    }
}

/// Look for `<key> ... <1-5>` within ~15 non-digit characters of the
/// keyword, case-insensitive. Fault-tolerant last resort for prose
/// responses that mention a rating without valid JSON.
fn regex_scan_dimension(text: &str, keys: &[&str]) -> Option<RubricScore> {
    for key in keys {
        let pattern = format!(r"(?i){}\D{{0,15}}?([1-5])(?:\.\d+)?", regex::escape(key));
        let re = match regex::Regex::new(&pattern) {
            Ok(re) => re,
            Err(_) => continue,
        };
        if let Some(caps) = re.captures(text) {
            if let Some(m) = caps.get(1) {
                if let Ok(score) = m.as_str().parse::<f32>() {
                    return Some(RubricScore {
                        score,
                        justification: None,
                    });
                }
            }
        }
    }
    None
}

// ─── Ollama judge client (synchronous, mirrors broca-external-baseline) ────

#[derive(Serialize)]
struct OllamaChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct OllamaChatOptions {
    temperature: f32,
    num_predict: usize,
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaChatMessage<'a>>,
    stream: bool,
    options: OllamaChatOptions,
}

#[derive(Deserialize)]
struct OllamaChatResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatResponseMessage,
}

const JUDGE_SYSTEM_PROMPT: &str = "You are a poetry critic evaluating machine-generated poems \
along four dimensions, each scored as an integer 1 (poor) to 5 (excellent): NOVELTY (originality \
and surprise of imagery/word choice), COHERENCE (does it hang together as a unified thought), \
EMOTIONAL_RESONANCE (does it evoke feeling), CRAFT (control of meter, imagery, word choice). \
Respond with ONLY a single JSON object and nothing else — no markdown fences, no preamble — in \
exactly this shape: {\"novelty\": {\"score\": <1-5>, \"justification\": \"<one sentence>\"}, \
\"coherence\": {\"score\": <1-5>, \"justification\": \"<one sentence>\"}, \"emotional_resonance\": \
{\"score\": <1-5>, \"justification\": \"<one sentence>\"}, \"craft\": {\"score\": <1-5>, \
\"justification\": \"<one sentence>\"}}";

fn judge_user_prompt(form_label: &str, poem_text: &str) -> String {
    format!("Poem (form: {form_label}):\n\n{poem_text}\n\nScore it now.")
}

fn ollama_available(base_url: &str, timeout: Duration) -> Result<(), String> {
    let tags_url = format!("{base_url}/api/tags");
    let agent = ureq::AgentBuilder::new().timeout(timeout).build();
    agent
        .get(&tags_url)
        .call()
        .map(|_| ())
        .map_err(|e| format!("Ollama not reachable at {base_url}: {e}"))
}

fn call_ollama_judge(
    base_url: &str,
    model: &str,
    form_label: &str,
    poem_text: &str,
    max_tokens: usize,
    timeout: Duration,
) -> Result<String> {
    let url = format!("{base_url}/api/chat");
    let user_prompt = judge_user_prompt(form_label, poem_text);
    let req = OllamaChatRequest {
        model,
        messages: vec![
            OllamaChatMessage {
                role: "system",
                content: JUDGE_SYSTEM_PROMPT,
            },
            OllamaChatMessage {
                role: "user",
                content: &user_prompt,
            },
        ],
        stream: false,
        options: OllamaChatOptions {
            temperature: 0.2,
            num_predict: max_tokens,
        },
    };
    let body = serde_json::to_string(&req).context("encoding judge request JSON")?;
    let agent = ureq::AgentBuilder::new().timeout(timeout).build();
    let response_body = agent
        .post(&url)
        .set("Content-Type", "application/json")
        .send_string(&body)
        .with_context(|| format!("posting judge request to {url}"))?
        .into_string()
        .context("reading judge response body")?;
    let parsed: OllamaChatResponse =
        serde_json::from_str(&response_body).context("decoding judge chat response JSON")?;
    Ok(parsed.message.content)
}

// ─── Poem generation (mirrors creative_bridge.rs's Poetry arm channel-build) ─

/// The three forms this eval covers, in a fixed order for deterministic reports.
fn forms() -> Vec<(&'static str, CreativeGating)> {
    vec![
        ("haiku", CreativeGating::haiku()),
        ("tanka", CreativeGating::tanka()),
        ("free_verse", CreativeGating::free_verse()),
    ]
}

/// Build a `ThoughtChannels` for poem `poem_idx` of `n` within `form_idx`
/// (0=haiku, 1=tanka, 2=free_verse). Mirrors the core channel mapping
/// `creative_bridge.rs`'s Poetry arm uses (epistemic=certain, then
/// emotion/consciousness from live state) but sweeps valence/arousal/warmth
/// and a form-appropriate consciousness band deterministically across the
/// batch so poems differ instead of repeating the identical input N times.
fn build_channels(form_idx: usize, poem_idx: usize, n: usize) -> ThoughtChannels {
    let mut channels = ThoughtChannels::default();
    // Epistemic ordinal 0 = certain: art speaks with full voice (matches
    // creative_bridge.rs's Poetry arm).
    channels.set_epistemic(0.0);

    let t = if n > 1 {
        poem_idx as f32 / (n - 1) as f32
    } else {
        0.5
    };

    let valence = -0.8 + 1.6 * t;
    let arousal = 0.2 + 0.7 * ((t * 2.0 * std::f32::consts::PI).sin() * 0.5 + 0.5);
    let warmth = 0.3 + 0.6 * (1.0 - t);
    channels.set_emotion(valence, arousal, warmth);

    // Consciousness band mirrors `select_creative_gating`'s thresholds
    // (< 0.5 haiku, < 0.75 tanka, else free verse) for realism, though the
    // form itself is selected explicitly here, not derived from this value.
    let consciousness = match form_idx {
        0 => 0.10 + 0.35 * t,
        1 => 0.50 + 0.20 * t,
        _ => 0.75 + 0.20 * t,
    };
    let meta_depth = 0.2 + 0.6 * t;
    let coherence = 0.4 + 0.5 * (1.0 - t);
    channels.set_consciousness(consciousness, meta_depth, coherence);

    channels
}

fn generate_form_poems(
    generator: &mut BrocaGenerator,
    form_label: &str,
    gating: &CreativeGating,
    form_idx: usize,
    n: usize,
    seed_base: u64,
    max_gen_tokens: Option<usize>,
) -> Vec<PoemRecord> {
    let form = gating
        .form_constraint
        .clone()
        .unwrap_or(PoeticForm::FreeVerse);

    {
        let cfg = generator.config_mut();
        // Art doesn't hedge: disable epistemic gating per the preset's weight.
        cfg.enable_epistemic_gate = gating.epistemic_gate_weight > 0.0;
        if let Some(penalty) = gating.repetition_penalty_override {
            cfg.repetition_penalty = penalty;
        }
        if let Some(tokens) = max_gen_tokens {
            cfg.gating.base_max_tokens = tokens;
        }
    }

    let mut records = Vec::with_capacity(n);
    for i in 0..n {
        {
            let cfg = generator.config_mut();
            cfg.sampling_seed = Some(seed_base + (form_idx as u64 * 1000) + i as u64);
        }
        let channels = build_channels(form_idx, i, n);
        let result = generator.generate(&channels);
        let validation = validate_poem(&result.text, &form);
        records.push(PoemRecord {
            form: form_label.to_string(),
            index: i,
            text: result.text,
            line_syllable_counts: validation.line_syllable_counts,
            target_syllable_counts: validation.target_counts,
            valid: validation.valid,
            errors: validation.errors,
            judge_scores: None,
        });
    }
    records
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn run(opts: Options) -> Result<()> {
    let genesis = GenesisSeed::from_phrase(&opts.genesis_phrase);
    let load_result = if opts.allow_checkpoint_recovery {
        BrocaGenerator::from_checkpoint_allow_checksum_mismatch(&opts.checkpoint_path, &genesis)
    } else {
        BrocaGenerator::from_checkpoint(&opts.checkpoint_path, &genesis)
    };
    let (mut generator, _, _, _) =
        load_result.with_context(|| format!("loading checkpoint {}", opts.checkpoint_path))?;

    let mut all_poems: Vec<PoemRecord> = Vec::new();
    let mut compliance = Vec::new();
    for (form_idx, (form_label, gating)) in forms().into_iter().enumerate() {
        let records = generate_form_poems(
            &mut generator,
            form_label,
            &gating,
            form_idx,
            opts.n_per_form,
            opts.seed,
            opts.max_gen_tokens,
        );
        let refs: Vec<&PoemRecord> = records.iter().collect();
        compliance.push(compliance_stats_for_records(form_label, &refs));
        all_poems.extend(records);
    }

    // Phase 2.4: LLM judge, best-effort. Never let judge failure block the
    // Phase 2.5 compliance report, which is already fully computed above.
    let judge = if opts.skip_judge {
        JudgeAggregate {
            available: false,
            model: opts.judge_model.clone(),
            url: opts.judge_base_url.clone(),
            unavailable_reason: Some("--skip-judge was passed".to_string()),
            n_poems: all_poems.len(),
            n_request_failures: 0,
            n_parse_failures: 0,
            n_scored: 0,
            mean_novelty: None,
            mean_coherence: None,
            mean_emotional_resonance: None,
            mean_craft: None,
            limitation_note: JUDGE_LIMITATION_NOTE.to_string(),
        }
    } else {
        let timeout = Duration::from_secs(opts.judge_timeout_secs);
        match ollama_available(&opts.judge_base_url, timeout) {
            Err(reason) => JudgeAggregate {
                available: false,
                model: opts.judge_model.clone(),
                url: opts.judge_base_url.clone(),
                unavailable_reason: Some(reason),
                n_poems: all_poems.len(),
                n_request_failures: 0,
                n_parse_failures: 0,
                n_scored: 0,
                mean_novelty: None,
                mean_coherence: None,
                mean_emotional_resonance: None,
                mean_craft: None,
                limitation_note: JUDGE_LIMITATION_NOTE.to_string(),
            },
            Ok(()) => {
                let mut n_request_failures = 0usize;
                let mut n_parse_failures = 0usize;
                for poem in all_poems.iter_mut() {
                    match call_ollama_judge(
                        &opts.judge_base_url,
                        &opts.judge_model,
                        &poem.form,
                        &poem.text,
                        opts.judge_max_tokens,
                        timeout,
                    ) {
                        Ok(raw) => {
                            let parsed = parse_judge_response(&raw);
                            if parsed.is_none() {
                                n_parse_failures += 1;
                            }
                            poem.judge_scores = parsed;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Ollama judge request failed");
                            n_request_failures += 1;
                            poem.judge_scores = None;
                        }
                    }
                }
                let (mean_novelty, mean_coherence, mean_emotional_resonance, mean_craft, n_scored) =
                    aggregate_judge_scores(&all_poems);
                JudgeAggregate {
                    available: true,
                    model: opts.judge_model.clone(),
                    url: opts.judge_base_url.clone(),
                    unavailable_reason: None,
                    n_poems: all_poems.len(),
                    n_request_failures,
                    n_parse_failures,
                    n_scored,
                    mean_novelty,
                    mean_coherence,
                    mean_emotional_resonance,
                    mean_craft,
                    limitation_note: JUDGE_LIMITATION_NOTE.to_string(),
                }
            }
        }
    };

    let report = PoetryEvalReport {
        schema_version: 1,
        checkpoint_path: opts.checkpoint_path.clone(),
        genesis_phrase: opts.genesis_phrase.clone(),
        n_per_form: opts.n_per_form,
        compliance,
        judge,
        poems: all_poems,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &opts.json_out {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Error: {e}");
            print_usage();
            process::exit(1);
        }
    };
    if let Err(e) = run(opts) {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Report JSON structure ──

    #[test]
    fn report_round_trips_through_json() {
        let report = PoetryEvalReport {
            schema_version: 1,
            checkpoint_path: "checkpoint.bin".to_string(),
            genesis_phrase: "symthaea luminous dynamics".to_string(),
            n_per_form: 2,
            compliance: vec![compliance_stats(
                "haiku",
                &[vec![5, 7, 5]],
                &[vec![5, 7, 5]],
                &[true],
                &[vec![]],
            )],
            judge: JudgeAggregate {
                available: true,
                model: "gemma4:e2b".to_string(),
                url: "http://127.0.0.1:11434".to_string(),
                unavailable_reason: None,
                n_poems: 2,
                n_request_failures: 0,
                n_parse_failures: 1,
                n_scored: 1,
                mean_novelty: Some(4.0),
                mean_coherence: Some(3.5),
                mean_emotional_resonance: None,
                mean_craft: Some(4.5),
                limitation_note: JUDGE_LIMITATION_NOTE.to_string(),
            },
            poems: vec![
                PoemRecord {
                    form: "haiku".to_string(),
                    index: 0,
                    text: "an old silent pond\na frog jumps into the pond\nsplash silence again"
                        .to_string(),
                    line_syllable_counts: vec![5, 7, 5],
                    target_syllable_counts: vec![5, 7, 5],
                    valid: true,
                    errors: vec![],
                    judge_scores: Some(JudgeScores {
                        novelty: Some(RubricScore {
                            score: 4.0,
                            justification: Some("fresh imagery".to_string()),
                        }),
                        coherence: None,
                        emotional_resonance: None,
                        craft: None,
                    }),
                },
                PoemRecord {
                    form: "haiku".to_string(),
                    index: 1,
                    text: "one line only".to_string(),
                    line_syllable_counts: vec![4],
                    target_syllable_counts: vec![5, 7, 5],
                    valid: false,
                    errors: vec![ValidationError::TooFewLines {
                        expected: 3,
                        got: 1,
                    }],
                    judge_scores: None,
                },
            ],
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        let restored: PoetryEvalReport = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.schema_version, 1);
        assert_eq!(restored.checkpoint_path, "checkpoint.bin");
        assert_eq!(restored.compliance.len(), 1);
        assert_eq!(restored.compliance[0].form, "haiku");
        assert_eq!(restored.poems.len(), 2);
        assert!(restored.poems[0].valid);
        assert!(!restored.poems[1].valid);
        assert_eq!(
            restored.poems[1].errors,
            vec![ValidationError::TooFewLines {
                expected: 3,
                got: 1
            }]
        );
        assert_eq!(
            restored.poems[0]
                .judge_scores
                .as_ref()
                .unwrap()
                .novelty
                .as_ref()
                .unwrap()
                .score,
            4.0
        );
        assert!(restored.poems[1].judge_scores.is_none());
        assert!(restored.judge.available);
        assert_eq!(restored.judge.mean_emotional_resonance, None);
        assert_eq!(restored.judge.limitation_note, JUDGE_LIMITATION_NOTE);
    }

    // ── Compliance aggregation math ──

    #[test]
    fn compliance_stats_all_passing() {
        let stats = compliance_stats(
            "haiku",
            &[vec![5, 7, 5], vec![5, 7, 5]],
            &[vec![5, 7, 5], vec![5, 7, 5]],
            &[true, true],
            &[vec![], vec![]],
        );
        assert_eq!(stats.n, 2);
        assert_eq!(stats.compliant_count, 2);
        assert!((stats.compliance_rate - 1.0).abs() < 1e-6);
        assert_eq!(stats.mean_syllable_deviation, Some(0.0));
        assert_eq!(
            stats.error_counts,
            ErrorTypeCounts {
                too_few_lines: 0,
                too_many_lines: 0,
                wrong_syllable_count: 0,
            }
        );
    }

    #[test]
    fn compliance_stats_mixed_errors_computes_rate_and_deviation() {
        // Poem 0: valid haiku (deviation 0 across 3 lines).
        // Poem 1: wrong syllable counts (deviations 2, 0, 3 => sum 5 over 3 lines).
        // Poem 2: too few lines (only 1 line present, target has 3, but only
        //         the lines that exist are zipped against targets: 1 line,
        //         deviation |4-5|=1).
        let stats = compliance_stats(
            "haiku",
            &[vec![5, 7, 5], vec![7, 7, 8], vec![4]],
            &[vec![5, 7, 5], vec![5, 7, 5], vec![5, 7, 5]],
            &[true, false, false],
            &[
                vec![],
                vec![ValidationError::WrongSyllableCount {
                    line: 0,
                    expected: 5,
                    got: 7,
                }],
                vec![ValidationError::TooFewLines {
                    expected: 3,
                    got: 1,
                }],
            ],
        );
        assert_eq!(stats.n, 3);
        assert_eq!(stats.compliant_count, 1);
        assert!((stats.compliance_rate - (1.0 / 3.0)).abs() < 1e-6);
        // Total deviation: poem0 = 0+0+0=0; poem1 = |7-5|+|7-7|+|8-5| = 2+0+3=5;
        // poem2 = |4-5| = 1. Total = 6 over (3+3+1)=7 lines => 6/7.
        let expected_mean = 6.0 / 7.0;
        assert!(
            (stats.mean_syllable_deviation.unwrap() - expected_mean).abs() < 1e-6,
            "got {:?}",
            stats.mean_syllable_deviation
        );
        assert_eq!(
            stats.error_counts,
            ErrorTypeCounts {
                too_few_lines: 1,
                too_many_lines: 0,
                wrong_syllable_count: 1,
            }
        );
    }

    #[test]
    fn compliance_stats_too_many_lines_counted() {
        let stats = compliance_stats(
            "haiku",
            &[vec![5, 7, 5, 5]],
            &[vec![5, 7, 5]],
            &[false],
            &[vec![ValidationError::TooManyLines {
                expected: 3,
                got: 4,
            }]],
        );
        assert_eq!(stats.error_counts.too_many_lines, 1);
        assert_eq!(stats.compliant_count, 0);
        assert!((stats.compliance_rate - 0.0).abs() < 1e-6);
    }

    #[test]
    fn compliance_stats_free_verse_has_no_deviation_signal() {
        // Free verse poems have empty target_syllable_counts — no lines have
        // a target to compare against, so mean_syllable_deviation is None
        // (not 0.0, which would misleadingly imply perfect adherence).
        let stats = compliance_stats(
            "free_verse",
            &[vec![4, 9, 6], vec![3, 3]],
            &[vec![], vec![]],
            &[true, true],
            &[vec![], vec![]],
        );
        assert_eq!(stats.mean_syllable_deviation, None);
        assert!((stats.compliance_rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compliance_stats_empty_set() {
        let stats = compliance_stats("haiku", &[], &[], &[], &[]);
        assert_eq!(stats.n, 0);
        assert_eq!(stats.compliant_count, 0);
        assert!((stats.compliance_rate - 0.0).abs() < 1e-6);
        assert_eq!(stats.mean_syllable_deviation, None);
    }

    // ── Judge-response parser ──

    #[test]
    fn parse_judge_response_clean_json() {
        let raw = r#"{"novelty": {"score": 4, "justification": "fresh imagery"},
            "coherence": {"score": 3, "justification": "mostly hangs together"},
            "emotional_resonance": {"score": 5, "justification": "quite moving"},
            "craft": {"score": 2, "justification": "meter is uneven"}}"#;
        let parsed = parse_judge_response(raw).expect("should parse clean JSON");
        assert_eq!(parsed.novelty.unwrap().score, 4.0);
        assert_eq!(parsed.coherence.unwrap().score, 3.0);
        assert_eq!(parsed.emotional_resonance.unwrap().score, 5.0);
        assert_eq!(parsed.craft.unwrap().score, 2.0);
    }

    #[test]
    fn parse_judge_response_json_wrapped_in_prose_and_fences() {
        let raw = "Sure, here you go:\n```json\n{\"novelty\": {\"score\": 3, \"justification\": \"ok\"}, \"coherence\": {\"score\": 4, \"justification\": \"solid\"}}\n```\nHope that helps!";
        let parsed = parse_judge_response(raw).expect("should extract embedded JSON");
        assert_eq!(parsed.novelty.unwrap().score, 3.0);
        assert_eq!(parsed.coherence.unwrap().score, 4.0);
        // Dimensions not present in the embedded JSON stay None, not fabricated.
        assert!(parsed.emotional_resonance.is_none());
        assert!(parsed.craft.is_none());
    }

    #[test]
    fn parse_judge_response_regex_fallback_on_prose_with_ratings() {
        // Not valid JSON at all, but mentions ratings inline — the
        // fault-tolerant regex fallback should recover what it can.
        let raw = "I'd rate the Craft: 4 out of 5, quite polished. Coherence 3/5 though.";
        let parsed = parse_judge_response(raw).expect("should recover via regex fallback");
        assert_eq!(parsed.craft.unwrap().score, 4.0);
        assert_eq!(parsed.coherence.unwrap().score, 3.0);
        assert!(parsed.novelty.is_none());
    }

    #[test]
    fn parse_judge_response_pure_prose_returns_none() {
        // No JSON, no dimension keywords with a nearby digit — nothing to
        // recover. Must return None, never a fabricated default.
        let raw = "Interesting use of imagery. Would read again sometime.";
        assert!(parse_judge_response(raw).is_none());
    }

    #[test]
    fn parse_judge_response_empty_returns_none() {
        assert!(parse_judge_response("").is_none());
        assert!(parse_judge_response("   \n  ").is_none());
    }

    #[test]
    fn judge_scores_is_empty() {
        assert!(JudgeScores::default().is_empty());
        let scores = JudgeScores {
            novelty: Some(RubricScore {
                score: 3.0,
                justification: None,
            }),
            ..Default::default()
        };
        assert!(!scores.is_empty());
    }

    // ── Judge aggregation ──

    #[test]
    fn aggregate_judge_scores_computes_per_dimension_means_ignoring_none() {
        let poems = vec![
            PoemRecord {
                form: "haiku".to_string(),
                index: 0,
                text: "x".to_string(),
                line_syllable_counts: vec![],
                target_syllable_counts: vec![],
                valid: true,
                errors: vec![],
                judge_scores: Some(JudgeScores {
                    novelty: Some(RubricScore {
                        score: 4.0,
                        justification: None,
                    }),
                    coherence: Some(RubricScore {
                        score: 2.0,
                        justification: None,
                    }),
                    emotional_resonance: None,
                    craft: None,
                }),
            },
            PoemRecord {
                form: "haiku".to_string(),
                index: 1,
                text: "y".to_string(),
                line_syllable_counts: vec![],
                target_syllable_counts: vec![],
                valid: true,
                errors: vec![],
                judge_scores: Some(JudgeScores {
                    novelty: Some(RubricScore {
                        score: 2.0,
                        justification: None,
                    }),
                    coherence: None,
                    emotional_resonance: None,
                    craft: None,
                }),
            },
            PoemRecord {
                form: "haiku".to_string(),
                index: 2,
                text: "z".to_string(),
                line_syllable_counts: vec![],
                target_syllable_counts: vec![],
                valid: true,
                errors: vec![],
                judge_scores: None, // request failed for this one
            },
        ];
        let (novelty, coherence, emo, craft, n_scored) = aggregate_judge_scores(&poems);
        assert_eq!(novelty, Some(3.0)); // (4+2)/2
        assert_eq!(coherence, Some(2.0)); // only one sample
        assert_eq!(emo, None);
        assert_eq!(craft, None);
        assert_eq!(n_scored, 2);
    }

    #[test]
    fn aggregate_judge_scores_all_none_yields_all_none() {
        let poems = vec![PoemRecord {
            form: "haiku".to_string(),
            index: 0,
            text: "x".to_string(),
            line_syllable_counts: vec![],
            target_syllable_counts: vec![],
            valid: true,
            errors: vec![],
            judge_scores: None,
        }];
        let (novelty, coherence, emo, craft, n_scored) = aggregate_judge_scores(&poems);
        assert_eq!(novelty, None);
        assert_eq!(coherence, None);
        assert_eq!(emo, None);
        assert_eq!(craft, None);
        assert_eq!(n_scored, 0);
    }

    // ── build_channels sanity (deterministic, varies across the batch) ──

    #[test]
    fn build_channels_varies_across_batch() {
        let c0 = build_channels(0, 0, 10);
        let c5 = build_channels(0, 5, 10);
        let c9 = build_channels(0, 9, 10);
        assert_ne!(c0.channels, c5.channels);
        assert_ne!(c5.channels, c9.channels);
    }

    #[test]
    fn build_channels_single_poem_does_not_panic() {
        // n=1 would divide by zero in a naive (i / (n-1)) without the guard.
        let _ = build_channels(1, 0, 1);
    }
}
