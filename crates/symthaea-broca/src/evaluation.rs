//! Evaluation harness: perplexity, English ratio, and per-intent coherence.
//!
//! Measures generation quality after training:
//! - **Perplexity**: exp(avg cross-entropy) via teacher-forced forward pass
//! - **English word ratio**: fraction of generated tokens that are real vocabulary words
//! - **Per-intent breakdown**: quality metrics grouped by semantic intent

use std::collections::HashMap;

use crate::encoder::ThoughtChannels;
use crate::generator::BrocaGenerator;
use crate::training::TrainingDataset;

/// Configuration for evaluation.
pub struct EvalConfig {
    /// Dataset to evaluate on.
    pub dataset: TrainingDataset,
    /// Compute teacher-forced perplexity.
    pub compute_perplexity: bool,
    /// Compute English word ratio on generated text.
    pub compute_english_ratio: bool,
    /// Break down metrics by semantic intent.
    pub per_intent_breakdown: bool,
    /// Maximum tokens to generate per sample (for English ratio / coherence).
    pub max_gen_tokens: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            dataset: TrainingDataset::default(),
            compute_perplexity: true,
            compute_english_ratio: true,
            per_intent_breakdown: true,
            max_gen_tokens: 64,
        }
    }
}

/// Overall evaluation results.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// exp(avg cross-entropy), teacher-forced.
    pub perplexity: f32,
    /// Fraction of tokens that are real vocabulary words (not byte/special).
    pub english_word_ratio: f32,
    /// Average final coherence across generated samples.
    pub avg_coherence: f32,
    /// Per-intent quality breakdown.
    pub intent_scores: HashMap<String, IntentScore>,
    /// Number of samples evaluated.
    pub num_samples: usize,
}

/// Per-intent quality metrics.
#[derive(Debug, Clone)]
pub struct IntentScore {
    pub perplexity: f32,
    pub english_ratio: f32,
    pub avg_coherence: f32,
    pub count: usize,
}

const INTENT_NAMES: [&str; 8] = [
    "Acknowledge",
    "Answer",
    "Clarify",
    "Propose",
    "Uncertainty",
    "Reflect",
    "Continue",
    "Unknown",
];

/// Identify the active intent from channels (argmax of channels 0..8).
fn active_intent(channels: &[f32; 20]) -> &'static str {
    let idx = (0..8)
        .max_by(|&a, &b| channels[a].total_cmp(&channels[b]))
        .unwrap_or(7);
    INTENT_NAMES[idx]
}

/// Cross-entropy loss for a single position (read-only, no weight updates).
fn cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    if target >= logits.len() {
        return f32::INFINITY;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let log_softmax_target = (logits[target] - max_logit) - sum_exp.ln();
    -log_softmax_target
}

/// Compute English word ratio: fraction of token IDs that correspond to
/// multi-character vocabulary words (not single bytes, not special tokens).
pub fn english_word_ratio(token_ids: &[u32], tokenizer: &crate::tokenizer::BpeTokenizer) -> f32 {
    if token_ids.is_empty() {
        return 0.0;
    }

    let mut english_count = 0usize;
    let mut total_count = 0usize;

    for &id in token_ids {
        // Skip special tokens entirely
        if tokenizer.is_special(id) {
            continue;
        }

        total_count += 1;
        let token_str = tokenizer.token_str(id);

        // A "real English token" is a multi-character alphabetic string
        // (not a single byte, not a byte-escape like <0xFF>)
        if token_str.len() >= 2
            && !token_str.starts_with("<0x")
            && token_str.chars().any(|c| c.is_alphabetic())
        {
            english_count += 1;
        }
    }

    if total_count == 0 {
        return 0.0;
    }

    english_count as f32 / total_count as f32
}

/// Run full evaluation: perplexity + generation quality + per-intent breakdown.
pub fn evaluate(generator: &mut BrocaGenerator, config: &EvalConfig) -> EvalResult {
    let mut total_ce = 0.0f32;
    let mut total_ce_tokens = 0usize;
    let mut total_english_ratio = 0.0f32;
    let mut total_coherence = 0.0f32;
    let mut gen_count = 0usize;
    let mut intent_accum: HashMap<String, (f32, f32, f32, f32, usize, usize)> = HashMap::new();
    // (sum_ce, sum_ce_tokens_f, sum_english_ratio, sum_coherence, gen_count, count)

    for pair in &config.dataset.pairs {
        if pair.target_ids.is_empty() {
            continue;
        }

        let channels = ThoughtChannels {
            channels: pair.channels,
        };
        let intent = active_intent(&pair.channels).to_string();

        // --- Perplexity: teacher-forced forward pass ---
        let mut pair_ce = 0.0f32;
        let mut pair_tokens = 0usize;

        if config.compute_perplexity {
            let thought_hv = generator.encoder().encode(&channels);
            generator.controller_mut().reset();
            let mut prev_token = generator.tokenizer().thought_id;
            let window = pair.target_ids.len().min(config.max_gen_tokens);

            for (pos, &target_id) in pair.target_ids[..window].iter().enumerate() {
                let logits = generator
                    .controller_mut()
                    .forward_step(&thought_hv, prev_token, pos);
                let loss = cross_entropy_loss(&logits, target_id as usize);
                pair_ce += loss;
                pair_tokens += 1;
                prev_token = target_id;
            }

            total_ce += pair_ce;
            total_ce_tokens += pair_tokens;
        }

        // --- Generation quality: English ratio + coherence ---
        let mut pair_english_ratio = 0.0f32;
        let mut pair_coherence = 0.0f32;

        if config.compute_english_ratio {
            let result = generator.generate(&channels);
            pair_english_ratio = english_word_ratio(&result.token_ids, generator.tokenizer());
            pair_coherence = result.final_coherence;
            total_english_ratio += pair_english_ratio;
            total_coherence += pair_coherence;
            gen_count += 1;
        }

        // --- Per-intent accumulation ---
        if config.per_intent_breakdown {
            let entry = intent_accum
                .entry(intent)
                .or_insert((0.0, 0.0, 0.0, 0.0, 0, 0));
            if pair_tokens > 0 {
                entry.0 += pair_ce;
                entry.1 += pair_tokens as f32;
            }
            entry.2 += pair_english_ratio;
            entry.3 += pair_coherence;
            if config.compute_english_ratio {
                entry.4 += 1; // generation count for this intent
            }
            entry.5 += 1;
        }
    }

    // --- Aggregate ---
    let perplexity = if total_ce_tokens > 0 {
        (total_ce / total_ce_tokens as f32).exp()
    } else {
        f32::INFINITY
    };

    let english_word_ratio_avg = if gen_count > 0 {
        total_english_ratio / gen_count as f32
    } else {
        0.0
    };

    let avg_coherence = if gen_count > 0 {
        total_coherence / gen_count as f32
    } else {
        0.0
    };

    // Per-intent scores
    let mut intent_scores = HashMap::new();
    for (intent, (sum_ce, sum_ce_tok, sum_er, sum_coh, intent_gen_count, count)) in &intent_accum {
        let ppl = if *sum_ce_tok > 0.0 {
            (sum_ce / sum_ce_tok).exp()
        } else {
            f32::INFINITY
        };
        let er = if *intent_gen_count > 0 {
            sum_er / *intent_gen_count as f32
        } else {
            0.0
        };
        let coh = if *intent_gen_count > 0 {
            sum_coh / *intent_gen_count as f32
        } else {
            0.0
        };
        intent_scores.insert(
            intent.clone(),
            IntentScore {
                perplexity: ppl,
                english_ratio: er,
                avg_coherence: coh,
                count: *count,
            },
        );
    }

    EvalResult {
        perplexity,
        english_word_ratio: english_word_ratio_avg,
        avg_coherence,
        intent_scores,
        num_samples: config.dataset.len(),
    }
}

/// Format an evaluation result as a human-readable report.
pub fn format_eval_report(result: &EvalResult) -> String {
    let mut s = String::new();
    s.push_str("=== Broca Evaluation Report ===\n\n");
    s.push_str(&format!("Samples:           {}\n", result.num_samples));
    s.push_str(&format!("Perplexity:        {:.4}\n", result.perplexity));
    s.push_str(&format!(
        "English ratio:     {:.4}\n",
        result.english_word_ratio
    ));
    s.push_str(&format!("Avg coherence:     {:.4}\n", result.avg_coherence));

    if !result.intent_scores.is_empty() {
        s.push_str("\n--- Per-Intent Breakdown ---\n");
        s.push_str(&format!(
            "{:<14} {:>8} {:>10} {:>6}\n",
            "Intent", "PPL", "English%", "N"
        ));
        s.push_str(&format!(
            "{:<14} {:>8} {:>10} {:>6}\n",
            "------", "---", "--------", "-"
        ));

        let mut intents: Vec<_> = result.intent_scores.iter().collect();
        intents.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (intent, score) in intents {
            s.push_str(&format!(
                "{:<14} {:>8.2} {:>9.1}% {:>6}\n",
                intent,
                score.perplexity,
                score.english_ratio * 100.0,
                score.count
            ));
        }
    }

    s
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIQUID-MAMBA EVALUATION (requires mamba feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for Liquid-Mamba evaluation.
#[cfg(feature = "mamba")]
pub struct LiquidMambaEvalConfig {
    /// Dataset to evaluate on.
    pub dataset: TrainingDataset,
    /// Compute teacher-forced perplexity through frozen Mamba.
    pub compute_perplexity: bool,
    /// Compute English word ratio using Mamba's GPT-2 tokenizer.
    pub compute_english_ratio: bool,
    /// Break down metrics by semantic intent.
    pub per_intent_breakdown: bool,
    /// Maximum tokens to generate per sample.
    pub max_gen_tokens: usize,
    /// Run consciousness gating test (Certain vs Unknown hedging comparison).
    pub consciousness_gating_test: bool,
}

#[cfg(feature = "mamba")]
impl Default for LiquidMambaEvalConfig {
    fn default() -> Self {
        Self {
            dataset: TrainingDataset::default(),
            compute_perplexity: true,
            compute_english_ratio: true,
            per_intent_breakdown: true,
            max_gen_tokens: 64,
            consciousness_gating_test: true,
        }
    }
}

/// Liquid-Mamba evaluation results.
#[cfg(feature = "mamba")]
#[derive(Debug, Clone)]
pub struct LiquidMambaEvalResult {
    /// Base evaluation results (perplexity, English ratio, coherence, intent breakdown).
    pub base: EvalResult,
    /// Mean semantic prediction error (HDC round-trip reconstruction).
    pub avg_semantic_pe: f32,
    /// Mean effective rank of projection bottleneck activations.
    pub avg_effective_rank: f32,
    /// Consciousness gating verification: (certain_hedging%, unknown_hedging%).
    /// If `unknown > certain`, the consciousness bridge is steering generation.
    pub gating_verification: Option<(f32, f32)>,
    /// Distinct-1: fraction of unique unigrams across all generated text.
    pub distinct_1: f32,
    /// Distinct-2: fraction of unique bigrams across all generated text.
    pub distinct_2: f32,
    /// Mean cosine similarity between input thought HV and output centroid HV.
    pub avg_thought_output_similarity: f32,
    /// PE trend (least-squares slope over last 16 PE entries) after all generations.
    pub pe_trend: f32,
    /// Mean PE across all generations in history buffer.
    pub pe_mean: f32,
    /// Standard deviation of PE across all generations in history buffer.
    pub pe_std_dev: f32,
}

/// Compute distinct-n: fraction of unique n-grams out of total n-grams.
///
/// Measures lexical diversity — higher is better (less repetition).
#[cfg(feature = "mamba")]
fn distinct_n(words: &[String], n: usize) -> f32 {
    if words.len() < n || n == 0 {
        return 0.0;
    }
    let mut ngrams = std::collections::HashSet::new();
    let total = words.len() - n + 1;
    for window in words.windows(n) {
        let ngram: String = window.join(" ");
        ngrams.insert(ngram);
    }
    ngrams.len() as f32 / total as f32
}

/// Cosine similarity between two f32 slices.
#[cfg(feature = "mamba")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Hedging tokens that indicate epistemic uncertainty in generated text.
#[cfg(feature = "mamba")]
const HEDGING_WORDS: &[&str] = &[
    "perhaps",
    "maybe",
    "might",
    "possibly",
    "uncertain",
    "unclear",
    "likely",
    "probably",
    "could",
    "seem",
];

/// Count the fraction of tokens that contain hedging words.
#[cfg(feature = "mamba")]
fn hedging_ratio(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }
    let hedging_count = words
        .iter()
        .filter(|w| HEDGING_WORDS.iter().any(|h| w.contains(h)))
        .count();
    hedging_count as f32 / words.len() as f32
}

/// Compute English word ratio using Mamba's GPT-2 tokenizer.
///
/// A token counts as "English" if it decodes to a multi-character alphabetic string
/// (not a single byte, not a byte-escape).
#[cfg(feature = "mamba")]
pub fn english_word_ratio_mamba(
    token_ids: &[u32],
    wrapper: &dyn crate::mamba::MambaBackend,
) -> f32 {
    if token_ids.is_empty() {
        return 0.0;
    }

    let mut english_count = 0usize;
    let mut total_count = 0usize;
    let eos_id = wrapper.eos_token_id();

    for &id in token_ids {
        // Skip EOS/special
        if id == eos_id {
            continue;
        }
        total_count += 1;

        if let Ok(token_str) = wrapper.decode_token(id) {
            let trimmed = token_str.trim();
            if trimmed.len() >= 2 && trimmed.chars().any(|c| c.is_alphabetic()) {
                english_count += 1;
            }
        }
    }

    if total_count == 0 {
        return 0.0;
    }
    english_count as f32 / total_count as f32
}

/// Run full Liquid-Mamba evaluation: perplexity + generation quality + projection health.
#[cfg(feature = "mamba")]
pub fn evaluate_liquid_mamba(
    gen: &mut crate::liquid_mamba::LiquidMambaGenerator,
    config: &LiquidMambaEvalConfig,
) -> LiquidMambaEvalResult {
    let mut total_ce = 0.0f32;
    let mut total_ce_tokens = 0usize;
    let mut total_english_ratio = 0.0f32;
    let mut total_coherence = 0.0f32;
    let mut total_semantic_pe = 0.0f32;
    let mut gen_count = 0usize;
    let mut intent_accum: HashMap<String, (f32, f32, f32, f32, usize, usize)> = HashMap::new();
    let mut all_output_hvs: Vec<symthaea_core::hdc::ContinuousHV> = Vec::new();
    let mut all_generated_words: Vec<String> = Vec::new();
    let mut total_thought_output_sim = 0.0f32;
    let mut sim_count = 0usize;

    for pair in &config.dataset.pairs {
        if pair.target_ids.is_empty() && pair.target_text.is_empty() {
            continue;
        }

        let channels = ThoughtChannels {
            channels: pair.channels,
        };
        let intent = active_intent(&pair.channels).to_string();

        // --- Perplexity: teacher-forced through frozen Mamba ---
        let mut pair_ce = 0.0f32;
        let mut pair_tokens = 0usize;

        if config.compute_perplexity && !pair.target_text.is_empty() {
            // Encode thought to HDC → SSM space, inject into Mamba
            let thought_hv = gen.encoder().encode(&channels);
            let ssm_context = gen.projection().project_to_ssm(&thought_hv);

            gen.mamba_mut().reset();
            if gen.mamba_mut().inject_initial_context(&ssm_context).is_ok() {
                // Tokenize target with Mamba's tokenizer
                if let Ok(target_ids) = gen.mamba().encode(&pair.target_text) {
                    let eos_id = gen.mamba().eos_token_id();
                    let mut prev_token = eos_id;
                    let window = target_ids.len().min(config.max_gen_tokens);

                    for &target_id in &target_ids[..window] {
                        if let Ok(logits) = gen.mamba_mut().forward_one_token(prev_token) {
                            let loss = cross_entropy_loss(&logits, target_id as usize);
                            pair_ce += loss;
                            pair_tokens += 1;
                        }
                        prev_token = target_id;
                    }
                }
            }

            total_ce += pair_ce;
            total_ce_tokens += pair_tokens;
        }

        // --- Generation quality ---
        let mut pair_english_ratio = 0.0f32;
        let mut pair_coherence = 0.0f32;

        if config.compute_english_ratio {
            let thought_hv = gen.encoder().encode(&channels);
            let result = gen.generate(&channels);
            pair_english_ratio = english_word_ratio_mamba(&result.token_ids, gen.mamba());
            pair_coherence = result.final_coherence;
            total_english_ratio += pair_english_ratio;
            total_coherence += pair_coherence;
            total_semantic_pe += result.semantic_pe;
            gen_count += 1;

            // Collect words for distinct-n
            let words: Vec<String> = result
                .text
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();
            all_generated_words.extend(words);

            // Thought-output similarity: centroid of output HVs vs input thought HV
            if !result.output_hvs.is_empty() {
                let dim = thought_hv.dim();
                let mut centroid = vec![0.0f32; dim];
                for hv in &result.output_hvs {
                    for (c, v) in centroid.iter_mut().zip(hv.as_slice().iter()) {
                        *c += v;
                    }
                }
                let n = result.output_hvs.len() as f32;
                for c in centroid.iter_mut() {
                    *c /= n;
                }
                let sim = cosine_similarity(thought_hv.as_slice(), &centroid);
                total_thought_output_sim += sim;
                sim_count += 1;
            }

            // Collect output HVs for effective rank (sample up to 4)
            for hv in result.output_hvs.iter().take(4) {
                all_output_hvs.push(hv.clone());
            }
        }

        // --- Per-intent accumulation ---
        if config.per_intent_breakdown {
            let entry = intent_accum
                .entry(intent)
                .or_insert((0.0, 0.0, 0.0, 0.0, 0, 0));
            if pair_tokens > 0 {
                entry.0 += pair_ce;
                entry.1 += pair_tokens as f32;
            }
            entry.2 += pair_english_ratio;
            entry.3 += pair_coherence;
            if config.compute_english_ratio {
                entry.4 += 1;
            }
            entry.5 += 1;
        }
    }

    // --- Aggregate base metrics ---
    let perplexity = if total_ce_tokens > 0 {
        (total_ce / total_ce_tokens as f32).exp()
    } else {
        f32::INFINITY
    };

    let english_word_ratio_avg = if gen_count > 0 {
        total_english_ratio / gen_count as f32
    } else {
        0.0
    };

    let avg_coherence = if gen_count > 0 {
        total_coherence / gen_count as f32
    } else {
        0.0
    };

    let avg_semantic_pe = if gen_count > 0 {
        total_semantic_pe / gen_count as f32
    } else {
        1.0
    };

    // Effective rank from collected output HVs
    let avg_effective_rank = if all_output_hvs.len() >= 4 {
        gen.projection().effective_rank(&all_output_hvs)
    } else {
        0.0
    };

    // Per-intent scores
    let mut intent_scores = HashMap::new();
    for (intent, (sum_ce, sum_ce_tok, sum_er, sum_coh, intent_gen_count, count)) in &intent_accum {
        let ppl = if *sum_ce_tok > 0.0 {
            (sum_ce / sum_ce_tok).exp()
        } else {
            f32::INFINITY
        };
        let er = if *intent_gen_count > 0 {
            sum_er / *intent_gen_count as f32
        } else {
            0.0
        };
        let coh = if *intent_gen_count > 0 {
            sum_coh / *intent_gen_count as f32
        } else {
            0.0
        };
        intent_scores.insert(
            intent.clone(),
            IntentScore {
                perplexity: ppl,
                english_ratio: er,
                avg_coherence: coh,
                count: *count,
            },
        );
    }

    let base = EvalResult {
        perplexity,
        english_word_ratio: english_word_ratio_avg,
        avg_coherence,
        intent_scores,
        num_samples: config.dataset.len(),
    };

    // --- Consciousness gating test ---
    let gating_verification =
        if config.consciousness_gating_test && !config.dataset.pairs.is_empty() {
            consciousness_gating_test(gen, &config.dataset)
        } else {
            None
        };

    // --- Diversity metrics ---
    let d1 = distinct_n(&all_generated_words, 1);
    let d2 = distinct_n(&all_generated_words, 2);

    let avg_thought_output_similarity = if sim_count > 0 {
        total_thought_output_sim / sim_count as f32
    } else {
        0.0
    };

    // PE history statistics from the generator's ring buffer
    let (pe_mean, pe_std_dev, pe_trend) = gen.pe_stats();

    LiquidMambaEvalResult {
        base,
        avg_semantic_pe,
        avg_effective_rank,
        gating_verification,
        distinct_1: d1,
        distinct_2: d2,
        avg_thought_output_similarity,
        pe_trend,
        pe_mean,
        pe_std_dev,
    }
}

/// Run consciousness gating test: compare hedging tokens under Certain vs Unknown epistemic states.
#[cfg(feature = "mamba")]
fn consciousness_gating_test(
    gen: &mut crate::liquid_mamba::LiquidMambaGenerator,
    dataset: &TrainingDataset,
) -> Option<(f32, f32)> {
    // Sample up to 20 pairs for the gating test (avoid running full dataset twice)
    let sample_size = dataset.pairs.len().min(20);
    let mut certain_hedging_total = 0.0f32;
    let mut unknown_hedging_total = 0.0f32;
    let mut test_count = 0usize;

    for pair in dataset.pairs.iter().take(sample_size) {
        // Generate with Certain epistemic (0.0)
        let mut certain_channels = ThoughtChannels {
            channels: pair.channels,
        };
        certain_channels.set_epistemic(0.0);
        let certain_result = gen.generate(&certain_channels);

        // Generate with Unknown epistemic (3.0)
        let mut unknown_channels = ThoughtChannels {
            channels: pair.channels,
        };
        unknown_channels.set_epistemic(3.0);
        let unknown_result = gen.generate(&unknown_channels);

        certain_hedging_total += hedging_ratio(&certain_result.text);
        unknown_hedging_total += hedging_ratio(&unknown_result.text);
        test_count += 1;
    }

    if test_count == 0 {
        return None;
    }

    let certain_pct = certain_hedging_total / test_count as f32;
    let unknown_pct = unknown_hedging_total / test_count as f32;
    Some((certain_pct, unknown_pct))
}

/// Format a Liquid-Mamba evaluation result as a human-readable report.
#[cfg(feature = "mamba")]
pub fn format_liquid_mamba_eval_report(result: &LiquidMambaEvalResult) -> String {
    let mut s = String::new();
    s.push_str("=== Liquid-Mamba Evaluation Report ===\n\n");
    s.push_str(&format!("Samples:           {}\n", result.base.num_samples));
    s.push_str(&format!(
        "Perplexity:        {:.4}\n",
        result.base.perplexity
    ));
    s.push_str(&format!(
        "English ratio:     {:.4}\n",
        result.base.english_word_ratio
    ));
    s.push_str(&format!(
        "Avg coherence:     {:.4}\n",
        result.base.avg_coherence
    ));
    s.push_str(&format!(
        "Avg semantic PE:   {:.4}\n",
        result.avg_semantic_pe
    ));
    s.push_str(&format!(
        "Effective rank:    {:.2}\n",
        result.avg_effective_rank
    ));
    s.push_str(&format!("Distinct-1:        {:.4}\n", result.distinct_1));
    s.push_str(&format!("Distinct-2:        {:.4}\n", result.distinct_2));
    s.push_str(&format!(
        "Thought-output sim:{:.4}\n",
        result.avg_thought_output_similarity
    ));
    s.push_str(&format!("PE trend:          {:.6}\n", result.pe_trend));
    s.push_str(&format!("PE mean:           {:.4}\n", result.pe_mean));
    s.push_str(&format!("PE std dev:        {:.4}\n", result.pe_std_dev));

    if let Some((certain, unknown)) = result.gating_verification {
        s.push_str(&format!("\n--- Consciousness Gating Test ---\n"));
        s.push_str(&format!("Certain hedging:   {:.2}%\n", certain * 100.0));
        s.push_str(&format!("Unknown hedging:   {:.2}%\n", unknown * 100.0));
        if unknown > certain {
            s.push_str("Result:            PASS (Unknown hedges more than Certain)\n");
        } else {
            s.push_str("Result:            FAIL (consciousness gating not yet effective)\n");
        }
    }

    if !result.base.intent_scores.is_empty() {
        s.push_str("\n--- Per-Intent Breakdown ---\n");
        s.push_str(&format!(
            "{:<14} {:>8} {:>10} {:>6}\n",
            "Intent", "PPL", "English%", "N"
        ));
        s.push_str(&format!(
            "{:<14} {:>8} {:>10} {:>6}\n",
            "------", "---", "--------", "-"
        ));

        let mut intents: Vec<_> = result.base.intent_scores.iter().collect();
        intents.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (intent, score) in intents {
            s.push_str(&format!(
                "{:<14} {:>8.2} {:>9.1}% {:>6}\n",
                intent,
                score.perplexity,
                score.english_ratio * 100.0,
                score.count
            ));
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::LanguageControllerConfig;
    use crate::gating::GatingConfig;
    use crate::generator::{BrocaConfig, SamplingStrategy};
    use crate::tokenizer::BpeTokenizer;
    use crate::training::TrainingPair;
    use symthaea_core::genesis::GenesisSeed;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-broca-eval")
    }

    fn test_config() -> BrocaConfig {
        BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: 32,
                max_seq_len: 16,
                ..Default::default()
            },
            gating: GatingConfig {
                base_max_tokens: 20,
                ..Default::default()
            },
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: false,
            enable_semantic_veto: false,
            ..Default::default()
        }
    }

    fn make_dataset(gen: &BrocaGenerator) -> TrainingDataset {
        let tok = gen.tokenizer().clone();
        let mut dataset = TrainingDataset::default();
        let channels = ThoughtChannels::default();
        for text in &["hello world", "the cat", "is good"] {
            dataset.push(TrainingPair::new(channels, text.to_string(), &tok));
        }
        dataset
    }

    #[test]
    fn test_english_ratio_all_english() {
        let tok = BpeTokenizer::default_minimal();
        // Encode a sentence of known English words
        let ids = tok.encode("the cat is on the mat");
        let ratio = english_word_ratio(&ids, &tok);
        assert!(
            ratio > 0.3,
            "Known English sentence should have high English ratio: {ratio}"
        );
    }

    #[test]
    fn test_english_ratio_empty() {
        let tok = BpeTokenizer::default_minimal();
        let ratio = english_word_ratio(&[], &tok);
        assert!((ratio - 0.0).abs() < 1e-6, "Empty should be 0.0");
    }

    #[test]
    fn test_perplexity_finite() {
        let genesis = test_genesis();
        let config = test_config();
        let mut gen = BrocaGenerator::new(&genesis, config);
        let dataset = make_dataset(&gen);

        let eval_config = EvalConfig {
            dataset,
            compute_perplexity: true,
            compute_english_ratio: false,
            per_intent_breakdown: false,
            max_gen_tokens: 16,
        };

        let result = evaluate(&mut gen, &eval_config);
        assert!(
            result.perplexity.is_finite(),
            "Perplexity should be finite: {}",
            result.perplexity
        );
        assert!(
            result.perplexity > 0.0,
            "Perplexity should be positive: {}",
            result.perplexity
        );
    }

    #[test]
    fn test_evaluate_basic() {
        let genesis = test_genesis();
        let config = test_config();
        let mut gen = BrocaGenerator::new(&genesis, config);
        let dataset = make_dataset(&gen);

        let eval_config = EvalConfig {
            dataset,
            compute_perplexity: true,
            compute_english_ratio: true,
            per_intent_breakdown: false,
            max_gen_tokens: 16,
        };

        let result = evaluate(&mut gen, &eval_config);
        assert_eq!(result.num_samples, 3);
        assert!(result.perplexity.is_finite());
        assert!(result.english_word_ratio >= 0.0);
        assert!(result.english_word_ratio <= 1.0);
    }

    #[test]
    fn test_per_intent_breakdown() {
        let genesis = test_genesis();
        let config = test_config();
        let mut gen = BrocaGenerator::new(&genesis, config);
        let tok = gen.tokenizer().clone();

        let mut dataset = TrainingDataset::default();
        // Two different intents
        let answer_ch = ThoughtChannels::with_intent(1);
        let clarify_ch = ThoughtChannels::with_intent(2);
        dataset.push(TrainingPair::new(answer_ch, "yes it is".to_string(), &tok));
        dataset.push(TrainingPair::new(
            clarify_ch,
            "I mean that".to_string(),
            &tok,
        ));

        let eval_config = EvalConfig {
            dataset,
            compute_perplexity: true,
            compute_english_ratio: false,
            per_intent_breakdown: true,
            max_gen_tokens: 16,
        };

        let result = evaluate(&mut gen, &eval_config);
        assert!(
            result.intent_scores.contains_key("Answer"),
            "Should have Answer intent"
        );
        assert!(
            result.intent_scores.contains_key("Clarify"),
            "Should have Clarify intent"
        );
        assert_eq!(result.intent_scores["Answer"].count, 1);
        assert_eq!(result.intent_scores["Clarify"].count, 1);
    }

    #[test]
    fn test_format_report() {
        let mut intent_scores = HashMap::new();
        intent_scores.insert(
            "Answer".to_string(),
            IntentScore {
                perplexity: 45.2,
                english_ratio: 0.65,
                avg_coherence: 0.5,
                count: 10,
            },
        );

        let result = EvalResult {
            perplexity: 50.0,
            english_word_ratio: 0.7,
            avg_coherence: 0.5,
            intent_scores,
            num_samples: 10,
        };

        let report = format_eval_report(&result);
        assert!(report.contains("Evaluation Report"), "Should have header");
        assert!(report.contains("50.0"), "Should contain perplexity");
        assert!(report.contains("Answer"), "Should contain intent name");
        assert!(report.contains("10"), "Should contain sample count");
    }

    #[test]
    fn test_cross_entropy_oob_returns_inf() {
        let logits = vec![1.0, 2.0, 3.0];
        assert!(
            cross_entropy_loss(&logits, 999).is_infinite(),
            "Out-of-bounds target should return INFINITY"
        );
    }

    #[test]
    fn test_perplexity_no_tokens_is_inf() {
        let genesis = test_genesis();
        let config = test_config();
        let mut gen = BrocaGenerator::new(&genesis, config);

        // Empty dataset → no CE tokens → perplexity should be INFINITY
        let eval_config = EvalConfig {
            dataset: TrainingDataset::default(),
            compute_perplexity: true,
            compute_english_ratio: false,
            per_intent_breakdown: false,
            max_gen_tokens: 16,
        };

        let result = evaluate(&mut gen, &eval_config);
        assert!(
            result.perplexity.is_infinite(),
            "Perplexity with no tokens should be INFINITY, got: {}",
            result.perplexity
        );
    }

    #[test]
    fn test_active_intent_detection() {
        let ch = ThoughtChannels::with_intent(3); // Propose
        let intent = active_intent(&ch.channels);
        assert_eq!(intent, "Propose");

        let ch2 = ThoughtChannels::with_intent(0); // Acknowledge
        let intent2 = active_intent(&ch2.channels);
        assert_eq!(intent2, "Acknowledge");
    }

    #[cfg(feature = "mamba")]
    mod liquid_mamba_tests {
        use super::*;

        #[test]
        fn test_hedging_ratio_with_hedging() {
            let text = "Perhaps the answer is maybe correct, possibly even likely true";
            let ratio = hedging_ratio(text);
            assert!(ratio > 0.2, "Should detect hedging words: {ratio}");
        }

        #[test]
        fn test_hedging_ratio_without_hedging() {
            let text = "The answer is definitely correct and true";
            let ratio = hedging_ratio(text);
            assert!(ratio < 0.1, "Should have low hedging ratio: {ratio}");
        }

        #[test]
        fn test_hedging_ratio_empty() {
            assert!((hedging_ratio("") - 0.0).abs() < 1e-6);
        }

        #[test]
        fn test_format_liquid_mamba_report() {
            let result = LiquidMambaEvalResult {
                base: EvalResult {
                    perplexity: 120.5,
                    english_word_ratio: 0.45,
                    avg_coherence: 0.3,
                    intent_scores: HashMap::new(),
                    num_samples: 5,
                },
                avg_semantic_pe: 0.72,
                avg_effective_rank: 18.5,
                gating_verification: Some((0.05, 0.15)),
                distinct_1: 0.85,
                distinct_2: 0.92,
                avg_thought_output_similarity: 0.35,
                pe_trend: -0.01,
                pe_mean: 0.65,
                pe_std_dev: 0.12,
            };

            let report = format_liquid_mamba_eval_report(&result);
            assert!(report.contains("Liquid-Mamba"), "Should have L-M header");
            assert!(report.contains("120.5"), "Should contain perplexity");
            assert!(report.contains("semantic PE"), "Should contain semantic PE");
            assert!(report.contains("Effective rank"), "Should contain rank");
            assert!(
                report.contains("PASS"),
                "Should pass gating test (0.15 > 0.05)"
            );
        }

        #[test]
        fn test_distinct_n_all_unique() {
            let words: Vec<String> = vec!["the", "cat", "sat", "on", "mat"]
                .into_iter()
                .map(String::from)
                .collect();
            let d1 = distinct_n(&words, 1);
            assert!((d1 - 1.0).abs() < 1e-6, "All unique unigrams: d1={d1}");
            let d2 = distinct_n(&words, 2);
            assert!((d2 - 1.0).abs() < 1e-6, "All unique bigrams: d2={d2}");
        }

        #[test]
        fn test_distinct_n_repetitive() {
            let words: Vec<String> = vec!["the", "the", "the", "the"]
                .into_iter()
                .map(String::from)
                .collect();
            let d1 = distinct_n(&words, 1);
            assert!((d1 - 0.25).abs() < 1e-6, "One unique out of 4: d1={d1}");
            let d2 = distinct_n(&words, 2);
            // All bigrams are "the the" — 1 unique out of 3
            assert!(
                (d2 - 1.0 / 3.0).abs() < 1e-3,
                "One unique bigram of 3: d2={d2}"
            );
        }

        #[test]
        fn test_distinct_n_empty() {
            let words: Vec<String> = vec![];
            assert!((distinct_n(&words, 1) - 0.0).abs() < 1e-6);
            assert!((distinct_n(&words, 2) - 0.0).abs() < 1e-6);
        }

        #[test]
        fn test_cosine_similarity_identical() {
            let a = vec![1.0, 2.0, 3.0];
            let b = vec![1.0, 2.0, 3.0];
            let sim = cosine_similarity(&a, &b);
            assert!((sim - 1.0).abs() < 1e-5, "Identical vectors: sim={sim}");
        }

        #[test]
        fn test_cosine_similarity_orthogonal() {
            let a = vec![1.0, 0.0, 0.0];
            let b = vec![0.0, 1.0, 0.0];
            let sim = cosine_similarity(&a, &b);
            assert!(sim.abs() < 1e-5, "Orthogonal vectors: sim={sim}");
        }

        #[test]
        fn test_format_liquid_mamba_report_gating_fail() {
            let result = LiquidMambaEvalResult {
                base: EvalResult {
                    perplexity: 200.0,
                    english_word_ratio: 0.1,
                    avg_coherence: 0.1,
                    intent_scores: HashMap::new(),
                    num_samples: 1,
                },
                avg_semantic_pe: 0.9,
                avg_effective_rank: 2.0,
                gating_verification: Some((0.10, 0.05)),
                distinct_1: 0.3,
                distinct_2: 0.4,
                avg_thought_output_similarity: 0.1,
                pe_trend: 0.0,
                pe_mean: 0.9,
                pe_std_dev: 0.0,
            };

            let report = format_liquid_mamba_eval_report(&result);
            assert!(
                report.contains("FAIL"),
                "Should fail gating test (0.05 < 0.10)"
            );
        }

        #[test]
        #[ignore = "Requires network access to download mamba-130m"]
        fn test_liquid_mamba_perplexity_finite() {
            use crate::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
            use crate::training::TrainingPair;
            use symthaea_core::genesis::GenesisSeed;

            let genesis = GenesisSeed::from_phrase("test-lm-eval");
            let config = LiquidMambaConfig {
                max_tokens: 16,
                ..Default::default()
            };
            let mut gen = LiquidMambaGenerator::new(&genesis, config)
                .expect("Failed to create LiquidMambaGenerator");

            let mut dataset = TrainingDataset::default();
            let channels = ThoughtChannels::with_intent(1);
            dataset.pairs.push(TrainingPair {
                channels: channels.channels,
                target_text: "hello world".to_string(),
                target_ids: vec![],
            });

            let eval_config = LiquidMambaEvalConfig {
                dataset,
                compute_perplexity: true,
                compute_english_ratio: false,
                per_intent_breakdown: false,
                max_gen_tokens: 16,
                consciousness_gating_test: false,
            };

            let result = evaluate_liquid_mamba(&mut gen, &eval_config);
            assert!(
                result.base.perplexity.is_finite(),
                "Perplexity should be finite: {}",
                result.base.perplexity
            );
            assert!(
                result.base.perplexity > 0.0,
                "Perplexity should be positive: {}",
                result.base.perplexity
            );
        }

        #[test]
        #[ignore = "Requires network access to download mamba-130m"]
        fn test_liquid_mamba_gating_verification() {
            use crate::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
            use crate::training::TrainingPair;
            use symthaea_core::genesis::GenesisSeed;

            let genesis = GenesisSeed::from_phrase("test-lm-gating");
            let config = LiquidMambaConfig {
                max_tokens: 32,
                ..Default::default()
            };
            let mut gen = LiquidMambaGenerator::new(&genesis, config)
                .expect("Failed to create LiquidMambaGenerator");

            let mut dataset = TrainingDataset::default();
            for intent in 0..4 {
                let channels = ThoughtChannels::with_intent(intent);
                dataset.pairs.push(TrainingPair {
                    channels: channels.channels,
                    target_text: "the answer is clear".to_string(),
                    target_ids: vec![],
                });
            }

            let eval_config = LiquidMambaEvalConfig {
                dataset,
                compute_perplexity: false,
                compute_english_ratio: false,
                per_intent_breakdown: false,
                max_gen_tokens: 32,
                consciousness_gating_test: true,
            };

            let result = evaluate_liquid_mamba(&mut gen, &eval_config);
            assert!(
                result.gating_verification.is_some(),
                "Gating verification should produce results"
            );
            let (certain, unknown) = result.gating_verification.unwrap();
            assert!(certain.is_finite(), "Certain hedging should be finite");
            assert!(unknown.is_finite(), "Unknown hedging should be finite");
            // With a trained projection, unknown_hedging > certain_hedging.
            // With random projection, this may not hold — just verify they're computed.
        }

        #[test]
        fn test_evaluate_liquid_mamba_mock() {
            use crate::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
            use crate::training::TrainingPair;
            use symthaea_core::genesis::GenesisSeed;

            let genesis = GenesisSeed::from_phrase("test-lm-eval-mock");
            let config = LiquidMambaConfig {
                max_tokens: 16,
                ..Default::default()
            };
            let mut gen = LiquidMambaGenerator::with_mock(&genesis, config);

            let mut dataset = TrainingDataset::default();
            for intent in 0..3 {
                let ch = ThoughtChannels::with_intent(intent);
                dataset.pairs.push(TrainingPair {
                    channels: ch.channels,
                    target_text: "hello world test".to_string(),
                    target_ids: vec![],
                });
            }

            let eval_config = LiquidMambaEvalConfig {
                dataset,
                compute_perplexity: true,
                compute_english_ratio: true,
                per_intent_breakdown: true,
                max_gen_tokens: 16,
                consciousness_gating_test: true,
            };

            let result = evaluate_liquid_mamba(&mut gen, &eval_config);
            assert_eq!(result.base.num_samples, 3);
            assert!(result.base.perplexity.is_finite() || result.base.perplexity.is_infinite());
            assert!(result.avg_semantic_pe.is_finite());
            assert!(result.distinct_1.is_finite());
            assert!(result.distinct_2.is_finite());
            assert!(result.pe_mean.is_finite());
            assert!(result.gating_verification.is_some());
        }
    }
}
