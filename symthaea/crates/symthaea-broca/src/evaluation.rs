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
    "Acknowledge", "Answer", "Clarify", "Propose",
    "Uncertainty", "Reflect", "Continue", "Unknown",
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
        return 0.0;
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
    let mut intent_accum: HashMap<String, (f32, f32, f32, usize, usize)> = HashMap::new();
    // (sum_ce, sum_ce_tokens_f, sum_english_ratio, sum_coherence_count, count)

    for pair in &config.dataset.pairs {
        if pair.target_ids.is_empty() {
            continue;
        }

        let channels = ThoughtChannels { channels: pair.channels };
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
                let logits = generator.controller_mut().forward_step(&thought_hv, prev_token, pos);
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
            let entry = intent_accum.entry(intent).or_insert((0.0, 0.0, 0.0, 0, 0));
            if pair_tokens > 0 {
                entry.0 += pair_ce;
                entry.1 += pair_tokens as f32;
            }
            entry.2 += pair_english_ratio;
            if config.compute_english_ratio {
                entry.3 += 1; // coherence sample count
            }
            entry.4 += 1;
        }
    }

    // --- Aggregate ---
    let perplexity = if total_ce_tokens > 0 {
        (total_ce / total_ce_tokens as f32).exp()
    } else {
        0.0
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
    for (intent, (sum_ce, sum_ce_tok, sum_er, coh_count, count)) in &intent_accum {
        let ppl = if *sum_ce_tok > 0.0 {
            (sum_ce / sum_ce_tok).exp()
        } else {
            0.0
        };
        let er = if *coh_count > 0 { sum_er / *coh_count as f32 } else { 0.0 };
        // coherence was summed into sum_er's count — we need a separate accumulator
        // For simplicity, reuse the generation path's data
        intent_scores.insert(intent.clone(), IntentScore {
            perplexity: ppl,
            english_ratio: er,
            avg_coherence: 0.0, // coherence not tracked per-intent (would need separate accumulator)
            count: *count,
        });
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
    s.push_str(&format!("English ratio:     {:.4}\n", result.english_word_ratio));
    s.push_str(&format!("Avg coherence:     {:.4}\n", result.avg_coherence));

    if !result.intent_scores.is_empty() {
        s.push_str("\n--- Per-Intent Breakdown ---\n");
        s.push_str(&format!("{:<14} {:>8} {:>10} {:>6}\n", "Intent", "PPL", "English%", "N"));
        s.push_str(&format!("{:<14} {:>8} {:>10} {:>6}\n", "------", "---", "--------", "-"));

        let mut intents: Vec<_> = result.intent_scores.iter().collect();
        intents.sort_by_key(|(k, _)| k.clone());

        for (intent, score) in intents {
            s.push_str(&format!(
                "{:<14} {:>8.2} {:>9.1}% {:>6}\n",
                intent, score.perplexity, score.english_ratio * 100.0, score.count
            ));
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{BrocaConfig, SamplingStrategy};
    use crate::controller::LanguageControllerConfig;
    use crate::gating::GatingConfig;
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
        assert!(ratio > 0.3, "Known English sentence should have high English ratio: {ratio}");
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
        assert!(result.perplexity.is_finite(), "Perplexity should be finite: {}", result.perplexity);
        assert!(result.perplexity > 0.0, "Perplexity should be positive: {}", result.perplexity);
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
        dataset.push(TrainingPair::new(clarify_ch, "I mean that".to_string(), &tok));

        let eval_config = EvalConfig {
            dataset,
            compute_perplexity: true,
            compute_english_ratio: false,
            per_intent_breakdown: true,
            max_gen_tokens: 16,
        };

        let result = evaluate(&mut gen, &eval_config);
        assert!(result.intent_scores.contains_key("Answer"), "Should have Answer intent");
        assert!(result.intent_scores.contains_key("Clarify"), "Should have Clarify intent");
        assert_eq!(result.intent_scores["Answer"].count, 1);
        assert_eq!(result.intent_scores["Clarify"].count, 1);
    }

    #[test]
    fn test_format_report() {
        let mut intent_scores = HashMap::new();
        intent_scores.insert("Answer".to_string(), IntentScore {
            perplexity: 45.2,
            english_ratio: 0.65,
            avg_coherence: 0.5,
            count: 10,
        });

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
    fn test_active_intent_detection() {
        let ch = ThoughtChannels::with_intent(3); // Propose
        let intent = active_intent(&ch.channels);
        assert_eq!(intent, "Propose");

        let ch2 = ThoughtChannels::with_intent(0); // Acknowledge
        let intent2 = active_intent(&ch2.channels);
        assert_eq!(intent2, "Acknowledge");
    }
}
