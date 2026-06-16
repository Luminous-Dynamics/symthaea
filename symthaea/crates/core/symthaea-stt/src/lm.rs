// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Language Model and Beam Search Decoder
//!
//! Provides N-gram language models and beam search decoding for improved
//! transcription accuracy. Integrates acoustic scores from HDC/LTC with
//! language model probabilities.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     BEAM SEARCH DECODER                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Phoneme     ──►  Beam     ──►  LM Scoring  ──►  Best Path      │
//! │  Posteriors      Search        (N-gram)          Selection      │
//! │                                                                  │
//! │  score = α * acoustic + (1-α) * lm + β * word_bonus            │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Default beam width for search
pub const DEFAULT_BEAM_WIDTH: usize = 10;

/// Default language model weight (α)
pub const DEFAULT_LM_WEIGHT: f32 = 0.3;

/// N-gram language model
///
/// Supports unigrams, bigrams, and trigrams with backoff smoothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramLM {
    /// Order of the model (1, 2, or 3)
    pub order: usize,
    /// Unigram log probabilities: P(w)
    unigrams: HashMap<String, f32>,
    /// Bigram log probabilities: P(w | w-1)
    bigrams: HashMap<(String, String), f32>,
    /// Trigram log probabilities: P(w | w-2, w-1)
    trigrams: HashMap<(String, String, String), f32>,
    /// Backoff weights for bigrams
    bigram_backoff: HashMap<String, f32>,
    /// Backoff weights for trigrams
    trigram_backoff: HashMap<(String, String), f32>,
    /// Unknown word log probability
    pub unk_prob: f32,
    /// Total vocabulary size
    pub vocab_size: usize,
}

impl Default for NgramLM {
    fn default() -> Self {
        Self::new(2) // Default to bigram
    }
}

impl NgramLM {
    /// Create a new N-gram language model
    pub fn new(order: usize) -> Self {
        assert!((1..=3).contains(&order), "Order must be 1, 2, or 3");

        Self {
            order,
            unigrams: HashMap::new(),
            bigrams: HashMap::new(),
            trigrams: HashMap::new(),
            bigram_backoff: HashMap::new(),
            trigram_backoff: HashMap::new(),
            unk_prob: -10.0, // Very low probability for unknown words
            vocab_size: 0,
        }
    }

    /// Train language model from text corpus
    pub fn train(&mut self, corpus: &[Vec<String>]) {
        // Count n-grams
        let mut unigram_counts: HashMap<String, usize> = HashMap::new();
        let mut bigram_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut trigram_counts: HashMap<(String, String, String), usize> = HashMap::new();

        let mut total_tokens = 0usize;

        for sentence in corpus {
            // Add start/end tokens
            let mut tokens = vec!["<s>".to_string()];
            tokens.extend(sentence.iter().cloned());
            tokens.push("</s>".to_string());

            for (i, token) in tokens.iter().enumerate() {
                // Unigrams
                *unigram_counts.entry(token.clone()).or_insert(0) += 1;
                total_tokens += 1;

                // Bigrams
                if i > 0 && self.order >= 2 {
                    let prev = tokens[i - 1].clone();
                    *bigram_counts.entry((prev, token.clone())).or_insert(0) += 1;
                }

                // Trigrams
                if i > 1 && self.order >= 3 {
                    let prev2 = tokens[i - 2].clone();
                    let prev1 = tokens[i - 1].clone();
                    *trigram_counts
                        .entry((prev2, prev1, token.clone()))
                        .or_insert(0) += 1;
                }
            }
        }

        self.vocab_size = unigram_counts.len();

        // Convert counts to log probabilities with add-k smoothing
        let k = 0.01; // Smoothing constant
        let vocab_f = self.vocab_size as f32;
        let total_f = total_tokens as f32;

        // Unigrams: P(w) = (count(w) + k) / (N + k*V)
        for (word, count) in &unigram_counts {
            let prob = ((*count as f32 + k) / (total_f + k * vocab_f)).ln();
            self.unigrams.insert(word.clone(), prob);
        }

        // Bigrams: P(w2 | w1) = (count(w1, w2) + k) / (count(w1) + k*V)
        if self.order >= 2 {
            for ((w1, w2), count) in &bigram_counts {
                let w1_count = *unigram_counts.get(w1).unwrap_or(&1) as f32;
                let prob = ((*count as f32 + k) / (w1_count + k * vocab_f)).ln();
                self.bigrams.insert((w1.clone(), w2.clone()), prob);
            }

            // Backoff weights (simplified: just use 0.4)
            for word in unigram_counts.keys() {
                self.bigram_backoff.insert(word.clone(), 0.4_f32.ln());
            }
        }

        // Trigrams: P(w3 | w1, w2) = (count(w1, w2, w3) + k) / (count(w1, w2) + k*V)
        if self.order >= 3 {
            for ((w1, w2, w3), count) in &trigram_counts {
                let bigram_count =
                    *bigram_counts.get(&(w1.clone(), w2.clone())).unwrap_or(&1) as f32;
                let prob = ((*count as f32 + k) / (bigram_count + k * vocab_f)).ln();
                self.trigrams
                    .insert((w1.clone(), w2.clone(), w3.clone()), prob);
            }

            // Backoff weights
            for (w1, w2) in bigram_counts.keys() {
                self.trigram_backoff
                    .insert((w1.clone(), w2.clone()), 0.4_f32.ln());
            }
        }
    }

    /// Score a word given history (returns log probability)
    pub fn score(&self, word: &str, history: &[String]) -> f32 {
        match self.order {
            1 => self.unigram_score(word),
            2 => self.bigram_score(word, history),
            3 => self.trigram_score(word, history),
            _ => self.unk_prob,
        }
    }

    fn unigram_score(&self, word: &str) -> f32 {
        *self.unigrams.get(word).unwrap_or(&self.unk_prob)
    }

    fn bigram_score(&self, word: &str, history: &[String]) -> f32 {
        if history.is_empty() {
            return self.unigram_score(word);
        }

        let prev = history.last().unwrap();

        if let Some(&prob) = self.bigrams.get(&(prev.clone(), word.to_string())) {
            prob
        } else {
            // Backoff to unigram
            let backoff = *self.bigram_backoff.get(prev).unwrap_or(&0.0);
            backoff + self.unigram_score(word)
        }
    }

    fn trigram_score(&self, word: &str, history: &[String]) -> f32 {
        if history.len() < 2 {
            return self.bigram_score(word, history);
        }

        let prev2 = &history[history.len() - 2];
        let prev1 = &history[history.len() - 1];

        if let Some(&prob) = self
            .trigrams
            .get(&(prev2.clone(), prev1.clone(), word.to_string()))
        {
            prob
        } else {
            // Backoff to bigram
            let backoff = *self
                .trigram_backoff
                .get(&(prev2.clone(), prev1.clone()))
                .unwrap_or(&0.0);
            backoff + self.bigram_score(word, history)
        }
    }

    /// Score an entire sentence
    pub fn score_sentence(&self, words: &[String]) -> f32 {
        let mut score = 0.0;
        let mut history = vec!["<s>".to_string()];

        for word in words {
            score += self.score(word, &history);
            history.push(word.clone());

            // Keep history bounded
            if history.len() > self.order {
                history.remove(0);
            }
        }

        // Add end-of-sentence score
        score += self.score("</s>", &history);

        score
    }

    /// Calculate perplexity on test corpus
    pub fn perplexity(&self, corpus: &[Vec<String>]) -> f32 {
        let mut total_log_prob = 0.0;
        let mut total_words = 0;

        for sentence in corpus {
            total_log_prob += self.score_sentence(sentence);
            total_words += sentence.len() + 1; // +1 for </s>
        }

        // PP = exp(-1/N * sum(log P(w_i)))
        (-total_log_prob / total_words as f32).exp()
    }

    /// Save model to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| std::io::Error::other(e.to_string()))?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load model from file
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Load from ARPA format (standard LM format)
    pub fn load_arpa<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut lm = Self::new(3);
        let mut section = 0; // 0=header, 1=unigrams, 2=bigrams, 3=trigrams

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with("\\1-grams:") {
                section = 1;
                continue;
            } else if line.starts_with("\\2-grams:") {
                section = 2;
                continue;
            } else if line.starts_with("\\3-grams:") {
                section = 3;
                continue;
            } else if line.starts_with("\\end\\") {
                break;
            } else if line.starts_with('\\') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match section {
                1 if parts.len() >= 2 => {
                    // Unigram: prob word [backoff]
                    if let Ok(prob) = parts[0].parse::<f32>() {
                        let word = parts[1].to_string();
                        let log_prob = prob * 10.0_f32.ln(); // Convert from log10
                        lm.unigrams.insert(word.clone(), log_prob);

                        if parts.len() >= 3 {
                            if let Ok(backoff) = parts[2].parse::<f32>() {
                                lm.bigram_backoff.insert(word, backoff * 10.0_f32.ln());
                            }
                        }
                    }
                }
                2 if parts.len() >= 3 => {
                    // Bigram: prob w1 w2 [backoff]
                    if let Ok(prob) = parts[0].parse::<f32>() {
                        let w1 = parts[1].to_string();
                        let w2 = parts[2].to_string();
                        let log_prob = prob * 10.0_f32.ln();
                        lm.bigrams.insert((w1.clone(), w2), log_prob);

                        if parts.len() >= 4 {
                            if let Ok(backoff) = parts[3].parse::<f32>() {
                                lm.trigram_backoff
                                    .insert((w1, parts[2].to_string()), backoff * 10.0_f32.ln());
                            }
                        }
                    }
                }
                3 if parts.len() >= 4 => {
                    // Trigram: prob w1 w2 w3
                    if let Ok(prob) = parts[0].parse::<f32>() {
                        let log_prob = prob * 10.0_f32.ln();
                        lm.trigrams.insert(
                            (
                                parts[1].to_string(),
                                parts[2].to_string(),
                                parts[3].to_string(),
                            ),
                            log_prob,
                        );
                    }
                }
                _ => {}
            }
        }

        lm.vocab_size = lm.unigrams.len();
        lm.order = if !lm.trigrams.is_empty() {
            3
        } else if !lm.bigrams.is_empty() {
            2
        } else {
            1
        };

        Ok(lm)
    }
}

/// Beam search hypothesis
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Sequence of tokens (phonemes or words)
    pub tokens: Vec<String>,
    /// Acoustic score (log probability)
    pub acoustic_score: f32,
    /// Language model score (log probability)
    pub lm_score: f32,
    /// Combined score
    pub score: f32,
}

impl Hypothesis {
    /// Create empty hypothesis
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            acoustic_score: 0.0,
            lm_score: 0.0,
            score: 0.0,
        }
    }

    /// Extend hypothesis with a new token
    pub fn extend(&self, token: String, acoustic: f32, lm: f32, lm_weight: f32) -> Self {
        let mut new_tokens = self.tokens.clone();
        new_tokens.push(token);

        let new_acoustic = self.acoustic_score + acoustic;
        let new_lm = self.lm_score + lm;
        let new_score = (1.0 - lm_weight) * new_acoustic + lm_weight * new_lm;

        Self {
            tokens: new_tokens,
            acoustic_score: new_acoustic,
            lm_score: new_lm,
            score: new_score,
        }
    }
}

impl Default for Hypothesis {
    fn default() -> Self {
        Self::new()
    }
}

/// Beam search decoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamConfig {
    /// Number of hypotheses to keep
    pub beam_width: usize,
    /// Language model weight (0.0 to 1.0)
    pub lm_weight: f32,
    /// Word insertion bonus (encourages longer sequences)
    pub word_bonus: f32,
    /// Minimum token probability threshold
    pub min_prob: f32,
    /// Enable length normalization
    pub length_norm: bool,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: DEFAULT_BEAM_WIDTH,
            lm_weight: DEFAULT_LM_WEIGHT,
            word_bonus: 0.0,
            min_prob: -20.0, // Log probability threshold
            length_norm: true,
        }
    }
}

/// Beam search decoder
///
/// Decodes acoustic scores into token sequences using beam search
/// with optional language model rescoring.
pub struct BeamDecoder {
    /// Configuration
    pub config: BeamConfig,
    /// Language model (optional)
    lm: Option<NgramLM>,
    /// Current beam
    beam: Vec<Hypothesis>,
    /// Vocabulary (token to index)
    vocab: Vec<String>,
}

impl BeamDecoder {
    /// Create new decoder
    pub fn new(config: BeamConfig) -> Self {
        Self {
            config,
            lm: None,
            beam: vec![Hypothesis::new()],
            vocab: Vec::new(),
        }
    }

    /// Create decoder with default configuration
    pub fn default_config() -> Self {
        Self::new(BeamConfig::default())
    }

    /// Set the language model
    pub fn with_lm(mut self, lm: NgramLM) -> Self {
        self.lm = Some(lm);
        self
    }

    /// Set the vocabulary
    pub fn with_vocab(mut self, vocab: Vec<String>) -> Self {
        self.vocab = vocab;
        self
    }

    /// Reset decoder state
    pub fn reset(&mut self) {
        self.beam = vec![Hypothesis::new()];
    }

    /// Process one frame of acoustic scores
    ///
    /// # Arguments
    /// * `scores` - Log probabilities for each token in vocabulary
    pub fn step(&mut self, scores: &[f32]) {
        if scores.len() != self.vocab.len() && !self.vocab.is_empty() {
            eprintln!(
                "Warning: score length {} != vocab size {}",
                scores.len(),
                self.vocab.len()
            );
        }

        let mut new_beam: Vec<Hypothesis> = Vec::new();

        // For each hypothesis in the beam
        for hyp in &self.beam {
            // Try extending with each token
            for (i, &acoustic_score) in scores.iter().enumerate() {
                if acoustic_score < self.config.min_prob {
                    continue;
                }

                let token = if i < self.vocab.len() {
                    self.vocab[i].clone()
                } else {
                    format!("T{i}")
                };

                // Get LM score
                let lm_score = if let Some(ref lm) = self.lm {
                    lm.score(&token, &hyp.tokens)
                } else {
                    0.0
                };

                // Apply word bonus
                let bonus = if !token.is_empty() && !token.starts_with('<') {
                    self.config.word_bonus
                } else {
                    0.0
                };

                let new_hyp = hyp.extend(
                    token,
                    acoustic_score + bonus,
                    lm_score,
                    self.config.lm_weight,
                );

                new_beam.push(new_hyp);
            }
        }

        // Prune to beam width
        new_beam.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        new_beam.truncate(self.config.beam_width);

        self.beam = new_beam;
    }

    /// Process a batch of acoustic scores (entire sequence)
    pub fn decode(&mut self, all_scores: &[Vec<f32>]) -> Vec<Hypothesis> {
        self.reset();

        for scores in all_scores {
            self.step(scores);
        }

        self.finalize()
    }

    /// Finalize decoding and return best hypotheses
    pub fn finalize(&mut self) -> Vec<Hypothesis> {
        // Apply length normalization if enabled
        if self.config.length_norm {
            for hyp in &mut self.beam {
                let len = hyp.tokens.len().max(1) as f32;
                hyp.score /= len;
            }

            // Re-sort after normalization
            self.beam.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        self.beam.clone()
    }

    /// Get the best hypothesis
    pub fn best(&self) -> Option<&Hypothesis> {
        self.beam.first()
    }

    /// Get N-best hypotheses
    pub fn nbest(&self, n: usize) -> Vec<&Hypothesis> {
        self.beam.iter().take(n).collect()
    }
}

/// Phoneme-to-word decoder
///
/// Uses a pronunciation dictionary to convert phoneme sequences to words.
pub struct PhonemeToWordDecoder {
    /// Pronunciation dictionary: word -> phoneme sequence
    dict: HashMap<String, Vec<String>>,
    /// Reverse dictionary: phoneme sequence -> words
    reverse_dict: HashMap<Vec<String>, Vec<String>>,
}

impl PhonemeToWordDecoder {
    /// Create new phoneme decoder
    pub fn new() -> Self {
        Self {
            dict: HashMap::new(),
            reverse_dict: HashMap::new(),
        }
    }

    /// Load pronunciation dictionary (CMU format)
    pub fn load_dict<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with(";;;") {
                continue;
            }

            let parts: Vec<&str> = line.splitn(2, "  ").collect();
            if parts.len() != 2 {
                continue;
            }

            let word = parts[0].to_lowercase();
            // Remove variant suffix (e.g., WORD(1) -> word)
            let word = word.split('(').next().unwrap_or(&word).to_string();

            let phonemes: Vec<String> = parts[1]
                .split_whitespace()
                .map(|p| {
                    // Remove stress markers
                    p.chars().filter(|c| !c.is_ascii_digit()).collect()
                })
                .collect();

            self.dict.insert(word.clone(), phonemes.clone());
            self.reverse_dict.entry(phonemes).or_default().push(word);
        }

        Ok(())
    }

    /// Decode phoneme sequence to words using greedy matching
    pub fn decode(&self, phonemes: &[String]) -> Vec<String> {
        let mut words = Vec::new();
        let mut i = 0;

        while i < phonemes.len() {
            let mut best_match: Option<(String, usize)> = None;

            // Try longest match first
            for len in (1..=(phonemes.len() - i).min(15)).rev() {
                let seq: Vec<String> = phonemes[i..i + len].to_vec();

                if let Some(word_list) = self.reverse_dict.get(&seq) {
                    if let Some(word) = word_list.first() {
                        best_match = Some((word.clone(), len));
                        break;
                    }
                }
            }

            if let Some((word, len)) = best_match {
                words.push(word);
                i += len;
            } else {
                // Unknown phoneme, emit as-is
                words.push(format!("<{}>", phonemes[i]));
                i += 1;
            }
        }

        words
    }

    /// Get phonemes for a word
    pub fn get_phonemes(&self, word: &str) -> Option<&Vec<String>> {
        self.dict.get(&word.to_lowercase())
    }

    /// Check if word is in dictionary
    pub fn contains(&self, word: &str) -> bool {
        self.dict.contains_key(&word.to_lowercase())
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.dict.len()
    }
}

impl Default for PhonemeToWordDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined decoder: acoustic -> phonemes -> words
pub struct CombinedDecoder {
    /// Beam decoder for phoneme sequences
    beam_decoder: BeamDecoder,
    /// Phoneme to word decoder
    phoneme_decoder: PhonemeToWordDecoder,
    /// Word-level language model
    word_lm: Option<NgramLM>,
}

impl CombinedDecoder {
    /// Create new combined decoder
    pub fn new(beam_config: BeamConfig) -> Self {
        Self {
            beam_decoder: BeamDecoder::new(beam_config),
            phoneme_decoder: PhonemeToWordDecoder::new(),
            word_lm: None,
        }
    }

    /// Set phoneme language model
    pub fn with_phoneme_lm(mut self, lm: NgramLM) -> Self {
        self.beam_decoder = self.beam_decoder.with_lm(lm);
        self
    }

    /// Set word language model
    pub fn with_word_lm(mut self, lm: NgramLM) -> Self {
        self.word_lm = Some(lm);
        self
    }

    /// Load pronunciation dictionary
    pub fn load_dict<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        self.phoneme_decoder.load_dict(path)
    }

    /// Set phoneme vocabulary
    pub fn with_vocab(mut self, vocab: Vec<String>) -> Self {
        self.beam_decoder = self.beam_decoder.with_vocab(vocab);
        self
    }

    /// Decode acoustic scores to text
    pub fn decode(&mut self, scores: &[Vec<f32>]) -> String {
        // First pass: acoustic to phonemes
        let hypotheses = self.beam_decoder.decode(scores);

        if hypotheses.is_empty() {
            return String::new();
        }

        // Convert phonemes to words for each hypothesis
        let mut word_hypotheses: Vec<(Vec<String>, f32)> = Vec::new();

        for hyp in &hypotheses {
            let words = self.phoneme_decoder.decode(&hyp.tokens);

            // Rescore with word LM if available
            let word_score = if let Some(ref lm) = self.word_lm {
                lm.score_sentence(&words)
            } else {
                0.0
            };

            // Combined score
            let total_score = hyp.score + 0.3 * word_score;
            word_hypotheses.push((words, total_score));
        }

        // Sort by combined score
        word_hypotheses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return best hypothesis
        word_hypotheses
            .first()
            .map(|(words, _)| words.join(" "))
            .unwrap_or_default()
    }

    /// Decode and return N-best results
    pub fn decode_nbest(&mut self, scores: &[Vec<f32>], n: usize) -> Vec<(String, f32)> {
        let hypotheses = self.beam_decoder.decode(scores);

        let mut word_hypotheses: Vec<(String, f32)> = Vec::new();

        for hyp in hypotheses.iter().take(n * 2) {
            let words = self.phoneme_decoder.decode(&hyp.tokens);

            let word_score = if let Some(ref lm) = self.word_lm {
                lm.score_sentence(&words)
            } else {
                0.0
            };

            let total_score = hyp.score + 0.3 * word_score;
            word_hypotheses.push((words.join(" "), total_score));
        }

        word_hypotheses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        word_hypotheses.truncate(n);

        word_hypotheses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_lm_training() {
        let corpus = vec![
            vec!["the".to_string(), "cat".to_string(), "sat".to_string()],
            vec!["the".to_string(), "dog".to_string(), "ran".to_string()],
            vec!["the".to_string(), "cat".to_string(), "ran".to_string()],
        ];

        let mut lm = NgramLM::new(2);
        lm.train(&corpus);

        assert!(lm.vocab_size > 0);
        assert!(lm.unigrams.contains_key("cat"));
        assert!(
            lm.bigrams
                .contains_key(&("the".to_string(), "cat".to_string()))
        );
    }

    #[test]
    fn test_ngram_scoring() {
        let corpus = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["hello".to_string(), "there".to_string()],
        ];

        let mut lm = NgramLM::new(2);
        lm.train(&corpus);

        // "hello" should have higher probability than unknown word
        let hello_score = lm.score("hello", &[]);
        let unknown_score = lm.score("xyz", &[]);

        assert!(hello_score > unknown_score);
    }

    #[test]
    fn test_sentence_scoring() {
        let corpus = vec![
            vec!["the".to_string(), "quick".to_string(), "fox".to_string()],
            vec!["the".to_string(), "lazy".to_string(), "dog".to_string()],
        ];

        let mut lm = NgramLM::new(2);
        lm.train(&corpus);

        let known = vec!["the".to_string(), "quick".to_string(), "fox".to_string()];
        let unknown = vec!["xyz".to_string(), "abc".to_string(), "def".to_string()];

        let known_score = lm.score_sentence(&known);
        let unknown_score = lm.score_sentence(&unknown);

        assert!(known_score > unknown_score);
    }

    #[test]
    fn test_beam_decoder() {
        let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let config = BeamConfig {
            beam_width: 3,
            lm_weight: 0.0,
            ..Default::default()
        };

        let mut decoder = BeamDecoder::new(config).with_vocab(vocab);

        // Simulate two frames
        let scores1 = vec![-1.0, -2.0, -3.0]; // "a" is most likely
        let scores2 = vec![-2.0, -1.0, -3.0]; // "b" is most likely

        let results = decoder.decode(&[scores1, scores2]);

        assert!(!results.is_empty());
        assert_eq!(results[0].tokens.len(), 2);
    }

    #[test]
    fn test_hypothesis_extension() {
        let hyp = Hypothesis::new();
        let extended = hyp.extend("hello".to_string(), -1.0, -0.5, 0.3);

        assert_eq!(extended.tokens, vec!["hello"]);
        assert!(extended.acoustic_score < 0.0);
    }

    #[test]
    fn test_perplexity() {
        let corpus = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "c".to_string()],
        ];

        let mut lm = NgramLM::new(2);
        lm.train(&corpus);

        let pp = lm.perplexity(&corpus);

        // Perplexity should be positive and finite
        assert!(pp > 0.0);
        assert!(pp.is_finite());
    }
}
