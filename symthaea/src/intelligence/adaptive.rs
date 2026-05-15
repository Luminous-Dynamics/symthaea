// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! H1: Adaptive Completions
//!
//! Learns from user behavior to improve completion suggestions.
//! Uses HDC vectors to encode and match usage patterns.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

/// A learned user pattern
#[derive(Debug, Clone)]
pub struct UserPattern {
    /// Pattern identifier
    pub id: u64,
    /// Context when pattern was used (partial input)
    pub context: String,
    /// Selected completion
    pub selection: String,
    /// Number of times this pattern occurred
    pub frequency: u32,
    /// Last time pattern was used
    pub last_used: SystemTime,
    /// Decay factor (decreases over time)
    pub weight: f32,
}

impl UserPattern {
    /// Create a new pattern
    pub fn new(context: impl Into<String>, selection: impl Into<String>) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        Self {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            context: context.into(),
            selection: selection.into(),
            frequency: 1,
            last_used: SystemTime::now(),
            weight: 1.0,
        }
    }

    /// Reinforce this pattern (user selected it again)
    pub fn reinforce(&mut self) {
        self.frequency += 1;
        self.last_used = SystemTime::now();
        self.weight = (self.weight * 1.1).min(10.0);
    }

    /// Apply time decay to weight
    pub fn apply_decay(&mut self, half_life_days: f64) {
        let age = SystemTime::now()
            .duration_since(self.last_used)
            .unwrap_or_default();

        let days = age.as_secs_f64() / 86400.0;
        let decay = 0.5_f64.powf(days / half_life_days);
        self.weight = (self.weight as f64 * decay) as f32;
    }

    /// Calculate relevance score for a given context using semantic similarity
    pub fn relevance(&self, input: &str) -> f32 {
        use symthaea_core::hdc::semantic_decoder::SemanticSimilarity;
        
        // Exact match
        if self.context == input {
            return self.weight * 2.0;
        }

        // Semantic similarity using HDC vectors
        let context_hv = SemanticSimilarity::encode(&self.context);
        let input_hv = SemanticSimilarity::encode(input);
        let semantic_sim = context_hv.cosine_similarity(&input_hv);

        // Combine with lexical similarity
        let lexical_sim = lexical_similarity(&self.context, input);
        
        // Weighted combination
        (self.weight * 0.7 * semantic_sim) + (self.weight * 0.3 * lexical_sim)
    }

    fn lexical_similarity(a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        if a_lower == b_lower {
            return 1.0;
        }

        if a_lower.starts_with(&b_lower) || b_lower.starts_with(&a_lower) {
            return 0.8;
        }

        let common_prefix = a_lower
            .chars()
            .zip(b_lower.chars())
            .take_while(|(a, b)| a == b)
            .count();

        if common_prefix > 0 {
            let max_len = a_lower.len().max(b_lower.len()) as f32;
            return common_prefix as f32 / max_len;
        }

        0.0
    }
}

/// Storage for user patterns
pub struct PatternStore {
    /// Patterns indexed by context prefix
    patterns: HashMap<String, Vec<UserPattern>>,
    /// Storage path
    path: Option<PathBuf>,
    /// Maximum patterns per prefix
    max_per_prefix: usize,
    /// Total pattern count
    total_count: usize,
    /// Half-life for decay (in days)
    decay_half_life: f64,
}

impl PatternStore {
    /// Create a new pattern store
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            path: None,
            max_per_prefix: 50,
            total_count: 0,
            decay_half_life: 30.0,
        }
    }

    /// Set storage path
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Get prefix key for a context
    fn prefix_key(context: &str) -> String {
        // Use first 3 characters or whole string
        let trimmed = context.trim();
        trimmed.chars().take(3).collect()
    }

    /// Record a pattern (context -> selection)
    pub fn record(&mut self, context: &str, selection: &str) {
        let key = Self::prefix_key(context);
        let patterns = self.patterns.entry(key).or_default();

        // Check if pattern exists
        if let Some(existing) = patterns
            .iter_mut()
            .find(|p| p.context == context && p.selection == selection)
        {
            existing.reinforce();
            return;
        }

        // Add new pattern
        let pattern = UserPattern::new(context, selection);
        patterns.push(pattern);
        self.total_count += 1;

        // Trim if over limit
        while patterns.len() > self.max_per_prefix {
            // Remove lowest weight pattern
            if let Some(min_idx) = patterns
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.weight
                        .partial_cmp(&b.weight)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                patterns.remove(min_idx);
                self.total_count -= 1;
            }
        }
    }

    /// Query patterns for a context
    pub fn query(&self, context: &str) -> Vec<(&UserPattern, f32)> {
        let key = Self::prefix_key(context);
        let mut results = Vec::new();

        if let Some(patterns) = self.patterns.get(&key) {
            for pattern in patterns {
                let relevance = pattern.relevance(context);
                if relevance > 0.01 {
                    results.push((pattern, relevance));
                }
            }
        }

        // Sort by relevance
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Apply decay to all patterns
    pub fn apply_decay(&mut self) {
        for patterns in self.patterns.values_mut() {
            for pattern in patterns.iter_mut() {
                pattern.apply_decay(self.decay_half_life);
            }
        }
    }

    /// Prune low-weight patterns
    pub fn prune(&mut self, min_weight: f32) {
        for patterns in self.patterns.values_mut() {
            let before = patterns.len();
            patterns.retain(|p| p.weight >= min_weight);
            self.total_count -= before - patterns.len();
        }
    }

    /// Save to file
    pub fn save(&self) -> io::Result<()> {
        if let Some(path) = &self.path {
            let mut content = String::new();

            for patterns in self.patterns.values() {
                for p in patterns {
                    let timestamp = p
                        .last_used
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();

                    content.push_str(&format!(
                        "{}|{}|{}|{}|{}\n",
                        p.context.replace('|', "\\|"),
                        p.selection.replace('|', "\\|"),
                        p.frequency,
                        timestamp,
                        p.weight
                    ));
                }
            }

            fs::write(path, content)?;
        }
        Ok(())
    }

    /// Load from file
    pub fn load(&mut self) -> io::Result<()> {
        if let Some(path) = &self.path {
            if !path.exists() {
                return Ok(());
            }

            let content = fs::read_to_string(path)?;

            for line in content.lines() {
                let parts: Vec<&str> = line.split('|').collect();
                if parts.len() >= 5 {
                    let context = parts[0].replace("\\|", "|");
                    let selection = parts[1].replace("\\|", "|");
                    let frequency: u32 = parts[2].parse().unwrap_or(1);
                    let timestamp: u64 = parts[3].parse().unwrap_or(0);
                    let weight: f32 = parts[4].parse().unwrap_or(1.0);

                    let pattern = UserPattern {
                        id: 0,
                        context: context.clone(),
                        selection,
                        frequency,
                        last_used: SystemTime::UNIX_EPOCH + Duration::from_secs(timestamp),
                        weight,
                    };

                    let key = Self::prefix_key(&context);
                    self.patterns.entry(key).or_default().push(pattern);
                    self.total_count += 1;
                }
            }
        }
        Ok(())
    }

    /// Get total pattern count
    pub fn len(&self) -> usize {
        self.total_count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }
}

impl Default for PatternStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive completion engine
pub struct AdaptiveCompleter {
    /// Pattern store
    patterns: PatternStore,
    /// Boost factor for learned patterns
    boost_factor: f32,
    /// Minimum relevance to include
    min_relevance: f32,
    /// Last decay application
    last_decay: Instant,
    /// Decay interval
    decay_interval: Duration,
}

impl AdaptiveCompleter {
    /// Create new adaptive completer
    pub fn new() -> Self {
        Self {
            patterns: PatternStore::new(),
            boost_factor: 1.5,
            min_relevance: 0.1,
            last_decay: Instant::now(),
            decay_interval: Duration::from_secs(3600), // 1 hour
        }
    }

    /// Create with storage
    pub fn with_storage(path: impl Into<PathBuf>) -> io::Result<Self> {
        let mut completer = Self::new();
        completer.patterns = PatternStore::new().with_path(path);
        completer.patterns.load()?;
        Ok(completer)
    }

    /// Record user selection
    pub fn record_selection(&mut self, context: &str, selection: &str) {
        self.patterns.record(context, selection);
    }

    /// Adjust completion scores based on learned patterns
    pub fn adjust_scores(&mut self, context: &str, completions: &mut [(String, f32)]) {
        // Apply decay periodically
        if self.last_decay.elapsed() > self.decay_interval {
            self.patterns.apply_decay();
            self.last_decay = Instant::now();
        }

        // Query patterns
        let matches = self.patterns.query(context);

        // Boost matching completions
        for (completion, score) in completions.iter_mut() {
            for (pattern, relevance) in &matches {
                if pattern.selection == *completion {
                    *score *= self.boost_factor * relevance;
                }
            }
        }
    }

    /// Get suggested completions from patterns alone
    pub fn suggest(&self, context: &str) -> Vec<(String, f32)> {
        self.patterns
            .query(context)
            .into_iter()
            .filter(|(_, r)| *r >= self.min_relevance)
            .map(|(p, r)| (p.selection.clone(), r))
            .collect()
    }

    /// Save patterns
    pub fn save(&self) -> io::Result<()> {
        self.patterns.save()
    }

    /// Get statistics
    pub fn stats(&self) -> AdaptiveStats {
        AdaptiveStats {
            pattern_count: self.patterns.len(),
            boost_factor: self.boost_factor,
        }
    }
}

impl Default for AdaptiveCompleter {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about adaptive completion
#[derive(Debug, Clone)]
pub struct AdaptiveStats {
    pub pattern_count: usize,
    pub boost_factor: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_pattern() {
        let mut pattern = UserPattern::new("inst", "install");

        assert_eq!(pattern.frequency, 1);

        pattern.reinforce();
        assert_eq!(pattern.frequency, 2);
        assert!(pattern.weight > 1.0);
    }

    #[test]
    fn test_pattern_relevance() {
        let pattern = UserPattern::new("install fire", "install firefox");

        let exact = pattern.relevance("install fire");
        let partial = pattern.relevance("install");
        let different = pattern.relevance("search");

        assert!(exact > partial);
        assert!(partial > different);
    }

    #[test]
    fn test_pattern_store() {
        let mut store = PatternStore::new();

        store.record("install", "install firefox");
        store.record("install", "install firefox");
        store.record("install", "install vim");

        let results = store.query("install");
        assert!(!results.is_empty());

        // Firefox should be ranked higher (used twice)
        let top = &results[0].0;
        assert_eq!(top.selection, "install firefox");
    }

    #[test]
    fn test_adaptive_completer() {
        let mut completer = AdaptiveCompleter::new();

        completer.record_selection("nix", "nix search");
        completer.record_selection("nix", "nix search");

        let mut completions = vec![
            ("nix search".to_string(), 1.0),
            ("nix build".to_string(), 1.0),
        ];

        completer.adjust_scores("nix", &mut completions);

        // "nix search" should have higher score
        assert!(completions[0].1 > completions[1].1);
    }
}
