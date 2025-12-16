//! Cerebellum - Procedural Memory System
//!
//! Week 2 Days 3-4: The Reflex Engine
//!
//! Revolutionary Insight:
//! "Reflexive Promotion" - Episodic memories that are recalled repeatedly
//! (5+ times) automatically promote to procedural reflexes for instant (<10ms) execution.
//!
//! The Cerebellum is the FAST PATH - pure muscle memory with no conscious thought.
//! It learns patterns, workflows, and context-aware responses through repetition.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use qp_trie::Trie;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument};

use crate::memory::{MemoryTrace, EmotionalValence};

/// Execution context for context-aware reflexes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// Current working directory
    pub cwd: Option<String>,

    /// Time of day (0-23 hours)
    pub hour: u8,

    /// Current emotional state
    pub emotion: EmotionalValence,

    /// Active tags/domains
    pub tags: Vec<String>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let hour = ((now % 86400) / 3600) as u8; // Unix timestamp → hour of day

        Self {
            cwd: None,
            hour,
            emotion: EmotionalValence::Neutral,
            tags: Vec::new(),
        }
    }
}

/// A learned skill - compiled procedural memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Skill name/trigger
    pub name: String,

    /// Command sequence to execute
    pub sequence: Vec<String>,

    /// Historical success rate (0.0 to 1.0)
    pub success_rate: f32,

    /// Total times executed
    pub execution_count: usize,

    /// Context tags this skill applies to
    pub context_tags: Vec<String>,

    /// Last used timestamp
    pub last_used: u64,

    /// Promotion timestamp (when it became a reflex)
    pub promoted_at: u64,
}

impl Skill {
    /// Create new skill from memory trace
    pub fn from_memory(trace: &MemoryTrace) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            name: trace.content.clone(),
            sequence: vec![trace.content.clone()],
            success_rate: 0.9, // Assume high success if promoted
            execution_count: 0,
            context_tags: trace.tags.clone(),
            last_used: now,
            promoted_at: now,
        }
    }

    /// Update skill after execution
    pub fn record_execution(&mut self, success: bool) {
        self.execution_count += 1;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_used = now;

        // Exponential moving average for success rate
        let alpha = 0.1; // Smoothing factor
        let outcome = if success { 1.0 } else { 0.0 };
        self.success_rate = alpha * outcome + (1.0 - alpha) * self.success_rate;
    }

    /// Check if skill matches execution context
    pub fn matches_context(&self, context: &ExecutionContext) -> bool {
        // If skill has no context requirements, it matches everything
        if self.context_tags.is_empty() {
            return true;
        }

        // Check if any skill tag matches context tags
        self.context_tags.iter()
            .any(|skill_tag| context.tags.contains(skill_tag))
    }

    /// Decay score - skills decay if unused
    pub fn decay_score(&self) -> f32 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let days_unused = (now - self.last_used) as f32 / 86400.0;

        // Decay formula: success_rate * e^(-decay_rate * days)
        let decay_rate = 0.01; // 1% decay per day
        self.success_rate * (-decay_rate * days_unused).exp()
    }
}

/// Workflow chain - learned action sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowChain {
    /// The command that triggers this chain
    pub trigger: String,

    /// The likely next command
    pub next: String,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Number of times this sequence occurred
    pub occurrence_count: usize,
}

/// The Cerebellum Actor - Procedural Memory & Reflex Engine
///
/// Implements:
/// 1. Reflexive Promotion: Episodic → Procedural migration
/// 2. Skill Chains: Workflow learning
/// 3. Context-Aware Execution: Time, location, emotional state
/// 4. Fuzzy Matching: Typo tolerance
pub struct CerebellumActor {
    /// The Skill Trie (Prefix Tree)
    /// Key: Command pattern bytes
    /// Value: The Skill
    skills: Trie<Vec<u8>, Skill>,

    /// Practice counters for promotion
    /// Key: Command pattern
    /// Value: (count, last_access_timestamp)
    practice_counts: HashMap<String, (usize, u64)>,

    /// Workflow chains (learned sequences)
    /// Key: Current command
    /// Value: List of likely next commands
    chains: HashMap<String, Vec<WorkflowChain>>,

    /// Last executed command (for chain learning)
    last_command: Option<String>,

    /// Promotion threshold (default: 5 recalls)
    promotion_threshold: usize,

    /// Fuzzy match tolerance (edit distance)
    fuzzy_tolerance: usize,

    /// Decay threshold for demotion
    decay_threshold: f32,
}

impl CerebellumActor {
    /// Create new Cerebellum with default settings
    pub fn new() -> Self {
        Self {
            skills: Trie::new(),
            practice_counts: HashMap::new(),
            chains: HashMap::new(),
            last_command: None,
            promotion_threshold: 5, // "Rule of 5"
            fuzzy_tolerance: 2,     // Allow 2-char edits
            decay_threshold: 0.3,   // Demote if below 30%
        }
    }

    /// Create Cerebellum with custom settings
    pub fn with_config(promotion_threshold: usize, fuzzy_tolerance: usize, decay_threshold: f32) -> Self {
        Self {
            skills: Trie::new(),
            practice_counts: HashMap::new(),
            chains: HashMap::new(),
            last_command: None,
            promotion_threshold,
            fuzzy_tolerance,
            decay_threshold,
        }
    }

    /// Attempt reflexive execution (FAST PATH <10ms)
    ///
    /// Returns Some(Skill) if a reflex matches, None otherwise
    #[instrument(skip(self))]
    pub fn try_reflex(&self, input: &str, context: &ExecutionContext) -> Option<Skill> {
        // 1. Try exact match
        if let Some(skill) = self.skills.get(input.as_bytes()) {
            if skill.matches_context(context) {
                info!(
                    skill = %skill.name,
                    success_rate = skill.success_rate,
                    "⚡ Reflex matched (exact)"
                );
                return Some(skill.clone());
            }
        }

        // 2. Try prefix match (autocomplete)
        let mut matches: Vec<_> = self.skills
            .iter_prefix(input.as_bytes())
            .filter_map(|(_, skill)| {
                if skill.matches_context(context) {
                    Some(skill.clone())
                } else {
                    None
                }
            })
            .collect();

        if !matches.is_empty() {
            // Sort by decay score (best skill first)
            matches.sort_by(|a, b| {
                b.decay_score().partial_cmp(&a.decay_score()).unwrap()
            });

            let best = &matches[0];
            info!(
                skill = %best.name,
                success_rate = best.success_rate,
                "⚡ Reflex matched (prefix)"
            );
            return Some(best.clone());
        }

        // 3. Try fuzzy match (typo tolerance)
        if let Some(skill) = self.fuzzy_match(input, context) {
            info!(
                skill = %skill.name,
                success_rate = skill.success_rate,
                "⚡ Reflex matched (fuzzy)"
            );
            return Some(skill);
        }

        // 4. No reflex found
        None
    }

    /// Fuzzy match for typo tolerance
    fn fuzzy_match(&self, input: &str, context: &ExecutionContext) -> Option<Skill> {
        let mut best_match: Option<(Skill, usize)> = None;

        for (key_bytes, skill) in self.skills.iter() {
            if !skill.matches_context(context) {
                continue;
            }

            if let Ok(key) = String::from_utf8(key_bytes.to_vec()) {
                let distance = strsim::levenshtein(input, &key);

                if distance <= self.fuzzy_tolerance {
                    match &best_match {
                        None => best_match = Some((skill.clone(), distance)),
                        Some((_, best_dist)) if distance < *best_dist => {
                            best_match = Some((skill.clone(), distance));
                        }
                        _ => {}
                    }
                }
            }
        }

        best_match.map(|(skill, _)| skill)
    }

    /// Practice a pattern (called when Hippocampus recalls memory)
    ///
    /// The "Reflexive Promotion" logic:
    /// If a memory is recalled 5+ times, promote it to a procedural reflex
    #[instrument(skip(self))]
    pub fn practice(&mut self, pattern: &str, trace: &MemoryTrace) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = self.practice_counts
            .entry(pattern.to_string())
            .or_insert((0, now));

        entry.0 += 1;
        entry.1 = now;

        let count = entry.0;

        // The "Rule of 5": Promote after 5 practices
        if count >= self.promotion_threshold {
            // Only promote if emotion is not negative
            if trace.emotion != EmotionalValence::Negative {
                let skill = Skill::from_memory(trace);
                self.skills.insert(pattern.as_bytes().to_vec(), skill.clone());

                info!(
                    pattern = %pattern,
                    count = count,
                    "🎓 PROMOTION: Pattern promoted to procedural reflex"
                );

                // Clear practice counter (it's now a reflex)
                self.practice_counts.remove(pattern);
            } else {
                info!(
                    pattern = %pattern,
                    "⚠️  Promotion blocked: Negative emotional valence"
                );
            }
        } else {
            info!(
                pattern = %pattern,
                count = count,
                threshold = self.promotion_threshold,
                "🏋️ Practicing... ({}/{})",
                count, self.promotion_threshold
            );
        }
    }

    /// Learn workflow chain (command sequences)
    ///
    /// Called after each command execution to learn patterns like:
    /// "nix build" → "nix run" → "git commit"
    #[instrument(skip(self))]
    pub fn learn_chain(&mut self, current_command: &str) {
        if let Some(last) = &self.last_command {
            // Check if chain exists
            let chains = self.chains.entry(last.clone()).or_insert_with(Vec::new);

            // Find existing chain index
            let existing_index = chains.iter().position(|c| c.next == current_command);

            if let Some(index) = existing_index {
                // Update existing chain
                chains[index].occurrence_count += 1;

                // Calculate total AFTER updating count
                let total: usize = chains.iter().map(|c| c.occurrence_count).sum();
                chains[index].confidence = chains[index].occurrence_count as f32 / total as f32;
            } else {
                // Create new chain
                chains.push(WorkflowChain {
                    trigger: last.clone(),
                    next: current_command.to_string(),
                    confidence: 1.0 / (chains.len() + 1) as f32, // Initial confidence
                    occurrence_count: 1,
                });
            }

            info!(
                from = %last,
                to = %current_command,
                "🔗 Workflow chain learned"
            );
        }

        // Update last command
        self.last_command = Some(current_command.to_string());
    }

    /// Suggest next action in workflow
    ///
    /// Returns the most likely next command based on learned chains
    pub fn suggest_next(&self, current_command: &str) -> Option<WorkflowChain> {
        if let Some(chains) = self.chains.get(current_command) {
            // Return highest confidence chain
            chains.iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                .cloned()
        } else {
            None
        }
    }

    /// Record execution outcome
    ///
    /// Updates skill success rate and checks for demotion
    #[instrument(skip(self))]
    pub fn record_execution(&mut self, pattern: &str, success: bool) {
        if let Some(skill) = self.skills.get_mut(pattern.as_bytes()) {
            skill.record_execution(success);

            info!(
                pattern = %pattern,
                success = success,
                new_rate = skill.success_rate,
                "📊 Execution recorded"
            );

            // Check for demotion (skill decay)
            let decay = skill.decay_score();
            if decay < self.decay_threshold {
                info!(
                    pattern = %pattern,
                    decay_score = decay,
                    threshold = self.decay_threshold,
                    "⬇️  DEMOTION: Skill decayed below threshold, removing reflex"
                );
                self.skills.remove(pattern.as_bytes());

                // Restore practice counter (it's episodic again)
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                self.practice_counts.insert(pattern.to_string(), (0, now));
            }
        }
    }

    /// Get skill by pattern
    pub fn get_skill(&self, pattern: &str) -> Option<&Skill> {
        self.skills.get(pattern.as_bytes())
    }

    /// Count total skills
    pub fn skill_count(&self) -> usize {
        self.skills.count()
    }

    /// Get practice count for pattern
    pub fn practice_count(&self, pattern: &str) -> usize {
        self.practice_counts
            .get(pattern)
            .map(|(count, _)| *count)
            .unwrap_or(0)
    }

    /// Get all workflow chains
    pub fn get_chains(&self) -> &HashMap<String, Vec<WorkflowChain>> {
        &self.chains
    }

    /// Statistics about procedural memory
    pub fn stats(&self) -> CerebellumStats {
        let total_skills = self.skill_count();
        let avg_success_rate = if total_skills > 0 {
            self.skills.iter()
                .map(|(_, skill)| skill.success_rate)
                .sum::<f32>() / total_skills as f32
        } else {
            0.0
        };

        let total_chains: usize = self.chains.values()
            .map(|chains| chains.len())
            .sum();

        CerebellumStats {
            total_skills,
            avg_success_rate,
            patterns_practicing: self.practice_counts.len(),
            total_chains,
        }
    }
}

/// Statistics about procedural memory
#[derive(Debug, Clone)]
pub struct CerebellumStats {
    pub total_skills: usize,
    pub avg_success_rate: f32,
    pub patterns_practicing: usize,
    pub total_chains: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_trace(content: &str, tags: Vec<String>, emotion: EmotionalValence) -> MemoryTrace {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        MemoryTrace {
            id: 0,
            timestamp: now,
            encoding: vec![0.0; 10_000],
            hdc_encoding: None,  // Week 14 Day 3: Optional HDC encoding
            emotion,
            tags,
            content: content.to_string(),
            recall_count: 0,
            strength: 0.5,
        }
    }

    #[test]
    fn test_cerebellum_creation() {
        let cerebellum = CerebellumActor::new();
        assert_eq!(cerebellum.skill_count(), 0);
        assert_eq!(cerebellum.promotion_threshold, 5);
    }

    #[test]
    fn test_practice_and_promotion() {
        let mut cerebellum = CerebellumActor::new();
        let trace = create_test_trace(
            "git push",
            vec!["git".to_string()],
            EmotionalValence::Neutral,
        );

        // Practice 4 times (not promoted yet)
        for i in 0..4 {
            cerebellum.practice("git push", &trace);
            assert_eq!(cerebellum.practice_count("git push"), i + 1);
        }

        // Not promoted yet
        assert_eq!(cerebellum.skill_count(), 0);

        // 5th practice triggers promotion
        cerebellum.practice("git push", &trace);

        // After promotion: skill exists, practice counter cleared
        assert_eq!(cerebellum.skill_count(), 1);
        assert!(cerebellum.get_skill("git push").is_some());
        assert_eq!(cerebellum.practice_count("git push"), 0); // Cleared after promotion
    }

    #[test]
    fn test_reflex_exact_match() {
        let mut cerebellum = CerebellumActor::new();
        let trace = create_test_trace(
            "nix build",
            vec!["nixos".to_string()],
            EmotionalValence::Positive,
        );

        // Promote skill
        for _ in 0..5 {
            cerebellum.practice("nix build", &trace);
        }

        // Try reflex
        let context = ExecutionContext {
            tags: vec!["nixos".to_string()],
            ..Default::default()
        };

        let result = cerebellum.try_reflex("nix build", &context);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "nix build");
    }

    #[test]
    fn test_reflex_prefix_match() {
        let mut cerebellum = CerebellumActor::new();
        let trace = create_test_trace(
            "nix build",
            vec![],
            EmotionalValence::Neutral,
        );

        // Promote skill
        for _ in 0..5 {
            cerebellum.practice("nix build", &trace);
        }

        // Try prefix
        let context = ExecutionContext::default();
        let result = cerebellum.try_reflex("nix bu", &context);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "nix build");
    }

    #[test]
    fn test_fuzzy_match_typo() {
        let mut cerebellum = CerebellumActor::new();
        let trace = create_test_trace(
            "nix build",
            vec![],
            EmotionalValence::Neutral,
        );

        // Promote skill
        for _ in 0..5 {
            cerebellum.practice("nix build", &trace);
        }

        // Try with typo (1-char edit)
        let context = ExecutionContext::default();
        let result = cerebellum.try_reflex("nix buld", &context);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "nix build");
    }

    #[test]
    fn test_context_filtering() {
        let mut cerebellum = CerebellumActor::new();
        let trace = create_test_trace(
            "run server",
            vec!["backend".to_string()],
            EmotionalValence::Neutral,
        );

        // Promote skill with context
        for _ in 0..5 {
            cerebellum.practice("run server", &trace);
        }

        // Try with matching context
        let context_match = ExecutionContext {
            tags: vec!["backend".to_string()],
            ..Default::default()
        };
        assert!(cerebellum.try_reflex("run server", &context_match).is_some());

        // Try with non-matching context
        let context_no_match = ExecutionContext {
            tags: vec!["frontend".to_string()],
            ..Default::default()
        };
        assert!(cerebellum.try_reflex("run server", &context_no_match).is_none());
    }

    #[test]
    fn test_workflow_chains() {
        let mut cerebellum = CerebellumActor::new();

        // Learn a workflow: build → test → commit
        cerebellum.learn_chain("nix build");
        cerebellum.learn_chain("nix test");
        cerebellum.learn_chain("git commit");

        // Check suggestions
        let suggestion = cerebellum.suggest_next("nix build");
        assert!(suggestion.is_some());
        assert_eq!(suggestion.unwrap().next, "nix test");

        let suggestion2 = cerebellum.suggest_next("nix test");
        assert!(suggestion2.is_some());
        assert_eq!(suggestion2.unwrap().next, "git commit");
    }

    #[test]
    fn test_success_rate_tracking() {
        let mut cerebellum = CerebellumActor::new();
        let trace = create_test_trace(
            "deploy",
            vec![],
            EmotionalValence::Neutral,
        );

        // Promote skill
        for _ in 0..5 {
            cerebellum.practice("deploy", &trace);
        }

        let skill = cerebellum.get_skill("deploy").unwrap();
        let initial_rate = skill.success_rate;

        // Record successful execution
        cerebellum.record_execution("deploy", true);
        let skill = cerebellum.get_skill("deploy").unwrap();
        assert!(skill.success_rate >= initial_rate);

        // Record failed execution
        cerebellum.record_execution("deploy", false);
        let skill = cerebellum.get_skill("deploy").unwrap();
        // Success rate should decrease
        assert!(skill.success_rate < 0.9);
    }

    #[test]
    fn test_negative_emotion_blocks_promotion() {
        let mut cerebellum = CerebellumActor::new();
        let trace = create_test_trace(
            "dangerous command",
            vec![],
            EmotionalValence::Negative,
        );

        // Practice 5 times with negative emotion
        for _ in 0..5 {
            cerebellum.practice("dangerous command", &trace);
        }

        // Should NOT be promoted
        assert_eq!(cerebellum.skill_count(), 0);
        assert!(cerebellum.get_skill("dangerous command").is_none());
    }

    #[test]
    fn test_stats() {
        let mut cerebellum = CerebellumActor::new();
        let trace1 = create_test_trace("cmd1", vec![], EmotionalValence::Positive);
        let trace2 = create_test_trace("cmd2", vec![], EmotionalValence::Neutral);

        // Promote 2 skills
        for _ in 0..5 {
            cerebellum.practice("cmd1", &trace1);
            cerebellum.practice("cmd2", &trace2);
        }

        // Learn a chain
        cerebellum.learn_chain("cmd1");
        cerebellum.learn_chain("cmd2");

        let stats = cerebellum.stats();
        assert_eq!(stats.total_skills, 2);
        assert_eq!(stats.total_chains, 1);
    }
}
