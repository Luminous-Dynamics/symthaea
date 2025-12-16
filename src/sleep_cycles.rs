/*!
Sleep Cycles - Homeostatic Memory Consolidation

Solves the "Memory Growth Problem" via biological sleep mechanisms:
1. **Synaptic Scaling**: Weakens unused memories, strengthens important ones
2. **Memory Consolidation**: Transfers short-term → long-term
3. **Garbage Collection**: Prunes redundant/forgotten memories
4. **Pattern Extraction**: Discovers recurring patterns during sleep

Inspired by mammalian sleep cycles and synaptic homeostasis theory.
*/

use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Memory importance threshold (below this = candidate for pruning)
const IMPORTANCE_THRESHOLD: f32 = 0.1;

/// Decay rate per sleep cycle (0.0 = no decay, 1.0 = full decay)
const DECAY_RATE: f32 = 0.05;

/// Memory consolidation threshold (above this = move to long-term)
const CONSOLIDATION_THRESHOLD: f32 = 0.7;

/// A memory entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Hypervector content (10,000D)
    pub content: Vec<i8>,

    /// Importance weight (0.0 to 1.0)
    pub importance: f32,

    /// Access count (how often recalled)
    pub access_count: usize,

    /// Last access time
    pub last_access: SystemTime,

    /// Creation time
    pub created_at: SystemTime,

    /// Memory type
    pub memory_type: MemoryType,
}

/// Memory types with different consolidation rules
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    /// Short-term working memory (volatile)
    ShortTerm,

    /// Long-term consolidated memory (persistent)
    LongTerm,

    /// Procedural memory (skills, never pruned)
    Procedural,

    /// Episodic memory (experiences, pruned by importance)
    Episodic,
}

/// Sleep Cycle Manager
pub struct SleepCycleManager {
    /// Short-term memory store
    short_term: Arc<DashMap<String, MemoryEntry>>,

    /// Long-term memory store
    long_term: Arc<DashMap<String, MemoryEntry>>,

    /// Sleep statistics
    cycles_completed: Arc<std::sync::Mutex<usize>>,
    memories_pruned: Arc<std::sync::Mutex<usize>>,
    memories_consolidated: Arc<std::sync::Mutex<usize>>,

    /// Configuration
    config: SleepConfig,
}

/// Sleep configuration
#[derive(Debug, Clone)]
pub struct SleepConfig {
    /// Enable automatic sleep cycles
    pub auto_sleep: bool,

    /// Sleep cycle interval (e.g., every N operations)
    pub cycle_interval: usize,

    /// Importance decay rate
    pub decay_rate: f32,

    /// Consolidation threshold
    pub consolidation_threshold: f32,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            auto_sleep: true,
            cycle_interval: 1000,  // Sleep every 1000 operations
            decay_rate: DECAY_RATE,
            consolidation_threshold: CONSOLIDATION_THRESHOLD,
        }
    }
}

impl SleepCycleManager {
    /// Create new sleep cycle manager
    pub fn new(config: SleepConfig) -> Self {
        Self {
            short_term: Arc::new(DashMap::new()),
            long_term: Arc::new(DashMap::new()),
            cycles_completed: Arc::new(std::sync::Mutex::new(0)),
            memories_pruned: Arc::new(std::sync::Mutex::new(0)),
            memories_consolidated: Arc::new(std::sync::Mutex::new(0)),
            config,
        }
    }

    /// Store memory in short-term
    pub fn remember(&self, key: String, content: Vec<i8>, memory_type: MemoryType) {
        let entry = MemoryEntry {
            content,
            importance: 1.0,  // Start with max importance
            access_count: 0,
            last_access: SystemTime::now(),
            created_at: SystemTime::now(),
            memory_type,
        };

        self.short_term.insert(key, entry);
    }

    /// Recall memory (updates access stats)
    pub fn recall(&self, key: &str) -> Option<Vec<i8>> {
        // Try short-term first
        if let Some(mut entry) = self.short_term.get_mut(key) {
            entry.access_count += 1;
            entry.last_access = SystemTime::now();
            entry.importance = (entry.importance + 0.1).min(1.0);  // Boost importance
            return Some(entry.content.clone());
        }

        // Try long-term
        if let Some(mut entry) = self.long_term.get_mut(key) {
            entry.access_count += 1;
            entry.last_access = SystemTime::now();
            return Some(entry.content.clone());
        }

        None
    }

    /// Execute sleep cycle (consolidation + pruning)
    pub async fn sleep(&self) -> Result<SleepReport> {
        tracing::info!("😴 Entering sleep cycle...");

        let mut report = SleepReport::default();

        // Phase 1: Synaptic Scaling (decay unused memories)
        self.synaptic_scaling(&mut report);

        // Phase 2: Memory Consolidation (short-term → long-term)
        self.consolidate_memories(&mut report);

        // Phase 3: Garbage Collection (prune unimportant)
        self.prune_memories(&mut report);

        // Phase 4: Pattern Extraction (discover recurring patterns)
        self.extract_patterns(&mut report);

        // Update statistics
        *self.cycles_completed.lock().unwrap() += 1;
        *self.memories_pruned.lock().unwrap() += report.pruned;
        *self.memories_consolidated.lock().unwrap() += report.consolidated;

        tracing::info!("✨ Sleep cycle complete: {}", report);

        Ok(report)
    }

    /// Phase 1: Synaptic Scaling
    fn synaptic_scaling(&self, report: &mut SleepReport) {
        let now = SystemTime::now();

        for mut entry in self.short_term.iter_mut() {
            // Calculate time since last access
            let time_unused = now
                .duration_since(entry.last_access)
                .unwrap_or(Duration::ZERO)
                .as_secs() as f32;

            // Decay importance based on time unused (exponential decay)
            let decay_factor = (-time_unused / 3600.0).exp();  // Decay over hours
            entry.importance *= decay_factor * (1.0 - self.config.decay_rate);

            report.scaled += 1;
        }
    }

    /// Phase 2: Memory Consolidation
    fn consolidate_memories(&self, report: &mut SleepReport) {
        let keys_to_consolidate: Vec<String> = self
            .short_term
            .iter()
            .filter(|entry| {
                entry.importance >= self.config.consolidation_threshold
                    && entry.memory_type != MemoryType::Procedural
            })
            .map(|entry| entry.key().clone())
            .collect();

        for key in keys_to_consolidate {
            if let Some((k, mut entry)) = self.short_term.remove(&key) {
                entry.memory_type = MemoryType::LongTerm;
                self.long_term.insert(k, entry);
                report.consolidated += 1;
            }
        }
    }

    /// Phase 3: Garbage Collection
    fn prune_memories(&self, report: &mut SleepReport) {
        // Prune short-term memories below threshold
        let keys_to_prune: Vec<String> = self
            .short_term
            .iter()
            .filter(|entry| {
                entry.importance < IMPORTANCE_THRESHOLD
                    && entry.memory_type != MemoryType::Procedural
            })
            .map(|entry| entry.key().clone())
            .collect();

        for key in keys_to_prune {
            self.short_term.remove(&key);
            report.pruned += 1;
        }
    }

    /// Phase 4: Pattern Extraction (placeholder)
    fn extract_patterns(&self, report: &mut SleepReport) {
        // TODO: Use resonator networks to find recurring patterns
        // For now, just count pattern candidates

        let pattern_candidates = self
            .short_term
            .iter()
            .filter(|entry| entry.access_count > 5)
            .count();

        report.patterns_extracted = pattern_candidates;
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            short_term_count: self.short_term.len(),
            long_term_count: self.long_term.len(),
            cycles_completed: *self.cycles_completed.lock().unwrap(),
            memories_pruned: *self.memories_pruned.lock().unwrap(),
            memories_consolidated: *self.memories_consolidated.lock().unwrap(),
        }
    }

    /// Force garbage collection (manual sleep)
    pub async fn force_sleep(&self) -> Result<SleepReport> {
        self.sleep().await
    }

    /// Check if sleep cycle is needed
    pub fn should_sleep(&self) -> bool {
        if !self.config.auto_sleep {
            return false;
        }

        // Sleep if short-term memory is getting large
        self.short_term.len() > 1000
    }
}

/// Sleep cycle report
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SleepReport {
    pub scaled: usize,
    pub consolidated: usize,
    pub pruned: usize,
    pub patterns_extracted: usize,
}

impl std::fmt::Display for SleepReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Scaled: {}, Consolidated: {}, Pruned: {}, Patterns: {}",
            self.scaled, self.consolidated, self.pruned, self.patterns_extracted
        )
    }
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub short_term_count: usize,
    pub long_term_count: usize,
    pub cycles_completed: usize,
    pub memories_pruned: usize,
    pub memories_consolidated: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sleep_cycle_manager() {
        let manager = SleepCycleManager::new(SleepConfig::default());

        // Store some memories
        manager.remember("test1".to_string(), vec![1; 10_000], MemoryType::ShortTerm);
        manager.remember("test2".to_string(), vec![2; 10_000], MemoryType::ShortTerm);

        // Recall one to boost importance
        for _ in 0..10 {
            manager.recall("test1");
        }

        // Sleep cycle
        let report = manager.sleep().await.unwrap();

        assert!(report.scaled > 0);
    }

    #[test]
    fn test_memory_recall() {
        let manager = SleepCycleManager::new(SleepConfig::default());

        let content = vec![42i8; 10_000];
        manager.remember("test".to_string(), content.clone(), MemoryType::ShortTerm);

        let recalled = manager.recall("test");
        assert_eq!(recalled, Some(content));
    }

    #[test]
    fn test_stats_tracking() {
        let manager = SleepCycleManager::new(SleepConfig::default());

        manager.remember("test".to_string(), vec![1; 10_000], MemoryType::ShortTerm);

        let stats = manager.stats();
        assert_eq!(stats.short_term_count, 1);
    }
}
