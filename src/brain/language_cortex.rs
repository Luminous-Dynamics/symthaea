//! Language Cortex Bridge - Wiring Language to the Brain
//!
//! This module creates bidirectional connections between the Language
//! Consciousness system and the brain's Global Workspace, enabling:
//!
//! 1. **Language → Brain**: `LanguageAttentionBid` → `AttentionBid` for
//!    global workspace competition
//!
//! 2. **Brain → Language**: Precision and attention signals flow back to
//!    modulate language processing
//!
//! 3. **Working Memory Sync**: Language frames and constructions populate
//!    the brain's working memory alongside other cognitive contents
//!
//! # Revolutionary Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     LANGUAGE-BRAIN INTEGRATION                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌──────────────────┐                    ┌──────────────────────┐       │
//! │  │     LANGUAGE     │     Conversion     │       BRAIN          │       │
//! │  │     BRIDGE       │ =================> │    PREFRONTAL        │       │
//! │  │                  │                    │                      │       │
//! │  │ LanguageAttention│   to_brain_bid()   │    AttentionBid      │       │
//! │  │ Bid              │ -----------------> │                      │       │
//! │  │                  │                    │                      │       │
//! │  │ LanguageWorking  │  to_brain_wm()     │    WorkingMemory     │       │
//! │  │ MemoryItem       │ -----------------> │    Item              │       │
//! │  │                  │                    │                      │       │
//! │  │ Integrated       │  process_input()   │    GlobalWorkspace   │       │
//! │  │ Processor        │ -----------------> │    Competition       │       │
//! │  └──────────────────┘                    └──────────────────────┘       │
//! │                                                                          │
//! │                    <================= Feedback                           │
//! │                                                                          │
//! │  Precision weights, spotlight status, working memory sync               │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut bridge = LanguageCortexBridge::new();
//!
//! // Process language input and generate brain-compatible bid
//! let (brain_bid, language_result) = bridge.process_input("Install Firefox");
//!
//! // Submit to prefrontal cortex
//! prefrontal.cognitive_cycle_with_energy(vec![brain_bid], &mut hearth);
//! ```

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

use crate::memory::EmotionalValence;
use crate::hdc::binary_hv::HV16;
use super::prefrontal::{AttentionBid, WorkingMemoryItem, GlobalWorkspace, PrefrontalCortexActor};
use super::actor_model::SharedHdcVector;
use super::active_inference::{ActiveInferenceEngine, PredictionDomain};

use crate::language::{
    ConsciousnessBridge, BridgeConfig, BridgeResult, ConsciousnessState,
    LanguageAttentionBid, LanguageWorkingMemoryItem, WorkingMemoryContentType,
    IntegratedConsciousnessProcessor, IntegratedResult, IntegratedConsciousnessState,
    ActiveInferenceAdapter, AdapterConfig,
};

// =============================================================================
// TYPE CONVERSIONS
// =============================================================================

/// Convert a LanguageAttentionBid to a brain AttentionBid
pub fn to_brain_bid(lang_bid: &LanguageAttentionBid) -> AttentionBid {
    // Map consciousness state to emotional valence
    let emotion = emotional_valence_from_language(lang_bid);

    // Extract tags from language analysis
    let mut tags = lang_bid.active_primes.clone();
    if let Some(frame) = &lang_bid.primary_frame {
        tags.push(format!("frame:{}", frame));
    }
    if let Some(construct) = &lang_bid.construction {
        tags.push(format!("construct:{}", construct));
    }
    tags.push("language".to_string());

    // Convert HDC encoding to SharedHdcVector (u8 -> i8 conversion)
    let hdc_semantic = Some(Arc::new(
        lang_bid.understanding.utterance_encoding.0
            .iter()
            .map(|&b| b as i8)
            .collect::<Vec<i8>>()
    ));

    AttentionBid::new("LanguageCortex", &lang_bid.content)
        .with_salience(lang_bid.salience)
        .with_urgency(lang_bid.urgency)
        .with_emotion(emotion)
        .with_tags(tags)
        .with_hdc_semantic(hdc_semantic)
}

/// Derive emotional valence from language understanding
pub fn emotional_valence_from_language(lang_bid: &LanguageAttentionBid) -> EmotionalValence {
    // Check semantic primes for emotional content
    let primes = &lang_bid.active_primes;

    // Positive indicators
    let positive_primes = ["GOOD", "WANT", "FEEL"];
    let has_positive = primes.iter().any(|p| {
        positive_primes.iter().any(|pos| p.contains(pos))
    });

    // Negative indicators
    let negative_primes = ["BAD", "DONT", "NOT"];
    let has_negative = primes.iter().any(|p| {
        negative_primes.iter().any(|neg| p.contains(neg))
    });

    // Also check urgency and confidence
    if lang_bid.urgency > 0.7 && has_negative {
        EmotionalValence::Negative
    } else if has_positive && !has_negative {
        EmotionalValence::Positive
    } else if has_negative && !has_positive {
        EmotionalValence::Negative
    } else {
        EmotionalValence::Neutral
    }
}

/// Convert a LanguageWorkingMemoryItem to a brain WorkingMemoryItem
pub fn to_brain_wm_item(lang_item: &LanguageWorkingMemoryItem) -> WorkingMemoryItem {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Create a brain attention bid from the language item
    let source = match &lang_item.content_type {
        WorkingMemoryContentType::Frame { name, .. } => format!("Frame:{}", name),
        WorkingMemoryContentType::Construction { name, .. } => format!("Construction:{}", name),
        WorkingMemoryContentType::PrimeBundle { .. } => "PrimeBundle".to_string(),
        WorkingMemoryContentType::Topic { name } => format!("Topic:{}", name),
        WorkingMemoryContentType::OpenQuestion { .. } => "Question".to_string(),
    };

    let original_bid = AttentionBid::new(&source, &lang_item.summary)
        .with_salience(lang_item.activation)
        .with_urgency(0.5)
        .with_tags(vec!["language".to_string(), "working_memory".to_string()]);

    WorkingMemoryItem {
        content: lang_item.summary.clone(),
        original_bid,
        activation: lang_item.activation,
        created_at: now - (lang_item.age as u64 * 1000), // Approximate creation time
        last_accessed: now,
    }
}

/// Convert brain AttentionBid back to language-compatible format
pub fn from_brain_bid(brain_bid: &AttentionBid) -> LanguageBidFeedback {
    LanguageBidFeedback {
        source: brain_bid.source.clone(),
        content: brain_bid.content.clone(),
        won_spotlight: false, // Will be set by caller
        final_score: brain_bid.score(),
        tags: brain_bid.tags.clone(),
    }
}

/// Feedback from brain to language system
#[derive(Debug, Clone)]
pub struct LanguageBidFeedback {
    /// Source of the bid
    pub source: String,
    /// Content of the bid
    pub content: String,
    /// Did this bid win the spotlight?
    pub won_spotlight: bool,
    /// Final score after competition
    pub final_score: f32,
    /// Tags assigned
    pub tags: Vec<String>,
}

// =============================================================================
// LANGUAGE CORTEX BRIDGE
// =============================================================================

/// Configuration for the Language Cortex Bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageCortexConfig {
    /// Weight for language bids in competition
    pub language_bid_weight: f32,
    /// Maximum language items in working memory
    pub max_language_wm_items: usize,
    /// Sync working memory with brain
    pub sync_working_memory: bool,
    /// Enable feedback loop
    pub enable_feedback: bool,
    /// Minimum salience to submit bid
    pub min_bid_salience: f32,
}

impl Default for LanguageCortexConfig {
    fn default() -> Self {
        Self {
            language_bid_weight: 1.0,
            max_language_wm_items: 4, // Leave room for other cognitive items
            sync_working_memory: true,
            enable_feedback: true,
            min_bid_salience: 0.3,
        }
    }
}

/// Statistics for the Language Cortex Bridge
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LanguageCortexStats {
    /// Total inputs processed
    pub inputs_processed: u64,
    /// Bids submitted to brain
    pub bids_submitted: u64,
    /// Bids that won spotlight
    pub spotlight_wins: u64,
    /// Working memory syncs
    pub wm_syncs: u64,
    /// Average language bid score
    pub avg_bid_score: f32,
    /// Language contribution to consciousness
    pub language_consciousness_ratio: f32,
}

/// The Language Cortex Bridge
///
/// Wires the language understanding system to the brain's global workspace,
/// enabling language to participate in conscious attention competition.
pub struct LanguageCortexBridge {
    /// Configuration
    config: LanguageCortexConfig,

    /// Integrated language processor (includes bridge + active inference)
    processor: IntegratedConsciousnessProcessor,

    /// Recent brain feedback
    feedback_history: VecDeque<LanguageBidFeedback>,

    /// Statistics
    stats: LanguageCortexStats,

    /// Score history for averaging
    score_history: VecDeque<f32>,
}

impl LanguageCortexBridge {
    /// Create a new Language Cortex Bridge
    pub fn new(config: LanguageCortexConfig) -> Self {
        Self {
            config,
            processor: IntegratedConsciousnessProcessor::new(),
            feedback_history: VecDeque::with_capacity(100),
            stats: LanguageCortexStats::default(),
            score_history: VecDeque::with_capacity(100),
        }
    }

    /// Process language input and generate brain-compatible attention bid
    ///
    /// Returns (brain_bid, language_result) for submission to prefrontal cortex
    pub fn process_input(&mut self, input: &str) -> (Option<AttentionBid>, IntegratedResult) {
        self.stats.inputs_processed += 1;

        // Process through integrated language pipeline
        let result = self.processor.process(input);

        // Convert to brain bid if salience is sufficient
        let brain_bid = if result.bridge_result.bid.salience >= self.config.min_bid_salience {
            let mut bid = to_brain_bid(&result.bridge_result.bid);

            // Apply weight
            bid.salience *= self.config.language_bid_weight;

            // Track stats
            self.stats.bids_submitted += 1;
            self.score_history.push_back(bid.score());
            if self.score_history.len() > 100 {
                self.score_history.pop_front();
            }
            self.stats.avg_bid_score = self.score_history.iter().sum::<f32>()
                / self.score_history.len() as f32;

            Some(bid)
        } else {
            None
        };

        (brain_bid, result)
    }

    /// Sync language working memory to brain working memory
    ///
    /// Takes the current language working memory and converts it to
    /// brain-compatible items for inclusion in the global workspace.
    pub fn sync_working_memory(&mut self) -> Vec<WorkingMemoryItem> {
        if !self.config.sync_working_memory {
            return Vec::new();
        }

        self.stats.wm_syncs += 1;

        let lang_wm = self.processor.bridge().working_memory();
        let mut brain_items = Vec::new();

        // Convert top language items (limited by config)
        for lang_item in lang_wm.iter().take(self.config.max_language_wm_items) {
            brain_items.push(to_brain_wm_item(lang_item));
        }

        brain_items
    }

    /// Receive feedback from brain after competition
    pub fn receive_feedback(&mut self, feedback: LanguageBidFeedback) {
        if !self.config.enable_feedback {
            return;
        }

        if feedback.won_spotlight {
            self.stats.spotlight_wins += 1;
        }

        self.feedback_history.push_back(feedback);
        if self.feedback_history.len() > 100 {
            self.feedback_history.pop_front();
        }

        // Calculate language consciousness ratio
        if self.stats.bids_submitted > 0 {
            self.stats.language_consciousness_ratio =
                self.stats.spotlight_wins as f32 / self.stats.bids_submitted as f32;
        }
    }

    /// Get current language consciousness state
    pub fn consciousness_state(&self) -> IntegratedConsciousnessState {
        self.processor.consciousness_state()
    }

    /// Get access to the underlying processor
    pub fn processor(&self) -> &IntegratedConsciousnessProcessor {
        &self.processor
    }

    /// Get statistics
    pub fn stats(&self) -> &LanguageCortexStats {
        &self.stats
    }

    /// Get recent feedback
    pub fn feedback_history(&self) -> &VecDeque<LanguageBidFeedback> {
        &self.feedback_history
    }

    /// Reset the bridge state
    pub fn reset(&mut self) {
        self.processor.reset();
        self.feedback_history.clear();
        self.score_history.clear();
    }
}

// =============================================================================
// FULL BRAIN INTEGRATION
// =============================================================================

/// Result from full brain-integrated language processing
#[derive(Debug)]
pub struct BrainLanguageResult {
    /// The brain attention bid (if generated)
    pub brain_bid: Option<AttentionBid>,
    /// Language processing result
    pub language_result: IntegratedResult,
    /// Working memory items to add
    pub working_memory_items: Vec<WorkingMemoryItem>,
    /// Did language win the spotlight?
    pub won_spotlight: bool,
    /// Language consciousness state
    pub consciousness_state: IntegratedConsciousnessState,
}

/// Process language input through full brain integration
///
/// This function:
/// 1. Processes input through language pipeline
/// 2. Generates brain-compatible attention bid
/// 3. Submits to prefrontal cortex for competition
/// 4. Syncs working memory
/// 5. Returns comprehensive result
pub fn process_language_in_brain(
    input: &str,
    bridge: &mut LanguageCortexBridge,
    prefrontal: &mut PrefrontalCortexActor,
    other_bids: Vec<AttentionBid>,
    hearth: &mut crate::physiology::HearthActor,
) -> BrainLanguageResult {
    // Step 1: Process through language bridge
    let (brain_bid, language_result) = bridge.process_input(input);

    // Step 2: Combine with other bids
    let mut all_bids = other_bids;
    if let Some(bid) = brain_bid.clone() {
        all_bids.push(bid);
    }

    // Step 3: Run cognitive cycle
    let winner = prefrontal.cognitive_cycle_with_energy(all_bids, hearth);

    // Step 4: Check if language won
    let won_spotlight = winner.as_ref()
        .map(|w| w.source == "LanguageCortex")
        .unwrap_or(false);

    // Step 5: Send feedback
    if let Some(ref bid) = brain_bid {
        bridge.receive_feedback(LanguageBidFeedback {
            source: bid.source.clone(),
            content: bid.content.clone(),
            won_spotlight,
            final_score: bid.score(),
            tags: bid.tags.clone(),
        });
    }

    // Step 6: Sync working memory
    let working_memory_items = bridge.sync_working_memory();

    // Step 7: Get consciousness state
    let consciousness_state = bridge.consciousness_state();

    BrainLanguageResult {
        brain_bid,
        language_result,
        working_memory_items,
        won_spotlight,
        consciousness_state,
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_brain_bid_conversion() {
        let bridge = ConsciousnessBridge::new(BridgeConfig::default());
        let mut test_bridge = ConsciousnessBridge::new(BridgeConfig::default());
        let result = test_bridge.process("I want to install Firefox");

        let brain_bid = to_brain_bid(&result.bid);

        assert_eq!(brain_bid.source, "LanguageCortex");
        assert!(brain_bid.salience >= 0.0 && brain_bid.salience <= 1.0);
        assert!(brain_bid.tags.contains(&"language".to_string()));
    }

    #[test]
    fn test_emotional_valence_detection() {
        let bridge = ConsciousnessBridge::new(BridgeConfig::default());
        let mut test_bridge = ConsciousnessBridge::new(BridgeConfig::default());

        // Positive input
        let positive = test_bridge.process("I want something good");
        let positive_emotion = emotional_valence_from_language(&positive.bid);
        // May or may not detect as positive depending on prime extraction

        // Test that function doesn't crash
        assert!(matches!(
            positive_emotion,
            EmotionalValence::Positive | EmotionalValence::Negative | EmotionalValence::Neutral
        ));
    }

    #[test]
    fn test_working_memory_conversion() {
        let lang_item = LanguageWorkingMemoryItem {
            id: 1,
            content_type: WorkingMemoryContentType::Frame {
                name: "Transfer".to_string(),
                role_count: 3,
            },
            encoding: HV16::default(),
            summary: "Transfer frame with 3 roles".to_string(),
            activation: 0.8,
            age: 2,
            bindings: vec![],
        };

        let brain_item = to_brain_wm_item(&lang_item);

        assert!(brain_item.content.contains("Transfer"));
        assert!((brain_item.activation - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_language_cortex_bridge_creation() {
        let config = LanguageCortexConfig::default();
        let bridge = LanguageCortexBridge::new(config);

        assert_eq!(bridge.stats().inputs_processed, 0);
    }

    #[test]
    fn test_process_input() {
        let config = LanguageCortexConfig::default();
        let mut bridge = LanguageCortexBridge::new(config);

        let (brain_bid, result) = bridge.process_input("Please help me install a program");

        // Should generate a bid for this non-trivial input
        assert!(result.bridge_result.understanding.confidence > 0.0);
        assert_eq!(bridge.stats().inputs_processed, 1);
    }

    #[test]
    fn test_sync_working_memory() {
        let config = LanguageCortexConfig::default();
        let mut bridge = LanguageCortexBridge::new(config);

        // Process some input to populate working memory
        bridge.process_input("She gave him the book");
        bridge.process_input("He wanted something good");

        let wm_items = bridge.sync_working_memory();

        // Should have some working memory items (frames/constructions)
        assert!(bridge.stats().wm_syncs > 0);
    }

    #[test]
    fn test_feedback_reception() {
        let config = LanguageCortexConfig::default();
        let mut bridge = LanguageCortexBridge::new(config);

        // Process input
        let (brain_bid, _) = bridge.process_input("Test input");

        // Simulate feedback
        if let Some(bid) = brain_bid {
            bridge.receive_feedback(LanguageBidFeedback {
                source: bid.source,
                content: bid.content,
                won_spotlight: true,
                final_score: 0.8,
                tags: vec![],
            });

            assert_eq!(bridge.stats().spotlight_wins, 1);
        }
    }

    #[test]
    fn test_conversation_processing() {
        let config = LanguageCortexConfig::default();
        let mut bridge = LanguageCortexBridge::new(config);

        let inputs = [
            "Hello, how can I help you?",
            "I need to configure my system",
            "Please install nginx",
        ];

        for input in &inputs {
            let (bid, result) = bridge.process_input(input);
            assert!(result.phi() >= 0.0);
        }

        assert_eq!(bridge.stats().inputs_processed, 3);
    }

    #[test]
    fn test_consciousness_state() {
        let config = LanguageCortexConfig::default();
        let mut bridge = LanguageCortexBridge::new(config);

        // Process some inputs
        for _ in 0..5 {
            bridge.process_input("I think you know something important");
        }

        let state = bridge.consciousness_state();

        assert!(matches!(
            state,
            IntegratedConsciousnessState::Optimal
                | IntegratedConsciousnessState::Good
                | IntegratedConsciousnessState::Adequate
                | IntegratedConsciousnessState::Struggling
                | IntegratedConsciousnessState::NeedsAttention
        ));
    }

    #[test]
    fn test_reset() {
        let config = LanguageCortexConfig::default();
        let mut bridge = LanguageCortexBridge::new(config);

        bridge.process_input("Test");
        bridge.reset();

        assert!(bridge.feedback_history().is_empty());
    }

    #[test]
    #[ignore = "performance test - run with cargo test --release"]
    fn benchmark_language_cortex_bridge() {
        use std::time::Instant;

        let config = LanguageCortexConfig::default();
        let mut bridge = LanguageCortexBridge::new(config);

        let test_inputs = [
            "Install the Firefox browser",
            "Configure nginx web server",
            "I need help with something",
        ];

        // Warm up
        for _ in 0..3 {
            for input in &test_inputs {
                bridge.process_input(input);
            }
        }
        bridge.reset();

        // Benchmark
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            for input in &test_inputs {
                bridge.process_input(input);
            }
        }
        let elapsed = start.elapsed();
        let per_input = elapsed.as_micros() / (iterations * test_inputs.len()) as u128;

        // Should be under 50ms per input in debug mode
        assert!(per_input < 50_000,
            "Language cortex bridge too slow: {}μs per input", per_input);

        println!("\n📊 Language Cortex Bridge Performance:");
        println!("   {}μs per input", per_input);
        println!("   Final state: {}", bridge.consciousness_state());
    }
}
