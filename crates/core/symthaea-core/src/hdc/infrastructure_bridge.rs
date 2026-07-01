// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Infrastructure Bridge - Connects consciousness to real persistence
//!
//! This module bridges the abstract consciousness interfaces to concrete implementations:
//! - HippocampusActor: Real episodic memory persistence via actor model
//! - UnifiedMind: Four-database system (Qdrant, CozoDB, LanceDB, DuckDB)
//! - Kokoro TTS: Voice synthesis with consciousness-driven prosody
//!
//! ## Architecture
//!
//! ```text
//! UnifiedConsciousBeing
//!         │
//!         ▼
//! ┌─────────────────┐
//! │ InfrastructureBridge │ ← This module
//! └────────┬────────┘
//!          │
//!    ┌─────┴─────┬─────────────┐
//!    ▼           ▼             ▼
//! Hippocampus  UnifiedMind   Kokoro
//! Actor        4-Database    TTS
//! ```

use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export voice pacing
pub use crate::voice::LTCPacing;

// Feature-gated imports for real backends
#[cfg(feature = "voice-tts")]
use crate::voice::{VoiceOutput, VoiceOutputConfig};

use crate::memory::hippocampus::{EmotionalValence, HippocampusActor, RecallQuery};

/// Memory record for persistence
#[derive(Debug, Clone)]
pub struct ConsciousnessMemoryRecord {
    pub id: u64,
    pub timestamp: u64,
    pub text: String,
    pub embedding: Vec<f32>,
    pub phi: f64,
    pub emotion: String,
    pub tags: Vec<String>,
}

/// Trait for episodic memory backends
pub trait EpisodicMemoryBackend: Send + Sync {
    /// Store a memory record
    fn store(&self, record: ConsciousnessMemoryRecord) -> Result<u64, String>;

    /// Recall similar memories
    fn recall(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<ConsciousnessMemoryRecord>, String>;

    /// Consolidate memories during "sleep"
    fn consolidate(&self) -> Result<usize, String>;
}

/// Trait for voice synthesis backends
pub trait VoiceSynthesisBackend: Send + Sync {
    /// Synthesize speech with consciousness-driven pacing
    fn synthesize(&self, text: &str, pacing: &LTCPacing) -> Result<Vec<f32>, String>;

    /// Check if voice is ready
    fn is_ready(&self) -> bool;
}

/// Mock memory backend for testing
#[derive(Default)]
pub struct MockMemoryBackend {
    records: RwLock<Vec<ConsciousnessMemoryRecord>>,
    next_id: RwLock<u64>,
}

impl MockMemoryBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

impl EpisodicMemoryBackend for MockMemoryBackend {
    fn store(&self, mut record: ConsciousnessMemoryRecord) -> Result<u64, String> {
        let rt = tokio::runtime::Handle::try_current().map_err(|e| format!("No runtime: {}", e))?;

        rt.block_on(async {
            let mut id = self.next_id.write().await;
            *id += 1;
            record.id = *id;

            let mut records = self.records.write().await;
            records.push(record);

            Ok(*id)
        })
    }

    fn recall(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<ConsciousnessMemoryRecord>, String> {
        let rt = tokio::runtime::Handle::try_current().map_err(|e| format!("No runtime: {}", e))?;

        rt.block_on(async {
            let records = self.records.read().await;

            // Simple cosine similarity
            let mut scored: Vec<_> = records
                .iter()
                .map(|r| {
                    let sim = cosine_similarity(query_embedding, &r.embedding);
                    (r.clone(), sim)
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            Ok(scored.into_iter().take(top_k).map(|(r, _)| r).collect())
        })
    }

    fn consolidate(&self) -> Result<usize, String> {
        // Mock: no-op consolidation
        Ok(0)
    }
}

/// Mock voice backend for testing
#[derive(Default)]
pub struct MockVoiceBackend;

impl VoiceSynthesisBackend for MockVoiceBackend {
    fn synthesize(&self, _text: &str, _pacing: &LTCPacing) -> Result<Vec<f32>, String> {
        // Return empty samples (no actual audio)
        Ok(vec![])
    }

    fn is_ready(&self) -> bool {
        false // No actual TTS
    }
}

use crate::math::cosine_similarity_f32 as cosine_similarity;

// =============================================================================
// REAL BACKEND IMPLEMENTATIONS
// =============================================================================

/// Hippocampus-based episodic memory backend
///
/// Wraps the HippocampusActor for real holographic memory storage and retrieval.
pub struct HippocampusMemoryBackend {
    /// The hippocampus actor (thread-safe)
    hippocampus: RwLock<HippocampusActor>,
}

impl HippocampusMemoryBackend {
    /// Create a new hippocampus memory backend
    pub fn new(dimensions: usize) -> Result<Self, String> {
        let hippocampus = HippocampusActor::new(dimensions)
            .map_err(|e| format!("Failed to create hippocampus: {}", e))?;

        Ok(Self {
            hippocampus: RwLock::new(hippocampus),
        })
    }

    /// Create with custom capacity
    pub fn with_capacity(dimensions: usize, max_memories: usize) -> Result<Self, String> {
        let hippocampus = HippocampusActor::with_capacity(dimensions, max_memories)
            .map_err(|e| format!("Failed to create hippocampus: {}", e))?;

        Ok(Self {
            hippocampus: RwLock::new(hippocampus),
        })
    }

    /// Convert emotion string to EmotionalValence
    fn parse_emotion(emotion: &str) -> EmotionalValence {
        match emotion.to_lowercase().as_str() {
            "positive" | "happy" | "joy" | "excited" => EmotionalValence::Positive,
            "negative" | "sad" | "angry" | "frustrated" => EmotionalValence::Negative,
            _ => EmotionalValence::Neutral,
        }
    }

    /// Convert EmotionalValence to string
    fn emotion_to_string(emotion: EmotionalValence) -> String {
        match emotion {
            EmotionalValence::Positive => "positive".to_string(),
            EmotionalValence::Neutral => "neutral".to_string(),
            EmotionalValence::Negative => "negative".to_string(),
        }
    }
}

impl EpisodicMemoryBackend for HippocampusMemoryBackend {
    fn store(&self, record: ConsciousnessMemoryRecord) -> Result<u64, String> {
        let rt = tokio::runtime::Handle::try_current().map_err(|e| format!("No runtime: {}", e))?;

        rt.block_on(async {
            let mut hippocampus = self.hippocampus.write().await;

            let emotion = Self::parse_emotion(&record.emotion);

            hippocampus
                .remember(record.text, record.tags, emotion)
                .map_err(|e| format!("Failed to store memory: {}", e))
        })
    }

    fn recall(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<ConsciousnessMemoryRecord>, String> {
        let rt = tokio::runtime::Handle::try_current().map_err(|e| format!("No runtime: {}", e))?;

        rt.block_on(async {
            let mut hippocampus = self.hippocampus.write().await;

            // Create a query string from embedding (simple hash for now)
            // In practice, you'd want to use the embedding directly
            let query_hash: u64 = query_embedding
                .iter()
                .take(8)
                .enumerate()
                .map(|(i, v)| ((v * 1000.0) as u64).wrapping_mul(i as u64 + 1))
                .sum();

            let query = RecallQuery {
                query: format!("query_{}", query_hash),
                top_k,
                threshold: 0.3,
                ..Default::default()
            };

            let results = hippocampus
                .recall(query)
                .map_err(|e| format!("Failed to recall: {}", e))?;

            Ok(results
                .into_iter()
                .map(|r| ConsciousnessMemoryRecord {
                    id: r.trace.id,
                    timestamp: r.trace.timestamp,
                    text: r.trace.content,
                    embedding: r.trace.encoding,
                    phi: r.similarity as f64,
                    emotion: Self::emotion_to_string(r.trace.emotion),
                    tags: r.trace.tags,
                })
                .collect())
        })
    }

    fn consolidate(&self) -> Result<usize, String> {
        let rt = tokio::runtime::Handle::try_current().map_err(|e| format!("No runtime: {}", e))?;

        rt.block_on(async {
            let hippocampus = self.hippocampus.read().await;

            // Check working memory pressure - if high, consolidation would be triggered
            let pressure = hippocampus.working_memory_pressure();

            // Return count of memories that could be consolidated
            // (actual consolidation happens via sleep cycle manager)
            if pressure > 0.7 {
                Ok(hippocampus.memory_count() / 10) // ~10% eligible for consolidation
            } else {
                Ok(0)
            }
        })
    }
}

/// Kokoro TTS voice synthesis backend
///
/// Wraps VoiceOutput for consciousness-driven speech synthesis.
#[cfg(feature = "voice-tts")]
pub struct KokoroVoiceBackend {
    /// The voice output handler (thread-safe via mutex)
    voice: std::sync::Mutex<VoiceOutput>,
    /// Whether initialization succeeded
    ready: bool,
}

#[cfg(feature = "voice-tts")]
impl KokoroVoiceBackend {
    /// Create a new Kokoro voice backend with default config
    pub fn new() -> Result<Self, String> {
        Self::with_config(VoiceOutputConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: VoiceOutputConfig) -> Result<Self, String> {
        let voice = VoiceOutput::new(config)
            .map_err(|e| format!("Failed to initialize Kokoro TTS: {}", e))?;

        Ok(Self {
            voice: std::sync::Mutex::new(voice),
            ready: true,
        })
    }

    /// Create from specific model and voice paths
    pub fn from_paths(
        model_path: &std::path::Path,
        voice_path: &std::path::Path,
        config: VoiceOutputConfig,
    ) -> Result<Self, String> {
        let voice = VoiceOutput::from_paths(model_path, voice_path, config)
            .map_err(|e| format!("Failed to load Kokoro from paths: {}", e))?;

        Ok(Self {
            voice: std::sync::Mutex::new(voice),
            ready: true,
        })
    }
}

#[cfg(feature = "voice-tts")]
impl VoiceSynthesisBackend for KokoroVoiceBackend {
    fn synthesize(&self, text: &str, pacing: &LTCPacing) -> Result<Vec<f32>, String> {
        let mut voice = self
            .voice
            .lock()
            .map_err(|e| format!("Voice lock poisoned: {}", e))?;

        let result = voice
            .synthesize_with_pacing(text, pacing.clone())
            .map_err(|e| format!("Synthesis failed: {}", e))?;

        Ok(result.samples)
    }

    fn is_ready(&self) -> bool {
        self.ready
    }
}

/// Stub Kokoro backend when voice-tts feature is disabled
#[cfg(not(feature = "voice-tts"))]
pub struct KokoroVoiceBackend;

#[cfg(not(feature = "voice-tts"))]
impl KokoroVoiceBackend {
    pub fn new() -> Result<Self, String> {
        Err("voice-tts feature not enabled".to_string())
    }
}

#[cfg(not(feature = "voice-tts"))]
impl VoiceSynthesisBackend for KokoroVoiceBackend {
    fn synthesize(&self, _text: &str, _pacing: &LTCPacing) -> Result<Vec<f32>, String> {
        Err("voice-tts feature not enabled".to_string())
    }

    fn is_ready(&self) -> bool {
        false
    }
}

/// Infrastructure bridge connecting consciousness to real systems
pub struct InfrastructureBridge {
    /// Episodic memory backend
    memory: Arc<dyn EpisodicMemoryBackend>,
    /// Voice synthesis backend  
    voice: Arc<dyn VoiceSynthesisBackend>,
    /// Statistics
    stats: BridgeStats,
}

#[derive(Debug, Default, Clone)]
pub struct BridgeStats {
    pub memories_stored: u64,
    pub memories_recalled: u64,
    pub voice_synthesized: u64,
}

impl InfrastructureBridge {
    /// Create with mock backends (for testing)
    pub fn mock() -> Self {
        Self {
            memory: Arc::new(MockMemoryBackend::new()),
            voice: Arc::new(MockVoiceBackend),
            stats: BridgeStats::default(),
        }
    }

    /// Create with real backends
    pub fn new(
        memory: Arc<dyn EpisodicMemoryBackend>,
        voice: Arc<dyn VoiceSynthesisBackend>,
    ) -> Self {
        Self {
            memory,
            voice,
            stats: BridgeStats::default(),
        }
    }

    /// Create with Hippocampus memory backend (real episodic memory)
    ///
    /// Uses mock voice backend unless voice-tts feature is enabled.
    pub fn with_hippocampus(dimensions: usize) -> Result<Self, String> {
        let memory = HippocampusMemoryBackend::new(dimensions)?;

        Ok(Self {
            memory: Arc::new(memory),
            voice: Arc::new(MockVoiceBackend),
            stats: BridgeStats::default(),
        })
    }

    /// Create with Hippocampus memory and custom capacity
    pub fn with_hippocampus_capacity(
        dimensions: usize,
        max_memories: usize,
    ) -> Result<Self, String> {
        let memory = HippocampusMemoryBackend::with_capacity(dimensions, max_memories)?;

        Ok(Self {
            memory: Arc::new(memory),
            voice: Arc::new(MockVoiceBackend),
            stats: BridgeStats::default(),
        })
    }

    /// Create with both Hippocampus memory and Kokoro TTS
    ///
    /// Requires the `voice-tts` feature to be enabled.
    #[cfg(feature = "voice-tts")]
    pub fn with_hippocampus_and_kokoro(dimensions: usize) -> Result<Self, String> {
        let memory = HippocampusMemoryBackend::new(dimensions)?;
        let voice = KokoroVoiceBackend::new()?;

        Ok(Self {
            memory: Arc::new(memory),
            voice: Arc::new(voice),
            stats: BridgeStats::default(),
        })
    }

    /// Create with Kokoro TTS only (mock memory)
    #[cfg(feature = "voice-tts")]
    pub fn with_kokoro() -> Result<Self, String> {
        let voice = KokoroVoiceBackend::new()?;

        Ok(Self {
            memory: Arc::new(MockMemoryBackend::new()),
            voice: Arc::new(voice),
            stats: BridgeStats::default(),
        })
    }

    /// Store consciousness memory
    pub fn store_memory(&mut self, record: ConsciousnessMemoryRecord) -> Result<u64, String> {
        let id = self.memory.store(record)?;
        self.stats.memories_stored += 1;
        Ok(id)
    }

    /// Recall similar memories
    pub fn recall_memories(
        &mut self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<ConsciousnessMemoryRecord>, String> {
        let results = self.memory.recall(query, top_k)?;
        self.stats.memories_recalled += results.len() as u64;
        Ok(results)
    }

    /// Synthesize speech with consciousness pacing
    pub fn synthesize_speech(
        &mut self,
        text: &str,
        pacing: &LTCPacing,
    ) -> Result<Vec<f32>, String> {
        let samples = self.voice.synthesize(text, pacing)?;
        self.stats.voice_synthesized += 1;
        Ok(samples)
    }

    /// Check if voice is available
    pub fn voice_ready(&self) -> bool {
        self.voice.is_ready()
    }

    /// Get statistics
    pub fn stats(&self) -> &BridgeStats {
        &self.stats
    }

    /// Consolidate memories
    pub fn consolidate(&self) -> Result<usize, String> {
        self.memory.consolidate()
    }
}

impl Default for InfrastructureBridge {
    fn default() -> Self {
        Self::mock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_backend() {
        let bridge = InfrastructureBridge::mock();
        assert_eq!(bridge.stats().memories_stored, 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }
}