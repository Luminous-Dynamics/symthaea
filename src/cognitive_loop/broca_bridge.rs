//! Broca language bridge: consciousness-gated thought-to-text generation.
//!
//! Wraps the Broca SSM language center with consciousness signal extraction
//! and generation management. Feature-gated by `ssm_language`.

#[cfg(feature = "ssm_language")]
use symthaea_broca::{BrocaConfig, BrocaGenerator, GenerationResult, ThoughtChannels};

#[cfg(feature = "ssm_language")]
use symthaea_core::genesis::GenesisSeed;

/// Compact consciousness signals for language generation gating.
#[derive(Debug, Clone, Copy, Default)]
pub struct BrocaConsciousnessSignals {
    /// Epistemic confidence (0=out-of-domain .. 1=certain).
    pub epistemic_confidence: f32,
    /// Emotional valence (-1=negative .. 1=positive).
    pub emotional_valence: f32,
    /// Emotional arousal (0=calm .. 1=excited).
    pub emotional_arousal: f32,
    /// Emotional warmth (0=cold .. 1=warm).
    pub emotional_warmth: f32,
    /// Consciousness level / psi (0..1).
    pub consciousness_level: f32,
    /// Meta-awareness (0..1).
    pub meta_awareness: f32,
    /// Coherence (0..1).
    pub coherence: f32,
}

// Re-export telemetry type from the types module.
pub use super::types::BrocaGenerationTelemetry;

/// Manager wrapping BrocaGenerator with consciousness gating.
#[cfg(feature = "ssm_language")]
pub struct BrocaManager {
    generator: BrocaGenerator,
    pub(crate) last_telemetry: BrocaGenerationTelemetry,
    /// Minimum consciousness level required to generate (default 0.1).
    pub(crate) consciousness_threshold: f32,
}

#[cfg(feature = "ssm_language")]
impl BrocaManager {
    /// Default checkpoint path relative to the symthaea crate root.
    const DEFAULT_CHECKPOINT: &'static str = "crates/symthaea-broca/data/broca-cfc-v2.bin";

    /// Create a new BrocaManager, optionally loading from a checkpoint.
    ///
    /// If `checkpoint_path` is `Some`, loads weights from that file.
    /// If `None`, tries the default checkpoint at `crates/symthaea-broca/data/broca-cfc-v2.bin`.
    /// If neither path exists, creates a fresh (untrained) generator.
    pub fn new(
        genesis: &GenesisSeed,
        config: BrocaConfig,
        checkpoint_path: Option<&str>,
    ) -> Self {
        let generator = Self::try_load_checkpoint(checkpoint_path, genesis)
            .unwrap_or_else(|| BrocaGenerator::new(genesis, config));

        Self {
            generator,
            last_telemetry: BrocaGenerationTelemetry::default(),
            consciousness_threshold: 0.1,
        }
    }

    /// Attempt to load a BrocaGenerator from a checkpoint file.
    ///
    /// Tries `explicit_path` first, then `DEFAULT_CHECKPOINT`. Returns `None`
    /// if no checkpoint can be loaded (missing file, corrupt data, etc.).
    fn try_load_checkpoint(
        explicit_path: Option<&str>,
        genesis: &GenesisSeed,
    ) -> Option<BrocaGenerator> {
        let paths_to_try: Vec<&str> = match explicit_path {
            Some(p) => vec![p],
            None => vec![Self::DEFAULT_CHECKPOINT],
        };

        for path in paths_to_try {
            if !std::path::Path::new(path).exists() {
                tracing::debug!("Broca checkpoint not found at {path}, skipping");
                continue;
            }
            match BrocaGenerator::from_checkpoint(path, genesis) {
                Ok((gen, _adam, _proj, _lm_config)) => {
                    tracing::info!("Loaded Broca checkpoint from {path}");
                    return Some(gen);
                }
                Err(e) => {
                    tracing::warn!("Failed to load Broca checkpoint from {path}: {e}");
                }
            }
        }

        tracing::info!("No Broca checkpoint loaded, creating fresh generator");
        None
    }

    /// Generate text from consciousness signals, gated by consciousness level.
    ///
    /// Returns `None` if consciousness is too low (below threshold).
    /// Populates `ThoughtChannels` from the provided signals and delegates
    /// to `BrocaGenerator::generate()`.
    pub fn generate(
        &mut self,
        signals: BrocaConsciousnessSignals,
    ) -> Option<GenerationResult> {
        // Gate: don't generate if consciousness too low
        if signals.consciousness_level < self.consciousness_threshold {
            self.last_telemetry = BrocaGenerationTelemetry {
                consciousness_gated: true,
                ..Default::default()
            };
            return None;
        }

        let start = std::time::Instant::now();

        // Build thought channels from consciousness signals
        let mut channels = ThoughtChannels::default();
        // Map epistemic confidence to epistemic ordinal (invert: 1.0=Certain→0, 0.0=OutOfDomain→4)
        channels.set_epistemic((1.0 - signals.epistemic_confidence) * 4.0);
        channels.set_emotion(
            signals.emotional_valence,
            signals.emotional_arousal,
            signals.emotional_warmth,
        );
        channels.set_consciousness(
            signals.consciousness_level,
            signals.meta_awareness,
            signals.coherence,
        );

        let result = self.generator.generate(&channels);

        let elapsed = start.elapsed();
        self.last_telemetry = BrocaGenerationTelemetry {
            generated: true,
            token_count: result.num_tokens,
            final_coherence: result.final_coherence,
            veto_triggered: result.veto_triggered,
            generation_time_us: elapsed.as_micros() as u64,
            consciousness_gated: false,
            ..Default::default()
        };

        Some(result)
    }

    /// Get the most recent generation telemetry.
    pub fn last_telemetry(&self) -> &BrocaGenerationTelemetry {
        &self.last_telemetry
    }

    /// Get a reference to the underlying generator.
    pub fn generator(&self) -> &BrocaGenerator {
        &self.generator
    }

    /// Get a mutable reference to the underlying generator (for training).
    pub fn generator_mut(&mut self) -> &mut BrocaGenerator {
        &mut self.generator
    }
}
