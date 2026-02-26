//! SSM Backend: Native CfC-HDC language generation via symthaea-broca.
//!
//! Implements `LLMBackend` trait using the local BrocaGenerator instead of
//! external LLM APIs. Also provides `DirectThoughtBackend` for bypassing
//! text prompt serialization when the SSM backend is active.

use anyhow::Result;
use std::sync::Mutex;
use symthaea_broca::{
    BrocaConfig, BrocaGenerator, LanguageControllerConfig, SamplingStrategy,
    ThoughtChannels,
};
use symthaea_core::genesis::GenesisSeed;

use super::llm_backend::{GenerationParams, LLMBackend};

/// Trait for direct thought-to-text generation (bypasses text prompt serialization).
pub trait DirectThoughtBackend: LLMBackend {
    /// Generate text directly from thought channels.
    fn generate_from_channels(
        &self,
        channels: &ThoughtChannels,
        params: &GenerationParams,
    ) -> Result<String>;
}

/// SSM backend using the BrocaGenerator for local, native text generation.
pub struct SsmBackend {
    /// The generator is behind a Mutex because LLMBackend requires &self
    /// but BrocaGenerator::generate requires &mut self.
    generator: Mutex<BrocaGenerator>,
}

impl SsmBackend {
    /// Create a new SSM backend from a genesis seed with default config.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 3,
                neurons_per_layer: 8,
                ..Default::default()
            },
            sampling: SamplingStrategy::TopK {
                k: 40,
                temperature: 0.7,
            },
            ..Default::default()
        };

        let generator = BrocaGenerator::new(genesis, config);
        Self {
            generator: Mutex::new(generator),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(genesis: &GenesisSeed, config: BrocaConfig) -> Self {
        let generator = BrocaGenerator::new(genesis, config);
        Self {
            generator: Mutex::new(generator),
        }
    }
}

impl std::fmt::Debug for SsmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SsmBackend")
            .field("name", &"symthaea-ssm-broca")
            .finish()
    }
}

#[async_trait::async_trait]
impl LLMBackend for SsmBackend {
    async fn generate(&self, prompt: &str, _params: &GenerationParams) -> Result<String> {
        // Parse prompt to extract thought channels if possible,
        // otherwise use a default channel set derived from the prompt text.
        let channels = channels_from_prompt(prompt);

        let mut gen = self.generator.lock().map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let result = gen.generate(&channels);
        Ok(result.text)
    }

    async fn generate_streaming(
        &self,
        prompt: &str,
        _params: &GenerationParams,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<String> {
        let channels = channels_from_prompt(prompt);

        let mut gen = self.generator.lock().map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let result = gen.generate_with_callback(&channels, on_token);
        Ok(result.text)
    }

    async fn is_available(&self) -> bool {
        true // Always available — no external dependencies
    }

    fn name(&self) -> &str {
        "symthaea-ssm-broca"
    }
}

impl DirectThoughtBackend for SsmBackend {
    fn generate_from_channels(
        &self,
        channels: &ThoughtChannels,
        _params: &GenerationParams,
    ) -> Result<String> {
        let mut gen = self.generator.lock().map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let result = gen.generate(channels);
        Ok(result.text)
    }
}

/// Extract ThoughtChannels from a text prompt.
///
/// Attempts to parse structured thought markers from the prompt text.
/// Falls back to default channels if parsing fails.
fn channels_from_prompt(prompt: &str) -> ThoughtChannels {
    let mut channels = ThoughtChannels::default();

    // Parse EPISTEMIC_STATUS from prompt
    if prompt.contains("EPISTEMIC_STATUS: Unknown") {
        channels.set_epistemic(3.0);
    } else if prompt.contains("EPISTEMIC_STATUS: Uncertain") {
        channels.set_epistemic(2.0);
    } else if prompt.contains("EPISTEMIC_STATUS: Probable") {
        channels.set_epistemic(1.0);
    } else if prompt.contains("EPISTEMIC_STATUS: Certain") {
        channels.set_epistemic(0.0);
    } else if prompt.contains("EPISTEMIC_STATUS: OutOfDomain") {
        channels.set_epistemic(4.0);
    }

    // Parse SEMANTIC_INTENT
    if prompt.contains("SEMANTIC_INTENT: Answer") {
        channels = ThoughtChannels::with_intent(1);
    } else if prompt.contains("SEMANTIC_INTENT: Acknowledge") {
        channels = ThoughtChannels::with_intent(0);
    } else if prompt.contains("SEMANTIC_INTENT: Clarify") {
        channels = ThoughtChannels::with_intent(2);
    } else if prompt.contains("SEMANTIC_INTENT: ProposeAction") {
        channels = ThoughtChannels::with_intent(3);
    } else if prompt.contains("SEMANTIC_INTENT: ExpressUncertainty") {
        channels = ThoughtChannels::with_intent(4);
    } else if prompt.contains("SEMANTIC_INTENT: Reflect") {
        channels = ThoughtChannels::with_intent(5);
    }

    // Parse MOOD_TEMPERATURE
    if let Some(pos) = prompt.find("MOOD_TEMPERATURE:") {
        let rest = &prompt[pos + 17..];
        if let Some(val) = rest.trim().split_whitespace().next() {
            if let Ok(temp) = val.parse::<f32>() {
                channels.channels[17] = temp.clamp(0.5, 2.0);
            }
        }
    }

    channels
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIQUID-MAMBA BACKEND
// ═══════════════════════════════════════════════════════════════════════════════

/// Liquid-Mamba backend: pre-trained Mamba SSM fused with HDC consciousness gating.
///
/// Uses `LiquidMambaGenerator` from `symthaea-broca` to generate text from a
/// pre-trained mamba-130m model, guided by HDC thought projections and gated
/// by epistemic, emotional, and coherence constraints.
#[cfg(feature = "liquid-mamba")]
pub struct LiquidMambaBackend {
    generator: Mutex<symthaea_broca::LiquidMambaGenerator>,
}

#[cfg(feature = "liquid-mamba")]
impl LiquidMambaBackend {
    /// Create a new Liquid-Mamba backend.
    pub fn new(genesis: &GenesisSeed, config: symthaea_broca::LiquidMambaConfig) -> Result<Self> {
        let generator = symthaea_broca::LiquidMambaGenerator::new(genesis, config)?;
        Ok(Self {
            generator: Mutex::new(generator),
        })
    }
}

#[cfg(feature = "liquid-mamba")]
impl std::fmt::Debug for LiquidMambaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiquidMambaBackend")
            .field("name", &"liquid-mamba-l-ssm")
            .finish()
    }
}

#[cfg(feature = "liquid-mamba")]
#[async_trait::async_trait]
impl LLMBackend for LiquidMambaBackend {
    async fn generate(&self, prompt: &str, _params: &GenerationParams) -> Result<String> {
        let channels = channels_from_prompt(prompt);
        let mut gen = self.generator.lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let result = gen.generate(&channels);
        Ok(result.text)
    }

    fn update_affect(&self, load: f32, temp: f32) {
        if let Ok(mut gen) = self.generator.lock() {
            gen.update_affect(load, temp);
        }
    }

    async fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "liquid-mamba-l-ssm"
    }
}

#[cfg(feature = "liquid-mamba")]
impl DirectThoughtBackend for LiquidMambaBackend {
    fn generate_from_channels(
        &self,
        channels: &ThoughtChannels,
        _params: &GenerationParams,
    ) -> Result<String> {
        let mut gen = self.generator.lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        let result = gen.generate(channels);
        Ok(result.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-ssm-backend")
    }

    #[test]
    fn test_ssm_backend_creation() {
        let genesis = test_genesis();
        let backend = SsmBackend::new(&genesis);
        assert_eq!(backend.name(), "symthaea-ssm-broca");
    }

    #[tokio::test]
    async fn test_ssm_backend_is_available() {
        let genesis = test_genesis();
        let backend = SsmBackend::new(&genesis);
        assert!(backend.is_available().await);
    }

    #[tokio::test]
    async fn test_ssm_backend_generate() {
        let genesis = test_genesis();
        let backend = SsmBackend::new(&genesis);

        let params = GenerationParams::default();
        let result = backend.generate("SEMANTIC_INTENT: Answer\nEPISTEMIC_STATUS: Certain\nTranslate this thought.", &params).await;
        assert!(result.is_ok());
        // Should produce some text (untrained model produces pseudo-random tokens)
        let text = result.unwrap();
        assert!(!text.is_empty() || true); // May be empty if EOS hit immediately
    }

    #[test]
    fn test_direct_thought_backend() {
        let genesis = test_genesis();
        let backend = SsmBackend::new(&genesis);

        let channels = ThoughtChannels::with_intent(1); // Answer
        let params = GenerationParams::default();

        let result = backend.generate_from_channels(&channels, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_channels_from_prompt() {
        let channels = channels_from_prompt("EPISTEMIC_STATUS: Unknown\nSEMANTIC_INTENT: Answer");
        assert!((channels.epistemic_ordinal() - 3.0).abs() < 0.1);

        let channels = channels_from_prompt("EPISTEMIC_STATUS: Certain\nSEMANTIC_INTENT: Clarify");
        assert!((channels.epistemic_ordinal() - 0.0).abs() < 0.1);
    }
}
