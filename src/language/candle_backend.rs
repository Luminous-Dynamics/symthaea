// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama as qlm;
use hf_hub::{Repo, api::sync::Api};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use super::llm_backend::{GenerationParams, LLMBackend};

/// Affective Logits Processor that warps vocabulary based on physics.
pub struct AffectiveLogitsProcessor {
    inner: LogitsProcessor,
    repetition_penalty: f32,
}

impl AffectiveLogitsProcessor {
    pub fn new(seed: u64, temperature: f32, repetition_penalty: f32, top_k: Option<usize>) -> Self {
        // Build the appropriate sampling strategy with top-k baked in
        let sampling = match top_k {
            Some(k) => Sampling::TopK {
                k,
                temperature: temperature as f64,
            },
            None => Sampling::All {
                temperature: temperature as f64,
            },
        };
        Self {
            inner: LogitsProcessor::from_sampling(seed, sampling),
            repetition_penalty,
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        // Top-K is now handled internally by the LogitsProcessor sampling strategy
        let token = self.inner.sample(logits)?;
        Ok(token)
    }
}

/// In-process Candle backend for Llama-3-GGUF (quantized).
pub struct CandleBackend {
    model: Arc<Mutex<qlm::ModelWeights>>,
    tokenizer: Tokenizer,
    device: Device,
    /// Shared affective state
    thermodynamic_load: Arc<std::sync::atomic::AtomicU32>, // bits of f32
    mood_temperature: Arc<std::sync::atomic::AtomicU32>,
    /// Currently active LoRA adapter ID
    current_lora: Arc<Mutex<Option<String>>>,
}

impl CandleBackend {
    pub fn new(model_id: &str, _revision: &str, file_name: &str) -> Result<Self> {
        let device = Device::Cpu; // Default to CPU for 6W edge logic, can be optimized later
        let api = Api::new()?;
        let repo = api.repo(Repo::model(model_id.to_string()));
        let model_path = repo.get(file_name)?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;

        // Load GGUF model via quantized_llama ModelWeights
        let mut file = std::fs::File::open(&model_path)?;
        let model_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        let model = qlm::ModelWeights::from_gguf(model_content, &mut file, &device)?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer,
            device,
            thermodynamic_load: Arc::new(std::sync::atomic::AtomicU32::new(0.0f32.to_bits())),
            mood_temperature: Arc::new(std::sync::atomic::AtomicU32::new(1.0f32.to_bits())),
            current_lora: Arc::new(Mutex::new(None)),
        })
    }

    pub fn set_affect(&self, load: f32, temp: f32) {
        self.thermodynamic_load
            .store(load.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.mood_temperature
            .store(temp.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    fn get_affect(&self) -> (f32, f32) {
        let load = f32::from_bits(
            self.thermodynamic_load
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        let temp = f32::from_bits(
            self.mood_temperature
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        (load, temp)
    }
}

#[async_trait::async_trait]
impl LLMBackend for CandleBackend {
    fn apply_lora(&self, lora_id: &str, _delta: &[u8]) -> Result<()> {
        // In a full implementation, we'd use candle-nn::lora to wrap layers
        // For RC1, we track the ID to simulate the dialect shift
        tracing::info!(id = %lora_id, "Applying linguistic LoRA adapter (Collective Dialect)");
        let mut current = self.current_lora.blocking_lock();
        *current = Some(lora_id.to_string());
        Ok(())
    }

    async fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec();
        let (load, mood_temp) = self.get_affect();

        // 1. Affective Logit Warping
        // High load -> Low repetition penalty (stuttering), Small Top-K (limited vocabulary)
        // Rested -> High repetition penalty (diverse), Large Top-K
        let rep_penalty = 1.2 - (load * 0.2); // 1.2 (rested) to 1.0 (exhausted)
        let top_k = if load > 0.8 { Some(10) } else { Some(50) };
        let temperature = mood_temp.clamp(0.1, 2.0);

        let mut logits_processor =
            AffectiveLogitsProcessor::new(42, temperature, rep_penalty, top_k);
        let mut generated_text = String::new();

        let mut model = self.model.lock().await;

        for _i in 0..params.max_tokens {
            let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
            // quantized_llama::ModelWeights::forward takes (x, index_pos)
            let logits = model.forward(&input, tokens.len())?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);

            let token_text = self
                .tokenizer
                .decode(&[next_token], true)
                .map_err(|e| anyhow!(e))?;
            generated_text.push_str(&token_text);

            if token_text.contains("<|end_of_text|>") || token_text.contains("</s>") {
                break;
            }

            // 3. Interleaved Zero-Idle Attention: Yield to tokio after every token
            tokio::task::yield_now().await;
        }

        Ok(generated_text)
    }

    async fn is_available(&self) -> bool {
        true // In-process is always available once loaded
    }

    fn name(&self) -> &str {
        "Candle (In-Process)"
    }

    fn update_affect(&self, load: f32, temp: f32) {
        self.set_affect(load, temp);
    }
}
