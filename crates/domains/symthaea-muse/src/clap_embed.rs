// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real CLAP audio embeddings via the `laion/clap-htsat-unfused` ONNX audio
//! tower (exported by `Xenova/clap-htsat-unfused`), for real Fréchet Audio
//! Distance in [`crate::creative_bench::FadScore`].
//!
//! Downloads `onnx/audio_model.onnx` from HuggingFace Hub on first use
//! (cached at `~/.cache/huggingface/hub/`, same pattern as
//! `symthaea-embeddings/src/qwen3/hub.rs`) and runs it via `ort`.
//!
//! The ONNX graph's only input is `input_features`
//! (`[batch, 1, N_FRAMES, 64]` f32) and its only output is `audio_embeds`
//! (`[batch, 512]` f32) — verified by inspecting the exported graph directly
//! (the `is_longer` flag that HF's Python API exposes is metadata for the
//! *fusion* variant only; the unfused audio tower's graph doesn't use it).
//!
//! # Runtime requirement: `ORT_DYLIB_PATH`
//! `ort`'s `load-dynamic` feature (used here) does not bundle ONNX Runtime —
//! it loads `libonnxruntime.so` at *runtime*, not link time, so `cargo
//! build`/`cargo check` succeed with no ONNX Runtime installed at all, but
//! any actual [`ClapEmbedder::new`]/[`ClapEmbedder::from_model_path`] call
//! panics with "cannot open shared object file" unless `ORT_DYLIB_PATH`
//! points at a real `libonnxruntime.so` (e.g. from
//! `nix shell nixpkgs#onnxruntime` or the `symthaea` flake's `gpu` devShell,
//! which already exports it).

use anyhow::Result;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::clap_mel::{self, N_FRAMES};

const AUDIO_TOWER_REPO: &str = "Xenova/clap-htsat-unfused";
const AUDIO_TOWER_FILE: &str = "onnx/audio_model.onnx";
/// The fp32 text tower (501 MB, one-time cached download). fp32 is chosen
/// deliberately over the fp16/quantized exports: audio and text embeddings
/// are compared by cosine in ONE shared space, and mixing a quantized text
/// tower with the fp32 audio tower would introduce an asymmetric drift we
/// can't audit.
const TEXT_TOWER_FILE: &str = "onnx/text_model.onnx";
const TOKENIZER_FILE: &str = "tokenizer.json";
pub const EMBEDDING_DIM: usize = 512;

/// Downloads (or reuses the cached) CLAP audio-tower ONNX model and returns
/// its local path.
pub fn ensure_audio_model() -> Result<String> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| anyhow::anyhow!("Failed to create HF Hub API: {e}"))?;
    let repo = api.model(AUDIO_TOWER_REPO.to_string());
    let path = repo
        .get(AUDIO_TOWER_FILE)
        .map_err(|e| anyhow::anyhow!("Failed to download {AUDIO_TOWER_FILE}: {e}"))?;
    Ok(path.to_string_lossy().to_string())
}

/// A loaded CLAP audio-tower ONNX session, ready to embed mono waveforms.
pub struct ClapEmbedder {
    session: Session,
}

impl ClapEmbedder {
    /// Load the CLAP audio tower from an explicit ONNX file path.
    pub fn from_model_path(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Download (or reuse the cache) and load the CLAP audio tower.
    pub fn new() -> Result<Self> {
        let model_path = ensure_audio_model()?;
        Self::from_model_path(&model_path)
    }

    /// Embed a mono waveform sampled at [`clap_mel::SAMPLE_RATE`] into a
    /// 512-d CLAP audio embedding.
    pub fn embed(&mut self, mono_waveform_48k: &[f64]) -> Result<Vec<f32>> {
        let filters = clap_mel::mel_filters();
        let mel = clap_mel::prepare_mel_input(mono_waveform_48k, filters);
        debug_assert_eq!(mel.len(), N_FRAMES);

        let mut flat = Vec::with_capacity(N_FRAMES * 64);
        for frame in &mel {
            flat.extend_from_slice(frame);
        }

        let shape = vec![1i64, 1, N_FRAMES as i64, 64];
        let input_tensor = Tensor::from_array((shape, flat))?;

        let outputs = self
            .session
            .run(ort::inputs!["input_features" => input_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}

/// Downloads (or reuses the cached) CLAP text-tower ONNX model and returns
/// its local path.
pub fn ensure_text_model() -> Result<String> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| anyhow::anyhow!("Failed to create HF Hub API: {e}"))?;
    let repo = api.model(AUDIO_TOWER_REPO.to_string());
    let path = repo
        .get(TEXT_TOWER_FILE)
        .map_err(|e| anyhow::anyhow!("Failed to download {TEXT_TOWER_FILE}: {e}"))?;
    Ok(path.to_string_lossy().to_string())
}

/// Downloads (or reuses the cached) RoBERTa tokenizer for the text tower.
pub fn ensure_tokenizer() -> Result<String> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| anyhow::anyhow!("Failed to create HF Hub API: {e}"))?;
    let repo = api.model(AUDIO_TOWER_REPO.to_string());
    let path = repo
        .get(TOKENIZER_FILE)
        .map_err(|e| anyhow::anyhow!("Failed to download {TOKENIZER_FILE}: {e}"))?;
    Ok(path.to_string_lossy().to_string())
}

/// The CLAP TEXT tower: embeds a natural-language prompt into the SAME
/// projected 512-d space as [`ClapEmbedder`]'s audio embeddings, so
/// `cosine(text, audio)` is meaningful — this is what makes "warm nostalgic
/// waltz"-style prompt steering possible (generate N candidates, rank by
/// similarity to the prompt).
pub struct ClapTextEmbedder {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
}

impl ClapTextEmbedder {
    /// Load from explicit file paths (offline/test path).
    pub fn from_paths(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        Ok(Self { session, tokenizer })
    }

    /// Download (or reuse the cache) and load the CLAP text tower.
    pub fn new() -> Result<Self> {
        let model_path = ensure_text_model()?;
        let tokenizer_path = ensure_tokenizer()?;
        Self::from_paths(&model_path, &tokenizer_path)
    }

    /// Embed a text prompt into a 512-d CLAP text embedding.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        // `true` = add special tokens; the repo's tokenizer.json carries the
        // RoBERTa post-processor, so <s>/</s> wrapping happens here.
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let seq = ids.len() as i64;
        let input_ids = Tensor::from_array((vec![1i64, seq], ids))?;
        // The exported graph's SOLE input is `input_ids` — no attention mask
        // (verified by probing the ONNX session's signature directly, same
        // method as the audio tower's `is_longer` discovery; a batch-of-one
        // unpadded sequence needs no mask).
        let outputs = self.session.run(ort::inputs!["input_ids" => input_ids])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            data.len() == EMBEDDING_DIM,
            "text tower returned {} dims, expected {EMBEDDING_DIM} — wrong output head?",
            data.len()
        );
        Ok(data.to_vec())
    }
}

/// Cosine similarity between two embeddings (e.g. a text prompt and an
/// audio render, both from the CLAP towers' shared space).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tone(duration_s: f64, seed: u64) -> Vec<f64> {
        let sr = clap_mel::SAMPLE_RATE as f64;
        let n = (duration_s * sr) as usize;
        // Deterministic tiny LCG for reproducible "noise", matching the
        // spirit of the Python fixture generator without depending on it.
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((state >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.1;
                0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
                    + noise
            })
            .collect()
    }

    #[test]
    #[ignore] // Requires network access to download the ~117MB ONNX model.
    fn embed_matches_python_onnxruntime_fixture() {
        let fixture_path = std::env::var("CLAP_ONNX_EMBED_FIXTURE").unwrap_or_else(|_| {
            "/tmp/claude-1000/-srv-luminous-dynamics/2883ee98-a7b6-4844-b220-f8d2496e7baa/scratchpad/clap_onnx_embed_fixture.json"
                .to_string()
        });
        let mel_fixture_path = std::env::var("CLAP_MEL_FIXTURE").unwrap_or_else(|_| {
            "/tmp/claude-1000/-srv-luminous-dynamics/2883ee98-a7b6-4844-b220-f8d2496e7baa/scratchpad/clap_mel_fixture.json"
                .to_string()
        });

        let embed_fixture: std::collections::HashMap<String, Vec<f32>> =
            serde_json::from_str(&std::fs::read_to_string(&fixture_path).unwrap()).unwrap();
        let mel_fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&mel_fixture_path).unwrap()).unwrap();

        let model_path = std::env::var("CLAP_AUDIO_MODEL_PATH")
            .unwrap_or_else(|_| ensure_audio_model().expect("download audio tower"));
        let mut embedder = ClapEmbedder::from_model_path(&model_path).unwrap();

        for (name, expected) in &embed_fixture {
            let waveform: Vec<f64> = mel_fixture[name]["waveform"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let got = embedder.embed(&waveform).unwrap();
            assert_eq!(got.len(), EMBEDDING_DIM);
            assert_eq!(got.len(), expected.len());
            let mae: f32 = got
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / got.len() as f32;
            assert!(
                mae < 0.01,
                "[{name}] embedding mean abs error too high: {mae}"
            );
        }
    }

    #[test]
    fn make_tone_is_deterministic() {
        let a = make_tone(1.0, 42);
        let b = make_tone(1.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn cosine_similarity_fundamentals() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        let c = [2.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
        assert!(
            (cosine_similarity(&a, &c) - 1.0).abs() < 1e-6,
            "scale-invariant"
        );
        assert!(cosine_similarity(&a, &b).abs() < 1e-6, "orthogonal → 0");
        let neg = [-1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &neg) + 1.0).abs() < 1e-6);
        assert_eq!(
            cosine_similarity(&[0.0f32; 3], &a),
            0.0,
            "zero vector guarded"
        );
    }

    /// Cross-modal sanity: the text tower must land in the audio tower's
    /// space well enough that a musically-apt prompt beats an absurd one for
    /// a real render. Network + ORT gated.
    #[test]
    #[ignore] // Downloads the ~501MB text tower + needs ORT_DYLIB_PATH.
    fn text_and_audio_towers_share_a_meaningful_space() {
        let mut audio = ClapEmbedder::new().expect("audio tower");
        let mut text = ClapTextEmbedder::new().expect("text tower");
        // A 5s render-shaped stand-in: harmonic tone (see make_tone) — not
        // real music, but tonal/harmonic vs the contrast prompt below.
        let tone = make_tone(5.0, 7);
        let audio_emb = audio.embed(&tone).unwrap();
        let apt = text.embed("a sustained harmonic musical tone").unwrap();
        let absurd = text
            .embed("a person reading a shopping list aloud")
            .unwrap();
        assert_eq!(apt.len(), EMBEDDING_DIM);
        let sim_apt = cosine_similarity(&audio_emb, &apt);
        let sim_absurd = cosine_similarity(&audio_emb, &absurd);
        assert!(
            sim_apt > sim_absurd,
            "apt prompt must outrank the absurd one: {sim_apt} vs {sim_absurd}"
        );
    }
}
