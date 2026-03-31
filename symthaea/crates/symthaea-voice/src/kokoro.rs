// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Kokoro TTS: neural text-to-speech via ONNX Runtime.
//!
//! Downloads the 82M parameter model from HuggingFace on first use.
//! Produces human-quality speech at 24kHz. Feature-gated: `kokoro`.

// Logging stubs (avoid tracing dependency)
#[cfg(feature = "kokoro")]
macro_rules! info { ($($t:tt)*) => { eprintln!("[kokoro] {}", format!($($t)*)) }; }
#[cfg(feature = "kokoro")]
macro_rules! warn { ($($t:tt)*) => { eprintln!("[kokoro WARN] {}", format!($($t)*)) }; }

/// Kokoro engine configuration.
#[derive(Debug, Clone)]
pub struct KokoroConfig {
    pub repo_id: String,
    pub model_filename: String,
    pub voices_filename: String,
    pub sample_rate: u32,
    pub default_voice: usize,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            repo_id: "onnx-community/Kokoro-82M-v1.0-ONNX".to_string(),
            model_filename: "onnx/model.onnx".to_string(),
            voices_filename: "voices/af_heart.bin".to_string(), // warm female voice
            sample_rate: 24000,
            default_voice: 0,
        }
    }
}

/// Kokoro TTS engine.
pub struct KokoroEngine {
    #[cfg(feature = "kokoro")]
    session: ort::session::Session,
    #[cfg(feature = "kokoro")]
    voices: Vec<Vec<f32>>,
    config: KokoroConfig,
}

impl KokoroEngine {
    /// Load the Kokoro model. Downloads from HuggingFace if not cached.
    #[cfg(feature = "kokoro")]
    pub fn load(config: KokoroConfig) -> Option<Self> {
        info!("Loading Kokoro TTS from {}...", config.repo_id);

        let api = match hf_hub::api::sync::Api::new() {
            Ok(a) => a,
            Err(e) => { warn!("HF API init failed: {}", e); return None; }
        };
        let repo = api.model(config.repo_id.clone());

        let model_path = match repo.get(&config.model_filename) {
            Ok(p) => { info!("Model at: {:?}", p); p },
            Err(e) => { warn!("Model download failed: {}", e); return None; }
        };
        let session = match ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(&model_path))
        {
            Ok(s) => { info!("ONNX session created"); s },
            Err(e) => { warn!("ONNX session failed: {}", e); return None; }
        };

        let voices = match repo.get(&config.voices_filename) {
            Ok(path) => {
                info!("Voice file at: {:?}", path);
                let data = std::fs::read(&path).unwrap_or_default();
                let v = parse_voice_pack(&data);
                info!("Loaded {} voice(s), embed dim={}", v.len(), v.first().map(|x| x.len()).unwrap_or(0));
                v
            }
            Err(e) => {
                warn!("Voice download failed: {}. Using zero voice.", e);
                vec![vec![0.0f32; 256]]
            }
        };

        info!("Kokoro loaded: {} voices", voices.len());
        Some(Self { session, voices, config })
    }

    #[cfg(not(feature = "kokoro"))]
    pub fn load(_config: KokoroConfig) -> Option<Self> {
        None
    }

    /// Synthesize text to audio at 24kHz.
    #[cfg(feature = "kokoro")]
    pub fn synthesize(&mut self, text: &str, voice_id: Option<usize>) -> Option<Vec<f32>> {
        let phoneme_ids = text_to_kokoro_ids(text);
        info!("Kokoro synthesize: '{}' → {} phoneme IDs", text, phoneme_ids.len());
        if phoneme_ids.is_empty() { warn!("No phoneme IDs!"); return None; }

        let voice_idx = voice_id.unwrap_or(self.config.default_voice);
        let voice_embed = self.voices.get(voice_idx).or(self.voices.first())?;

        let seq_len = phoneme_ids.len();
        let input_ids: Vec<i64> = phoneme_ids.iter().map(|&id| id as i64).collect();

        let ids_tensor = ort::value::Tensor::from_array(
            (vec![1i64, seq_len as i64], input_ids)
        ).ok()?;
        let style_tensor = ort::value::Tensor::from_array(
            (vec![1i64, voice_embed.len() as i64], voice_embed.clone())
        ).ok()?;
        let speed_tensor = ort::value::Tensor::from_array(
            (vec![1i64], vec![1.0f32])
        ).ok()?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids_tensor,
            "style" => style_tensor,
            "speed" => speed_tensor,
        ]).ok()?;

        // Try "audio" output, then first output
        if let Some(out) = outputs.get("audio") {
            if let Ok((_shape, data)) = out.try_extract_tensor::<f32>() {
                if !data.is_empty() { return Some(data.to_vec()); }
            }
        }
        if let Ok((_shape, data)) = outputs[0].try_extract_tensor::<f32>() {
            if !data.is_empty() { return Some(data.to_vec()); }
        }
        None
    }

    #[cfg(not(feature = "kokoro"))]
    pub fn synthesize(&mut self, _text: &str, _voice_id: Option<usize>) -> Option<Vec<f32>> {
        None
    }

    pub fn sample_rate(&self) -> u32 { self.config.sample_rate }
}

/// Convert text to Kokoro-compatible phoneme IDs (Misaki format).
/// Uses a simple character-level mapping for now.
fn text_to_kokoro_ids(text: &str) -> Vec<u32> {
    // Misaki phoneme vocabulary (simplified subset)
    let mut ids = Vec::new();
    for ch in text.to_lowercase().chars() {
        let id = match ch {
            'a' => 16, 'b' => 18, 'c' => 22, 'd' => 19, 'e' => 4,
            'f' => 23, 'g' => 24, 'h' => 25, 'i' => 1, 'j' => 33,
            'k' => 22, 'l' => 27, 'm' => 28, 'n' => 29, 'o' => 11,
            'p' => 31, 'q' => 22, 'r' => 26, 's' => 32, 't' => 20,
            'u' => 7, 'v' => 34, 'w' => 35, 'x' => 32, 'y' => 2,
            'z' => 38, ' ' => 14, '.' => 42, ',' => 41, '!' => 43,
            '?' => 44, _ => continue,
        };
        ids.push(id);
    }
    ids
}

#[cfg(feature = "kokoro")]
fn parse_voice_pack(data: &[u8]) -> Vec<Vec<f32>> {
    let embed_dim = 256;
    let bytes_per_embed = embed_dim * 4;
    if data.len() < bytes_per_embed {
        return vec![vec![0.0f32; embed_dim]];
    }
    let num_voices = data.len() / bytes_per_embed;
    (0..num_voices).map(|i| {
        let start = i * bytes_per_embed;
        (0..embed_dim).map(|j| {
            let offset = start + j * 4;
            f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]])
        }).collect()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kokoro_ids_from_text() {
        let ids = text_to_kokoro_ids("hello");
        assert!(!ids.is_empty());
        assert!(ids.len() == 5);
    }

    #[test]
    fn config_default() {
        let cfg = KokoroConfig::default();
        assert_eq!(cfg.sample_rate, 24000);
    }
}
