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

/// Convert text to Kokoro token IDs via CMUdict IPA → Kokoro tokenizer.
fn text_to_kokoro_ids(text: &str) -> Vec<u32> {
    let phonemes = crate::g2p::text_to_phonemes(text);
    let mut ids = Vec::new();

    for ph in &phonemes {
        // Map our IPA phonemes to Kokoro's tokenizer vocabulary
        let token_ids = ipa_to_kokoro_tokens(ph.ipa);
        ids.extend(token_ids);
    }

    ids
}

/// Map IPA symbol to Kokoro tokenizer IDs.
/// Vocabulary from onnx-community/Kokoro-82M-v1.0-ONNX/tokenizer.json
fn ipa_to_kokoro_tokens(ipa: &str) -> Vec<u32> {
    match ipa {
        // Vowels
        "ɑ" | "ɑː" => vec![69],     // ɑ
        "æ"         => vec![72],     // æ
        "ʌ"         => vec![63],     // u (closest approximation)
        "ɒ"         => vec![71],     // ɒ
        "ɔ" | "ɔː" => vec![76],     // ɔ
        "ə"         => vec![83],     // ə
        "ɜː"       => vec![87],     // ɜ
        "ɛ"         => vec![86],     // ɛ
        "ɪ"         => vec![102],    // ɪ
        "iː" | "i"  => vec![51],    // i
        "ʊ"         => vec![63],     // u
        "uː" | "u"  => vec![63],    // u
        // Diphthongs
        "eɪ"        => vec![47, 102], // e + ɪ
        "aɪ"        => vec![24, 25],  // A + I (Kokoro diphthong tokens)
        "oʊ"        => vec![57, 63],  // o + u
        "aʊ"        => vec![43, 63],  // a + u
        "ɔɪ"        => vec![76, 102], // ɔ + ɪ
        // Consonants
        "b"         => vec![44],
        "d"         => vec![46],
        "f"         => vec![48],
        "ɡ"         => vec![92],     // ɡ
        "h"         => vec![50],
        "k"         => vec![53],
        "l"         => vec![54],
        "m"         => vec![55],
        "n"         => vec![56],
        "ŋ"         => vec![112],    // ŋ
        "p"         => vec![58],
        "ɹ"         => vec![60],     // r
        "s"         => vec![61],
        "t"         => vec![62],
        "v"         => vec![64],
        "w"         => vec![65],
        "j"         => vec![52],     // j
        "z"         => vec![68],
        "ʃ"         => vec![35],     // S (Kokoro uses uppercase for IPA ʃ)
        "ʒ"         => vec![68],     // z (approximation)
        "θ"         => vec![36],     // T (Kokoro uses uppercase for IPA θ)
        "ð"         => vec![81],     // ð
        "tʃ"        => vec![20],     // ʦ approximation
        "dʒ"        => vec![82],     // ʤ
        // Silence/space
        " "         => vec![16],     // space
        ""          => vec![],
        // Punctuation
        "."         => vec![4],
        ","         => vec![3],
        "!"         => vec![5],
        "?"         => vec![6],
        // Compound
        "ks"        => vec![53, 61], // k + s
        // Default: try character-level
        other => {
            other.chars().filter_map(|c| {
                match c {
                    'a'..='z' => Some(c as u32 - 'a' as u32 + 43),
                    _ => None,
                }
            }).collect()
        }
    }
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
