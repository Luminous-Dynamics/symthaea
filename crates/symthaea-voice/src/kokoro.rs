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
    /// Consciousness-driven speed (set before each synthesize call).
    pub speed: Option<f32>,
    /// Consciousness-driven voice blend (interpolate between voice embeddings).
    /// (valence, arousal) → position in voice space.
    pub voice_blend: Option<(f32, f32)>,
}

impl KokoroEngine {
    /// Load the Kokoro model. Downloads from HuggingFace if not cached.
    #[cfg(feature = "kokoro")]
    pub fn load(config: KokoroConfig) -> Option<Self> {
        info!("Loading Kokoro TTS from {}...", config.repo_id);

        let api = match hf_hub::api::sync::Api::new() {
            Ok(a) => a,
            Err(e) => {
                warn!("HF API init failed: {}", e);
                return None;
            }
        };
        let repo = api.model(config.repo_id.clone());

        let model_path = match repo.get(&config.model_filename) {
            Ok(p) => {
                info!("Model at: {:?}", p);
                p
            }
            Err(e) => {
                warn!("Model download failed: {}", e);
                return None;
            }
        };
        let session = match ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(&model_path))
        {
            Ok(s) => {
                info!("ONNX session created");
                s
            }
            Err(e) => {
                warn!("ONNX session failed: {}", e);
                return None;
            }
        };

        let voices = match repo.get(&config.voices_filename) {
            Ok(path) => {
                info!("Voice file at: {:?}", path);
                let data = std::fs::read(&path).unwrap_or_default();
                let v = parse_voice_pack(&data);
                info!(
                    "Loaded {} voice(s), embed dim={}",
                    v.len(),
                    v.first().map(|x| x.len()).unwrap_or(0)
                );
                v
            }
            Err(e) => {
                warn!("Voice download failed: {}. Using zero voice.", e);
                vec![vec![0.0f32; 256]]
            }
        };

        info!("Kokoro loaded: {} voices", voices.len());
        Some(Self {
            session,
            voices,
            config,
            speed: None,
            voice_blend: None,
        })
    }

    #[cfg(not(feature = "kokoro"))]
    pub fn load(_config: KokoroConfig) -> Option<Self> {
        None
    }

    /// Synthesize text to audio at 24kHz.
    #[cfg(feature = "kokoro")]
    pub fn synthesize(&mut self, text: &str, voice_id: Option<usize>) -> Option<Vec<f32>> {
        let raw_ids = text_to_kokoro_ids(text);
        info!(
            "Kokoro synthesize: '{}' → {} tokens: {:?}",
            text,
            raw_ids.len(),
            &raw_ids
        );
        if raw_ids.is_empty() {
            warn!("No phoneme IDs!");
            return None;
        }

        // Pad with 0 at start and end (model expects this)
        let mut phoneme_ids = Vec::with_capacity(raw_ids.len() + 2);
        phoneme_ids.push(0); // start pad
        phoneme_ids.extend(&raw_ids);
        phoneme_ids.push(0); // end pad

        // Voice indexing: voices[token_count] for base voice
        let token_count = phoneme_ids.len().min(self.voices.len().saturating_sub(1));
        let base_voice = self
            .voices
            .get(token_count)
            .or_else(|| self.voices.last())
            .or_else(|| self.voices.first())?;

        // LIQUID KOKORO v2: blend voice embeddings from consciousness state
        let voice_embed = if let Some((valence, arousal)) = self.voice_blend {
            // Map (valence, arousal) to a secondary voice index
            // High valence = warmer voice (higher index), low = cooler
            // High arousal = brighter voice
            let blend_idx = ((0.5 + valence * 0.3 + arousal * 0.2) * self.voices.len() as f32)
                .clamp(0.0, self.voices.len() as f32 - 1.0) as usize;
            let blend_voice = self.voices.get(blend_idx).unwrap_or(base_voice);

            // Interpolate: 70% base (token-indexed) + 30% consciousness-blended
            let t = 0.3;
            let blended: Vec<f32> = base_voice
                .iter()
                .zip(blend_voice.iter())
                .map(|(&a, &b)| a * (1.0 - t) + b * t)
                .collect();
            info!(
                "Voice blend: base={}, blend={} (v={:.2} a={:.2})",
                token_count, blend_idx, valence, arousal
            );
            blended
        } else {
            info!("Voice index: {} (token count)", token_count);
            base_voice.clone()
        };

        let seq_len = phoneme_ids.len();
        let input_ids: Vec<i64> = phoneme_ids.iter().map(|&id| id as i64).collect();

        let ids_tensor =
            ort::value::Tensor::from_array((vec![1i64, seq_len as i64], input_ids)).ok()?;
        let embed_len = voice_embed.len();
        let style_tensor =
            ort::value::Tensor::from_array((vec![1i64, embed_len as i64], voice_embed)).ok()?;
        // LIQUID KOKORO: consciousness drives speech rate
        // High arousal/confidence = faster (1.1x), low/confused = slower (0.75x)
        let speed = self.speed.unwrap_or(0.9);
        let speed_tensor = ort::value::Tensor::from_array((vec![1i64], vec![speed])).ok()?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "style" => style_tensor,
                "speed" => speed_tensor,
            ])
            .ok()?;

        // Extract audio output with detailed logging
        if let Some(out) = outputs.get("audio") {
            if let Ok((shape, data)) = out.try_extract_tensor::<f32>() {
                info!(
                    "Kokoro output 'audio': shape={:?}, samples={}",
                    shape,
                    data.len()
                );
                if !data.is_empty() {
                    return Some(data.to_vec());
                }
            }
        }
        if let Ok((shape, data)) = outputs[0].try_extract_tensor::<f32>() {
            info!(
                "Kokoro output[0]: shape={:?}, samples={}",
                shape,
                data.len()
            );
            if !data.is_empty() {
                return Some(data.to_vec());
            }
        }
        warn!("Kokoro produced no audio");
        None
    }

    #[cfg(not(feature = "kokoro"))]
    pub fn synthesize(&mut self, _text: &str, _voice_id: Option<usize>) -> Option<Vec<f32>> {
        None
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

/// Convert text to Kokoro token IDs.
/// Uses espeak-ng for G2P (what Kokoro was trained on) with CMUdict fallback.
fn text_to_kokoro_ids(text: &str) -> Vec<u32> {
    // Try espeak-ng first (native Kokoro G2P)
    let ipa_string = if let Some(ipa) = espeak_ipa(text) {
        ipa
    } else {
        // Fallback to CMUdict if espeak-ng not available
        warn!("espeak-ng not available, falling back to CMUdict");
        let mut ipa = String::new();
        for word in text.split_whitespace() {
            let (clean, punct) = split_punctuation(word);
            if !clean.is_empty() {
                let phonemes = crate::g2p::text_to_phonemes(&clean);
                for ph in &phonemes {
                    if ph.ipa != " " {
                        ipa.push_str(ph.ipa);
                    }
                }
            }
            if let Some(p) = punct {
                ipa.push_str(p);
            }
            ipa.push(' ');
        }
        ipa.trim().to_string()
    };
    info!("IPA string: '{}'", ipa_string);

    // Tokenize: handle diphthongs first, then character-by-character
    let mut ids = Vec::new();
    let chars: Vec<char> = ipa_string.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Check for diphthongs (2-char sequences → Kokoro uppercase tokens)
        if i + 1 < chars.len() {
            let pair = format!("{}{}", chars[i], chars[i + 1]);
            if let Some(token) = diphthong_to_kokoro(&pair) {
                ids.extend(token);
                i += 2;
                continue;
            }
        }
        // Single character
        if let Some(id) = char_to_kokoro_id(chars[i]) {
            ids.push(id);
        } else {
            warn!(
                "Unknown char '{}' (U+{:04X}) — skipped",
                chars[i], chars[i] as u32
            );
        }
        i += 1;
    }

    ids
}

/// Map diphthongs to Kokoro's uppercase token pairs.
fn diphthong_to_kokoro(pair: &str) -> Option<Vec<u32>> {
    match pair {
        "aɪ" => Some(vec![24, 25]), // A + I (as in "I", "my", "light")
        "eɪ" => Some(vec![47, 25]), // e + I (as in "say", "day")
        "oʊ" => Some(vec![57, 39]), // o + W (as in "go", "slow")
        "aʊ" => Some(vec![43, 39]), // a + W (as in "how", "now")
        "ɔɪ" => Some(vec![76, 25]), // ɔ + I (as in "boy", "joy")
        _ => None,
    }
}

/// Map a single character to Kokoro token ID.
/// Based on tokenizer.json vocabulary.
fn char_to_kokoro_id(ch: char) -> Option<u32> {
    match ch {
        '$' => Some(0),
        ';' => Some(1),
        ':' => Some(2),
        ',' => Some(3),
        '.' => Some(4),
        '!' => Some(5),
        '?' => Some(6),
        ' ' => Some(16),
        'A' => Some(24),
        'I' => Some(25),
        'O' => Some(31),
        'Q' => Some(33),
        'S' => Some(35),
        'T' => Some(36),
        'W' => Some(39),
        'Y' => Some(41),
        'a' => Some(43),
        'b' => Some(44),
        'c' => Some(45),
        'd' => Some(46),
        'e' => Some(47),
        'f' => Some(48),
        'h' => Some(50),
        'i' => Some(51),
        'j' => Some(52),
        'k' => Some(53),
        'l' => Some(54),
        'm' => Some(55),
        'n' => Some(56),
        'o' => Some(57),
        'p' => Some(58),
        'q' => Some(59),
        'r' => Some(60),
        's' => Some(61),
        't' => Some(62),
        'u' => Some(63),
        'v' => Some(64),
        'w' => Some(65),
        'x' => Some(66),
        'y' => Some(67),
        'z' => Some(68),
        'ɑ' => Some(69),
        'ɐ' => Some(70),
        'ɒ' => Some(71),
        'æ' => Some(72),
        'ɔ' => Some(76),
        'ð' => Some(81),
        'ə' => Some(83),
        'ɚ' => Some(85),
        'ɛ' => Some(86),
        'ɜ' => Some(87),
        'ɡ' => Some(92),
        'ɪ' => Some(102),
        'ŋ' => Some(112),
        'ʃ' => Some(35),        // same as 'S'
        'ʌ' => Some(138),       // CORRECT: ʌ=138, NOT 63 (u)
        'ʊ' => Some(135),       // CORRECT: ʊ=135
        'ʒ' => Some(147),       // CORRECT: ʒ=147
        'θ' => Some(119),       // CORRECT: θ=119
        'ɹ' => Some(123),       // CORRECT: ɹ=123
        'ː' => Some(158),       // CORRECT: ː=158 — DO NOT SKIP (controls vowel length!)
        'ˈ' => Some(156),       // primary stress
        'ˌ' => Some(157),       // secondary stress
        'ʔ' => Some(148),       // glottal stop
        'ɾ' => Some(125),       // tap
        'ʃ' => Some(131),       // ʃ=131
        'ʧ' | 'ʨ' => Some(133), // affricate
        'ʤ' => Some(82),        // voiced affricate
        'ɚ' => Some(85),        // rhotacized schwa (espeak uses this)
        'ɐ' => Some(70),        // near-open central (espeak uses this)
        'ʲ' => Some(164),       // palatalization (espeak adds this)
        'ʰ' => Some(162),       // aspiration
        _ => None,
    }
}

/// Get IPA from espeak-ng (what Kokoro was trained on).
fn espeak_ipa(text: &str) -> Option<String> {
    let output = std::process::Command::new("espeak-ng")
        .args(["-v", "en-us", "-q", "--ipa", text])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let ipa = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if ipa.is_empty() {
        return None;
    }

    info!("espeak IPA: '{}'", ipa);
    Some(ipa)
}

fn split_punctuation(word: &str) -> (String, Option<&'static str>) {
    let trimmed = word.trim_end_matches(|c: char| ".,!?;:".contains(c));
    let punct = if trimmed.len() < word.len() {
        match word.as_bytes()[word.len() - 1] {
            b'.' => Some("."),
            b',' => Some(","),
            b'!' => Some("!"),
            b'?' => Some("?"),
            _ => None,
        }
    } else {
        None
    };
    (trimmed.to_string(), punct)
}

/// Map IPA symbol to Kokoro tokenizer IDs.
/// Vocabulary from onnx-community/Kokoro-82M-v1.0-ONNX/tokenizer.json
fn ipa_to_kokoro_tokens(ipa: &str) -> Vec<u32> {
    match ipa {
        // Vowels
        "ɑ" | "ɑː" => vec![69], // ɑ
        "æ" => vec![72],        // æ
        "ʌ" => vec![63],        // u (closest approximation)
        "ɒ" => vec![71],        // ɒ
        "ɔ" | "ɔː" => vec![76], // ɔ
        "ə" => vec![83],        // ə
        "ɜː" => vec![87],       // ɜ
        "ɛ" => vec![86],        // ɛ
        "ɪ" => vec![102],       // ɪ
        "iː" | "i" => vec![51], // i
        "ʊ" => vec![63],        // u
        "uː" | "u" => vec![63], // u
        // Diphthongs
        "eɪ" => vec![47, 102], // e + ɪ
        "aɪ" => vec![24, 25],  // A + I (Kokoro diphthong tokens)
        "oʊ" => vec![57, 63],  // o + u
        "aʊ" => vec![43, 63],  // a + u
        "ɔɪ" => vec![76, 102], // ɔ + ɪ
        // Consonants
        "b" => vec![44],
        "d" => vec![46],
        "f" => vec![48],
        "ɡ" => vec![92], // ɡ
        "h" => vec![50],
        "k" => vec![53],
        "l" => vec![54],
        "m" => vec![55],
        "n" => vec![56],
        "ŋ" => vec![112], // ŋ
        "p" => vec![58],
        "ɹ" => vec![60], // r
        "s" => vec![61],
        "t" => vec![62],
        "v" => vec![64],
        "w" => vec![65],
        "j" => vec![52], // j
        "z" => vec![68],
        "ʃ" => vec![35],  // S (Kokoro uses uppercase for IPA ʃ)
        "ʒ" => vec![68],  // z (approximation)
        "θ" => vec![36],  // T (Kokoro uses uppercase for IPA θ)
        "ð" => vec![81],  // ð
        "tʃ" => vec![20], // ʦ approximation
        "dʒ" => vec![82], // ʤ
        // Silence/space
        " " => vec![16], // space
        "" => vec![],
        // Punctuation
        "." => vec![4],
        "," => vec![3],
        "!" => vec![5],
        "?" => vec![6],
        // Compound
        "ks" => vec![53, 61], // k + s
        // Default: try character-level
        other => other
            .chars()
            .filter_map(|c| match c {
                'a'..='z' => Some(c as u32 - 'a' as u32 + 43),
                _ => None,
            })
            .collect(),
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
    (0..num_voices)
        .map(|i| {
            let start = i * bytes_per_embed;
            (0..embed_dim)
                .map(|j| {
                    let offset = start + j * 4;
                    f32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ])
                })
                .collect()
        })
        .collect()
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
