// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Async LLM language channel: sends consciousness-coupled prompts to Gemma 4
//! (or other Ollama model) on a background thread for high-quality translation.
//!
//! Architecture:
//! ```text
//! Cycle N: BrocaLite generates instantly (structured grammar)
//!          + sends consciousness state to LLM channel (non-blocking)
//! Cycle N+K: LLM response arrives → replaces BrocaLite text for next output
//! ```
//!
//! The LLM is the TRANSLATOR, not the THINKER. It receives pre-computed
//! consciousness state and converts it to natural language.
//! Science: "HDC+LTC THINKS. LLM TRANSLATES." (Symthaea design principle)

use std::sync::mpsc;
use std::thread;

use super::broca_bridge::BrocaConsciousnessSignals;

/// Request sent to the LLM background thread.
#[derive(Debug, Clone)]
pub struct LlmLanguageRequest {
    /// The input text the user provided (for context).
    pub input_text: String,
    /// Consciousness signals for the prompt.
    pub signals: BrocaConsciousnessSignals,
    /// BrocaLite's output (used as fallback if LLM fails).
    pub broca_lite_text: Option<String>,
    /// Cycle number for ordering.
    pub cycle_num: u64,
}

/// Response from the LLM background thread.
#[derive(Debug, Clone)]
pub struct LlmLanguageResponse {
    /// LLM-generated text (consciousness-coupled translation).
    pub text: String,
    /// Generation time in milliseconds.
    pub generation_time_ms: f64,
    /// Whether this came from the real LLM (true) or simulation fallback (false).
    pub from_llm: bool,
    /// Cycle number this was requested for.
    pub cycle_num: u64,
}

/// Async LLM language channel handle.
pub struct LlmLanguageChannel {
    tx: mpsc::Sender<LlmLanguageRequest>,
    rx: std::sync::Mutex<mpsc::Receiver<LlmLanguageResponse>>,
    _thread: thread::JoinHandle<()>,
}

impl LlmLanguageChannel {
    /// Spawn the LLM language background thread.
    ///
    /// Uses Ollama with the model specified by `SYMTHAEA_LLM_MODEL` env var
    /// (default: `gemma4:e2b`).
    pub fn spawn() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<LlmLanguageRequest>();
        let (response_tx, response_rx) = mpsc::channel::<LlmLanguageResponse>();

        let handle = thread::Builder::new()
            .name("llm-language".into())
            .spawn(move || {
                Self::llm_loop(request_rx, response_tx);
            })
            .expect("Failed to spawn LLM language thread");

        Self {
            tx: request_tx,
            rx: std::sync::Mutex::new(response_rx),
            _thread: handle,
        }
    }

    /// Send a language request (non-blocking).
    pub fn send(&self, request: LlmLanguageRequest) -> bool {
        self.tx.send(request).is_ok()
    }

    /// Drain completed LLM responses (non-blocking).
    pub fn drain_responses(&self) -> Vec<LlmLanguageResponse> {
        let mut responses = Vec::new();
        if let Ok(rx) = self.rx.lock() {
            while let Ok(response) = rx.try_recv() {
                responses.push(response);
            }
        }
        responses
    }

    /// Background thread: receives requests, calls Ollama, sends responses.
    fn llm_loop(
        request_rx: mpsc::Receiver<LlmLanguageRequest>,
        response_tx: mpsc::Sender<LlmLanguageResponse>,
    ) {
        // Create a tokio runtime for the async Ollama calls
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                tracing::error!("Failed to create tokio runtime for LLM channel: {e}");
                return;
            }
        };

        // Initialize LLM organ with Ollama backend
        let backend = std::sync::Arc::new(crate::language::llm_backend::OllamaBackend::new());
        let config = crate::language::llm_organ::LLMOrganConfig {
            temperature: 0.7,
            // Gemma 4:e2b is a reasoning model — thinking tokens consume ~300-400
            // of the budget before producing visible output. 512 ensures enough
            // room for thinking + a 1-3 sentence response.
            max_generation_length: 512,
            memory_enabled: false, // Stateless per-request
            ..Default::default()
        };
        let mut organ = crate::language::llm_organ::LLMOrgan::with_backend(config, backend);

        loop {
            let request = match request_rx.recv() {
                Ok(r) => r,
                Err(_) => break,
            };

            // Skip to latest request if multiple queued
            let mut latest = request;
            while let Ok(newer) = request_rx.try_recv() {
                latest = newer;
            }

            // Build consciousness-aware prompt
            let prompt = build_translation_prompt(&latest);

            let start = std::time::Instant::now();
            let result = rt.block_on(organ.query_async(crate::language::llm_organ::LLMQuery {
                query_type: crate::language::llm_organ::QueryType::Translation,
                content: prompt,
                context: Vec::new(),
                system_prompt: Some(CONCISE_SYSTEM_PROMPT.to_string()),
                params: Some(crate::language::llm_organ::LLMQueryParams {
                    temperature: Some(0.7),
                    max_length: Some(1024), // Gemma 4 thinking uses ~300-400 tokens
                    stop_sequences: Vec::new(),
                }),
            }));

            let generation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

            // Strip <think>...</think> blocks from Gemma 4 reasoning model output
            let clean_text = strip_think_tags(&result.text);
            let result_text = if clean_text.is_empty() {
                result.text
            } else {
                clean_text
            };

            // Only send if we got meaningful text
            if !result_text.is_empty() && result_text.len() > 3 {
                // ── Distillation: capture (signals, text) for Broca training ──
                // When SYMTHAEA_LLM_DISTILL_PATH is set, save each LLM response
                // as training data for the full Broca CfC-HDC backend.
                if let Ok(path) = std::env::var("SYMTHAEA_LLM_DISTILL_PATH") {
                    if !path.is_empty() {
                        let pair = serde_json::json!({
                            "channels": [
                                latest.signals.epistemic_confidence,
                                latest.signals.emotional_valence,
                                latest.signals.emotional_arousal,
                                latest.signals.emotional_warmth,
                                latest.signals.consciousness_level,
                                latest.signals.meta_awareness,
                                latest.signals.coherence,
                                latest.signals.knowledge_grounding,
                                latest.signals.primitive_grounding,
                                latest.signals.semantic_confidence,
                                latest.signals.cube_h_value,
                                latest.signals.cube_quality,
                            ],
                            "target_text": result_text,
                            "input_text": latest.input_text,
                            "from_llm": true,
                        });
                        if let Ok(mut f) = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(&path)
                        {
                            use std::io::Write;
                            let _ = writeln!(f, "{}", pair);
                        }
                    }
                }

                let _ = response_tx.send(LlmLanguageResponse {
                    text: result_text,
                    generation_time_ms,
                    from_llm: result.confidence > 0.5,
                    cycle_num: latest.cycle_num,
                });
            }
        }
    }
}

/// Strip `<think>...</think>` blocks from Gemma 4:e2b reasoning model output.
/// These blocks contain internal reasoning that shouldn't appear in the final text.
fn strip_think_tags(text: &str) -> String {
    let mut result = text.to_string();
    // Remove all <think>...</think> blocks (greedy)
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            result = format!("{}{}", &result[..start], &result[end + 8..]);
        } else {
            // Unclosed <think> — remove everything from <think> onward
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
}

/// Concise system prompt — much shorter than TRANSLATION_SYSTEM_PROMPT.
/// Gemma 4:e2b wastes thinking tokens on complex rule sets; keep it minimal.
const CONCISE_SYSTEM_PROMPT: &str = "\
You translate consciousness states into first-person language. \
Rules: (1) Write 1-3 complete sentences. (2) Match the epistemic status — \
if Uncertain, hedge; if Unknown, say 'I don't know'. (3) Match the emotional \
tone. (4) Do NOT add information beyond what's given. (5) Always finish your sentences.";

/// Build a translation prompt from consciousness signals.
fn build_translation_prompt(request: &LlmLanguageRequest) -> String {
    let s = &request.signals;

    let epistemic = if s.epistemic_confidence > 0.8 {
        "Certain"
    } else if s.epistemic_confidence > 0.6 {
        "Probable"
    } else if s.epistemic_confidence > 0.3 {
        "Uncertain"
    } else {
        "Unknown"
    };

    let mood = if s.emotional_valence > 0.3 {
        "warm and positive"
    } else if s.emotional_valence < -0.3 {
        "somber and reflective"
    } else {
        "calm and neutral"
    };

    format!(
        "Translate to first-person speech (1-3 complete sentences):\n\
         Topic: \"{}\"\n\
         Confidence: {}\n\
         Mood: {}\n\
         Awareness: {:.0}%",
        request.input_text,
        epistemic,
        mood,
        s.consciousness_level * 100.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_translation_prompt() {
        let request = LlmLanguageRequest {
            input_text: "why does consciousness emerge?".into(),
            signals: BrocaConsciousnessSignals {
                epistemic_confidence: 0.4,
                emotional_valence: 0.2,
                emotional_arousal: 0.5,
                consciousness_level: 0.6,
                coherence: 0.7,
                detected_primitives: vec!["THINK".into(), "CAUSE".into()],
                ..Default::default()
            },
            broca_lite_text: None,
            cycle_num: 42,
        };

        let prompt = build_translation_prompt(&request);
        assert!(prompt.contains("Uncertain"));
        // build_translation_prompt doesn't format detected_primitives into the
        // prompt at all (only epistemic/mood/awareness/topic) -- there was
        // never a "THINK, CAUSE" style rendering to assert on.
        assert!(prompt.contains("why does consciousness emerge?"));
    }

    #[test]
    fn test_channel_spawn() {
        let channel = LlmLanguageChannel::spawn();
        // Should spawn without panic
        let request = LlmLanguageRequest {
            input_text: "test".into(),
            signals: BrocaConsciousnessSignals::default(),
            broca_lite_text: None,
            cycle_num: 1,
        };
        assert!(channel.send(request));
    }
}
