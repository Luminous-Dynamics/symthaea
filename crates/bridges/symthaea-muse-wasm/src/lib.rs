// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WASM bridge for consciousness-driven music synthesis.
//!
//! Exposes [`StreamingSynth`] to JavaScript via `wasm-bindgen`, producing
//! interleaved stereo PCM that can be consumed by an `AudioWorklet` or
//! `ScriptProcessorNode`.
//!
//! Designed to work with the AudioWorklet bridge at
//! `symthaea-muse/js/consciousness-audio-bridge.js`.
//!
//! # Quick Start (JavaScript)
//!
//! ```js
//! import init, { MuseEngine } from './pkg/symthaea_muse_wasm.js';
//! await init();
//! const engine = new MuseEngine(44100, 1024);
//! engine.set_params(0.5, 0.0, 0.5); // arousal, valence, phi
//! const pcm = engine.render_chunk(); // Float32Array [L, R, L, R, ...]
//! ```

use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::{MuseConfig, MusicalState};
use wasm_bindgen::prelude::*;

// ─── Logging helper ─────────────────────────────────────────────────────────

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

// ─── MuseEngine ─────────────────────────────────────────────────────────────

/// WASM-accessible consciousness-driven synthesis engine.
///
/// Wraps `StreamingSynth` and exposes a JSON + scalar API for state updates,
/// plus `render_chunk()` for pulling interleaved stereo PCM.
#[wasm_bindgen]
pub struct MuseEngine {
    synth: StreamingSynth,
    sample_rate: u32,
    chunk_size: usize,
    /// Cached state for `set_params` partial updates.
    current_state: MusicalState,
}

#[wasm_bindgen]
impl MuseEngine {
    /// Create a new synthesis engine.
    ///
    /// * `sample_rate` - Audio sample rate in Hz (typically 44100 or 48000).
    /// * `chunk_size`  - Number of stereo frames per `render_chunk()` call
    ///                   (e.g. 1024 for ~23 ms at 44.1 kHz).
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: u32, chunk_size: u32) -> Self {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        let config = MuseConfig::default();
        let synth = StreamingSynth::new(config, sample_rate);

        console_log!(
            "MuseEngine: sample_rate={}, chunk_size={}",
            sample_rate,
            chunk_size
        );

        Self {
            synth,
            sample_rate,
            chunk_size: chunk_size as usize,
            current_state: MusicalState::default(),
        }
    }

    /// Update consciousness state from a JSON object.
    ///
    /// Expected shape (all fields optional, missing ones keep previous value):
    /// ```json
    /// {
    ///   "arousal": 0.5,
    ///   "valence": 0.0,
    ///   "consciousness_level": 0.5,
    ///   "dopamine": 0.5,
    ///   "serotonin": 0.5,
    ///   "noradrenaline": 0.3,
    ///   "prediction_error": 0.1,
    ///   "harmony_activations": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    /// }
    /// ```
    pub fn update_state(&mut self, state_json: &str) -> Result<(), JsValue> {
        let state: MusicalState = serde_json::from_str(state_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid state JSON: {e}")))?;
        self.current_state = state.clone();
        self.synth.update_state(&state);
        Ok(())
    }

    /// Update state from individual V-A-Phi parameters (simpler API for sliders).
    ///
    /// Neuromodulators are derived from arousal/valence using psychophysiological
    /// heuristics (Schultz 1997: DA ~ reward/arousal, Jacobs & Azmitia 1992:
    /// 5-HT ~ positive valence, Berridge & Waterhouse 2003: NE ~ arousal).
    pub fn set_params(&mut self, arousal: f32, valence: f32, phi: f32) {
        let state = MusicalState {
            arousal,
            valence,
            consciousness_level: phi,
            dopamine: arousal * 0.6 + 0.2,
            serotonin: (valence + 1.0) * 0.4 + 0.1,
            noradrenaline: arousal * 0.5 + 0.1,
            prediction_error: 0.1,
            harmony_activations: self.current_state.harmony_activations,
        };
        self.current_state = state.clone();
        self.synth.update_state(&state);
    }

    /// Update a single harmony activation by index (0-7).
    pub fn set_harmony(&mut self, index: u32, value: f32) {
        if (index as usize) < 8 {
            self.current_state.harmony_activations[index as usize] = value.clamp(0.0, 1.0);
            self.synth.update_state(&self.current_state);
        }
    }

    /// Set all eight harmony activations at once from a JS Float32Array.
    pub fn set_harmonies(&mut self, values: &[f32]) {
        let n = values.len().min(8);
        self.current_state.harmony_activations[..n].copy_from_slice(&values[..n]);
        self.synth.update_state(&self.current_state);
    }

    /// Render one chunk of stereo audio.
    ///
    /// Returns interleaved stereo f32 samples `[L, R, L, R, ...]`.
    /// Length = `chunk_size * 2` (or the synth's internal chunk size if
    /// `chunk_size` is smaller than the synth minimum).
    pub fn render_chunk(&mut self) -> Vec<f32> {
        let frames = self.synth.render_chunk();
        let mut interleaved = Vec::with_capacity(frames.len() * 2);
        for [l, r] in &frames {
            interleaved.push(*l);
            interleaved.push(*r);
        }
        interleaved
    }

    /// Get current engine metrics as a JSON string.
    pub fn get_metrics(&self) -> String {
        serde_json::json!({
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "arousal": self.current_state.arousal,
            "valence": self.current_state.valence,
            "consciousness_level": self.current_state.consciousness_level,
            "dopamine": self.current_state.dopamine,
            "serotonin": self.current_state.serotonin,
            "noradrenaline": self.current_state.noradrenaline,
        })
        .to_string()
    }

    /// Get current sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get chunk size in stereo frames.
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size as u32
    }
}
