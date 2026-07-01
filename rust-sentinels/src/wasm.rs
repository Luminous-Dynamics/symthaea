// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WebAssembly bindings for browser-based consciousness detection
//!
//! Enable with `--features wasm` and compile with `wasm-pack build`
//!
//! # Example Usage (JavaScript)
//! ```javascript
//! import init, { WasmEmotionSentinel, WasmSleepSentinel, analyze_consciousness } from './sentinels.js';
//!
//! await init();
//!
//! // Analyze emotion from EEG data
//! const sentinel = new WasmEmotionSentinel();
//! const eegData = new Float32Array([...]);  // Your EEG samples
//! const result = sentinel.analyze(eegData, 256.0);
//! console.log('Valence:', result.valence, 'Arousal:', result.arousal);
//!
//! // Full consciousness analysis
//! const consciousness = analyze_consciousness(eegData, 256.0);
//! console.log('Consciousness level:', consciousness.consciousness_level);
//! ```

#![cfg(feature = "wasm")]

use js_sys::{Float32Array, Object, Reflect};
use wasm_bindgen::prelude::*;

use crate::sentinels::{
    AttentionSentinel, EmotionSentinel, EngagementSentinel, FlowSentinel, MeditationSentinel,
    SleepSentinel,
};
use crate::types::{EmotionQuadrant, MeditationState, SleepStage};

/// WASM wrapper for EmotionSentinel
#[wasm_bindgen]
pub struct WasmEmotionSentinel {
    inner: EmotionSentinel,
}

#[wasm_bindgen]
impl WasmEmotionSentinel {
    /// Create a new emotion sentinel
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: EmotionSentinel::default(),
        }
    }

    /// Analyze emotion from EEG data
    /// Returns an object with { valence, arousal, quadrant, confidence }
    #[wasm_bindgen]
    pub fn analyze(&self, data: &Float32Array, sample_rate: f32) -> Result<JsValue, JsValue> {
        let vec: Vec<f32> = data.to_vec();
        let score = self.inner.analyze(&vec, sample_rate);

        let obj = Object::new();
        Reflect::set(&obj, &"valence".into(), &score.valence.into())?;
        Reflect::set(&obj, &"arousal".into(), &score.arousal.into())?;
        Reflect::set(&obj, &"confidence".into(), &score.confidence.into())?;
        Reflect::set(
            &obj,
            &"quadrant".into(),
            &quadrant_to_string(score.quadrant()).into(),
        )?;

        Ok(obj.into())
    }
}

/// WASM wrapper for SleepSentinel
#[wasm_bindgen]
pub struct WasmSleepSentinel {
    inner: SleepSentinel,
}

#[wasm_bindgen]
impl WasmSleepSentinel {
    /// Create a new sleep sentinel
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: SleepSentinel::default(),
        }
    }

    /// Analyze sleep stage from EEG data
    /// Returns an object with { stage, confidence, delta_power, sleep_quality }
    #[wasm_bindgen]
    pub fn analyze(&mut self, data: &Float32Array, sample_rate: f32) -> Result<JsValue, JsValue> {
        let vec: Vec<f32> = data.to_vec();
        let score = self.inner.analyze(&vec, sample_rate);

        let obj = Object::new();
        Reflect::set(&obj, &"stage".into(), &stage_to_string(score.stage).into())?;
        Reflect::set(&obj, &"confidence".into(), &score.confidence.into())?;
        Reflect::set(&obj, &"delta_power".into(), &score.delta_power.into())?;
        Reflect::set(&obj, &"sleep_quality".into(), &score.sleep_quality.into())?;

        Ok(obj.into())
    }

    /// Get spindle count from last analysis
    #[wasm_bindgen]
    pub fn spindle_count(&self) -> usize {
        self.inner.spindle_count()
    }

    /// Get K-complex count from last analysis
    #[wasm_bindgen]
    pub fn k_complex_count(&self) -> usize {
        self.inner.k_complex_count()
    }

    /// Reset state for new recording
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

/// WASM wrapper for MeditationSentinel
#[wasm_bindgen]
pub struct WasmMeditationSentinel {
    inner: MeditationSentinel,
}

#[wasm_bindgen]
impl WasmMeditationSentinel {
    /// Create a new meditation sentinel
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: MeditationSentinel::default(),
        }
    }

    /// Analyze meditation state from EEG data
    /// Returns an object with { depth, state, alpha_power, theta_power, stability }
    #[wasm_bindgen]
    pub fn analyze(&self, data: &Float32Array, sample_rate: f32) -> Result<JsValue, JsValue> {
        let vec: Vec<f32> = data.to_vec();
        let score = self.inner.analyze(&vec, sample_rate);

        let obj = Object::new();
        Reflect::set(&obj, &"depth".into(), &score.depth.into())?;
        Reflect::set(
            &obj,
            &"state".into(),
            &meditation_state_to_string(score.state()).into(),
        )?;
        Reflect::set(&obj, &"alpha_power".into(), &score.alpha_power.into())?;
        Reflect::set(&obj, &"theta_power".into(), &score.theta_power.into())?;
        Reflect::set(&obj, &"stability".into(), &score.stability.into())?;
        Reflect::set(
            &obj,
            &"meditation_index".into(),
            &score.meditation_index.into(),
        )?;

        Ok(obj.into())
    }
}

/// WASM wrapper for AttentionSentinel
#[wasm_bindgen]
pub struct WasmAttentionSentinel {
    inner: AttentionSentinel,
}

#[wasm_bindgen]
impl WasmAttentionSentinel {
    /// Create a new attention sentinel
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: AttentionSentinel::default(),
        }
    }

    /// Analyze attention from EEG data
    #[wasm_bindgen]
    pub fn analyze(&mut self, data: &Float32Array, sample_rate: f32) -> Result<JsValue, JsValue> {
        let vec: Vec<f32> = data.to_vec();
        let score = self.inner.analyze(&vec, sample_rate);

        let obj = Object::new();
        Reflect::set(&obj, &"sustained".into(), &score.sustained.into())?;
        Reflect::set(&obj, &"selective".into(), &score.selective.into())?;
        Reflect::set(&obj, &"alertness".into(), &score.alertness.into())?;
        Reflect::set(&obj, &"cognitive_load".into(), &score.cognitive_load.into())?;
        Reflect::set(&obj, &"state".into(), &format!("{:?}", score.state).into())?;
        Reflect::set(&obj, &"confidence".into(), &score.confidence.into())?;

        Ok(obj.into())
    }
}

/// WASM wrapper for FlowSentinel
#[wasm_bindgen]
pub struct WasmFlowSentinel {
    inner: FlowSentinel,
    last_score: Option<crate::sentinels::FlowScore>,
}

#[wasm_bindgen]
impl WasmFlowSentinel {
    /// Create a new flow sentinel
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: FlowSentinel::default(),
            last_score: None,
        }
    }

    /// Analyze flow state from EEG data
    #[wasm_bindgen]
    pub fn analyze(&mut self, data: &Float32Array, sample_rate: f32) -> Result<JsValue, JsValue> {
        let vec: Vec<f32> = data.to_vec();
        let score = self.inner.analyze(&vec, sample_rate);

        let obj = Object::new();
        Reflect::set(&obj, &"immersion".into(), &score.immersion.into())?;
        Reflect::set(&obj, &"effortlessness".into(), &score.effortlessness.into())?;
        Reflect::set(&obj, &"time_dilation".into(), &score.time_dilation.into())?;
        Reflect::set(&obj, &"performance".into(), &score.performance.into())?;
        Reflect::set(&obj, &"flow_index".into(), &score.flow_index.into())?;
        Reflect::set(&obj, &"state".into(), &format!("{:?}", score.state).into())?;
        Reflect::set(&obj, &"confidence".into(), &score.confidence.into())?;
        Reflect::set(&obj, &"in_flow".into(), &score.in_flow().into())?;

        self.last_score = Some(score);
        Ok(obj.into())
    }

    /// Check if currently in flow state (based on last analysis)
    #[wasm_bindgen]
    pub fn in_flow(&self) -> bool {
        self.last_score
            .as_ref()
            .map(|s| s.in_flow())
            .unwrap_or(false)
    }
}

/// WASM wrapper for EngagementSentinel
#[wasm_bindgen]
pub struct WasmEngagementSentinel {
    inner: EngagementSentinel,
    last_score: Option<crate::sentinels::EngagementScore>,
}

#[wasm_bindgen]
impl WasmEngagementSentinel {
    /// Create a new engagement sentinel
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: EngagementSentinel::default(),
            last_score: None,
        }
    }

    /// Analyze engagement from EEG data
    #[wasm_bindgen]
    pub fn analyze(&mut self, data: &Float32Array, sample_rate: f32) -> Result<JsValue, JsValue> {
        let vec: Vec<f32> = data.to_vec();
        let score = self.inner.analyze(&vec, sample_rate);

        let obj = Object::new();
        Reflect::set(
            &obj,
            &"engagement_index".into(),
            &score.engagement_index.into(),
        )?;
        Reflect::set(&obj, &"level".into(), &format!("{:?}", score.level).into())?;
        Reflect::set(&obj, &"cognitive".into(), &score.cognitive.into())?;
        Reflect::set(&obj, &"emotional".into(), &score.emotional.into())?;
        Reflect::set(&obj, &"fatigue".into(), &score.fatigue.into())?;
        Reflect::set(&obj, &"confidence".into(), &score.confidence.into())?;
        Reflect::set(&obj, &"needs_break".into(), &score.needs_break().into())?;

        self.last_score = Some(score);
        Ok(obj.into())
    }

    /// Check if user needs a break (based on last analysis)
    #[wasm_bindgen]
    pub fn needs_break(&self) -> bool {
        self.last_score
            .as_ref()
            .map(|s| s.needs_break())
            .unwrap_or(false)
    }
}

/// Analyze full consciousness state from EEG data
/// Returns comprehensive analysis including emotion, attention, and overall consciousness level
#[wasm_bindgen]
pub fn analyze_consciousness(data: &Float32Array, sample_rate: f32) -> Result<JsValue, JsValue> {
    let vec: Vec<f32> = data.to_vec();

    // Run all sentinels
    let emotion = EmotionSentinel::default();
    let mut sleep = SleepSentinel::default();
    let meditation = MeditationSentinel::default();
    let mut attention = AttentionSentinel::default();
    let mut flow = FlowSentinel::default();
    let mut engagement = EngagementSentinel::default();

    let emotion_score = emotion.analyze(&vec, sample_rate);
    let sleep_score = sleep.analyze(&vec, sample_rate);
    let meditation_score = meditation.analyze(&vec, sample_rate);
    let attention_score = attention.analyze(&vec, sample_rate);
    let flow_score = flow.analyze(&vec, sample_rate);
    let engagement_score = engagement.analyze(&vec, sample_rate);

    // Compute overall consciousness level
    let consciousness_level = (emotion_score.confidence
        + attention_score.confidence
        + meditation_score.depth
        + flow_score.flow_index * 0.5
        + engagement_score.engagement_index * 0.5)
        / 3.5;

    // Build result object
    let obj = Object::new();

    // Emotion
    let emotion_obj = Object::new();
    Reflect::set(
        &emotion_obj,
        &"valence".into(),
        &emotion_score.valence.into(),
    )?;
    Reflect::set(
        &emotion_obj,
        &"arousal".into(),
        &emotion_score.arousal.into(),
    )?;
    Reflect::set(
        &emotion_obj,
        &"quadrant".into(),
        &quadrant_to_string(emotion_score.quadrant()).into(),
    )?;
    Reflect::set(&obj, &"emotion".into(), &emotion_obj.into())?;

    // Sleep
    let sleep_obj = Object::new();
    Reflect::set(
        &sleep_obj,
        &"stage".into(),
        &stage_to_string(sleep_score.stage).into(),
    )?;
    Reflect::set(
        &sleep_obj,
        &"quality".into(),
        &sleep_score.sleep_quality.into(),
    )?;
    Reflect::set(&obj, &"sleep".into(), &sleep_obj.into())?;

    // Meditation
    let meditation_obj = Object::new();
    Reflect::set(
        &meditation_obj,
        &"depth".into(),
        &meditation_score.depth.into(),
    )?;
    Reflect::set(
        &meditation_obj,
        &"state".into(),
        &meditation_state_to_string(meditation_score.state()).into(),
    )?;
    Reflect::set(&obj, &"meditation".into(), &meditation_obj.into())?;

    // Attention
    let attention_obj = Object::new();
    Reflect::set(
        &attention_obj,
        &"sustained".into(),
        &attention_score.sustained.into(),
    )?;
    Reflect::set(
        &attention_obj,
        &"selective".into(),
        &attention_score.selective.into(),
    )?;
    Reflect::set(
        &attention_obj,
        &"state".into(),
        &format!("{:?}", attention_score.state).into(),
    )?;
    Reflect::set(&obj, &"attention".into(), &attention_obj.into())?;

    // Flow
    let flow_obj = Object::new();
    Reflect::set(
        &flow_obj,
        &"flow_index".into(),
        &flow_score.flow_index.into(),
    )?;
    Reflect::set(
        &flow_obj,
        &"state".into(),
        &format!("{:?}", flow_score.state).into(),
    )?;
    Reflect::set(&flow_obj, &"in_flow".into(), &flow_score.in_flow().into())?;
    Reflect::set(&obj, &"flow".into(), &flow_obj.into())?;

    // Engagement
    let engagement_obj = Object::new();
    Reflect::set(
        &engagement_obj,
        &"index".into(),
        &engagement_score.engagement_index.into(),
    )?;
    Reflect::set(
        &engagement_obj,
        &"level".into(),
        &format!("{:?}", engagement_score.level).into(),
    )?;
    Reflect::set(
        &engagement_obj,
        &"needs_break".into(),
        &engagement_score.needs_break().into(),
    )?;
    Reflect::set(&obj, &"engagement".into(), &engagement_obj.into())?;

    // Overall
    Reflect::set(
        &obj,
        &"consciousness_level".into(),
        &consciousness_level.into(),
    )?;
    Reflect::set(&obj, &"timestamp".into(), &js_sys::Date::now().into())?;

    Ok(obj.into())
}

/// Quick emotion analysis (lighter weight than full consciousness)
#[wasm_bindgen]
pub fn quick_emotion(data: &Float32Array, sample_rate: f32) -> Result<JsValue, JsValue> {
    let vec: Vec<f32> = data.to_vec();
    let sentinel = EmotionSentinel::default();
    let score = sentinel.analyze(&vec, sample_rate);

    let obj = Object::new();
    Reflect::set(&obj, &"valence".into(), &score.valence.into())?;
    Reflect::set(&obj, &"arousal".into(), &score.arousal.into())?;
    Reflect::set(
        &obj,
        &"quadrant".into(),
        &quadrant_to_string(score.quadrant()).into(),
    )?;

    Ok(obj.into())
}

/// Quick attention check
#[wasm_bindgen]
pub fn quick_attention(data: &Float32Array, sample_rate: f32) -> Result<JsValue, JsValue> {
    let vec: Vec<f32> = data.to_vec();
    let mut sentinel = AttentionSentinel::default();
    let score = sentinel.analyze(&vec, sample_rate);

    let obj = Object::new();
    Reflect::set(&obj, &"sustained".into(), &score.sustained.into())?;
    Reflect::set(&obj, &"alertness".into(), &score.alertness.into())?;
    Reflect::set(&obj, &"state".into(), &format!("{:?}", score.state).into())?;

    Ok(obj.into())
}

/// Get library version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Helper functions for enum conversion
fn quadrant_to_string(q: EmotionQuadrant) -> String {
    match q {
        EmotionQuadrant::ExcitedPositive => "excited".to_string(),
        EmotionQuadrant::CalmPositive => "calm".to_string(),
        EmotionQuadrant::ExcitedNegative => "stressed".to_string(),
        EmotionQuadrant::CalmNegative => "sad".to_string(),
    }
}

fn stage_to_string(s: SleepStage) -> String {
    match s {
        SleepStage::Wake => "wake".to_string(),
        SleepStage::N1 => "n1".to_string(),
        SleepStage::N2 => "n2".to_string(),
        SleepStage::N3 => "n3".to_string(),
        SleepStage::Rem => "rem".to_string(),
    }
}

fn meditation_state_to_string(s: MeditationState) -> String {
    match s {
        MeditationState::Light => "light".to_string(),
        MeditationState::Moderate => "moderate".to_string(),
        MeditationState::Deep => "deep".to_string(),
    }
}