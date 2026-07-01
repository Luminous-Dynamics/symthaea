// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sentinels - Consciousness State Detection Library
//!
//! This library provides biosignal analysis for detecting consciousness states
//! from EEG and other physiological signals.
//!
//! ## The Consciousness Trilogy (Validated)
//!
//! - **EmotionSentinel**: Detects emotional valence and arousal (Proof of Joy)
//! - **SleepSentinel**: Classifies sleep stages (Proof of Rest)
//! - **MeditationSentinel**: Measures meditation depth (Proof of Focus)
//!
//! ## Extended Proofs
//!
//! - **AttentionSentinel**: Detects sustained/selective attention (Proof of Attention)
//! - **FlowSentinel**: Identifies optimal performance states (Proof of Flow)
//! - **EngagementSentinel**: Measures cognitive/emotional engagement (Proof of Engagement)
//!
//! ## Quick Start
//!
//! ```rust
//! use sentinels::{analyze_consciousness, AnalysisMode};
//!
//! // Generate or load EEG data (30 seconds at 256 Hz)
//! let data: Vec<f32> = vec![0.0; 256 * 30];
//!
//! // Analyze consciousness state
//! let poc = analyze_consciousness(&data, 256.0, AnalysisMode::Auto).unwrap();
//!
//! println!("State: {:?}", poc.state);
//! println!("Consciousness Level: {:.2}", poc.consciousness_level);
//! println!("Wellbeing Score: {:.2}", poc.wellbeing_score);
//! ```
//!
//! ## Features
//!
//! - `python`: Enable Python bindings via PyO3
//! - `wasm`: Enable WebAssembly bindings

pub mod error;
pub mod hardware;
pub mod sentinels;
pub mod signal;
pub mod types;

// Optional modules
#[cfg(feature = "python")]
mod python;

#[cfg(feature = "wasm")]
mod wasm;

pub use error::SentinelError;
pub use types::*;

// Re-export hardware abstraction
pub use hardware::{
    DeviceConfig, DeviceError, DeviceInfo, EegDevice, EegSample, MuseAdapter, MuseConfig,
    MuseModel, OpenBciAdapter, OpenBciBoard, OpenBciConfig,
};

// Re-export all sentinels and their types
pub use sentinels::{
    AttentionConfig,
    AttentionScore,
    // Extended Proofs
    AttentionSentinel,
    AttentionState,
    EmotionQuadrant,
    EmotionScore,
    // Trilogy
    EmotionSentinel,
    EngagementConfig,
    EngagementLevel,
    EngagementScore,
    EngagementSentinel,
    FlowConfig,
    FlowScore,
    FlowSentinel,
    FlowState,
    KComplexEvent,
    MeditationScore,
    MeditationSentinel,
    MeditationState,
    SleepConfig,
    SleepScore,
    SleepSentinel,
    SleepStage,
    SpectralRatios,
    SpindleEvent,
};

/// Analysis mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisMode {
    /// Run all six Sentinels (Trilogy + Extended)
    Auto,
    /// Run only the Consciousness Trilogy
    Trilogy,
    /// Emotion analysis only
    Emotion,
    /// Sleep staging only
    Sleep,
    /// Meditation analysis only
    Meditation,
    /// Attention analysis only
    Attention,
    /// Flow state analysis only
    Flow,
    /// Engagement analysis only
    Engagement,
    /// Extended Proofs only (Attention + Flow + Engagement)
    Extended,
}

/// Extended Proof of Consciousness result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtendedPoC {
    /// Timestamp of analysis
    pub timestamp: f64,
    /// Primary consciousness state
    pub state: ConsciousnessState,
    /// Overall consciousness level (0-1)
    pub consciousness_level: f32,
    /// Overall wellbeing score (0-1)
    pub wellbeing_score: f32,
    /// Performance potential (0-1)
    pub performance_potential: f32,
    /// Learning readiness (0-1)
    pub learning_readiness: f32,

    // Trilogy
    pub emotion: Option<EmotionScore>,
    pub sleep: Option<SleepScore>,
    pub meditation: Option<MeditationScore>,

    // Extended Proofs
    pub attention: Option<AttentionScore>,
    pub flow: Option<FlowScore>,
    pub engagement: Option<EngagementScore>,
}

impl ExtendedPoC {
    /// Serialize to JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Main entry point for consciousness analysis
///
/// # Arguments
///
/// * `data` - EEG samples (single channel)
/// * `sample_rate` - Sampling frequency in Hz
/// * `mode` - Which Sentinels to run
///
/// # Returns
///
/// `ProofOfConsciousness` containing all analysis results
///
/// # Errors
///
/// Returns `SentinelError` if analysis fails
pub fn analyze_consciousness(
    data: &[f32],
    sample_rate: f32,
    mode: AnalysisMode,
) -> Result<ProofOfConsciousness, SentinelError> {
    if data.len() < (sample_rate * 2.0) as usize {
        return Err(SentinelError::InsufficientData {
            required: (sample_rate * 2.0) as usize,
            provided: data.len(),
        });
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let run_trilogy = matches!(mode, AnalysisMode::Auto | AnalysisMode::Trilogy);

    let emotion = if run_trilogy || matches!(mode, AnalysisMode::Emotion) {
        Some(EmotionSentinel::default().analyze(data, sample_rate))
    } else {
        None
    };

    let sleep = if run_trilogy || matches!(mode, AnalysisMode::Sleep) {
        Some(SleepSentinel::default().analyze(data, sample_rate))
    } else {
        None
    };

    let meditation = if run_trilogy || matches!(mode, AnalysisMode::Meditation) {
        Some(MeditationSentinel::default().analyze(data, sample_rate))
    } else {
        None
    };

    let (state, consciousness_level, wellbeing_score) =
        compute_overall_state(&emotion, &sleep, &meditation);

    Ok(ProofOfConsciousness {
        timestamp,
        state,
        consciousness_level,
        wellbeing_score,
        emotion,
        sleep,
        meditation,
    })
}

/// Analyze with Extended Proofs (all six Sentinels)
///
/// Returns an `ExtendedPoC` with attention, flow, and engagement scores.
pub fn analyze_extended(
    data: &[f32],
    sample_rate: f32,
    mode: AnalysisMode,
) -> Result<ExtendedPoC, SentinelError> {
    if data.len() < (sample_rate * 2.0) as usize {
        return Err(SentinelError::InsufficientData {
            required: (sample_rate * 2.0) as usize,
            provided: data.len(),
        });
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let run_trilogy = matches!(mode, AnalysisMode::Auto | AnalysisMode::Trilogy);
    let run_extended = matches!(mode, AnalysisMode::Auto | AnalysisMode::Extended);

    // Trilogy
    let emotion = if run_trilogy || matches!(mode, AnalysisMode::Emotion) {
        Some(EmotionSentinel::default().analyze(data, sample_rate))
    } else {
        None
    };

    let sleep = if run_trilogy || matches!(mode, AnalysisMode::Sleep) {
        Some(SleepSentinel::default().analyze(data, sample_rate))
    } else {
        None
    };

    let meditation = if run_trilogy || matches!(mode, AnalysisMode::Meditation) {
        Some(MeditationSentinel::default().analyze(data, sample_rate))
    } else {
        None
    };

    // Extended Proofs
    let attention = if run_extended || matches!(mode, AnalysisMode::Attention) {
        Some(AttentionSentinel::default().analyze(data, sample_rate))
    } else {
        None
    };

    let flow = if run_extended || matches!(mode, AnalysisMode::Flow) {
        Some(FlowSentinel::default().analyze(data, sample_rate))
    } else {
        None
    };

    let engagement = if run_extended || matches!(mode, AnalysisMode::Engagement) {
        Some(EngagementSentinel::default().analyze(data, sample_rate))
    } else {
        None
    };

    // Compute overall state with extended info
    let (state, consciousness_level, wellbeing_score) = compute_extended_state(
        &emotion,
        &sleep,
        &meditation,
        &attention,
        &flow,
        &engagement,
    );

    // Calculate performance potential (from attention + flow)
    let performance_potential = match (&attention, &flow) {
        (Some(a), Some(f)) => (a.attention_index / 3.0 * 0.5 + f.flow_index * 0.5).min(1.0),
        (Some(a), None) => a.attention_index / 3.0,
        (None, Some(f)) => f.flow_index,
        _ => 0.5,
    };

    // Calculate learning readiness (from attention + engagement)
    let learning_readiness = match (&attention, &engagement) {
        (Some(a), Some(e)) => {
            let base = (a.attention_index / 3.0 * 0.4 + e.engagement_index / 3.0 * 0.4).min(1.0);
            if e.fatigue > 0.5 {
                base * 0.5
            } else {
                base + (1.0 - e.fatigue) * 0.2
            }
        }
        (Some(a), None) => a.attention_index / 3.0,
        (None, Some(e)) => e.engagement_index / 3.0,
        _ => 0.5,
    };

    Ok(ExtendedPoC {
        timestamp,
        state,
        consciousness_level,
        wellbeing_score,
        performance_potential,
        learning_readiness,
        emotion,
        sleep,
        meditation,
        attention,
        flow,
        engagement,
    })
}

fn compute_overall_state(
    emotion: &Option<EmotionScore>,
    sleep: &Option<SleepScore>,
    meditation: &Option<MeditationScore>,
) -> (ConsciousnessState, f32, f32) {
    let mut state = ConsciousnessState::Relaxed;
    let mut consciousness_level = 0.5;
    let mut wellbeing = 0.5;

    // Sleep takes priority
    if let Some(s) = sleep {
        if s.stage != SleepStage::Wake {
            match s.stage {
                SleepStage::N3 => {
                    state = ConsciousnessState::DeepSleep;
                    consciousness_level = 0.1;
                    wellbeing = 0.6 + 0.4 * s.sleep_quality;
                }
                SleepStage::Rem => {
                    state = ConsciousnessState::Rem;
                    consciousness_level = 0.4;
                    wellbeing = 0.5;
                }
                _ => {
                    state = ConsciousnessState::LightSleep;
                    consciousness_level = 0.3;
                    wellbeing = 0.4 + 0.2 * s.sleep_quality;
                }
            }
            return (state, consciousness_level, wellbeing);
        }
    }

    // Awake states
    consciousness_level = 0.7;

    if let Some(m) = meditation {
        if m.depth > 0.6 {
            state = ConsciousnessState::Meditative;
            consciousness_level = 0.8 + 0.2 * m.depth;
            wellbeing = 0.7 + 0.3 * m.stability;
        } else if m.depth > 0.4 {
            state = ConsciousnessState::Focused;
            consciousness_level = 0.75;
            wellbeing = 0.65;
        }
    }

    if let Some(e) = emotion {
        wellbeing = 0.5 + 0.3 * e.valence + 0.2 * (1.0 - e.arousal);
        wellbeing = wellbeing.clamp(0.0, 1.0);

        if e.arousal > 0.6 && e.valence < -0.3 {
            state = ConsciousnessState::Stressed;
            consciousness_level = 0.9;
            wellbeing = 0.3;
        }

        if e.arousal < 0.3 && state != ConsciousnessState::Meditative {
            state = ConsciousnessState::Relaxed;
            consciousness_level = 0.6;
        }
    }

    // Flow state detection
    if consciousness_level > 0.8 && wellbeing > 0.7 {
        if matches!(
            state,
            ConsciousnessState::Focused | ConsciousnessState::Meditative
        ) {
            state = ConsciousnessState::Flow;
        }
    }

    (state, consciousness_level, wellbeing)
}

fn compute_extended_state(
    emotion: &Option<EmotionScore>,
    sleep: &Option<SleepScore>,
    meditation: &Option<MeditationScore>,
    attention: &Option<AttentionScore>,
    flow: &Option<FlowScore>,
    engagement: &Option<EngagementScore>,
) -> (ConsciousnessState, f32, f32) {
    // Start with base computation
    let (mut state, mut consciousness_level, mut wellbeing) =
        compute_overall_state(emotion, sleep, meditation);

    // If awake, use Extended Proofs to refine
    if !matches!(
        state,
        ConsciousnessState::DeepSleep | ConsciousnessState::LightSleep | ConsciousnessState::Rem
    ) {
        // Flow state from FlowSentinel
        if let Some(f) = flow {
            if f.flow_index > 0.6 {
                state = ConsciousnessState::Flow;
                consciousness_level = 0.9 + 0.1 * f.flow_index;
                wellbeing = wellbeing.max(0.8);
            }
        }

        // Focused state from AttentionSentinel
        if let Some(a) = attention {
            if a.attention_index > 2.0 && state != ConsciousnessState::Flow {
                state = ConsciousnessState::Focused;
                consciousness_level = consciousness_level.max(0.8);
            }
        }

        // Adjust wellbeing based on engagement
        if let Some(e) = engagement {
            if e.level == EngagementLevel::High {
                wellbeing = wellbeing.max(0.7);
            } else if e.level == EngagementLevel::Overload {
                wellbeing = wellbeing * 0.7;
                state = ConsciousnessState::Stressed;
            }
        }
    }

    (state, consciousness_level.min(1.0), wellbeing.min(1.0))
}

// Full Python bindings in external file
#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_analysis() {
        let data: Vec<f32> = (0..256 * 30)
            .map(|i| {
                let t = i as f32 / 256.0;
                (10.0 * t * std::f32::consts::TAU).sin() * 0.5
            })
            .collect();

        let result = analyze_consciousness(&data, 256.0, AnalysisMode::Auto);
        assert!(result.is_ok());

        let poc = result.unwrap();
        assert!(poc.consciousness_level >= 0.0 && poc.consciousness_level <= 1.0);
        assert!(poc.wellbeing_score >= 0.0 && poc.wellbeing_score <= 1.0);
    }

    #[test]
    fn test_extended_analysis() {
        let data: Vec<f32> = (0..256 * 30)
            .map(|i| {
                let t = i as f32 / 256.0;
                (10.0 * t * std::f32::consts::TAU).sin() * 0.5
            })
            .collect();

        let result = analyze_extended(&data, 256.0, AnalysisMode::Auto);
        assert!(result.is_ok());

        let poc = result.unwrap();
        assert!(poc.attention.is_some());
        assert!(poc.flow.is_some());
        assert!(poc.engagement.is_some());
        assert!(poc.performance_potential >= 0.0 && poc.performance_potential <= 1.0);
        assert!(poc.learning_readiness >= 0.0 && poc.learning_readiness <= 1.0);
    }

    #[test]
    fn test_insufficient_data() {
        let data: Vec<f32> = vec![0.0; 100];
        let result = analyze_consciousness(&data, 256.0, AnalysisMode::Auto);
        assert!(matches!(
            result,
            Err(SentinelError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_attention_only() {
        let data: Vec<f32> = (0..256 * 5)
            .map(|i| {
                let t = i as f32 / 256.0;
                (6.0 * t * std::f32::consts::TAU).sin() * 0.5 // Theta
            })
            .collect();

        let result = analyze_extended(&data, 256.0, AnalysisMode::Attention);
        assert!(result.is_ok());
        let poc = result.unwrap();
        assert!(poc.attention.is_some());
        assert!(poc.emotion.is_none()); // Only attention was requested
    }
}