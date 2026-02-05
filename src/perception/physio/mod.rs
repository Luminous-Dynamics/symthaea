//! Physiological Signal Processing
//!
//! This module provides tools for processing physiological signals:
//! - EDF file parsing (European Data Format)
//! - Sleep stage classification (Project Hypnos)
//! - Emotion detection from multi-channel EEG (Project Pathos)
//! - Meditation/Flow state detection (Project Nous)
//! - Consciousness state detection
//!
//! ## The Consciousness Trilogy
//!
//! ### Project Hypnos (Sleep/Unconscious)
//! The SleepSentinel uses LTC dynamics to detect consciousness states
//! from EEG signals. Protocol: "Proof of Rest"
//!
//! ### Project Pathos (Emotion/Subconscious)
//! The EmotionSentinel uses Frontal Asymmetry and Beta/Alpha ratios
//! to detect Valence/Arousal. Protocol: "Proof of Joy"
//!
//! ### Project Nous (Meditation/Superconscious)
//! The MeditationSentinel uses Frontal Midline Theta, Alpha Coherence,
//! and Gamma Synchrony to detect Flow/Focus states. Protocol: "Proof of Focus"
//!
//! All three Sentinels are now active and using the unified_ltc module.

pub mod edf_loader;
pub mod emotion_detector;
pub mod meditation_detector;
pub mod entropy;  // Permutation Entropy for consciousness detection
pub mod sleep_sentinel;

pub use edf_loader::{EdfFile, EdfSignal, SleepAnnotation, SleepStage};
pub use emotion_detector::{
    EmotionSentinel, EmotionalState, EmotionCategory, EmotionChannel,
    EmotionEEG, FrontalAsymmetry, ArousalDetector, EmotionSimulator,
    FrequencyBand,
};
pub use meditation_detector::{
    MeditationSentinel, MeditationState, MeditationCategory, MeditationChannel,
    MeditationEEG, FrontalMidlineTheta, AlphaCoherence, GammaSynchrony,
    MeditationSimulator,
};
pub use entropy::{permutation_entropy_order3, permutation_entropy_order4, weighted_permutation_entropy};
pub use sleep_sentinel::{
    SleepSentinel, ConsciousnessState, SleepSentinelConfig,
    IntegrationMetrics, SleepSentinelStats, CalibrationState,
    AdaptiveThresholdConfig, CalibrationData,
};
