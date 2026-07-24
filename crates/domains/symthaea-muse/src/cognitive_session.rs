// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible temporal cognition for the narrow Sonata intervention path.
//!
//! V6 deliberately stopped at a clean theory/world-model/policy boundary while
//! Studio still constructed a bounded `MusicInferenceResult`. This module
//! replaces that synthetic hand-off with an actual temporal trajectory through
//! Symthaea's HDC/CfC and active-inference components.
//!
//! The current sensory stream is symbolic rather than renderer-derived: every
//! frame is obtained from a declared score region and converted into six bounded
//! proxies. The proxies are then encoded into an HDC state, evolved through a
//! CfC network, and passed to `MusicalInferenceEngine`. This is evidence that the
//! real cognitive stack participated in action selection; it is not evidence
//! that symbolic proxies equal perception or listener experience.

use crate::MusicalState;
use crate::audio_feedback::AudioFeatures;
use crate::cognitive_bridge::{
    CognitiveGoal, CognitiveObligationDemand, CognitiveSection, SymbolicMusicObservation,
};
use crate::musical_inference::{
    MusicInferenceLearningStats, MusicInferenceResult, MusicalInferenceEngine,
};
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_music_theory::{
    Duration, ObligationLedger, PlannedSonataSection, ScoreCognitiveProfile, SonataRealization,
    SonataSectionKind, profile_score_region,
};

/// Version of the temporal cognitive-session evidence contract.
pub const COGNITIVE_SESSION_VERSION: &str = "sonata-hdc-cfc-fep-v2";
/// The session is intentionally compact: enough dimensions to retain a stable
/// temporal state without serializing full production-size hypervectors.
pub const COGNITIVE_SESSION_HDC_DIMENSION: usize = 64;

/// Frozen parameters that make a cognitive session interpretable and replayable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveSessionConfig {
    /// Deterministic genesis phrase suffix. The piece seed is appended at run time.
    pub genesis_namespace: String,
    /// Number of equal observation windows per planned Sonata section.
    pub windows_per_section: u8,
    /// Fraction of each bounded sensory channel supplied by the CfC temporal state.
    pub temporal_blend: f32,
    /// Strength of sensory feedback into the evolving musical state.
    pub state_feedback_strength: f32,
    /// HDC dimension used by the compact temporal encoder.
    pub hdc_dimension: usize,
    /// CfC layer widths retained as provenance.
    pub cfc_layer_sizes: Vec<usize>,
}

impl Default for CognitiveSessionConfig {
    fn default() -> Self {
        Self {
            genesis_namespace: "symthaea-muse-sonata-cognition".into(),
            windows_per_section: 2,
            temporal_blend: 0.20,
            state_feedback_strength: 0.20,
            hdc_dimension: COGNITIVE_SESSION_HDC_DIMENSION,
            cfc_layer_sizes: vec![16, 16, 8],
        }
    }
}

impl CognitiveSessionConfig {
    fn bounded(mut self) -> Self {
        self.windows_per_section = self.windows_per_section.clamp(1, 8);
        self.temporal_blend = self.temporal_blend.clamp(0.0, 0.5);
        self.state_feedback_strength = self.state_feedback_strength.clamp(0.0, 1.0);
        self.hdc_dimension = self.hdc_dimension.clamp(16, 4096);
        if self.cfc_layer_sizes.is_empty() {
            self.cfc_layer_sizes = vec![16, 16, 8];
        }
        self
    }
}

/// Serializable six-channel sensory vector matching the FEP music observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CognitiveSensoryVector {
    pub spectral_centroid: f32,
    pub spectral_flux: f32,
    pub rhythm_entropy: f32,
    pub harmonic_tension: f32,
    pub rms_energy: f32,
    pub zero_crossing_rate: f32,
}

impl CognitiveSensoryVector {
    fn bounded(self) -> Self {
        Self {
            spectral_centroid: self.spectral_centroid.clamp(0.0, 1.0),
            spectral_flux: self.spectral_flux.clamp(0.0, 1.0),
            rhythm_entropy: self.rhythm_entropy.clamp(0.0, 1.0),
            harmonic_tension: self.harmonic_tension.clamp(0.0, 1.0),
            rms_energy: self.rms_energy.clamp(0.0, 1.0),
            zero_crossing_rate: self.zero_crossing_rate.clamp(0.0, 1.0),
        }
    }

    fn all_finite(self) -> bool {
        [
            self.spectral_centroid,
            self.spectral_flux,
            self.rhythm_entropy,
            self.harmonic_tension,
            self.rms_energy,
            self.zero_crossing_rate,
        ]
        .into_iter()
        .all(f32::is_finite)
    }
}

impl From<CognitiveSensoryVector> for AudioFeatures {
    fn from(value: CognitiveSensoryVector) -> Self {
        Self {
            spectral_centroid: value.spectral_centroid,
            spectral_flux: value.spectral_flux,
            rhythm_entropy: value.rhythm_entropy,
            harmonic_tension: value.harmonic_tension,
            rms_energy: value.rms_energy,
            zero_crossing_rate: value.zero_crossing_rate,
        }
    }
}

/// One observed region and the real cognitive state transition it produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveSessionFrame {
    pub sequence: u32,
    pub section: SonataSectionKind,
    pub start: Duration,
    pub end: Duration,
    pub symbolic_profile: ScoreCognitiveProfile,
    /// Direct deterministic score-side sensory proxies.
    pub raw_sensory: CognitiveSensoryVector,
    /// Sensory vector after bounded HDC/CfC temporal modulation.
    pub temporal_sensory: CognitiveSensoryVector,
    /// Similarity of this HDC observation to the preceding observation.
    pub input_similarity_to_previous: Option<f32>,
    /// Similarity of the CfC output state to the preceding output state.
    pub temporal_similarity_to_previous: Option<f32>,
    /// Non-cryptographic fingerprints retained to detect accidental trace changes.
    pub input_state_fingerprint: String,
    pub temporal_state_fingerprint: String,
    pub state_before: MusicalState,
    pub state_after: MusicalState,
    /// True only when the selected FEP action was committed as the cause of the
    /// next observation, enabling temporal-difference learning.
    pub fep_action_committed: bool,
    pub inference: MusicInferenceResult,
}

/// Auditable trajectory used to obtain one terminal cognitive action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveSessionTrace {
    pub session_version: String,
    pub backend: String,
    pub seed: u64,
    /// Explicit FEP action-selection seed derived from the piece seed and the
    /// frozen session namespace.
    pub fep_rng_seed: u64,
    pub config: CognitiveSessionConfig,
    /// Six-channel expected-observation goal installed in the FEP engine.
    pub fep_goal_preferences: Vec<f64>,
    pub fep_goal_precision: f64,
    pub frames: Vec<CognitiveSessionFrame>,
    pub engine_cycle_count: u64,
    pub fep_learning: MusicInferenceLearningStats,
    pub terminal_inference: MusicInferenceResult,
    pub terminal_state: MusicalState,
    /// FNV-1a over canonical JSON with this field cleared. This is an integrity
    /// fingerprint, not a cryptographic artifact digest.
    pub session_fingerprint: String,
}

impl CognitiveSessionTrace {
    pub fn terminal_inference(&self) -> &MusicInferenceResult {
        &self.terminal_inference
    }

    /// Build the symbolic bridge observation from the state actually reached by
    /// the temporal session and the theory ledger at the decision boundary.
    pub fn bridge_observation(
        &self,
        obligations: &ObligationLedger,
        now: Duration,
        section: CognitiveSection,
        active_goal: Option<CognitiveGoal>,
        goal_urgency: f32,
    ) -> SymbolicMusicObservation {
        let state = &self.terminal_state;
        let (dominant_harmony, dominant_harmony_activation) = state
            .harmony_activations
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap_or((0, 0.0));
        let pressure = obligations.pressure_at(now);
        let obligation_demands = obligations
            .pending()
            .into_iter()
            .map(|item| CognitiveObligationDemand {
                id: item.id,
                priority: item.priority,
                due_by: item.due_by,
                overdue: item.is_due_at(now),
                kind: item.kind.clone(),
            })
            .collect();

        SymbolicMusicObservation {
            section,
            active_goal,
            goal_urgency: goal_urgency.clamp(0.0, 1.0),
            valence: state.valence.clamp(-1.0, 1.0),
            arousal: state.arousal.clamp(0.0, 1.0),
            prediction_error: state.prediction_error.clamp(0.0, 1.0),
            consciousness_level: state.consciousness_level.clamp(0.0, 1.0),
            dominant_harmony,
            dominant_harmony_activation,
            pending_obligations: pressure.pending_count,
            overdue_obligations: obligations
                .overdue_at(now)
                .into_iter()
                .map(|item| item.id)
                .collect(),
            obligation_demands,
            obligation_pressure: pressure.weighted_pressure,
        }
    }

    pub fn validate(&self) -> Vec<CognitiveSessionIssue> {
        let mut issues = Vec::new();
        if self.session_version != COGNITIVE_SESSION_VERSION {
            issues.push(CognitiveSessionIssue::UnsupportedVersion(
                self.session_version.clone(),
            ));
        }
        if self.backend != "symthaea-hdc-cfc-fep" {
            issues.push(CognitiveSessionIssue::UnexpectedBackend(
                self.backend.clone(),
            ));
        }
        if self.frames.is_empty() {
            issues.push(CognitiveSessionIssue::MissingFrames);
        }
        for (index, frame) in self.frames.iter().enumerate() {
            if frame.sequence as usize != index {
                issues.push(CognitiveSessionIssue::NonContiguousFrameSequence {
                    expected: index as u32,
                    found: frame.sequence,
                });
            }
            if frame.end.beats() <= frame.start.beats() {
                issues.push(CognitiveSessionIssue::InvalidFrameRegion(frame.sequence));
            }
            if matches!(
                frame.section,
                SonataSectionKind::RecapitulationPrimary
                    | SonataSectionKind::RecapitulationSecondary
            ) {
                issues.push(CognitiveSessionIssue::DecisionBoundaryLeakage(
                    frame.sequence,
                ));
            }
            if !frame.raw_sensory.all_finite() || !frame.temporal_sensory.all_finite() {
                issues.push(CognitiveSessionIssue::NonFiniteSensory(frame.sequence));
            }
            if !inference_is_finite(&frame.inference) {
                issues.push(CognitiveSessionIssue::NonFiniteInference(frame.sequence));
            }
            if !frame.fep_action_committed {
                issues.push(CognitiveSessionIssue::UncommittedFepAction(frame.sequence));
            }
        }
        if self.engine_cycle_count != self.frames.len() as u64 {
            issues.push(CognitiveSessionIssue::CycleCountMismatch {
                expected: self.frames.len() as u64,
                found: self.engine_cycle_count,
            });
        }
        let expected_fep_seed = derive_fep_rng_seed(self.seed, &self.config.genesis_namespace);
        if self.fep_rng_seed != expected_fep_seed {
            issues.push(CognitiveSessionIssue::FepSeedMismatch {
                expected: expected_fep_seed,
                found: self.fep_rng_seed,
            });
        }
        if self.fep_goal_preferences.len() != 6
            || self
                .fep_goal_preferences
                .iter()
                .any(|value| !value.is_finite())
            || !self.fep_goal_precision.is_finite()
            || self.fep_goal_precision <= 0.0
        {
            issues.push(CognitiveSessionIssue::InvalidFepGoal);
        } else if let Some(first) = self.frames.first() {
            let (expected_preferences, expected_precision) =
                MusicalInferenceEngine::emotion_goal(&first.state_before);
            if self.fep_goal_preferences != expected_preferences
                || self.fep_goal_precision != expected_precision
            {
                issues.push(CognitiveSessionIssue::FepGoalMismatch);
            }
        }
        let expected_transitions = self.frames.len().saturating_sub(1);
        if self.fep_learning.committed_actions != self.frames.len() as u64
            || self.fep_learning.td_transition_history_size != expected_transitions
            || (expected_transitions > 0 && self.fep_learning.td_total_updates == 0)
            || !self.fep_learning.td_average_error.is_finite()
            || !self.fep_learning.td_average_prediction_accuracy.is_finite()
        {
            issues.push(CognitiveSessionIssue::InvalidFepLearningEvidence);
        }
        if !inference_is_finite(&self.terminal_inference) {
            issues.push(CognitiveSessionIssue::NonFiniteTerminalInference);
        }
        if let Some(last) = self.frames.last()
            && last.inference != self.terminal_inference
        {
            issues.push(CognitiveSessionIssue::TerminalInferenceMismatch);
        }
        if self.session_fingerprint != session_fingerprint(self) {
            issues.push(CognitiveSessionIssue::FingerprintMismatch);
        }
        issues
    }

    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

/// Invalid or internally inconsistent cognitive-session evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveSessionIssue {
    UnsupportedVersion(String),
    UnexpectedBackend(String),
    MissingFrames,
    NonContiguousFrameSequence { expected: u32, found: u32 },
    InvalidFrameRegion(u32),
    DecisionBoundaryLeakage(u32),
    NonFiniteSensory(u32),
    NonFiniteInference(u32),
    UncommittedFepAction(u32),
    CycleCountMismatch { expected: u64, found: u64 },
    FepSeedMismatch { expected: u64, found: u64 },
    InvalidFepGoal,
    FepGoalMismatch,
    InvalidFepLearningEvidence,
    NonFiniteTerminalInference,
    TerminalInferenceMismatch,
    FingerprintMismatch,
}

/// Failure to construct a real temporal session from the declared score plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveSessionError {
    MissingPlannedSections,
    MissingTargetSection,
    InvalidObservationRegion,
    NoObservationFrames,
    InvalidSessionEvidence(Vec<CognitiveSessionIssue>),
}

/// Run the real HDC/CfC/FEP trajectory from the opening through the primary
/// recapitulation decision boundary.
pub fn run_sonata_cognitive_session(
    realization: &SonataRealization,
    initial_state: &MusicalState,
    seed: u64,
    config: CognitiveSessionConfig,
) -> Result<CognitiveSessionTrace, CognitiveSessionError> {
    let config = config.bounded();
    if realization.plan.sections.is_empty() {
        return Err(CognitiveSessionError::MissingPlannedSections);
    }
    if !realization
        .plan
        .sections
        .iter()
        .any(|section| section.kind == SonataSectionKind::RecapitulationPrimary)
    {
        return Err(CognitiveSessionError::MissingTargetSection);
    }

    let genesis_label = format!("{}:{seed}", config.genesis_namespace);
    let genesis = GenesisSeed::from_phrase(&genesis_label);
    let network_config = UnifiedNetworkConfig {
        layer_sizes: config.cfc_layer_sizes.clone(),
        neuron_config: UnifiedConfig {
            dimension: config.hdc_dimension,
            tau_base: 0.20,
            backbone_tau: 0.80,
            ..UnifiedConfig::default()
        },
        use_layer_binding: true,
        skip_connections: true,
    };
    let mut temporal_network = HdcLtcUnifiedNetwork::from_genesis(network_config, &genesis);
    let fep_rng_seed = derive_fep_rng_seed(seed, &config.genesis_namespace);
    let mut inference_engine = MusicalInferenceEngine::new_with_seed(fep_rng_seed);
    inference_engine.set_emotion_anchor(initial_state);
    let fep_goal_preferences = inference_engine.goal_preferences().to_vec();
    let fep_goal_precision = inference_engine.goal_precision();
    let mut evolving_state = initial_state.clone();
    let mut frames = Vec::new();
    let mut previous_profile = None;
    let mut previous_input: Option<ContinuousHV> = None;
    let mut previous_output: Option<ContinuousHV> = None;

    for section in &realization.plan.sections {
        // The target return must remain unseen. The terminal action is chosen
        // prospectively from the completed development at the boundary where
        // the primary recapitulation begins.
        if section.kind == SonataSectionKind::RecapitulationPrimary {
            break;
        }
        for (start, end) in observation_windows(section, config.windows_per_section) {
            let profile = profile_score_region(&realization.score, start, end)
                .ok_or(CognitiveSessionError::InvalidObservationRegion)?;
            let raw_sensory = symbolic_sensory(profile, previous_profile);
            let input = encode_observation(
                &genesis,
                config.hdc_dimension,
                section.kind,
                frames.len() as u32,
                raw_sensory,
            );
            let dt = ((end.saturating_sub(start)).seconds(f64::from(realization.score.tempo_bpm))
                as f32)
                .clamp(0.01, 4.0);
            temporal_network.evolve_closed_form(dt, &input);
            let output = temporal_network.output();
            let temporal_sensory =
                apply_temporal_modulation(raw_sensory, &output, config.temporal_blend);
            let state_before = evolving_state.clone();
            let audio_features: AudioFeatures = temporal_sensory.into();
            let inference = inference_engine.infer_and_commit(&audio_features);
            audio_features.modulate_state(&mut evolving_state, config.state_feedback_strength);
            inference_engine.apply_action(&inference, &mut evolving_state);
            let frame = CognitiveSessionFrame {
                sequence: frames.len() as u32,
                section: section.kind,
                start,
                end,
                symbolic_profile: profile,
                raw_sensory,
                temporal_sensory,
                input_similarity_to_previous: previous_input
                    .as_ref()
                    .map(|previous| input.similarity(previous)),
                temporal_similarity_to_previous: previous_output
                    .as_ref()
                    .map(|previous| output.similarity(previous)),
                input_state_fingerprint: hypervector_fingerprint(&input),
                temporal_state_fingerprint: hypervector_fingerprint(&output),
                state_before,
                state_after: evolving_state.clone(),
                fep_action_committed: true,
                inference,
            };
            previous_profile = Some(profile);
            previous_input = Some(input);
            previous_output = Some(output);
            frames.push(frame);
        }
    }

    let terminal_inference = frames
        .last()
        .map(|frame| frame.inference.clone())
        .ok_or(CognitiveSessionError::NoObservationFrames)?;
    let mut trace = CognitiveSessionTrace {
        session_version: COGNITIVE_SESSION_VERSION.into(),
        backend: "symthaea-hdc-cfc-fep".into(),
        seed,
        fep_rng_seed,
        config,
        fep_goal_preferences,
        fep_goal_precision,
        frames,
        engine_cycle_count: inference_engine.cycle_count(),
        fep_learning: inference_engine.learning_stats(),
        terminal_inference,
        terminal_state: evolving_state,
        session_fingerprint: String::new(),
    };
    trace.session_fingerprint = session_fingerprint(&trace);
    let issues = trace.validate();
    if issues.is_empty() {
        Ok(trace)
    } else {
        Err(CognitiveSessionError::InvalidSessionEvidence(issues))
    }
}

fn observation_windows(
    section: &PlannedSonataSection,
    windows_per_section: u8,
) -> Vec<(Duration, Duration)> {
    let width = section
        .end
        .saturating_sub(section.start)
        .scale(1, windows_per_section as i64);
    let mut windows = Vec::with_capacity(windows_per_section as usize);
    let mut start = section.start;
    for index in 0..windows_per_section {
        let end = if index + 1 == windows_per_section {
            section.end
        } else {
            start + width
        };
        if end.beats() > start.beats() {
            windows.push((start, end));
        }
        start = end;
    }
    windows
}

fn symbolic_sensory(
    profile: ScoreCognitiveProfile,
    previous: Option<ScoreCognitiveProfile>,
) -> CognitiveSensoryVector {
    let spectral_flux = previous.map_or(0.0, |prior| {
        ((profile.tension - prior.tension).abs()
            + (profile.density - prior.density).abs()
            + (profile.familiarity - prior.familiarity).abs()
            + (profile.tonal_displacement - prior.tonal_displacement).abs())
            * 0.25
    });
    let voice_energy = (profile.active_voice_count as f32 / 4.0).clamp(0.0, 1.0);
    CognitiveSensoryVector {
        spectral_centroid: 0.15 + 0.70 * profile.tonal_displacement,
        spectral_flux,
        rhythm_entropy: 0.60 * profile.density + 0.40 * (1.0 - profile.familiarity),
        harmonic_tension: profile.tension,
        rms_energy: 0.55 * profile.density + 0.45 * voice_energy,
        zero_crossing_rate: 0.15 * (1.0 - profile.familiarity) + 0.10 * profile.tension,
    }
    .bounded()
}

fn derive_fep_rng_seed(seed: u64, namespace: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in namespace
        .as_bytes()
        .iter()
        .copied()
        .chain(seed.to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    if hash == 0 { 0x9E3779B97F4A7C15 } else { hash }
}

fn encode_observation(
    genesis: &GenesisSeed,
    dimension: usize,
    section: SonataSectionKind,
    sequence: u32,
    sensory: CognitiveSensoryVector,
) -> ContinuousHV {
    let channels = [
        (sensory.spectral_centroid, "centroid"),
        (sensory.spectral_flux, "flux"),
        (sensory.rhythm_entropy, "rhythm"),
        (sensory.harmonic_tension, "tension"),
        (sensory.rms_energy, "energy"),
        (sensory.zero_crossing_rate, "noise"),
    ];
    let mut encoded = ContinuousHV::zero(dimension);
    for (value, label) in channels {
        let basis = genesis.hv(&format!("session:{label}"), dimension);
        encoded = encoded.add(&basis.scale(value));
    }
    let section_basis = genesis.hv(section_label(section), dimension);
    encoded = encoded.add(&section_basis.scale(0.35));
    let phase_basis = genesis.hv(&format!("session:phase:{}", sequence % 16), dimension);
    encoded.add(&phase_basis.scale(0.10)).normalize()
}

fn section_label(section: SonataSectionKind) -> &'static str {
    match section {
        SonataSectionKind::ExpositionPrimary => "session:section:exposition-primary",
        SonataSectionKind::ExpositionSecondary => "session:section:exposition-secondary",
        SonataSectionKind::Development => "session:section:development",
        SonataSectionKind::RecapitulationPrimary => "session:section:recapitulation-primary",
        SonataSectionKind::RecapitulationSecondary => "session:section:recapitulation-secondary",
    }
}

fn apply_temporal_modulation(
    raw: CognitiveSensoryVector,
    output: &ContinuousHV,
    blend: f32,
) -> CognitiveSensoryVector {
    let latent = |index: usize| {
        output
            .values
            .get(index)
            .copied()
            .map(sigmoid)
            .unwrap_or(0.5)
    };
    let mix = |base: f32, index: usize| base * (1.0 - blend) + latent(index) * blend;
    CognitiveSensoryVector {
        spectral_centroid: mix(raw.spectral_centroid, 0),
        spectral_flux: mix(raw.spectral_flux, 1),
        rhythm_entropy: mix(raw.rhythm_entropy, 2),
        harmonic_tension: mix(raw.harmonic_tension, 3),
        rms_energy: mix(raw.rms_energy, 4),
        zero_crossing_rate: mix(raw.zero_crossing_rate, 5),
    }
    .bounded()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value.clamp(-8.0, 8.0)).exp())
}

fn hypervector_fingerprint(hv: &ContinuousHV) -> String {
    let mut hash = FNV_OFFSET_BASIS;
    for value in &hv.values {
        hash = fnv1a(hash, &value.to_bits().to_le_bytes());
    }
    format!("fnv1a64:{hash:016x}")
}

fn session_fingerprint(trace: &CognitiveSessionTrace) -> String {
    let mut canonical = trace.clone();
    canonical.session_fingerprint.clear();
    let bytes = serde_json::to_vec(&canonical).unwrap_or_default();
    let hash = fnv1a(FNV_OFFSET_BASIS, &bytes);
    format!("fnv1a64:{hash:016x}")
}

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn inference_is_finite(inference: &MusicInferenceResult) -> bool {
    [
        inference.free_energy,
        inference.prediction_error,
        inference.surprise,
        inference.learning_rate_mod,
        inference.sensory_precision,
        inference.prior_precision,
    ]
    .into_iter()
    .all(f64::is_finite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{
        MusicalIntent, PitchClass, Style, VoiceRole, compose_sonata_with_plan,
    };

    fn realization(seed: u64) -> SonataRealization {
        let intent = MusicalIntent {
            seed,
            tonic: PitchClass::C,
            ..MusicalIntent::default()
        };
        compose_sonata_with_plan(&intent, &Style::Sonata.spec()).unwrap()
    }

    #[test]
    fn temporal_session_runs_real_hdc_cfc_and_fep_cycles() {
        let trace = run_sonata_cognitive_session(
            &realization(41),
            &MusicalState::default(),
            41,
            CognitiveSessionConfig::default(),
        )
        .unwrap();

        assert_eq!(trace.backend, "symthaea-hdc-cfc-fep");
        assert_eq!(trace.engine_cycle_count, trace.frames.len() as u64);
        assert!(trace.frames.len() >= 6);
        assert!(trace.is_valid(), "{:?}", trace.validate());
        assert!(trace.frames.iter().all(|frame| !matches!(
            frame.section,
            SonataSectionKind::RecapitulationPrimary | SonataSectionKind::RecapitulationSecondary
        )));
        assert!(
            trace
                .frames
                .iter()
                .skip(1)
                .all(|frame| frame.input_similarity_to_previous.is_some())
        );
    }

    #[test]
    fn identical_inputs_replay_exactly() {
        let first = run_sonata_cognitive_session(
            &realization(59),
            &MusicalState::default(),
            59,
            CognitiveSessionConfig::default(),
        )
        .unwrap();
        let second = run_sonata_cognitive_session(
            &realization(59),
            &MusicalState::default(),
            59,
            CognitiveSessionConfig::default(),
        )
        .unwrap();

        assert_eq!(first.fep_rng_seed, second.fep_rng_seed);
        assert_eq!(first.session_fingerprint, second.session_fingerprint);
        assert_eq!(first, second);
    }

    #[test]
    fn tampered_fep_seed_is_rejected() {
        let mut trace = run_sonata_cognitive_session(
            &realization(61),
            &MusicalState::default(),
            61,
            CognitiveSessionConfig::default(),
        )
        .unwrap();
        trace.fep_rng_seed ^= 1;
        assert!(
            trace
                .validate()
                .iter()
                .any(|issue| matches!(issue, CognitiveSessionIssue::FepSeedMismatch { .. }))
        );
    }

    #[test]
    fn target_region_changes_cannot_leak_into_pre_target_session() {
        let baseline_realization = realization(67);
        let mut altered = baseline_realization.clone();
        let target = altered
            .plan
            .sections
            .iter()
            .find(|section| section.kind == SonataSectionKind::RecapitulationPrimary)
            .unwrap();
        if let Some(note) = altered.score.notes.iter_mut().find(|note| {
            note.role == VoiceRole::Melody
                && note.onset.beats() >= target.start.beats()
                && note.onset.beats() < target.end.beats()
        }) {
            note.pitch = note.pitch.transpose(5);
        }

        let baseline = run_sonata_cognitive_session(
            &baseline_realization,
            &MusicalState::default(),
            67,
            CognitiveSessionConfig::default(),
        )
        .unwrap();
        let changed = run_sonata_cognitive_session(
            &altered,
            &MusicalState::default(),
            67,
            CognitiveSessionConfig::default(),
        )
        .unwrap();

        assert_eq!(baseline, changed);
    }

    #[test]
    fn earlier_score_history_changes_session_evidence() {
        let mut altered = realization(73);
        let development = altered
            .plan
            .sections
            .iter()
            .find(|section| section.kind == SonataSectionKind::Development)
            .unwrap();
        if let Some(note) = altered.score.notes.iter_mut().find(|note| {
            note.role == VoiceRole::Melody
                && note.onset.beats() >= development.start.beats()
                && note.onset.beats() < development.end.beats()
        }) {
            note.pitch = note.pitch.transpose(6);
        }

        let baseline = run_sonata_cognitive_session(
            &realization(73),
            &MusicalState::default(),
            73,
            CognitiveSessionConfig::default(),
        )
        .unwrap();
        let changed = run_sonata_cognitive_session(
            &altered,
            &MusicalState::default(),
            73,
            CognitiveSessionConfig::default(),
        )
        .unwrap();

        assert_ne!(baseline.session_fingerprint, changed.session_fingerprint);
        assert_ne!(
            baseline.frames[4].input_state_fingerprint,
            changed.frames[4].input_state_fingerprint
        );
    }

    #[test]
    fn tampered_session_fingerprint_is_rejected() {
        let mut trace = run_sonata_cognitive_session(
            &realization(91),
            &MusicalState::default(),
            91,
            CognitiveSessionConfig::default(),
        )
        .unwrap();
        trace.frames[0].raw_sensory.harmonic_tension = 1.0;
        assert!(
            trace
                .validate()
                .contains(&CognitiveSessionIssue::FingerprintMismatch)
        );
    }
}
