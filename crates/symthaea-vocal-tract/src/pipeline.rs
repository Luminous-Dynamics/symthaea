//! Complete vocal tract pipeline: encoder → controller → FEP agent.
//!
//! Manages dual-rate processing:
//! - 200Hz motor: controller.forward() → FormantFrame → vocoder
//! - 10Hz cognitive: encoder.encode() → update cached HV; fep_agent.tick() → modulate
//!
//! Also includes `ProsodyContext` for F0 declination, stress boost, and energy
//! ASR envelope post-processing.

use crate::controller::{SpeakerProfile, VocalTractConfig, VocalTractController};
use crate::encoder::{VocalTractHdcEncoder, VoiceCognitiveState};
use crate::fep::{VocalTractFepAgent, VocalTractObservation};
use crate::types::{FormantFrame, SourceType};
use symthaea_core::genesis::{GenesisCovenant, GenesisSeed};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// ═══════════════════════════════════════════════════════════════════════════════
// PROSODY CONTEXT
// ═══════════════════════════════════════════════════════════════════════════════

/// Context for prosody post-processing: F0 declination, stress boost, energy envelope.
#[derive(Debug, Clone, Copy)]
pub struct ProsodyContext {
    /// Progress through the entire utterance (0.0 → 1.0).
    pub utterance_progress: f32,
    /// Progress through the current phoneme (0.0 → 1.0).
    pub phoneme_progress: f32,
    /// Stress level (0 = unstressed, 1 = primary, 2 = secondary).
    pub stress: u8,
    /// Base F0 for the utterance (from config or speaker profile).
    pub base_f0: f32,
    /// Emotional arousal (0.0–1.0) — maps to F0 range expansion.
    pub arousal: f32,
}

impl Default for ProsodyContext {
    fn default() -> Self {
        Self {
            utterance_progress: 0.0,
            phoneme_progress: 0.5,
            stress: 0,
            base_f0: 120.0,
            arousal: 0.5,
        }
    }
}

impl ProsodyContext {
    /// Apply prosody post-processing to a FormantFrame.
    ///
    /// - **F0**: declination (0.85→1.0 over utterance) × stress boost × arousal range
    /// - **Energy**: ASR envelope (10% attack, sustain, 15% release) × stress factor
    pub fn apply_prosody(&self, frame: &mut FormantFrame) {
        // F0 declination: starts at 1.0, falls to 0.85 by end of utterance
        let declination = 1.0 - 0.15 * self.utterance_progress;

        // Stress boost: primary=1.10×, secondary=1.05×, none=1.0×
        let stress_f0_boost = match self.stress {
            1 => 1.10,
            2 => 1.05,
            _ => 1.0,
        };

        // Arousal maps to F0 range (more arousal = wider pitch swings)
        let arousal_factor = 0.9 + 0.2 * self.arousal;

        frame.f0 = self.base_f0 * declination * stress_f0_boost * arousal_factor;
        frame.f0 = frame.f0.clamp(50.0, 500.0);

        // Energy ASR envelope within the phoneme
        let asr_envelope = if self.phoneme_progress < 0.10 {
            // Attack: ramp up over first 10%
            self.phoneme_progress / 0.10
        } else if self.phoneme_progress > 0.85 {
            // Release: ramp down over last 15%
            (1.0 - self.phoneme_progress) / 0.15
        } else {
            // Sustain
            1.0
        };

        // Stress energy boost: primary=1.2×, secondary=1.1×, unstressed=0.9×
        let stress_energy = match self.stress {
            1 => 1.2,
            2 => 1.1,
            _ => 0.9,
        };

        frame.energy = (frame.energy * asr_envelope * stress_energy).clamp(0.0, 1.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VOCAL TRACT PIPELINE
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete vocal tract pipeline: encoder → controller → FEP agent.
///
/// Manages dual-rate processing:
/// - 200Hz motor: controller.forward() → FormantFrame → vocoder
/// - 10Hz cognitive: encoder.encode() → update cached HV; fep_agent.tick() → modulate
pub struct VocalTractPipeline {
    /// HDC encoder: cognitive state → 16,384D ContinuousHV
    pub encoder: VocalTractHdcEncoder,
    /// LTC controller: ContinuousHV → FormantFrame
    pub controller: VocalTractController,
    /// FEP active inference agent: voice metrics → tau/LR modulation
    pub fep_agent: VocalTractFepAgent,
    /// Cached cognitive HV (updated at 10Hz, used at 200Hz)
    cached_hv: ContinuousHV,
    /// Counter for dual-rate scheduling (every 20 motor frames = 1 cognitive tick)
    motor_frame_count: usize,
    /// Motor frames per cognitive tick (200Hz / 10Hz = 20)
    frames_per_cognitive_tick: usize,
    /// Cumulative time in seconds (set on each FormantFrame).
    cumulative_time: f32,
    /// Genesis seed for phoneme HV generation.
    genesis: GenesisSeed,
    /// Cache of phoneme identity HVs (phoneme name → HV).
    phoneme_hv_cache: std::collections::HashMap<String, ContinuousHV>,
    /// Cached cognitive channels for prosody head (updated at 10Hz, used at 200Hz).
    cached_cognitive_channels: Option<[f32; 12]>,
    /// Previous phoneme name (for detecting phoneme transitions).
    prev_phoneme: Option<String>,
    /// Previous phoneme's bound HV (for coarticulation blending during transitions).
    prev_phoneme_bound_hv: Option<ContinuousHV>,
    /// Counter of frames elapsed since phoneme changed (for blend scheduling).
    coarticulation_counter: usize,
    /// Number of frames over which to blend between old/new phoneme HVs (80ms at 200Hz).
    coarticulation_frames: usize,
    /// Phoneme name → manner of articulation (for setting source_type on output frames).
    phoneme_manner_map: std::collections::HashMap<String, SourceType>,
}

impl VocalTractPipeline {
    /// Create a new pipeline from a genesis seed.
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            encoder: VocalTractHdcEncoder::new(genesis, 32),
            controller: VocalTractController::new(genesis, &VocalTractConfig::default()),
            fep_agent: VocalTractFepAgent::new(),
            cached_hv: ContinuousHV::zero(HDC_DIMENSION),
            motor_frame_count: 0,
            frames_per_cognitive_tick: 20,
            cumulative_time: 0.0,
            genesis: genesis.clone(),
            phoneme_hv_cache: std::collections::HashMap::new(),
            cached_cognitive_channels: None,
            prev_phoneme: None,
            prev_phoneme_bound_hv: None,
            coarticulation_counter: 0,
            coarticulation_frames: 16, // 80ms at 200Hz
            phoneme_manner_map: std::collections::HashMap::new(),
        }
    }

    /// Create a new pipeline with a specific speaker profile.
    ///
    /// Derives a speaker-specific genesis via `GenesisCovenant`, applies the
    /// profile's base_f0 to config, modulates tau and formant scaling.
    pub fn new_with_speaker(genesis: &GenesisSeed, profile: &SpeakerProfile) -> Self {
        // Derive a speaker-specific genesis namespace
        let speaker_genesis = GenesisCovenant::new(genesis, &format!("speaker::{}", profile.name));
        let speaker_seed = speaker_genesis.seed();

        let config = VocalTractConfig {
            base_f0: profile.base_f0,
            ..VocalTractConfig::default()
        };

        let mut controller = VocalTractController::new(speaker_seed, &config);
        controller.modulate_tau(profile.tau_factor);

        Self {
            encoder: VocalTractHdcEncoder::new(speaker_seed, 32),
            controller,
            fep_agent: VocalTractFepAgent::new(),
            cached_hv: ContinuousHV::zero(HDC_DIMENSION),
            motor_frame_count: 0,
            frames_per_cognitive_tick: 20,
            cumulative_time: 0.0,
            genesis: speaker_seed.clone(),
            phoneme_hv_cache: std::collections::HashMap::new(),
            cached_cognitive_channels: None,
            prev_phoneme: None,
            prev_phoneme_bound_hv: None,
            coarticulation_counter: 0,
            coarticulation_frames: 16,
            phoneme_manner_map: std::collections::HashMap::new(),
        }
    }

    /// Set the phoneme manner map for source_type propagation.
    ///
    /// Maps phoneme names (ARPABET) to their manner of articulation. When
    /// `tick_phoneme()` produces a frame, it sets `frame.source_type` from
    /// this map so the vocoder uses the correct excitation signal.
    pub fn set_manner_map(&mut self, map: std::collections::HashMap<String, SourceType>) {
        self.phoneme_manner_map = map;
    }

    /// Register a single phoneme's manner of articulation.
    pub fn register_phoneme_manner(&mut self, phoneme: &str, manner: SourceType) {
        self.phoneme_manner_map
            .insert(phoneme.to_string(), manner);
    }

    /// Get or create a cached phoneme identity HV.
    ///
    /// Uses the genesis seed to deterministically create a unique HV for each
    /// phoneme name (e.g., "AH", "IY"). Cached for O(1) repeat lookups.
    pub fn get_or_create_phoneme_hv(&mut self, phoneme: &str) -> ContinuousHV {
        if let Some(hv) = self.phoneme_hv_cache.get(phoneme) {
            return hv.clone();
        }
        let hv = self
            .genesis
            .hv(&format!("phoneme::{phoneme}"), HDC_DIMENSION);
        self.phoneme_hv_cache
            .insert(phoneme.to_string(), hv.clone());
        hv
    }

    /// Run one motor frame (200Hz).
    ///
    /// - Every `frames_per_cognitive_tick` frames: re-encode cognitive state,
    ///   optionally run FEP tick if metrics provided.
    /// - Every frame: evolve controller with cached HV → produce FormantFrame.
    /// - If `phoneme` is provided, the cognitive HV is bound with a phoneme
    ///   identity HV so the controller can distinguish different phonemes.
    pub fn tick(
        &mut self,
        cognitive_state: &VoiceCognitiveState,
        metrics: Option<&VocalTractObservation>,
        dt: f32,
    ) -> FormantFrame {
        self.tick_phoneme(cognitive_state, metrics, dt, None)
    }

    /// Run one motor frame with phoneme-aware routing.
    ///
    /// Like `tick()`, but accepts an optional phoneme name. When provided,
    /// the cached cognitive HV is bound with a phoneme identity HV via
    /// `bind()`, injecting phoneme identity while preserving cognitive state.
    pub fn tick_phoneme(
        &mut self,
        cognitive_state: &VoiceCognitiveState,
        metrics: Option<&VocalTractObservation>,
        dt: f32,
        phoneme: Option<&str>,
    ) -> FormantFrame {
        // Cognitive tick (10Hz)
        if self.motor_frame_count % self.frames_per_cognitive_tick == 0 {
            self.cached_hv = self.encoder.encode(cognitive_state);
            self.cached_cognitive_channels = Some(cognitive_state.to_channels());

            // FEP modulation if we have metrics
            if let Some(m) = metrics {
                // Learn from previous action's outcome before selecting new action
                self.fep_agent.learn(m);
                let fep_result = self.fep_agent.tick(m);
                self.controller.modulate_tau(fep_result.tau_factor);
                let current_lr = self.controller.learning_rate();
                self.controller
                    .set_learning_rate(current_lr * fep_result.learning_rate_factor);
            }
        }

        self.motor_frame_count += 1;

        // Coarticulation blending: interpolate old→new phoneme HV over transition.
        // When phoneme changes, save the old bound HV and linearly blend toward
        // the new bound HV over `coarticulation_frames` (80ms at 200Hz).
        //
        // Optimization: skip bind() when phoneme is unchanged and this is not
        // a cognitive re-encode frame. Saves ~0.5-1ms for ~90% of frames.
        let effective_hv = if let Some(ph) = phoneme {
            let is_same = self.prev_phoneme.as_deref() == Some(ph);
            let is_reencode =
                (self.motor_frame_count - 1) % self.frames_per_cognitive_tick == 0;

            let new_bound = if is_same && !is_reencode && self.coarticulation_counter >= self.coarticulation_frames {
                // Cache hit: same phoneme, no re-encode, past blend window
                if let Some(ref cached) = self.prev_phoneme_bound_hv {
                    cached.clone()
                } else {
                    let phoneme_hv = self.get_or_create_phoneme_hv(ph);
                    let bound = self.cached_hv.bind(&phoneme_hv);
                    self.prev_phoneme_bound_hv = Some(bound.clone());
                    bound
                }
            } else {
                let phoneme_hv = self.get_or_create_phoneme_hv(ph);
                let bound = self.cached_hv.bind(&phoneme_hv);

                if !is_same {
                    // Phoneme changed — save old HV for blending
                    if let Some(old_ph) = self.prev_phoneme.take() {
                        let old_phoneme_hv = self.get_or_create_phoneme_hv(&old_ph);
                        self.prev_phoneme_bound_hv =
                            Some(self.cached_hv.bind(&old_phoneme_hv));
                        self.coarticulation_counter = 0;
                    }
                } else {
                    // Same phoneme, re-encode — refresh cached bound
                    self.prev_phoneme_bound_hv = Some(bound.clone());
                }

                bound
            };

            self.prev_phoneme = Some(ph.to_string());

            // Blend if within coarticulation window
            if self.coarticulation_counter < self.coarticulation_frames {
                self.coarticulation_counter += 1;
                if let Some(ref prev_hv) = self.prev_phoneme_bound_hv {
                    let t = self.coarticulation_counter as f32
                        / self.coarticulation_frames as f32;
                    prev_hv.scale(1.0 - t).add(&new_bound.scale(t))
                } else {
                    new_bound
                }
            } else {
                new_bound
            }
        } else {
            self.prev_phoneme = None;
            self.prev_phoneme_bound_hv = None;
            self.cached_hv.clone()
        };

        // Adaptive rate limiting: tighter during steady state, looser during transitions
        if phoneme.is_some() && self.coarticulation_counter < self.coarticulation_frames {
            self.controller
                .set_max_formant_delta(self.controller.config().transition_max_delta);
        } else {
            self.controller
                .set_max_formant_delta(self.controller.config().steady_max_delta);
        }

        // Motor tick (200Hz): evolve network + produce formants with prosody head
        let mut frame = self.controller.forward_with_prosody(
            &effective_hv,
            dt,
            self.cached_cognitive_channels.as_ref(),
        );
        frame.time = self.cumulative_time;
        self.cumulative_time += dt;

        // Set source_type from phoneme manner map (vocoder uses this for excitation)
        if let Some(ph) = phoneme {
            if let Some(&manner) = self.phoneme_manner_map.get(ph) {
                frame.source_type = manner;
            }
        }

        frame
    }

    /// Run one motor frame with phoneme routing + prosody post-processing.
    ///
    /// Combines `tick_phoneme()` with `ProsodyContext::apply_prosody()` for
    /// F0 declination, stress boost, and energy ASR envelope.
    pub fn tick_with_prosody(
        &mut self,
        cognitive_state: &VoiceCognitiveState,
        metrics: Option<&VocalTractObservation>,
        dt: f32,
        phoneme: Option<&str>,
        prosody: &ProsodyContext,
    ) -> FormantFrame {
        let mut frame = self.tick_phoneme(cognitive_state, metrics, dt, phoneme);
        prosody.apply_prosody(&mut frame);
        frame
    }

    /// Reset the entire pipeline.
    pub fn reset(&mut self) {
        self.encoder.reset();
        self.controller.reset();
        self.fep_agent.reset();
        self.cached_hv = ContinuousHV::zero(HDC_DIMENSION);
        self.motor_frame_count = 0;
        self.cumulative_time = 0.0;
        self.phoneme_hv_cache.clear();
        self.cached_cognitive_channels = None;
        self.prev_phoneme = None;
        self.prev_phoneme_bound_hv = None;
        self.coarticulation_counter = 0;
    }

    /// Get current cumulative time in seconds.
    pub fn cumulative_time(&self) -> f32 {
        self.cumulative_time
    }

    /// Get a reference to the stored genesis seed.
    pub fn genesis(&self) -> &GenesisSeed {
        &self.genesis
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::SpeakerProfile;
    use crate::encoder::VoiceCognitiveState;
    use crate::fep::VocalTractObservation;
    use crate::types::FormantFrame;
    use symthaea_core::genesis::GenesisSeed;

    #[test]
    fn test_pipeline_time_tracking() {
        let genesis = GenesisSeed::from_phrase("test-time-tracking");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();
        let dt = 0.005; // 200Hz

        // Run 40 frames
        let mut last_frame = pipeline.tick(&state, None, dt);
        for _ in 1..40 {
            last_frame = pipeline.tick(&state, None, dt);
        }

        // Last frame should have time = 39 * 0.005 = 0.195
        let expected_time = 39.0 * dt;
        assert!(
            (last_frame.time - expected_time).abs() < 1e-4,
            "Expected time ~{}, got {}",
            expected_time,
            last_frame.time
        );

        // Cumulative time should be 40 * 0.005 = 0.200
        assert!(
            (pipeline.cumulative_time() - 40.0 * dt).abs() < 1e-4,
            "Expected cumulative ~{}, got {}",
            40.0 * dt,
            pipeline.cumulative_time()
        );

        // Reset should clear time
        pipeline.reset();
        assert!((pipeline.cumulative_time()).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_fep_learning() {
        let genesis = GenesisSeed::from_phrase("test-fep-learning");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let state = VoiceCognitiveState::default();
        let metrics = VocalTractObservation {
            articulation_score: 0.6,
            formant_accuracy: 0.5,
            pitch_stability: 0.7,
            coarticulation_smoothness: 0.6,
            duration_accuracy: 0.5,
            energy_consistency: 0.6,
        };

        // Run 60 frames (3 cognitive ticks with FEP feedback)
        for _ in 0..60 {
            pipeline.tick(&state, Some(&metrics), 0.005);
        }

        // FEP agent should have ticked 3 times
        assert_eq!(pipeline.fep_agent.tick_count(), 3);
        // TD learning should have been triggered (learn is called before tick on 2nd+ cognitive tick)
        assert!(
            pipeline.fep_agent.stats().td_updates > 0,
            "TD learner should have updates after multiple cognitive ticks"
        );
    }

    #[test]
    fn test_pipeline_end_to_end() {
        let genesis = GenesisSeed::from_phrase("test-vocal-pipeline");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Run 40 frames (2 cognitive ticks)
        for _ in 0..40 {
            let frame = pipeline.tick(&state, None, 0.005);
            assert!(frame.f1 >= 200.0 && frame.f1 <= 1000.0);
            assert!(frame.energy >= 0.0 && frame.energy <= 1.0);
        }
    }

    #[test]
    fn test_pipeline_dual_rate() {
        let genesis = GenesisSeed::from_phrase("test-dual-rate");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let state1 = VoiceCognitiveState {
            emotional_arousal: 0.2,
            ..Default::default()
        };
        let state2 = VoiceCognitiveState {
            emotional_arousal: 0.9,
            ..Default::default()
        };

        // First cognitive tick (frame 0) with state1
        let frame_a = pipeline.tick(&state1, None, 0.005);

        // Frames 1-19 still use cached HV from state1
        for _ in 1..20 {
            pipeline.tick(&state1, None, 0.005);
        }

        // Frame 20: new cognitive tick with state2 (different arousal)
        let frame_b = pipeline.tick(&state2, None, 0.005);

        // Frame 21: still using state2's cached HV
        let frame_c = pipeline.tick(&state2, None, 0.005);

        // frame_a and frame_b used different cognitive inputs at re-encode boundaries
        // frame_b and frame_c used the same cognitive HV (frame_c at motor-only tick)
        assert!(frame_a.f1.is_finite());
        assert!(frame_b.f1.is_finite());
        assert!(frame_c.f1.is_finite());
    }

    #[test]
    fn test_pipeline_with_fep_feedback() {
        let genesis = GenesisSeed::from_phrase("test-fep-feedback");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let state = VoiceCognitiveState::default();
        let metrics = VocalTractObservation {
            articulation_score: 0.8,
            formant_accuracy: 0.7,
            pitch_stability: 0.9,
            coarticulation_smoothness: 0.8,
            duration_accuracy: 0.7,
            energy_consistency: 0.8,
        };

        // Run with FEP feedback
        for _ in 0..40 {
            let frame = pipeline.tick(&state, Some(&metrics), 0.005);
            assert!(frame.f1.is_finite());
        }

        // FEP agent should have ticked twice (at frames 0 and 20)
        assert_eq!(pipeline.fep_agent.tick_count(), 2);
    }

    #[test]
    fn test_phoneme_hv_caching() {
        let genesis = GenesisSeed::from_phrase("test-phoneme-cache");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let hv1 = pipeline.get_or_create_phoneme_hv("AH");
        let hv2 = pipeline.get_or_create_phoneme_hv("AH");

        // Same phoneme → identical HV (from cache)
        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-5,
            "Cached HVs should be identical"
        );

        // Different phoneme → different HV
        let hv3 = pipeline.get_or_create_phoneme_hv("IY");
        assert!(
            hv1.similarity(&hv3) < 0.5,
            "Different phonemes should have dissimilar HVs: sim={}",
            hv1.similarity(&hv3)
        );
    }

    #[test]
    fn test_phoneme_routing_different_output() {
        let genesis = GenesisSeed::from_phrase("test-phoneme-routing");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Run 40 frames with /AH/ (enough for LTC to evolve state)
        let mut frames_ah = Vec::new();
        for _ in 0..40 {
            frames_ah.push(pipeline.tick_phoneme(&state, None, 0.005, Some("AH")));
        }

        pipeline.reset();

        // Run 40 frames with /IY/
        let mut frames_iy = Vec::new();
        for _ in 0..40 {
            frames_iy.push(pipeline.tick_phoneme(&state, None, 0.005, Some("IY")));
        }

        // Compute total trajectory divergence across all 9 formant dims
        let total_diff: f32 = frames_ah
            .iter()
            .zip(frames_iy.iter())
            .map(|(a, b)| {
                (a.f1 - b.f1).abs()
                    + (a.f2 - b.f2).abs()
                    + (a.f3 - b.f3).abs()
                    + (a.f0 - b.f0).abs()
            })
            .sum();

        assert!(
            total_diff > 1.0,
            "Different phonemes should produce different formant trajectories: total_diff={total_diff:.2}"
        );
    }

    #[test]
    fn test_phoneme_none_backward_compat() {
        let genesis = GenesisSeed::from_phrase("test-backward-compat");
        let mut pipeline1 = VocalTractPipeline::new(&genesis);
        let mut pipeline2 = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // tick() (no phoneme) should produce same result as tick_phoneme(None)
        let frame1 = pipeline1.tick(&state, None, 0.005);
        let frame2 = pipeline2.tick_phoneme(&state, None, 0.005, None);

        assert!(
            (frame1.f1 - frame2.f1).abs() < 1e-4,
            "tick() and tick_phoneme(None) should be equivalent"
        );
        assert!((frame1.f0 - frame2.f0).abs() < 1e-4);
        assert!((frame1.energy - frame2.energy).abs() < 1e-6);
    }

    #[test]
    fn test_phoneme_bind_changes_similarity() {
        let genesis = GenesisSeed::from_phrase("test-bind-sim");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        // Get a cognitive HV
        let cognitive_hv = ContinuousHV::random(HDC_DIMENSION, 42);
        let phoneme_hv = pipeline.get_or_create_phoneme_hv("AH");

        // bind() should produce a vector dissimilar from the original
        let bound = cognitive_hv.bind(&phoneme_hv);
        let sim = cognitive_hv.similarity(&bound);
        assert!(sim < 0.5, "bind() should decorrelate: sim={}", sim);

        // But binding with same phoneme twice should give same result
        let bound2 = cognitive_hv.bind(&phoneme_hv);
        assert!(
            (bound.similarity(&bound2) - 1.0).abs() < 1e-5,
            "Same bind should be deterministic"
        );
    }

    #[test]
    fn test_prosody_f0_declination() {
        let base_f0 = 120.0;
        // Frame at utterance start
        let mut frame_start = FormantFrame {
            f0: 200.0, // will be overwritten by prosody
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        let ctx_start = ProsodyContext {
            utterance_progress: 0.0,
            phoneme_progress: 0.5,
            stress: 0,
            base_f0,
            arousal: 0.5,
        };
        ctx_start.apply_prosody(&mut frame_start);

        // Frame at utterance end
        let mut frame_end = FormantFrame {
            f0: 200.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        let ctx_end = ProsodyContext {
            utterance_progress: 1.0,
            phoneme_progress: 0.5,
            stress: 0,
            base_f0,
            arousal: 0.5,
        };
        ctx_end.apply_prosody(&mut frame_end);

        // F0 should decline over the utterance
        assert!(
            frame_start.f0 > frame_end.f0,
            "F0 should decline: start={:.1}, end={:.1}",
            frame_start.f0,
            frame_end.f0
        );

        // Declination magnitude: 15% over full utterance
        let ratio = frame_end.f0 / frame_start.f0;
        assert!(
            (ratio - 0.85).abs() < 0.05,
            "Expected ~15% declination, got ratio={ratio:.3}"
        );
    }

    #[test]
    fn test_prosody_stress_boost() {
        let base_f0 = 120.0;
        let mid_ctx = |stress: u8| ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            stress,
            base_f0,
            arousal: 0.5,
        };

        let mut frame_unstressed = FormantFrame {
            f0: 200.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        mid_ctx(0).apply_prosody(&mut frame_unstressed);

        let mut frame_primary = FormantFrame {
            f0: 200.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        mid_ctx(1).apply_prosody(&mut frame_primary);

        let mut frame_secondary = FormantFrame {
            f0: 200.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        mid_ctx(2).apply_prosody(&mut frame_secondary);

        // Primary stress > secondary > unstressed
        assert!(
            frame_primary.f0 > frame_secondary.f0,
            "Primary stress should have higher F0: primary={:.1}, secondary={:.1}",
            frame_primary.f0,
            frame_secondary.f0
        );
        assert!(
            frame_secondary.f0 > frame_unstressed.f0,
            "Secondary stress should have higher F0 than unstressed: secondary={:.1}, unstressed={:.1}",
            frame_secondary.f0,
            frame_unstressed.f0
        );

        // Energy: stressed > unstressed
        assert!(
            frame_primary.energy > frame_unstressed.energy,
            "Stressed phoneme should have higher energy"
        );
    }

    #[test]
    fn test_prosody_energy_envelope() {
        let base_ctx = |phoneme_progress: f32| ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress,
            stress: 1,
            base_f0: 120.0,
            arousal: 0.5,
        };

        // Attack phase (near start)
        let mut frame_attack = FormantFrame {
            f0: 120.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        base_ctx(0.02).apply_prosody(&mut frame_attack);

        // Sustain phase (middle)
        let mut frame_sustain = FormantFrame {
            f0: 120.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        base_ctx(0.5).apply_prosody(&mut frame_sustain);

        // Release phase (near end)
        let mut frame_release = FormantFrame {
            f0: 120.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        base_ctx(0.95).apply_prosody(&mut frame_release);

        // Attack < sustain (ramping up)
        assert!(
            frame_attack.energy < frame_sustain.energy,
            "Attack energy should be lower than sustain: attack={:.3}, sustain={:.3}",
            frame_attack.energy,
            frame_sustain.energy
        );

        // Release < sustain (ramping down)
        assert!(
            frame_release.energy < frame_sustain.energy,
            "Release energy should be lower than sustain: release={:.3}, sustain={:.3}",
            frame_release.energy,
            frame_sustain.energy
        );
    }

    #[test]
    fn test_speaker_male_vs_female_differ() {
        let genesis = GenesisSeed::from_phrase("test-speaker-profiles");
        let state = VoiceCognitiveState::default();

        let mut male_pipeline =
            VocalTractPipeline::new_with_speaker(&genesis, &SpeakerProfile::male());
        let mut female_pipeline =
            VocalTractPipeline::new_with_speaker(&genesis, &SpeakerProfile::female());

        // Run 40 frames each
        let mut male_frames = Vec::new();
        let mut female_frames = Vec::new();
        for _ in 0..40 {
            male_frames.push(male_pipeline.tick_phoneme(&state, None, 0.005, Some("AH")));
            female_frames.push(female_pipeline.tick_phoneme(&state, None, 0.005, Some("AH")));
        }

        // Formant trajectories should differ between male and female
        let total_diff: f32 = male_frames
            .iter()
            .zip(female_frames.iter())
            .map(|(m, f)| (m.f1 - f.f1).abs() + (m.f2 - f.f2).abs() + (m.f0 - f.f0).abs())
            .sum();

        assert!(
            total_diff > 10.0,
            "Male and female voices should produce different formants: total_diff={total_diff:.2}"
        );
    }

    #[test]
    fn test_speaker_profile_f0_applied() {
        let profile = SpeakerProfile::female();
        assert!(
            (profile.base_f0 - 220.0).abs() < 1.0,
            "Female base_f0 should be 220Hz"
        );
        assert!((profile.formant_scale - 1.15).abs() < 0.01);

        let child = SpeakerProfile::child();
        assert!(
            (child.base_f0 - 300.0).abs() < 1.0,
            "Child base_f0 should be 300Hz"
        );
        assert!((child.formant_scale - 1.30).abs() < 0.01);

        let male = SpeakerProfile::male();
        assert!(
            (male.base_f0 - 120.0).abs() < 1.0,
            "Male base_f0 should be 120Hz"
        );
    }

    #[test]
    fn test_speaker_genesis_determinism() {
        let genesis = GenesisSeed::from_phrase("test-speaker-determinism");
        let profile = SpeakerProfile::female();
        let state = VoiceCognitiveState::default();

        let mut p1 = VocalTractPipeline::new_with_speaker(&genesis, &profile);
        let mut p2 = VocalTractPipeline::new_with_speaker(&genesis, &profile);

        let frame1 = p1.tick(&state, None, 0.005);
        let frame2 = p2.tick(&state, None, 0.005);

        assert!(
            (frame1.f1 - frame2.f1).abs() < 1e-4,
            "Same genesis + profile → same output: f1={} vs {}",
            frame1.f1,
            frame2.f1
        );
        assert!((frame1.f0 - frame2.f0).abs() < 1e-4);
    }

    #[test]
    fn test_coarticulation_blending() {
        let genesis = GenesisSeed::from_phrase("test-coarticulation");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Run 30 frames of /AH/
        for _ in 0..30 {
            pipeline.tick_phoneme(&state, None, 0.005, Some("AH"));
        }

        // Transition to /IY/ — should blend over 16 frames
        let mut transition_frames = Vec::new();
        for _ in 0..30 {
            transition_frames.push(pipeline.tick_phoneme(&state, None, 0.005, Some("IY")));
        }

        // Run the same transition WITHOUT coarticulation (fresh pipeline, hard switch)
        let mut pipeline2 = VocalTractPipeline::new(&genesis);
        // Disable coarticulation by setting frames to 0
        pipeline2.coarticulation_frames = 0;

        for _ in 0..30 {
            pipeline2.tick_phoneme(&state, None, 0.005, Some("AH"));
        }

        let mut hard_frames = Vec::new();
        for _ in 0..30 {
            hard_frames.push(pipeline2.tick_phoneme(&state, None, 0.005, Some("IY")));
        }

        // The blended transition should differ from the hard switch in early frames
        let early_diff: f32 = transition_frames[..8]
            .iter()
            .zip(hard_frames[..8].iter())
            .map(|(a, b)| (a.f1 - b.f1).abs() + (a.f2 - b.f2).abs())
            .sum();

        // Late frames should converge (both approaches reach the same /IY/ target)
        let late_diff: f32 = transition_frames[20..]
            .iter()
            .zip(hard_frames[20..].iter())
            .map(|(a, b)| (a.f1 - b.f1).abs() + (a.f2 - b.f2).abs())
            .sum();

        // Blending should produce measurable difference in early frames
        assert!(
            early_diff > 0.1,
            "Coarticulation should differ from hard switch in early frames: diff={early_diff:.2}"
        );

        // Late frames should be closer (blending complete)
        assert!(
            late_diff < early_diff * 2.0,
            "Late frames should converge: early_diff={early_diff:.2}, late_diff={late_diff:.2}"
        );
    }

    #[test]
    fn test_phoneme_bind_cache_consistency() {
        // Verify that cached bind produces same output as uncached
        let genesis = GenesisSeed::from_phrase("test-bind-cache");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Run 5 frames with /AH/ (first triggers bind, rest use cache)
        let mut frames = Vec::new();
        for _ in 0..5 {
            frames.push(pipeline.tick_phoneme(&state, None, 0.005, Some("AH")));
        }

        // All frames should be valid (cache doesn't corrupt output)
        for (i, f) in frames.iter().enumerate() {
            assert!(
                f.f1 >= 200.0 && f.f1 <= 1000.0,
                "Frame {i}: F1={} out of range",
                f.f1
            );
            assert!(f.f1.is_finite(), "Frame {i}: F1 is not finite");
        }

        // Frames 2-4 (cached) should be close to frame 1 (uncached)
        // since the LTC network evolves smoothly with the same input
        let f1_drift: f32 = frames[1..].iter().map(|f| (f.f1 - frames[0].f1).abs()).sum::<f32>();
        assert!(
            f1_drift < 200.0,
            "Cached frames should be consistent with uncached: drift={f1_drift:.1}"
        );
    }

    #[test]
    fn test_adaptive_rate_limiting() {
        let genesis = GenesisSeed::from_phrase("test-adaptive-rate");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Verify config defaults
        assert!(
            (pipeline.controller.config().steady_max_delta - 12.0).abs() < 1e-4,
            "steady_max_delta should default to 12.0"
        );
        assert!(
            (pipeline.controller.config().transition_max_delta - 20.0).abs() < 1e-4,
            "transition_max_delta should default to 20.0"
        );

        // Train briefly so the network can distinguish AH vs IY
        let targets = vec![
            ("AH", crate::types::FormantTarget::vowel(520.0, 1190.0, 2390.0, 80.0)),
            ("IY", crate::types::FormantTarget::vowel(270.0, 2290.0, 3010.0, 100.0)),
        ];
        let target_refs: Vec<(&str, &crate::types::FormantTarget)> =
            targets.iter().map(|(name, t)| (*name, t)).collect();
        pipeline
            .controller
            .train_on_phoneme_targets(&genesis, &target_refs, 20);

        // Run 40 frames of /AH/ to establish steady state
        for _ in 0..40 {
            pipeline.tick_phoneme(&state, None, 0.005, Some("AH"));
        }

        // Collect steady-state frames (still /AH/)
        let mut steady_f1 = Vec::new();
        for _ in 0..20 {
            let frame = pipeline.tick_phoneme(&state, None, 0.005, Some("AH"));
            steady_f1.push(frame.f1);
        }

        // Transition to /IY/ — collect transition frames
        let mut transition_f1 = Vec::new();
        for _ in 0..20 {
            let frame = pipeline.tick_phoneme(&state, None, 0.005, Some("IY"));
            transition_f1.push(frame.f1);
        }

        // Verify steady-state delta respects 12.0 Hz/frame limit
        let steady_max: f32 = steady_f1
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            steady_max <= 12.5,
            "Steady-state F1 delta should be ≤12 Hz/frame: got {:.2}",
            steady_max
        );

        // Verify transition delta respects 20.0 Hz/frame limit
        let transition_max: f32 = transition_f1
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            transition_max <= 20.5,
            "Transition F1 delta should be ≤20 Hz/frame: got {:.2}",
            transition_max
        );
    }

    #[test]
    fn test_source_type_propagation() {
        use crate::types::SourceType;

        let genesis = GenesisSeed::from_phrase("test-source-type");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Register manner for specific phonemes
        pipeline.register_phoneme_manner("P", SourceType::Stop);
        pipeline.register_phoneme_manner("S", SourceType::Fricative);
        pipeline.register_phoneme_manner("M", SourceType::Nasal);
        pipeline.register_phoneme_manner("AH", SourceType::Vowel);

        // Stop consonant
        let frame_p = pipeline.tick_phoneme(&state, None, 0.005, Some("P"));
        assert_eq!(
            frame_p.source_type,
            SourceType::Stop,
            "P should produce Stop source type"
        );

        // Fricative
        let frame_s = pipeline.tick_phoneme(&state, None, 0.005, Some("S"));
        assert_eq!(
            frame_s.source_type,
            SourceType::Fricative,
            "S should produce Fricative source type"
        );

        // Nasal
        let frame_m = pipeline.tick_phoneme(&state, None, 0.005, Some("M"));
        assert_eq!(
            frame_m.source_type,
            SourceType::Nasal,
            "M should produce Nasal source type"
        );

        // Vowel
        let frame_ah = pipeline.tick_phoneme(&state, None, 0.005, Some("AH"));
        assert_eq!(
            frame_ah.source_type,
            SourceType::Vowel,
            "AH should produce Vowel source type"
        );

        // No phoneme → keeps controller default (Silent from ..Default::default())
        pipeline.reset();
        let frame_none = pipeline.tick_phoneme(&state, None, 0.005, None);
        assert_eq!(
            frame_none.source_type,
            SourceType::Silent,
            "No phoneme should keep controller default (Silent)"
        );
    }
}
