// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HDC/LTC motor control driving a functional area-tract waveguide.

use anyhow::Result;
use symthaea_core::genesis::GenesisSeed;
use symthaea_vocal_tract::{
    ArticulationPlace, ArticulatoryProjection, ArticulatoryState, BranchedWaveguideConfig,
    BranchedWaveguideV2, ConstrictionManner, FunctionalTractRenderer, FunctionalVoiceIdentity,
    GestureFrame, GesturePlanner, IdentityAnatomy, IdentityPhysiology, ResidualDetailModel,
    SignedUnit, TransmissionLineReference, UnitInterval, VocalTractPipeline,
    encoder::VoiceCognitiveState,
    types::{FormantFrame, SourceType},
};

use super::formant_targets::FormantDatabase;
use super::singing_engine::{SingingVoiceEngine, VocalPerformance, VocalStem};

pub const FUNCTIONAL_MOTOR_RATE: f32 = 200.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionalRendererKind {
    Kl24BaselineV1,
    BranchedWaveguideV2,
    TransmissionLineReference,
}

#[derive(Debug, Clone)]
struct PhoneSpan {
    start: f32,
    end: f32,
    phoneme: String,
    syllable: usize,
}

fn ipa_to_arpa(ipa: &str) -> &'static str {
    match ipa.trim_matches(|c: char| c == 'ː' || c.is_ascii_digit()) {
        "i" | "iː" => "IY",
        "ɪ" => "IH",
        "e" | "eɪ" => "EY",
        "ɛ" => "EH",
        "æ" => "AE",
        "ɑ" | "ɑː" | "ɒ" => "AA",
        "ɔ" | "ɔː" => "AO",
        "o" | "oʊ" => "OW",
        "ʊ" => "UH",
        "u" | "uː" => "UW",
        "ʌ" => "AH",
        "ə" => "AX",
        "ɝ" | "ɜ" | "ɜː" => "ER",
        "aɪ" => "AY",
        "aʊ" => "AW",
        "ɔɪ" => "OY",
        "p" => "P",
        "b" => "B",
        "t" => "T",
        "d" => "D",
        "k" => "K",
        "g" | "ɡ" => "G",
        "f" => "F",
        "v" => "V",
        "θ" => "TH",
        "ð" => "DH",
        "s" => "S",
        "z" => "Z",
        "ʃ" => "SH",
        "ʒ" => "ZH",
        "h" => "HH",
        "tʃ" => "CH",
        "dʒ" => "JH",
        "m" => "M",
        "n" => "N",
        "ŋ" => "NG",
        "l" => "L",
        "r" | "ɹ" => "R",
        "w" => "W",
        "j" => "Y",
        _ => "AX",
    }
}

fn phone_spans(performance: &VocalPerformance) -> Vec<PhoneSpan> {
    let mut spans = Vec::new();
    for (syllable_index, syllable) in performance.syllables.iter().enumerate() {
        let total: f32 = syllable.notes().map(|note| note.duration).sum();
        let consonant_total: f32 = syllable
            .phonemes
            .iter()
            .filter(|phone| !phone.is_vowel)
            .map(|phone| phone.natural_duration_s.max(0.015))
            .sum();
        let consonant_scale = ((total - 0.06).max(0.0) / consonant_total.max(1e-6)).min(1.0);
        let vowel_weight: f32 = syllable
            .phonemes
            .iter()
            .filter(|phone| phone.is_vowel)
            .map(|phone| phone.natural_duration_s.max(0.01))
            .sum();
        let vowel_budget = (total - consonant_total * consonant_scale).max(0.03);
        let mut cursor = (syllable.note.start_time - syllable.consonant_advance_s).max(0.0);
        for phone in &syllable.phonemes {
            let duration = if phone.is_vowel {
                vowel_budget * phone.natural_duration_s.max(0.01) / vowel_weight.max(1e-6)
            } else {
                phone.natural_duration_s.max(0.015) * consonant_scale
            }
            .max(0.005);
            spans.push(PhoneSpan {
                start: cursor,
                end: cursor + duration,
                phoneme: ipa_to_arpa(&phone.ipa).to_string(),
                syllable: syllable_index,
            });
            cursor += duration;
        }
    }
    spans
}

fn interpolate_curve(points: &[(f32, f32)], time: f32, fallback: f32) -> f32 {
    let Some(&(first_t, first_v)) = points.first() else {
        return fallback;
    };
    if time <= first_t {
        return first_v;
    }
    for pair in points.windows(2) {
        let (a_t, a_v) = pair[0];
        let (b_t, b_v) = pair[1];
        if time <= b_t {
            let ratio = ((time - a_t) / (b_t - a_t).max(1e-6)).clamp(0.0, 1.0);
            return a_v + ratio * (b_v - a_v);
        }
    }
    points.last().map(|point| point.1).unwrap_or(fallback)
}

fn expression(
    performance: &VocalPerformance,
    time: f32,
    field: fn(&super::singing_engine::ExpressionPoint) -> f32,
) -> f32 {
    let mut points: Vec<_> = performance
        .phrases
        .iter()
        .flat_map(|phrase| phrase.expression_curve.iter())
        .map(|point| (point.time_s, field(point)))
        .collect();
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    interpolate_curve(&points, time, 0.5).clamp(0.0, 1.0)
}

fn gesture_for_phoneme(
    phoneme: Option<&str>,
    state: &ArticulatoryState,
    time_s: f32,
) -> GestureFrame {
    let mut gesture = GestureFrame {
        time_s,
        f0_hz: state.f0,
        energy: UnitInterval::new(state.energy),
        glottal_adduction: UnitInterval::new(0.18 + 0.78 * state.voicing),
        vocal_fold_tension: UnitInterval::new(
            ((state.f0.max(50.0) / 440.0).log2() * 0.16 + 0.52).clamp(0.0, 1.0),
        ),
        respiratory_effort: UnitInterval::new((0.25 + 0.75 * state.energy).clamp(0.0, 1.0)),
        ..Default::default()
    };
    match phoneme.unwrap_or("SIL") {
        "IY" => set_vowel(&mut gesture, 0.24, 0.82, 0.92, 0.28),
        "IH" => set_vowel(&mut gesture, 0.34, 0.58, 0.72, 0.20),
        "EY" => set_vowel(&mut gesture, 0.38, 0.46, 0.78, 0.18),
        "EH" => set_vowel(&mut gesture, 0.51, 0.20, 0.62, 0.12),
        "AE" => set_vowel(&mut gesture, 0.75, -0.45, 0.68, 0.05),
        "AA" => set_vowel(&mut gesture, 0.82, -0.62, -0.55, 0.05),
        "AO" => set_vowel(&mut gesture, 0.68, -0.28, -0.62, 0.62),
        "OW" => set_vowel(&mut gesture, 0.48, 0.18, -0.58, 0.78),
        "UH" => set_vowel(&mut gesture, 0.42, 0.42, -0.50, 0.55),
        "UW" => set_vowel(&mut gesture, 0.28, 0.72, -0.76, 0.92),
        "AH" | "AX" => set_vowel(&mut gesture, 0.58, -0.10, -0.08, 0.12),
        "ER" => set_vowel(&mut gesture, 0.44, 0.18, 0.22, 0.36),
        "M" => set_consonant(
            &mut gesture,
            ArticulationPlace::Labial,
            ConstrictionManner::Nasal,
            true,
        ),
        "N" => set_consonant(
            &mut gesture,
            ArticulationPlace::Alveolar,
            ConstrictionManner::Nasal,
            true,
        ),
        "NG" => set_consonant(
            &mut gesture,
            ArticulationPlace::Velar,
            ConstrictionManner::Nasal,
            true,
        ),
        "P" => set_consonant(
            &mut gesture,
            ArticulationPlace::Labial,
            ConstrictionManner::Stop,
            false,
        ),
        "B" => set_consonant(
            &mut gesture,
            ArticulationPlace::Labial,
            ConstrictionManner::Stop,
            true,
        ),
        "T" => set_consonant(
            &mut gesture,
            ArticulationPlace::Alveolar,
            ConstrictionManner::Stop,
            false,
        ),
        "D" => set_consonant(
            &mut gesture,
            ArticulationPlace::Alveolar,
            ConstrictionManner::Stop,
            true,
        ),
        "K" => set_consonant(
            &mut gesture,
            ArticulationPlace::Velar,
            ConstrictionManner::Stop,
            false,
        ),
        "G" => set_consonant(
            &mut gesture,
            ArticulationPlace::Velar,
            ConstrictionManner::Stop,
            true,
        ),
        "F" => set_consonant(
            &mut gesture,
            ArticulationPlace::Labial,
            ConstrictionManner::Fricative,
            false,
        ),
        "V" => set_consonant(
            &mut gesture,
            ArticulationPlace::Labial,
            ConstrictionManner::Fricative,
            true,
        ),
        "TH" => set_consonant(
            &mut gesture,
            ArticulationPlace::Dental,
            ConstrictionManner::Fricative,
            false,
        ),
        "DH" => set_consonant(
            &mut gesture,
            ArticulationPlace::Dental,
            ConstrictionManner::Fricative,
            true,
        ),
        "S" => set_consonant(
            &mut gesture,
            ArticulationPlace::Alveolar,
            ConstrictionManner::Fricative,
            false,
        ),
        "Z" => set_consonant(
            &mut gesture,
            ArticulationPlace::Alveolar,
            ConstrictionManner::Fricative,
            true,
        ),
        "SH" => set_consonant(
            &mut gesture,
            ArticulationPlace::Postalveolar,
            ConstrictionManner::Fricative,
            false,
        ),
        "ZH" => set_consonant(
            &mut gesture,
            ArticulationPlace::Postalveolar,
            ConstrictionManner::Fricative,
            true,
        ),
        "CH" => set_consonant(
            &mut gesture,
            ArticulationPlace::Postalveolar,
            ConstrictionManner::Affricate,
            false,
        ),
        "JH" => set_consonant(
            &mut gesture,
            ArticulationPlace::Postalveolar,
            ConstrictionManner::Affricate,
            true,
        ),
        "L" | "R" => set_consonant(
            &mut gesture,
            ArticulationPlace::Alveolar,
            ConstrictionManner::Approximant,
            true,
        ),
        "Y" => set_consonant(
            &mut gesture,
            ArticulationPlace::Palatal,
            ConstrictionManner::Approximant,
            true,
        ),
        "W" => {
            set_consonant(
                &mut gesture,
                ArticulationPlace::Labial,
                ConstrictionManner::Approximant,
                true,
            );
            gesture.lip_protrusion = UnitInterval::new(0.85);
        }
        "HH" => set_consonant(
            &mut gesture,
            ArticulationPlace::Glottal,
            ConstrictionManner::Fricative,
            false,
        ),
        _ => {
            gesture.energy = UnitInterval::new(0.0);
            gesture.glottal_adduction = UnitInterval::new(0.0);
        }
    }
    gesture
}

fn set_vowel(gesture: &mut GestureFrame, jaw: f32, height: f32, frontness: f32, rounding: f32) {
    gesture.jaw_aperture = UnitInterval::new(jaw);
    gesture.tongue_body_height = SignedUnit::new(height);
    gesture.tongue_body_frontness = SignedUnit::new(frontness);
    gesture.lip_aperture = UnitInterval::new((0.18 + 0.82 * jaw).clamp(0.0, 1.0));
    gesture.lip_protrusion = UnitInterval::new(rounding);
    gesture.constriction_manner = ConstrictionManner::Open;
    gesture.target_place = ArticulationPlace::None;
}

fn set_consonant(
    gesture: &mut GestureFrame,
    place: ArticulationPlace,
    manner: ConstrictionManner,
    voiced: bool,
) {
    gesture.target_place = place;
    gesture.constriction_manner = manner;
    gesture.tongue_tip_location = SignedUnit::new(2.0 * place.normalized_location() - 1.0);
    gesture.tongue_tip_constriction = UnitInterval::new(match manner {
        ConstrictionManner::Stop | ConstrictionManner::Affricate => 1.0,
        ConstrictionManner::Fricative => 0.82,
        ConstrictionManner::Approximant => 0.45,
        ConstrictionManner::Nasal => 0.95,
        _ => 0.0,
    });
    gesture.velum_opening = UnitInterval::new(if matches!(manner, ConstrictionManner::Nasal) {
        1.0
    } else {
        0.0
    });
    gesture.glottal_adduction = UnitInterval::new(if voiced { 0.78 } else { 0.08 });
}

pub struct FunctionalSingingEngine {
    pipeline: VocalTractPipeline,
    articulatory_head: ArticulatoryProjection,
    identity: FunctionalVoiceIdentity,
    anatomy: IdentityAnatomy,
    physiology: IdentityPhysiology,
    renderer_kind: FunctionalRendererKind,
    residual_detail: Option<ResidualDetailModel>,
    sample_rate: u32,
}

impl FunctionalSingingEngine {
    pub fn new(sample_rate: u32) -> Self {
        Self::with_identity(sample_rate, FunctionalVoiceIdentity::muse())
    }

    /// Construct a singer from a compact, procedural physiology. This keeps
    /// vocal identity separate from linguistic/motor control and makes it
    /// possible to audition voices without cloning a recorded person.
    pub fn with_identity(sample_rate: u32, identity: FunctionalVoiceIdentity) -> Self {
        let anatomy = IdentityAnatomy::procedural(
            "LegacyProcedural",
            identity.tract_length_cm,
            identity.tract_length_cm * 0.74,
            identity.pharynx_scale,
            identity.oral_scale,
            identity.lip_area_cm2 / 1.35,
        );
        Self::with_embodiment(
            sample_rate,
            identity,
            anatomy,
            IdentityPhysiology::default(),
            FunctionalRendererKind::Kl24BaselineV1,
        )
    }

    pub fn new_v2(sample_rate: u32) -> Self {
        Self::with_embodiment(
            sample_rate,
            FunctionalVoiceIdentity::muse(),
            IdentityAnatomy::velvet(),
            IdentityPhysiology::default(),
            FunctionalRendererKind::BranchedWaveguideV2,
        )
    }

    pub fn with_embodiment(
        sample_rate: u32,
        identity: FunctionalVoiceIdentity,
        anatomy: IdentityAnatomy,
        physiology: IdentityPhysiology,
        renderer_kind: FunctionalRendererKind,
    ) -> Self {
        let genesis = GenesisSeed::from_phrase("symthaea-functional-singing-v1");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let database = FormantDatabase::new();
        for phoneme in database.all_phonemes() {
            if let Some(target) = database.lookup(&phoneme) {
                pipeline.register_phoneme_manner(&phoneme, target.manner);
                pipeline.register_phoneme_voicing(&phoneme, target.is_voiced);
            }
        }
        // Fit the physical output head directly. The legacy acoustic head is
        // intentionally left unrefined because its formants are not consumed
        // by this backend; pitch, energy and source metadata are overridden by
        // the performance trajectory below.
        let mut articulatory_head = ArticulatoryProjection::neutral(&identity);
        let mut latents = Vec::new();
        let mut physical_targets = Vec::new();
        for phoneme in database.all_phonemes() {
            if let Some(target) = database.lookup(&phoneme) {
                latents.push(pipeline.controller.phoneme_latent(&genesis, &phoneme, 20));
                let acoustic = FormantFrame::from_target(
                    target,
                    220.0,
                    if target.is_vowel { 0.8 } else { 0.5 },
                    0.0,
                );
                physical_targets.push(ArticulatoryState::from_formant_frame(&acoustic, &identity));
            }
        }
        let _fitted = articulatory_head.fit(&latents, &physical_targets, 0.01);
        pipeline.reset();
        Self {
            pipeline,
            articulatory_head,
            identity,
            anatomy,
            physiology,
            renderer_kind,
            residual_detail: None,
            sample_rate,
        }
    }

    pub fn identity(&self) -> &FunctionalVoiceIdentity {
        &self.identity
    }

    pub fn anatomy(&self) -> &IdentityAnatomy {
        &self.anatomy
    }

    pub fn renderer_kind(&self) -> FunctionalRendererKind {
        self.renderer_kind
    }

    /// Install a bounded detail-only model. Invalid or overly powerful models
    /// are rejected by the domain-level safety constraints.
    pub fn set_residual_detail(&mut self, model: Option<ResidualDetailModel>) -> Result<()> {
        if let Some(candidate) = &model {
            candidate
                .validate()
                .map_err(|error| anyhow::anyhow!(error))?;
        }
        self.residual_detail = model;
        Ok(())
    }

    /// Generate explicit tract states directly from the post-CfC latent
    /// trajectory. The formant head supplies pitch/source metadata but is not
    /// inverted to create the live area function.
    pub fn motor_states(
        &mut self,
        performance: &VocalPerformance,
    ) -> Result<Vec<ArticulatoryState>> {
        performance.validate()?;
        self.pipeline.reset();
        let spans = phone_spans(performance);
        let pitch: Vec<_> = performance
            .pitch_curve
            .iter()
            .map(|point| (point.time_s, point.frequency_hz))
            .collect();
        let duration = performance.duration_s();
        let count = (duration * FUNCTIONAL_MOTOR_RATE).ceil() as usize;
        let dt = 1.0 / FUNCTIONAL_MOTOR_RATE;
        Ok((0..count)
            .map(|index| {
                let time = index as f32 * dt;
                let active = spans
                    .iter()
                    .find(|span| time >= span.start && time < span.end);
                let energy = expression(performance, time, |point| point.energy);
                let breathiness = expression(performance, time, |point| point.breathiness);
                let tension = expression(performance, time, |point| point.tension);
                let cognitive = VoiceCognitiveState {
                    emotional_valence: 0.35 + energy * 0.35,
                    emotional_arousal: (0.25 + tension * 0.55).clamp(0.0, 1.0),
                    consciousness_level: 0.82,
                    articulation_quality: 0.92,
                    rate_stability: 0.92,
                    integrated_phi: 0.75,
                    ..Default::default()
                };
                let mut frame = self.pipeline.tick_phoneme(
                    &cognitive,
                    None,
                    dt,
                    active.map(|span| span.phoneme.as_str()),
                );
                if let Some(span) = active {
                    let syllable = &performance.syllables[span.syllable];
                    let mut f0 = interpolate_curve(&pitch, time, syllable.note.frequency);
                    let syllable_duration =
                        (syllable.end_time_s() - syllable.note.start_time).max(0.001);
                    let progress =
                        ((time - syllable.note.start_time) / syllable_duration).clamp(0.0, 1.0);
                    if progress >= syllable.vibrato_onset && syllable.vibrato_depth_cents > 0.0 {
                        let cents = syllable.vibrato_depth_cents
                            * (std::f32::consts::TAU * syllable.vibrato_rate_hz * time).sin();
                        f0 *= 2.0f32.powf(cents / 1200.0);
                    }
                    frame.f0 = f0;
                    frame.energy = (energy * syllable.energy).sqrt().clamp(0.0, 1.0);
                    frame.voicing *= 1.0 - breathiness * 0.08;
                } else {
                    frame.energy = 0.0;
                    frame.voicing = 0.0;
                    frame.source_type = SourceType::Silent;
                }
                frame.time = time;
                let latent = self.pipeline.controller.latent_output();
                self.articulatory_head.project(&latent, &frame)
            })
            .collect())
    }

    /// Cacheable, intention-relative motor trajectory. Unlike the legacy
    /// `motor_states`, this contains no absolute tube geometry.
    pub fn gesture_frames(&mut self, performance: &VocalPerformance) -> Result<Vec<GestureFrame>> {
        let states = self.motor_states(performance)?;
        let spans = phone_spans(performance);
        Ok(states
            .iter()
            .enumerate()
            .map(|(index, state)| {
                let time = index as f32 / FUNCTIONAL_MOTOR_RATE;
                let phoneme = spans
                    .iter()
                    .find(|span| time >= span.start && time < span.end)
                    .map(|span| span.phoneme.as_str());
                gesture_for_phoneme(phoneme, state, time)
            })
            .collect())
    }

    pub fn physical_frames(
        &mut self,
        performance: &VocalPerformance,
    ) -> Result<Vec<symthaea_vocal_tract::PhysicalTractFrame>> {
        let gestures = self.gesture_frames(performance)?;
        let mut planner = GesturePlanner::default();
        planner
            .realize_sequence(
                &gestures,
                &self.anatomy,
                &self.physiology,
                1.0 / FUNCTIONAL_MOTOR_RATE,
            )
            .map_err(|error| anyhow::anyhow!(error))
    }
}

impl Default for FunctionalSingingEngine {
    fn default() -> Self {
        Self::new(48_000)
    }
}

impl SingingVoiceEngine for FunctionalSingingEngine {
    fn id(&self) -> &str {
        match self.renderer_kind {
            FunctionalRendererKind::Kl24BaselineV1 => "functional-tract-kl24-baseline-v1",
            FunctionalRendererKind::BranchedWaveguideV2 => "functional-tract-branched-v2",
            FunctionalRendererKind::TransmissionLineReference => "functional-tract-reference-v1",
        }
    }

    fn render(&mut self, performance: &VocalPerformance) -> Result<VocalStem> {
        let dry = match self.renderer_kind {
            FunctionalRendererKind::Kl24BaselineV1 => {
                let states = self.motor_states(performance)?;
                let mut renderer = FunctionalTractRenderer::new(
                    self.identity.clone(),
                    symthaea_vocal_tract::FunctionalTractConfig {
                        sample_rate: self.sample_rate,
                        ..Default::default()
                    },
                );
                renderer.synthesize_states(&states, FUNCTIONAL_MOTOR_RATE)
            }
            FunctionalRendererKind::BranchedWaveguideV2 => {
                let frames = self.physical_frames(performance)?;
                let mut renderer = BranchedWaveguideV2::new(BranchedWaveguideConfig {
                    output_sample_rate: self.sample_rate,
                    ..Default::default()
                });
                renderer
                    .render_frames(&frames, FUNCTIONAL_MOTOR_RATE)
                    .map_err(|error| anyhow::anyhow!(error))?
                    .final_output
            }
            FunctionalRendererKind::TransmissionLineReference => {
                let frames = self.physical_frames(performance)?;
                let mut renderer = TransmissionLineReference::new(self.sample_rate);
                renderer
                    .render_frames(&frames, FUNCTIONAL_MOTOR_RATE)
                    .map_err(|error| anyhow::anyhow!(error))?
                    .final_output
            }
        };
        let samples = if let Some(model) = &self.residual_detail {
            model
                .process(&dry)
                .map_err(|error| anyhow::anyhow!(error))?
        } else {
            dry
        };
        let stem = VocalStem {
            samples,
            sample_rate: self.sample_rate,
            backend: self.id().to_string(),
        };
        stem.validate()?;
        Ok(stem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_muse::Note;

    #[test]
    fn functional_engine_renders_a_procedural_voice() {
        let performance = VocalPerformance::from_melody(
            "light",
            &[Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.16,
                velocity: 0.72,
            }],
            "en",
        )
        .unwrap();
        let mut engine = FunctionalSingingEngine::new(24_000);
        let stem = engine.render(&performance).unwrap();
        assert_eq!(stem.backend, "functional-tract-hdc-cfc");
        assert!(stem.samples.iter().any(|sample| sample.abs() > 1e-6));
        assert!(stem.samples.iter().all(|sample| sample.is_finite()));
    }
}
