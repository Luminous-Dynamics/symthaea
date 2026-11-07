// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Score-conditioned adapter for the existing Muse formant-frame + FormantVocoder
//! path. This is a conventional source-filter baseline, not the unintegrated
//! physical waveguide/gesture prototype described in the physiology research.

use anyhow::{Result, bail};

use symthaea_muse::singing_bridge::{render_sung_frames, sing};
use symthaea_vocal_tract::types::{FormantFrame, SourceType};

use super::singing_engine::{
    CapabilitySupport, ControlCapability, SingingBackendCapabilities, SingingControl,
    SingingVoiceEngine, VocalPerformance, VocalStem,
};
use super::vocoder::{FormantVocoder, VocoderConfig};

const FRAME_RATE_HZ: f32 = 200.0;
const FRAME_DT_S: f32 = 1.0 / FRAME_RATE_HZ;
const BACKEND_ID: &str = "formant-vocoder-singing";

/// An honest, executable baseline over the already-integrated Muse
/// sing -> render_sung_frames -> FormantVocoder path.
///
/// It accepts only a VocalPerformance whose phones exactly match the current
/// English Muse G2P result. This prevents the renderer from silently singing
/// a different pronunciation than the frozen performance plan.
pub struct FormantSingingEngine {
    vocoder: FormantVocoder,
    sample_rate: u32,
}

impl Default for FormantSingingEngine {
    fn default() -> Self {
        Self::new(24_000).expect("24 kHz is a valid FormantVocoder sample rate")
    }
}

impl FormantSingingEngine {
    pub fn new(sample_rate: u32) -> Result<Self> {
        if sample_rate < 8_000 || sample_rate % FRAME_RATE_HZ as u32 != 0 {
            bail!(
                "FormantSingingEngine sample rate must be >= 8 kHz and divisible by {FRAME_RATE_HZ} Hz; got {sample_rate}"
            );
        }
        let config = VocoderConfig {
            sample_rate,
            ..VocoderConfig::default()
        };
        Ok(Self {
            vocoder: FormantVocoder::with_config(config),
            sample_rate,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn aligned_syllables(
        &self,
        performance: &VocalPerformance,
    ) -> Result<Vec<symthaea_muse::singing_bridge::SungSyllable>> {
        performance.validate()?;
        if performance.language != "en" {
            bail!(
                "FormantSingingEngine supports the current English Muse G2P frontend only; got {:?}",
                performance.language
            );
        }

        let notes: Vec<_> = performance
            .syllables
            .iter()
            .flat_map(|syllable| syllable.notes().copied())
            .collect();
        if notes.is_empty() {
            bail!("formant singing requires at least one score note");
        }

        let mut previous_end_s = 0.0f32;
        for (index, note) in notes.iter().enumerate() {
            let end_s = note.start_time + note.duration;
            if note.start_time < previous_end_s - 0.000_01 {
                bail!("score note {index} overlaps or reorders a preceding note");
            }
            previous_end_s = end_s;
        }

        let aligned = sing(&performance.lyrics, &notes);
        if aligned.len() != notes.len() {
            bail!(
                "Muse G2P alignment returned {} syllable/note events for {} score notes",
                aligned.len(),
                notes.len()
            );
        }

        let mut cursor = 0usize;
        for (syllable_index, expected) in performance.syllables.iter().enumerate() {
            let observed = &aligned[cursor];
            if !same_phones(&expected.phonemes, &observed.phonemes) {
                bail!(
                    "formant renderer G2P phone sequence differs from frozen VocalPerformance at syllable {syllable_index}"
                );
            }
            if !same_note(&expected.note, &observed.note) {
                bail!("formant renderer note binding differs at syllable {syllable_index}");
            }
            cursor += 1;

            if !expected.melisma_notes.is_empty() {
                let sustained = expected
                    .phonemes
                    .iter()
                    .rev()
                    .find(|phone| phone.is_vowel)
                    .ok_or_else(|| anyhow::anyhow!(
                        "syllable {syllable_index} has melisma notes but no vowel nucleus"
                    ))?;
                for (melisma_index, note) in expected.melisma_notes.iter().enumerate() {
                    let observed_melisma = &aligned[cursor];
                    if observed_melisma.phonemes.len() != 1
                        || observed_melisma.phonemes[0].ipa != sustained.ipa
                        || !observed_melisma.phonemes[0].is_vowel
                    {
                        bail!(
                            "G2P melisma event {melisma_index} in syllable {syllable_index} does not sustain its designated vowel"
                        );
                    }
                    if !same_note(note, &observed_melisma.note) {
                        bail!(
                            "formant renderer melisma note binding differs in syllable {syllable_index}"
                        );
                    }
                    cursor += 1;
                }
            }
        }
        if cursor != aligned.len() {
            bail!("formant renderer left unverified G2P note events");
        }
        Ok(aligned)
    }
}

fn same_note(left: &symthaea_muse::Note, right: &symthaea_muse::Note) -> bool {
    left.frequency.to_bits() == right.frequency.to_bits()
        && left.start_time.to_bits() == right.start_time.to_bits()
        && left.duration.to_bits() == right.duration.to_bits()
        && left.velocity.to_bits() == right.velocity.to_bits()
}

fn same_phones(
    expected: &[super::singing_engine::VocalPhoneme],
    observed: &[symthaea_vocal_tract::speech::g2p::Phoneme],
) -> bool {
    expected.len() == observed.len()
        && expected.iter().zip(observed).all(|(expected, observed)| {
            expected.ipa == observed.ipa
                && expected.is_vowel == observed.is_vowel
                && expected.stress == observed.stress
        })
}

/// Put irregularly timed formant frames on a fixed 200 Hz timeline. Long gaps
/// become explicit silent frames; nearby frames are interpolated. FormantVocoder
/// consumes uniformly sampled frames, so leaving gaps implicit would compress
/// musical rests into the average frame interval.
fn complete_timeline(raw: &[FormantFrame], duration_s: f32) -> Result<Vec<FormantFrame>> {
    if raw.is_empty() || !duration_s.is_finite() || duration_s <= 0.0 {
        bail!("cannot complete an empty or invalid formant timeline");
    }
    for (index, frame) in raw.iter().enumerate() {
        if !frame.time.is_finite()
            || frame.time < 0.0
            || frame.time > duration_s + FRAME_DT_S
            || (index > 0 && frame.time <= raw[index - 1].time)
        {
            bail!("raw formant frame {index} has invalid or non-monotonic timing");
        }
    }

    let intervals = (duration_s * FRAME_RATE_HZ).ceil().max(1.0) as usize;
    let frame_count = intervals + 1;
    let mut completed = Vec::with_capacity(frame_count);
    let mut right_index = 0usize;

    for frame_index in 0..frame_count {
        let time_s = frame_index as f32 * FRAME_DT_S;
        while right_index + 1 < raw.len() && raw[right_index + 1].time <= time_s {
            right_index += 1;
        }

        let frame = if time_s < raw[0].time - FRAME_DT_S * 0.5
            || time_s > raw[raw.len() - 1].time + FRAME_DT_S * 0.5
        {
            FormantFrame::silent(time_s)
        } else if right_index + 1 < raw.len() {
            let left = &raw[right_index];
            let right = &raw[right_index + 1];
            let gap_s = right.time - left.time;
            if gap_s > FRAME_DT_S * 1.5 {
                FormantFrame::silent(time_s)
            } else if gap_s <= f32::EPSILON {
                left.clone()
            } else {
                left.lerp(right, ((time_s - left.time) / gap_s).clamp(0.0, 1.0))
            }
        } else {
            raw[raw.len() - 1].clone()
        };
        let mut frame = frame;
        frame.time = time_s;
        completed.push(frame);
    }
    Ok(completed)
}

fn capability(
    control: SingingControl,
    support: CapabilitySupport,
    rationale: &str,
) -> ControlCapability {
    ControlCapability {
        control,
        support,
        rationale: rationale.to_string(),
    }
}

impl SingingVoiceEngine for FormantSingingEngine {
    fn id(&self) -> &str {
        BACKEND_ID
    }

    fn capabilities(&self) -> SingingBackendCapabilities {
        SingingBackendCapabilities::new(
            BACKEND_ID,
            vec![
                capability(
                    SingingControl::LyricText,
                    CapabilitySupport::Conditional,
                    "English-only Muse G2P; emitted phones must match the frozen performance",
                ),
                capability(
                    SingingControl::Language,
                    CapabilitySupport::Conditional,
                    "only language='en' is accepted",
                ),
                capability(
                    SingingControl::ExactPhoneSequence,
                    CapabilitySupport::Conditional,
                    "current Muse G2P is re-run solely to verify exact IPA, vowel and stress identity",
                ),
                capability(
                    SingingControl::SyllableAlignment,
                    CapabilitySupport::Conditional,
                    "uses Muse maximal-onset English syllabification; linguistic correctness is not guaranteed",
                ),
                capability(
                    SingingControl::NotePitch,
                    CapabilitySupport::Native,
                    "note frequency is assigned to the corresponding sung syllable",
                ),
                capability(
                    SingingControl::NoteDuration,
                    CapabilitySupport::Approximated,
                    "phoneme frames are allocated within each score-note duration on a 5 ms frame grid",
                ),
                capability(
                    SingingControl::AbsoluteOnsetAndRests,
                    CapabilitySupport::Approximated,
                    "absolute note times are quantized to a 5 ms frame grid with explicit silent frames",
                ),
                capability(
                    SingingControl::ConsonantAnticipation,
                    CapabilitySupport::Unsupported,
                    "VocalPerformance consonant_advance_s is not consumed by the legacy formant-frame bridge",
                ),
                capability(
                    SingingControl::Melisma,
                    CapabilitySupport::Approximated,
                    "one designated vowel is kept across extra notes, but note-envelope ramps can reveal note boundaries",
                ),
                capability(
                    SingingControl::ContinuousPitchCurve,
                    CapabilitySupport::Unsupported,
                    "renderer follows discrete note targets, not the arbitrary VocalPerformance pitch_curve",
                ),
                capability(
                    SingingControl::PhraseExpressionCurve,
                    CapabilitySupport::Unsupported,
                    "phrase expression points are not mapped into formant-frame controls",
                ),
                capability(
                    SingingControl::Breathiness,
                    CapabilitySupport::Unsupported,
                    "per-syllable breathiness is not represented by the legacy FormantFrame contract",
                ),
                capability(
                    SingingControl::Energy,
                    CapabilitySupport::Native,
                    "note velocity controls the frame energy envelope",
                ),
                capability(
                    SingingControl::Tension,
                    CapabilitySupport::Unsupported,
                    "tension controls are not consumed by the legacy formant renderer",
                ),
                capability(
                    SingingControl::Vibrato,
                    CapabilitySupport::Approximated,
                    "uses a fixed 5.5 Hz / 30 cent vibrato independent of per-syllable settings",
                ),
                capability(
                    SingingControl::Portamento,
                    CapabilitySupport::Unsupported,
                    "custom per-syllable portamento values are not consumed",
                ),
                capability(
                    SingingControl::Articulation,
                    CapabilitySupport::Approximated,
                    "articulatory manner derives from phoneme source type and fixed natural-duration heuristics",
                ),
                capability(
                    SingingControl::VoiceIdentity,
                    CapabilitySupport::Unsupported,
                    "uses the default source-filter voice; no identity reference or trained voice profile is accepted",
                ),
                capability(
                    SingingControl::LocalEdits,
                    CapabilitySupport::Unsupported,
                    "renderer synthesizes a whole performance and has no localized edit API",
                ),
            ],
        )
        .expect("static formant singing capability manifest is complete")
    }

    fn render(&mut self, performance: &VocalPerformance) -> Result<VocalStem> {
        let aligned = self.aligned_syllables(performance)?;
        let raw = render_sung_frames(&aligned, FRAME_RATE_HZ);
        let frames = complete_timeline(&raw, performance.duration_s())?;
        let samples = self.vocoder.synthesize(&frames);
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
    use crate::voice::singing_engine::Note;

    fn note(start_time: f32, duration: f32, frequency: f32) -> Note {
        Note {
            frequency,
            start_time,
            duration,
            velocity: 0.7,
        }
    }

    #[test]
    fn backend_declares_real_control_boundaries() {
        let engine = FormantSingingEngine::default();
        let capabilities = engine.capabilities();
        capabilities.validate().unwrap();
        assert_eq!(
            capabilities.support(SingingControl::AbsoluteOnsetAndRests),
            CapabilitySupport::Native
        );
        assert_eq!(
            capabilities.support(SingingControl::Melisma),
            CapabilitySupport::Approximated
        );
        assert_eq!(
            capabilities.support(SingingControl::VoiceIdentity),
            CapabilitySupport::Unsupported
        );
    }

    #[test]
    fn formant_engine_renders_the_frozen_g2p_plan_to_finite_audio() {
        let mut engine = FormantSingingEngine::default();
        let performance =
            VocalPerformance::from_melody("light", &[note(0.0, 0.5, 261.63)], "en").unwrap();
        let stem = engine.render(&performance).unwrap();
        assert_eq!(stem.sample_rate, 24_000);
        assert_eq!(stem.backend, BACKEND_ID);
        assert!(!stem.samples.is_empty());
        assert!(stem.samples.iter().all(|sample| sample.is_finite()));
        assert!(stem.samples.iter().any(|sample| sample.abs() > 1e-7));
    }

    #[test]
    fn formant_engine_preserves_an_initial_rest_on_the_uniform_timeline() {
        let raw = vec![
            FormantFrame {
                f1: 500.0,
                f2: 1500.0,
                f3: 2500.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 220.0,
                energy: 0.7,
                voicing: 1.0,
                time: 0.02,
                source_type: SourceType::Vowel,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            },
            FormantFrame {
                f1: 500.0,
                f2: 1500.0,
                f3: 2500.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 220.0,
                energy: 0.7,
                voicing: 1.0,
                time: 0.03,
                source_type: SourceType::Vowel,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            },
        ];
        let frames = complete_timeline(&raw, 0.03).unwrap();
        assert_eq!(frames.len(), 7);
        assert_eq!(frames[0].source_type, SourceType::Silent);
        assert_eq!(frames[1].source_type, SourceType::Silent);
        assert_eq!(frames[2].source_type, SourceType::Vowel);
        assert_eq!(frames[6].time, 0.03);
    }

    #[test]
    fn formant_engine_rejects_a_phone_plan_that_does_not_match_current_g2p() {
        let mut engine = FormantSingingEngine::default();
        let mut performance =
            VocalPerformance::from_melody("light", &[note(0.0, 0.5, 261.63)], "en").unwrap();
        performance.syllables[0].phonemes[0].ipa = "x".into();
        let error = engine.render(&performance).unwrap_err().to_string();
        assert!(error.contains("phone sequence differs"), "{error}");
    }

    #[test]
    fn formant_engine_rejects_overlapping_score_notes() {
        let mut engine = FormantSingingEngine::default();
        let mut performance = VocalPerformance::from_melody(
            "light bright",
            &[note(0.0, 0.5, 261.63), note(0.5, 0.5, 293.66)],
            "en",
        )
        .unwrap();
        performance.syllables[1].note.start_time = 0.4;
        let error = engine.render(&performance).unwrap_err().to_string();
        assert!(error.contains("overlaps"), "{error}");
    }
}
