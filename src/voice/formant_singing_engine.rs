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
    CapabilitySupport, ControlCapability, ExpressionPoint, SingingBackendCapabilities,
    SingingControl, SingingRenderPolicy, SingingVoiceEngine, VocalPerformance, VocalStem,
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
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "syllable {syllable_index} has melisma notes but no vowel nucleus"
                        )
                    })?;
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

/// Replace Muse's fixed vibrato with the per-note controls frozen in VocalPerformance.
fn apply_note_pitch_controls(
    frames: &mut [FormantFrame],
    performance: &VocalPerformance,
) {
    let mut previous_frequency: Option<f32> = None;
    let mut previous_end = 0.0f32;
    for syllable in &performance.syllables {
        for note in syllable.notes() {
            let note_start = note.start_time;
            let note_end = note.start_time + note.duration;
            let gap_before = note_start - previous_end;
            let from_frequency = if gap_before > FRAME_DT_S * 1.5 {
                note.frequency
            } else {
                previous_frequency.unwrap_or(note.frequency)
            };
            let transition_s = note.duration * syllable.portamento.clamp(0.0, 1.0);
            let vibrato_start = note_start + note.duration * syllable.vibrato_onset.clamp(0.0, 1.0);
            let vibrato_ramp_s = 0.05f32;
            let edge_ramp_s = 0.015f32;

            for frame in frames.iter_mut().filter(|frame| {
                frame.time >= note_start && frame.time < note_end
            }) {
                let elapsed = (frame.time - note_start).max(0.0);
                let transition = if transition_s > f32::EPSILON {
                    (elapsed / transition_s).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                // Interpolate in log-frequency so the transition is uniform in cents.
                let pitch = if from_frequency > 0.0 && note.frequency > 0.0 {
                    from_frequency
                        * (note.frequency / from_frequency).powf(transition)
                } else {
                    note.frequency
                };

                let attack = (elapsed / edge_ramp_s).clamp(0.0, 1.0);
                let release = ((note_end - frame.time) / edge_ramp_s).clamp(0.0, 1.0);
                let envelope = attack.min(release);
                let vibrato_gate = if syllable.vibrato_depth_cents > 0.0
                    && syllable.vibrato_rate_hz > 0.0
                    && frame.time >= vibrato_start
                {
                    let onset_gate = ((frame.time - vibrato_start) / vibrato_ramp_s).clamp(0.0, 1.0);
                    onset_gate * release
                } else {
                    0.0
                };
                let phase = std::f32::consts::TAU
                    * syllable.vibrato_rate_hz
                    * (frame.time - vibrato_start).max(0.0);
                let vibrato_cents =
                    syllable.vibrato_depth_cents * phase.sin() * vibrato_gate;
                let vibrato_ratio = 2.0f32.powf(vibrato_cents / 1200.0);
                frame.f0 = pitch * envelope * vibrato_ratio;
            }

            previous_frequency = Some(note.frequency);
            previous_end = note_end;
        }
    }
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

/// Interpolate a phrase-level expression target at an absolute performance time.
fn expression_at(curve: &[ExpressionPoint], time_s: f32) -> Option<ExpressionPoint> {
    let first = *curve.first()?;
    let last = *curve.last()?;
    if time_s <= first.time_s {
        return Some(first);
    }
    if time_s >= last.time_s {
        return Some(last);
    }
    let right = curve.partition_point(|point| point.time_s < time_s);
    let left = right.saturating_sub(1);
    let from = curve[left];
    let to = curve[right];
    let span = (to.time_s - from.time_s).max(f32::EPSILON);
    let t = ((time_s - from.time_s) / span).clamp(0.0, 1.0);
    Some(ExpressionPoint {
        time_s,
        energy: from.energy + (to.energy - from.energy) * t,
        breathiness: from.breathiness + (to.breathiness - from.breathiness) * t,
        tension: from.tension + (to.tension - from.tension) * t,
    })
}

/// Apply the controls that the legacy source-filter frame can express.
///
/// Energy is applied as a ratio against the phrase's mean syllable energy.
/// Breathiness is approximated by lowering periodic voicing and slightly
/// widening resonances. Tension is approximated by narrowing bandwidths.
/// This does not change glottal Rd, create a separate aspiration envelope,
/// or synthesize phrase inhalations; those remain explicit capability limits.
fn apply_phrase_expression(frames: &mut [FormantFrame], performance: &VocalPerformance) {
    for phrase in &performance.phrases {
        let phrase_syllables = &performance.syllables[phrase.syllable_start..phrase.syllable_end];
        let Some(first) = phrase_syllables.first() else {
            continue;
        };
        let Some(last) = phrase_syllables.last() else {
            continue;
        };
        let start_s = first.note.start_time;
        let end_s = last.end_time_s();
        let base_energy = phrase_syllables
            .iter()
            .map(|syllable| syllable.energy)
            .sum::<f32>()
            / phrase_syllables.len() as f32;

        for frame in frames
            .iter_mut()
            .filter(|frame| frame.time >= start_s && frame.time <= end_s)
        {
            let Some(expression) = expression_at(&phrase.expression_curve, frame.time) else {
                continue;
            };
            if base_energy > f32::EPSILON {
                frame.energy = (frame.energy * (expression.energy / base_energy)).clamp(0.0, 1.0);
            }
            let breathiness = expression.breathiness.clamp(0.0, 1.0);
            let tension = expression.tension.clamp(0.0, 1.0);
            if matches!(
                frame.source_type,
                SourceType::Vowel | SourceType::Liquid | SourceType::Nasal
            ) {
                frame.voicing = (frame.voicing * (1.0 - 0.25 * breathiness)).clamp(0.0, 1.0);
            }
            let breath_bandwidth = 1.0 + 0.08 * breathiness;
            let tension_bandwidth = 1.2 - 0.4 * tension;
            let bandwidth_scale = breath_bandwidth * tension_bandwidth;
            frame.b1 = (frame.b1 * bandwidth_scale).max(20.0);
            frame.b2 = (frame.b2 * bandwidth_scale).max(20.0);
            frame.b3 = (frame.b3 * bandwidth_scale).max(20.0);
        }
    }
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
                    CapabilitySupport::Approximated,
                    "energy scales frame amplitude, breathiness lowers periodic voicing and widens resonances, and tension narrows resonator bandwidths; no separate glottal-Rd or breath-event model is applied",
                ),
                capability(
                    SingingControl::PhraseBreathTiming,
                    CapabilitySupport::Unsupported,
                    "phrase breath durations are not synthesized; only score-note timing gaps become silent frames",
                ),
                capability(
                    SingingControl::Breathiness,
                    CapabilitySupport::Approximated,
                    "phrase breathiness reduces periodic voicing and widens formant bandwidths; aspiration level itself is not independently controlled",
                ),
                capability(
                    SingingControl::Energy,
                    CapabilitySupport::Native,
                    "note velocity controls the frame energy envelope",
                ),
                capability(
                    SingingControl::Tension,
                    CapabilitySupport::Approximated,
                    "phrase tension narrows formant bandwidths; vocal-fold tension and glottal source shape are not modeled",
                ),
                capability(
                    SingingControl::Vibrato,
                    CapabilitySupport::Native,
                    "per-syllable vibrato onset, rate, and depth drive the F0 curve with attack/release gating",
                ),
                capability(
                    SingingControl::Portamento,
                    CapabilitySupport::Native,
                    "per-syllable portamento fraction drives a log-frequency glide from the preceding contiguous note",
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
        let mut raw = raw;
        apply_note_pitch_controls(&mut raw, performance);
        let mut frames = complete_timeline(&raw, performance.duration_s())?;
        apply_phrase_expression(&mut frames, performance);
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
            CapabilitySupport::Approximated
        );
        assert_eq!(
            capabilities.support(SingingControl::ConsonantAnticipation),
            CapabilitySupport::Unsupported
        );
        assert_eq!(
            capabilities.support(SingingControl::Melisma),
            CapabilitySupport::Approximated
        );
        assert_eq!(
            capabilities.support(SingingControl::VoiceIdentity),
            CapabilitySupport::Unsupported
        );
        assert_eq!(
            capabilities.support(SingingControl::PhraseExpressionCurve),
            CapabilitySupport::Approximated
        );
        assert_eq!(
            capabilities.support(SingingControl::Breathiness),
            CapabilitySupport::Approximated
        );
        assert_eq!(
            capabilities.support(SingingControl::Tension),
            CapabilitySupport::Approximated
        );
        assert_eq!(
            capabilities.support(SingingControl::PhraseBreathTiming),
            CapabilitySupport::Unsupported
        );
        SingingRenderPolicy::phrase_expression_comparison()
            .validate_against(&capabilities)
            .unwrap();
        assert!(
            SingingRenderPolicy::phrase_breath_comparison()
                .validate_against(&capabilities)
                .is_err()
        );
    }

    #[test]
    fn score_pitch_and_velocity_have_measurable_frame_level_responses() {
        let mut engine = FormantSingingEngine::default();
        let performance_for = |frequency: f32, velocity: f32| {
            VocalPerformance::from_melody(
                "light",
                &[Note {
                    frequency,
                    start_time: 0.0,
                    duration: 0.6,
                    velocity,
                }],
                "en",
            )
            .unwrap()
        };

        let low = performance_for(220.0, 0.2);
        let high = performance_for(440.0, 0.8);
        let low_aligned = engine.aligned_syllables(&low).unwrap();
        let high_aligned = engine.aligned_syllables(&high).unwrap();
        let low_frames = render_sung_frames(&low_aligned, FRAME_RATE_HZ);
        let high_frames = render_sung_frames(&high_aligned, FRAME_RATE_HZ);

        let mean_middle = |frames: &[FormantFrame], field: fn(&FormantFrame) -> f32| {
            let values: Vec<_> = frames
                .iter()
                .filter(|frame| (0.15..=0.45).contains(&frame.time))
                .map(field)
                .collect();
            assert!(!values.is_empty(), "middle steady-state frames must exist");
            values.iter().sum::<f32>() / values.len() as f32
        };
        let low_pitch = mean_middle(&low_frames, |frame| frame.f0);
        let high_pitch = mean_middle(&high_frames, |frame| frame.f0);
        let low_energy = mean_middle(&low_frames, |frame| frame.energy);
        let high_energy = mean_middle(&high_frames, |frame| frame.energy);

        assert!(
            high_pitch > low_pitch * 1.8,
            "doubling the score frequency must measurably raise F0: low={low_pitch}, high={high_pitch}"
        );
        assert!(
            high_energy > low_energy * 1.8,
            "increasing note velocity must measurably raise the frame energy: low={low_energy}, high={high_energy}"
        );
    }

    #[test]
    fn phrase_expression_modulates_formant_energy_voicing_and_bandwidth() {
        let note = note(0.0, 0.5, 220.0);
        let mut performance = VocalPerformance::from_melody("light", &[note], "en").unwrap();
        performance.phrases[0].expression_curve = vec![
            ExpressionPoint {
                time_s: 0.0,
                energy: 0.35,
                breathiness: 0.8,
                tension: 0.9,
            },
            ExpressionPoint {
                time_s: 0.5,
                energy: 0.35,
                breathiness: 0.8,
                tension: 0.9,
            },
        ];
        let mut frames = vec![FormantFrame {
            f1: 500.0,
            f2: 1500.0,
            f3: 2500.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 220.0,
            energy: 0.8,
            voicing: 0.9,
            time: 0.25,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        }];

        apply_phrase_expression(&mut frames, &performance);
        assert!((frames[0].energy - 0.4).abs() < 1e-5);
        assert!((frames[0].voicing - 0.72).abs() < 1e-5);
        let bandwidth_scale = (1.0 + 0.08 * 0.8) * (1.2 - 0.4 * 0.9);
        assert!((frames[0].b1 - 60.0 * bandwidth_scale).abs() < 1e-4);
        assert!((frames[0].b2 - 90.0 * bandwidth_scale).abs() < 1e-4);
    }

    #[test]
    fn per_syllable_vibrato_onset_rate_and_depth_control_the_rendered_f0() {
        let note = note(0.0, 1.0, 220.0);
        let mut performance = VocalPerformance::from_melody("light", &[note], "en").unwrap();
        performance.syllables[0].portamento = 0.0;
        performance.syllables[0].vibrato_onset = 0.25;
        performance.syllables[0].vibrato_rate_hz = 5.0;
        performance.syllables[0].vibrato_depth_cents = 20.0;

        let aligned = FormantSingingEngine::default()
            .aligned_syllables(&performance)
            .unwrap();
        let mut frames = render_sung_frames(&aligned, FRAME_RATE_HZ);
        apply_note_pitch_controls(&mut frames, &performance);

        let before_onset: Vec<_> = frames
            .iter()
            .filter(|frame| frame.time >= 0.10 && frame.time <= 0.20)
            .map(|frame| frame.f0)
            .collect();
        assert!(!before_onset.is_empty());
        let before_spread = before_onset.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - before_onset.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            before_spread < 0.1,
            "vibrato must remain inactive before its onset: spread={before_spread}"
        );

        let after_onset: Vec<_> = frames
            .iter()
            .filter(|frame| frame.time >= 0.45 && frame.time <= 0.90)
            .map(|frame| frame.f0)
            .collect();
        assert!(!after_onset.is_empty());
        let after_spread = after_onset.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - after_onset.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            after_spread > 3.0,
            "configured vibrato depth must measurably modulate F0: spread={after_spread}"
        );
        let mean = after_onset.iter().sum::<f32>() / after_onset.len() as f32;
        assert!((mean - 220.0).abs() < 2.0, "vibrato should remain centered on the score pitch: {mean}");
    }

    #[test]
    fn per_syllable_portamento_glides_in_log_frequency_and_reaches_target() {
        let notes = [note(0.0, 0.5, 220.0), note(0.5, 0.5, 440.0)];
        let mut performance =
            VocalPerformance::from_melody("light bright", &notes, "en").unwrap();
        performance.syllables[0].vibrato_depth_cents = 0.0;
        performance.syllables[1].vibrato_depth_cents = 0.0;
        performance.syllables[1].vibrato_onset = 1.0;
        performance.syllables[1].portamento = 0.5;

        let aligned = FormantSingingEngine::default()
            .aligned_syllables(&performance)
            .unwrap();
        let mut frames = render_sung_frames(&aligned, FRAME_RATE_HZ);
        apply_note_pitch_controls(&mut frames, &performance);

        let at_transition = frames
            .iter()
            .find(|frame| (frame.time - 0.60).abs() < 0.002)
            .expect("frame at 100ms into portamento");
        let expected = 220.0 * 2.0f32.powf(0.4);
        assert!(
            (at_transition.f0 - expected).abs() < 3.0,
            "log-frequency portamento at 20% of glide should be {expected}, got {}",
            at_transition.f0
        );
        let at_target = frames
            .iter()
            .find(|frame| (frame.time - 0.76).abs() < 0.002)
            .expect("frame after portamento completes");
        assert!(
            (at_target.f0 - 440.0).abs() < 3.0,
            "portamento must reach the target note: {}",
            at_target.f0
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
