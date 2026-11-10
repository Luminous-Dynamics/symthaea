// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Score-conditioned adapter for the existing Muse formant-frame + FormantVocoder
//! path. This is a conventional source-filter baseline, not the unintegrated
//! physical waveguide/gesture prototype described in the physiology research.

use anyhow::{Result, bail};

use symthaea_muse::singing_bridge::{render_sung_frames, sing};
use symthaea_pronunciation_plan::{
    BeatDomainScore, BeatScoreEvent, Phone, PhoneKind as ResolvedPhoneKind, PronunciationPlan,
    ScoreEventKind, SyllableNoteBinding, TempoChange, VocalPhraseSpan, VocalScoreAlignment,
};
use symthaea_vocal_tract::speech::formants::formant_target_pub;
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

    /// Capabilities of the explicit pronunciation-plan + score-graph API.
    ///
    /// These differ from `capabilities()`, which describes the legacy
    /// VocalPerformance v2 route. The resolved-plan route consumes selected
    /// phones and note events directly; it does not receive vibrato,
    /// portamento, phrase-expression or voice-identity controls.
    pub fn resolved_plan_capabilities() -> Result<SingingBackendCapabilities> {
        SingingBackendCapabilities::new(
            format!("{BACKEND_ID}-resolved-plan-v1"),
            vec![
                capability(
                    SingingControl::LyricText,
                    CapabilitySupport::Unsupported,
                    "the resolved-plan route consumes a selected pronunciation plan, not raw lyric text",
                ),
                capability(
                    SingingControl::Language,
                    CapabilitySupport::Conditional,
                    "only language='en' or 'en-US' is accepted by the calibrated legacy formant inventory",
                ),
                capability(
                    SingingControl::ExactPhoneSequence,
                    CapabilitySupport::Native,
                    "selected canonical phone tokens are rendered directly without rerunning G2P",
                ),
                capability(
                    SingingControl::SyllableAlignment,
                    CapabilitySupport::Native,
                    "explicit word/syllable-to-score interval bindings are validated before rendering",
                ),
                capability(
                    SingingControl::NotePitch,
                    CapabilitySupport::Native,
                    "each score event supplies the exact target F0 for its interval",
                ),
                capability(
                    SingingControl::NoteDuration,
                    CapabilitySupport::Approximated,
                    "score intervals are preserved, while phones are allocated by fixed duration heuristics on a 5 ms frame grid",
                ),
                capability(
                    SingingControl::AbsoluteOnsetAndRests,
                    CapabilitySupport::Approximated,
                    "absolute score time and explicit rests are materialized on a 5 ms frame grid",
                ),
                capability(
                    SingingControl::ConsonantAnticipation,
                    CapabilitySupport::Unsupported,
                    "the resolved score graph does not carry a consonant-anticipation control",
                ),
                capability(
                    SingingControl::Melisma,
                    CapabilitySupport::Approximated,
                    "a designated vowel is held across score events with a continuous envelope, while F0 still changes at note boundaries",
                ),
                capability(
                    SingingControl::ContinuousPitchCurve,
                    CapabilitySupport::Unsupported,
                    "the resolved-plan route follows discrete score-event F0 targets",
                ),
                capability(
                    SingingControl::PhraseExpressionCurve,
                    CapabilitySupport::Unsupported,
                    "no phrase-expression curve is supplied to this route",
                ),
                capability(
                    SingingControl::PhraseBreathTiming,
                    CapabilitySupport::Unsupported,
                    "phrase breath-before/after controls are not synthesized",
                ),
                capability(
                    SingingControl::Breathiness,
                    CapabilitySupport::Unsupported,
                    "the resolved-plan route has no independent breathiness input",
                ),
                capability(
                    SingingControl::Energy,
                    CapabilitySupport::Native,
                    "score-event velocity scales frame energy",
                ),
                capability(
                    SingingControl::Tension,
                    CapabilitySupport::Unsupported,
                    "the resolved-plan route has no vocal-fold tension input",
                ),
                capability(
                    SingingControl::Vibrato,
                    CapabilitySupport::Unsupported,
                    "the resolved-plan route does not receive a vibrato control",
                ),
                capability(
                    SingingControl::Portamento,
                    CapabilitySupport::Unsupported,
                    "the resolved-plan route does not receive a portamento control",
                ),
                capability(
                    SingingControl::Articulation,
                    CapabilitySupport::Approximated,
                    "source type distinguishes vowel, nasal, liquid, stop and fricative classes; durations use fixed heuristics",
                ),
                capability(
                    SingingControl::VoiceIdentity,
                    CapabilitySupport::Unsupported,
                    "the resolved-plan route uses the default source-filter voice",
                ),
                capability(
                    SingingControl::LocalEdits,
                    CapabilitySupport::Unsupported,
                    "the route renders a complete plan and has no localized edit API",
                ),
            ],
        )
    }

    /// Render a fully selected pronunciation and explicit score/alignment graph.
    ///
    /// Unlike the compatibility VocalPerformance path, this method does not
    /// call Muse G2P. The caller must supply a validated, selected pronunciation
    /// plan and a score alignment that assigns each note interval exactly once.
    /// Unsupported formant phones fail closed rather than taking the legacy
    /// formant table's default-vowel branch.
    /// Materialize a beat-domain source score and render its selected pronunciation.
    ///
    /// Syllable bindings use offsets inside the materialized seconds-domain
    /// events, while the original tempo map and beat events remain available to
    /// the caller for source receipts and reproducibility.
    pub fn render_beat_domain_plan(
        &mut self,
        pronunciation: &PronunciationPlan,
        score: &BeatDomainScore,
        syllable_bindings: &[SyllableNoteBinding],
        phrases: &[VocalPhraseSpan],
    ) -> Result<VocalStem> {
        let alignment = VocalScoreAlignment {
            schema_version: 1,
            events: score.materialize_seconds().map_err(anyhow::Error::msg)?,
            syllable_bindings: syllable_bindings.to_vec(),
            phrases: phrases.to_vec(),
        };
        self.render_resolved_plan(pronunciation, &alignment)
    }

    /// Policy-checked resolved-plan rendering. Controlled comparisons should
    /// use this entry point so the selected API route's capability manifest is
    /// validated before audio is produced.
    pub fn render_resolved_plan_with_policy(
        &mut self,
        pronunciation: &PronunciationPlan,
        alignment: &VocalScoreAlignment,
        policy: &SingingRenderPolicy,
    ) -> Result<VocalStem> {
        policy.validate_against(&Self::resolved_plan_capabilities()?)?;
        self.render_resolved_plan(pronunciation, alignment)
    }

    /// Beat-domain counterpart to `render_resolved_plan_with_policy`.
    pub fn render_beat_domain_plan_with_policy(
        &mut self,
        pronunciation: &PronunciationPlan,
        score: &BeatDomainScore,
        syllable_bindings: &[SyllableNoteBinding],
        phrases: &[VocalPhraseSpan],
        policy: &SingingRenderPolicy,
    ) -> Result<VocalStem> {
        policy.validate_against(&Self::resolved_plan_capabilities()?)?;
        self.render_beat_domain_plan(pronunciation, score, syllable_bindings, phrases)
    }

    pub fn render_resolved_plan(
        &mut self,
        pronunciation: &PronunciationPlan,
        alignment: &VocalScoreAlignment,
    ) -> Result<VocalStem> {
        let frames = self.resolved_plan_frames(pronunciation, alignment)?;
        let samples = self.vocoder.synthesize(&frames);
        let stem = VocalStem {
            samples,
            sample_rate: self.sample_rate,
            backend: format!("{BACKEND_ID}-resolved-plan-v1"),
        };
        stem.validate()?;
        Ok(stem)
    }

    fn resolved_plan_frames(
        &self,
        pronunciation: &PronunciationPlan,
        alignment: &VocalScoreAlignment,
    ) -> Result<Vec<FormantFrame>> {
        validate_formant_language(&pronunciation.language)?;
        pronunciation
            .validate_renderable()
            .map_err(anyhow::Error::msg)?;
        alignment
            .validate_against(pronunciation)
            .map_err(anyhow::Error::msg)?;

        let final_event = alignment
            .events
            .last()
            .ok_or_else(|| anyhow::anyhow!("resolved score has no events"))?;
        let duration_s = final_event.start_s + final_event.duration_s;
        if !duration_s.is_finite() || duration_s <= 0.0 || duration_s > 3_600.0 {
            bail!("resolved score duration must be finite and within (0, 3600] seconds");
        }

        let mut segments = Vec::new();
        for (binding_index, binding) in alignment.syllable_bindings.iter().enumerate() {
            let word = pronunciation.words.get(binding.word_index).ok_or_else(|| {
                anyhow::anyhow!("binding {binding_index} references missing word")
            })?;
            let selected_id = word.selected_candidate_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("binding {binding_index} has no selected candidate")
            })?;
            let candidate = word
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == selected_id)
                .ok_or_else(|| {
                    anyhow::anyhow!("binding {binding_index} selects a missing candidate")
                })?;
            let syllable = candidate
                .syllables
                .get(binding.syllable_index)
                .ok_or_else(|| {
                    anyhow::anyhow!("binding {binding_index} has no pronunciation syllable")
                })?;

            // A multi-note melisma is one sustained vowel event, not a fresh
            // vowel attack on every note. Compute its envelope once across the
            // first vowel onset through the final vowel offset; note F0 targets
            // may still change at the score-event boundaries.
            let melisma_envelope = if binding.spans.len() > 1 {
                let phone_index = binding.melisma_phone_index.ok_or_else(|| {
                    anyhow::anyhow!(
                        "binding {binding_index} has multiple spans without a melisma phone"
                    )
                })?;
                let first_span = &binding.spans[0];
                let first_event = &alignment.events[first_span.event_index];
                let first_start_s = first_event.start_s + first_span.start_offset_s;
                let first_end_s = first_event.start_s + first_span.end_offset_s;
                let first_phone_refs: Vec<&Phone> =
                    syllable.phones[..=phone_index].iter().collect();
                let first_durations = allocate_resolved_phone_durations(
                    &first_phone_refs,
                    first_end_s - first_start_s,
                    syllable.stress,
                    binding_index,
                )?;
                let vowel_onset_s =
                    first_start_s + first_durations[..phone_index].iter().sum::<f32>();

                let last_span = binding
                    .spans
                    .last()
                    .expect("multi-span binding is non-empty");
                let last_event = &alignment.events[last_span.event_index];
                let last_start_s = last_event.start_s + last_span.start_offset_s;
                let last_end_s = last_event.start_s + last_span.end_offset_s;
                let last_phone_refs: Vec<&Phone> = syllable.phones[phone_index..].iter().collect();
                let last_durations = allocate_resolved_phone_durations(
                    &last_phone_refs,
                    last_end_s - last_start_s,
                    syllable.stress,
                    binding_index,
                )?;
                let vowel_release_s = last_start_s + last_durations[0];
                Some((vowel_onset_s, vowel_release_s))
            } else {
                None
            };

            for (span_index, span) in binding.spans.iter().enumerate() {
                let event = alignment.events.get(span.event_index).ok_or_else(|| {
                    anyhow::anyhow!("binding {binding_index} references missing score event")
                })?;
                let (frequency_hz, velocity) = match event.kind {
                    ScoreEventKind::SungNote {
                        frequency_hz,
                        velocity,
                    } => (frequency_hz, velocity),
                    ScoreEventKind::Rest => {
                        bail!("binding {binding_index} attempted to render a Rest event")
                    }
                };
                let start_s = event.start_s + span.start_offset_s;
                let end_s = event.start_s + span.end_offset_s;
                let span_duration_s = end_s - start_s;
                if !start_s.is_finite()
                    || !end_s.is_finite()
                    || !span_duration_s.is_finite()
                    || span_duration_s <= 0.0
                {
                    bail!("binding {binding_index} has invalid materialized timing");
                }

                let phone_indices: Vec<usize> = if binding.spans.len() == 1 {
                    (0..syllable.phones.len()).collect()
                } else {
                    let phone_index = binding.melisma_phone_index.ok_or_else(|| {
                        anyhow::anyhow!(
                            "binding {binding_index} has additional note spans without a designated melisma vowel"
                        )
                    })?;
                    if syllable.phones.iter().skip(phone_index + 1).any(|phone| {
                        matches!(
                            phone.kind,
                            ResolvedPhoneKind::Vowel | ResolvedPhoneKind::SyllabicConsonant
                        )
                    }) {
                        bail!(
                            "binding {binding_index} melisma phone must be the final syllabic nucleus"
                        );
                    }
                    if span_index == 0 {
                        (0..=phone_index).collect()
                    } else if span_index + 1 == binding.spans.len() {
                        (phone_index..syllable.phones.len()).collect()
                    } else {
                        vec![phone_index]
                    }
                };
                let phones: Vec<&Phone> = phone_indices
                    .iter()
                    .map(|index| &syllable.phones[*index])
                    .collect();
                let durations = allocate_resolved_phone_durations(
                    &phones,
                    span_duration_s,
                    syllable.stress,
                    binding_index,
                )?;
                let mut phone_start_s = start_s;
                for ((phone, phone_index), phone_duration_s) in
                    phones.iter().zip(phone_indices).zip(durations)
                {
                    let (f1, f2, f3, source_type) = strict_formant_target(phone)?;
                    let phone_end_s = phone_start_s + phone_duration_s;
                    let stress_gain = match syllable.stress {
                        1 => 1.08,
                        2 => 1.04,
                        _ => 1.0,
                    };
                    let voiced = match source_type {
                        SourceType::Vowel | SourceType::Nasal | SourceType::Liquid => 0.92,
                        SourceType::Stop | SourceType::Fricative | SourceType::Affricate => {
                            if is_voiced_consonant(&phone.ipa) {
                                0.58
                            } else {
                                0.0
                            }
                        }
                        SourceType::Silent => 0.0,
                    };
                    let base_energy = (velocity * stress_gain).clamp(0.0, 1.0);
                    let (envelope_start_s, envelope_end_s) =
                        if Some(phone_index) == binding.melisma_phone_index {
                            melisma_envelope.unwrap_or((phone_start_s, phone_end_s))
                        } else {
                            (phone_start_s, phone_end_s)
                        };
                    segments.push(ResolvedPhoneSegment {
                        start_s: phone_start_s,
                        end_s: phone_end_s,
                        envelope_start_s,
                        envelope_end_s,
                        f1,
                        f2,
                        f3,
                        f0: frequency_hz,
                        energy: base_energy,
                        voicing: voiced,
                        source_type,
                    });
                    phone_start_s = phone_end_s;
                }
                if (phone_start_s - end_s).abs() > 0.000_01 {
                    bail!("binding {binding_index} phone durations do not cover its score span");
                }
            }
        }

        segments.sort_by(|left, right| left.start_s.total_cmp(&right.start_s));
        for (index, segment) in segments.iter().enumerate() {
            if index > 0 && segment.start_s < segments[index - 1].end_s - 0.000_01 {
                bail!("resolved phone segments overlap after materialization");
            }
        }

        let frame_intervals = (duration_s * FRAME_RATE_HZ).ceil().max(1.0) as usize;
        if frame_intervals > 720_000 {
            bail!("resolved score exceeds the one-hour formant-frame limit");
        }
        let mut frames = Vec::with_capacity(frame_intervals + 1);
        let mut segment_index = 0usize;
        for frame_index in 0..=frame_intervals {
            let time_s = frame_index as f32 * FRAME_DT_S;
            while segment_index < segments.len() && segments[segment_index].end_s <= time_s {
                segment_index += 1;
            }
            let frame = if let Some(segment) = segments.get(segment_index)
                && time_s >= segment.start_s
                && time_s < segment.end_s
            {
                let attack = ((time_s - segment.envelope_start_s) / 0.015).clamp(0.0, 1.0);
                let release = ((segment.envelope_end_s - time_s) / 0.015).clamp(0.0, 1.0);
                let envelope = attack.min(release);
                FormantFrame {
                    f1: segment.f1,
                    f2: segment.f2,
                    f3: segment.f3,
                    b1: 60.0,
                    b2: 80.0,
                    b3: 100.0,
                    f0: segment.f0,
                    energy: segment.energy * envelope,
                    voicing: segment.voicing,
                    time: time_s,
                    source_type: segment.source_type,
                    nasal_zero_freq: 0.0,
                    nasal_zero_bw: 0.0,
                }
            } else {
                FormantFrame::silent(time_s)
            };
            frames.push(frame);
        }
        Ok(frames)
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

fn validate_formant_language(language: &str) -> Result<()> {
    if !matches!(language, "en" | "en-US") {
        bail!(
            "FormantSingingEngine resolved-plan renderer currently supports language='en' or 'en-US' only; got {language:?}"
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ResolvedPhoneSegment {
    start_s: f32,
    end_s: f32,
    envelope_start_s: f32,
    envelope_end_s: f32,
    f1: f32,
    f2: f32,
    f3: f32,
    f0: f32,
    energy: f32,
    voicing: f32,
    source_type: SourceType,
}

fn allocate_resolved_phone_durations(
    phones: &[&Phone],
    total_duration_s: f32,
    stress: u8,
    binding_index: usize,
) -> Result<Vec<f32>> {
    if phones.is_empty() || !total_duration_s.is_finite() || total_duration_s <= 0.0 {
        bail!("binding {binding_index} requires phones and a positive finite duration");
    }
    let consonants = phones
        .iter()
        .filter(|phone| phone.kind == ResolvedPhoneKind::Consonant)
        .count();
    let nuclei: Vec<_> = phones
        .iter()
        .filter(|phone| {
            matches!(
                phone.kind,
                ResolvedPhoneKind::Vowel | ResolvedPhoneKind::SyllabicConsonant
            )
        })
        .collect();
    if nuclei.is_empty() {
        bail!("binding {binding_index} has no vowel or syllabic nucleus");
    }

    const CONSONANT_DURATION_S: f32 = 0.045;
    const MIN_NUCLEUS_DURATION_S: f32 = 0.025;
    let consonant_budget = consonants as f32 * CONSONANT_DURATION_S;
    let nucleus_budget = total_duration_s - consonant_budget;
    if nucleus_budget < nuclei.len() as f32 * MIN_NUCLEUS_DURATION_S {
        bail!("binding {binding_index} is too short to preserve its consonants and vowel nucleus");
    }
    let nucleus_weights: Vec<f32> = nuclei
        .iter()
        .map(|phone| {
            let kind_weight = if phone.kind == ResolvedPhoneKind::SyllabicConsonant {
                0.8
            } else {
                1.0
            };
            let stress_weight = match stress {
                1 => 1.1,
                2 => 1.05,
                _ => 1.0,
            };
            kind_weight * stress_weight
        })
        .collect();
    let total_nucleus_weight = nucleus_weights.iter().sum::<f32>();
    let mut nucleus_index = 0usize;
    let mut durations = Vec::with_capacity(phones.len());
    for phone in phones {
        if phone.kind == ResolvedPhoneKind::Consonant {
            durations.push(CONSONANT_DURATION_S);
        } else {
            let duration = nucleus_budget * nucleus_weights[nucleus_index] / total_nucleus_weight;
            durations.push(duration);
            nucleus_index += 1;
        }
    }
    Ok(durations)
}

fn strict_formant_target(phone: &Phone) -> Result<(f32, f32, f32, SourceType)> {
    let supported = matches!(
        phone.ipa.as_str(),
        "iː" | "i"
            | "ɪ"
            | "ɛ"
            | "æ"
            | "ɑ"
            | "ɑː"
            | "ɒ"
            | "ɔ"
            | "ɔː"
            | "ʌ"
            | "ʊ"
            | "uː"
            | "u"
            | "ə"
            | "ɜː"
            | "eɪ"
            | "aɪ"
            | "oʊ"
            | "aʊ"
            | "m"
            | "n"
            | "ŋ"
            | "p"
            | "b"
            | "t"
            | "d"
            | "k"
            | "ɡ"
            | "f"
            | "v"
            | "θ"
            | "ð"
            | "s"
            | "z"
            | "ʃ"
            | "ʒ"
            | "h"
            | "tʃ"
            | "dʒ"
            | "l"
            | "ɹ"
            | "w"
            | "j"
    );
    if !supported {
        bail!(
            "resolved IPA phone {:?} is outside the calibrated legacy formant inventory",
            phone.ipa
        );
    }
    let (f1, f2, f3, source) = formant_target_pub(&phone.ipa);
    match phone.kind {
        ResolvedPhoneKind::Vowel if source != SourceType::Vowel => {
            bail!(
                "resolved vowel {:?} maps to a non-vowel formant source",
                phone.ipa
            )
        }
        ResolvedPhoneKind::SyllabicConsonant
            if !matches!(source, SourceType::Nasal | SourceType::Liquid) =>
        {
            bail!(
                "syllabic consonant {:?} has no supported syllabic source type",
                phone.ipa
            )
        }
        ResolvedPhoneKind::Consonant if source == SourceType::Vowel => {
            bail!("resolved consonant {:?} maps to a vowel source", phone.ipa)
        }
        _ => {}
    }
    Ok((f1, f2, f3, source))
}

fn is_voiced_consonant(ipa: &str) -> bool {
    matches!(ipa, "b" | "d" | "ɡ" | "v" | "ð" | "z" | "ʒ" | "dʒ")
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
fn apply_note_pitch_controls(frames: &mut [FormantFrame], performance: &VocalPerformance) {
    const EDGE_RAMP_S: f32 = 0.015;
    for syllable in &performance.syllables {
        for note in syllable.notes() {
            let note_start = note.start_time;
            let note_end = note_start + note.duration;
            for frame in frames
                .iter_mut()
                .filter(|frame| frame.time >= note_start && frame.time < note_end)
            {
                let Some(target_pitch) = performance.target_pitch_at(frame.time) else {
                    continue;
                };
                let elapsed = (frame.time - note_start).max(0.0);
                let attack = (elapsed / EDGE_RAMP_S).clamp(0.0, 1.0);
                let release = ((note_end - frame.time) / EDGE_RAMP_S).clamp(0.0, 1.0);
                frame.f0 = target_pitch * attack.min(release);
            }
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
    use symthaea_pronunciation_plan::{
        CandidateProvenance, CandidateSource, Phone, PhoneKind, PronunciationCandidate,
        PronunciationSyllable, ResolutionState, ScoreEvent, ScoreEventSpan, SelectionAuthority,
        SelectionReceipt, SyllableNoteBinding, VocalPhraseSpan, WordPronunciation,
    };

    fn note(start_time: f32, duration: f32, frequency: f32) -> Note {
        Note {
            frequency,
            start_time,
            duration,
            velocity: 0.7,
        }
    }

    fn selected_word(
        surface: &str,
        start_byte: usize,
        candidate_id: &str,
        phones: Vec<Phone>,
    ) -> WordPronunciation {
        let candidate = PronunciationCandidate {
            candidate_id: candidate_id.to_string(),
            provenance: CandidateProvenance {
                source: CandidateSource::HumanReviewed,
                provider_id: "fixture-reviewer".to_string(),
                provider_version: "1".to_string(),
                source_digest: format!("blake3:{}", "0".repeat(64)),
            },
            raw_provider_score: None,
            syllables: vec![PronunciationSyllable { stress: 1, phones }],
        };
        WordPronunciation {
            start_byte,
            end_byte: start_byte + surface.len(),
            surface: surface.to_string(),
            state: ResolutionState::Selected,
            candidates: vec![candidate],
            selected_candidate_id: Some(candidate_id.to_string()),
            selection_receipt: Some(SelectionReceipt {
                authority: SelectionAuthority::HumanReviewer,
                evidence_reference: format!("review:{candidate_id}"),
            }),
        }
    }

    fn phone(ipa: &str, kind: PhoneKind) -> Phone {
        Phone {
            ipa: ipa.to_string(),
            kind,
        }
    }

    fn resolved_read_now_fixture() -> (PronunciationPlan, VocalScoreAlignment) {
        let pronunciation = PronunciationPlan {
            schema_version: 1,
            language: "en-US".to_string(),
            source_text: "read now".to_string(),
            words: vec![
                selected_word(
                    "read",
                    0,
                    "read-past",
                    vec![
                        phone("ɹ", PhoneKind::Consonant),
                        phone("ɛ", PhoneKind::Vowel),
                        phone("d", PhoneKind::Consonant),
                    ],
                ),
                selected_word(
                    "now",
                    5,
                    "now-standard",
                    vec![
                        phone("n", PhoneKind::Consonant),
                        phone("aʊ", PhoneKind::Vowel),
                    ],
                ),
            ],
        };
        let alignment = VocalScoreAlignment {
            schema_version: 1,
            events: vec![
                ScoreEvent {
                    start_s: 0.0,
                    duration_s: 0.4,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 220.0,
                        velocity: 0.7,
                    },
                },
                ScoreEvent {
                    start_s: 0.4,
                    duration_s: 0.2,
                    kind: ScoreEventKind::Rest,
                },
                ScoreEvent {
                    start_s: 0.6,
                    duration_s: 0.4,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 246.94,
                        velocity: 0.7,
                    },
                },
            ],
            syllable_bindings: vec![
                SyllableNoteBinding {
                    word_index: 0,
                    syllable_index: 0,
                    spans: vec![ScoreEventSpan {
                        event_index: 0,
                        start_offset_s: 0.0,
                        end_offset_s: 0.4,
                    }],
                    melisma_phone_index: None,
                },
                SyllableNoteBinding {
                    word_index: 1,
                    syllable_index: 0,
                    spans: vec![ScoreEventSpan {
                        event_index: 2,
                        start_offset_s: 0.0,
                        end_offset_s: 0.4,
                    }],
                    melisma_phone_index: None,
                },
            ],
            phrases: vec![VocalPhraseSpan {
                syllable_start: 0,
                syllable_end: 2,
                breath_before_s: 0.0,
                breath_after_s: 0.1,
            }],
        };
        (pronunciation, alignment)
    }

    #[test]
    fn resolved_plan_renderer_uses_selected_phones_without_rerunning_g2p() {
        let (pronunciation, alignment) = resolved_read_now_fixture();
        let mut engine = FormantSingingEngine::default();
        let frames = engine
            .resolved_plan_frames(&pronunciation, &alignment)
            .unwrap();
        let frame_at = |time_s: f32| {
            frames
                .iter()
                .find(|frame| (frame.time - time_s).abs() < 1e-5)
                .unwrap()
        };
        assert_eq!(frame_at(0.20).source_type, SourceType::Vowel);
        assert_eq!(frame_at(0.50).source_type, SourceType::Silent);
        assert_eq!(frame_at(0.70).source_type, SourceType::Nasal);
        assert_eq!(frame_at(0.90).source_type, SourceType::Vowel);

        let stem = engine
            .render_resolved_plan(&pronunciation, &alignment)
            .unwrap();
        stem.validate().unwrap();
        assert_eq!(stem.backend, "formant-vocoder-singing-resolved-plan-v1");
        assert_eq!(stem.sample_rate, 24_000);
    }

    #[test]
    fn resolved_plan_renderer_allows_two_syllables_to_share_one_long_note() {
        let (mut pronunciation, _) = resolved_read_now_fixture();
        let alignment = VocalScoreAlignment {
            schema_version: 1,
            events: vec![ScoreEvent {
                start_s: 0.0,
                duration_s: 1.0,
                kind: ScoreEventKind::SungNote {
                    frequency_hz: 220.0,
                    velocity: 0.7,
                },
            }],
            syllable_bindings: vec![
                SyllableNoteBinding {
                    word_index: 0,
                    syllable_index: 0,
                    spans: vec![ScoreEventSpan {
                        event_index: 0,
                        start_offset_s: 0.0,
                        end_offset_s: 0.45,
                    }],
                    melisma_phone_index: None,
                },
                SyllableNoteBinding {
                    word_index: 1,
                    syllable_index: 0,
                    spans: vec![ScoreEventSpan {
                        event_index: 0,
                        start_offset_s: 0.45,
                        end_offset_s: 1.0,
                    }],
                    melisma_phone_index: None,
                },
            ],
            phrases: vec![VocalPhraseSpan {
                syllable_start: 0,
                syllable_end: 2,
                breath_before_s: 0.0,
                breath_after_s: 0.0,
            }],
        };
        pronunciation.source_text = "read now".to_string();
        let mut engine = FormantSingingEngine::default();
        let frames = engine
            .resolved_plan_frames(&pronunciation, &alignment)
            .unwrap();
        let at_end_of_first = frames
            .iter()
            .find(|frame| (frame.time - 0.35).abs() < 1e-5)
            .unwrap();
        let at_start_of_second = frames
            .iter()
            .find(|frame| (frame.time - 0.55).abs() < 1e-5)
            .unwrap();
        assert_eq!(at_end_of_first.source_type, SourceType::Vowel);
        assert_eq!(at_start_of_second.source_type, SourceType::Nasal);
    }

    #[test]
    fn resolved_plan_renderer_sustains_only_designated_vowel_across_melisma() {
        let word = selected_word(
            "read",
            0,
            "read-past",
            vec![
                phone("ɹ", PhoneKind::Consonant),
                phone("ɛ", PhoneKind::Vowel),
                phone("d", PhoneKind::Consonant),
            ],
        );
        let pronunciation = PronunciationPlan {
            schema_version: 1,
            language: "en-US".to_string(),
            source_text: "read".to_string(),
            words: vec![word],
        };
        let alignment = VocalScoreAlignment {
            schema_version: 1,
            events: vec![
                ScoreEvent {
                    start_s: 0.0,
                    duration_s: 0.5,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 220.0,
                        velocity: 0.7,
                    },
                },
                ScoreEvent {
                    start_s: 0.5,
                    duration_s: 0.5,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 246.94,
                        velocity: 0.7,
                    },
                },
            ],
            syllable_bindings: vec![SyllableNoteBinding {
                word_index: 0,
                syllable_index: 0,
                spans: vec![
                    ScoreEventSpan {
                        event_index: 0,
                        start_offset_s: 0.0,
                        end_offset_s: 0.5,
                    },
                    ScoreEventSpan {
                        event_index: 1,
                        start_offset_s: 0.0,
                        end_offset_s: 0.5,
                    },
                ],
                melisma_phone_index: Some(1),
            }],
            phrases: vec![VocalPhraseSpan {
                syllable_start: 0,
                syllable_end: 1,
                breath_before_s: 0.0,
                breath_after_s: 0.0,
            }],
        };
        let mut engine = FormantSingingEngine::default();
        let frames = engine
            .resolved_plan_frames(&pronunciation, &alignment)
            .unwrap();
        let second_note = frames
            .iter()
            .find(|frame| (frame.time - 0.7).abs() < 1e-5)
            .unwrap();
        assert_eq!(second_note.source_type, SourceType::Vowel);
        assert!((second_note.f0 - 246.94).abs() < 0.01);

        // The vowel is one continuous melisma event: the first sample frame
        // after a tied-note boundary must not restart its attack envelope.
        let before_boundary = frames
            .iter()
            .find(|frame| (frame.time - 0.495).abs() < 1e-5)
            .unwrap();
        let at_boundary = frames
            .iter()
            .find(|frame| (frame.time - 0.5).abs() < 1e-5)
            .unwrap();
        assert!(before_boundary.energy > 0.5);
        assert!(at_boundary.energy > 0.5);
        assert!(
            (before_boundary.energy - at_boundary.energy).abs() < 0.01,
            "melisma envelope must remain continuous: before={}, at={}",
            before_boundary.energy,
            at_boundary.energy
        );
    }

    #[test]
    fn beat_domain_renderer_materializes_tempo_changes_and_explicit_rests() {
        let (pronunciation, _) = resolved_read_now_fixture();
        let score = BeatDomainScore {
            schema_version: 1,
            tempo_map: vec![
                TempoChange {
                    start_beat: 0.0,
                    quarter_bpm: 120.0,
                },
                TempoChange {
                    start_beat: 4.0,
                    quarter_bpm: 60.0,
                },
            ],
            events: vec![
                BeatScoreEvent {
                    start_beat: 0.0,
                    duration_beats: 2.0,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 220.0,
                        velocity: 0.7,
                    },
                },
                BeatScoreEvent {
                    start_beat: 2.0,
                    duration_beats: 2.0,
                    kind: ScoreEventKind::Rest,
                },
                BeatScoreEvent {
                    start_beat: 4.0,
                    duration_beats: 2.0,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 246.94,
                        velocity: 0.7,
                    },
                },
            ],
        };
        let bindings = vec![
            SyllableNoteBinding {
                word_index: 0,
                syllable_index: 0,
                spans: vec![ScoreEventSpan {
                    event_index: 0,
                    start_offset_s: 0.0,
                    end_offset_s: 1.0,
                }],
                melisma_phone_index: None,
            },
            SyllableNoteBinding {
                word_index: 1,
                syllable_index: 0,
                spans: vec![ScoreEventSpan {
                    event_index: 2,
                    start_offset_s: 0.0,
                    end_offset_s: 2.0,
                }],
                melisma_phone_index: None,
            },
        ];
        let phrases = [VocalPhraseSpan {
            syllable_start: 0,
            syllable_end: 2,
            breath_before_s: 0.0,
            breath_after_s: 0.1,
        }];
        let mut engine = FormantSingingEngine::default();
        let stem = engine
            .render_beat_domain_plan(&pronunciation, &score, &bindings, &phrases)
            .unwrap();
        assert_eq!(stem.sample_rate, 24_000);

        let frames = engine
            .resolved_plan_frames(
                &pronunciation,
                &VocalScoreAlignment {
                    schema_version: 1,
                    events: score.materialize_seconds().unwrap(),
                    syllable_bindings: bindings,
                    phrases: phrases.to_vec(),
                },
            )
            .unwrap();
        let frame_at = |time_s: f32| {
            frames
                .iter()
                .find(|frame| (frame.time - time_s).abs() < 1e-5)
                .unwrap()
        };
        assert_eq!(frame_at(0.5).source_type, SourceType::Vowel);
        assert_eq!(frame_at(1.5).source_type, SourceType::Silent);
        assert_eq!(frame_at(3.0).source_type, SourceType::Vowel);
        assert!((frame_at(3.0).f0 - 246.94).abs() < 0.01);
    }

    #[test]
    fn resolved_plan_renderer_rejects_composite_phone_sequences_as_single_phones() {
        let mut pronunciation = resolved_read_now_fixture().0;
        pronunciation.words[0].candidates[0].syllables[0].phones[0].ipa = "ks".to_string();
        let alignment = resolved_read_now_fixture().1;
        let mut engine = FormantSingingEngine::default();
        let error = engine
            .resolved_plan_frames(&pronunciation, &alignment)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("outside the calibrated legacy formant inventory"),
            "{error}"
        );
    }

    #[test]
    fn resolved_melisma_releases_coda_only_after_the_final_sustained_vowel() {
        let word = selected_word(
            "read",
            0,
            "read-past",
            vec![
                phone("ɹ", PhoneKind::Consonant),
                phone("ɛ", PhoneKind::Vowel),
                phone("d", PhoneKind::Consonant),
            ],
        );
        let pronunciation = PronunciationPlan {
            schema_version: 1,
            language: "en-US".to_string(),
            source_text: "read".to_string(),
            words: vec![word],
        };
        let alignment = VocalScoreAlignment {
            schema_version: 1,
            events: vec![
                ScoreEvent {
                    start_s: 0.0,
                    duration_s: 0.5,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 220.0,
                        velocity: 0.7,
                    },
                },
                ScoreEvent {
                    start_s: 0.5,
                    duration_s: 0.5,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 246.94,
                        velocity: 0.7,
                    },
                },
            ],
            syllable_bindings: vec![SyllableNoteBinding {
                word_index: 0,
                syllable_index: 0,
                spans: vec![
                    ScoreEventSpan {
                        event_index: 0,
                        start_offset_s: 0.0,
                        end_offset_s: 0.5,
                    },
                    ScoreEventSpan {
                        event_index: 1,
                        start_offset_s: 0.0,
                        end_offset_s: 0.5,
                    },
                ],
                melisma_phone_index: Some(1),
            }],
            phrases: vec![VocalPhraseSpan {
                syllable_start: 0,
                syllable_end: 1,
                breath_before_s: 0.0,
                breath_after_s: 0.0,
            }],
        };
        let mut engine = FormantSingingEngine::default();
        let frames = engine
            .resolved_plan_frames(&pronunciation, &alignment)
            .unwrap();
        let vowel_during_melisma = frames
            .iter()
            .find(|frame| (frame.time - 0.7).abs() < 1e-5)
            .unwrap();
        let coda_after_melisma = frames
            .iter()
            .find(|frame| (frame.time - 0.95).abs() < 1e-5)
            .unwrap();
        assert_eq!(vowel_during_melisma.source_type, SourceType::Vowel);
        assert_eq!(coda_after_melisma.source_type, SourceType::Stop);
        assert!((vowel_during_melisma.f0 - 246.94).abs() < 0.01);
    }

    #[test]
    fn resolved_plan_renderer_rejects_phones_outside_calibrated_inventory() {
        let mut pronunciation = resolved_read_now_fixture().0;
        pronunciation.words[0].candidates[0].syllables[0].phones[0].ipa = "ɣ".to_string();
        let alignment = resolved_read_now_fixture().1;
        let mut engine = FormantSingingEngine::default();
        let error = engine
            .resolved_plan_frames(&pronunciation, &alignment)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("outside the calibrated legacy formant inventory"),
            "{error}"
        );
    }

    #[test]
    fn resolved_plan_policy_blocks_controls_that_this_route_does_not_receive() {
        let (pronunciation, alignment) = resolved_read_now_fixture();
        let mut engine = FormantSingingEngine::default();
        let error = engine
            .render_resolved_plan_with_policy(
                &pronunciation,
                &alignment,
                &SingingRenderPolicy::phrase_expression_comparison(),
            )
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("cannot satisfy required control PhraseExpressionCurve"),
            "{error}"
        );

        let stem = engine
            .render_resolved_plan_with_policy(
                &pronunciation,
                &alignment,
                &SingingRenderPolicy::score_conditioned_comparison(),
            )
            .unwrap();
        stem.validate().unwrap();
    }

    #[test]
    fn resolved_plan_capabilities_do_not_inherit_legacy_expression_claims() {
        let capabilities = FormantSingingEngine::resolved_plan_capabilities().unwrap();
        capabilities.validate().unwrap();
        assert_eq!(
            capabilities.support(SingingControl::Vibrato),
            CapabilitySupport::Unsupported
        );
        assert_eq!(
            capabilities.support(SingingControl::Portamento),
            CapabilitySupport::Unsupported
        );
        assert_eq!(
            capabilities.support(SingingControl::PhraseExpressionCurve),
            CapabilitySupport::Unsupported
        );
        SingingRenderPolicy::score_conditioned_comparison()
            .validate_against(&capabilities)
            .unwrap();
        assert!(
            SingingRenderPolicy::phrase_expression_comparison()
                .validate_against(&capabilities)
                .is_err()
        );
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
        let before_spread = before_onset
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
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
        let after_spread = after_onset
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            - after_onset.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            after_spread > 3.0,
            "configured vibrato depth must measurably modulate F0: spread={after_spread}"
        );
        let mean = after_onset.iter().sum::<f32>() / after_onset.len() as f32;
        assert!(
            (mean - 220.0).abs() < 2.0,
            "vibrato should remain centered on the score pitch: {mean}"
        );
    }

    #[test]
    fn per_syllable_portamento_glides_in_log_frequency_and_reaches_target() {
        let notes = [note(0.0, 0.5, 220.0), note(0.5, 0.5, 440.0)];
        let mut performance = VocalPerformance::from_melody("light bright", &notes, "en").unwrap();
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
    #[test]
    fn resolved_plan_renderer_rejects_unqualified_non_english_profiles() {
        let (mut pronunciation, alignment) = resolved_read_now_fixture();
        pronunciation.language = "xh-ZA".to_string();
        let mut engine = FormantSingingEngine::default();
        let error = engine
            .resolved_plan_frames(&pronunciation, &alignment)
            .unwrap_err()
            .to_string();
        assert!(error.contains("language='en' or 'en-US' only"), "{error}");
    }
}
