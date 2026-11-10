// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing pronunciation lattice.
//!
//! A renderer must consume a resolved plan. It must not rerun G2P, silently
//! choose a pronunciation variant, or invent a phone when a word is OOV.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PronunciationPlan {
    pub schema_version: u32,
    pub language: String,
    pub source_text: String,
    pub words: Vec<WordPronunciation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WordPronunciation {
    pub start_byte: usize,
    pub end_byte: usize,
    pub surface: String,
    pub state: ResolutionState,
    pub candidates: Vec<PronunciationCandidate>,
    pub selected_candidate_id: Option<String>,
    pub selection_receipt: Option<SelectionReceipt>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionState {
    NeedsReview,
    Selected,
    OutOfVocabulary,
    Unsupported,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PronunciationCandidate {
    pub candidate_id: String,
    pub provenance: CandidateProvenance,
    /// Raw provider score only. It is not a calibrated probability and must
    /// never be compared across providers without an explicit calibration.
    pub raw_provider_score: Option<f32>,
    pub syllables: Vec<PronunciationSyllable>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateProvenance {
    pub source: CandidateSource,
    pub provider_id: String,
    pub provider_version: String,
    /// Digest/revision of the exact lexicon, engine, annotation bundle, or
    /// override source. A mutable path or display name is not sufficient.
    pub source_digest: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateSource {
    Dictionary,
    RuleG2p,
    ModelG2p,
    HumanReviewed,
    UserOverride,
    CommunityReviewed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PronunciationSyllable {
    pub stress: u8,
    pub phones: Vec<Phone>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Phone {
    /// One canonical normalized IPA phone token. Syllable stress and
    /// boundaries are represented in typed fields; a conventional length
    /// diacritic may remain attached to the phone (for example, "iː").
    pub ipa: String,
    pub kind: PhoneKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneKind {
    Vowel,
    Consonant,
    SyllabicConsonant,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionReceipt {
    pub authority: SelectionAuthority,
    /// Non-empty reference to a review/override record; avoid storing personal
    /// reviewer identity in the pronunciation artifact itself.
    pub evidence_reference: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionAuthority {
    HumanReviewer,
    User,
    CommunityReview,
    ApprovedPolicy,
}

impl PronunciationPlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported pronunciation-plan schema version {}",
                self.schema_version
            ));
        }
        if self.language.trim().is_empty()
            || self.language.bytes().any(|byte| byte.is_ascii_whitespace())
        {
            return Err("language must be an explicit non-empty tag".into());
        }
        if self.source_text.trim().is_empty() {
            return Err("source_text must not be empty".into());
        }
        if self.words.is_empty() {
            return Err("at least one word span is required".into());
        }

        let mut previous_end = 0usize;
        for (word_index, word) in self.words.iter().enumerate() {
            if word.start_byte >= word.end_byte || word.end_byte > self.source_text.len() {
                return Err(format!("word {word_index} has an invalid byte span"));
            }
            if !self.source_text.is_char_boundary(word.start_byte)
                || !self.source_text.is_char_boundary(word.end_byte)
            {
                return Err(format!("word {word_index} span splits a UTF-8 code point"));
            }
            if word.start_byte < previous_end {
                return Err(format!("word {word_index} overlaps or is out of order"));
            }
            if word.surface != self.source_text[word.start_byte..word.end_byte] {
                return Err(format!(
                    "word {word_index} surface does not match source_text"
                ));
            }
            if word.surface.chars().any(char::is_whitespace) {
                return Err(format!(
                    "word {word_index} span contains whitespace; each pronunciation span must contain one lexical token"
                ));
            }
            validate_word_state(word_index, word)?;

            previous_end = word.end_byte;
        }

        // Every non-whitespace source character must belong to exactly one
        // declared word span. Text cannot disappear between tokenization and G2P.
        for (byte_index, character) in self.source_text.char_indices() {
            if character.is_whitespace() {
                continue;
            }
            let covered = self
                .words
                .iter()
                .any(|word| byte_index >= word.start_byte && byte_index < word.end_byte);
            if !covered {
                return Err(format!(
                    "source character {character:?} at byte {byte_index} is not covered by a word span"
                ));
            }
        }

        Ok(())
    }

    /// A renderer may consume only fully selected pronunciations. Review,
    /// OOV, and unsupported words must be resolved upstream, not guessed
    /// during synthesis.
    pub fn validate_renderable(&self) -> Result<(), String> {
        self.validate()?;
        for (word_index, word) in self.words.iter().enumerate() {
            if word.state != ResolutionState::Selected {
                return Err(format!(
                    "word {word_index} is {:?}; rendering requires an explicit selected pronunciation",
                    word.state
                ));
            }
        }
        Ok(())
    }
}

fn valid_source_digest(value: &str) -> bool {
    let Some((algorithm, encoded)) = value.split_once(':') else {
        return false;
    };
    let expected_length = match algorithm {
        "blake3" | "sha256" => 64,
        "git-sha1" => 40,
        _ => return false,
    };
    encoded.len() == expected_length
        && encoded
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_word_state(index: usize, word: &WordPronunciation) -> Result<(), String> {
    let mut ids = std::collections::BTreeSet::new();
    for candidate in &word.candidates {
        if candidate.candidate_id.trim().is_empty() || !ids.insert(candidate.candidate_id.as_str())
        {
            return Err(format!(
                "word {index} has an empty or duplicate candidate ID"
            ));
        }
        if candidate.provenance.provider_id.trim().is_empty()
            || candidate.provenance.provider_version.trim().is_empty()
            || !valid_source_digest(&candidate.provenance.source_digest)
        {
            return Err(format!(
                "candidate {} has incomplete or malformed provenance digest",
                candidate.candidate_id
            ));
        }
        if candidate
            .raw_provider_score
            .is_some_and(|score| !score.is_finite())
        {
            return Err(format!(
                "candidate {} has a non-finite raw score",
                candidate.candidate_id
            ));
        }
        if candidate.syllables.is_empty() {
            return Err(format!(
                "candidate {} has no syllables",
                candidate.candidate_id
            ));
        }
        for (syllable_index, syllable) in candidate.syllables.iter().enumerate() {
            if syllable.stress > 2 || syllable.phones.is_empty() {
                return Err(format!(
                    "candidate {} syllable {syllable_index} has invalid stress or no phones",
                    candidate.candidate_id
                ));
            }
            let has_nucleus = syllable
                .phones
                .iter()
                .any(|phone| matches!(phone.kind, PhoneKind::Vowel | PhoneKind::SyllabicConsonant));
            if !has_nucleus {
                return Err(format!(
                    "candidate {} syllable {syllable_index} has no vowel or syllabic nucleus",
                    candidate.candidate_id
                ));
            }
            for (phone_index, phone) in syllable.phones.iter().enumerate() {
                if phone.ipa.trim().is_empty()
                    || phone.ipa.chars().any(char::is_whitespace)
                    || phone.ipa.contains('ˈ')
                    || phone.ipa.contains('ˌ')
                    || phone.ipa == "ː"
                {
                    return Err(format!(
                        "candidate {} syllable {syllable_index} phone {phone_index} is not one normalized IPA token",
                        candidate.candidate_id
                    ));
                }
            }
        }
    }

    match word.state {
        ResolutionState::NeedsReview => {
            if word.candidates.is_empty()
                || word.selected_candidate_id.is_some()
                || word.selection_receipt.is_some()
            {
                return Err(format!(
                    "word {index} needs review and must retain candidates without a selection"
                ));
            }
        }
        ResolutionState::Selected => {
            let Some(selected) = word.selected_candidate_id.as_deref() else {
                return Err(format!("word {index} is selected without a candidate ID"));
            };
            if !ids.contains(selected) {
                return Err(format!(
                    "word {index} selects unknown pronunciation candidate {selected}"
                ));
            }
            let Some(receipt) = &word.selection_receipt else {
                return Err(format!("word {index} selection has no evidence receipt"));
            };
            if receipt.evidence_reference.trim().is_empty() {
                return Err(format!(
                    "word {index} selection evidence reference is empty"
                ));
            }
        }
        ResolutionState::OutOfVocabulary | ResolutionState::Unsupported => {
            if !word.candidates.is_empty()
                || word.selected_candidate_id.is_some()
                || word.selection_receipt.is_some()
            {
                return Err(format!(
                    "word {index} OOV/unsupported state must not invent or select a pronunciation"
                ));
            }
        }
    }
    Ok(())
}

/// Score event consumed by a vocal performance. Rests are explicit and remain
/// unbound; a renderer must not turn them into sustained vowels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreEvent {
    pub start_s: f32,
    pub duration_s: f32,
    pub kind: ScoreEventKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreEventKind {
    SungNote { frequency_hz: f32, velocity: f32 },
    Rest,
}

/// A piecewise-constant tempo change in quarter-note beats per minute.
/// The first entry in a beat-domain score must begin at beat zero.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TempoChange {
    pub start_beat: f64,
    pub quarter_bpm: f64,
}

/// A score event expressed in musical beat coordinates rather than seconds.
/// Seconds are materialized only at the renderer boundary from the tempo map.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeatScoreEvent {
    pub start_beat: f64,
    pub duration_beats: f64,
    pub kind: ScoreEventKind,
}

/// Source-of-truth score timeline with explicit tempo map and beat-domain events.
/// This is additive to the seconds-domain score alignment: renderers still
/// receive canonical seconds after conversion, while receipts can retain the
/// original beats and tempo changes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeatDomainScore {
    pub schema_version: u32,
    pub tempo_map: Vec<TempoChange>,
    pub events: Vec<BeatScoreEvent>,
}

impl BeatDomainScore {
    const BEAT_EPSILON: f64 = 1e-9;

    fn validate_tempo_map(&self) -> Result<(), String> {
        if self.tempo_map.is_empty() {
            return Err("beat-domain score requires an explicit tempo map".into());
        }
        if self.tempo_map[0].start_beat.abs() > Self::BEAT_EPSILON {
            return Err("tempo map must begin at beat zero".into());
        }
        let mut previous_beat = -f64::INFINITY;
        for (index, change) in self.tempo_map.iter().enumerate() {
            if !change.start_beat.is_finite()
                || change.start_beat < 0.0
                || !change.quarter_bpm.is_finite()
                || !(10.0..=1000.0).contains(&change.quarter_bpm)
                || change.start_beat <= previous_beat
            {
                return Err(format!(
                    "tempo change {index} must have a strictly increasing finite beat and quarter BPM in [10, 1000]"
                ));
            }
            previous_beat = change.start_beat;
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported beat-domain score schema version {}",
                self.schema_version
            ));
        }
        self.validate_tempo_map()?;
        if self.events.is_empty() {
            return Err("beat-domain score requires at least one score event".into());
        }

        let mut previous_end = 0.0f64;
        for (index, event) in self.events.iter().enumerate() {
            if !event.start_beat.is_finite()
                || event.start_beat < 0.0
                || !event.duration_beats.is_finite()
                || event.duration_beats <= 0.0
            {
                return Err(format!("beat score event {index} has invalid timing"));
            }
            if (event.start_beat - previous_end).abs() > Self::BEAT_EPSILON {
                return Err(format!(
                    "beat score event {index} is overlapping or has an implicit gap; represent gaps as explicit Rest events"
                ));
            }
            let end = event.start_beat + event.duration_beats;
            if !end.is_finite() || end <= event.start_beat {
                return Err(format!("beat score event {index} has an invalid end beat"));
            }
            if let ScoreEventKind::SungNote {
                frequency_hz,
                velocity,
            } = event.kind
            {
                if !frequency_hz.is_finite()
                    || frequency_hz <= 0.0
                    || !velocity.is_finite()
                    || !(0.0..=1.0).contains(&velocity)
                {
                    return Err(format!(
                        "beat score event {index} has invalid pitch/velocity"
                    ));
                }
            }
            previous_end = end;
        }
        let final_seconds = self.seconds_at_beat(previous_end)?;
        if !final_seconds.is_finite() || final_seconds <= 0.0 {
            return Err("beat-domain score produces an invalid final time".into());
        }
        Ok(())
    }

    /// Convert a musical beat position to absolute seconds by integrating the
    /// piecewise-constant quarter-note tempo map. Tempo changes may occur
    /// inside a note; the event's seconds duration then reflects both tempos.
    pub fn seconds_at_beat(&self, beat: f64) -> Result<f64, String> {
        if !beat.is_finite() || beat < 0.0 {
            return Err("beat position must be finite and non-negative".into());
        }
        self.validate_tempo_map()?;

        let mut seconds = 0.0f64;
        for (index, change) in self.tempo_map.iter().enumerate() {
            let segment_start = change.start_beat;
            if beat <= segment_start {
                break;
            }
            let segment_end = self
                .tempo_map
                .get(index + 1)
                .map(|next| next.start_beat.min(beat))
                .unwrap_or(beat);
            if segment_end > segment_start {
                if !change.quarter_bpm.is_finite() || change.quarter_bpm <= 0.0 {
                    return Err(format!("tempo change {index} has invalid BPM"));
                }
                seconds += (segment_end - segment_start) * 60.0 / change.quarter_bpm;
            }
            if beat <= segment_end {
                break;
            }
        }
        if !seconds.is_finite() {
            return Err("tempo map conversion overflowed".into());
        }
        Ok(seconds)
    }

    /// Materialize a renderer-compatible seconds timeline without mutating or
    /// discarding the beat-domain source representation.
    pub fn materialize_seconds(&self) -> Result<Vec<ScoreEvent>, String> {
        self.validate()?;
        self.events
            .iter()
            .enumerate()
            .map(|(index, event)| {
                let start_s = self.seconds_at_beat(event.start_beat)?;
                let end_s = self.seconds_at_beat(event.start_beat + event.duration_beats)?;
                let duration_s = end_s - start_s;
                let start_s_f32 = start_s as f32;
                let duration_s_f32 = duration_s as f32;
                if !start_s_f32.is_finite()
                    || !duration_s.is_finite()
                    || duration_s <= 0.0
                    || !duration_s_f32.is_finite()
                    || duration_s_f32 <= 0.0
                {
                    return Err(format!(
                        "beat score event {index} cannot be represented as a finite positive f32 timeline"
                    ));
                }
                Ok(ScoreEvent {
                    start_s: start_s_f32,
                    duration_s: duration_s_f32,
                    kind: event.kind,
                })
            })
            .collect()
    }
}

/// Ordered note-index assignment for one resolved pronunciation syllable.
/// Multiple consecutive note indices represent a melisma over one explicitly
/// designated vowel phone from the selected pronunciation candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreEventSpan {
    /// Index into the ordered score event list.
    pub event_index: usize,
    /// Half-open interval, in seconds relative to the score event's onset.
    /// Multiple syllables may occupy disjoint sub-intervals of one long note.
    pub start_offset_s: f32,
    pub end_offset_s: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyllableNoteBinding {
    pub word_index: usize,
    pub syllable_index: usize,
    /// One or more ordered intervals within sung score events. One syllable
    /// may span multiple consecutive notes (melisma); several syllables may
    /// occupy different intervals inside the same note.
    pub spans: Vec<ScoreEventSpan>,
    /// Required when this syllable spans multiple score events. It identifies
    /// the exact vowel nucleus sustained across those events.
    pub melisma_phone_index: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VocalPhraseSpan {
    /// Half-open range into the flattened pronunciation syllable sequence.
    pub syllable_start: usize,
    pub syllable_end: usize,
    pub breath_before_s: f32,
    pub breath_after_s: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VocalScoreAlignment {
    pub schema_version: u32,
    pub events: Vec<ScoreEvent>,
    pub syllable_bindings: Vec<SyllableNoteBinding>,
    pub phrases: Vec<VocalPhraseSpan>,
}

impl VocalScoreAlignment {
    /// Validate the score mapping against a fully resolved pronunciation plan.
    ///
    /// Every selected syllable is represented once. Every sung note interval
    /// is covered exactly once by one or more syllable intervals. This permits
    /// both one-syllable-to-many-notes melisma and many-syllables-to-one-note
    /// subdivision, while forbidding overlap, gaps, and silent event loss.
    /// Rests remain explicit and unbound.
    pub fn validate_against(&self, pronunciation: &PronunciationPlan) -> Result<(), String> {
        const TIME_EPSILON_S: f32 = 0.000_01;

        pronunciation.validate_renderable()?;
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported vocal score-alignment schema version {}",
                self.schema_version
            ));
        }
        if self.events.is_empty() || self.syllable_bindings.is_empty() || self.phrases.is_empty() {
            return Err("score events, syllable bindings, and phrases are required".into());
        }
        if self.events[0].start_s.abs() > TIME_EPSILON_S {
            return Err(
                "score must begin at time zero; represent an opening gap as an explicit Rest event"
                    .into(),
            );
        }

        let mut previous_event_end = 0.0f32;
        for (event_index, event) in self.events.iter().enumerate() {
            if !event.start_s.is_finite()
                || event.start_s < 0.0
                || !event.duration_s.is_finite()
                || event.duration_s <= 0.0
            {
                return Err(format!("score event {event_index} has invalid timing"));
            }
            let end = event.start_s + event.duration_s;
            if !end.is_finite() || end <= event.start_s {
                return Err(format!("score event {event_index} has invalid end time"));
            }
            if event.start_s < previous_event_end - TIME_EPSILON_S {
                return Err(format!(
                    "score event {event_index} overlaps or is out of chronological order"
                ));
            }
            if event_index > 0 && (event.start_s - previous_event_end).abs() > TIME_EPSILON_S {
                return Err(format!(
                    "score event {event_index} has an implicit gap; represent it as an explicit Rest event"
                ));
            }
            if let ScoreEventKind::SungNote {
                frequency_hz,
                velocity,
            } = event.kind
            {
                if !frequency_hz.is_finite()
                    || frequency_hz <= 0.0
                    || !velocity.is_finite()
                    || !(0.0..=1.0).contains(&velocity)
                {
                    return Err(format!(
                        "sung score event {event_index} has invalid pitch/velocity"
                    ));
                }
            }
            previous_event_end = end;
        }

        let mut expected_syllables: Vec<(usize, usize)> = Vec::new();
        for (word_index, word) in pronunciation.words.iter().enumerate() {
            let Some(selected_id) = word.selected_candidate_id.as_deref() else {
                return Err(format!("word {word_index} has no selected pronunciation"));
            };
            let Some(selected) = word
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == selected_id)
            else {
                return Err(format!("word {word_index} selects a missing pronunciation"));
            };
            for syllable_index in 0..selected.syllables.len() {
                expected_syllables.push((word_index, syllable_index));
            }
        }

        if self.syllable_bindings.len() != expected_syllables.len() {
            return Err(format!(
                "syllable binding count {} does not match selected pronunciation count {}",
                self.syllable_bindings.len(),
                expected_syllables.len()
            ));
        }

        // Collect every sub-interval assigned within each score event. The
        // final coverage pass proves that intervals partition each sung note.
        let mut event_coverage: Vec<Vec<(f32, f32, usize)>> = vec![Vec::new(); self.events.len()];
        let mut previous_binding_end_s: Option<f32> = None;

        for (binding_index, (binding, expected)) in self
            .syllable_bindings
            .iter()
            .zip(expected_syllables.iter())
            .enumerate()
        {
            if (binding.word_index, binding.syllable_index) != *expected {
                return Err(format!(
                    "syllable binding {binding_index} is out of pronunciation order; expected word {} syllable {}",
                    expected.0, expected.1
                ));
            }
            if binding.spans.is_empty() {
                return Err(format!(
                    "syllable binding {binding_index} has no score spans"
                ));
            }

            let mut previous_span: Option<&ScoreEventSpan> = None;
            let mut seen_events = std::collections::BTreeSet::new();
            for span in &binding.spans {
                let Some(event) = self.events.get(span.event_index) else {
                    return Err(format!(
                        "syllable binding {binding_index} references missing score event {}",
                        span.event_index
                    ));
                };
                if !matches!(event.kind, ScoreEventKind::SungNote { .. }) {
                    return Err(format!(
                        "syllable binding {binding_index} assigns a rest event {}",
                        span.event_index
                    ));
                }
                if !span.start_offset_s.is_finite()
                    || !span.end_offset_s.is_finite()
                    || span.start_offset_s < 0.0
                    || span.end_offset_s <= span.start_offset_s
                    || span.end_offset_s > event.duration_s + TIME_EPSILON_S
                {
                    return Err(format!(
                        "syllable binding {binding_index} has an invalid interval in score event {}",
                        span.event_index
                    ));
                }
                if !seen_events.insert(span.event_index) {
                    return Err(format!(
                        "syllable binding {binding_index} assigns the same score event more than once"
                    ));
                }
                if let Some(previous) = previous_span {
                    if span.event_index != previous.event_index + 1 {
                        return Err(format!(
                            "syllable binding {binding_index} multi-note span must use consecutive score events"
                        ));
                    }
                    let previous_event = &self.events[previous.event_index];
                    if previous.end_offset_s + TIME_EPSILON_S < previous_event.duration_s
                        || span.start_offset_s > TIME_EPSILON_S
                    {
                        return Err(format!(
                            "syllable binding {binding_index} has a gap between consecutive melisma notes"
                        ));
                    }
                }

                let event_start_s = event.start_s + span.start_offset_s;
                let event_end_s = event.start_s + span.end_offset_s;
                if previous_binding_end_s
                    .is_some_and(|previous_end| event_start_s < previous_end - TIME_EPSILON_S)
                {
                    return Err(format!(
                        "syllable binding {binding_index} overlaps or reorders a preceding syllable"
                    ));
                }
                previous_binding_end_s = Some(event_end_s);
                event_coverage[span.event_index].push((
                    span.start_offset_s,
                    span.end_offset_s.min(event.duration_s),
                    binding_index,
                ));
                previous_span = Some(span);
            }

            let word = &pronunciation.words[binding.word_index];
            let Some(selected_id) = word.selected_candidate_id.as_deref() else {
                return Err(format!(
                    "word {} has no selected pronunciation",
                    binding.word_index
                ));
            };
            let Some(selected) = word
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == selected_id)
            else {
                return Err(format!(
                    "word {} selects a missing pronunciation",
                    binding.word_index
                ));
            };
            let Some(syllable) = selected.syllables.get(binding.syllable_index) else {
                return Err(format!(
                    "syllable binding {binding_index} references a missing pronunciation syllable"
                ));
            };
            if binding.spans.len() > 1 {
                let Some(phone_index) = binding.melisma_phone_index else {
                    return Err(format!(
                        "multi-note syllable binding {binding_index} requires an explicit melisma vowel"
                    ));
                };
                let Some(phone) = syllable.phones.get(phone_index) else {
                    return Err(format!(
                        "syllable binding {binding_index} melisma phone index is out of range"
                    ));
                };
                if phone.kind != PhoneKind::Vowel {
                    return Err(format!(
                        "syllable binding {binding_index} melisma must sustain a vowel phone"
                    ));
                }
            } else if binding.melisma_phone_index.is_some() {
                return Err(format!(
                    "single-event syllable binding {binding_index} must not declare a melisma phone"
                ));
            }
        }

        for (event_index, event) in self.events.iter().enumerate() {
            match event.kind {
                ScoreEventKind::Rest if !event_coverage[event_index].is_empty() => {
                    return Err(format!("rest event {event_index} cannot be bound"));
                }
                ScoreEventKind::SungNote { .. } => {
                    let mut segments = event_coverage[event_index].clone();
                    segments.sort_by(|left, right| left.0.total_cmp(&right.0));
                    let mut covered_until = 0.0f32;
                    for (start_s, end_s, binding_index) in segments {
                        if start_s > covered_until + TIME_EPSILON_S {
                            return Err(format!(
                                "sung score event {event_index} has an unassigned interval before syllable binding {binding_index}"
                            ));
                        }
                        if start_s < covered_until - TIME_EPSILON_S {
                            return Err(format!(
                                "sung score event {event_index} has overlapping syllable intervals at binding {binding_index}"
                            ));
                        }
                        covered_until = end_s;
                    }
                    if (covered_until - event.duration_s).abs() > TIME_EPSILON_S {
                        return Err(format!(
                            "sung score event {event_index} is not fully covered by syllable intervals"
                        ));
                    }
                }
                ScoreEventKind::Rest => {}
            }
        }

        let mut next_syllable = 0usize;
        for (phrase_index, phrase) in self.phrases.iter().enumerate() {
            if phrase.syllable_start != next_syllable
                || phrase.syllable_start >= phrase.syllable_end
                || phrase.syllable_end > expected_syllables.len()
                || !phrase.breath_before_s.is_finite()
                || phrase.breath_before_s < 0.0
                || !phrase.breath_after_s.is_finite()
                || phrase.breath_after_s < 0.0
            {
                return Err(format!(
                    "phrase {phrase_index} has invalid or non-partitioning syllable range/breath timing"
                ));
            }
            next_syllable = phrase.syllable_end;
        }
        if next_syllable != expected_syllables.len() {
            return Err("phrase ranges do not cover every pronunciation syllable".into());
        }
        Ok(())
    }
}

/// Typed articulatory/phonological features attached to a language-specific
/// inventory phone. Multi-character orthographic graphemes are normalized to
/// one of these IPA tokens before a PronunciationPlan is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneFeature {
    Voiced,
    Nasal,
    Syllabic,
    Aspirated,
    Ejective,
    Implosive,
    Click,
    ClickDental,
    ClickAlveolar,
    ClickPostAlveolar,
    ClickLateral,
    LengthLong,
    LengthShort,
    VowelHigh,
    VowelMid,
    VowelLow,
    VowelFront,
    VowelCentral,
    VowelBack,
    Rounded,
    PlaceBilabial,
    PlaceLabiodental,
    PlaceDental,
    PlaceAlveolar,
    PlacePostAlveolar,
    PlacePalatal,
    PlaceVelar,
    PlaceUvular,
    PlaceGlottal,
}

/// One canonical phone in a versioned language/dialect inventory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageProfilePhone {
    pub ipa: String,
    pub kind: PhoneKind,
    pub features: Vec<PhoneFeature>,
}

/// Evidence and rights metadata for one external lexical/phonological resource.
/// The content is referenced by immutable source revision and blob digest; it
/// is never bundled implicitly by including this receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageResourceReceipt {
    pub resource_id: String,
    pub source_repository: String,
    pub source_revision: String,
    pub resource_path: String,
    pub source_digest: String,
    pub license_document_path: String,
    pub license_document_digest: String,
    pub license_expression: String,
    pub attribution: String,
    pub rights_status: ResourceRightsStatus,
    pub legal_review_reference: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceRightsStatus {
    MetadataAudited,
    LegalReviewedForDistribution,
    Excluded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageTask {
    GraphemeToPhoneme,
    Syllabification,
    Morphology,
    TonePrediction,
    SpeechSynthesis,
    SingingSynthesis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageTaskStatus {
    NotEvaluated,
    ResearchOnly,
    Qualified,
    Unsupported,
}

/// Capability status is per task. G2P or ASR support never implies singing
/// support; Qualified requires an external evidence receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageTaskCapability {
    pub task: LanguageTask,
    pub status: LanguageTaskStatus,
    pub evidence_reference: Option<String>,
    pub evidence_digest: Option<String>,
}

/// Canonical, versioned dialect/orthography/phone inventory contract.
///
/// The digest covers a canonical serialization of every field except
/// profile_digest itself. The profile loader/registry computes and verifies
/// the digest over the exact stored manifest bytes before passing this value
/// to the contract; this crate validates the digest format and its binding
/// to a pronunciation plan, but does not recanonicalize external JSON.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageProfileManifest {
    pub schema_version: u32,
    pub profile_id: String,
    pub profile_digest: String,
    pub language_tag: String,
    pub dialect: String,
    pub orthography_id: String,
    pub inventory_version: String,
    pub phone_inventory: Vec<LanguageProfilePhone>,
    pub resources: Vec<LanguageResourceReceipt>,
    pub capabilities: Vec<LanguageTaskCapability>,
}

/// A pronunciation plan bound to one exact language-profile identity. The
/// complete profile is supplied separately at validation/render time so the
/// plan can remain compact while carrying an immutable profile reference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfiledPronunciationPlan {
    pub schema_version: u32,
    pub language_profile_id: String,
    pub language_profile_digest: String,
    pub pronunciation: PronunciationPlan,
}

impl LanguageProfileManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported language-profile schema version {}",
                self.schema_version
            ));
        }
        if self.profile_id.trim().is_empty()
            || self.language_tag.trim().is_empty()
            || self
                .language_tag
                .bytes()
                .any(|byte| byte.is_ascii_whitespace())
            || self.dialect.trim().is_empty()
            || self.orthography_id.trim().is_empty()
            || self.inventory_version.trim().is_empty()
        {
            return Err("language profile requires ID, exact source Git revision, BCP-47 tag, dialect, orthography and inventory version".into());
        }
        if !valid_source_digest(&self.profile_digest) {
            return Err("language profile requires a valid canonical profile digest".into());
        }
        if self.phone_inventory.is_empty() {
            return Err("language profile requires a non-empty phone inventory".into());
        }
        let mut phones = std::collections::BTreeSet::new();
        let mut has_vowel = false;
        let mut has_consonant = false;
        for phone in &self.phone_inventory {
            if phone.ipa.trim().is_empty()
                || phone.ipa.chars().any(char::is_whitespace)
                || !phones.insert(phone.ipa.as_str())
            {
                return Err(format!(
                    "language profile {} has an empty, whitespace-containing or duplicate IPA phone",
                    self.profile_id
                ));
            }
            if phone.features.is_empty() {
                return Err(format!(
                    "phone {:?} has no articulatory features",
                    phone.ipa
                ));
            }
            let features: std::collections::BTreeSet<_> = phone.features.iter().copied().collect();
            if features.len() != phone.features.len() {
                return Err(format!("phone {:?} has duplicate features", phone.ipa));
            }
            match phone.kind {
                PhoneKind::Vowel => {
                    has_vowel = true;
                    if features.iter().any(|feature| {
                        matches!(
                            feature,
                            PhoneFeature::Click
                                | PhoneFeature::ClickDental
                                | PhoneFeature::ClickAlveolar
                                | PhoneFeature::ClickPostAlveolar
                                | PhoneFeature::ClickLateral
                                | PhoneFeature::Aspirated
                                | PhoneFeature::Ejective
                                | PhoneFeature::Implosive
                        )
                    }) {
                        return Err(format!("vowel {:?} has consonant-only features", phone.ipa));
                    }
                }
                PhoneKind::Consonant => {
                    has_consonant = true;
                    if features.contains(&PhoneFeature::Syllabic) {
                        return Err(format!(
                            "phone {:?} is marked syllabic but its kind is consonant",
                            phone.ipa
                        ));
                    }
                }
                PhoneKind::SyllabicConsonant => {
                    has_consonant = true;
                    if !features.contains(&PhoneFeature::Syllabic) {
                        return Err(format!(
                            "syllabic consonant {:?} lacks the syllabic feature",
                            phone.ipa
                        ));
                    }
                }
            }
            let click_places = [
                PhoneFeature::ClickDental,
                PhoneFeature::ClickAlveolar,
                PhoneFeature::ClickPostAlveolar,
                PhoneFeature::ClickLateral,
            ]
            .iter()
            .filter(|feature| features.contains(feature))
            .count();
            if features.contains(&PhoneFeature::Click) != (click_places == 1) {
                return Err(format!(
                    "click phone {:?} must have exactly one explicit click place",
                    phone.ipa
                ));
            }
            if click_places > 0 && phone.kind != PhoneKind::Consonant {
                return Err(format!("click phone {:?} must be a consonant", phone.ipa));
            }
        }
        if !has_vowel || !has_consonant {
            return Err("language profile inventory must include vowels and consonants".into());
        }

        if self.resources.is_empty() {
            return Err("language profile requires explicit resource/license receipts".into());
        }
        let mut resource_ids = std::collections::BTreeSet::new();
        for resource in &self.resources {
            if resource.resource_id.trim().is_empty()
                || !resource_ids.insert(resource.resource_id.as_str())
                || resource.source_repository.trim().is_empty()
                || !valid_source_digest(&format!("git-sha1:{}", resource.source_revision))
                || !valid_source_digest(&resource.source_digest)
                || !valid_source_digest(&resource.license_document_digest)
                || resource.license_expression.trim().is_empty()
                || resource.license_expression == "NOASSERTION"
                || resource.attribution.trim().is_empty()
                || !safe_resource_path(&resource.resource_path)
                || !safe_resource_path(&resource.license_document_path)
            {
                return Err(format!(
                    "resource {:?} has incomplete provenance/license metadata",
                    resource.resource_id
                ));
            }
            match resource.rights_status {
                ResourceRightsStatus::Excluded => {
                    return Err(format!(
                        "excluded resource {:?} cannot be present in an active language profile",
                        resource.resource_id
                    ));
                }
                ResourceRightsStatus::LegalReviewedForDistribution => {
                    if resource
                        .legal_review_reference
                        .as_deref()
                        .is_none_or(str::is_empty)
                    {
                        return Err(format!(
                            "resource {:?} claims legal review without an evidence reference",
                            resource.resource_id
                        ));
                    }
                }
                ResourceRightsStatus::MetadataAudited => {}
            }
        }

        let mut tasks = std::collections::BTreeSet::new();
        for capability in &self.capabilities {
            if !tasks.insert(capability.task) {
                return Err(format!(
                    "language profile has duplicate capability {:?}",
                    capability.task
                ));
            }
            if let Some(reference) = capability.evidence_reference.as_deref()
                && reference.trim().is_empty()
            {
                return Err(format!(
                    "capability {:?} has an empty evidence reference",
                    capability.task
                ));
            }
            if let Some(digest) = capability.evidence_digest.as_deref()
                && !valid_source_digest(digest)
            {
                return Err(format!(
                    "capability {:?} has an invalid evidence digest",
                    capability.task
                ));
            }
            if capability.status == LanguageTaskStatus::Qualified
                && (capability
                    .evidence_reference
                    .as_deref()
                    .is_none_or(str::is_empty)
                    || capability.evidence_digest.is_none())
            {
                return Err(format!(
                    "qualified capability {:?} requires evidence reference and digest",
                    capability.task
                ));
            }
        }
        for required in [
            LanguageTask::GraphemeToPhoneme,
            LanguageTask::Syllabification,
        ] {
            if !tasks.contains(&required) {
                return Err(format!(
                    "language profile must explicitly state {required:?} capability"
                ));
            }
        }
        Ok(())
    }

    pub fn capability(&self, task: LanguageTask) -> Option<&LanguageTaskCapability> {
        self.capabilities
            .iter()
            .find(|capability| capability.task == task)
    }
}

fn safe_resource_path(value: &str) -> bool {
    !value.trim().is_empty()
        && !value.starts_with('/')
        && !value.contains('\\')
        && value
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

impl ProfiledPronunciationPlan {
    pub fn validate_against(&self, profile: &LanguageProfileManifest) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported profiled-pronunciation schema version {}",
                self.schema_version
            ));
        }
        profile.validate()?;
        self.pronunciation.validate()?;
        if self.language_profile_id != profile.profile_id
            || self.language_profile_digest != profile.profile_digest
        {
            return Err("pronunciation plan language-profile ID/digest mismatch".into());
        }
        if self.pronunciation.language != profile.language_tag {
            return Err(format!(
                "pronunciation language {:?} does not match profile language {:?}",
                self.pronunciation.language, profile.language_tag
            ));
        }

        for (word_index, word) in self.pronunciation.words.iter().enumerate() {
            for candidate in &word.candidates {
                for (syllable_index, syllable) in candidate.syllables.iter().enumerate() {
                    for (phone_index, phone) in syllable.phones.iter().enumerate() {
                        let Some(profile_phone) = profile
                            .phone_inventory
                            .iter()
                            .find(|profile_phone| profile_phone.ipa == phone.ipa)
                        else {
                            return Err(format!(
                                "word {word_index} candidate {} syllable {syllable_index} phone {phone_index} {:?} is outside profile {} inventory",
                                candidate.candidate_id, phone.ipa, profile.profile_id
                            ));
                        };
                        if profile_phone.kind != phone.kind {
                            return Err(format!(
                                "word {word_index} candidate {} phone {:?} kind {:?} conflicts with profile kind {:?}",
                                candidate.candidate_id, phone.ipa, phone.kind, profile_phone.kind
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// A task is production-qualified only when the exact language profile
    /// carries a Qualified status with an evidence receipt. ResearchOnly and
    /// NotEvaluated profiles may still be used in explicitly labelled studies.
    pub fn validate_for_task(
        &self,
        profile: &LanguageProfileManifest,
        task: LanguageTask,
    ) -> Result<(), String> {
        self.validate_against(profile)?;
        self.pronunciation.validate_renderable()?;
        let capability = profile.capability(task).ok_or_else(|| {
            format!(
                "language profile {} does not declare capability {task:?}",
                profile.profile_id
            )
        })?;
        if capability.status != LanguageTaskStatus::Qualified {
            return Err(format!(
                "language profile {} task {task:?} is {:?}, not Qualified",
                profile.profile_id, capability.status
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance(source: CandidateSource, provider: &str) -> CandidateProvenance {
        CandidateProvenance {
            source,
            provider_id: provider.into(),
            provider_version: "1.0.0".into(),
            source_digest: format!("blake3:{}", "0".repeat(64)),
        }
    }

    fn candidate(id: &str, vowel: &str, source: CandidateSource) -> PronunciationCandidate {
        PronunciationCandidate {
            candidate_id: id.into(),
            provenance: provenance(source, "fixture-g2p"),
            raw_provider_score: Some(0.42),
            syllables: vec![PronunciationSyllable {
                stress: 1,
                phones: vec![
                    Phone {
                        ipa: "ɹ".into(),
                        kind: PhoneKind::Consonant,
                    },
                    Phone {
                        ipa: vowel.into(),
                        kind: PhoneKind::Vowel,
                    },
                    Phone {
                        ipa: "d".into(),
                        kind: PhoneKind::Consonant,
                    },
                ],
            }],
        }
    }

    fn word(
        surface: &str,
        start_byte: usize,
        state: ResolutionState,
        candidates: Vec<PronunciationCandidate>,
    ) -> WordPronunciation {
        WordPronunciation {
            start_byte,
            end_byte: start_byte + surface.len(),
            surface: surface.into(),
            state,
            candidates,
            selected_candidate_id: None,
            selection_receipt: None,
        }
    }

    fn xhosa_profile_fixture() -> LanguageProfileManifest {
        let source_revision = "59a93768d72b5804f8b87bc3dd409320c8f0c328";
        LanguageProfileManifest {
            schema_version: 1,
            profile_id: "xh-ZA-za-lex-nchlt-2017".into(),
            profile_digest: format!("blake3:{}", "1".repeat(64)),
            language_tag: "xh-ZA".into(),
            dialect: "isiXhosa NCHLT 2013 resource profile; research only".into(),
            orthography_id: "za-lex-xho-graphemeset".into(),
            inventory_version: "za-lex-xho-phonemeset-2017".into(),
            phone_inventory: vec![
                LanguageProfilePhone {
                    ipa: "ǀ".into(),
                    kind: PhoneKind::Consonant,
                    features: vec![PhoneFeature::Click, PhoneFeature::ClickDental],
                },
                LanguageProfilePhone {
                    ipa: "a".into(),
                    kind: PhoneKind::Vowel,
                    features: vec![PhoneFeature::VowelLow, PhoneFeature::VowelBack],
                },
                LanguageProfilePhone {
                    ipa: "m".into(),
                    kind: PhoneKind::SyllabicConsonant,
                    features: vec![
                        PhoneFeature::Syllabic,
                        PhoneFeature::Nasal,
                        PhoneFeature::Voiced,
                    ],
                },
            ],
            resources: vec![LanguageResourceReceipt {
                resource_id: "za-lex-xho-phonemeset".into(),
                source_repository: "https://github.com/NWU-MuST/za_lex".into(),
                source_revision: source_revision.into(),
                resource_path: "data/xho/phonemeset.json".into(),
                source_digest: format!("git-sha1:{}", "2".repeat(40)),
                license_document_path: "data/xho/COPYRIGHT".into(),
                license_document_digest: format!("git-sha1:{}", "3".repeat(40)),
                license_expression: "CC-BY-3.0".into(),
                attribution: "DAC, CSIR and NWU".into(),
                rights_status: ResourceRightsStatus::MetadataAudited,
                legal_review_reference: None,
            }],
            capabilities: vec![
                LanguageTaskCapability {
                    task: LanguageTask::GraphemeToPhoneme,
                    status: LanguageTaskStatus::ResearchOnly,
                    evidence_reference: Some("upstream-rule-fixture:2017".into()),
                    evidence_digest: Some(format!("git-sha1:{}", "4".repeat(40))),
                },
                LanguageTaskCapability {
                    task: LanguageTask::Syllabification,
                    status: LanguageTaskStatus::ResearchOnly,
                    evidence_reference: Some("upstream-syllabifier-fixture:2017".into()),
                    evidence_digest: Some(format!("git-sha1:{}", "5".repeat(40))),
                },
                LanguageTaskCapability {
                    task: LanguageTask::SingingSynthesis,
                    status: LanguageTaskStatus::ResearchOnly,
                    evidence_reference: None,
                    evidence_digest: None,
                },
                LanguageTaskCapability {
                    task: LanguageTask::Morphology,
                    status: LanguageTaskStatus::NotEvaluated,
                    evidence_reference: None,
                    evidence_digest: None,
                },
                LanguageTaskCapability {
                    task: LanguageTask::TonePrediction,
                    status: LanguageTaskStatus::Unsupported,
                    evidence_reference: Some("NCHLT-2013-phone-set-does-not-encode-tone".into()),
                    evidence_digest: None,
                },
                LanguageTaskCapability {
                    task: LanguageTask::SpeechSynthesis,
                    status: LanguageTaskStatus::NotEvaluated,
                    evidence_reference: None,
                    evidence_digest: None,
                },
            ],
        }
    }

    fn xhosa_profiled_plan() -> ProfiledPronunciationPlan {
        let profile = xhosa_profile_fixture();
        let mut word = word(
            "ca",
            0,
            ResolutionState::Selected,
            vec![PronunciationCandidate {
                candidate_id: "click-a".into(),
                provenance: provenance(CandidateSource::CommunityReviewed, "xhosa-fixture"),
                raw_provider_score: None,
                syllables: vec![PronunciationSyllable {
                    stress: 0,
                    phones: vec![
                        Phone {
                            ipa: "ǀ".into(),
                            kind: PhoneKind::Consonant,
                        },
                        Phone {
                            ipa: "a".into(),
                            kind: PhoneKind::Vowel,
                        },
                    ],
                }],
            }],
        );
        word.selected_candidate_id = Some("click-a".into());
        word.selection_receipt = Some(SelectionReceipt {
            authority: SelectionAuthority::CommunityReview,
            evidence_reference: "review-fixture:xhosa-click-ca".into(),
        });
        ProfiledPronunciationPlan {
            schema_version: 1,
            language_profile_id: profile.profile_id.clone(),
            language_profile_digest: profile.profile_digest.clone(),
            pronunciation: PronunciationPlan {
                schema_version: 1,
                language: "xh-ZA".into(),
                source_text: "ca".into(),
                words: vec![word],
            },
        }
    }

    #[test]
    fn versioned_language_profile_accepts_click_phone_and_exact_plan_binding() {
        let profile = xhosa_profile_fixture();
        let plan = xhosa_profiled_plan();
        assert!(profile.validate().is_ok());
        assert!(plan.validate_against(&profile).is_ok());
    }

    #[test]
    fn profiled_plan_rejects_unknown_phone_wrong_language_and_wrong_digest() {
        let profile = xhosa_profile_fixture();
        let mut unknown_phone = xhosa_profiled_plan();
        unknown_phone.pronunciation.words[0].candidates[0].syllables[0].phones[0].ipa = "ǁ".into();
        assert!(
            unknown_phone
                .validate_against(&profile)
                .unwrap_err()
                .contains("outside profile")
        );

        let mut wrong_language = xhosa_profiled_plan();
        wrong_language.pronunciation.language = "en-ZA".into();
        assert!(
            wrong_language
                .validate_against(&profile)
                .unwrap_err()
                .contains("does not match profile language")
        );

        let mut wrong_digest = xhosa_profiled_plan();
        wrong_digest.language_profile_digest = format!("blake3:{}", "f".repeat(64));
        assert!(
            wrong_digest
                .validate_against(&profile)
                .unwrap_err()
                .contains("ID/digest mismatch")
        );
    }

    #[test]
    fn profile_rejects_click_without_unique_place_and_syllabic_phone_without_feature() {
        let mut profile = xhosa_profile_fixture();
        profile.phone_inventory[0].features = vec![PhoneFeature::Click];
        assert!(
            profile
                .validate()
                .unwrap_err()
                .contains("exactly one explicit click place")
        );

        let mut profile = xhosa_profile_fixture();
        profile.phone_inventory[2].features = vec![PhoneFeature::Nasal, PhoneFeature::Voiced];
        assert!(
            profile
                .validate()
                .unwrap_err()
                .contains("lacks the syllabic feature")
        );
    }

    #[test]
    fn metadata_only_profile_cannot_qualify_singing_rendering() {
        let profile = xhosa_profile_fixture();
        let plan = xhosa_profiled_plan();
        let error = plan
            .validate_for_task(&profile, LanguageTask::SingingSynthesis)
            .unwrap_err();
        assert!(error.contains("ResearchOnly, not Qualified"), "{error}");
    }

    #[test]
    fn qualified_language_capability_requires_evidence_receipt() {
        let mut profile = xhosa_profile_fixture();
        let singing = profile
            .capabilities
            .iter_mut()
            .find(|capability| capability.task == LanguageTask::SingingSynthesis)
            .unwrap();
        singing.status = LanguageTaskStatus::Qualified;
        singing.evidence_reference = None;
        singing.evidence_digest = None;
        assert!(
            profile
                .validate()
                .unwrap_err()
                .contains("requires evidence reference and digest")
        );
    }

    #[test]
    fn language_profile_requires_an_exact_immutable_resource_revision() {
        let mut profile = xhosa_profile_fixture();
        profile.resources[0].source_revision = "main".into();
        let error = profile.validate().unwrap_err();
        assert!(error.contains("incomplete provenance/license metadata"), "{error}");

        let mut profile = xhosa_profile_fixture();
        profile.resources[0].source_revision = "59A93768D72B5804F8B87BC3DD409320C8F0C328".into();
        let error = profile.validate().unwrap_err();
        assert!(error.contains("incomplete provenance/license metadata"), "{error}");
    }

    #[test]
    fn language_profile_rejects_excluded_or_unlicensed_resource() {
        let mut profile = xhosa_profile_fixture();
        profile.resources[0].rights_status = ResourceRightsStatus::Excluded;
        assert!(
            profile
                .validate()
                .unwrap_err()
                .contains("excluded resource")
        );

        let mut profile = xhosa_profile_fixture();
        profile.resources[0].license_expression = "NOASSERTION".into();
        assert!(
            profile
                .validate()
                .unwrap_err()
                .contains("incomplete provenance/license metadata")
        );
    }

    #[test]
    fn contextual_heteronym_retains_multiple_candidates_until_review() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![word(
                "read",
                0,
                ResolutionState::NeedsReview,
                vec![
                    candidate("past", "ɛ", CandidateSource::Dictionary),
                    candidate("present", "iː", CandidateSource::ModelG2p),
                ],
            )],
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn selected_candidate_requires_a_receipt() {
        let mut selected = word(
            "read",
            0,
            ResolutionState::Selected,
            vec![candidate("present", "iː", CandidateSource::Dictionary)],
        );
        selected.selected_candidate_id = Some("present".into());
        selected.selection_receipt = Some(SelectionReceipt {
            authority: SelectionAuthority::HumanReviewer,
            evidence_reference: "review:fixture-001".into(),
        });
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![selected],
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn out_of_vocabulary_is_explicit_and_contains_no_guessed_phone() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "xqzpl".into(),
            words: vec![word("xqzpl", 0, ResolutionState::OutOfVocabulary, vec![])],
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn unrepresented_source_text_cannot_be_silently_dropped() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "light bright".into(),
            words: vec![word(
                "light",
                0,
                ResolutionState::NeedsReview,
                vec![candidate("light-1", "aɪ", CandidateSource::Dictionary)],
            )],
        };
        let error = plan.validate().unwrap_err();
        assert!(error.contains("not covered"), "{error}");
    }

    #[test]
    fn one_pronunciation_span_cannot_merge_multiple_whitespace_separated_words() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "hello world".into(),
            words: vec![word(
                "hello world",
                0,
                ResolutionState::NeedsReview,
                vec![candidate("merged", "ɛ", CandidateSource::Dictionary)],
            )],
        };
        let error = plan.validate().unwrap_err();
        assert!(error.contains("contains whitespace"), "{error}");
    }

    #[test]
    fn spans_must_be_valid_utf8_boundaries_and_match_the_surface() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "café".into(),
            words: vec![word(
                "caf",
                0,
                ResolutionState::NeedsReview,
                vec![candidate("cafe", "æ", CandidateSource::Dictionary)],
            )],
        };
        assert!(plan.validate().is_err());

        let split = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "café".into(),
            words: vec![WordPronunciation {
                start_byte: 0,
                end_byte: 4,
                surface: "caf".into(),
                state: ResolutionState::NeedsReview,
                candidates: vec![candidate("cafe", "æ", CandidateSource::Dictionary)],
                selected_candidate_id: None,
                selection_receipt: None,
            }],
        };
        assert!(split.validate().is_err());
    }

    #[test]
    fn unresolved_or_unknown_candidate_cannot_be_marked_selected() {
        let mut unknown = word(
            "read",
            0,
            ResolutionState::Selected,
            vec![candidate("past", "ɛ", CandidateSource::Dictionary)],
        );
        unknown.selected_candidate_id = Some("missing".into());
        unknown.selection_receipt = Some(SelectionReceipt {
            authority: SelectionAuthority::User,
            evidence_reference: "user-choice:1".into(),
        });
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "read".into(),
            words: vec![unknown],
        };
        assert!(
            plan.validate()
                .unwrap_err()
                .contains("unknown pronunciation")
        );
    }

    #[test]
    fn phones_must_be_normalized_tokens_and_syllables_need_a_nucleus() {
        let mut invalid = candidate("bad", "ɛ", CandidateSource::Dictionary);
        invalid.syllables[0].phones[1].ipa = "ˈɛ".into();
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "read".into(),
            words: vec![word("read", 0, ResolutionState::NeedsReview, vec![invalid])],
        };
        assert!(plan.validate().unwrap_err().contains("normalized IPA"));

        let mut no_nucleus = candidate("no-nucleus", "ɛ", CandidateSource::Dictionary);
        no_nucleus.syllables[0]
            .phones
            .retain(|phone| phone.kind == PhoneKind::Consonant);
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "read".into(),
            words: vec![word(
                "read",
                0,
                ResolutionState::NeedsReview,
                vec![no_nucleus],
            )],
        };
        assert!(plan.validate().unwrap_err().contains("no vowel"));
    }

    #[test]
    fn non_finite_provider_scores_and_incomplete_provenance_fail_closed() {
        assert!(!valid_source_digest(""));
        assert!(!valid_source_digest("fixture"));
        assert!(!valid_source_digest("blake3:fixture"));
        assert!(!valid_source_digest(&format!("blake3:{}", "A".repeat(64))));
        assert!(!valid_source_digest(&format!("md5:{}", "0".repeat(32))));
        assert!(valid_source_digest(&format!("blake3:{}", "0".repeat(64))));
        assert!(valid_source_digest(&format!("sha256:{}", "a".repeat(64))));
        assert!(valid_source_digest(&format!("git-sha1:{}", "1".repeat(40))));

        let mut bad_score = candidate("bad-score", "ɛ", CandidateSource::ModelG2p);
        bad_score.raw_provider_score = Some(f32::NAN);
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "read".into(),
            words: vec![word(
                "read",
                0,
                ResolutionState::NeedsReview,
                vec![bad_score],
            )],
        };
        assert!(plan.validate().unwrap_err().contains("non-finite"));

        let mut bad_source = candidate("bad-source", "ɛ", CandidateSource::ModelG2p);
        bad_source.provenance.source_digest.clear();
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "read".into(),
            words: vec![word(
                "read",
                0,
                ResolutionState::NeedsReview,
                vec![bad_source],
            )],
        };
        assert!(plan.validate().unwrap_err().contains("provenance digest"));
    }
    #[test]
    fn unresolved_heteronym_is_valid_but_not_renderable() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![word(
                "read",
                0,
                ResolutionState::NeedsReview,
                vec![
                    candidate("past", "ɛ", CandidateSource::Dictionary),
                    candidate("present", "iː", CandidateSource::ModelG2p),
                ],
            )],
        };
        assert!(plan.validate().is_ok());
        assert!(
            plan.validate_renderable()
                .unwrap_err()
                .contains("rendering requires")
        );
    }

    #[test]
    fn resolved_pronunciation_is_renderable() {
        let mut selected = word(
            "read",
            0,
            ResolutionState::Selected,
            vec![candidate("past", "ɛ", CandidateSource::Dictionary)],
        );
        selected.selected_candidate_id = Some("past".into());
        selected.selection_receipt = Some(SelectionReceipt {
            authority: SelectionAuthority::HumanReviewer,
            evidence_reference: "review:read-past-001".into(),
        });
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![selected],
        };
        assert!(plan.validate_renderable().is_ok());
    }

    fn selected_word(surface: &str, start_byte: usize, vowel: &str) -> WordPronunciation {
        let mut candidate = PronunciationCandidate {
            candidate_id: format!("{surface}-variant-1"),
            provenance: provenance(CandidateSource::HumanReviewed, "review-fixture"),
            raw_provider_score: None,
            syllables: vec![PronunciationSyllable {
                stress: 1,
                phones: vec![
                    Phone {
                        ipa: "ɹ".into(),
                        kind: PhoneKind::Consonant,
                    },
                    Phone {
                        ipa: vowel.into(),
                        kind: PhoneKind::Vowel,
                    },
                    Phone {
                        ipa: "d".into(),
                        kind: PhoneKind::Consonant,
                    },
                ],
            }],
        };
        candidate.provenance.source_digest = format!("blake3:{}", "0".repeat(64));
        WordPronunciation {
            start_byte,
            end_byte: start_byte + surface.len(),
            surface: surface.into(),
            state: ResolutionState::Selected,
            selected_candidate_id: Some(candidate.candidate_id.clone()),
            candidates: vec![candidate],
            selection_receipt: Some(SelectionReceipt {
                authority: SelectionAuthority::HumanReviewer,
                evidence_reference: "review:fixture".into(),
            }),
        }
    }

    fn alignment_fixture() -> (PronunciationPlan, VocalScoreAlignment) {
        let pronunciation = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read now".into(),
            words: vec![selected_word("read", 0, "ɛ"), selected_word("now", 5, "aʊ")],
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
                    duration_s: 0.4,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 246.94,
                        velocity: 0.7,
                    },
                },
                ScoreEvent {
                    start_s: 0.8,
                    duration_s: 0.4,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 261.63,
                        velocity: 0.7,
                    },
                },
            ],
            syllable_bindings: vec![
                SyllableNoteBinding {
                    word_index: 0,
                    syllable_index: 0,
                    spans: vec![
                        ScoreEventSpan {
                            event_index: 0,
                            start_offset_s: 0.0,
                            end_offset_s: 0.4,
                        },
                        ScoreEventSpan {
                            event_index: 1,
                            start_offset_s: 0.0,
                            end_offset_s: 0.4,
                        },
                    ],
                    melisma_phone_index: Some(1),
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
                breath_after_s: 0.2,
            }],
        };
        (pronunciation, alignment)
    }

    #[test]
    fn medial_melisma_is_valid_and_phrase_partition_is_explicit() {
        let (pronunciation, alignment) = alignment_fixture();
        assert!(alignment.validate_against(&pronunciation).is_ok());
    }

    #[test]
    fn melisma_must_sustain_a_vowel_not_a_consonant() {
        let (pronunciation, mut alignment) = alignment_fixture();
        alignment.syllable_bindings[0].melisma_phone_index = Some(0);
        let error = alignment.validate_against(&pronunciation).unwrap_err();
        assert!(error.contains("must sustain a vowel"), "{error}");
    }

    #[test]
    fn every_sung_note_must_be_assigned_once_and_rests_must_remain_unbound() {
        let (pronunciation, mut alignment) = alignment_fixture();
        alignment.syllable_bindings[0].spans = vec![ScoreEventSpan {
            event_index: 0,
            start_offset_s: 0.0,
            end_offset_s: 0.4,
        }];
        alignment.syllable_bindings[0].melisma_phone_index = None;
        let error = alignment.validate_against(&pronunciation).unwrap_err();
        assert!(error.contains("not fully covered"), "{error}");

        let (pronunciation, mut alignment) = alignment_fixture();
        alignment.events[1].kind = ScoreEventKind::Rest;
        let error = alignment.validate_against(&pronunciation).unwrap_err();
        assert!(error.contains("assigns a rest"), "{error}");
    }

    #[test]
    fn multiple_syllables_can_share_one_long_note_without_overlap() {
        let pronunciation = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read now".into(),
            words: vec![selected_word("read", 0, "ɛ"), selected_word("now", 5, "aʊ")],
        };
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
                breath_after_s: 0.1,
            }],
        };
        assert!(alignment.validate_against(&pronunciation).is_ok());
    }

    #[test]
    fn syllable_intervals_must_cover_each_note_without_gaps_or_overlap() {
        let pronunciation = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read now".into(),
            words: vec![selected_word("read", 0, "ɛ"), selected_word("now", 5, "aʊ")],
        };
        let mut alignment = VocalScoreAlignment {
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
                        end_offset_s: 0.4,
                    }],
                    melisma_phone_index: None,
                },
                SyllableNoteBinding {
                    word_index: 1,
                    syllable_index: 0,
                    spans: vec![ScoreEventSpan {
                        event_index: 0,
                        start_offset_s: 0.5,
                        end_offset_s: 1.0,
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
        let error = alignment.validate_against(&pronunciation).unwrap_err();
        assert!(error.contains("unassigned interval"), "{error}");

        alignment.syllable_bindings[1].spans[0].start_offset_s = 0.35;
        let error = alignment.validate_against(&pronunciation).unwrap_err();
        assert!(
            error.contains("overlaps or reorders a preceding syllable")
                || error.contains("overlapping syllable intervals"),
            "{error}"
        );
    }

    #[test]
    fn melisma_cannot_skip_an_explicit_rest() {
        let pronunciation = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read now".into(),
            words: vec![selected_word("read", 0, "ɛ"), selected_word("now", 5, "aʊ")],
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
                    duration_s: 0.1,
                    kind: ScoreEventKind::Rest,
                },
                ScoreEvent {
                    start_s: 0.5,
                    duration_s: 0.4,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 246.94,
                        velocity: 0.7,
                    },
                },
                ScoreEvent {
                    start_s: 0.9,
                    duration_s: 0.4,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 261.63,
                        velocity: 0.7,
                    },
                },
            ],
            syllable_bindings: vec![
                SyllableNoteBinding {
                    word_index: 0,
                    syllable_index: 0,
                    spans: vec![
                        ScoreEventSpan {
                            event_index: 0,
                            start_offset_s: 0.0,
                            end_offset_s: 0.4,
                        },
                        ScoreEventSpan {
                            event_index: 2,
                            start_offset_s: 0.0,
                            end_offset_s: 0.4,
                        },
                    ],
                    melisma_phone_index: Some(1),
                },
                SyllableNoteBinding {
                    word_index: 1,
                    syllable_index: 0,
                    spans: vec![ScoreEventSpan {
                        event_index: 3,
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
        let error = alignment.validate_against(&pronunciation).unwrap_err();
        assert!(error.contains("consecutive score events"), "{error}");
    }

    #[test]
    fn phrase_ranges_must_partition_every_syllable_once() {
        let (pronunciation, mut alignment) = alignment_fixture();
        alignment.phrases[0].syllable_start = 1;
        let error = alignment.validate_against(&pronunciation).unwrap_err();
        assert!(error.contains("phrase 0"), "{error}");
    }

    #[test]
    fn score_events_must_have_explicit_rests_for_timing_gaps() {
        let (pronunciation, mut alignment) = alignment_fixture();
        alignment.events[1].start_s = 0.5;
        let error = alignment.validate_against(&pronunciation).unwrap_err();
        assert!(error.contains("explicit Rest"), "{error}");

        let (pronunciation, mut alignment) = alignment_fixture();
        alignment.events[0].start_s = 0.25;
        let error = alignment.validate_against(&pronunciation).unwrap_err();
        assert!(error.contains("begin at time zero"), "{error}");
    }

    #[test]
    fn note_assignments_cannot_overlap_or_reorder() {
        let (pronunciation, mut alignment) = alignment_fixture();
        alignment.syllable_bindings[1].spans = vec![ScoreEventSpan {
            event_index: 1,
            start_offset_s: 0.0,
            end_offset_s: 0.4,
        }];
        let error = alignment.validate_against(&pronunciation).unwrap_err();
        assert!(
            error.contains("overlaps or reorders") || error.contains("overlapping"),
            "{error}"
        );
    }
    #[test]
    fn beat_domain_score_integrates_tempo_changes_inside_score_events() {
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
                    duration_beats: 4.0,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 246.94,
                        velocity: 0.7,
                    },
                },
                BeatScoreEvent {
                    start_beat: 6.0,
                    duration_beats: 2.0,
                    kind: ScoreEventKind::Rest,
                },
            ],
        };

        let seconds = score.materialize_seconds().unwrap();
        assert_eq!(seconds.len(), 3);
        assert!((seconds[0].start_s - 0.0).abs() < 1e-6);
        assert!((seconds[0].duration_s - 1.0).abs() < 1e-6);
        assert!((seconds[1].start_s - 1.0).abs() < 1e-6);
        assert!((seconds[1].duration_s - 3.0).abs() < 1e-6);
        assert!((seconds[2].start_s - 4.0).abs() < 1e-6);
        assert!((seconds[2].duration_s - 2.0).abs() < 1e-6);
        assert!(matches!(seconds[2].kind, ScoreEventKind::Rest));
        assert!((score.seconds_at_beat(8.0).unwrap() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn beat_domain_score_rejects_implicit_gaps_and_malformed_tempo_maps() {
        let mut score = BeatDomainScore {
            schema_version: 1,
            tempo_map: vec![TempoChange {
                start_beat: 0.0,
                quarter_bpm: 120.0,
            }],
            events: vec![
                BeatScoreEvent {
                    start_beat: 0.0,
                    duration_beats: 1.0,
                    kind: ScoreEventKind::SungNote {
                        frequency_hz: 220.0,
                        velocity: 0.7,
                    },
                },
                BeatScoreEvent {
                    start_beat: 1.5,
                    duration_beats: 1.0,
                    kind: ScoreEventKind::Rest,
                },
            ],
        };
        assert!(score.validate().unwrap_err().contains("implicit gap"));

        score.events[1].start_beat = 1.0;
        score.tempo_map = vec![TempoChange {
            start_beat: 1.0,
            quarter_bpm: 120.0,
        }];
        assert!(score.validate().unwrap_err().contains("begin at beat zero"));

        score.tempo_map = vec![
            TempoChange {
                start_beat: 0.0,
                quarter_bpm: 120.0,
            },
            TempoChange {
                start_beat: 0.0,
                quarter_bpm: 90.0,
            },
        ];
        assert!(
            score
                .validate()
                .unwrap_err()
                .contains("strictly increasing")
        );
    }

    #[test]
    fn beat_domain_score_rejects_time_overflow_at_renderer_precision() {
        let score = BeatDomainScore {
            schema_version: 1,
            tempo_map: vec![TempoChange {
                start_beat: 0.0,
                quarter_bpm: 120.0,
            }],
            events: vec![BeatScoreEvent {
                start_beat: 0.0,
                duration_beats: 1.0e60,
                kind: ScoreEventKind::Rest,
            }],
        };
        assert!(
            score
                .materialize_seconds()
                .unwrap_err()
                .contains("finite positive f32 timeline")
        );
    }
}
