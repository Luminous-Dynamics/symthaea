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
    /// BLAKE3 digest of the exact serialized pronunciation candidate that was
    /// selected. This detects later edits to phones, syllables, scores, or
    /// provenance while retaining an otherwise unchanged receipt.
    pub candidate_digest: String,
    /// BLAKE3 digest binding the selected candidate to the exact source text,
    /// language, lexical byte span, and schema version.
    pub selection_digest: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionAuthority {
    HumanReviewer,
    User,
    CommunityReview,
    ApprovedPolicy,
}

#[derive(Serialize)]
struct CandidateDigestPayload<'a> {
    domain: &'static str,
    candidate: &'a PronunciationCandidate,
}

#[derive(Serialize)]
struct SelectionDigestPayload<'a> {
    domain: &'static str,
    schema_version: u32,
    language: &'a str,
    source_text: &'a str,
    start_byte: usize,
    end_byte: usize,
    surface: &'a str,
    selected_candidate_id: &'a str,
    candidate_digest: &'a str,
}

fn candidate_digest(candidate: &PronunciationCandidate) -> Result<String, String> {
    let payload = CandidateDigestPayload {
        domain: "luminous-dynamics/pronunciation-candidate/v1",
        candidate,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize pronunciation candidate: {error}"))?;
    Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
}

fn selection_digest(
    schema_version: u32,
    language: &str,
    source_text: &str,
    word: &WordPronunciation,
    selected_candidate_id: &str,
    candidate_digest: &str,
) -> Result<String, String> {
    let payload = SelectionDigestPayload {
        domain: "luminous-dynamics/pronunciation-selection/v1",
        schema_version,
        language,
        source_text,
        start_byte: word.start_byte,
        end_byte: word.end_byte,
        surface: &word.surface,
        selected_candidate_id,
        candidate_digest,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize pronunciation selection binding: {error}"))?;
    Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
}

impl SelectionReceipt {
    /// Create an integrity-bound receipt after choosing a candidate. The
    /// authority and evidence reference still require an independently
    /// verifiable review/override record; these digests bind that record to
    /// the exact candidate and lexical context in this plan.
    pub fn bind(
        authority: SelectionAuthority,
        evidence_reference: impl Into<String>,
        schema_version: u32,
        language: &str,
        source_text: &str,
        word: &WordPronunciation,
        candidate: &PronunciationCandidate,
    ) -> Result<Self, String> {
        let candidate_digest = candidate_digest(candidate)?;
        let selection_digest = selection_digest(
            schema_version,
            language,
            source_text,
            word,
            &candidate.candidate_id,
            &candidate_digest,
        )?;
        Ok(Self {
            authority,
            evidence_reference: evidence_reference.into(),
            candidate_digest,
            selection_digest,
        })
    }
}

impl PronunciationPlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 2 {
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
            validate_word_state(
                word_index,
                word,
                self.schema_version,
                &self.language,
                &self.source_text,
            )?;

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

fn validate_word_state(
    index: usize,
    word: &WordPronunciation,
    schema_version: u32,
    language: &str,
    source_text: &str,
) -> Result<(), String> {
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
            let selected_candidate = word
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == selected)
                .expect("selected candidate ID was checked above");
            let expected_candidate_digest = candidate_digest(selected_candidate)?;
            if receipt.candidate_digest != expected_candidate_digest {
                return Err(format!(
                    "word {index} selection receipt does not bind the exact selected pronunciation candidate"
                ));
            }
            let expected_selection_digest = selection_digest(
                schema_version,
                language,
                source_text,
                word,
                selected,
                &expected_candidate_digest,
            )?;
            if receipt.selection_digest != expected_selection_digest {
                return Err(format!(
                    "word {index} selection receipt does not bind the exact language/text/span context"
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

    fn selected_receipt(
        language: &str,
        source_text: &str,
        word: &WordPronunciation,
        candidate_index: usize,
    ) -> SelectionReceipt {
        SelectionReceipt::bind(
            SelectionAuthority::HumanReviewer,
            "review:fixture",
            1,
            language,
            source_text,
            word,
            &word.candidates[candidate_index],
        )
        .unwrap()
    }

    #[test]
    fn contextual_heteronym_retains_multiple_candidates_until_review() {
        let plan = PronunciationPlan {
            schema_version: 2,
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
        selected.selection_receipt = Some(selected_receipt("en-US", "read", &selected, 0));
        let plan = PronunciationPlan {
            schema_version: 2,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![selected],
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn changing_selected_phones_invalidates_the_old_receipt() {
        let mut selected = word(
            "read",
            0,
            ResolutionState::Selected,
            vec![candidate("past", "ɛ", CandidateSource::Dictionary)],
        );
        selected.selected_candidate_id = Some("past".into());
        selected.selection_receipt = Some(selected_receipt("en-US", "read", &selected, 0));
        let mut plan = PronunciationPlan {
            schema_version: 2,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![selected],
        };
        plan.words[0].candidates[0].syllables[0].phones[1].ipa = "æ".into();
        let error = plan.validate().unwrap_err();
        assert!(error.contains("exact selected pronunciation candidate"), "{error}");
    }

    #[test]
    fn changing_language_or_source_context_invalidates_the_old_receipt() {
        let mut selected = word(
            "read",
            0,
            ResolutionState::Selected,
            vec![candidate("past", "ɛ", CandidateSource::Dictionary)],
        );
        selected.selected_candidate_id = Some("past".into());
        selected.selection_receipt = Some(selected_receipt("en-US", "read", &selected, 0));
        let mut plan = PronunciationPlan {
            schema_version: 2,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![selected],
        };
        plan.language = "en-GB".into();
        let error = plan.validate().unwrap_err();
        assert!(error.contains("exact language/text/span context"), "{error}");
    }

    #[test]
    fn out_of_vocabulary_is_explicit_and_contains_no_guessed_phone() {
        let plan = PronunciationPlan {
            schema_version: 2,
            language: "en-US".into(),
            source_text: "xqzpl".into(),
            words: vec![word("xqzpl", 0, ResolutionState::OutOfVocabulary, vec![])],
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn unrepresented_source_text_cannot_be_silently_dropped() {
        let plan = PronunciationPlan {
            schema_version: 2,
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
            schema_version: 2,
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
            schema_version: 2,
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
            schema_version: 2,
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
        unknown.selection_receipt = Some(selected_receipt("en", "read", &unknown, 0));
        let plan = PronunciationPlan {
            schema_version: 2,
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
            schema_version: 2,
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
            schema_version: 2,
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
            schema_version: 2,
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
            schema_version: 2,
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
            schema_version: 2,
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
        selected.selection_receipt = Some(selected_receipt("en-US", "read", &selected, 0));
        let plan = PronunciationPlan {
            schema_version: 2,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![selected],
        };
        assert!(plan.validate_renderable().is_ok());
    }

    fn selected_word(
        surface: &str,
        start_byte: usize,
        vowel: &str,
        source_text: &str,
    ) -> WordPronunciation {
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
        let mut word = WordPronunciation {
            start_byte,
            end_byte: start_byte + surface.len(),
            surface: surface.into(),
            state: ResolutionState::Selected,
            selected_candidate_id: Some(candidate.candidate_id.clone()),
            candidates: vec![candidate],
            selection_receipt: None,
        };
        word.selection_receipt = Some(selected_receipt("en-US", source_text, &word, 0));
        word
    }

    fn alignment_fixture() -> (PronunciationPlan, VocalScoreAlignment) {
        let pronunciation = PronunciationPlan {
            schema_version: 2,
            language: "en-US".into(),
            source_text: "read now".into(),
            words: vec![
                selected_word("read", 0, "ɛ", "read now"),
                selected_word("now", 5, "aʊ", "read now"),
            ],
        };
        let alignment = VocalScoreAlignment {
            schema_version: 2,
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
            schema_version: 2,
            language: "en-US".into(),
            source_text: "read now".into(),
            words: vec![selected_word("read", 0, "ɛ"), selected_word("now", 5, "aʊ")],
        };
        let alignment = VocalScoreAlignment {
            schema_version: 2,
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
            schema_version: 2,
            language: "en-US".into(),
            source_text: "read now".into(),
            words: vec![selected_word("read", 0, "ɛ"), selected_word("now", 5, "aʊ")],
        };
        let mut alignment = VocalScoreAlignment {
            schema_version: 2,
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
            schema_version: 2,
            language: "en-US".into(),
            source_text: "read now".into(),
            words: vec![selected_word("read", 0, "ɛ"), selected_word("now", 5, "aʊ")],
        };
        let alignment = VocalScoreAlignment {
            schema_version: 2,
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
}
