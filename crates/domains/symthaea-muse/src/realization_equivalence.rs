// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Intent-level symbolic realization qualification.
//!
//! A paired target is one witness, not the definition of correctness. This
//! module supports a conservative first step: a source-bound, event-scoped
//! pitch-choice contract whose accepted set contains multiple MIDI pitches.
//! A candidate passes when it selects any allowed pitch and preserves every
//! other score/event property exactly. The optional benchmark reference is
//! diagnostic only; it does not decide acceptance.
//!
//! The current Score schema has no persistent note UUID. Event IDs therefore
//! refer to the onset-ordered Score::events() stream. Events are joined by a
//! pitch-independent identity key. Duplicate keys are ambiguous and fail
//! closed rather than being guessed by nearest onset.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use symthaea_music_theory::{Score, ScoreNote};

pub const SYMBOLIC_REALIZATION_CONTRACT_VERSION: &str =
    "muse-symbolic-realization-contract-v1";
pub const SYMBOLIC_REALIZATION_REPORT_VERSION: &str =
    "muse-symbolic-realization-report-v1";
pub const MAX_REALIZATION_EVENTS: usize = 4096;
pub const MAX_ALLOWED_PITCHES: usize = 128;

const SCORE_DOMAIN: &[u8] = b"luminous-dynamics-muse-score-content-v1\0";
const CONTRACT_DOMAIN: &[u8] = b"luminous-dynamics-muse-symbolic-realization-contract-v1\0";
const REPORT_DOMAIN: &[u8] = b"luminous-dynamics-muse-symbolic-realization-report-v1\0";

/// Stable source-score event identity. Index is into Score::events().
pub fn score_event_id(index: usize) -> String {
    format!("score_event_index:{index}")
}

fn parse_score_event_id(value: &str) -> Option<usize> {
    let suffix = value.strip_prefix("score_event_index:")?;
    if suffix.is_empty() || (suffix.len() > 1 && suffix.starts_with('0')) {
        return None;
    }
    if !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    suffix.parse().ok()
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_component(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Hash the complete serialized Score using the same domain as Muse's
/// full-score render binding. This is exact-content identity, not equivalence.
pub fn score_content_sha256(score: &Score) -> Result<String, SymbolicRealizationError> {
    let bytes =
        serde_json::to_vec(score).map_err(|_| SymbolicRealizationError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(SCORE_DOMAIN);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// One explicit edit intent with a set of acceptable event-level realizations.
///
/// The accepted pitch list must contain at least two distinct MIDI notes and
/// must exclude the source pitch, so returning the source cannot pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicRealizationContract {
    pub contract_version: String,
    pub contract_id: String,
    pub source_score_sha256: String,
    pub target_event_id: String,
    pub allowed_midi_notes: Vec<u8>,
    /// Optional paired target used for diagnostics. It never defines acceptance.
    pub reference_score_sha256: Option<String>,
    /// SHA-256 over the canonical contract with this field blank.
    pub contract_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicRealizationError {
    InvalidContract(&'static str),
    InvalidScore(&'static str),
    SourceBindingMismatch,
    ReferenceBindingMismatch,
    SerializationFailure,
}

impl std::fmt::Display for SymbolicRealizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "symbolic realization qualification failed: {self:?}")
    }
}

impl std::error::Error for SymbolicRealizationError {}

impl SymbolicRealizationContract {
    /// Create a contract bound to a source Score and optional paired reference.
    pub fn new(
        contract_id: impl Into<String>,
        source_score: &Score,
        target_event_index: usize,
        allowed_midi_notes: Vec<u8>,
        reference_score: Option<&Score>,
    ) -> Result<Self, SymbolicRealizationError> {
        validate_score(source_score)?;
        let source_events = source_score.events();
        let source_note = source_events
            .get(target_event_index)
            .ok_or(SymbolicRealizationError::InvalidContract(
                "target event index is outside the source Score",
            ))?;
        ensure_unambiguous_source_events(&source_events)?;

        let mut contract = Self {
            contract_version: SYMBOLIC_REALIZATION_CONTRACT_VERSION.to_owned(),
            contract_id: contract_id.into(),
            source_score_sha256: score_content_sha256(source_score)?,
            target_event_id: score_event_id(target_event_index),
            allowed_midi_notes,
            reference_score_sha256: reference_score
                .map(score_content_sha256)
                .transpose()?,
            contract_sha256: String::new(),
        };
        contract.seal()?;
        if contract
            .allowed_midi_notes
            .contains(&source_note.pitch.midi())
        {
            return Err(SymbolicRealizationError::InvalidContract(
                "allowed pitch set includes the source pitch and therefore permits a no-op",
            ));
        }
        if let Some(reference) = reference_score {
            validate_score(reference)?;
        }
        Ok(contract)
    }

    /// Canonicalize the accepted pitch set and seal the contract.
    pub fn seal(&mut self) -> Result<(), SymbolicRealizationError> {
        self.allowed_midi_notes.sort_unstable();
        self.contract_sha256.clear();
        self.validate_fields()?;
        self.contract_sha256 = self.compute_sha256()?;
        Ok(())
    }

    /// Check intrinsic contract structure and its content receipt.
    pub fn verify(&self) -> Result<bool, SymbolicRealizationError> {
        self.validate_fields()?;
        Ok(self.compute_sha256()? == self.contract_sha256)
    }

    fn validate_fields(&self) -> Result<(), SymbolicRealizationError> {
        if self.contract_version != SYMBOLIC_REALIZATION_CONTRACT_VERSION {
            return Err(SymbolicRealizationError::InvalidContract(
                "unsupported contract version",
            ));
        }
        if self.contract_id.trim().is_empty() || self.contract_id.len() > 256 {
            return Err(SymbolicRealizationError::InvalidContract(
                "contract ID must be non-empty and bounded",
            ));
        }
        if !is_lower_hex_sha256(&self.source_score_sha256) {
            return Err(SymbolicRealizationError::InvalidContract(
                "source score digest must be lowercase SHA-256",
            ));
        }
        if !self.contract_sha256.is_empty() && !is_lower_hex_sha256(&self.contract_sha256) {
            return Err(SymbolicRealizationError::InvalidContract(
                "contract receipt must be lowercase SHA-256",
            ));
        }
        if parse_score_event_id(&self.target_event_id).is_none() {
            return Err(SymbolicRealizationError::InvalidContract(
                "target event ID must use canonical score_event_index:N form",
            ));
        }
        if self.allowed_midi_notes.len() < 2
            || self.allowed_midi_notes.len() > MAX_ALLOWED_PITCHES
        {
            return Err(SymbolicRealizationError::InvalidContract(
                "allowed pitch set must contain 2..=128 alternatives",
            ));
        }
        if self
            .allowed_midi_notes
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(SymbolicRealizationError::InvalidContract(
                "allowed pitch set must be sorted and unique",
            ));
        }
        if self
            .reference_score_sha256
            .as_ref()
            .is_some_and(|digest| !is_lower_hex_sha256(digest))
        {
            return Err(SymbolicRealizationError::InvalidContract(
                "optional reference digest must be lowercase SHA-256",
            ));
        }
        Ok(())
    }

    fn compute_sha256(&self) -> Result<String, SymbolicRealizationError> {
        let mut unsigned = self.clone();
        unsigned.contract_sha256.clear();
        let bytes = serde_json::to_vec(&unsigned)
            .map_err(|_| SymbolicRealizationError::SerializationFailure)?;
        let mut hasher = Sha256::new();
        hasher.update(CONTRACT_DOMAIN);
        hash_component(&mut hasher, &bytes);
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Check exact source binding and ensure the target event has unique identity.
    pub fn verify_for_source(&self, source_score: &Score) -> Result<(), SymbolicRealizationError> {
        if !self.verify()? {
            return Err(SymbolicRealizationError::InvalidContract(
                "contract receipt did not verify",
            ));
        }
        validate_score(source_score)?;
        if score_content_sha256(source_score)? != self.source_score_sha256 {
            return Err(SymbolicRealizationError::SourceBindingMismatch);
        }
        let source_events = source_score.events();
        ensure_unambiguous_source_events(&source_events)?;
        let index = parse_score_event_id(&self.target_event_id)
            .ok_or(SymbolicRealizationError::InvalidContract(
                "target event ID is malformed",
            ))?;
        let note = source_events
            .get(index)
            .ok_or(SymbolicRealizationError::InvalidContract(
                "target event ID is outside the bound source Score",
            ))?;
        if self.allowed_midi_notes.contains(&note.pitch.midi()) {
            return Err(SymbolicRealizationError::InvalidContract(
                "allowed pitch set includes the bound source pitch",
            ));
        }
        Ok(())
    }
}

/// Candidate failure categories. Reference-match status remains diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolicRealizationViolation {
    EventCountMismatch,
    AmbiguousEventIdentity,
    MissingOrModifiedSourceEvent,
    UnexpectedCandidateEvent,
    RequestedPitchNotAllowed,
    ProtectedPitchChanged,
    ScoreContextChanged,
}

/// Replayable candidate qualification report. Target similarity is diagnostic;
/// it never contributes to the accepted decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicRealizationReport {
    pub report_version: String,
    pub contract_sha256: String,
    pub source_score_sha256: String,
    pub candidate_score_sha256: String,
    pub reference_score_sha256: Option<String>,
    pub exact_reference_match: Option<bool>,
    pub semantic_reference_match: Option<bool>,
    pub reference_satisfies_contract: Option<bool>,
    pub requested_change_satisfied: bool,
    pub protected_content_preserved: bool,
    pub event_count_preserved: bool,
    pub score_context_preserved: bool,
    pub accepted: bool,
    /// True only when the candidate is valid and semantically differs from the reference.
    pub accepted_alternative: bool,
    pub violations: Vec<SymbolicRealizationViolation>,
    pub report_sha256: String,
}

impl SymbolicRealizationReport {
    /// Receipt integrity plus all independent acceptance predicates.
    pub fn is_accepted(&self) -> bool {
        self.accepted
            && self.requested_change_satisfied
            && self.protected_content_preserved
            && self.event_count_preserved
            && self.score_context_preserved
            && self.report_sha256 == report_sha256(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EventIdentityKey {
    part: u16,
    onset_bits: u64,
    duration_bits: u64,
    velocity_bits: u32,
    role_code: u8,
    emphasis_code: u8,
    section_intensity_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EventSemanticKey {
    identity: EventIdentityKey,
    pitch_midi: u8,
}

fn role_code(note: &ScoreNote) -> u8 {
    match note.role {
        symthaea_music_theory::VoiceRole::Melody => 0,
        symthaea_music_theory::VoiceRole::Harmony => 1,
        symthaea_music_theory::VoiceRole::Bass => 2,
        symthaea_music_theory::VoiceRole::CounterMelody => 3,
    }
}

fn emphasis_code(note: &ScoreNote) -> u8 {
    match note.emphasis {
        symthaea_music_theory::Emphasis::Normal => 0,
        symthaea_music_theory::Emphasis::Climax => 1,
        symthaea_music_theory::Emphasis::Cadential => 2,
        symthaea_music_theory::Emphasis::PhraseStart => 3,
    }
}

/// All symbolic note properties except pitch. This is the join key for the
/// pitch-choice operation; changing any protected field breaks the join.
fn event_identity(note: &ScoreNote) -> EventIdentityKey {
    EventIdentityKey {
        part: note.part.0,
        onset_bits: note.onset.beats().to_bits(),
        duration_bits: note.duration.beats().to_bits(),
        velocity_bits: note.velocity.to_bits(),
        role_code: role_code(note),
        emphasis_code: emphasis_code(note),
        section_intensity_bits: note.section_intensity.to_bits(),
    }
}

fn event_semantic_key(note: &ScoreNote) -> EventSemanticKey {
    EventSemanticKey {
        identity: event_identity(note),
        pitch_midi: note.pitch.midi(),
    }
}

fn ensure_unambiguous_source_events(
    events: &[ScoreNote],
) -> Result<(), SymbolicRealizationError> {
    let mut seen = BTreeSet::new();
    for event in events {
        if !seen.insert(event_identity(event)) {
            return Err(SymbolicRealizationError::InvalidContract(
                "source Score has ambiguous event identity; persistent event IDs are required",
            ));
        }
    }
    Ok(())
}

fn validate_score(score: &Score) -> Result<(), SymbolicRealizationError> {
    if score.notes.is_empty() || score.notes.len() > MAX_REALIZATION_EVENTS {
        return Err(SymbolicRealizationError::InvalidScore(
            "Score must contain 1..=4096 notes",
        ));
    }
    if !score.tempo_bpm.is_finite()
        || score.tempo_bpm <= 0.0
        || score.meter == 0
        || !score.total_beats.beats().is_finite()
        || score.total_beats.beats() < 0.0
    {
        return Err(SymbolicRealizationError::InvalidScore(
            "tempo, meter, or total-beat context is invalid",
        ));
    }
    for note in &score.notes {
        let onset = note.onset.beats();
        let duration = note.duration.beats();
        if !onset.is_finite()
            || onset < 0.0
            || !duration.is_finite()
            || duration <= 0.0
            || !note.velocity.is_finite()
            || !(0.0..=1.0).contains(&note.velocity)
            || !note.section_intensity.is_finite()
            || note.section_intensity < 0.0
        {
            return Err(SymbolicRealizationError::InvalidScore(
                "note timing and dynamics must be finite and within declared bounds",
            ));
        }
    }
    Ok(())
}

fn score_context_equal(source: &Score, candidate: &Score) -> bool {
    source.key == candidate.key
        && source.tempo_bpm.to_bits() == candidate.tempo_bpm.to_bits()
        && source.meter == candidate.meter
        && source.total_beats.beats().to_bits() == candidate.total_beats.beats().to_bits()
}

fn semantic_score_equal(left: &Score, right: &Score) -> bool {
    if !score_context_equal(left, right) || left.notes.len() != right.notes.len() {
        return false;
    }
    let mut left_events: Vec<EventSemanticKey> =
        left.events().iter().map(event_semantic_key).collect();
    let mut right_events: Vec<EventSemanticKey> =
        right.events().iter().map(event_semantic_key).collect();
    left_events.sort_unstable();
    right_events.sort_unstable();
    left_events == right_events
}

fn candidate_qualification(
    contract: &SymbolicRealizationContract,
    source: &Score,
    candidate: &Score,
) -> Result<
    (
        bool,
        bool,
        bool,
        bool,
        Vec<SymbolicRealizationViolation>,
    ),
    SymbolicRealizationError,
> {
    validate_score(source)?;
    validate_score(candidate)?;

    let source_events = source.events();
    let candidate_events = candidate.events();
    let event_count_preserved = source_events.len() == candidate_events.len();
    let score_context_preserved = score_context_equal(source, candidate);
    let mut protected_content_preserved = score_context_preserved;
    let mut requested_change_satisfied = false;
    let mut violations = Vec::new();

    if !event_count_preserved {
        violations.push(SymbolicRealizationViolation::EventCountMismatch);
        protected_content_preserved = false;
    }
    if !score_context_preserved {
        violations.push(SymbolicRealizationViolation::ScoreContextChanged);
    }

    let target_index = parse_score_event_id(&contract.target_event_id).ok_or(
        SymbolicRealizationError::InvalidContract("target event ID is malformed"),
    )?;
    let source_target = source_events.get(target_index).ok_or(
        SymbolicRealizationError::InvalidContract("target event ID is outside the source Score"),
    )?;
    if contract.allowed_midi_notes.contains(&source_target.pitch.midi()) {
        return Err(SymbolicRealizationError::InvalidContract(
            "allowed pitch set includes the source pitch",
        ));
    }

    // Join by pitch-independent event identity. Repeated signatures are
    // ambiguous in the current Score schema and cannot be safely guessed.
    let mut source_keys = BTreeSet::new();
    for event in &source_events {
        if !source_keys.insert(event_identity(event)) {
            violations.push(SymbolicRealizationViolation::AmbiguousEventIdentity);
            protected_content_preserved = false;
        }
    }
    let mut candidate_by_key = BTreeMap::<EventIdentityKey, Vec<usize>>::new();
    for (index, event) in candidate_events.iter().enumerate() {
        candidate_by_key
            .entry(event_identity(event))
            .or_default()
            .push(index);
    }
    if candidate_by_key.values().any(|indices| indices.len() > 1) {
        violations.push(SymbolicRealizationViolation::AmbiguousEventIdentity);
        protected_content_preserved = false;
    }

    let mut consumed_candidate_indices = BTreeSet::new();
    let mut matched_target = false;
    for (source_index, source_event) in source_events.iter().enumerate() {
        let key = event_identity(source_event);
        let Some(candidate_indices) = candidate_by_key.get(&key) else {
            violations.push(SymbolicRealizationViolation::MissingOrModifiedSourceEvent);
            protected_content_preserved = false;
            continue;
        };
        if candidate_indices.len() != 1 {
            continue;
        }
        let candidate_index = candidate_indices[0];
        if !consumed_candidate_indices.insert(candidate_index) {
            violations.push(SymbolicRealizationViolation::AmbiguousEventIdentity);
            protected_content_preserved = false;
            continue;
        }
        let candidate_event = &candidate_events[candidate_index];
        if source_index == target_index {
            matched_target = true;
            let target_pitch = candidate_event.pitch.midi();
            if !contract.allowed_midi_notes.contains(&target_pitch) {
                violations.push(SymbolicRealizationViolation::RequestedPitchNotAllowed);
            } else if target_pitch != source_event.pitch.midi() {
                requested_change_satisfied = true;
            }
        } else if candidate_event.pitch != source_event.pitch {
            violations.push(SymbolicRealizationViolation::ProtectedPitchChanged);
            protected_content_preserved = false;
        }
    }

    if consumed_candidate_indices.len() != candidate_events.len() {
        violations.push(SymbolicRealizationViolation::UnexpectedCandidateEvent);
        protected_content_preserved = false;
    }
    if !matched_target {
        requested_change_satisfied = false;
        violations.push(SymbolicRealizationViolation::MissingOrModifiedSourceEvent);
        protected_content_preserved = false;
    }
    violations.sort();
    violations.dedup();

    Ok((
        requested_change_satisfied,
        protected_content_preserved,
        event_count_preserved,
        score_context_preserved,
        violations,
    ))
}

fn report_sha256(report: &SymbolicRealizationReport) -> String {
    let mut unsigned = report.clone();
    unsigned.report_sha256.clear();
    let bytes = serde_json::to_vec(&unsigned).expect("serializing report should not fail");
    let mut hasher = Sha256::new();
    hasher.update(REPORT_DOMAIN);
    hash_component(&mut hasher, &bytes);
    format!("{:x}", hasher.finalize())
}

/// Evaluate a candidate against the source-bound intent contract.
///
/// A paired target is one controlled witness. A candidate satisfying the
/// operation and preservation predicates is accepted even when its exact
/// content hash and semantic score differ from that reference.
pub fn evaluate_symbolic_realization(
    contract: &SymbolicRealizationContract,
    source: &Score,
    candidate: &Score,
    reference: Option<&Score>,
) -> Result<SymbolicRealizationReport, SymbolicRealizationError> {
    contract.verify_for_source(source)?;
    match (&contract.reference_score_sha256, reference) {
        (Some(expected), Some(reference_score)) => {
            validate_score(reference_score)?;
            if score_content_sha256(reference_score)? != *expected {
                return Err(SymbolicRealizationError::ReferenceBindingMismatch);
            }
        }
        (None, None) => {}
        _ => return Err(SymbolicRealizationError::ReferenceBindingMismatch),
    }

    let (requested, protected, count, context, violations) =
        candidate_qualification(contract, source, candidate)?;
    let candidate_sha = score_content_sha256(candidate)?;
    let reference_sha = reference.map(score_content_sha256).transpose()?;
    let exact_reference_match = reference_sha
        .as_ref()
        .map(|digest| digest == &candidate_sha);
    let semantic_reference_match =
        reference.map(|reference_score| semantic_score_equal(candidate, reference_score));
    let reference_satisfies_contract = match reference {
        Some(reference_score) => {
            let (request_ok, protected_ok, count_ok, context_ok, reference_violations) =
                candidate_qualification(contract, source, reference_score)?;
            Some(
                request_ok
                    && protected_ok
                    && count_ok
                    && context_ok
                    && reference_violations.is_empty(),
            )
        }
        None => None,
    };
    let accepted = requested && protected && count && context && violations.is_empty();
    let mut report = SymbolicRealizationReport {
        report_version: SYMBOLIC_REALIZATION_REPORT_VERSION.to_owned(),
        contract_sha256: contract.contract_sha256.clone(),
        source_score_sha256: contract.source_score_sha256.clone(),
        candidate_score_sha256: candidate_sha,
        reference_score_sha256: reference_sha,
        exact_reference_match,
        semantic_reference_match,
        reference_satisfies_contract,
        requested_change_satisfied: requested,
        protected_content_preserved: protected,
        event_count_preserved: count,
        score_context_preserved: context,
        accepted,
        accepted_alternative: accepted && semantic_reference_match == Some(false),
        violations,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_sha256(&report);
    Ok(report)
}

/// Recompute the full evaluation and require exact report equality.
pub fn verify_symbolic_realization_report(
    contract: &SymbolicRealizationContract,
    source: &Score,
    candidate: &Score,
    reference: Option<&Score>,
    supplied: &SymbolicRealizationReport,
) -> Result<bool, SymbolicRealizationError> {
    if supplied.report_sha256.is_empty()
        || supplied.report_sha256 != report_sha256(supplied)
    {
        return Ok(false);
    }
    let expected = evaluate_symbolic_realization(contract, source, candidate, reference)?;
    Ok(expected == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::harmony::Key;
    use symthaea_music_theory::pitch::{Pitch, PitchClass};
    use symthaea_music_theory::rhythm::Duration;
    use symthaea_music_theory::{Emphasis, PartId, VoiceRole};

    fn note(pitch: u8, onset: f64, duration: f64) -> ScoreNote {
        ScoreNote {
            part: PartId(0),
            pitch: Pitch::from_midi(pitch),
            onset: Duration::new((onset * 1000.0).round() as u32, 1000),
            duration: Duration::new((duration * 1000.0).round() as u32, 1000),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn source_score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(60, 0.0, 1.0));
        score.push(note(64, 1.0, 1.0));
        score.push(note(67, 2.0, 1.0));
        score
    }

    fn candidate_with_pitch(pitch: u8) -> Score {
        let mut score = source_score();
        score.notes[1].pitch = Pitch::from_midi(pitch);
        score
    }

    fn fixture(reference_pitch: u8) -> (Score, Score, SymbolicRealizationContract) {
        let source = source_score();
        let reference = candidate_with_pitch(reference_pitch);
        let contract = SymbolicRealizationContract::new(
            "muse-pitch-choice-v1",
            &source,
            1,
            vec![64, 67, 71],
            Some(&reference),
        )
        .expect("contract seals");
        (source, reference, contract)
    }

    #[test]
    fn valid_alternative_passes_when_it_differs_from_paired_reference() {
        let (source, reference, contract) = fixture(64);
        let candidate = candidate_with_pitch(71);
        let report = evaluate_symbolic_realization(
            &contract,
            &source,
            &candidate,
            Some(&reference),
        )
        .expect("candidate evaluates");
        assert!(report.is_accepted(), "{:?}", report.violations);
        assert_eq!(report.exact_reference_match, Some(false));
        assert_eq!(report.semantic_reference_match, Some(false));
        assert_eq!(report.reference_satisfies_contract, Some(true));
        assert!(report.accepted_alternative);
        assert!(verify_symbolic_realization_report(
            &contract,
            &source,
            &candidate,
            Some(&reference),
            &report,
        )
        .expect("replay succeeds"));
    }

    #[test]
    fn paired_reference_itself_is_a_valid_realization() {
        let (source, reference, contract) = fixture(67);
        let report = evaluate_symbolic_realization(
            &contract,
            &source,
            &reference,
            Some(&reference),
        )
        .expect("candidate evaluates");
        assert!(report.is_accepted());
        assert_eq!(report.exact_reference_match, Some(true));
        assert_eq!(report.semantic_reference_match, Some(true));
        assert!(!report.accepted_alternative);
    }

    #[test]
    fn no_op_is_rejected_even_though_it_maximizes_source_similarity() {
        let (source, reference, contract) = fixture(64);
        let report = evaluate_symbolic_realization(
            &contract,
            &source,
            &source,
            Some(&reference),
        )
        .expect("candidate evaluates");
        assert!(!report.is_accepted());
        assert!(!report.requested_change_satisfied);
        assert!(report.violations.contains(
            &SymbolicRealizationViolation::RequestedPitchNotAllowed
        ));
    }

    #[test]
    fn wrong_event_edit_is_rejected() {
        let (source, reference, contract) = fixture(64);
        let mut candidate = source.clone();
        candidate.notes[0].pitch = Pitch::from_midi(71);
        let report = evaluate_symbolic_realization(
            &contract,
            &source,
            &candidate,
            Some(&reference),
        )
        .expect("candidate evaluates");
        assert!(!report.is_accepted());
        assert!(!report.requested_change_satisfied);
        assert!(!report.protected_content_preserved);
    }

    #[test]
    fn unrequested_pitch_change_is_rejected() {
        let (source, reference, contract) = fixture(64);
        let mut candidate = candidate_with_pitch(71);
        candidate.notes[0].pitch = Pitch::from_midi(62);
        let report = evaluate_symbolic_realization(
            &contract,
            &source,
            &candidate,
            Some(&reference),
        )
        .expect("candidate evaluates");
        assert!(!report.is_accepted());
        assert!(report.violations.contains(
            &SymbolicRealizationViolation::ProtectedPitchChanged
        ));
    }

    #[test]
    fn collateral_timing_or_dynamics_change_breaks_event_identity() {
        let (source, reference, contract) = fixture(64);
        let mut candidate = candidate_with_pitch(71);
        candidate.notes[2].velocity = 0.8;
        let report = evaluate_symbolic_realization(
            &contract,
            &source,
            &candidate,
            Some(&reference),
        )
        .expect("candidate evaluates");
        assert!(!report.is_accepted());
        assert!(!report.protected_content_preserved);
        assert!(report.violations.contains(
            &SymbolicRealizationViolation::MissingOrModifiedSourceEvent
        ));
    }

    #[test]
    fn event_insertion_and_deletion_fail_closed() {
        let (source, reference, contract) = fixture(64);
        let mut candidate = candidate_with_pitch(71);
        candidate.notes.pop();
        let report = evaluate_symbolic_realization(
            &contract,
            &source,
            &candidate,
            Some(&reference),
        )
        .expect("candidate evaluates");
        assert!(!report.is_accepted());
        assert!(!report.event_count_preserved);
        assert!(report.violations.contains(
            &SymbolicRealizationViolation::EventCountMismatch
        ));
    }

    #[test]
    fn note_storage_reordering_is_semantically_not_a_new_realization() {
        let (source, reference, contract) = fixture(64);
        let mut candidate = candidate_with_pitch(71);
        candidate.notes.reverse();
        let report = evaluate_symbolic_realization(
            &contract,
            &source,
            &candidate,
            Some(&reference),
        )
        .expect("candidate evaluates");
        assert!(report.is_accepted(), "{:?}", report.violations);
        assert_eq!(report.exact_reference_match, Some(false));
        assert_eq!(report.semantic_reference_match, Some(false));
        assert!(report.accepted_alternative);
    }

    #[test]
    fn ambiguous_identical_source_event_identity_fails_closed() {
        let mut source = source_score();
        source.notes[1].onset = Duration::zero();
        source.notes[1].duration = source.notes[0].duration;
        source.notes[1].velocity = source.notes[0].velocity;
        source.notes[1].section_intensity = source.notes[0].section_intensity;
        assert!(matches!(
            SymbolicRealizationContract::new(
                "ambiguous-fixture",
                &source,
                1,
                vec![64, 71],
                None,
            ),
            Err(SymbolicRealizationError::InvalidContract(_))
        ));
    }

    #[test]
    fn source_pitch_cannot_appear_in_allowed_set() {
        let source = source_score();
        assert!(matches!(
            SymbolicRealizationContract::new(
                "no-op-allowed",
                &source,
                1,
                vec![60, 64, 71],
                None,
            ),
            Err(SymbolicRealizationError::InvalidContract(_))
        ));
    }

    #[test]
    fn stale_source_binding_and_receipt_tampering_reject() {
        let (source, reference, contract) = fixture(64);
        let candidate = candidate_with_pitch(71);
        let mut stale_source = source.clone();
        stale_source.notes[0].pitch = Pitch::from_midi(61);
        assert_eq!(
            contract.verify_for_source(&stale_source),
            Err(SymbolicRealizationError::SourceBindingMismatch)
        );

        let mut report = evaluate_symbolic_realization(
            &contract,
            &source,
            &candidate,
            Some(&reference),
        )
        .expect("candidate evaluates");
        report.accepted = false;
        assert!(!report.is_accepted());
        assert!(!verify_symbolic_realization_report(
            &contract,
            &source,
            &candidate,
            Some(&reference),
            &report,
        )
        .expect("tamper check returns false"));
    }
}
