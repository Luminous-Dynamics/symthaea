// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Closed-set transcript hypothesis selection.
//!
//! This module defines a safety boundary for systems that select among
//! immutable ASR candidates. A selector may return a candidate identifier or
//! abstain; it cannot return replacement transcript text.
//!
//! This is an integrity primitive, not an ASR model, MICL implementation,
//! confidence estimator, or proof that the selected candidate is correct.
//! Any accepted transcript is copied from the exact candidate token sequence.
//!
//! See issue #11120 and research contract #11119.

use std::collections::HashSet;
use std::fmt;

/// Stable identifier for one immutable transcript candidate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CandidateId(String);

impl CandidateId {
    /// Construct a candidate identifier.
    ///
    /// Identifiers must be non-empty after trimming. The original supplied
    /// value is retained exactly; trimming is validation only.
    pub fn new(value: impl Into<String>) -> Result<Self, CandidateSelectionError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(CandidateSelectionError::EmptyCandidateId);
        }
        Ok(Self(value))
    }

    /// Return the exact identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Provenance for the system that produced a candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateProvenance {
    producer_id: String,
    model_revision: String,
    decoder_config_sha256: String,
}

impl CandidateProvenance {
    /// Construct provenance for an acoustic candidate producer.
    pub fn new(
        producer_id: impl Into<String>,
        model_revision: impl Into<String>,
        decoder_config_sha256: impl Into<String>,
    ) -> Result<Self, CandidateSelectionError> {
        let producer_id = producer_id.into();
        let model_revision = model_revision.into();
        let decoder_config_sha256 = decoder_config_sha256.into();

        if producer_id.trim().is_empty() {
            return Err(CandidateSelectionError::EmptyProducerId);
        }
        if model_revision.trim().is_empty() {
            return Err(CandidateSelectionError::EmptyModelRevision);
        }
        if !is_sha256(&decoder_config_sha256) {
            return Err(CandidateSelectionError::InvalidDecoderConfigDigest);
        }

        Ok(Self {
            producer_id,
            model_revision,
            decoder_config_sha256,
        })
    }

    /// Producer name or provider identifier.
    pub fn producer_id(&self) -> &str {
        &self.producer_id
    }

    /// Immutable model/provider revision.
    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    /// SHA-256 of the decoding configuration.
    pub fn decoder_config_sha256(&self) -> &str {
        &self.decoder_config_sha256
    }
}

/// Immutable transcript candidate tied to one exact source-audio interval.
///
/// Fields are private so a validated candidate cannot be modified between
/// validation and materialization. Tokenization and normalization are
/// intentionally upstream responsibilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateHypothesis {
    id: CandidateId,
    tokens: Vec<String>,
    source_audio_sha256: String,
    source_start_frame: u64,
    source_end_frame: u64,
    provenance: CandidateProvenance,
}

impl CandidateHypothesis {
    /// Construct an immutable candidate for a source-audio interval.
    pub fn new(
        id: CandidateId,
        tokens: Vec<String>,
        source_audio_sha256: impl Into<String>,
        source_start_frame: u64,
        source_end_frame: u64,
        provenance: CandidateProvenance,
    ) -> Result<Self, CandidateSelectionError> {
        let source_audio_sha256 = source_audio_sha256.into();

        if !is_sha256(&source_audio_sha256) {
            return Err(CandidateSelectionError::InvalidSourceAudioDigest);
        }
        if source_start_frame >= source_end_frame {
            return Err(CandidateSelectionError::InvalidSourceFrameRange {
                start_frame: source_start_frame,
                end_frame: source_end_frame,
            });
        }

        Ok(Self {
            id,
            tokens,
            source_audio_sha256,
            source_start_frame,
            source_end_frame,
            provenance,
        })
    }

    /// Stable candidate identifier.
    pub fn id(&self) -> &CandidateId {
        &self.id
    }

    /// Exact original token sequence. No normalization or rewriting occurs.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Source-audio SHA-256 digest.
    pub fn source_audio_sha256(&self) -> &str {
        &self.source_audio_sha256
    }

    /// Inclusive source start frame.
    pub fn source_start_frame(&self) -> u64 {
        self.source_start_frame
    }

    /// Exclusive source end frame.
    pub fn source_end_frame(&self) -> u64 {
        self.source_end_frame
    }

    /// Producer and decoder provenance.
    pub fn provenance(&self) -> &CandidateProvenance {
        &self.provenance
    }
}

/// Why a selector declined to choose one candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstentionReason {
    /// None of the candidates has adequate acoustic support.
    NoCandidateAcousticallySupported,
    /// Candidates conflict and available evidence does not resolve them.
    ConflictingCandidates,
    /// The language/profile is outside the selector's qualified scope.
    OutOfScopeLanguage,
    /// Evidence is insufficient for a reliable choice.
    InsufficientEvidence,
    /// The recognition/selection resource budget was exhausted.
    PolicyBudgetExhausted,
}

/// A selector response.
///
/// There is deliberately no free-form transcript field. External selector
/// output must be parsed into this closed type before transcript materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionDecision {
    /// Select one existing immutable candidate.
    Select {
        /// Identifier of an item in the frozen candidate set.
        candidate_id: CandidateId,
    },
    /// Preserve uncertainty instead of inventing a transcript.
    Abstain {
        /// Typed reason for abstention.
        reason: AbstentionReason,
    },
}

/// Final validated selection state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedSelection {
    /// Exact selected candidate, or None when abstaining.
    selected_candidate: Option<CandidateHypothesis>,
    /// Number of candidates in the frozen set.
    candidate_count: usize,
    /// Selection/abstention outcome.
    status: SelectionStatus,
}

impl ValidatedSelection {
    /// Exact selected candidate, if a candidate was selected.
    pub fn selected_candidate(&self) -> Option<&CandidateHypothesis> {
        self.selected_candidate.as_ref()
    }

    /// Exact tokens to materialize. Returns None on abstention.
    pub fn selected_tokens(&self) -> Option<&[String]> {
        self.selected_candidate
            .as_ref()
            .map(CandidateHypothesis::tokens)
    }

    /// Number of candidates presented to the selector.
    pub fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    /// Whether the decision selected or abstained.
    pub fn status(&self) -> &SelectionStatus {
        &self.status
    }
}

/// Validated selection status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStatus {
    /// One candidate from the frozen set was selected.
    Selected,
    /// No transcript was materialized.
    Abstained(AbstentionReason),
}

/// Invalid candidate set or selector decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateSelectionError {
    /// No candidates were provided.
    EmptyCandidateSet,
    /// Candidate ID was empty.
    EmptyCandidateId,
    /// Candidate ID occurs more than once.
    DuplicateCandidateId(CandidateId),
    /// Source-audio SHA-256 is malformed.
    InvalidSourceAudioDigest,
    /// Decoder configuration SHA-256 is malformed.
    InvalidDecoderConfigDigest,
    /// Source frame interval is empty or reversed.
    InvalidSourceFrameRange {
        /// Inclusive start frame.
        start_frame: u64,
        /// Exclusive end frame.
        end_frame: u64,
    },
    /// Candidate pool contains different source-audio digests.
    MixedSourceAudio,
    /// Candidate pool contains different source intervals.
    MixedSourceIntervals,
    /// Producer identifier is empty.
    EmptyProducerId,
    /// Model revision is empty.
    EmptyModelRevision,
    /// Selector chose an ID not in the frozen candidate set.
    UnknownCandidateId(CandidateId),
}

impl fmt::Display for CandidateSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet => write!(f, "candidate set is empty"),
            Self::EmptyCandidateId => write!(f, "candidate ID must be non-empty"),
            Self::DuplicateCandidateId(id) => {
                write!(f, "duplicate candidate ID: {}", id.as_str())
            }
            Self::InvalidSourceAudioDigest => {
                write!(f, "source audio digest must be a 64-character SHA-256 hex string")
            }
            Self::InvalidDecoderConfigDigest => {
                write!(f, "decoder config digest must be a 64-character SHA-256 hex string")
            }
            Self::InvalidSourceFrameRange {
                start_frame,
                end_frame,
            } => write!(
                f,
                "source frame interval must be non-empty and ordered: {start_frame}..{end_frame}"
            ),
            Self::MixedSourceAudio => {
                write!(f, "all candidates must reference the same source audio digest")
            }
            Self::MixedSourceIntervals => {
                write!(f, "all candidates must reference the same source frame interval")
            }
            Self::EmptyProducerId => write!(f, "candidate producer ID must be non-empty"),
            Self::EmptyModelRevision => write!(f, "candidate model revision must be non-empty"),
            Self::UnknownCandidateId(id) => {
                write!(f, "selector returned unknown candidate ID: {}", id.as_str())
            }
        }
    }
}

impl std::error::Error for CandidateSelectionError {}

/// Validate a closed-set selector decision and copy the selected candidate.
///
/// The function verifies:
/// - a non-empty set of unique candidate IDs;
/// - identical source audio and source interval across candidates;
/// - a selected ID exists in the set exactly once.
///
/// The returned transcript, when selected, is an exact clone of the chosen
/// candidate's token sequence. A caller cannot supply replacement text through
/// the typed selector decision.
pub fn validate_closed_set_decision(
    candidates: &[CandidateHypothesis],
    decision: SelectionDecision,
) -> Result<ValidatedSelection, CandidateSelectionError> {
    let Some(first) = candidates.first() else {
        return Err(CandidateSelectionError::EmptyCandidateSet);
    };

    let mut ids = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.clone()) {
            return Err(CandidateSelectionError::DuplicateCandidateId(
                candidate.id.clone(),
            ));
        }
        if candidate.source_audio_sha256 != first.source_audio_sha256 {
            return Err(CandidateSelectionError::MixedSourceAudio);
        }
        if candidate.source_start_frame != first.source_start_frame
            || candidate.source_end_frame != first.source_end_frame
        {
            return Err(CandidateSelectionError::MixedSourceIntervals);
        }
    }

    match decision {
        SelectionDecision::Select { candidate_id } => {
            let selected = candidates
                .iter()
                .find(|candidate| candidate.id == candidate_id)
                .cloned()
                .ok_or_else(|| {
                    CandidateSelectionError::UnknownCandidateId(candidate_id.clone())
                })?;

            Ok(ValidatedSelection {
                selected_candidate: Some(selected),
                candidate_count: candidates.len(),
                status: SelectionStatus::Selected,
            })
        }
        SelectionDecision::Abstain { reason } => Ok(ValidatedSelection {
            selected_candidate: None,
            candidate_count: candidates.len(),
            status: SelectionStatus::Abstained(reason),
        }),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::{
        AbstentionReason, CandidateHypothesis, CandidateId, CandidateProvenance,
        CandidateSelectionError, SelectionDecision, SelectionStatus,
        validate_closed_set_decision,
    };

    const AUDIO_SHA: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DECODER_SHA: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const OTHER_AUDIO_SHA: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn candidate(
        id: &str,
        tokens: &[&str],
        audio_sha: &str,
        start: u64,
        end: u64,
        producer: &str,
    ) -> CandidateHypothesis {
        let id = CandidateId::new(id).expect("valid candidate ID");
        let provenance = CandidateProvenance::new(
            producer,
            "model-revision-1",
            DECODER_SHA,
        )
        .expect("valid provenance");
        CandidateHypothesis::new(
            id,
            tokens.iter().map(|token| (*token).to_string()).collect(),
            audio_sha,
            start,
            end,
            provenance,
        )
        .expect("valid candidate")
    }

    fn pool() -> Vec<CandidateHypothesis> {
        vec![
            candidate(
                "acoustic-a",
                &["ngiyabona", "hello"],
                AUDIO_SHA,
                0,
                16_000,
                "recognizer-a",
            ),
            candidate(
                "acoustic-b",
                &["ngiyabona", "hullo"],
                AUDIO_SHA,
                0,
                16_000,
                "recognizer-b",
            ),
        ]
    }

    #[test]
    fn selected_tokens_are_an_exact_copy_of_the_candidate() {
        let candidates = pool();
        let selected = validate_closed_set_decision(
            &candidates,
            SelectionDecision::Select {
                candidate_id: CandidateId::new("acoustic-b").unwrap(),
            },
        )
        .expect("candidate exists");

        assert_eq!(
            selected.selected_tokens(),
            Some(&["ngiyabona".to_string(), "hullo".to_string()][..])
        );
        assert_eq!(selected.candidate_count(), 2);
        assert_eq!(selected.status(), &SelectionStatus::Selected);
    }

    #[test]
    fn unknown_candidate_id_is_rejected() {
        let error = validate_closed_set_decision(
            &pool(),
            SelectionDecision::Select {
                candidate_id: CandidateId::new("invented-transcript").unwrap(),
            },
        );
        assert_eq!(
            error,
            Err(CandidateSelectionError::UnknownCandidateId(
                CandidateId::new("invented-transcript").unwrap()
            ))
        );
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let first = candidate("same", &["one"], AUDIO_SHA, 0, 100, "a");
        let second = candidate("same", &["two"], AUDIO_SHA, 0, 100, "b");
        let error = validate_closed_set_decision(
            &[first, second],
            SelectionDecision::Abstain {
                reason: AbstentionReason::ConflictingCandidates,
            },
        );
        assert_eq!(
            error,
            Err(CandidateSelectionError::DuplicateCandidateId(
                CandidateId::new("same").unwrap()
            ))
        );
    }

    #[test]
    fn empty_candidate_set_is_rejected_even_for_abstention() {
        let error = validate_closed_set_decision(
            &[],
            SelectionDecision::Abstain {
                reason: AbstentionReason::InsufficientEvidence,
            },
        );
        assert_eq!(error, Err(CandidateSelectionError::EmptyCandidateSet));
    }

    #[test]
    fn mixed_source_audio_is_rejected() {
        let candidates = vec![
            candidate("a", &["one"], AUDIO_SHA, 0, 100, "a"),
            candidate("b", &["two"], OTHER_AUDIO_SHA, 0, 100, "b"),
        ];
        let error = validate_closed_set_decision(
            &candidates,
            SelectionDecision::Select {
                candidate_id: CandidateId::new("a").unwrap(),
            },
        );
        assert_eq!(error, Err(CandidateSelectionError::MixedSourceAudio));
    }

    #[test]
    fn mixed_source_intervals_are_rejected() {
        let candidates = vec![
            candidate("a", &["one"], AUDIO_SHA, 0, 100, "a"),
            candidate("b", &["two"], AUDIO_SHA, 1, 100, "b"),
        ];
        let error = validate_closed_set_decision(
            &candidates,
            SelectionDecision::Select {
                candidate_id: CandidateId::new("a").unwrap(),
            },
        );
        assert_eq!(error, Err(CandidateSelectionError::MixedSourceIntervals));
    }

    #[test]
    fn invalid_source_frame_range_is_rejected() {
        let id = CandidateId::new("a").unwrap();
        let provenance =
            CandidateProvenance::new("a", "rev-1", DECODER_SHA).unwrap();
        let error = CandidateHypothesis::new(
            id,
            vec!["token".to_string()],
            AUDIO_SHA,
            200,
            100,
            provenance,
        );
        assert!(matches!(
            error,
            Err(CandidateSelectionError::InvalidSourceFrameRange { .. })
        ));
    }

    #[test]
    fn invalid_digest_is_rejected() {
        let id = CandidateId::new("a").unwrap();
        let provenance =
            CandidateProvenance::new("a", "rev-1", DECODER_SHA).unwrap();
        let error = CandidateHypothesis::new(
            id,
            vec!["token".to_string()],
            "not-a-digest",
            0,
            100,
            provenance,
        );
        assert_eq!(
            error,
            Err(CandidateSelectionError::InvalidSourceAudioDigest)
        );
    }

    #[test]
    fn empty_producer_or_revision_is_rejected() {
        assert_eq!(
            CandidateProvenance::new("", "rev-1", DECODER_SHA),
            Err(CandidateSelectionError::EmptyProducerId)
        );
        assert_eq!(
            CandidateProvenance::new("provider", "", DECODER_SHA),
            Err(CandidateSelectionError::EmptyModelRevision)
        );
    }

    #[test]
    fn abstention_is_preserved_without_materializing_a_transcript() {
        let selected = validate_closed_set_decision(
            &pool(),
            SelectionDecision::Abstain {
                reason: AbstentionReason::NoCandidateAcousticallySupported,
            },
        )
        .expect("valid abstention");

        assert_eq!(selected.selected_candidate(), None);
        assert_eq!(selected.selected_tokens(), None);
        assert_eq!(
            selected.status(),
            &SelectionStatus::Abstained(
                AbstentionReason::NoCandidateAcousticallySupported
            )
        );
    }

    #[test]
    fn candidate_ids_cannot_be_empty_or_whitespace() {
        assert_eq!(
            CandidateId::new(""),
            Err(CandidateSelectionError::EmptyCandidateId)
        );
        assert_eq!(
            CandidateId::new("   "),
            Err(CandidateSelectionError::EmptyCandidateId)
        );
    }
}
