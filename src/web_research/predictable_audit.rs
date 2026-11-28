// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictable Bernoulli audit selection from custodian-supplied random draws.
//!
//! This module implements the per-event audit decision only. It does not
//! generate entropy, choose the q-policy, adjudicate labels, or expose the
//! random draw to the candidate process. It binds each q_t to a pre-label
//! history digest and to externally frozen design/random-stream manifests.
//!
//! Probabilities use a dyadic grid with denominator 2^64 and numerator
//! multiples of 2^11. This restriction makes every returned f64 q_t exact:
//! q_t = (numerator >> 11) / 2^53. A custodian-supplied u64 draw is included
//! when draw < numerator. The q_t values are therefore exactly representable
//! and the Bernoulli sampling law has no floating-point threshold ambiguity.
//!
//! This is a with-replacement-across-time Bernoulli design, not the fixed-quota
//! SRSWOR design in stratified_audit.rs. q_t may adapt to recorded pre-label
//! history, but the q-policy and its dependence on that history must be audited
//! externally. Selected-label nonresponse is not modelled.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const PREDICTABLE_AUDIT_VERSION: &str = "symthaea-predictable-bernoulli-audit-v1";
pub const DYADIC_DENOMINATOR: u128 = 1_u128 << 64;
pub const DYADIC_NUMERATOR_STEP: u128 = 1_u128 << 11;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DyadicInclusionProbability {
    /// Numerator over the exact denominator 2^64. Must be a multiple of 2^11.
    /// The value 2^64 represents certainty and cannot be stored in a u64.
    pub numerator: u128,
}

impl DyadicInclusionProbability {
    pub fn validate(self) -> Result<(), PredictableAuditError> {
        if self.numerator == 0
            || self.numerator > DYADIC_DENOMINATOR
            || self.numerator % DYADIC_NUMERATOR_STEP != 0
        {
            return Err(PredictableAuditError::InvalidProbabilityNumerator(
                self.numerator,
            ));
        }
        Ok(())
    }

    /// Exact binary64 representation because the numerator is constrained to
    /// a 53-bit dyadic significand.
    pub fn as_f64(self) -> f64 {
        (self.numerator >> 11) as f64 / (1_u64 << 53) as f64
    }

    pub fn selects(self, random_word: u64) -> bool {
        (random_word as u128) < self.numerator
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictableAuditDesign {
    pub design_id: String,
    pub design_manifest_ref: String,
    pub design_manifest_sha256: String,
    pub random_stream_manifest_ref: String,
    pub random_stream_manifest_sha256: String,
    /// Deterministic lower bound on every conditional q_t in this horizon.
    pub minimum_inclusion_probability: DyadicInclusionProbability,
    /// Documents how the q-policy is computed from pre-label history.
    pub q_policy_id: String,
    pub q_policy_version: String,
}

#[derive(Debug, Clone)]
pub struct PredictableAuditEvent {
    pub sequence_number: usize,
    pub record_id: String,
    /// SHA-256 of the exact recorded pre-label history used by the q-policy.
    pub prelabel_history_sha256: String,
    /// q_t must be chosen before the current outcome is known.
    pub inclusion_probability: DyadicInclusionProbability,
    /// Supplied by the external custodian's independent random stream.
    /// This value is not included in the returned selection receipt.
    pub random_word: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictableAuditSelection {
    pub sequence_number: usize,
    pub record_id: String,
    pub prelabel_history_sha256: String,
    pub inclusion_probability_numerator: u128,
    pub inclusion_probability_denominator: u128,
    pub inclusion_probability_f64_bits: u64,
    pub selected_for_adjudication: bool,
    /// Index into the external random-stream manifest. The raw draw is not
    /// returned, so this receipt can be shared without revealing audit choice
    /// to the candidate process.
    pub random_stream_index: usize,
}

impl PredictableAuditSelection {
    pub fn inclusion_probability(&self) -> f64 {
        f64::from_bits(self.inclusion_probability_f64_bits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictableAuditReceipt {
    pub receipt_version: String,
    pub design_id: String,
    pub design_manifest_ref: String,
    pub design_manifest_sha256: String,
    pub random_stream_manifest_ref: String,
    pub random_stream_manifest_sha256: String,
    pub q_policy_id: String,
    pub q_policy_version: String,
    pub minimum_inclusion_probability_numerator: u128,
    pub minimum_inclusion_probability_f64_bits: u64,
    pub population_size: usize,
    pub selected_count: usize,
    pub selected_fraction: f64,
    /// Complete cohort; nonselected records remain present.
    pub selections: Vec<PredictableAuditSelection>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictableAuditError {
    EmptyDesignId,
    EmptyDesignManifestReference,
    InvalidDesignManifestDigest,
    EmptyRandomStreamManifestReference,
    InvalidRandomStreamManifestDigest,
    EmptyQPolicyIdentity,
    InvalidProbabilityNumerator(u128),
    ProbabilityBelowMinimum { sequence_number: usize },
    EmptyCohort,
    EmptyRecordId,
    DuplicateRecordId(String),
    NonSequentialIndex { expected: usize, actual: usize },
    InvalidPrelabelHistoryDigest { sequence_number: usize },
}

impl std::fmt::Display for PredictableAuditError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDesignId => write!(formatter, "audit design ID must not be empty"),
            Self::EmptyDesignManifestReference => {
                write!(formatter, "audit design manifest reference must not be empty")
            }
            Self::InvalidDesignManifestDigest => {
                write!(formatter, "audit design manifest SHA-256 is invalid")
            }
            Self::EmptyRandomStreamManifestReference => {
                write!(formatter, "random stream manifest reference must not be empty")
            }
            Self::InvalidRandomStreamManifestDigest => {
                write!(formatter, "random stream manifest SHA-256 is invalid")
            }
            Self::EmptyQPolicyIdentity => {
                write!(formatter, "q-policy ID and version must not be empty")
            }
            Self::InvalidProbabilityNumerator(value) => {
                write!(formatter, "invalid dyadic inclusion numerator: {value}")
            }
            Self::ProbabilityBelowMinimum { sequence_number } => {
                write!(
                    formatter,
                    "q_t is below the declared floor at sequence {sequence_number}"
                )
            }
            Self::EmptyCohort => write!(formatter, "predictable audit cohort must not be empty"),
            Self::EmptyRecordId => write!(formatter, "audit record ID must not be empty"),
            Self::DuplicateRecordId(id) => write!(formatter, "duplicate audit record ID: {id}"),
            Self::NonSequentialIndex { expected, actual } => write!(
                formatter,
                "audit sequence mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidPrelabelHistoryDigest { sequence_number } => write!(
                formatter,
                "invalid pre-label history SHA-256 at sequence {sequence_number}"
            ),
        }
    }
}

impl std::error::Error for PredictableAuditError {}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Execute a complete predictable Bernoulli audit schedule using custodian
/// supplied random words. The q_t value and pre-label history digest are
/// recorded before the current outcome; the draw itself stays out of the
/// returned receipt.
///
/// The result is a randomized audit request schedule. SOWRC may use q_t as its
/// predictable inclusion probability only when every selected request yields
/// an independent label and the external design satisfies conditional
/// ignorability. Nonresponse is a separate selection mechanism and must not be
/// silently treated as a successful audit.
pub fn build_predictable_audit_receipt(
    events: &[PredictableAuditEvent],
    design: &PredictableAuditDesign,
) -> Result<PredictableAuditReceipt, PredictableAuditError> {
    if design.design_id.trim().is_empty() {
        return Err(PredictableAuditError::EmptyDesignId);
    }
    if design.design_manifest_ref.trim().is_empty() {
        return Err(PredictableAuditError::EmptyDesignManifestReference);
    }
    if !is_sha256_hex(&design.design_manifest_sha256) {
        return Err(PredictableAuditError::InvalidDesignManifestDigest);
    }
    if design.random_stream_manifest_ref.trim().is_empty() {
        return Err(PredictableAuditError::EmptyRandomStreamManifestReference);
    }
    if !is_sha256_hex(&design.random_stream_manifest_sha256) {
        return Err(PredictableAuditError::InvalidRandomStreamManifestDigest);
    }
    if design.q_policy_id.trim().is_empty() || design.q_policy_version.trim().is_empty() {
        return Err(PredictableAuditError::EmptyQPolicyIdentity);
    }
    design.minimum_inclusion_probability.validate()?;
    if events.is_empty() {
        return Err(PredictableAuditError::EmptyCohort);
    }

    let mut record_ids = HashSet::with_capacity(events.len());
    let mut selections = Vec::with_capacity(events.len());
    let mut selected_count = 0usize;

    for (index, event) in events.iter().enumerate() {
        if event.sequence_number != index {
            return Err(PredictableAuditError::NonSequentialIndex {
                expected: index,
                actual: event.sequence_number,
            });
        }
        if event.record_id.trim().is_empty() {
            return Err(PredictableAuditError::EmptyRecordId);
        }
        if !record_ids.insert(event.record_id.as_str()) {
            return Err(PredictableAuditError::DuplicateRecordId(
                event.record_id.clone(),
            ));
        }
        if !is_sha256_hex(&event.prelabel_history_sha256) {
            return Err(PredictableAuditError::InvalidPrelabelHistoryDigest {
                sequence_number: index,
            });
        }
        event.inclusion_probability.validate()?;
        if event.inclusion_probability.numerator
            < design.minimum_inclusion_probability.numerator
        {
            return Err(PredictableAuditError::ProbabilityBelowMinimum {
                sequence_number: index,
            });
        }

        let selected = event.inclusion_probability.selects(event.random_word);
        if selected {
            selected_count += 1;
        }
        selections.push(PredictableAuditSelection {
            sequence_number: index,
            record_id: event.record_id.clone(),
            prelabel_history_sha256: event.prelabel_history_sha256.clone(),
            inclusion_probability_numerator: event.inclusion_probability.numerator,
            inclusion_probability_denominator: DYADIC_DENOMINATOR,
            inclusion_probability_f64_bits: event.inclusion_probability.as_f64().to_bits(),
            selected_for_adjudication: selected,
            random_stream_index: index,
        });
    }

    Ok(PredictableAuditReceipt {
        receipt_version: PREDICTABLE_AUDIT_VERSION.to_owned(),
        design_id: design.design_id.clone(),
        design_manifest_ref: design.design_manifest_ref.clone(),
        design_manifest_sha256: design.design_manifest_sha256.clone(),
        random_stream_manifest_ref: design.random_stream_manifest_ref.clone(),
        random_stream_manifest_sha256: design.random_stream_manifest_sha256.clone(),
        q_policy_id: design.q_policy_id.clone(),
        q_policy_version: design.q_policy_version.clone(),
        minimum_inclusion_probability_numerator: design
            .minimum_inclusion_probability
            .numerator,
        minimum_inclusion_probability_f64_bits: design
            .minimum_inclusion_probability
            .as_f64()
            .to_bits(),
        population_size: events.len(),
        selected_count,
        selected_fraction: selected_count as f64 / events.len() as f64,
        selections,
        limitations: vec![
            "Random words are supplied by an external custodian; this module does not generate entropy or authenticate the random stream.".to_owned(),
            "The q-policy and its dependence on the committed pre-label history must be audited externally; the receipt cannot prove predictability.".to_owned(),
            "Selection is a request for an independent label. Selected-label nonresponse is a separate observation mechanism and invalidates direct use of q_t unless modelled.".to_owned(),
            "The candidate must not receive selected status or the random stream before its predictions are frozen.".to_owned(),
            "This is an independent Bernoulli-per-event audit design, not fixed-quota SRSWOR. q_t may be history-adaptive; cross-event inclusion probabilities are not summarized by a simple product when q_t depends on prior observations.".to_owned(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn design() -> PredictableAuditDesign {
        PredictableAuditDesign {
            design_id: "predictable-audit-v1".to_owned(),
            design_manifest_ref: "custody://audit-policy/v1".to_owned(),
            design_manifest_sha256: digest('a'),
            random_stream_manifest_ref: "custody://random-stream/v1".to_owned(),
            random_stream_manifest_sha256: digest('b'),
            minimum_inclusion_probability: DyadicInclusionProbability {
                numerator: 1_u128 << 63,
            },
            q_policy_id: "risk-stratum-policy".to_owned(),
            q_policy_version: "1".to_owned(),
        }
    }

    fn event(index: usize, q_numerator: u128, random_word: u64) -> PredictableAuditEvent {
        PredictableAuditEvent {
            sequence_number: index,
            record_id: format!("event-{index}"),
            prelabel_history_sha256: digest('c'),
            inclusion_probability: DyadicInclusionProbability {
                numerator: q_numerator,
            },
            random_word,
        }
    }

    #[test]
    fn dyadic_probability_is_exact_and_covers_certainty() {
        let half = DyadicInclusionProbability {
            numerator: 1_u128 << 63,
        };
        assert_eq!(half.as_f64(), 0.5);
        assert!(!half.selects(1_u64 << 63));
        assert!(half.selects((1_u64 << 63) - 1));

        let certain = DyadicInclusionProbability {
            numerator: DYADIC_DENOMINATOR,
        };
        assert_eq!(certain.as_f64(), 1.0);
        assert!(certain.selects(u64::MAX));
    }

    #[test]
    fn selection_uses_eventwise_predictable_q_and_external_random_words() {
        let events = vec![
            event(0, 1_u128 << 63, 0),
            event(1, 3_u128 << 62, u64::MAX),
            event(2, DYADIC_DENOMINATOR, u64::MAX),
        ];
        let receipt = build_predictable_audit_receipt(&events, &design()).unwrap();
        assert_eq!(receipt.population_size, 3);
        assert_eq!(receipt.selected_count, 2);
        assert!(receipt.selections[0].selected_for_adjudication);
        assert!(!receipt.selections[1].selected_for_adjudication);
        assert!(receipt.selections[2].selected_for_adjudication);
        assert_eq!(receipt.selections[1].inclusion_probability(), 0.75);
    }

    #[test]
    fn exact_dyadic_grid_rejects_unrepresentable_probabilities() {
        for numerator in [0, 1, (1_u128 << 63) + 1, DYADIC_DENOMINATOR + 1] {
            assert!(matches!(
                DyadicInclusionProbability { numerator }.validate(),
                Err(PredictableAuditError::InvalidProbabilityNumerator(_))
            ));
        }
    }

    #[test]
    fn zero_or_below_floor_probability_fails_closed() {
        let events = vec![event(0, 1_u128 << 62, 0)];
        assert_eq!(
            build_predictable_audit_receipt(&events, &design()).unwrap_err(),
            PredictableAuditError::ProbabilityBelowMinimum { sequence_number: 0 }
        );
    }

    #[test]
    fn q_min_is_positive_and_on_the_exact_dyadic_grid() {
        for numerator in [0, 1, (1_u128 << 62) + 1, DYADIC_DENOMINATOR + 1] {
            let invalid = PredictableAuditDesign {
                minimum_inclusion_probability: DyadicInclusionProbability { numerator },
                ..design()
            };
            assert!(matches!(
                build_predictable_audit_receipt(&[event(0, 1_u128 << 63, 0)], &invalid),
                Err(PredictableAuditError::InvalidProbabilityNumerator(_))
            ));
        }
    }

    #[test]
    fn history_and_manifest_digests_are_validated() {
        let mut invalid_event = event(0, 1_u128 << 63, 0);
        invalid_event.prelabel_history_sha256 = "no-digest".to_owned();
        assert!(matches!(
            build_predictable_audit_receipt(&[invalid_event], &design()),
            Err(PredictableAuditError::InvalidPrelabelHistoryDigest { .. })
        ));

        let mut invalid_design = design();
        invalid_design.random_stream_manifest_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            build_predictable_audit_receipt(&[event(0, 1_u128 << 63, 0)], &invalid_design)
                .unwrap_err(),
            PredictableAuditError::InvalidRandomStreamManifestDigest
        );
    }

    #[test]
    fn event_sequence_and_record_identity_are_exact() {
        let mut bad_sequence = event(1, 1_u128 << 63, 0);
        assert_eq!(
            build_predictable_audit_receipt(&[bad_sequence.clone()], &design()).unwrap_err(),
            PredictableAuditError::NonSequentialIndex {
                expected: 0,
                actual: 1,
            }
        );
        bad_sequence.sequence_number = 0;
        let duplicate = vec![
            event(0, 1_u128 << 63, 0),
            PredictableAuditEvent {
                sequence_number: 1,
                record_id: "event-0".to_owned(),
                ..event(1, 1_u128 << 63, 0)
            },
        ];
        assert!(matches!(
            build_predictable_audit_receipt(&duplicate, &design()),
            Err(PredictableAuditError::DuplicateRecordId(_))
        ));
    }

    #[test]
    fn output_receipt_does_not_expose_raw_random_words() {
        let receipt = build_predictable_audit_receipt(
            &[event(0, 1_u128 << 63, 123456789)],
            &design(),
        )
        .unwrap();
        let serialized = serde_json::to_value(receipt).unwrap();
        assert!(serialized["selections"][0].get("random_word").is_none());
        assert!(serialized["selections"][0].get("selected_for_adjudication").is_some());
        assert!(serialized["selections"][0].get("prelabel_history_sha256").is_some());
    }

    #[test]
    fn q_can_adapt_across_events_without_using_current_outcomes() {
        // The caller supplies q_t as a function of the prior public history
        // digest. The audit event type contains no current outcome field.
        let events = vec![
            event(0, 1_u128 << 63, 0),
            event(1, 3_u128 << 62, 0),
            event(2, 1_u128 << 62, 0),
        ];
        let receipt = build_predictable_audit_receipt(&events, &design()).unwrap();
        assert_eq!(
            receipt
                .selections
                .iter()
                .map(PredictableAuditSelection::inclusion_probability)
                .collect::<Vec<_>>(),
            vec![0.5, 0.75, 0.25]
        );
    }
}
