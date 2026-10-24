// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, cycle-bound prediction-error evidence.
//!
//! A numeric prediction error is not sufficient provenance. This type keeps
//! validity status separate from the compatibility float used by older APIs.
//! Invalid evidence never carries a value, and deserialization validates the
//! same invariant as construction so malformed persisted records fail closed.

use serde::{Deserialize, Deserializer, Serialize};

/// Why a prediction-error comparison is or is not usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionEvidenceStatus {
    /// A finite comparison is available for the recorded observation cycle.
    Valid,
    /// The cognitive loop has not yet produced a prior prediction.
    NoPriorPrediction,
    /// A prediction exists but cannot be compared because its norm is zero.
    ZeroNormPrediction,
    /// The observation did not contain a comparable state.
    EmptyObservation,
    /// Prediction and observation dimensions do not match.
    DimensionMismatch,
    /// A purported comparison contains a NaN or infinite numeric value.
    NonFiniteInput,
    /// Evidence belongs to a different or out-of-order observation cycle.
    StaleOrOutOfOrder,
    /// The upstream source did not provide a trustworthy comparison.
    SourceUnavailable,
    /// The cycle was gated before a prediction comparison was evaluated.
    NotEvaluated,
}

/// A prediction error with status and observation provenance.
///
/// The numeric value is intentionally private and accessible only through
/// value_for_observation; invalid statuses must not carry a numeric value.
/// The observation cycle is always explicit. Prediction-cycle provenance is
/// optional while legacy call sites are being migrated.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PredictionErrorEvidence {
    status: PredictionEvidenceStatus,
    value: Option<f32>,
    prediction_cycle: Option<u64>,
    observation_cycle: u64,
}

#[derive(Deserialize)]
struct PredictionErrorEvidenceWire {
    status: PredictionEvidenceStatus,
    value: Option<f32>,
    prediction_cycle: Option<u64>,
    observation_cycle: u64,
}

impl<'de> Deserialize<'de> for PredictionErrorEvidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error as _;

        let wire = PredictionErrorEvidenceWire::deserialize(deserializer)?;
        match wire.status {
            PredictionEvidenceStatus::Valid => {
                let value = wire.value.ok_or_else(|| {
                    D::Error::custom("valid prediction evidence requires a value")
                })?;
                if !value.is_finite() {
                    return Err(D::Error::custom(
                        "valid prediction evidence must contain a finite value",
                    ));
                }
                if let Some(prediction_cycle) = wire.prediction_cycle {
                    if wire.observation_cycle.checked_sub(1) != Some(prediction_cycle) {
                        return Err(D::Error::custom(
                            "valid prediction evidence must come from the preceding cycle",
                        ));
                    }
                }
                Ok(Self {
                    status: wire.status,
                    value: Some(value),
                    prediction_cycle: wire.prediction_cycle,
                    observation_cycle: wire.observation_cycle,
                })
            }
            _ => {
                if wire.value.is_some() {
                    return Err(D::Error::custom(
                        "invalid prediction evidence must not contain a value",
                    ));
                }
                Ok(Self {
                    status: wire.status,
                    value: None,
                    prediction_cycle: wire.prediction_cycle,
                    observation_cycle: wire.observation_cycle,
                })
            }
        }
    }
}

/// Per-consumer monotonic cursor for exactly-once PE admission.
///
/// Each PE-dependent consumer owns its own cursor. A missing or non-finite
/// sample does not advance the cursor, so a valid retry for that observation
/// can still be admitted. A duplicate or out-of-order observation is rejected.
/// Independent non-PE inputs are not governed by this cursor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PredictionEvidenceCursor {
    last_admitted_observation_cycle: Option<u64>,
}

impl PredictionEvidenceCursor {
    /// Admit a finite PE sample only if its observation cycle is newer than
    /// every previously admitted PE sample for this consumer.
    pub fn admit(
        &mut self,
        observation_cycle: u64,
        sample: Option<f32>,
    ) -> Option<f32> {
        let value = sample.filter(|value| value.is_finite())?;
        if self
            .last_admitted_observation_cycle
            .is_some_and(|last_cycle| observation_cycle <= last_cycle)
        {
            return None;
        }
        self.last_admitted_observation_cycle = Some(observation_cycle);
        Some(value)
    }

    /// Highest observation cycle whose PE sample was admitted.
    #[inline]
    pub fn last_admitted_observation_cycle(&self) -> Option<u64> {
        self.last_admitted_observation_cycle
    }

    /// Restore the cursor from a validated consumer checkpoint.
    ///
    /// This is crate-visible so checkpoint owners can preserve exactly-once
    /// admission across restarts without exposing the cursor's representation.
    pub(crate) fn restore_last_admitted_observation_cycle(&mut self, cycle: Option<u64>) {
        self.last_admitted_observation_cycle = cycle;
    }

    /// Reset explicitly when starting a new independent stream/session.
    #[inline]
    pub fn reset(&mut self) {
        self.last_admitted_observation_cycle = None;
    }
}

impl PredictionErrorEvidence {
    /// Construct evidence from an explicit status and optional measurement.
    ///
    /// A Valid status without a finite value is downgraded to a non-usable
    /// status. Invalid statuses always discard the numeric payload.
    pub fn from_comparison(
        status: PredictionEvidenceStatus,
        value: Option<f32>,
        prediction_cycle: Option<u64>,
        observation_cycle: u64,
    ) -> Self {
        let (status, value) = match status {
            PredictionEvidenceStatus::Valid => match value {
                Some(value) if !value.is_finite() => {
                    (PredictionEvidenceStatus::NonFiniteInput, None)
                }
                Some(_)
                    if prediction_cycle.is_some()
                        && observation_cycle.checked_sub(1) != prediction_cycle =>
                {
                    (PredictionEvidenceStatus::StaleOrOutOfOrder, None)
                }
                Some(value) => (PredictionEvidenceStatus::Valid, Some(value)),
                None => (PredictionEvidenceStatus::SourceUnavailable, None),
            },
            invalid_status => (invalid_status, None),
        };
        Self {
            status,
            value,
            prediction_cycle,
            observation_cycle,
        }
    }

    /// Build valid evidence. Non-finite values are rejected rather than
    /// silently normalized to a sentinel.
    pub fn valid(
        value: f32,
        prediction_cycle: Option<u64>,
        observation_cycle: u64,
    ) -> Result<Self, &'static str> {
        if !value.is_finite() {
            return Err("prediction error must be finite");
        }
        if let Some(prediction_cycle) = prediction_cycle {
            if observation_cycle.checked_sub(1) != Some(prediction_cycle) {
                return Err("prediction cycle must immediately precede observation cycle");
            }
        }
        Ok(Self {
            status: PredictionEvidenceStatus::Valid,
            value: Some(value),
            prediction_cycle,
            observation_cycle,
        })
    }

    /// Build evidence for an explicitly invalid comparison.
    pub fn invalid(
        status: PredictionEvidenceStatus,
        prediction_cycle: Option<u64>,
        observation_cycle: u64,
    ) -> Result<Self, &'static str> {
        if status == PredictionEvidenceStatus::Valid {
            return Err("invalid evidence cannot use the Valid status");
        }
        Ok(Self {
            status,
            value: None,
            prediction_cycle,
            observation_cycle,
        })
    }

    /// Adapt legacy float-plus-boolean APIs without trusting the float when
    /// their validity flag is false. A non-finite value marked valid is
    /// downgraded to NonFiniteInput.
    pub fn from_compatibility_fields(
        value: f32,
        valid: bool,
        prediction_cycle: Option<u64>,
        observation_cycle: u64,
    ) -> Self {
        if !valid {
            return Self {
                status: PredictionEvidenceStatus::SourceUnavailable,
                value: None,
                prediction_cycle,
                observation_cycle,
            };
        }
        if !value.is_finite() {
            return Self {
                status: PredictionEvidenceStatus::NonFiniteInput,
                value: None,
                prediction_cycle,
                observation_cycle,
            };
        }
        Self::from_comparison(
            PredictionEvidenceStatus::Valid,
            Some(value),
            prediction_cycle,
            observation_cycle,
        )
    }

    /// Evidence classification.
    #[inline]
    pub fn status(&self) -> PredictionEvidenceStatus {
        self.status
    }

    /// Report staleness relative to a consumer's requested observation cycle.
    #[inline]
    pub fn status_for_observation(
        &self,
        observation_cycle: u64,
    ) -> PredictionEvidenceStatus {
        let source_cycle_inconsistent = self.prediction_cycle.is_some()
            && self.observation_cycle.checked_sub(1) != self.prediction_cycle;
        if self.status == PredictionEvidenceStatus::Valid
            && (self.observation_cycle != observation_cycle || source_cycle_inconsistent)
        {
            PredictionEvidenceStatus::StaleOrOutOfOrder
        } else {
            self.status
        }
    }

    /// Originating cycle for the prediction, when known.
    #[inline]
    pub fn prediction_cycle(&self) -> Option<u64> {
        self.prediction_cycle
    }

    /// Cycle whose observation was compared.
    #[inline]
    pub fn observation_cycle(&self) -> u64 {
        self.observation_cycle
    }

    /// Return a value only when the evidence is valid and belongs to the
    /// requested observation cycle. This prevents reusing an old sample as
    /// though it were a fresh outcome.
    #[inline]
    pub fn value_for_observation(&self, observation_cycle: u64) -> Option<f32> {
        if self.status_for_observation(observation_cycle) != PredictionEvidenceStatus::Valid {
            return None;
        }
        self.value.filter(|value| value.is_finite())
    }

    /// Whether this evidence can be consumed for the requested cycle.
    #[inline]
    pub fn is_usable_for(&self, observation_cycle: u64) -> bool {
        self.value_for_observation(observation_cycle).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::{PredictionErrorEvidence, PredictionEvidenceStatus};

    #[test]
    fn prediction_evidence_cursor_admits_each_cycle_once() {
        use super::PredictionEvidenceCursor;

        let mut cursor = PredictionEvidenceCursor::default();
        assert_eq!(cursor.admit(10, Some(0.3)), Some(0.3));
        assert_eq!(cursor.admit(10, Some(0.9)), None);
        assert_eq!(cursor.admit(9, Some(0.8)), None);
        assert_eq!(cursor.admit(11, Some(0.6)), Some(0.6));
        assert_eq!(cursor.last_admitted_observation_cycle(), Some(11));
    }

    #[test]
    fn missing_evidence_does_not_advance_idempotency_cursor() {
        use super::PredictionEvidenceCursor;

        let mut cursor = PredictionEvidenceCursor::default();
        assert_eq!(cursor.admit(5, None), None);
        assert_eq!(cursor.admit(5, Some(f32::NAN)), None);
        assert_eq!(cursor.last_admitted_observation_cycle(), None);
        assert_eq!(cursor.admit(5, Some(0.4)), Some(0.4));
    }

    #[test]
    fn prediction_evidence_cursor_resets_only_explicitly() {
        use super::PredictionEvidenceCursor;

        let mut cursor = PredictionEvidenceCursor::default();
        assert_eq!(cursor.admit(100, Some(0.2)), Some(0.2));
        assert_eq!(cursor.admit(1, Some(0.3)), None);
        cursor.reset();
        assert_eq!(cursor.admit(1, Some(0.3)), Some(0.3));
    }

    #[test]
    fn invalid_compatibility_payload_never_becomes_a_value() {
        for payload in [0.0, 1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let evidence =
                PredictionErrorEvidence::from_compatibility_fields(payload, false, Some(4), 5);
            assert_eq!(evidence.status(), PredictionEvidenceStatus::SourceUnavailable);
            assert_eq!(evidence.value_for_observation(5), None);
        }
    }

    #[test]
    fn non_finite_value_marked_valid_is_downgraded() {
        for payload in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let evidence =
                PredictionErrorEvidence::from_compatibility_fields(payload, true, Some(4), 5);
            assert_eq!(evidence.status(), PredictionEvidenceStatus::NonFiniteInput);
            assert_eq!(evidence.value_for_observation(5), None);
        }
    }

    #[test]
    fn valid_evidence_is_only_available_for_its_observation_cycle() {
        let evidence = PredictionErrorEvidence::valid(0.25, Some(4), 5).unwrap();
        assert_eq!(evidence.value_for_observation(5), Some(0.25));
        assert_eq!(evidence.value_for_observation(6), None);
        assert!(!evidence.is_usable_for(6));
    }

    #[test]
    fn not_evaluated_status_remains_missing_evidence() {
        let evidence = PredictionErrorEvidence::from_comparison(
            PredictionEvidenceStatus::NotEvaluated,
            Some(0.0),
            None,
            7,
        );
        assert_eq!(evidence.status(), PredictionEvidenceStatus::NotEvaluated);
        assert_eq!(evidence.value_for_observation(7), None);
    }

    #[test]
    fn explicit_comparison_status_cannot_leak_invalid_numeric_payloads() {
        for value in [Some(0.0), Some(1.0), Some(f32::NAN), Some(f32::INFINITY), None] {
            let evidence = PredictionErrorEvidence::from_comparison(
                PredictionEvidenceStatus::NoPriorPrediction,
                value,
                None,
                7,
            );
            assert_eq!(evidence.status(), PredictionEvidenceStatus::NoPriorPrediction);
            assert_eq!(evidence.value_for_observation(7), None);
        }

        let non_finite = PredictionErrorEvidence::from_comparison(
            PredictionEvidenceStatus::Valid,
            Some(f32::NAN),
            None,
            7,
        );
        assert_eq!(non_finite.status(), PredictionEvidenceStatus::NonFiniteInput);
        assert_eq!(non_finite.value_for_observation(7), None);
    }

    #[test]
    fn inconsistent_explicit_prediction_cycle_is_rejected_or_downgraded() {
        assert!(PredictionErrorEvidence::valid(0.5, Some(3), 5).is_err());

        let evidence = PredictionErrorEvidence::from_comparison(
            PredictionEvidenceStatus::Valid,
            Some(0.5),
            Some(3),
            5,
        );
        assert_eq!(evidence.status(), PredictionEvidenceStatus::StaleOrOutOfOrder);
        assert_eq!(evidence.value_for_observation(5), None);
    }

    #[test]
    fn invalid_constructor_rejects_valid_status() {
        assert!(
            PredictionErrorEvidence::invalid(PredictionEvidenceStatus::Valid, None, 1).is_err()
        );
        assert!(PredictionErrorEvidence::valid(f32::NAN, None, 1).is_err());
    }

    #[test]
    fn stale_evidence_is_classified_and_cannot_be_consumed() {
        let evidence = PredictionErrorEvidence::valid(0.5, Some(10), 11).unwrap();
        assert_eq!(
            evidence.status_for_observation(12),
            PredictionEvidenceStatus::StaleOrOutOfOrder
        );
        assert_eq!(evidence.value_for_observation(12), None);
    }

    #[test]
    fn deterministic_10k_payload_sweep_fails_closed() {
        // Fixed xorshift sequence: reproducible coverage without an extra
        // property-testing dependency. This tests the typed boundary, not all
        // downstream cognitive-loop consumers.
        let mut bits = 0xC0FF_EE01u32;
        for index in 0..10_000u64 {
            bits ^= bits << 13;
            bits ^= bits >> 17;
            bits ^= bits << 5;

            let payload = f32::from_bits(bits);
            let valid_flag = index % 3 != 0;
            let evidence = PredictionErrorEvidence::from_compatibility_fields(
                payload,
                valid_flag,
                Some(index.saturating_sub(1)),
                index,
            );

            let expected = if !valid_flag {
                None
            } else if payload.is_finite() {
                Some(payload)
            } else {
                None
            };
            assert_eq!(evidence.value_for_observation(index), expected);
            assert_eq!(evidence.value_for_observation(index.saturating_add(1)), None);
        }
    }

    #[test]
    fn serde_round_trip_preserves_valid_evidence() {
        let evidence = PredictionErrorEvidence::valid(0.75, Some(8), 9).unwrap();
        let json = serde_json::to_string(&evidence).unwrap();
        let restored: PredictionErrorEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, evidence);
        assert_eq!(restored.value_for_observation(9), Some(0.75));
    }

    #[test]
    fn serde_rejects_inconsistent_validity_and_value() {
        let missing = r#"{"status":"valid","value":null,"prediction_cycle":1,"observation_cycle":2}"#;
        assert!(serde_json::from_str::<PredictionErrorEvidence>(missing).is_err());

        let invalid_with_value =
            r#"{"status":"source_unavailable","value":1.0,"prediction_cycle":1,"observation_cycle":2}"#;
        assert!(serde_json::from_str::<PredictionErrorEvidence>(invalid_with_value).is_err());

        let skipped_cycle = r#"{"status":"valid","value":0.5,"prediction_cycle":3,"observation_cycle":5}"#;
        assert!(serde_json::from_str::<PredictionErrorEvidence>(skipped_cycle).is_err());

        let non_finite =
            r#"{"status":"valid","value":1e999,"prediction_cycle":1,"observation_cycle":2}"#;
        assert!(serde_json::from_str::<PredictionErrorEvidence>(non_finite).is_err());

        let stale = r#"{"status":"valid","value":0.5,"prediction_cycle":1,"observation_cycle":3}"#;
        assert!(serde_json::from_str::<PredictionErrorEvidence>(stale).is_err());
    }
}
