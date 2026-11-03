// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Precommitted, inspectable musical forecasts and append-only realization evidence.
//!
//! A forecast is a bounded statement about a future symbolic action, not a
//! transcript of hidden reasoning. The ledger distinguishes creation, display,
//! supersession and invalidation. Evaluation binds the forecast to the exact
//! input-history and immutable policy-identity digests that produced it.
//!
//! This module intentionally does not infer human anticipation, musician agency,
//! musical quality, or consciousness. Proper-score contributions from individual
//! fields are not a calibration claim; calibration requires a held-out population
//! of forecasts stratified by field, horizon, task and collaborator.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

pub const MUSICAL_FORECAST_SCHEMA_VERSION_V1: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastMusicalRoleV1 {
    Lead,
    Accompaniment,
    Bass,
    Counterline,
    Percussion,
    Pad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastArticulationV1 {
    Legato,
    Staccato,
    Tenuto,
    Accent,
    Marcato,
    Tremolo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastFieldV1 {
    BeatPosition,
    Duration,
    Pitch,
    Velocity,
    Role,
    Articulation,
}

impl ForecastFieldV1 {
    pub const ALL: [Self; 6] = [
        Self::BeatPosition,
        Self::Duration,
        Self::Pitch,
        Self::Velocity,
        Self::Role,
        Self::Articulation,
    ];
}

/// A probability is a prediction about whether one field will satisfy the
/// evaluator's declared match tolerance. Unavailable is explicit missingness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastConfidenceV1 {
    ProbabilityPpm { value: u32 },
    Unavailable { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastFieldConfidenceV1 {
    pub slot_id: String,
    pub field: ForecastFieldV1,
    pub confidence: ForecastConfidenceV1,
}

/// Symbolic event slot predicted by the agent. Beat positions and durations are
/// measured in micro-beats (1_000_000 units per quarter note); event slots are
/// stable within the forecast but need not expose a future renderer event ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicalForecastEventV1 {
    pub slot_id: String,
    pub beat_position_micros: u64,
    pub duration_micros: u64,
    pub pitch_midi: u8,
    pub velocity: u8,
    pub role: ForecastMusicalRoleV1,
    pub articulation: ForecastArticulationV1,
}

/// Forecast committed before the realization it describes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicalForecastV1 {
    pub schema_version: u32,
    pub forecast_id: String,
    pub session_id: String,
    pub turn_id: String,
    /// Monotonic sequence assigned by the caller and checked by the ledger.
    pub event_sequence: u64,
    /// Ticks from the declared clock domain; wall-clock timestamps are not
    /// substituted for monotonic order.
    pub clock_domain: String,
    pub created_at_tick: u64,
    pub target_window_start_tick: u64,
    pub target_window_end_tick: u64,
    pub expires_at_tick: u64,
    pub input_history_sha256: String,
    pub immutable_policy_identity_sha256: String,
    pub predicted_events: Vec<MusicalForecastEventV1>,
    /// Every predicted field has either a probability or an explicit unknown.
    pub confidence: Vec<ForecastFieldConfidenceV1>,
    /// A new record may supersede one earlier committed forecast. The older
    /// record remains in the ledger and is never rewritten.
    pub supersedes_forecast_id: Option<String>,
}

impl MusicalForecastV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != MUSICAL_FORECAST_SCHEMA_VERSION_V1 {
            return Err("unsupported musical forecast schema version".into());
        }
        for (name, value) in [
            ("forecast_id", self.forecast_id.as_str()),
            ("session_id", self.session_id.as_str()),
            ("turn_id", self.turn_id.as_str()),
            ("clock_domain", self.clock_domain.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{name} must not be empty"));
            }
        }
        if self.event_sequence == 0 {
            return Err("event_sequence must be positive".into());
        }
        if self.target_window_start_tick >= self.target_window_end_tick {
            return Err("target execution window must be non-empty".into());
        }
        if self.created_at_tick >= self.target_window_end_tick {
            return Err("forecast must be committed before the target window ends".into());
        }
        if self.expires_at_tick < self.target_window_end_tick {
            return Err("forecast expiry must cover the complete target window".into());
        }
        if !is_sha256(&self.input_history_sha256)
            || !is_sha256(&self.immutable_policy_identity_sha256)
        {
            return Err("input-history and immutable-policy identity must be lowercase SHA-256 digests".into());
        }
        if self.predicted_events.is_empty() {
            return Err("forecast must contain at least one predicted event".into());
        }

        let mut slots = BTreeSet::new();
        for event in &self.predicted_events {
            if event.slot_id.trim().is_empty() || !slots.insert(event.slot_id.as_str()) {
                return Err("predicted event slot IDs must be non-empty and unique".into());
            }
            if event.duration_micros == 0 {
                return Err(format!("event slot {} has zero duration", event.slot_id));
            }
        }

        let expected_confidence_count = self
            .predicted_events
            .len()
            .checked_mul(ForecastFieldV1::ALL.len())
            .ok_or_else(|| "confidence field count overflow".to_owned())?;
        if self.confidence.len() != expected_confidence_count {
            return Err("every predicted event field requires an explicit confidence or unavailable reason".into());
        }
        let mut confidence_keys = BTreeSet::new();
        for item in &self.confidence {
            if item.slot_id.trim().is_empty()
                || !slots.contains(item.slot_id.as_str())
                || !confidence_keys.insert((item.slot_id.as_str(), item.field))
            {
                return Err("confidence entries must map uniquely to predicted event fields".into());
            }
            match &item.confidence {
                ForecastConfidenceV1::ProbabilityPpm { value } if *value > 1_000_000 => {
                    return Err("confidence probability must be in 0..=1_000_000 ppm".into());
                }
                ForecastConfidenceV1::Unavailable { reason } if reason.trim().is_empty() => {
                    return Err("unavailable confidence requires a reason".into());
                }
                _ => {}
            }
        }
        if let Some(previous) = &self.supersedes_forecast_id {
            if previous.trim().is_empty() || previous == &self.forecast_id {
                return Err("superseded forecast ID must be non-empty and different".into());
            }
        }
        Ok(())
    }

    pub fn digest_sha256(&self) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        Ok(sha256_hex(&bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastDisplayReceiptV1 {
    pub forecast_id: String,
    pub recipient_id: String,
    pub displayed_at_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastInvalidationV1 {
    pub forecast_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum MusicalForecastLedgerEventKindV1 {
    Committed { forecast: MusicalForecastV1 },
    Displayed { receipt: ForecastDisplayReceiptV1 },
    Invalidated { invalidation: ForecastInvalidationV1 },
}

/// Append-only history. Fields are private so callers cannot silently rewrite
/// earlier previews or their displayed/supersession history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct MusicalForecastLedgerV1 {
    events: Vec<MusicalForecastLedgerEventV1>,
    last_sequence: u64,
    last_event_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicalForecastLedgerEventV1 {
    pub sequence: u64,
    pub event_tick: u64,
    pub event: MusicalForecastLedgerEventKindV1,
}

impl MusicalForecastLedgerV1 {
    pub fn events(&self) -> &[MusicalForecastLedgerEventV1] {
        &self.events
    }

    pub fn append_forecast(&mut self, forecast: MusicalForecastV1) -> Result<(), String> {
        forecast.validate()?;
        self.validate_next_event(forecast.event_sequence, forecast.created_at_tick)?;
        if self.forecast(&forecast.forecast_id).is_some() {
            return Err("forecast IDs are immutable and may be committed only once".into());
        }
        if let Some(previous_id) = &forecast.supersedes_forecast_id {
            let previous = self
                .forecast(previous_id)
                .ok_or_else(|| "superseded forecast must already exist in this ledger".to_owned())?;
            if previous.session_id != forecast.session_id {
                return Err("a forecast cannot supersede a different session".into());
            }
            if !self.is_active_at(previous_id, forecast.created_at_tick) {
                return Err("a forecast can supersede only an active prior forecast".into());
            }
        }
        let sequence = forecast.event_sequence;
        let event_tick = forecast.created_at_tick;
        self.events.push(MusicalForecastLedgerEventV1 {
            sequence,
            event_tick,
            event: MusicalForecastLedgerEventKindV1::Committed { forecast },
        });
        self.last_sequence = sequence;
        self.last_event_tick = event_tick;
        Ok(())
    }

    pub fn append_display_receipt(
        &mut self,
        sequence: u64,
        receipt: ForecastDisplayReceiptV1,
    ) -> Result<(), String> {
        let forecast = self
            .forecast(&receipt.forecast_id)
            .ok_or_else(|| "display receipt references an unknown forecast".to_owned())?;
        if receipt.recipient_id.trim().is_empty() {
            return Err("display recipient ID must not be empty".into());
        }
        if receipt.displayed_at_tick < forecast.created_at_tick {
            return Err("display cannot precede forecast commitment".into());
        }
        self.validate_next_event(sequence, receipt.displayed_at_tick)?;
        self.events.push(MusicalForecastLedgerEventV1 {
            sequence,
            event_tick: receipt.displayed_at_tick,
            event: MusicalForecastLedgerEventKindV1::Displayed { receipt },
        });
        self.last_sequence = sequence;
        self.last_event_tick = self.events.last().map_or(0, |item| item.event_tick);
        Ok(())
    }

    pub fn append_invalidation(
        &mut self,
        sequence: u64,
        event_tick: u64,
        invalidation: ForecastInvalidationV1,
    ) -> Result<(), String> {
        if self.forecast(&invalidation.forecast_id).is_none() {
            return Err("invalidation references an unknown forecast".into());
        }
        if invalidation.reason.trim().is_empty() {
            return Err("invalidation requires a reason".into());
        }
        if !self.is_active_at(&invalidation.forecast_id, event_tick) {
            return Err("only an active forecast can be invalidated".into());
        }
        self.validate_next_event(sequence, event_tick)?;
        self.events.push(MusicalForecastLedgerEventV1 {
            sequence,
            event_tick,
            event: MusicalForecastLedgerEventKindV1::Invalidated { invalidation },
        });
        self.last_sequence = sequence;
        self.last_event_tick = event_tick;
        Ok(())
    }

    pub fn forecast(&self, forecast_id: &str) -> Option<&MusicalForecastV1> {
        self.events.iter().find_map(|item| match &item.event {
            MusicalForecastLedgerEventKindV1::Committed { forecast }
                if forecast.forecast_id == forecast_id => Some(forecast),
            _ => None,
        })
    }

    pub fn displayed_before(&self, forecast_id: &str, event_tick: u64) -> bool {
        self.events.iter().any(|item| {
            item.event_tick <= event_tick
                && matches!(
                    &item.event,
                    MusicalForecastLedgerEventKindV1::Displayed { receipt }
                        if receipt.forecast_id == forecast_id
                )
        })
    }

    pub fn is_active_at(&self, forecast_id: &str, event_tick: u64) -> bool {
        if self.forecast(forecast_id).is_none() {
            return false;
        }
        !self.events.iter().any(|item| {
            if item.event_tick > event_tick {
                return false;
            }
            match &item.event {
                MusicalForecastLedgerEventKindV1::Committed { forecast } => {
                    forecast.supersedes_forecast_id.as_deref() == Some(forecast_id)
                }
                MusicalForecastLedgerEventKindV1::Invalidated { invalidation } => {
                    invalidation.forecast_id == forecast_id
                }
                MusicalForecastLedgerEventKindV1::Displayed { .. } => false,
            }
        })
    }

    fn validate_next_event(&self, sequence: u64, event_tick: u64) -> Result<(), String> {
        if sequence <= self.last_sequence {
            return Err("ledger event sequence must increase strictly".into());
        }
        if event_tick < self.last_event_tick {
            return Err("ledger event tick must be monotonic within its clock domain".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealizedMusicalEventV1 {
    pub slot_id: String,
    pub beat_position_micros: u64,
    pub duration_micros: u64,
    pub pitch_midi: u8,
    pub velocity: u8,
    pub role: ForecastMusicalRoleV1,
    pub articulation: ForecastArticulationV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastEvaluationContextV1 {
    pub session_id: String,
    pub input_history_sha256: String,
    pub immutable_policy_identity_sha256: String,
    pub realization_tick: u64,
    pub realized_events: Vec<RealizedMusicalEventV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastScoringPolicyV1 {
    pub beat_position_tolerance_micros: u64,
    pub duration_tolerance_micros: u64,
}

impl Default for ForecastScoringPolicyV1 {
    fn default() -> Self {
        Self {
            beat_position_tolerance_micros: 10_000,
            duration_tolerance_micros: 10_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastFieldOutcomeV1 {
    pub slot_id: String,
    pub field: ForecastFieldV1,
    pub correct: bool,
    pub confidence: ForecastConfidenceV1,
    /// One binary proper-score contribution on a 0..=1,000,000 scale.
    /// Aggregating this across one forecast is not a calibration estimate.
    pub brier_score_millionths: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastEvaluationV1 {
    pub forecast_id: String,
    pub forecast_sha256: String,
    pub committed_before_realization: bool,
    pub displayed_before_realization: bool,
    pub predicted_event_count: usize,
    pub matched_event_count: usize,
    pub missing_realized_event_count: usize,
    pub unpredicted_realized_event_count: usize,
    pub correct_field_count: usize,
    pub scored_field_count: usize,
    pub unavailable_confidence_count: usize,
    pub mean_brier_score_millionths: Option<u32>,
    pub field_outcomes: Vec<ForecastFieldOutcomeV1>,
}

pub fn evaluate_musical_forecast_v1(
    ledger: &MusicalForecastLedgerV1,
    forecast_id: &str,
    context: &ForecastEvaluationContextV1,
    scoring: ForecastScoringPolicyV1,
) -> Result<ForecastEvaluationV1, String> {
    let forecast = ledger
        .forecast(forecast_id)
        .ok_or_else(|| "forecast is not committed in the append-only ledger".to_owned())?;
    forecast.validate()?;
    if forecast.session_id != context.session_id {
        return Err("forecast session does not match realization session".into());
    }
    if forecast.input_history_sha256 != context.input_history_sha256 {
        return Err("input-history digest mismatch (non-contingent/yoked preview)".into());
    }
    if forecast.immutable_policy_identity_sha256 != context.immutable_policy_identity_sha256 {
        return Err("immutable policy identity digest mismatch".into());
    }
    if forecast.created_at_tick >= context.realization_tick {
        return Err("post-hoc forecast cannot be scored as anticipatory".into());
    }
    if context.realization_tick < forecast.target_window_start_tick
        || context.realization_tick > forecast.target_window_end_tick
        || context.realization_tick > forecast.expires_at_tick
    {
        return Err("forecast is stale or realization is outside its target window".into());
    }
    if !ledger.is_active_at(forecast_id, context.realization_tick) {
        return Err("forecast was superseded or invalidated before realization".into());
    }
    let mut actual_by_slot = std::collections::BTreeMap::new();
    for event in &context.realized_events {
        if event.slot_id.trim().is_empty()
            || actual_by_slot.insert(event.slot_id.as_str(), event).is_some()
        {
            return Err("realized event slot IDs must be non-empty and unique".into());
        }
        if event.duration_micros == 0 {
            return Err("realized event duration must be positive".into());
        }
    }

    let forecast_sha256 = forecast.digest_sha256()?;
    let mut outcomes = Vec::with_capacity(forecast.predicted_events.len() * 6);
    let mut matched_event_count = 0usize;
    let mut correct_field_count = 0usize;
    let mut unavailable_confidence_count = 0usize;
    let mut brier_sum = 0u128;
    let mut brier_count = 0u128;

    for predicted in &forecast.predicted_events {
        let actual = actual_by_slot.get(predicted.slot_id.as_str()).copied();
        if actual.is_some() {
            matched_event_count += 1;
        }
        for field in ForecastFieldV1::ALL {
            let correct = actual.is_some_and(|observed| {
                match field {
                    ForecastFieldV1::BeatPosition => predicted
                        .beat_position_micros
                        .abs_diff(observed.beat_position_micros)
                        <= scoring.beat_position_tolerance_micros,
                    ForecastFieldV1::Duration => predicted
                        .duration_micros
                        .abs_diff(observed.duration_micros)
                        <= scoring.duration_tolerance_micros,
                    ForecastFieldV1::Pitch => predicted.pitch_midi == observed.pitch_midi,
                    ForecastFieldV1::Velocity => predicted.velocity == observed.velocity,
                    ForecastFieldV1::Role => predicted.role == observed.role,
                    ForecastFieldV1::Articulation => predicted.articulation == observed.articulation,
                }
            });
            correct_field_count += usize::from(correct);
            let confidence = forecast
                .confidence
                .iter()
                .find(|item| item.slot_id == predicted.slot_id && item.field == field)
                .ok_or_else(|| "validated confidence entry disappeared".to_owned())?
                .confidence
                .clone();
            let brier = match &confidence {
                ForecastConfidenceV1::ProbabilityPpm { value } => {
                    let outcome_ppm = if correct { 1_000_000u64 } else { 0 };
                    let error = i64::from(*value) - outcome_ppm as i64;
                    let score = ((error * error) / 1_000_000) as u32;
                    brier_sum += u128::from(score);
                    brier_count += 1;
                    Some(score)
                }
                ForecastConfidenceV1::Unavailable { .. } => {
                    unavailable_confidence_count += 1;
                    None
                }
            };
            outcomes.push(ForecastFieldOutcomeV1 {
                slot_id: predicted.slot_id.clone(),
                field,
                correct,
                confidence,
                brier_score_millionths: brier,
            });
        }
    }

    let predicted_ids: BTreeSet<&str> = forecast
        .predicted_events
        .iter()
        .map(|event| event.slot_id.as_str())
        .collect();
    let realized_ids: BTreeSet<&str> = actual_by_slot.keys().copied().collect();
    let missing_realized_event_count = predicted_ids.difference(&realized_ids).count();
    let unpredicted_realized_event_count = realized_ids.difference(&predicted_ids).count();
    let mean_brier_score_millionths = if brier_count == 0 {
        None
    } else {
        Some((brier_sum / brier_count) as u32)
    };

    Ok(ForecastEvaluationV1 {
        forecast_id: forecast_id.to_owned(),
        forecast_sha256,
        committed_before_realization: true,
        displayed_before_realization: ledger.displayed_before(forecast_id, context.realization_tick),
        predicted_event_count: forecast.predicted_events.len(),
        matched_event_count,
        missing_realized_event_count,
        unpredicted_realized_event_count,
        correct_field_count,
        scored_field_count: outcomes.len(),
        unavailable_confidence_count,
        mean_brier_score_millionths,
        field_outcomes: outcomes,
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: &str) -> String {
        sha256_hex(seed.as_bytes())
    }

    fn event() -> MusicalForecastEventV1 {
        MusicalForecastEventV1 {
            slot_id: "next-note-0".into(),
            beat_position_micros: 2_000_000,
            duration_micros: 500_000,
            pitch_midi: 64,
            velocity: 88,
            role: ForecastMusicalRoleV1::Lead,
            articulation: ForecastArticulationV1::Accent,
        }
    }

    fn forecast(
        sequence: u64,
        created_at_tick: u64,
        supersedes_forecast_id: Option<String>,
    ) -> MusicalForecastV1 {
        let predicted = event();
        let confidence = ForecastFieldV1::ALL
            .into_iter()
            .map(|field| ForecastFieldConfidenceV1 {
                slot_id: predicted.slot_id.clone(),
                field,
                confidence: ForecastConfidenceV1::ProbabilityPpm { value: 900_000 },
            })
            .collect();
        MusicalForecastV1 {
            schema_version: MUSICAL_FORECAST_SCHEMA_VERSION_V1,
            forecast_id: format!("forecast-{sequence}"),
            session_id: "session-1".into(),
            turn_id: "turn-1".into(),
            event_sequence: sequence,
            clock_domain: "fixture-monotonic-v1".into(),
            created_at_tick,
            target_window_start_tick: 10,
            target_window_end_tick: 30,
            expires_at_tick: 30,
            input_history_sha256: digest("input-history"),
            immutable_policy_identity_sha256: digest("policy-identity"),
            predicted_events: vec![predicted],
            confidence,
            supersedes_forecast_id,
        }
    }

    fn realized_event() -> RealizedMusicalEventV1 {
        let predicted = event();
        RealizedMusicalEventV1 {
            slot_id: predicted.slot_id,
            beat_position_micros: predicted.beat_position_micros,
            duration_micros: predicted.duration_micros,
            pitch_midi: predicted.pitch_midi,
            velocity: predicted.velocity,
            role: predicted.role,
            articulation: predicted.articulation,
        }
    }

    fn context(realization_tick: u64) -> ForecastEvaluationContextV1 {
        ForecastEvaluationContextV1 {
            session_id: "session-1".into(),
            input_history_sha256: digest("input-history"),
            immutable_policy_identity_sha256: digest("policy-identity"),
            realization_tick,
            realized_events: vec![realized_event()],
        }
    }

    #[test]
    fn precommitted_forecast_matches_symbolic_realization() {
        let mut ledger = MusicalForecastLedgerV1::default();
        ledger.append_forecast(forecast(1, 5, None)).unwrap();
        ledger
            .append_display_receipt(
                2,
                ForecastDisplayReceiptV1 {
                    forecast_id: "forecast-1".into(),
                    recipient_id: "musician-1".into(),
                    displayed_at_tick: 6,
                },
            )
            .unwrap();

        let score = evaluate_musical_forecast_v1(
            &ledger,
            "forecast-1",
            &context(12),
            ForecastScoringPolicyV1::default(),
        )
        .unwrap();
        assert!(score.committed_before_realization);
        assert!(score.displayed_before_realization);
        assert_eq!(score.predicted_event_count, 1);
        assert_eq!(score.matched_event_count, 1);
        assert_eq!(score.correct_field_count, 6);
        assert_eq!(score.scored_field_count, 6);
        assert_eq!(score.mean_brier_score_millionths, Some(10_000));
    }

    #[test]
    fn post_hoc_forecast_is_rejected() {
        let mut late = forecast(1, 15, None);
        late.target_window_start_tick = 10;
        late.target_window_end_tick = 30;
        late.expires_at_tick = 30;
        let mut ledger = MusicalForecastLedgerV1::default();
        ledger.append_forecast(late).unwrap();
        let result = evaluate_musical_forecast_v1(
            &ledger,
            "forecast-1",
            &context(12),
            ForecastScoringPolicyV1::default(),
        );
        assert!(result.unwrap_err().contains("post-hoc"));
    }

    #[test]
    fn yoked_preview_with_different_input_history_is_rejected() {
        let mut ledger = MusicalForecastLedgerV1::default();
        ledger.append_forecast(forecast(1, 5, None)).unwrap();
        let mut wrong_history = context(12);
        wrong_history.input_history_sha256 = digest("different-session-history");
        let result = evaluate_musical_forecast_v1(
            &ledger,
            "forecast-1",
            &wrong_history,
            ForecastScoringPolicyV1::default(),
        );
        assert!(result.unwrap_err().contains("yoked"));
    }

    #[test]
    fn expired_preview_is_rejected_as_stale() {
        let mut ledger = MusicalForecastLedgerV1::default();
        ledger.append_forecast(forecast(1, 5, None)).unwrap();
        let mut late_context = context(31);
        late_context.realization_tick = 31;
        let result = evaluate_musical_forecast_v1(
            &ledger,
            "forecast-1",
            &late_context,
            ForecastScoringPolicyV1::default(),
        );
        assert!(result.unwrap_err().contains("stale"));
    }

    #[test]
    fn supersession_keeps_old_record_but_invalidates_it_for_later_realization() {
        let mut ledger = MusicalForecastLedgerV1::default();
        let first = forecast(1, 5, None);
        ledger.append_forecast(first.clone()).unwrap();
        let mut revision = forecast(2, 8, Some(first.forecast_id.clone()));
        revision.forecast_id = "forecast-2".into();
        ledger.append_forecast(revision).unwrap();

        assert_eq!(ledger.events().len(), 2);
        assert!(ledger.forecast("forecast-1").is_some());
        assert!(!ledger.is_active_at("forecast-1", 12));
        assert!(ledger.is_active_at("forecast-1", 7));
        assert!(ledger.is_active_at("forecast-2", 12));
        let result = evaluate_musical_forecast_v1(
            &ledger,
            "forecast-1",
            &context(12),
            ForecastScoringPolicyV1::default(),
        );
        assert!(result.unwrap_err().contains("superseded"));
    }

    #[test]
    fn supersession_cannot_branch_from_an_already_superseded_forecast() {
        let mut ledger = MusicalForecastLedgerV1::default();
        let first = forecast(1, 5, None);
        ledger.append_forecast(first.clone()).unwrap();
        let second = forecast(2, 8, Some(first.forecast_id.clone()));
        ledger.append_forecast(second).unwrap();
        let third = forecast(3, 9, Some(first.forecast_id.clone()));
        assert!(ledger.append_forecast(third).is_err());
    }

    #[test]
    fn unknown_confidence_is_explicit_and_not_scored_as_probability() {
        let mut item = forecast(1, 5, None);
        for confidence in &mut item.confidence {
            confidence.confidence = ForecastConfidenceV1::Unavailable {
                reason: "insufficient validation data".into(),
            };
        }
        let mut ledger = MusicalForecastLedgerV1::default();
        ledger.append_forecast(item).unwrap();
        let score = evaluate_musical_forecast_v1(
            &ledger,
            "forecast-1",
            &context(12),
            ForecastScoringPolicyV1::default(),
        )
        .unwrap();
        assert_eq!(score.unavailable_confidence_count, 6);
        assert_eq!(score.mean_brier_score_millionths, None);
    }

    #[test]
    fn immutable_policy_identity_mismatch_is_rejected() {
        let mut ledger = MusicalForecastLedgerV1::default();
        ledger.append_forecast(forecast(1, 5, None)).unwrap();
        let mut other_model = context(12);
        other_model.immutable_policy_identity_sha256 = digest("another-model");
        let result = evaluate_musical_forecast_v1(
            &ledger,
            "forecast-1",
            &other_model,
            ForecastScoringPolicyV1::default(),
        );
        assert!(result.unwrap_err().contains("identity digest"));
    }

    #[test]
    fn ledger_rejects_rewritten_sequence_or_display_before_commit() {
        let mut ledger = MusicalForecastLedgerV1::default();
        ledger.append_forecast(forecast(1, 5, None)).unwrap();
        assert!(ledger.append_forecast(forecast(1, 6, None)).is_err());
        assert!(ledger
            .append_display_receipt(
                2,
                ForecastDisplayReceiptV1 {
                    forecast_id: "forecast-1".into(),
                    recipient_id: "musician-1".into(),
                    displayed_at_tick: 4,
                },
            )
            .is_err());
    }

    #[test]
    fn confidence_coverage_is_mandatory_for_every_predicted_field() {
        let mut incomplete = forecast(1, 5, None);
        incomplete.confidence.pop();
        assert!(incomplete.validate().is_err());
    }
}
