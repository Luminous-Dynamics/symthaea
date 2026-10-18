// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A versioned, self-report-only record for studying music-listener response.
//! Separate from symbolic musical-outcome calibration: a score effect is not
//! a prediction about a person's affect. Missing answers remain missing.

use serde::{Deserialize, Serialize};

pub const LISTENER_RESPONSE_SCHEMA_VERSION: &str = "muse-listener-response-v1";

/// The outcome the listener says they want from this listening episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListeningIntent {
    MatchCurrentState,
    ShiftTowardTarget,
    SustainFocus,
    Energize,
    WindDown,
    Explore,
    Enjoy,
    Other,
}

/// Coarse context chosen by the listener rather than inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListeningActivity {
    Resting,
    WorkingOrStudying,
    MovingOrExercising,
    Socializing,
    Creating,
    Other,
}

/// One optional self-report vector, not an objective or clinical measurement.
/// Scales: valence [-1, 1]; arousal, tension, and energy [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct SelfReportedAffect {
    pub valence: Option<f32>,
    pub arousal: Option<f32>,
    pub tension: Option<f32>,
    pub energy: Option<f32>,
}

impl SelfReportedAffect {
    pub fn is_empty(&self) -> bool {
        self.valence.is_none()
            && self.arousal.is_none()
            && self.tension.is_none()
            && self.energy.is_none()
    }

    fn validate_into(&self, prefix: &str, issues: &mut Vec<ListenerResponseIssue>) {
        validate_bounded(self.valence, -1.0, 1.0, &format!("{prefix}.valence"), issues);
        validate_bounded(self.arousal, 0.0, 1.0, &format!("{prefix}.arousal"), issues);
        validate_bounded(self.tension, 0.0, 1.0, &format!("{prefix}.tension"), issues);
        validate_bounded(self.energy, 0.0, 1.0, &format!("{prefix}.energy"), issues);
    }
}

/// Optional mid-listening report. Time is relative to playback start, not wall clock.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AffectCheckpoint {
    pub elapsed_ms: u64,
    pub affect: SelfReportedAffect,
}

/// One listener-response episode associated with one exact audio artifact.
/// Identifiers must be opaque study-local tokens, never direct identifiers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ListenerResponseRecord {
    pub schema_version: String,
    pub episode_id: String,
    pub participant_token: String,
    pub presentation_token: String,
    /// SHA-256 of the exact audio bytes presented.
    pub audio_sha256: String,
    /// Optional symbolic score digest where a score exists.
    pub score_sha256: Option<String>,
    pub model_version: Option<String>,
    /// 1-based session-local indices; no precise timestamp is stored.
    pub session_index: u32,
    pub order_in_session: u32,
    pub intent: ListeningIntent,
    /// Required when the listener chooses a numeric target state.
    pub target_affect: Option<SelfReportedAffect>,
    pub activity: Option<ListeningActivity>,
    /// Listener-reported familiarity, 1 (not familiar) through 5 (very familiar).
    pub familiarity_1_to_5: Option<u8>,
    pub felt_before: Option<SelfReportedAffect>,
    pub during: Vec<AffectCheckpoint>,
    pub felt_after: Option<SelfReportedAffect>,
    /// Emotion/affect perceived as expressed by the music, distinct from felt state.
    pub perceived_music: Option<SelfReportedAffect>,
    pub enjoyment_1_to_7: Option<u8>,
    pub goal_attainment_1_to_7: Option<u8>,
    pub playback_completed: bool,
    pub consent_confirmed: bool,
    pub consent_protocol_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListenerResponseIssue {
    EmptyField { field: String },
    UnsupportedSchemaVersion { found: String },
    ConsentNotConfirmed,
    InvalidSha256 { field: String },
    InvalidAffectValue { field: String },
    InvalidOrdinal { field: String },
    MissingTargetForShiftIntent,
    NonIncreasingCheckpointTime,
}

impl ListenerResponseRecord {
    /// Validate data shape without treating incomplete self-report as invalid.
    /// Missing reports remain visible in coverage accounting.
    pub fn validate(&self) -> Vec<ListenerResponseIssue> {
        let mut issues = Vec::new();
        check_nonempty("schema_version", &self.schema_version, &mut issues);
        if self.schema_version != LISTENER_RESPONSE_SCHEMA_VERSION {
            issues.push(ListenerResponseIssue::UnsupportedSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if !self.consent_confirmed {
            issues.push(ListenerResponseIssue::ConsentNotConfirmed);
        }
        check_nonempty("episode_id", &self.episode_id, &mut issues);
        check_nonempty("participant_token", &self.participant_token, &mut issues);
        check_nonempty("presentation_token", &self.presentation_token, &mut issues);
        check_nonempty("consent_protocol_version", &self.consent_protocol_version, &mut issues);
        check_sha256("audio_sha256", &self.audio_sha256, &mut issues);

        if let Some(digest) = self.score_sha256.as_deref() {
            check_sha256("score_sha256", digest, &mut issues);
        }
        if self.model_version.as_deref().is_some_and(|value| value.trim().is_empty()) {
            issues.push(ListenerResponseIssue::EmptyField {
                field: "model_version".to_owned(),
            });
        }
        if self.session_index == 0 {
            issues.push(ListenerResponseIssue::InvalidOrdinal {
                field: "session_index".to_owned(),
            });
        }
        if self.order_in_session == 0 {
            issues.push(ListenerResponseIssue::InvalidOrdinal {
                field: "order_in_session".to_owned(),
            });
        }
        if self.familiarity_1_to_5.is_some_and(|value| !(1..=5).contains(&value)) {
            issues.push(ListenerResponseIssue::InvalidOrdinal {
                field: "familiarity_1_to_5".to_owned(),
            });
        }
        if self.enjoyment_1_to_7.is_some_and(|value| !(1..=7).contains(&value)) {
            issues.push(ListenerResponseIssue::InvalidOrdinal {
                field: "enjoyment_1_to_7".to_owned(),
            });
        }
        if self.goal_attainment_1_to_7.is_some_and(|value| !(1..=7).contains(&value)) {
            issues.push(ListenerResponseIssue::InvalidOrdinal {
                field: "goal_attainment_1_to_7".to_owned(),
            });
        }
        if self.intent == ListeningIntent::ShiftTowardTarget && self.target_affect.is_none() {
            issues.push(ListenerResponseIssue::MissingTargetForShiftIntent);
        }
        if let Some(target) = self.target_affect {
            target.validate_into("target_affect", &mut issues);
        }
        if let Some(before) = self.felt_before {
            before.validate_into("felt_before", &mut issues);
        }
        if let Some(after) = self.felt_after {
            after.validate_into("felt_after", &mut issues);
        }
        if let Some(perceived) = self.perceived_music {
            perceived.validate_into("perceived_music", &mut issues);
        }

        let mut previous_time = None;
        for (index, checkpoint) in self.during.iter().enumerate() {
            if previous_time.is_some_and(|previous| checkpoint.elapsed_ms <= previous) {
                issues.push(ListenerResponseIssue::NonIncreasingCheckpointTime);
                break;
            }
            previous_time = Some(checkpoint.elapsed_ms);
            checkpoint.affect.validate_into(&format!("during[{index}]"), &mut issues);
        }
        issues
    }

    /// Coverage helper, not a validity gate: empty episodes can represent
    /// skipped or declined feedback and must be accounted for separately.
    pub fn has_any_self_report(&self) -> bool {
        self.felt_before.is_some_and(|value| !value.is_empty())
            || self.during.iter().any(|point| !point.affect.is_empty())
            || self.felt_after.is_some_and(|value| !value.is_empty())
            || self.perceived_music.is_some_and(|value| !value.is_empty())
            || self.enjoyment_1_to_7.is_some()
            || self.goal_attainment_1_to_7.is_some()
    }
}

fn check_nonempty(field: &str, value: &str, issues: &mut Vec<ListenerResponseIssue>) {
    if value.trim().is_empty() {
        issues.push(ListenerResponseIssue::EmptyField { field: field.to_owned() });
    }
}

fn check_sha256(field: &str, value: &str, issues: &mut Vec<ListenerResponseIssue>) {
    let valid = value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit());
    if !valid {
        issues.push(ListenerResponseIssue::InvalidSha256 { field: field.to_owned() });
    }
}

fn validate_bounded(
    value: Option<f32>,
    minimum: f32,
    maximum: f32,
    field: &str,
    issues: &mut Vec<ListenerResponseIssue>,
) {
    if value.is_some_and(|value| !value.is_finite() || value < minimum || value > maximum) {
        issues.push(ListenerResponseIssue::InvalidAffectValue { field: field.to_owned() });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_record() -> ListenerResponseRecord {
        ListenerResponseRecord {
            schema_version: LISTENER_RESPONSE_SCHEMA_VERSION.to_owned(),
            episode_id: "episode-0001".to_owned(),
            participant_token: "participant-local-01".to_owned(),
            presentation_token: "presentation-0001".to_owned(),
            audio_sha256: "a".repeat(64),
            score_sha256: Some("b".repeat(64)),
            model_version: Some("muse-test-model-v1".to_owned()),
            session_index: 1,
            order_in_session: 1,
            intent: ListeningIntent::ShiftTowardTarget,
            target_affect: Some(SelfReportedAffect {
                valence: Some(0.4), arousal: Some(0.3), tension: None, energy: None,
            }),
            activity: Some(ListeningActivity::Resting),
            familiarity_1_to_5: Some(2),
            felt_before: Some(SelfReportedAffect {
                valence: Some(-0.2), arousal: Some(0.6), tension: Some(0.5), energy: Some(0.4),
            }),
            during: vec![
                AffectCheckpoint {
                    elapsed_ms: 10_000,
                    affect: SelfReportedAffect {
                        valence: Some(0.0), arousal: Some(0.5), tension: Some(0.4), energy: Some(0.5),
                    },
                },
                AffectCheckpoint {
                    elapsed_ms: 20_000,
                    affect: SelfReportedAffect {
                        valence: Some(0.2), arousal: Some(0.4), tension: Some(0.3), energy: Some(0.5),
                    },
                },
            ],
            felt_after: Some(SelfReportedAffect {
                valence: Some(0.3), arousal: Some(0.4), tension: Some(0.2), energy: Some(0.5),
            }),
            perceived_music: Some(SelfReportedAffect {
                valence: Some(0.5), arousal: Some(0.6), tension: Some(0.2), energy: Some(0.7),
            }),
            enjoyment_1_to_7: Some(6),
            goal_attainment_1_to_7: Some(5),
            playback_completed: true,
            consent_confirmed: true,
            consent_protocol_version: "pilot-consent-v1".to_owned(),
        }
    }

    #[test]
    fn valid_record_passes_and_round_trips_json() {
        let record = valid_record();
        assert!(record.validate().is_empty());
        assert!(record.has_any_self_report());
        let encoded = serde_json::to_string(&record).expect("serialize record");
        let decoded: ListenerResponseRecord =
            serde_json::from_str(&encoded).expect("deserialize record");
        assert_eq!(decoded, record);
    }

    #[test]
    fn invalid_digest_and_non_finite_or_out_of_range_values_fail_closed() {
        let mut record = valid_record();
        record.audio_sha256 = "not-a-digest".to_owned();
        record.felt_after = Some(SelfReportedAffect {
            valence: Some(f32::NAN), arousal: Some(1.1), tension: Some(-0.1), energy: Some(0.5),
        });
        let issues = record.validate();
        assert!(issues.iter().any(|issue| matches!(
            issue, ListenerResponseIssue::InvalidSha256 { field } if field == "audio_sha256"
        )));
        assert!(issues.iter().any(|issue| matches!(
            issue, ListenerResponseIssue::InvalidAffectValue { field } if field == "felt_after.valence"
        )));
        assert!(issues.iter().any(|issue| matches!(
            issue, ListenerResponseIssue::InvalidAffectValue { field } if field == "felt_after.arousal"
        )));
        assert!(issues.iter().any(|issue| matches!(
            issue, ListenerResponseIssue::InvalidAffectValue { field } if field == "felt_after.tension"
        )));
    }

    #[test]
    fn missing_self_report_is_retained_as_missing_not_rejected() {
        let mut record = valid_record();
        record.intent = ListeningIntent::Enjoy;
        record.target_affect = None;
        record.felt_before = None;
        record.felt_after = None;
        record.perceived_music = None;
        record.during.clear();
        record.enjoyment_1_to_7 = None;
        record.goal_attainment_1_to_7 = None;
        assert!(record.validate().is_empty());
        assert!(!record.has_any_self_report());
    }

    #[test]
    fn shift_intent_requires_a_listener_defined_target() {
        let mut record = valid_record();
        record.target_affect = None;
        assert!(record.validate().contains(&ListenerResponseIssue::MissingTargetForShiftIntent));
    }

    #[test]
    fn checkpoint_times_must_increase() {
        let mut record = valid_record();
        record.during[1].elapsed_ms = record.during[0].elapsed_ms;
        assert!(record.validate().contains(&ListenerResponseIssue::NonIncreasingCheckpointTime));
    }

    #[test]
    fn consent_must_be_confirmed() {
        let mut record = valid_record();
        record.consent_confirmed = false;
        assert!(record.validate().contains(&ListenerResponseIssue::ConsentNotConfirmed));
    }

    #[test]
    fn unknown_schema_versions_are_rejected() {
        let mut record = valid_record();
        record.schema_version = "future-unrecognized-version".to_owned();
        assert!(record.validate().iter().any(|issue| matches!(
            issue, ListenerResponseIssue::UnsupportedSchemaVersion { found }
                if found == "future-unrecognized-version"
        )));
    }

    #[test]
    fn invalid_ordinal_ratings_are_rejected() {
        let mut record = valid_record();
        record.familiarity_1_to_5 = Some(0);
        record.enjoyment_1_to_7 = Some(8);
        assert!(record.validate().iter().any(|issue| matches!(
            issue, ListenerResponseIssue::InvalidOrdinal { field } if field == "familiarity_1_to_5"
        )));
        assert!(record.validate().iter().any(|issue| matches!(
            issue, ListenerResponseIssue::InvalidOrdinal { field } if field == "enjoyment_1_to_7"
        )));
    }
}
