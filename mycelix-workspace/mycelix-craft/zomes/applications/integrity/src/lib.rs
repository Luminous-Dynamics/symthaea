#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Applications Integrity Zome
//!
//! Job application workflow with state machine validation.
//! State transitions are enforced at the integrity layer — invalid
//! transitions (e.g., Draft → Interview) are rejected before they
//! reach the DHT.

use hdi::prelude::*;

/// A job application submitted by a candidate.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct JobApplication {
    /// Hash of the JobPosting being applied to
    pub job_posting_hash: ActionHash,
    /// Applicant agent
    pub applicant: AgentPubKey,
    /// Cover message
    pub cover_message: Option<String>,
    /// Hashes of PublishedCredentials to include as resume
    pub resume_credential_hashes: Vec<ActionHash>,
    /// Current status
    pub status: ApplicationStatus,
    /// When submitted
    pub submitted_at: Timestamp,
    /// Last status change
    pub updated_at: Timestamp,
}

/// Application lifecycle status with strict state machine transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplicationStatus {
    Draft,
    Submitted,
    UnderReview,
    Interview,
    Offered,
    Accepted,
    Rejected,
    Withdrawn,
}

impl ApplicationStatus {
    /// Returns the set of valid next states from the current state.
    pub fn valid_transitions(&self) -> &'static [ApplicationStatus] {
        match self {
            Self::Draft => &[Self::Submitted, Self::Withdrawn],
            Self::Submitted => &[Self::UnderReview, Self::Withdrawn],
            Self::UnderReview => &[Self::Interview, Self::Rejected, Self::Withdrawn],
            Self::Interview => &[Self::Offered, Self::Rejected, Self::Withdrawn],
            Self::Offered => &[Self::Accepted, Self::Rejected, Self::Withdrawn],
            // Terminal states — no further transitions
            Self::Accepted => &[],
            Self::Rejected => &[],
            Self::Withdrawn => &[],
        }
    }

    /// Check if a transition to the target state is valid.
    pub fn can_transition_to(&self, target: &ApplicationStatus) -> bool {
        self.valid_transitions().contains(target)
    }

    /// Is this a terminal (final) state?
    pub fn is_terminal(&self) -> bool {
        self.valid_transitions().is_empty()
    }
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    JobApplication(JobApplication),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Applicant agent -> their applications
    AgentToApplication,
    /// Job posting -> applications received
    JobPostingToApplication,
}

pub fn validate_application(app: &JobApplication) -> ExternResult<ValidateCallbackResult> {
    if let Some(ref msg) = app.cover_message {
        if msg.len() > 5_000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Cover message cannot exceed 5000 characters".into(),
            ));
        }
    }
    if app.resume_credential_hashes.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot attach more than 20 credentials".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::JobApplication(app) => validate_application(&app),
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { original_action, action, .. } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let (action, new_entry) = match &update {
                OpUpdate::Entry { action, app_entry, .. } => (action, Some(app_entry)),
                OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => (action, None),
            };
            // Author check
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update their entries".into(),
                ));
            }
            // State machine check for JobApplication updates
            if let Some(EntryTypes::JobApplication(new_app)) = new_entry {
                let original_record = must_get_valid_record(action.original_action_address.clone())?;
                if let Some(Entry::App(bytes)) = original_record.entry().as_option() {
                    if let Ok(old_app) = JobApplication::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        if !old_app.status.can_transition_to(&new_app.status) {
                            return Ok(ValidateCallbackResult::Invalid(format!(
                                "Invalid state transition: {:?} -> {:?}. Valid: {:?}",
                                old_app.status,
                                new_app.status,
                                old_app.status.valid_transitions()
                            )));
                        }
                    }
                }
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- State machine tests ----

    #[test]
    fn test_draft_transitions() {
        let s = ApplicationStatus::Draft;
        assert!(s.can_transition_to(&ApplicationStatus::Submitted));
        assert!(s.can_transition_to(&ApplicationStatus::Withdrawn));
        assert!(!s.can_transition_to(&ApplicationStatus::Interview));
        assert!(!s.can_transition_to(&ApplicationStatus::Offered));
        assert!(!s.can_transition_to(&ApplicationStatus::Accepted));
    }

    #[test]
    fn test_submitted_transitions() {
        let s = ApplicationStatus::Submitted;
        assert!(s.can_transition_to(&ApplicationStatus::UnderReview));
        assert!(s.can_transition_to(&ApplicationStatus::Withdrawn));
        assert!(!s.can_transition_to(&ApplicationStatus::Draft));
        assert!(!s.can_transition_to(&ApplicationStatus::Interview));
    }

    #[test]
    fn test_under_review_transitions() {
        let s = ApplicationStatus::UnderReview;
        assert!(s.can_transition_to(&ApplicationStatus::Interview));
        assert!(s.can_transition_to(&ApplicationStatus::Rejected));
        assert!(s.can_transition_to(&ApplicationStatus::Withdrawn));
        assert!(!s.can_transition_to(&ApplicationStatus::Submitted));
        assert!(!s.can_transition_to(&ApplicationStatus::Offered));
    }

    #[test]
    fn test_interview_transitions() {
        let s = ApplicationStatus::Interview;
        assert!(s.can_transition_to(&ApplicationStatus::Offered));
        assert!(s.can_transition_to(&ApplicationStatus::Rejected));
        assert!(s.can_transition_to(&ApplicationStatus::Withdrawn));
        assert!(!s.can_transition_to(&ApplicationStatus::Draft));
    }

    #[test]
    fn test_offered_transitions() {
        let s = ApplicationStatus::Offered;
        assert!(s.can_transition_to(&ApplicationStatus::Accepted));
        assert!(s.can_transition_to(&ApplicationStatus::Rejected));
        assert!(s.can_transition_to(&ApplicationStatus::Withdrawn));
        assert!(!s.can_transition_to(&ApplicationStatus::Interview));
    }

    #[test]
    fn test_terminal_states() {
        assert!(ApplicationStatus::Accepted.is_terminal());
        assert!(ApplicationStatus::Rejected.is_terminal());
        assert!(ApplicationStatus::Withdrawn.is_terminal());
        assert!(!ApplicationStatus::Draft.is_terminal());
        assert!(!ApplicationStatus::Submitted.is_terminal());
    }

    #[test]
    fn test_full_happy_path() {
        let states = [
            ApplicationStatus::Draft,
            ApplicationStatus::Submitted,
            ApplicationStatus::UnderReview,
            ApplicationStatus::Interview,
            ApplicationStatus::Offered,
            ApplicationStatus::Accepted,
        ];
        for window in states.windows(2) {
            assert!(
                window[0].can_transition_to(&window[1]),
                "{:?} should transition to {:?}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_withdrawal_from_any_active_state() {
        let active = [
            ApplicationStatus::Draft,
            ApplicationStatus::Submitted,
            ApplicationStatus::UnderReview,
            ApplicationStatus::Interview,
            ApplicationStatus::Offered,
        ];
        for s in active {
            assert!(
                s.can_transition_to(&ApplicationStatus::Withdrawn),
                "{:?} should allow withdrawal",
                s
            );
        }
    }

    #[test]
    fn test_no_resurrection_from_terminal() {
        let terminals = [
            ApplicationStatus::Accepted,
            ApplicationStatus::Rejected,
            ApplicationStatus::Withdrawn,
        ];
        let all = [
            ApplicationStatus::Draft,
            ApplicationStatus::Submitted,
            ApplicationStatus::UnderReview,
            ApplicationStatus::Interview,
            ApplicationStatus::Offered,
            ApplicationStatus::Accepted,
            ApplicationStatus::Rejected,
            ApplicationStatus::Withdrawn,
        ];
        for t in terminals {
            for target in all {
                assert!(
                    !t.can_transition_to(&target),
                    "{:?} should not transition to {:?}",
                    t,
                    target
                );
            }
        }
    }

    // ---- Validation tests ----

    fn test_action_hash() -> ActionHash {
        let mut bytes = vec![0x84, 0x29, 0x24];
        bytes.extend(vec![0u8; 33]);
        ActionHash::from_raw_36(bytes)
    }

    fn test_agent() -> AgentPubKey {
        let mut bytes = vec![0x84, 0x20, 0x24];
        bytes.extend(vec![0u8; 36]);
        AgentPubKey::from_raw_39(bytes)
    }

    #[test]
    fn test_cover_message_too_long() {
        let app = JobApplication {
            job_posting_hash: test_action_hash(),
            applicant: test_agent(),
            cover_message: Some("x".repeat(5001)),
            resume_credential_hashes: vec![],
            status: ApplicationStatus::Draft,
            submitted_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        };
        assert!(matches!(validate_application(&app).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_too_many_credentials() {
        let app = JobApplication {
            job_posting_hash: test_action_hash(),
            applicant: test_agent(),
            cover_message: None,
            resume_credential_hashes: (0..21).map(|_| test_action_hash()).collect(),
            status: ApplicationStatus::Draft,
            submitted_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        };
        assert!(matches!(validate_application(&app).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    // ---- Supplementary exhaustive tests ----

    #[test]
    fn test_no_self_transitions() {
        let all = [
            ApplicationStatus::Draft,
            ApplicationStatus::Submitted,
            ApplicationStatus::UnderReview,
            ApplicationStatus::Interview,
            ApplicationStatus::Offered,
            ApplicationStatus::Accepted,
            ApplicationStatus::Rejected,
            ApplicationStatus::Withdrawn,
        ];
        for s in all {
            assert!(!s.can_transition_to(&s), "{:?} must not self-transition", s);
        }
    }

    #[test]
    fn test_exactly_three_terminal_states() {
        let all = [
            ApplicationStatus::Draft,
            ApplicationStatus::Submitted,
            ApplicationStatus::UnderReview,
            ApplicationStatus::Interview,
            ApplicationStatus::Offered,
            ApplicationStatus::Accepted,
            ApplicationStatus::Rejected,
            ApplicationStatus::Withdrawn,
        ];
        let terminal_count = all.iter().filter(|s| s.is_terminal()).count();
        assert_eq!(terminal_count, 3);
    }

    #[test]
    fn test_transition_count_per_state() {
        assert_eq!(ApplicationStatus::Draft.valid_transitions().len(), 2);
        assert_eq!(ApplicationStatus::Submitted.valid_transitions().len(), 2);
        assert_eq!(ApplicationStatus::UnderReview.valid_transitions().len(), 3);
        assert_eq!(ApplicationStatus::Interview.valid_transitions().len(), 3);
        assert_eq!(ApplicationStatus::Offered.valid_transitions().len(), 3);
        assert_eq!(ApplicationStatus::Accepted.valid_transitions().len(), 0);
        assert_eq!(ApplicationStatus::Rejected.valid_transitions().len(), 0);
        assert_eq!(ApplicationStatus::Withdrawn.valid_transitions().len(), 0);
    }

    #[test]
    fn test_rejection_only_from_review_interview_offered() {
        assert!(!ApplicationStatus::Draft.can_transition_to(&ApplicationStatus::Rejected));
        assert!(!ApplicationStatus::Submitted.can_transition_to(&ApplicationStatus::Rejected));
        assert!(ApplicationStatus::UnderReview.can_transition_to(&ApplicationStatus::Rejected));
        assert!(ApplicationStatus::Interview.can_transition_to(&ApplicationStatus::Rejected));
        assert!(ApplicationStatus::Offered.can_transition_to(&ApplicationStatus::Rejected));
    }

    #[test]
    fn test_draft_cannot_skip_to_accepted() {
        assert!(!ApplicationStatus::Draft.can_transition_to(&ApplicationStatus::Accepted));
        assert!(!ApplicationStatus::Draft.can_transition_to(&ApplicationStatus::Interview));
        assert!(!ApplicationStatus::Draft.can_transition_to(&ApplicationStatus::Offered));
    }

    #[test]
    fn test_valid_application_at_limit() {
        let app = JobApplication {
            job_posting_hash: test_action_hash(),
            applicant: test_agent(),
            cover_message: Some("x".repeat(5000)), // exactly at limit
            resume_credential_hashes: (0..20).map(|_| test_action_hash()).collect(), // exactly at limit
            status: ApplicationStatus::Draft,
            submitted_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        };
        assert_eq!(validate_application(&app).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_no_cover_message_valid() {
        let app = JobApplication {
            job_posting_hash: test_action_hash(),
            applicant: test_agent(),
            cover_message: None,
            resume_credential_hashes: vec![],
            status: ApplicationStatus::Draft,
            submitted_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        };
        assert_eq!(validate_application(&app).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_serde_roundtrip_all_statuses() {
        let all = [
            ApplicationStatus::Draft,
            ApplicationStatus::Submitted,
            ApplicationStatus::UnderReview,
            ApplicationStatus::Interview,
            ApplicationStatus::Offered,
            ApplicationStatus::Accepted,
            ApplicationStatus::Rejected,
            ApplicationStatus::Withdrawn,
        ];
        for s in all {
            let json = serde_json::to_string(&s).unwrap();
            let back: ApplicationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back, "Serde roundtrip failed for {:?}", s);
        }
    }
}
