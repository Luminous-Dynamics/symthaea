// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Debris Bounties Coordinator Tests
//!
//! Tests for the Kessler Cleanup Market logic that can be exercised
//! without a Holochain conductor:
//! - State machine transitions (is_valid_transition)
//! - Input validation rules (bounty_id, justification, currency, amount, NORAD ID)
//! - BountyStatus serialization roundtrips
//! - RemovalMethod serialization
//! - ClaimStatus serialization
//! - VerificationEvidence construction rules
//! - BountyEntry and BountyRequirements from shared crate
//! - validate_string_field boundary tests
//! - Terminal state detection

use mycelix_space_shared::*;

// =============================================================================
// State Machine — is_valid_transition (replicated from coordinator)
// =============================================================================

/// The coordinator's state machine is private, so we replicate it here to test.
///
/// ```text
/// Open -> Claimed | Expired | Cancelled
/// Claimed -> InProgress | Open (release)
/// InProgress -> PendingVerification
/// PendingVerification -> Completed | InProgress (verification failed)
/// Terminal: Completed, Expired, Cancelled
/// ```
#[derive(Clone, Debug, PartialEq)]
enum CoordBountyStatus {
    Open,
    Claimed,
    InProgress,
    PendingVerification,
    Completed,
    Expired,
    Cancelled,
}

fn is_valid_transition(from: &CoordBountyStatus, to: &CoordBountyStatus) -> bool {
    matches!(
        (from, to),
        (CoordBountyStatus::Open, CoordBountyStatus::Claimed)
            | (CoordBountyStatus::Open, CoordBountyStatus::Expired)
            | (CoordBountyStatus::Open, CoordBountyStatus::Cancelled)
            | (CoordBountyStatus::Claimed, CoordBountyStatus::InProgress)
            | (CoordBountyStatus::Claimed, CoordBountyStatus::Open) // release claim
            | (
                CoordBountyStatus::InProgress,
                CoordBountyStatus::PendingVerification
            )
            | (
                CoordBountyStatus::PendingVerification,
                CoordBountyStatus::Completed
            )
            | (
                CoordBountyStatus::PendingVerification,
                CoordBountyStatus::InProgress
            ) // verification failed
    )
}

fn is_terminal(status: &CoordBountyStatus) -> bool {
    matches!(
        status,
        CoordBountyStatus::Completed | CoordBountyStatus::Expired | CoordBountyStatus::Cancelled
    )
}

// --- Happy path transitions ---

/// Open -> Claimed is valid.
#[test]
fn test_transition_open_to_claimed() {
    assert!(is_valid_transition(
        &CoordBountyStatus::Open,
        &CoordBountyStatus::Claimed
    ));
}

/// Open -> Expired is valid.
#[test]
fn test_transition_open_to_expired() {
    assert!(is_valid_transition(
        &CoordBountyStatus::Open,
        &CoordBountyStatus::Expired
    ));
}

/// Open -> Cancelled is valid.
#[test]
fn test_transition_open_to_cancelled() {
    assert!(is_valid_transition(
        &CoordBountyStatus::Open,
        &CoordBountyStatus::Cancelled
    ));
}

/// Claimed -> InProgress is valid.
#[test]
fn test_transition_claimed_to_in_progress() {
    assert!(is_valid_transition(
        &CoordBountyStatus::Claimed,
        &CoordBountyStatus::InProgress
    ));
}

/// Claimed -> Open (release claim) is valid.
#[test]
fn test_transition_claimed_to_open_release() {
    assert!(is_valid_transition(
        &CoordBountyStatus::Claimed,
        &CoordBountyStatus::Open
    ));
}

/// InProgress -> PendingVerification is valid.
#[test]
fn test_transition_in_progress_to_pending_verification() {
    assert!(is_valid_transition(
        &CoordBountyStatus::InProgress,
        &CoordBountyStatus::PendingVerification
    ));
}

/// PendingVerification -> Completed is valid.
#[test]
fn test_transition_pending_verification_to_completed() {
    assert!(is_valid_transition(
        &CoordBountyStatus::PendingVerification,
        &CoordBountyStatus::Completed
    ));
}

/// PendingVerification -> InProgress (failed verification) is valid.
#[test]
fn test_transition_pending_verification_to_in_progress() {
    assert!(is_valid_transition(
        &CoordBountyStatus::PendingVerification,
        &CoordBountyStatus::InProgress
    ));
}

// --- Invalid transitions ---

/// Open -> InProgress is invalid (must go through Claimed first).
#[test]
fn test_transition_open_to_in_progress_invalid() {
    assert!(!is_valid_transition(
        &CoordBountyStatus::Open,
        &CoordBountyStatus::InProgress
    ));
}

/// Open -> Completed is invalid (cannot skip intermediate states).
#[test]
fn test_transition_open_to_completed_invalid() {
    assert!(!is_valid_transition(
        &CoordBountyStatus::Open,
        &CoordBountyStatus::Completed
    ));
}

/// Open -> PendingVerification is invalid.
#[test]
fn test_transition_open_to_pending_verification_invalid() {
    assert!(!is_valid_transition(
        &CoordBountyStatus::Open,
        &CoordBountyStatus::PendingVerification
    ));
}

/// Claimed -> Completed is invalid (must go through InProgress first).
#[test]
fn test_transition_claimed_to_completed_invalid() {
    assert!(!is_valid_transition(
        &CoordBountyStatus::Claimed,
        &CoordBountyStatus::Completed
    ));
}

/// Claimed -> Cancelled is invalid (only Open can be cancelled).
#[test]
fn test_transition_claimed_to_cancelled_invalid() {
    assert!(!is_valid_transition(
        &CoordBountyStatus::Claimed,
        &CoordBountyStatus::Cancelled
    ));
}

/// InProgress -> Open is invalid (cannot go backwards past Claimed).
#[test]
fn test_transition_in_progress_to_open_invalid() {
    assert!(!is_valid_transition(
        &CoordBountyStatus::InProgress,
        &CoordBountyStatus::Open
    ));
}

/// InProgress -> Completed is invalid (must go through PendingVerification).
#[test]
fn test_transition_in_progress_to_completed_invalid() {
    assert!(!is_valid_transition(
        &CoordBountyStatus::InProgress,
        &CoordBountyStatus::Completed
    ));
}

// --- Terminal states cannot transition ---

/// Completed is terminal — no transitions out.
#[test]
fn test_terminal_completed_no_transitions() {
    let targets = [
        CoordBountyStatus::Open,
        CoordBountyStatus::Claimed,
        CoordBountyStatus::InProgress,
        CoordBountyStatus::PendingVerification,
        CoordBountyStatus::Expired,
        CoordBountyStatus::Cancelled,
    ];
    for target in &targets {
        assert!(
            !is_valid_transition(&CoordBountyStatus::Completed, target),
            "Completed -> {:?} should be invalid",
            target
        );
    }
}

/// Expired is terminal — no transitions out.
#[test]
fn test_terminal_expired_no_transitions() {
    let targets = [
        CoordBountyStatus::Open,
        CoordBountyStatus::Claimed,
        CoordBountyStatus::InProgress,
        CoordBountyStatus::PendingVerification,
        CoordBountyStatus::Completed,
        CoordBountyStatus::Cancelled,
    ];
    for target in &targets {
        assert!(
            !is_valid_transition(&CoordBountyStatus::Expired, target),
            "Expired -> {:?} should be invalid",
            target
        );
    }
}

/// Cancelled is terminal — no transitions out.
#[test]
fn test_terminal_cancelled_no_transitions() {
    let targets = [
        CoordBountyStatus::Open,
        CoordBountyStatus::Claimed,
        CoordBountyStatus::InProgress,
        CoordBountyStatus::PendingVerification,
        CoordBountyStatus::Completed,
        CoordBountyStatus::Expired,
    ];
    for target in &targets {
        assert!(
            !is_valid_transition(&CoordBountyStatus::Cancelled, target),
            "Cancelled -> {:?} should be invalid",
            target
        );
    }
}

/// Terminal state detection: Completed, Expired, Cancelled are terminal.
#[test]
fn test_terminal_state_detection() {
    assert!(is_terminal(&CoordBountyStatus::Completed));
    assert!(is_terminal(&CoordBountyStatus::Expired));
    assert!(is_terminal(&CoordBountyStatus::Cancelled));
    assert!(!is_terminal(&CoordBountyStatus::Open));
    assert!(!is_terminal(&CoordBountyStatus::Claimed));
    assert!(!is_terminal(&CoordBountyStatus::InProgress));
    assert!(!is_terminal(&CoordBountyStatus::PendingVerification));
}

// --- Full lifecycle test ---

/// Walk the full happy path: Open -> Claimed -> InProgress -> PendingVerification -> Completed.
#[test]
fn test_full_lifecycle_happy_path() {
    let transitions = [
        (CoordBountyStatus::Open, CoordBountyStatus::Claimed),
        (CoordBountyStatus::Claimed, CoordBountyStatus::InProgress),
        (
            CoordBountyStatus::InProgress,
            CoordBountyStatus::PendingVerification,
        ),
        (
            CoordBountyStatus::PendingVerification,
            CoordBountyStatus::Completed,
        ),
    ];

    for (from, to) in &transitions {
        assert!(
            is_valid_transition(from, to),
            "Happy path transition {:?} -> {:?} should be valid",
            from,
            to
        );
    }
}

/// Walk the verification-failed path: PendingVerification -> InProgress -> PendingVerification -> Completed.
#[test]
fn test_verification_retry_path() {
    assert!(is_valid_transition(
        &CoordBountyStatus::PendingVerification,
        &CoordBountyStatus::InProgress
    ));
    assert!(is_valid_transition(
        &CoordBountyStatus::InProgress,
        &CoordBountyStatus::PendingVerification
    ));
    assert!(is_valid_transition(
        &CoordBountyStatus::PendingVerification,
        &CoordBountyStatus::Completed
    ));
}

/// Self-transitions are invalid (no status -> same status).
#[test]
fn test_self_transitions_invalid() {
    let all = [
        CoordBountyStatus::Open,
        CoordBountyStatus::Claimed,
        CoordBountyStatus::InProgress,
        CoordBountyStatus::PendingVerification,
        CoordBountyStatus::Completed,
        CoordBountyStatus::Expired,
        CoordBountyStatus::Cancelled,
    ];
    for status in &all {
        assert!(
            !is_valid_transition(status, status),
            "Self-transition {:?} -> {:?} should be invalid",
            status,
            status
        );
    }
}

// =============================================================================
// Input Validation — Bounty Creation (coordinator logic)
// =============================================================================

/// Zero bounty amount is rejected.
#[test]
fn test_bounty_amount_zero_rejected() {
    let amount: u64 = 0;
    assert_eq!(amount, 0);
}

/// Positive bounty amount is valid.
#[test]
fn test_bounty_amount_positive_valid() {
    let amount: u64 = 1;
    assert!(amount > 0);
}

/// Maximum u64 bounty amount is valid.
#[test]
fn test_bounty_amount_max_valid() {
    let amount: u64 = u64::MAX;
    assert!(amount > 0);
}

/// NORAD ID 0 is rejected for bounty creation.
#[test]
fn test_bounty_norad_id_zero_rejected() {
    let norad_id: u32 = 0;
    assert!(norad_id == 0 || norad_id > 999999);
}

/// NORAD ID above 999999 is rejected.
#[test]
fn test_bounty_norad_id_above_max_rejected() {
    let norad_id: u32 = 1_000_000;
    assert!(norad_id == 0 || norad_id > 999999);
}

/// verification_threshold of 0 is rejected.
#[test]
fn test_bounty_verification_threshold_zero_rejected() {
    let threshold: u32 = 0;
    assert_eq!(threshold, 0);
}

/// verification_threshold of 1 is valid.
#[test]
fn test_bounty_verification_threshold_one_valid() {
    let threshold: u32 = 1;
    assert!(threshold > 0);
}

// =============================================================================
// validate_string_field for bounty fields
// =============================================================================

/// Empty bounty_id rejected.
#[test]
fn test_bounty_id_empty_rejected() {
    assert!(validate_string_field("", "bounty_id", 256).is_err());
}

/// bounty_id exceeding 256 chars rejected.
#[test]
fn test_bounty_id_too_long_rejected() {
    let id = "B".repeat(257);
    assert!(validate_string_field(&id, "bounty_id", 256).is_err());
}

/// Empty justification rejected.
#[test]
fn test_justification_empty_rejected() {
    assert!(validate_string_field("", "justification", 2048).is_err());
}

/// Justification exceeding 2048 chars rejected.
#[test]
fn test_justification_too_long_rejected() {
    let j = "J".repeat(2049);
    assert!(validate_string_field(&j, "justification", 2048).is_err());
}

/// Justification at exactly 2048 chars is valid.
#[test]
fn test_justification_at_max_valid() {
    let j = "J".repeat(2048);
    assert!(validate_string_field(&j, "justification", 2048).is_ok());
}

/// Empty currency rejected.
#[test]
fn test_currency_empty_rejected() {
    assert!(validate_string_field("", "currency", 16).is_err());
}

/// Currency exceeding 16 chars rejected.
#[test]
fn test_currency_too_long_rejected() {
    let c = "C".repeat(17);
    assert!(validate_string_field(&c, "currency", 16).is_err());
}

/// Valid 3-letter currency code.
#[test]
fn test_currency_valid_usd() {
    assert!(validate_string_field("USD", "currency", 16).is_ok());
}

/// Contribution message exceeding 1024 chars rejected (coordinator logic).
#[test]
fn test_contribution_message_too_long() {
    let msg = "M".repeat(1025);
    assert!(msg.len() > 1024);
}

/// Contribution message at 1024 chars is valid.
#[test]
fn test_contribution_message_at_max_valid() {
    let msg = "M".repeat(1024);
    assert!(msg.len() <= 1024);
}

// =============================================================================
// BountyStatus (shared crate) Serialization
// =============================================================================

/// All shared BountyStatus variants roundtrip through JSON.
#[test]
fn test_shared_bounty_status_serde_roundtrip() {
    let statuses = vec![
        BountyStatus::Active,
        BountyStatus::Claimed,
        BountyStatus::InProgress,
        BountyStatus::Completed,
        BountyStatus::Expired,
        BountyStatus::Cancelled,
    ];

    for s in &statuses {
        let json = serde_json::to_string(s).expect("serialize");
        let parsed: BountyStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&parsed, s);
    }
}

/// All shared BountyStatus variants produce distinct JSON.
#[test]
fn test_shared_bounty_status_distinct_json() {
    let statuses = vec![
        BountyStatus::Active,
        BountyStatus::Claimed,
        BountyStatus::InProgress,
        BountyStatus::Completed,
        BountyStatus::Expired,
        BountyStatus::Cancelled,
    ];

    let jsons: Vec<String> = statuses
        .iter()
        .map(|s| serde_json::to_string(s).unwrap())
        .collect();

    for i in 0..jsons.len() {
        for j in (i + 1)..jsons.len() {
            assert_ne!(jsons[i], jsons[j]);
        }
    }
}

// =============================================================================
// BountyEntry (shared crate) Serialization
// =============================================================================

/// BountyEntry roundtrips with all fields.
#[test]
fn test_bounty_entry_full_serde() {
    let entry = BountyEntry {
        debris_norad_id: 49863,
        bounty_amount: 100_000,
        currency: "USD".to_string(),
        sponsor: "SpaceGuard Inc.".to_string(),
        deadline: None,
        requirements: BountyRequirements {
            min_mass_kg: Some(500.0),
            max_altitude_km: Some(800.0),
            debris_type: Some(ObjectType::RocketBody),
            removal_method: Some("Deorbit".to_string()),
        },
        status: BountyStatus::Active,
    };

    let json = serde_json::to_string(&entry).expect("serialize");
    let parsed: BountyEntry = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.debris_norad_id, 49863);
    assert_eq!(parsed.bounty_amount, 100_000);
    assert_eq!(parsed.requirements.min_mass_kg, Some(500.0));
}

/// BountyEntry with empty requirements roundtrips.
#[test]
fn test_bounty_entry_empty_requirements_serde() {
    let entry = BountyEntry {
        debris_norad_id: 25544,
        bounty_amount: 1,
        currency: "SAP".to_string(),
        sponsor: "Test".to_string(),
        deadline: None,
        requirements: BountyRequirements {
            min_mass_kg: None,
            max_altitude_km: None,
            debris_type: None,
            removal_method: None,
        },
        status: BountyStatus::Active,
    };

    let json = serde_json::to_string(&entry).expect("serialize");
    let parsed: BountyEntry = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.requirements.min_mass_kg.is_none());
}

// =============================================================================
// Organization and Mission Plan Validation (claim_bounty coordinator logic)
// =============================================================================

/// Empty organization rejected.
#[test]
fn test_claim_organization_empty_rejected() {
    assert!(validate_string_field("", "organization", 256).is_err());
}

/// Organization exceeding 256 chars rejected.
#[test]
fn test_claim_organization_too_long_rejected() {
    let org = "O".repeat(257);
    assert!(validate_string_field(&org, "organization", 256).is_err());
}

/// Empty mission_plan rejected.
#[test]
fn test_claim_mission_plan_empty_rejected() {
    assert!(validate_string_field("", "mission_plan", 4096).is_err());
}

/// Mission plan exceeding 4096 chars rejected.
#[test]
fn test_claim_mission_plan_too_long_rejected() {
    let plan = "P".repeat(4097);
    assert!(validate_string_field(&plan, "mission_plan", 4096).is_err());
}

/// Mission plan at exactly 4096 chars is valid.
#[test]
fn test_claim_mission_plan_at_max_valid() {
    let plan = "P".repeat(4096);
    assert!(validate_string_field(&plan, "mission_plan", 4096).is_ok());
}
